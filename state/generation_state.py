import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from core.models import ApiChunkResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Low-level helpers                                                          #
# --------------------------------------------------------------------------- #
def _build_u2b_full():
    # GPT/OpenAI table
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    u2b = {ord(chr(c)): b for b, c in zip(bs, cs)}
    # Qwen/Mistral table (bytes 0‑255 → U+2500+byte)
    for b in range(256):
        u2b[0x2500 + b] = b
    return u2b

_U2B = _build_u2b_full()

def fix_mojibake(text: str) -> str:
    buf, changed = bytearray(), False
    for ch in text:
        cp = ord(ch)
        if cp in _U2B:
            buf.append(_U2B[cp]); changed = True
        else:
            buf.extend(ch.encode('utf-8'))
    if not changed:
        return text
    try:
        return buf.decode('utf-8')
    except UnicodeDecodeError:
        return text

def fix_mojibake_iter(text: str, max_rounds: int = 3) -> str:
    prev = text
    for _ in range(max_rounds):
        cur = fix_mojibake(prev)
        if cur == prev:
            return cur
        prev = cur
    return prev


def _decode_token(token: str) -> str:
    """
    • Handles newline / leading-space markers exactly as before.
    • Then runs _repair_mojibake on the *rest* of the token string.
    """
    if not token:
        return token

    # Repair mojibake that is wholly contained in this single token
    token = fix_mojibake_iter(token)
    token = token.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")

    #return lead + token
    return token



# Convenience wrapper (still used elsewhere in the project)
def _tokens_to_text(tokens: List[str]) -> str:
    return "".join(_decode_token(t) for t in tokens)


# --------------------------------------------------------------------------- #
#  Special-token scrubbing                                                    #
# --------------------------------------------------------------------------- #
# A chat template's control tokens are *not* ordinary text.  Depending on the
# backend and on how a token happens to land in a streamed chunk, they can
# arrive as a `logprobs` token, inside the SSE `text` delta, or split across two
# chunks.  None of those forms may reach the dataset.
#
# The default pattern covers the chat-template families we actually meet
# (Gemma, Llama-3, ChatML, Mistral, Qwen) rather than trying to match every
# `<...>` span, which would eat legitimate prose and code.  Extra literal markers
# from `generation_params.end_of_turn_markers` are always added on top.
_DEFAULT_SPECIAL_TOKEN_PATTERN = (
    r"<\|?/?(?:"
    r"end_of_turn|start_of_turn|turn|end_of_text|eot_id|eom_id|eom|eot"
    r"|im_start|im_end|im_sep"
    r"|begin_of_text|end_of_text|start_header_id|end_header_id"
    r"|file_separator|pad|unk|bos|eos|sep|cls|mask"
    r")\|?>"
)


class SpecialTokenScrubber:
    """
    Removes chat-template control tokens from a growing text stream.

    Two things make this non-trivial:

    1.  A marker can be **split across chunks** (``"<tur"`` then ``"n|>"``).
        So text is only released once it can no longer be the prefix of a
        marker: the last ``holdback`` characters are withheld until either more
        text arrives or :meth:`flush` is called.
    2.  A marker can be **anywhere** in the chunk, not just at its tail, and may
        appear more than once.

    ``feed()`` returns only text that is safe to emit; ``flush()`` releases the
    held-back remainder at end of stream.
    """

    def __init__(
        self,
        markers: Optional[Sequence[str]] = None,
        pattern: Optional[str] = None,
    ) -> None:
        self.markers: List[str] = [
            str(m).strip() for m in (markers or []) if str(m).strip()
        ]

        self._literal_re = None
        if self.markers:
            # Longest first so "<end_of_turn>" wins over "<turn|>".
            alts = "|".join(re.escape(m) for m in sorted(self.markers, key=len, reverse=True))
            self._literal_re = re.compile(alts)

        pat = pattern if pattern is not None else _DEFAULT_SPECIAL_TOKEN_PATTERN
        self._pattern_re = re.compile(pat) if pat else None

        # How much of the tail must be withheld because it could still turn out to
        # be the *start* of a marker.  Driven by the longest configured marker;
        # the generic pattern needs at most 16 characters (its longest alternative
        # is `<start_of_turn>`, 15 chars, plus one for a not-yet-arrived `>`).
        # Capped so a lone `<` in ordinary prose never stalls the stream.
        longest_literal = max((len(m) for m in self.markers), default=0)
        self.holdback = min(32, max(longest_literal - 1, 16 if self._pattern_re else 0))

        self._pending = ""
        self.removed_count = 0

    # -- helpers ---------------------------------------------------------- #
    def _strip(self, text: str) -> str:
        if not text:
            return text
        before = text
        if self._literal_re is not None:
            text = self._literal_re.sub("", text)
        if self._pattern_re is not None:
            text = self._pattern_re.sub("", text)
        if text != before:
            self.removed_count += 1
        return text

    def is_marker(self, token: str) -> bool:
        """True when *token* (raw or decoded) is entirely a control token."""
        if not token:
            return False
        stripped = token.strip()
        if not stripped:
            return False
        for candidate in (token, stripped, _decode_token(token).strip()):
            if candidate in self.markers:
                return True
            if self._pattern_re is not None and self._pattern_re.fullmatch(candidate):
                return True
        return False

    # -- streaming API ---------------------------------------------------- #
    def feed(self, text: str) -> str:
        """Add *text*, return the part that is safe to emit now."""
        if text:
            self._pending += text
        if not self._pending:
            return ""

        cut = len(self._pending) - self.holdback
        if cut <= 0:
            return ""
        releasable, self._pending = self._pending[:cut], self._pending[cut:]
        return self._strip(releasable)

    def flush(self) -> str:
        """Release everything still held back (end of generation)."""
        pending, self._pending = self._pending, ""
        return self._strip(pending)

    def scrub(self, text: str) -> str:
        """One-shot scrub of a complete string (no hold-back)."""
        return self._strip(text)


# --------------------------------------------------------------------------- #
#  GenerationState                                                            #
# --------------------------------------------------------------------------- #
class GenerationState:
    """
    Holds the growing sequence of *raw* token strings **and** an
    incremental plain-text buffer that validators can query in O(1).

    Key optimisation artefacts
    --------------------------
    • `_decoded_buffer`:   the whole generated text so far, already decoded  
    • `_decoded_lens[i]`:  length of the decoded form of token *i*

    A character→token lookup is O(n) in the *window* length, not global len.
    """

    # ....................................................................... #
    #  Construction                                                           #
    # ....................................................................... #
    def __init__(self, prompt_string: str, log_init: bool = True) -> None:
        if not isinstance(prompt_string, str):
            raise TypeError("prompt_string must be a string")

        self.prompt_string: str                 = prompt_string
        self.generated_token_strings: List[str] = []

        # --- incremental caches -------------------------------------------
        self._decoded_buffer: str   = ""        # concat of decoded tokens
        self._decoded_lens:  List[int] = []     # len(decoded_tok_i)

        # --- logprobs cache (unchanged) -----------------------------------
        self.logprobs_cache: Dict[int, List[Tuple[str, float]]] = {}

        # Back-tracking builds one of these per candidate token to test it, so
        # this must not log: a single backtrack with 20 candidates produced 20
        # "GenerationState initialised" lines and buried the real events.
        if log_init:
            logger.info(
                "GenerationState initialised. Prompt length (chars) = %d",
                len(prompt_string),
            )

    @classmethod
    def hypothetical(cls, prompt_string: str, tokens: List[str]) -> "GenerationState":
        """Build a throw-away state for ``tokens`` without logging.

        Used by the back-tracker to test a candidate token.  It skips
        ``__init__``'s logging and builds the decoded caches directly, which is
        the only thing a validator needs; constructing a full state per
        candidate was both noisy and O(n) per candidate.
        """
        obj = cls.__new__(cls)
        obj.prompt_string = prompt_string
        obj.generated_token_strings = list(tokens)
        obj._decoded_lens = [len(_decode_token(t)) for t in tokens]
        obj._decoded_buffer = _tokens_to_text(obj.generated_token_strings)
        obj.logprobs_cache = {}
        return obj

    # ....................................................................... #
    #  Token stream maintenance                                               #
    # ....................................................................... #
    def append_chunk(self, chunk: ApiChunkResult) -> None:
        """
        Extend state with a fresh chunk returned by the API.
        """
        if not chunk.token_strings:
            logger.warning("append_chunk called with an empty chunk")
            return

        start = len(self.generated_token_strings)

        for raw_tok in chunk.token_strings:
            decoded = _decode_token(raw_tok)
            self.generated_token_strings.append(raw_tok)
            self._decoded_buffer += decoded
            self._decoded_lens.append(len(decoded))

        # cache per-token logprobs
        for rel_pos, alt_list in chunk.logprobs.items():
            abs_idx = start + rel_pos
            if abs_idx < len(self.generated_token_strings):
                self.logprobs_cache[abs_idx] = alt_list

    def truncate(self, generated_index: int) -> None:
        """
        Drop everything **from** generated_index onwards (inclusive).
        """
        if generated_index < 0 or generated_index >= len(self.generated_token_strings):
            return

        self.generated_token_strings = self.generated_token_strings[:generated_index]
        self._decoded_lens           = self._decoded_lens[:generated_index]
        self.logprobs_cache = {k: v for k, v in self.logprobs_cache.items()
                               if k < generated_index}

        # rebuild decoded buffer once – cheap compared with token loop in validators
        self._decoded_buffer = _tokens_to_text(self.generated_token_strings)

    def replace_token_string(self, generated_index: int, new_raw_token: str) -> None:
        """
        Overwrite a single token (used by back-tracking) and keep caches coherent.
        """
        if not (0 <= generated_index < len(self.generated_token_strings)):
            logger.error("replace_token_string: index out of range")
            return

        self.generated_token_strings[generated_index] = new_raw_token
        new_decoded = _decode_token(new_raw_token)
        self._decoded_lens[generated_index] = len(new_decoded)

        # rebuild decoded buffer – replacement is rare; clarity > micro-optim
        self._decoded_buffer = _tokens_to_text(self.generated_token_strings)

    # ....................................................................... #
    #  Fast views / helpers                                                   #
    # ....................................................................... #
    def get_generated_text(self) -> str:
        return self._decoded_buffer

    def get_full_text(self) -> str:
        return self.prompt_string + self._decoded_buffer

    def rebuild_decoded_buffer(self) -> None:
        """
        Recompute the plain-text buffer from the current token strings.

        Needed after token-level edits that bypass :meth:`replace_token_string`
        (e.g. removing an embedded control token from the middle of a chunk).
        """
        self._decoded_buffer = _tokens_to_text(self.generated_token_strings)
        self._decoded_lens = [len(_decode_token(t)) for t in self.generated_token_strings]

    def get_hypothetical_generated_text(
        self, truncate_index: int, next_token_string: str
    ) -> str:
        safe_ix = max(0, min(truncate_index, len(self.generated_token_strings)))
        part = (
            self.generated_token_strings[:safe_ix] + [next_token_string]
        )
        return _tokens_to_text(part)

    def get_generated_length(self) -> int:
        return len(self.generated_token_strings)

    def get_logprobs(self, generated_index: int):
        return self.logprobs_cache.get(generated_index)

    def get_token_string_at(self, generated_index: int) -> Optional[str]:
        if 0 <= generated_index < len(self.generated_token_strings):
            return self.generated_token_strings[generated_index]
        return None

    # ....................................................................... #
    #  NEW:  O(window) char→token mapper                                      #
    # ....................................................................... #
    def char_pos_to_token_index(self, char_pos: int) -> Optional[int]:
        """
        Convert a character offset **within `get_generated_text()`**
        to its corresponding token index.

        Linear in number of tokens until the offset – fine for the small
        validator scan window and far faster than re-decoding every call.
        """
        if char_pos < 0 or char_pos >= len(self._decoded_buffer):
            return None

        running = 0
        for idx, length in enumerate(self._decoded_lens):
            running += length
            if char_pos < running:
                return idx
        return None  # should not happen
