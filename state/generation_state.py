import logging
from typing import Dict, List, Optional, Tuple

from core.models import ApiChunkResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Low-level helpers                                                          #
# --------------------------------------------------------------------------- #
def _decode_token(token: str) -> str:
    if not token:
        return token

    # 1) GPT-BPE newline marker
    if token.startswith("Ċ"):
        token = "\n" + token[1:]

    # 2) GPT-BPE / SentencePiece leading-space markers
    if token.startswith("Ġ"):
        token = " " + token[1:]
    elif token.startswith("▁"):
        token = " " + token[1:]

    # 3) Strip any that remain inside the token
    return (
        token.replace("Ċ", "\n")
             .replace("Ġ", " ")
             .replace("▁", " ")
    )


# Convenience wrapper (still used elsewhere in the project)
def _tokens_to_text(tokens: List[str]) -> str:
    return "".join(_decode_token(t) for t in tokens)


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
    def __init__(self, prompt_string: str) -> None:
        if not isinstance(prompt_string, str):
            raise TypeError("prompt_string must be a string")

        self.prompt_string: str                 = prompt_string
        self.generated_token_strings: List[str] = []

        # --- incremental caches -------------------------------------------
        self._decoded_buffer: str   = ""        # concat of decoded tokens
        self._decoded_lens:  List[int] = []     # len(decoded_tok_i)

        # --- logprobs cache (unchanged) -----------------------------------
        self.logprobs_cache: Dict[int, List[Tuple[str, float]]] = {}

        logger.info(
            f"GenerationState initialised. Prompt length (chars) = "
            f"{len(prompt_string)}"
        )

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
