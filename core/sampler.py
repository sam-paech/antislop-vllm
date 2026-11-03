# core/sampler.py
"""
ApiAntiSlopSampler
==================

Coordinates chunk-wise text generation over any OpenAI-compatible
completion endpoint, applies a set of *validators* (phrase block-list,
regex block-list, …), and performs local back-tracking when a validator
flags an offending token.

The sampler never calls the remote API during back-tracking; it resamples
from the *top-logprob lists* already returned with each chunk, applying
temperature, min-p, top-p, and top-k exactly as normal decoding does.

If back-tracking fails (no viable alternative token in the cached list),
the violation is **suppressed** so validators won’t raise the same error
again.  Generation continues from that point.

All ban / back-track events are pushed to `self.events` for downstream
inspection or metrics.
"""
from __future__ import annotations

import json
import logging
import math
import unicodedata
import time
import random
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

import tiktoken # Added tiktoken

from api_client.base_client import BaseApiClient
from core.models import ApiChunkResult, ViolationInfo
from state.generation_state import (
    GenerationState,
    SpecialTokenScrubber,
    _decode_token,
    _tokens_to_text,
)
from validators.base_validator import BaseValidator
from validators.slop_phrase_validator import SlopPhraseValidator
from utils.sampler_helpers import apply_sampling_filters, select_tail_tokens
import csv, time, datetime, os
import requests
from threading import Lock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Global one-time cache for the “banned-prefix” token set
#  key  = chat-template model-id   →   value = frozenset(str tokens)
# ---------------------------------------------------------------------------
_BANNED_PREFIX_CACHE: dict[str, frozenset[str]] = {}
_BANNED_PREFIX_LOCK = Lock()

# Cached result of the streamed-logprobs probe, keyed by api base_url.
# Only *successful* probes are cached, so a backend that is briefly unavailable
# still gets re-probed on the next request instead of failing permanently.
_STREAM_PROBE_CACHE: dict[str, bool] = {}

# One-shot warnings.  A sampler is constructed per request in server mode, so any
# warning raised from __init__ is otherwise emitted once per request and drowns
# out the events that matter.
_TAIL_PREFIX_WARNED = False
_TIKTOKEN_WARNED: set[str] = set()


def _build_banned_prefix_set(chat_tpl, validators) -> frozenset[str]:
    """
    Return a *lower-cased* set containing:

    1. The first token (as produced by the HF tokenizer that your **model**
       will actually use) of every banned phrase / n-gram.
    2. All plain-text substrings that can *start* any banned phrase,
       lengths 2‥15 chars, **both with and without a leading space**.

    The caller later checks

        tok.lower()                       ∈ set   (raw token string)
        _decode_token(tok).lower()        ∈ set   (plain-text form)

    so we need only ASCII space variants – special markers (Ġ ) are handled
    by the first check.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        chat_tpl.model_id, trust_remote_code=True
    )

    # ------------------------------------------------------------
    # 1. collect source strings (banned phrases + banned n-grams)
    # ------------------------------------------------------------
    sources: list[str] = []
    for v in validators:
        if v.__class__.__name__ == "SlopPhraseValidator":
            sources.extend(v.slop_phrases_keys)
        elif v.__class__.__name__ == "NGramValidator":
            sources.extend(" ".join(t) for t in getattr(v, "banned_ngrams_tuples", []))

    if not sources:
        logging.getLogger(__name__).info(
            "tail-prefix filter initialised – no banned prefixes found."
        )
        return frozenset()

    out: set[str] = set()         # final result (lower-cased)

    # ------------------------------------------------------------
    # 2. token-based prefixes (same logic as before)
    # ------------------------------------------------------------
    variants: list[str] = []
    for s in sources:
        base = s.lstrip()
        variants.append(base)
        variants.append(" " + base)

    enc = tok(
        variants,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    for ids in enc["input_ids"]:
        if not ids:
            continue
        first_tok = tok.convert_ids_to_tokens(ids[0])
        if first_tok:
            out.add(first_tok.lower())

    # ------------------------------------------------------------
    # 3. plain-text substring prefixes (2‥15 characters)
    # ------------------------------------------------------------
    MAX_SUBLEN = 15
    MIN_SUBLEN = 2

    for s in sources:
        plain = s.lstrip().lower()
        upto  = min(MAX_SUBLEN, len(plain))
        for ln in range(MIN_SUBLEN, upto + 1):
            sub = plain[:ln]
            out.add(sub)           # no leading space
            out.add(" " + sub)     # with leading space

    logging.getLogger(__name__).info(
        "tail-prefix filter initialised – %d unique prefixes (token+substring).",
        len(out),
    )
    return frozenset(out)

def _is_repetition(
    existing_text: str,
    added_text: str,
    earlier_additions: Optional[List[str]] = None,
) -> bool:
    """True when *added_text* is the model repeating itself rather than continuing.

    Used to reject a "continuation" that is really a loop: after a backend stop is
    refused, the model is asked to keep writing, and a model that has already
    finished answers with a stock phrase.  Left unchecked those answers are appended
    until the attempt budget runs out -- the turn in the dataset that contains
    ``of course.`` 327 times.

    Two shapes of loop are recognised.  Both compare **word sequences**, not
    characters, so a sentence that merely reuses a common word is not caught:

    * the added words already appear in *existing_text* as one contiguous run, i.e.
      a verbatim restatement of something already said;
    * the added words repeat a fragment produced by an *earlier refused stop*.  This
      is the stock-phrase loop: each ``of course.`` is new text on its own, and only
      the fact that it has been said before makes it a loop.

    *earlier_additions* holds the fragments added by previous refused stops in this
    generation.  A fragment of fewer than two words is ignored, because a lone word
    ("and") collides by accident far too easily.
    """
    def _norm(text: str) -> str:
        # Pad with spaces so `in` compares whole words: without it "he said"
        # matched inside "she said" and a real continuation was rolled back.
        return " " + " ".join(text.lower().split()) + " "

    added_words = added_text.split()
    if len(added_words) < 2:
        return False
    needle = _norm(added_text)

    existing_words = existing_text.split()
    if len(existing_words) >= len(added_words):
        if needle in _norm(existing_text):
            return True

    for earlier in earlier_additions or ():
        earlier_text = _norm(earlier)
        if not earlier_text.strip():
            continue
        # Containment either way covers a verbatim repeat and a repeat of the same
        # stock phrase wrapped in a few extra words.
        if needle in earlier_text or earlier_text in needle:
            return True

    return False


def _scale_ban_strength(x: float) -> float:
    # Deciles map to decades: 0.1→0.9, 0.2→0.99, …, 0.9→0.999999999
    x = max(0.0, min(1.0, float(x)))
    if x >= 1.0:
        return 1.0
    return 1.0 - 10.0 ** (-10.0 * x)


class ApiAntiSlopSampler:
    """
    High-level controller for generation → validation → (optional)
    back-tracking.  No local tokenizer is required; all token strings and
    log-probs come from the API.
    """

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        api_client: BaseApiClient,
        validators: List[BaseValidator],
        config: Dict[str, object],
        on_ban_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_chunk_yielded_callback: Optional[Callable[[str, int], None]] = None,
        # Use model_name for tiktoken by default, fallback in code
        tiktoken_model_name_for_counting: Optional[str] = None,
        chat_template_formatter: Optional[Any] = None, 
    ) -> None:
        self.api_client = api_client
        self.validators = validators
        self.config = config

        # ── per-chunk timing CSV ───────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"timings_antislop_{ts}.csv"
        self._timings_path = os.getenv("ANTISLOP_TIMINGS", default_path)
        self._chunk_timings: list[tuple[int, int, float, float]] = []   # (chunk#, ctx_len, api_s, val_s)

        # Callbacks
        self.on_ban_event_callback = on_ban_event_callback
        self.on_chunk_yielded_callback = on_chunk_yielded_callback

        # ── tokenwise-dpo-pair capture ─────────────────────────────────────────
        self.ftpo_samples: Dict[str, Dict[str, Any]] = {}


        # Tiktoken encoding for internal token counting
        # Use the provided model name or default to cl100k_base
        _tiktoken_encoding_name = tiktoken_model_name_for_counting or "cl100k_base"
        _tiktoken_model_name = str(tiktoken_model_name_for_counting)
        try:
            # If tiktoken_model_name_for_counting is None, it will use "cl100k_base"
            # If it's a model name, it will try that.
            if tiktoken_model_name_for_counting:
                 self.tiktoken_encoding = tiktoken.encoding_for_model(tiktoken_model_name_for_counting)
            else:
                 self.tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            logger.info(f"ApiAntiSlopSampler using tiktoken encoding for '{_tiktoken_encoding_name}' for yielded chunk token counts.")
        except KeyError:
            if _tiktoken_model_name not in _TIKTOKEN_WARNED:
                _TIKTOKEN_WARNED.add(_tiktoken_model_name)
                logger.warning(
                    f"Tiktoken model '{tiktoken_model_name_for_counting}' not found. "
                    f"Falling back to 'cl100k_base' for yielded chunk token counts "
                    f"(this warning is logged once per model name)."
                )
            self.tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Error getting tiktoken encoding for '{_tiktoken_encoding_name}': {e}. "
                         f"Falling back to 'cl100k_base' for yielded chunk token counts.")
            self.tiktoken_encoding = tiktoken.get_encoding("cl100k_base")


        gen = config.get("generation_params", {})
        self.gen_config = config.get("generation_params", {})
        self.chunk_size = gen.get("chunk_size", 50)
        self.top_logprobs_count = gen.get("top_logprobs_count", 1)
        self.max_new_tokens = gen.get("max_new_tokens", 600)
        self.temperature = max(gen.get("temperature", 0.7), 1e-3)
        self.top_p = gen.get("top_p")
        self.top_k = gen.get("top_k")
        self.min_p = gen.get("min_p")
        self.stop_sequences = gen.get("stop_sequences")
        self.timeout = gen.get("timeout", 120)

        # ── end-of-turn marker list ─────────────────────────────────────────
        # Gemma-class models sampling at temperature > 0 emit their end-of-turn
        # marker (<end_of_turn> / <turn|>) mid-response.  vLLM reports that as
        # finish_reason="stop" and *strips the token itself* from the response,
        # so the marker list below cannot be used for detection -- only for
        # scrubbing tokens that do leak through.
        # NOTE: `[]` is falsy, so a plain `or DEFAULT` would silently re-enable
        # the feature for anyone who set `end_of_turn_markers: []` to turn it
        # off.  Only fall back to the defaults when the key is *absent*.
        eos_markers = gen.get("end_of_turn_markers")
        if eos_markers is None:
            eos_markers = ["<end_of_turn>", "<turn|>"]
        self.end_of_turn_markers: List[str] = [
            str(m).strip().lower() for m in eos_markers if str(m).strip()
        ]
        self.max_consecutive_stops = max(1, int(gen.get("max_consecutive_stops", 3)))
        # Consecutive *no-progress* early stops (safety valve); see _handle_stop.
        self._consecutive_stops = 0
        self.chat_formatter = chat_template_formatter
        self.force_backtrack = bool(config.get("backtracking", {}).get("force_backtrack", False))

        # ── stop / end-of-turn policy ───────────────────────────────────────
        # The original early-EOS feature only fired when a marker *token* showed
        # up in `logprobs.tokens`.  vLLM strips the stop token before
        # serialising a response, so that token never arrives and the feature
        # never fired: every premature `<end_of_turn>` silently truncated the
        # answer.  Detection now rests on observable evidence (did the attempt
        # produce text, and does that text end on a natural boundary); the
        # marker list is used only for *scrubbing* leaked control tokens.
        #
        # stop_on_clean_end: accept a backend stop that lands on a natural text
        #   boundary.  Set false to always keep generating — never truncate, at
        #   the cost of occasionally running on past the real end of turn.
        self.stop_on_clean_end = bool(gen.get("stop_on_clean_end", True))
        clean_end = gen.get("clean_end_chars")
        if clean_end is None:
            # Roleplay replies routinely end on an *action*, a trailing tilde, a
            # dash or a colon.  Leaving those out made the sampler refuse a
            # finished reply's stop and ask the model to keep writing.
            clean_end = [".", "!", "?", "…", '"', "'", "\u201d", "\u2019", ")", "]", "}", "\n",
                         "*", "_", "~", "\u2014", "\u2013", ":", ";", "\u00bb", "`"]
        self.clean_end_chars: Tuple[str, ...] = tuple(str(c) for c in clean_end if str(c))

        # Hard cap on upstream stream attempts per generation, so a backend that
        # closes the socket without [DONE] cannot spin forever.
        #
        # This budget covers *ordinary* attempts only: a backend stop that had to be
        # retried, a dropped connection, an empty chunk.  A restart entered to
        # resume after a back-track is progress by definition -- the token was
        # rewritten and generation moved on -- so it is counted against
        # `max_backtrack_restarts` instead.
        #
        # Sharing one counter between the two made a slop-heavy response end with
        # finish_reason='length' after a dozen back-tracks, mid-sentence, with text
        # still to come: the `...she whim` / `...the terror` truncations.
        self.max_stream_attempts = max(
            2, int(gen.get("max_stream_attempts", self.max_consecutive_stops * 4))
        )
        self.max_backtrack_restarts = max(
            2, int(gen.get("max_backtrack_restarts", 200))
        )

        # Ask the backend to suppress end-of-turn markers on *restart* requests.
        # After a premature EOS the continuation prompt ends exactly where the
        # model already chose to stop, so the same token is again the argmax and
        # the sampler would just burn its restart budget.  Backends that reject
        # `bad_words` disable this automatically (see api_client).
        #
        # Off by default.  `bad_words` holds for the whole request, so a model
        # whose stop was refused *cannot* end its turn on that attempt: a reply
        # that was in fact finished is padded with whatever it writes instead
        # (stock phrases, stage directions, a new turn).  When enabled, the ban
        # applies only to the one attempt that follows a refused stop, and that
        # attempt is capped at `restart_suppress_window` tokens.
        self.restart_suppress_markers = bool(gen.get("restart_suppress_markers", False))
        self.restart_suppress_window = max(1, int(gen.get("restart_suppress_window", 32)))

        # Text-level scrubber for control tokens that leak into the output.
        self.special_token_pattern = gen.get("special_token_pattern")

        # The model's own control tokens, read from the chat-template tokenizer.
        # The built-in pattern names a fixed set of families and missed 13 of
        # Gemma 4's 24 (`<|channel>`, `<channel|>`, `<|think|>`, `<|"|>`, ...),
        # while deleting harmless prose such as a literal "<mask>".  When the
        # tokenizer is known its list is used verbatim instead of the pattern.
        self.control_tokens: List[str] = []
        tok = getattr(chat_template_formatter, "tokenizer", None)
        if tok is not None:
            try:
                specials = set(getattr(tok, "all_special_tokens", []) or [])
                for added in (getattr(tok, "added_tokens_decoder", {}) or {}).values():
                    if getattr(added, "special", False):
                        specials.add(added.content)
                self.control_tokens = sorted(
                    t for t in specials
                    if isinstance(t, str) and len(t) > 2 and t.startswith("<") and t.endswith(">")
                )
            except Exception as e_tok:
                logger.warning("Could not read special tokens from the tokenizer: %s", e_tok)

        # Extra sampling fields passed through to the backend unchanged
        # (repetition_penalty, presence_penalty, frequency_penalty).  Set by the
        # API server from the client's request.
        self.extra_request_params: Dict[str, Any] = {}

        # Real reason the generation ended; surfaced by the API server so a
        # truncated record is distinguishable from a complete one.
        self.finish_reason: Optional[str] = None
        # Counts attempts that produced no new tokens (infinite-loop guard).
        self._no_progress_attempts = 0

        # Soft-ban strength (0..1). Applies to phrase/ngram, not regex.
        try:
            self.ban_strength = float(config.get("ban_strength", 1.0))
        except Exception:
            self.ban_strength = 1.0
        self.ban_strength = max(0.0, min(1.0, self.ban_strength))

        # Opt-in dump of unresolvable backtrack positions (debugging aid only).
        self.debug_fail_dump = bool(config.get("debug_fail_dump", False))


        # API retry policy (sampler-level)
        self.api_retry_attempts = int(gen.get("api_retry_attempts", 5))
        self.api_retry_delay_sec = float(gen.get("api_retry_delay_sec", 1.0))

        back = config.get("backtracking", {})
        self.max_retries_per_position = back.get("max_retries_per_position", 100)

        self.slop_info: Optional[Dict[str, object]] = None
        for v in validators:
            if isinstance(v, SlopPhraseValidator):
                self.slop_info = {
                    "keys": v.slop_phrases_keys,
                    "max_len": v.max_phrase_len,
                    "min_len": v.min_phrase_len,
                }
                break

        # Remember which alternative tokens we've already tried at each
        # generated position so we never pick the same word twice.
        # Key: generated_index (int) → set of raw token strings tried
        self._tried_alternatives = {}

        # Token indices at which each (validator, rule) was banned in this run,
        # to recognise a phrase the model keeps coming back to.
        self._ban_history: Dict[Tuple[str, str], List[int]] = {}

        self.events: List[Dict[str, object]] = []

        # ------------------------------------------------------------------ #
        #  Tail-buffer size: keep enough tokens so that *any* validator       #
        #  (regex, n-gram, phrase) can still match across a chunk boundary.   #
        #                                                                     #
        #  • Regex   → unlimited in principle → use a fixed safety window     #
        #                that you can raise in config.yaml:                   #
        #      regex_max_span_tokens: 999999    # default                        #
        #  • N-gram  → v.max_ngram_len tokens                                 #
        #  • Phrase  → (max_phrase_len + scan_window_base_size)/4 chars ≈     #
        #                number of tokens (rough estimate)                    #
        # ------------------------------------------------------------------ #
        regex_tail = int(config.get("regex_max_span_tokens", 999999))
        tail_tokens = regex_tail

        for v in validators:
            if v.__class__.__name__ == "NGramValidator":
                tail_tokens = max(tail_tokens, getattr(v, "max_ngram_len", 0) + 2)
            elif v.__class__.__name__ == "SlopPhraseValidator":
                est = (v.max_phrase_len + v.scan_window_base_size) // 4 + 2
                tail_tokens = max(tail_tokens, est)

        self.tail_keep_tokens = tail_tokens


        # ── tail prefix-filter (HF tokenizer only, parallel build) ──────────
        self.filter_tail_banned_prefix_tokens = bool(
            config.get("filter_tail_banned_prefix_tokens", True)
        )
        self._banned_prefix_token_strings: frozenset[str] = frozenset()

        if self.filter_tail_banned_prefix_tokens:
            if chat_template_formatter is None or not hasattr(chat_template_formatter, "model_id"):
                # A sampler is built per request in server mode, so this used to be
                # logged once per request.  Warn once per process instead.
                global _TAIL_PREFIX_WARNED
                if not _TAIL_PREFIX_WARNED:
                    _TAIL_PREFIX_WARNED = True
                    logger.warning(
                        "tail-prefix filter requested but no chat-template tokenizer "
                        "available – skipping filter (this warning is logged once)."
                    )
                self.filter_tail_banned_prefix_tokens = False
            else:
                cache_key = chat_template_formatter.model_id
                with _BANNED_PREFIX_LOCK:
                    cached = _BANNED_PREFIX_CACHE.get(cache_key)
                    if cached is None:                               # first time ever
                        start = time.time()
                        cached = _build_banned_prefix_set(chat_template_formatter, self.validators)
                        _BANNED_PREFIX_CACHE[cache_key] = cached
                        logger.info(
                            "Built banned-prefix token set in %.1fs (%d prefixes).",
                            time.time() - start, len(cached),
                        )
                self._banned_prefix_token_strings = cached





        logger.info(
            f"ApiAntiSlopSampler ready — chunk={self.chunk_size}, "
            f"max_new_tokens={self.max_new_tokens}, T={self.temperature}, "
            f"top_p={self.top_p}, top_k={self.top_k}, min_p={self.min_p}, "
            f"tail_keep_tokens={self.tail_keep_tokens}"
        )

        # NOTE: `request_mode` must be resolved *before* the round-trip warning
        # below, otherwise a stream-mode run is warned about chunk-mode prefill
        # amplification that does not apply to it.  The warning used to be emitted
        # unconditionally, so every stream-mode request logged it.
        self.request_mode = str(gen.get("request_mode", "chunk") or "chunk").strip().lower()
        self.regex_interval = int(config.get("regex_validation_interval", 20))

        # --- Upstream round-trip cost -------------------------------------
        # _generate_chunkwise() re-sends the whole prompt every chunk, so a
        # single request costs ~(max_new_tokens / chunk_size) sequential
        # prefill round-trips instead of one. That multiplies the backend
        # prefill load by a large factor and caps real throughput well below
        # the configured concurrency. Warn loudly rather than let it surprise
        # anyone sizing a deployment.  Stream mode is immune: it makes one
        # upstream request and does not re-send the prompt.
        if self.request_mode == "chunk" and self.chunk_size and self.max_new_tokens:
            round_trips = max(1, math.ceil(self.max_new_tokens / self.chunk_size))
            if round_trips >= 10:
                logger.warning(
                    f"chunk_size={self.chunk_size} with max_new_tokens={self.max_new_tokens} "
                    f"means ~{round_trips} sequential upstream round-trips per request, "
                    f"each re-sending the full prompt. This multiplies backend prefill "
                    f"cost by roughly {round_trips}x and will cap achievable throughput "
                    f"well below max_concurrent_requests. Consider a larger chunk_size or "
                    f"request_mode: stream."
                )

        if self.request_mode == "stream":
            if gen.get("probe_stream_logprobs", True):
                # The probe result is cached per endpoint: a sampler is created
                # per request in server mode, so an uncached probe meant one
                # extra round-trip *on every single request*.
                ok = self._probe_stream_logprobs()
                if not ok:
                    raise RuntimeError(
                        "request_mode=stream but backend did not return logprobs in SSE "
                        "events.  Use --request-mode=chunk instead."
                    )
            else:
                logger.info("Skipping streamed-logprobs probe (probe_stream_logprobs: false).")

    def _probe_stream_logprobs(self) -> bool:
        base_url = getattr(self.api_client, "base_url", None)
        if base_url is not None and base_url in _STREAM_PROBE_CACHE:
            return _STREAM_PROBE_CACHE[base_url]

        logger.info("Probing backend for streamed logprobs…")
        probe = None
        try:
            gen = self.api_client.generate_stream(
                prompt_text="Hello",
                max_tokens=1,
                top_logprobs=self.top_logprobs_count,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                timeout=15,
                stop_sequences=None,
            )
            try:
                probe = next(gen)
            finally:
                # Close the generator so the HTTP response is released; leaving
                # it open leaked a pooled connection on every probe.
                try:
                    gen.close()
                except Exception:
                    pass
            ok = bool(probe.logprobs)
        except Exception as e:
            logger.error("Probe failed: %s", e)
            ok = False

        if base_url is not None and ok:
            _STREAM_PROBE_CACHE[base_url] = True
        return ok


    def _run_validators(
        self, state: GenerationState, *, final: bool = False
    ) -> Optional[ViolationInfo]:
        """
        Query every validator and return the violation that starts at the
        **lowest token index**.  If two violations start at the same index
        we keep the one from the validator that appears first in
        self.validators so existing priority ties stay stable.

        *final* is forwarded to validators that understand it (phrase, n-gram).
        While the text is still growing they suppress matches that end at the end
        of the text, because the next token may still extend that word.  Without
        this, a ban on ``"la"`` fired the instant the model emitted ``" la"``,
        the back-tracker rewrote the token, the model re-emitted the same prefix,
        and generation looped at the same index — the repeated
        ``Back-tracking @tok=N orig=' la'`` pattern.
        """
        earliest: Optional[ViolationInfo] = None
        for v in self.validators:
            vio = self._check_one(v, state, final)
            if not vio:
                continue
            if (earliest is None) or (vio.violation_index < earliest.violation_index):
                earliest = vio
        return earliest

    @staticmethod
    def _check_one(
        validator: BaseValidator, state: GenerationState, final: bool
    ) -> Optional[ViolationInfo]:
        """Call *validator*, passing ``final`` only if it accepts the keyword.

        Third-party or older validators keep the single-argument signature; a
        TypeError from the keyword is not swallowed, so a genuine bug inside a
        validator still surfaces.
        """
        try:
            return validator.check(state, final=final)
        except TypeError as exc:
            if "final" not in str(exc):
                raise
            return validator.check(state)

    # ------------------------------------------------------------------ #
    #  Fast sanity-check for a candidate                                 #
    # ------------------------------------------------------------------ #
    def _check_hypothetical_state(
        self,
        state: GenerationState,
        truncate_idx: int,
        alt_token: str,
        final: bool = False,
    ) -> tuple[bool, tuple[str, str] | None]:
        """
        (passes_all, (validator_name, detail_key) | None)

        A violation blocks the candidate IFF the validator’s affected-token
        span includes the *candidate index*.

        For single-token validators (slop phrase, regex char map, …) that means
        `violation_index == truncate_idx`.

        For N-gram validator we treat the span as
            [violation_index, violation_index + len(ngram_tuple)-1].

        *final* is accepted for call-site compatibility but does not change the
        verdict: a candidate is always checked definitively.  See the comment
        below for why the streaming deferral must not apply here.
        """
        if not self.validators:
            return True, None

        # ⸺ build hypothetical state ------------------------
        # Create a minimal hypothetical state for validation
        # We need to properly initialize the decoded buffer for validators
        hypo_tokens = state.generated_token_strings[:truncate_idx] + [alt_token]
        # `hypothetical()` skips __init__'s logging: one back-track tests up to
        # `max_retries_per_position` candidates, and logging an initialised-state
        # line for each buried the actual events (20+ lines per back-track).
        hypo_state = GenerationState.hypothetical(state.prompt_string, hypo_tokens)

        # The candidate is checked with ``final=True`` even while the stream is
        # running, and this is deliberate.
        #
        # The deferral rule in the validators exists so the *live* stream does not
        # ban a half-written word: ``" la"`` may still grow into ``" land"``.  A
        # back-track candidate is a different situation -- we are choosing what to
        # put there, and the next validation pass will see the same text again.  A
        # candidate that already ends on a banned phrase is therefore not a fix, it
        # is the same violation one token later.
        #
        # Passing ``final=False`` here made every banned single word its own best
        # replacement: ``gripped`` was resampled as ``gripping`` (both banned),
        # then back to ``gripped``, burning the whole attempt budget and finally
        # emitting the banned word anyway -- the ``surge``/``spark`` and
        # ``gripped``/``gripping`` ping-pong, and the repeated
        # ``own own own`` text in the output.
        for v in self.validators:
            vio = self._check_one(v, hypo_state, True)
            if not vio:
                continue

            # ---------- does the violation involve the new token? ----------
            start = vio.violation_index
            end   = start                      # default: single token

            # n-gram gives us the length
            if v.__class__.__name__ == "NGramValidator":
                if isinstance(vio.details, dict) and "ngram_tuple" in vio.details:
                    end = start + len(vio.details["ngram_tuple"]) - 1

            # Ignore it if the span is wholly before the candidate position
            if end < truncate_idx:
                continue

            # otherwise block the candidate
            det = (
                (vio.details.get("ngram_string")
                or vio.details.get("phrase")
                or vio.details.get("pattern"))
                if isinstance(vio.details, dict) else "?"
            )
            return False, (v.__class__.__name__, det)

        return True, None



    

        # ------------------------------------------------------------------ #
    #  Back-tracking                                                     #
    # ------------------------------------------------------------------ #
    def _perform_backtrack(
        self,
        state: GenerationState,
        vio: ViolationInfo,
        final: bool = False,
    ) -> bool:
        """Resample the violating token from its cached top-logprob list.

        *final* says the text will not grow any more (the post-generation sweep).
        It is forwarded to the candidate check so that a replacement which merely
        *looks* fine because the word is unfinished is not accepted: in the final
        sweep there is no next chunk to complete it, so a match touching the end
        of the text is definitive.
        """
        idx         = vio.violation_index
        banned_token   = vio.original_token_string
        lp_list     = state.get_logprobs(idx)

        def _abort() -> bool:
            self._suppress_violation(vio)   # ← remember: vio is already in scope
            return False                       # [(token, logp)]

        # A rewrite needs the model to regenerate everything after the rewritten
        # token, because those tokens were sampled given the *old* one.  After
        # generation has stopped there is no next request, so rewriting one token
        # in place and keeping the stale tail splices two continuations together:
        # "It was not fear, but hunger." -> "It was. fear, but hunger.", and every
        # n-gram fix became "took deep breath".  Deleting the tail instead (the
        # behaviour before 2026.11.02.1930) truncated the turn.  Neither is
        # acceptable in a dataset, so a violation that is still present once
        # generation is over is left as written.  The generation loops resolve
        # end-of-text matches *before* accepting a stop, while they can still
        # regenerate (see _resolve_before_stop).
        if final:
            logger.warning(
                "Violation %r at token %d found after generation ended; leaving "
                "the text as written rather than splicing in a replacement.",
                vio.details.get("phrase") or vio.details.get("ngram_string")
                or vio.details.get("pattern") if isinstance(vio.details, dict) else vio.details,
                idx,
            )
            return _abort()

        if not lp_list:
            logger.error("Back-track failed — no logprobs available.")
            return _abort()

        # soft-ban strength; 1.0 = hard ban, 0.0 = no penalty
        try:
            ban_s = float(getattr(self, "ban_strength", 1.0))
        except Exception:
            ban_s = 1.0
        ban_s = _scale_ban_strength(max(0.0, min(1.0, ban_s)))
        include_banned_scaled_global = (vio.validator_type in {"slop_phrase", "ngram"} and ban_s < 1.0)


        # A phrase the context forces (a character called Mara against the
        # one-word entry "mara", a place name, a word the user just used) cannot
        # be avoided: each replacement is a junk token, the model tries the word
        # again one token later, and the reply fills with the junk
        # ("**\\\n\\\n\\\n..." in a real Gemma 4 run).  The per-position budget
        # never trips because every retry lands on a new index.  After three bans
        # of the same rule within 12 tokens, roll back to where the first one
        # started and allow the phrase for the rest of this request.
        rule = (
            (vio.details.get("suppression_detail_key") or vio.details.get("phrase")
             or vio.details.get("ngram_string") or vio.details.get("pattern"))
            if isinstance(vio.details, dict) else str(vio.details)
        )
        hist = self._ban_history.setdefault((vio.validator_type, str(rule)), [])
        hist.append(idx)
        if len(hist) >= 3 and max(hist[-3:]) - min(hist[-3:]) <= 12:
            first = min(hist[-3:])
            logger.warning(
                "Back-track: %r was banned 3 times within %d tokens; the context "
                "forces it.  Rolling back to token %d and allowing it for the rest "
                "of this request.", rule, max(hist[-3:]) - first, first,
            )
            for v in self.validators:
                if getattr(v, "validator_type", None) == vio.validator_type:
                    v._ignored.add((None, str(rule)))
            state.truncate(first)
            for k in list(self._tried_alternatives):
                if k >= first:
                    del self._tried_alternatives[k]
            return True

        tried_here   = self._tried_alternatives.setdefault(idx, set())
        logger.warning(
            f"Back-tracking @tok={idx} orig='{banned_token}' ban='{vio.details}'."
        )

        # Enforce the per-position budget.  `max_retries_per_position` was read
        # from config and never checked, so a position whose candidate list was
        # entirely banned could be re-entered without limit: each pass resampled
        # from the same cached logprobs, re-fired the same violation, and
        # generation made no progress (the repeated `Back-tracking @tok=N
        # orig=' la'` lines in the log).
        if len(tried_here) >= self.max_retries_per_position:
            logger.error(
                "Back-track: exhausted %d alternatives at token %d; "
                "suppressing the violation and continuing.",
                self.max_retries_per_position, idx,
            )
            return _abort()

        # ── utility: validate a single token once per call ───────────────
        validation_cache = {}              # tok_str -> bool
        def _is_valid(tok_str: str) -> bool:
            if tok_str in validation_cache:
                return validation_cache[tok_str]
            ok, _ = self._check_hypothetical_state(state, idx, tok_str, final=final)
            validation_cache[tok_str] = ok
            return ok

        # normalise “is banned” across raw/decoded to avoid mismatches.
        banned_decoded_lc = _decode_token(banned_token).lower()
        def _is_same_banned(tok_str: str) -> bool:
            if tok_str == banned_token:
                return True
            try:
                return _decode_token(tok_str).lower() == banned_decoded_lc
            except Exception:
                return False

        # ── build base (token, prob) list after temperature softmax ──────
        logits         = [lp for _, lp in lp_list]
        raw_p          = [math.exp(l) for l in logits]
        tempered       = [p ** (1 / self.temperature) for p in raw_p]
        base_weighted_pairs = [
            (tok, pt) for (tok, _), pt in zip(lp_list, tempered)
        ]

        # ─────────────────────────────────────────────────────────────────
        #  1. NEXT-TOKEN SELECTION  (honours --force-backtrack)
        #      NEW: for phrase/ngram with ban_strength<1 keep a downscaled
        #      banned token in the pool; if it wins we allow+supress+recheck.
        # ─────────────────────────────────────────────────────────────────

        def _build_pairs(temp, min_p, top_p, top_k):
            """Return candidate list [(tok, prob)] after all filters."""
            logits  = [lp for _, lp in lp_list]
            raw_p   = [math.exp(l) for l in logits]
            probs_t = [p ** (1.0 / max(temp, 1e-6)) for p in raw_p]

            # Establish the original sampler support before applying the
            # post-hoc ban.  Otherwise rejecting a dominant top token lowers
            # min-p's reference probability and admits tokens that normal
            # decoding could never have selected.
            pairs = apply_sampling_filters(
                [(tok, pt) for (tok, _), pt in zip(lp_list, probs_t)],
                min_p=min_p,
                top_p=top_p,
                top_k=top_k,
            )

            constrained_pairs = []
            for tok, pt in pairs:
                # vLLM's logprobs are computed before bad_words and stop handling,
                # so the alternatives list end-of-turn and other control tokens.
                # Choosing one puts e.g. `<turn|>` inside the assistant text, which
                # the next restart sends upstream as a real end of turn.
                if self._is_end_of_turn_token(tok):
                    continue
                if _is_same_banned(tok):
                    if include_banned_scaled_global:
                        pt *= (1.0 - ban_s)   # NEW: soft-ban scaling
                    else:
                        continue              # hard exclude
                constrained_pairs.append((tok, pt))

            return constrained_pairs

        # ── attempt ladder: default ⇒ relax T ⇒ drop min_p ⇒ drop top_p ⇒ drop top_k
        attempts = [
            (self.temperature, self.min_p, self.top_p, self.top_k),
        ]
        if getattr(self, "force_backtrack", False):
            attempts += [
                (1.0,              self.min_p, self.top_p, self.top_k),
                (1.0,              None,       self.top_p, self.top_k),
                (1.0,              None,       None,       self.top_k),
                (1.0,              None,       None,       None),
            ]

        valid_pairs = []
        for relax_idx, (temp, min_p, top_p, top_k) in enumerate(attempts):
            pairs = _build_pairs(temp, min_p, top_p, top_k)
            if not pairs:
                continue

            # NEW: allow the banned token only when soft-ban is active
            valid_pairs = []
            for tok, pt in pairs:
                if include_banned_scaled_global and _is_same_banned(tok):
                    valid_pairs.append((tok, pt))
                else:
                    if pt > 0 and tok not in tried_here and _is_valid(tok):
                        valid_pairs.append((tok, pt))
            if valid_pairs:
                break

        if not valid_pairs:
            # NOTE: this used to append to a hard-coded
            # /workspace/auto-antislop/fails.txt.  That path does not exist in a
            # normal deployment, so the debug write raised FileNotFoundError
            # *inside* error handling and killed the generation.  It is now opt-in
            # and writes next to the working directory.
            if self.debug_fail_dump:
                try:
                    dump_path = os.path.join(os.getcwd(), "antislop_backtrack_fails.txt")
                    with open(dump_path, "a", encoding="utf-8") as f:
                        f.write(
                            banned_token + ", " + str(self.min_p) + ", "
                            + str(self.top_logprobs_count) + "\n"
                            + json.dumps(lp_list) + "\n\n"
                        )
                except Exception as e_dump:
                    logger.debug("Could not write backtrack fail dump: %s", e_dump)
            logger.error("Back-track: no valid next-token candidates found.")
            return _abort()

        # sample replacement (or possibly the banned token if soft-ban is active)
        tokens, probs = zip(*valid_pairs)
        choice = random.choices(tokens, weights=probs, k=1)[0]

        # NEW: if soft-ban didn't stick, allow & suppress then re-run validators
        if include_banned_scaled_global and _is_same_banned(choice):
            self._suppress_violation(vio)
            next_vio = self._run_validators(state)
            if not next_vio:
                return True
            # continue fixing via the outer driver; we handled this position
            return True

        # ─────────────────────────────────────────────────────────────────
        # 2. TAIL-CANDIDATE SELECTION
        #     Here we are choosing tokens from the (short) tail of the 
        #     top_probs returned by the api. These become the "chosen"
        #     tokens in the chosen/rejected pairs for the FTPO dataset.
        # ─────────────────────────────────────────────────────────────────
        max_tail   = getattr(self, "max_chosen_tokens", 20)
        #tail_min_p = getattr(self, "tail_min_p", 0.03)
        tail_top_k = getattr(self, "tail_top_k", 50)
        if self.min_p:
            tail_min_p = self.min_p
        else:
            tail_min_p = 0.01

        tail_toks: list[str] = []
        if max_tail > 0:
            # Apply tail sampling filters before excluding the rejected token,
            # for the same reason as replacement selection above.
            tail_pairs = apply_sampling_filters(
                base_weighted_pairs,
                min_p=tail_min_p,
                top_p=None,
                top_k=tail_top_k,
            )

            # sort ascending prob ⇒ lowest-prob tokens first
            tail_pairs.sort(key=lambda tp: tp[1])

            #print('filtering tail tokens for rejected token:', banned_token)
            for tok, _ in tail_pairs:
                if (self.filter_tail_banned_prefix_tokens
                    and (tok.lower() in self._banned_prefix_token_strings or _decode_token(tok).lower() in self._banned_prefix_token_strings)):
                    #print('skipping', tok)
                    continue                      # skip prefixes of banned strings
                if '*' in tok:
                    # for some models, * is such a common continuation token that allowing
                    # it in chosen_ids leads to the model having major * repetition issues
                    continue
                # skip tokens that are <= 1 char after stripping whitespace
                if len(tok.strip()) <= 1:
                    continue
                
                # skip tokens that don't contain any alphanumeric chars
                if not any(c.isalnum() for c in tok):
                    continue
                if _is_same_banned(tok) or tok in tried_here:   # NEW: consistent skip
                    continue
                if self._is_end_of_turn_token(tok):
                    continue
                if not _is_valid(tok):
                    continue
                

                
                #print('adding', tok)
                tail_toks.append(tok)
                if len(tail_toks) >= max_tail:
                    break

        multi_chosen_decoded = [_decode_token(t) for t in tail_toks]

        # ── commit replacement ───────────────────────────────────────────
        #
        # Mid-generation the tail is *discarded on purpose*: the rewrite invalidates
        # everything after it, and the caller immediately asks the backend to
        # regenerate from that point.  Nothing is lost, because it is about to be
        # written again.
        #
        # The post-generation sweep is different: generation has already stopped, so
        # there is no next request.  Truncating there deleted every token after the
        # repaired one and nothing ever replaced them -- a complete, clean turn ended
        # mid-sentence at the last repaired slop word.  Against a 3 500-entry ban list
        # the sweep repairs something on most slop-heavy turns, which is why the tail
        # of the record was the part that went missing.
        #
        # In the sweep we therefore rewrite the one token and keep the rest.  The
        # tokens after it were produced under the *old* token, so the join can read
        # slightly oddly; that is strictly better than losing the remainder of the
        # turn, and it is what a reader of the dataset would expect.  A rewritten
        # token can also expose a new violation further along, which the sweep's
        # next pass handles.
        tried_here.add(choice)
        state.truncate(idx + 1)
        state.replace_token_string(idx, choice)

        # purge caches beyond idx
        for k in list(self._tried_alternatives):
            if k > idx:
                del self._tried_alternatives[k]
        # Suppressions past idx refer to text that no longer exists.  Left in
        # place, a regenerated occurrence of the same phrase that happens to land
        # on the same token index would be let through silently.
        for v in self.validators:
            ignored = getattr(v, "_ignored", None)
            if ignored:
                v._ignored = {key for key in ignored if key[0] is None or key[0] <= idx}

        logger.warning(
            f"    ✓ replacement='{choice}' "
            f"(T={self.temperature}, min_p={self.min_p}, "
            f"top_p={self.top_p}, top_k={self.top_k})"
        )

        # ── ftpo sample capture (legacy + new multi-fields) ──────────────
        if tail_toks:
            try:
                gen_so_far_tokens = state.generated_token_strings[:idx]
                gen_so_far_text   = _tokens_to_text(gen_so_far_tokens)

                context_chat = state.prompt_string + gen_so_far_text

                self.ftpo_samples[context_chat] = {
                    "prompt_raw":       state.prompt_string,
                    "generation_raw":   gen_so_far_text,
                    "context_with_chat_template": context_chat,

                    # legacy single-token keys
                    "chosen_decoded":   _decode_token(choice),
                    "rejected_decoded": _decode_token(banned_token),
                    "chosen_raw":       choice,
                    "rejected_raw":     banned_token,

                    # NEW multi-token keys
                    "multi_chosen_decoded":  multi_chosen_decoded,
                    "multi_chosen_raw":      tail_toks,
                    "multi_rejected_decoded": [_decode_token(banned_token)],
                    "multi_rejected_raw":     [banned_token],

                    "validator": {
                        "class": vio.validator_type,
                        "rule": (
                            vio.details.get("phrase")
                            or vio.details.get("ngram_string")
                            or vio.details.get("pattern")
                            or ""
                        ),
                        "subtype": (
                            vio.validator_type
                            if vio.validator_type != "ngram"
                            else (
                                "trigram"
                                if len(vio.details.get("ngram_tuple", [])) == 3
                                else "bigram"
                            )
                        ),
                    },
                    "stats": {},
                }
            except Exception as e_log:
                logger.error(f"ftpo-pair capture failed: {e_log}", exc_info=True)

        return True





    def _record_event(self, vio: ViolationInfo, fixed: bool) -> None:
        event = {
            "type":  vio.validator_type,
            "index": vio.violation_index,
            "details": vio.details,
            "original_token_string": vio.original_token_string,
            "fixed": fixed,
        }
        self.events.append(event)
        if self.on_ban_event_callback:
            try:
                self.on_ban_event_callback(event)
            except Exception as e_cb:
                logger.error("Error in on_ban_event_callback: %s", e_cb, exc_info=True)

    def _resolve_before_stop(self, state: GenerationState, emitted_upto: int,
                             can_restart: bool) -> bool:
        """Settle end-of-text violations before a stop is accepted.

        While the text grows, a match that touches its end is provisional (the
        next token may extend the word), and the n-gram validator defers almost
        everything.  Once the model stops, those matches are definitive.  This is
        the last moment a back-track can still be followed by regeneration, so
        fix them here rather than in a post-generation sweep.

        Returns True when a token was rewritten and the tail truncated: the
        caller must resume generation from there instead of ending the turn.
        A violation that cannot be regenerated (already emitted, or no restart
        budget left) is suppressed and the text is left as written.
        """
        for _ in range(64):
            vio = self._run_validators(state, final=True)
            if not vio:
                return False
            if can_restart and vio.violation_index >= emitted_upto:
                fixed = self._perform_backtrack(state, vio)
                self._record_event(vio, fixed)
                if fixed:
                    return True
                continue          # _perform_backtrack suppressed it
            self._suppress_violation(vio)
            self._record_event(vio, False)
        return False

    def _suppress_violation(self, vio: ViolationInfo) -> None:
        for v in self.validators:
            if getattr(v, "validator_type", None) == vio.validator_type:
                v.ignore_violation(vio)

    def _yield_text_and_callback(self, text_to_yield: str) -> str:
        """Helper to count tokens, call callback, and yield text."""
        if self.on_chunk_yielded_callback:
            try:
                num_tokens = len(self.tiktoken_encoding.encode(text_to_yield))
                self.on_chunk_yielded_callback(text_to_yield, num_tokens)
            except Exception as e_cb:
                logger.error(f"Error in on_chunk_yielded_callback: {e_cb}", exc_info=True)
        return text_to_yield

    def _run_validators_skip_regex(self, state, *, final: bool = False):
        earliest = None
        for v in self.validators:
            if v.__class__.__name__ == "RegexValidator":
                continue
            vio = self._check_one(v, state, final)
            if vio and (earliest is None or vio.violation_index < earliest.violation_index):
                earliest = vio
        return earliest

    # ------------------------------------------------------------------ #
    #  End-of-turn markers, scrubbing and stop policy                    #
    # ------------------------------------------------------------------ #
    def _new_scrubber(self) -> SpecialTokenScrubber:
        """Fresh text-level scrubber for one generation run."""
        pattern = self.special_token_pattern
        if pattern is None and self.control_tokens:
            pattern = ""          # exact list from the tokenizer instead
        return SpecialTokenScrubber(
            markers=list(self.end_of_turn_markers) + self.control_tokens,
            pattern=pattern,
        )

    def _is_end_of_turn_token(self, token: str) -> bool:
        """True if *token* is a control/end-of-turn token.

        Matching happens on the raw form, the stripped form and the decoded form
        because the same special token is rendered differently by different
        backends (canonical name vs. tokenizer-internal spelling).
        """
        if not token:
            return False
        scrubber = getattr(self, "_probe_scrubber", None)
        if scrubber is None:
            scrubber = self._new_scrubber()
            self._probe_scrubber = scrubber
        return scrubber.is_marker(token)

    def _sanitize_chunk_eos(self, chunk: "ApiChunkResult") -> int:
        """Remove control tokens from *chunk* **in place**, anywhere in it.

        The previous version only looked at the *tail* of
        ``chunk.token_strings``, so a chunk shaped like
        ``["text", "<turn|>", "\n"]`` kept the marker.  Removal is now done at
        every position, and the per-token logprob map is re-indexed so
        back-tracking still sees the right alternatives.

        Returns the number of tokens removed.
        """
        if not chunk.token_strings:
            return 0

        kept: List[str] = []
        kept_logprobs: Dict[int, List[Tuple[str, float]]] = {}
        removed = 0

        for old_idx, tok in enumerate(chunk.token_strings):
            if self._is_end_of_turn_token(tok):
                removed += 1
                continue
            if old_idx in chunk.logprobs:
                kept_logprobs[len(kept)] = chunk.logprobs[old_idx]
            kept.append(tok)

        if removed:
            chunk.token_strings = kept
            chunk.logprobs = kept_logprobs
        return removed

    # -- evidence helpers ------------------------------------------------- #
    def _ends_cleanly(self, text: str) -> bool:
        """True when *text* ends on a natural boundary (sentence/paren/newline).

        Used to distinguish "the model finished its turn" from "the model
        emitted EOS mid-sentence", which is the failure mode that truncated
        ~35% of a temperature-1.0 Gemma dataset.
        """
        if not text:
            return False
        stripped = text.rstrip()
        if not stripped:
            # Only trailing whitespace: a newline is a legitimate end.
            return text.endswith("\n")
        if not self.stop_on_clean_end:
            return False
        if stripped.endswith(self.clean_end_chars):
            return True
        # An emoji or other symbol (\u2665, \U0001F60F) is a deliberate ending.
        last = stripped[-1]
        if last in "\ufe0f\u200d" and len(stripped) > 1:
            last = stripped[-2]
        return unicodedata.category(last) in ("So", "Sk", "Sm")

    def _hit_stop_sequence(self, text: str) -> bool:
        if not self.stop_sequences or not text:
            return False
        return any(s and s in text for s in self.stop_sequences)

    def _handle_stop(
        self,
        reason: Optional[str],
        had_eos_marker: bool = False,
        attempt_text: str = "",
        produced_tokens: bool = True,
    ) -> bool:
        """Decide how to react to a backend stop signal.

        Returns True to accept the stop (terminate the run), False to reject it
        and keep generating.

        The decision deliberately does **not** require having seen a marker
        token.  vLLM detokenises the output and drops the stop token before
        serialising a streamed chunk, so a premature ``<end_of_turn>`` arrives
        as a plain ``finish_reason="stop"`` with no marker anywhere.  Keying on
        the marker meant the old code always accepted such a stop, truncating
        the answer — which is exactly the reported bug.

        Rules:
          * no reason                → not a stop at all; keep generating.
          * ``length``               → per-request budget only; keep generating.
          * ``error`` / ``timeout``  → transport failure; keep generating (the
                                       caller retries) unless the attempt made
                                       progress and we are out of budget.
          * a user ``stop_sequence`` was found in ``attempt_text``
                                     → genuinely terminal.
          * ``stop`` / ``stream_end`` with text ending on a natural boundary
                                     → genuinely terminal.
          * ``stop`` / ``stream_end`` mid-sentence
                                     → premature EOS; keep generating until
                                       ``max_consecutive_stops`` *consecutive
                                       no-progress* attempts, then give up.
        """
        if not reason:
            return False

        self.finish_reason = reason

        if reason == "length":
            return False

        if reason in ("error", "timeout"):
            # Transport-level failure.  Never treat it as a completed
            # generation: the caller's retry loop resumes from the partial text.
            if not produced_tokens:
                self._no_progress_attempts += 1
            else:
                self._no_progress_attempts = 0
            return self._no_progress_attempts >= self.max_consecutive_stops

        # A configured stop sequence is an explicit instruction from the caller.
        if self._hit_stop_sequence(attempt_text):
            logger.info("Stop sequence matched; ending generation.")
            self._consecutive_stops = 0
            self._no_progress_attempts = 0
            return True

        if self._ends_cleanly(attempt_text):
            self._consecutive_stops = 0
            self._no_progress_attempts = 0
            return True

        # Mid-sentence stop: the model tried to end its turn too early.
        #
        # The strike counter only advances when the attempt produced *nothing*,
        # so a model that writes 500 good tokens and then emits a stray EOS is
        # not penalised for it.  That keeps the anti-infinite-loop guarantee
        # (a model that immediately re-emits EOS every time still terminates)
        # without truncating otherwise-healthy generations.
        if produced_tokens:
            self._consecutive_stops = 0
        self._consecutive_stops += 1

        if self._consecutive_stops < self.max_consecutive_stops:
            logger.warning(
                "Premature end of turn (finish_reason=%r, %d/%d consecutive "
                "no-progress stops, marker_seen=%s); continuing generation.",
                reason, self._consecutive_stops, self.max_consecutive_stops,
                had_eos_marker,
            )
            return False

        logger.warning(
            "Giving up on premature end of turn after %d consecutive "
            "no-progress stops (finish_reason=%r).",
            self.max_consecutive_stops, reason,
        )
        self._consecutive_stops = 0
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_p: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Generates text, handling validation and back-tracking.

        Args:
            prompt: The initial text to generate from.
            max_new_tokens: Optional override for the maximum number of new tokens.
            temperature: Optional override for the sampling temperature.
            min_p: Optional override for the min_p sampling parameter.
            top_p: Optional override for nucleus sampling (top-p).
            top_k: Optional override for top-k filtering.
        """
        # Determine effective generation parameters, overriding defaults if provided.
        effective_max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        effective_temperature = self.temperature if temperature is None else temperature
        effective_min_p = self.min_p if min_p is None else min_p
        effective_top_p = self.top_p if top_p is None else top_p
        effective_top_k = self.top_k if top_k is None else top_k

        # Back-tracking resamples from cached logprobs with self.temperature /
        # min_p / top_p / top_k.  Those held the config-file values, so with a
        # client that sends its own settings (SDG does) a replacement token was
        # drawn under different sampling than the rest of the reply.
        self.temperature = max(effective_temperature, 1e-3)
        self.min_p = effective_min_p
        self.top_p = effective_top_p
        self.top_k = effective_top_k

        # fresh early-stop safety-valve budget for this generation run
        self._consecutive_stops = 0
        self._no_progress_attempts = 0
        self.finish_reason = None
        self._probe_scrubber = self._new_scrubber()

        if self.request_mode == "stream":
            yield from self._generate_streamwise(
                prompt,
                max_new_tokens=effective_max_new_tokens,
                temperature=effective_temperature,
                min_p=effective_min_p,
                top_p=effective_top_p,
                top_k=effective_top_k,
            )
        else:
            yield from self._generate_chunkwise(
                prompt,
                max_new_tokens=effective_max_new_tokens,
                temperature=effective_temperature,
                min_p=effective_min_p,
                top_p=effective_top_p,
                top_k=effective_top_k,
            )

    # ------------------------------------------------------------------ #
    #  Streaming generation with restart + retry                          #
    # ------------------------------------------------------------------ #
    def _generate_streamwise(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        min_p: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
    ):
        """Generate token-by-token over an SSE stream.

        Each pass of the outer loop is one *attempt*.  An attempt ends when the
        backend stops, when a validator fires (→ backtrack and restart), or when
        the transport fails (→ retry from the text produced so far).  Text is
        released only after passing through the scrubber and the tail hold-back,
        so neither a control token nor a half-written marker can escape.
        """
        state = GenerationState(prompt)
        last_yield = 0
        regex_since = 0
        scrubber = self._new_scrubber()

        # The stream holds back the last `holdback` characters so a marker split
        # across two chunks is never emitted.  That means the caller sees text
        # one holdback behind what has actually been generated.
        pending_scrub = ""

        def release(text: str, final: bool = False) -> Optional[str]:
            """Push *text* through the scrubber; return what may be emitted."""
            nonlocal pending_scrub
            if final:
                pending_scrub += scrubber.flush()
                out, pending_scrub = pending_scrub, ""
            else:
                out = scrubber.feed(text)
            if not out:
                return None
            return out

        # Two independent budgets.  `attempts` bounds plain retries (a rejected
        # stop, a dropped connection); `backtrack_restarts` bounds restarts entered
        # to resume after a back-track, which always made progress.  Charging both
        # to one counter truncated healthy generations -- see the note in __init__.
        attempts = 0
        backtrack_restarts = 0
        # Length of the generated text at the last *accepted* attempt, used to
        # recognise a continuation that only repeats what is already there.
        accepted_text = ""
        # Fragments added by continuations whose stop was refused.  A model that has
        # already finished answers every request with the same stock phrase; keeping
        # them is what lets the second and later repetitions be recognised.
        refused_fragments: List[str] = []
        # True while the attempt in flight was entered to continue past a stop the
        # sampler refused.  Its text is held back until the stop is seen, so a
        # repetition can still be rolled back.
        continuation_attempt = False
        # Set when a stop is refused: the *next* attempt is a continuation and its
        # text must be held back until its own stop is seen.
        next_is_continuation = False
        # Set only when the backend *stopped* and the sampler refused the stop.
        # A `length` finish (the per-request budget, including the capped
        # suppressed attempt itself) also leads to a continuation, but the model
        # did not try to end its turn there, so it must not ban the marker.
        suppress_next = False
        transport_failed = False

        while state.get_generated_length() < max_new_tokens:
            # Each iteration starts as a fresh attempt.  The stop handler sets the
            # continuation flag for the attempt it asks for.
            continuation_attempt = next_is_continuation
            next_is_continuation = False
            suppress_this = suppress_next
            suppress_next = False
            attempts += 1
            if attempts > self.max_stream_attempts:
                logger.error(
                    "Reached max_stream_attempts=%d; ending generation with "
                    "%d tokens.", self.max_stream_attempts, state.get_generated_length(),
                )
                # Out of attempts because the backend kept failing is an error,
                # not a length cap: the client must not store the text as a reply.
                self.finish_reason = "error" if transport_failed else "length"
                break
            if backtrack_restarts > self.max_backtrack_restarts:
                logger.error(
                    "Reached max_backtrack_restarts=%d; ending generation with "
                    "%d tokens (finish_reason will be reported as 'length').",
                    self.max_backtrack_restarts, state.get_generated_length(),
                )
                self.finish_reason = "length"
                break

            restart_stream = False
            natural_end = False
            transport_failed = False
            last_reason: Optional[str] = None
            remaining = max_new_tokens - state.get_generated_length()
            api_prompt = state.prompt_string + state.get_generated_text()

            len_before_attempt = state.get_generated_length()
            attempt_text_start = len(state.get_generated_text())
            saw_eos_marker = False
            # Tokens this attempt added before any back-track rewrote them.  Used
            # to tell "the model continued the response" from "the model repeated
            # what it had already said" when a backend stop is rejected.
            new_token_strings: List[str] = []

            # Ask the backend not to re-emit the marker we just stripped.  After
            # a premature EOS the continuation prompt ends exactly where the
            # model already chose to stop, so without this the same token wins
            # the argmax again and every restart is wasted.
            # Only the attempt that directly follows a refused stop is affected.
            # This used to test `attempts > 1`, which stays true for the rest of
            # the generation after one refused stop (or one dropped connection),
            # so the model could never end its turn again and wrote until
            # max_tokens: the repetition loops and long tails in the dataset.
            suppress = None
            if (self.restart_suppress_markers and self.end_of_turn_markers
                    and suppress_this):
                if not getattr(self.api_client, "suppress_tokens_unsupported", False):
                    suppress = self.end_of_turn_markers
                    remaining = min(remaining, self.restart_suppress_window)

            stream_kwargs = dict(self.extra_request_params)
            if suppress:
                stream_kwargs["suppress_tokens"] = suppress

            # One cancellation handle per stream.  The client is shared across all
            # concurrent requests in server mode, so the old client-wide
            # `cancel_current_stream()` aborted *everyone's* stream whenever one
            # request backtracked — producing the "Stream closed without [DONE]"
            # storm and truncating healthy generations.
            cancel_token = None
            if hasattr(self.api_client, "new_stream_token"):
                cancel_token = self.api_client.new_stream_token()
                stream_kwargs["cancel_token"] = cancel_token

            stream = self.api_client.generate_stream(
                prompt_text     = api_prompt,
                max_tokens      = remaining,
                top_logprobs    = self.top_logprobs_count,
                temperature     = temperature,
                top_p           = top_p,
                top_k           = top_k,
                min_p           = min_p,
                timeout         = self.timeout,
                stop_sequences  = self.stop_sequences,
                **stream_kwargs,
            )

            try:
                for chunk in stream:
                    # Scrub control tokens anywhere in the chunk, before
                    # anything is appended or emitted.
                    if self._sanitize_chunk_eos(chunk):
                        saw_eos_marker = True

                    if not chunk.token_strings:
                        # Control-only chunk ([DONE], a final empty chunk, or a
                        # chunk that held nothing but markers).
                        if chunk.finish_reason:
                            last_reason = chunk.finish_reason
                            break
                        continue

                    pre_len = state.get_generated_length()
                    state.append_chunk(chunk)
                    delta = state.get_generated_length() - pre_len
                    regex_since += delta

                    if delta:
                        new_token_strings.extend(
                            state.generated_token_strings[pre_len:]
                        )

                    if delta and self.on_chunk_yielded_callback:
                        try:
                            # empty text → caller knows this count is
                            # "accepted, not yet flushed"
                            self.on_chunk_yielded_callback("", delta)
                        except Exception as e_cb:
                            logger.error(
                                "Throughput callback failed: %s", e_cb, exc_info=True
                            )

                    run_regex = regex_since >= self.regex_interval
                    vio = (
                        self._run_validators(state)
                        if run_regex
                        else self._run_validators_skip_regex(state)
                    )

                    # ---------- violation handling --------------------------
                    if vio:
                        if vio.validator_type != "regex":
                            vio_full = self._run_validators(state)
                            vio = vio_full or vio

                        # Cancel and drain THIS stream before backtracking so the
                        # HTTP connection is fully released.  Scoped to this
                        # stream only: cancelling every stream on the shared client
                        # broke concurrent requests.
                        if cancel_token is not None:
                            self.api_client.cancel_stream(cancel_token)
                        else:
                            self.api_client.cancel_current_stream()
                        try:
                            for _ in stream:
                                pass
                        except Exception:
                            pass

                        time.sleep(0.1)
                        fixed = self._perform_backtrack(state, vio)

                        logger.info(
                            "Backtracking complete (fixed=%s), resuming generation. "
                            "State: %d tokens, %d chars",
                            fixed, state.get_generated_length(),
                            len(state.get_generated_text()),
                        )

                        # Back-tracking rewrites the tail, so the "already emitted"
                        # watermark may now point past the end of the state.  Left
                        # alone it made the next safe-prefix slice empty for the
                        # rest of the run, silently dropping all later text.
                        if last_yield > state.get_generated_length():
                            last_yield = state.get_generated_length()

                        try:
                            self.api_client.reset_connection_pool()
                        except Exception as e:
                            logger.warning("Failed to reset connection pool: %s", e)

                        event = {
                            "type":  vio.validator_type,
                            "index": vio.violation_index,
                            "details": vio.details,
                            "original_token_string": vio.original_token_string,
                            "fixed": fixed,
                        }
                        self.events.append(event)
                        if self.on_ban_event_callback:
                            try:
                                self.on_ban_event_callback(event)
                            except Exception as e_cb:
                                logger.error(
                                    "Error in on_ban_event_callback: %s",
                                    e_cb, exc_info=True,
                                )

                        if not fixed:
                            self._suppress_violation(vio)

                        restart_stream = True
                        regex_since = 0
                        break

                    # ---------- yield the safe prefix ------------------------
                    # A *continuation* attempt (one entered because a previous
                    # backend stop was refused) is held back until its stop is
                    # seen, because it may turn out to be the model repeating
                    # itself.  Text released here could not be taken back, and
                    # the loop that produced `of course.` 327 times was emitted
                    # exactly this way.  Everything else streams normally.
                    if not continuation_attempt:
                        safe_upto = max(
                            0, state.get_generated_length() - self.tail_keep_tokens
                        )
                        if safe_upto > last_yield:
                            text_out = _tokens_to_text(
                                state.generated_token_strings[last_yield:safe_upto]
                            )
                            last_yield = safe_upto
                            if text_out:
                                emitted = release(text_out)
                                if emitted:
                                    yield self._yield_text_and_callback(emitted)

                    if run_regex:
                        regex_since = 0

                    # A terminal reason can ride on a chunk that also carries
                    # text; latch it and stop consuming this attempt.
                    if chunk.finish_reason:
                        last_reason = chunk.finish_reason
                        break

            except Exception as e_stream:
                # Transport failure mid-stream (timeout, dropped connection,
                # malformed event).  The partial text is already in `state`, so a
                # retry resumes from exactly that point instead of throwing the
                # generation away — this used to truncate the record outright.
                transport_failed = True
                logger.warning(
                    "Stream attempt %d failed after %d new tokens: %s",
                    attempts, state.get_generated_length() - len_before_attempt, e_stream,
                )
                if attempts >= self.api_retry_attempts + self.max_stream_attempts:
                    raise

            attempt_text = state.get_generated_text()[attempt_text_start:]
            produced_tokens = state.get_generated_length() > len_before_attempt

            # ---- decide what to do with this attempt --------------------
            if restart_stream:
                # Validator fired; backtrack already rewrote the tail.  This is a
                # restart after *progress* (a token was rewritten), so it is
                # charged to the back-track budget rather than the retry budget,
                # and the retry counter is refunded so a long series of fixes
                # cannot exhaust it.
                backtrack_restarts += 1
                attempts -= 1
                if not produced_tokens:
                    self._no_progress_attempts += 1
                else:
                    self._no_progress_attempts = 0
                if self._no_progress_attempts >= self.max_stream_attempts:
                    logger.error(
                        "Backtracking made no progress after %d attempts; "
                        "stopping to avoid an infinite loop.",
                        self._no_progress_attempts,
                    )
                    break
                continue

            if transport_failed:
                # Never treat a broken connection as a completed generation.
                if produced_tokens:
                    self._no_progress_attempts = 0
                    # Small backoff before resuming; the text so far is kept.
                    if self.api_retry_delay_sec:
                        time.sleep(self.api_retry_delay_sec)
                else:
                    self._no_progress_attempts += 1
                    if self.api_retry_delay_sec:
                        time.sleep(self.api_retry_delay_sec)
                    if self._no_progress_attempts >= self.max_consecutive_stops:
                        logger.error(
                            "Stream failed %d times with no progress; giving up.",
                            self._no_progress_attempts,
                        )
                        self.finish_reason = "error"
                        break
                continue

            reason = last_reason

            if reason is None:
                # The attempt ended without a terminal chunk.  Treat it as a
                # dropped connection and retry rather than silently truncating.
                logger.warning(
                    "Stream attempt %d ended without a finish_reason; retrying.",
                    attempts,
                )
                if produced_tokens:
                    self._no_progress_attempts = 0
                else:
                    self._no_progress_attempts += 1
                    if self._no_progress_attempts >= self.max_consecutive_stops:
                        self.finish_reason = "error"
                        break
                continue

            # The clean-end test looks at the whole reply, as chunk mode does.
            # Judging only this attempt's text made an immediate stop on a retry
            # (empty attempt) look like a mid-sentence stop, so a finished reply's
            # stop was refused again.
            if self._handle_stop(
                reason,
                had_eos_marker=saw_eos_marker,
                attempt_text=state.get_generated_text(),
                produced_tokens=produced_tokens,
            ):
                if self._resolve_before_stop(
                    state, last_yield,
                    backtrack_restarts < self.max_backtrack_restarts
                    and state.get_generated_length() < max_new_tokens,
                ):
                    backtrack_restarts += 1
                    attempts -= 1
                    regex_since = 0
                    accepted_text = state.get_generated_text()
                    continue
                # The stop is genuine, so anything held back during this attempt
                # was real continuation text after all.  Flush it now.
                if continuation_attempt:
                    safe_upto = max(
                        0, state.get_generated_length() - self.tail_keep_tokens
                    )
                    if safe_upto > last_yield:
                        text_out = _tokens_to_text(
                            state.generated_token_strings[last_yield:safe_upto]
                        )
                        last_yield = safe_upto
                        if text_out:
                            emitted = release(text_out)
                            if emitted:
                                yield self._yield_text_and_callback(emitted)
                natural_end = True
                break

            # Rejected stop.  Before resuming, decide whether this attempt was
            # actually a *continuation*.
            #
            # A rejected stop is normally the model stopping early, so the next
            # attempt writes the rest of the sentence.  But when the model simply
            # had nothing left to say, asking again produces the same stock
            # phrase over and over -- the `of course.` x327 turn in the dataset.
            # That text is emitted verbatim today, because a rejected stop is
            # reported as `produced_tokens=True` and therefore resets the strike
            # counter on every attempt.
            #
            # A continuation that adds no new words is a no-progress attempt: it
            # is rolled back and counted, so the `max_consecutive_stops` valve
            # documented in README can actually fire.
            added_text = _tokens_to_text(new_token_strings)
            if produced_tokens and _is_repetition(
                accepted_text, added_text, refused_fragments
            ):
                logger.warning(
                    "Attempt %d only repeated the text already generated; "
                    "rolling it back (%d/%d no-progress attempts).",
                    attempts, self._no_progress_attempts + 1,
                    self.max_consecutive_stops,
                )
                state.truncate(len_before_attempt)
                if last_yield > state.get_generated_length():
                    last_yield = state.get_generated_length()
                self._no_progress_attempts += 1
                if self._no_progress_attempts >= self.max_consecutive_stops:
                    logger.warning(
                        "Ending turn: the model repeats itself instead of "
                        "continuing (finish_reason=%r).", reason,
                    )
                    self.finish_reason = reason
                    break
                # The retry is still a continuation, so its text must be held back
                # as well; otherwise it is emitted before it can be judged.
                next_is_continuation = True
                continue

            if produced_tokens:
                self._no_progress_attempts = 0
                accepted_text = state.get_generated_text()
                if continuation_attempt and added_text.strip():
                    refused_fragments.append(added_text)
            else:
                self._no_progress_attempts += 1
                if self._no_progress_attempts >= self.max_stream_attempts:
                    logger.error(
                        "No progress after %d stream attempts; ending generation.",
                        self._no_progress_attempts,
                    )
                    break

            # The stop was refused, so the next attempt is a continuation and its
            # text must be held back until its own stop is seen.
            next_is_continuation = True
            suppress_next = reason in ("stop", "stream_end")

            # Final sweep: the tail may still violate a regex.
            if regex_since > 0:
                vio = self._run_validators(state)
                if vio:
                    time.sleep(0.1)
                    fixed = self._perform_backtrack(state, vio)
                    event = {
                        "type":  vio.validator_type,
                        "index": vio.violation_index,
                        "details": vio.details,
                        "original_token_string": vio.original_token_string,
                        "fixed": fixed,
                    }
                    self.events.append(event)
                    if self.on_ban_event_callback:
                        try:
                            self.on_ban_event_callback(event)
                        except Exception as e_cb:
                            logger.error(
                                "Error in on_ban_event_callback: %s",
                                e_cb, exc_info=True,
                            )
                    if not fixed:
                        self._suppress_violation(vio)
                    if last_yield > state.get_generated_length():
                        last_yield = state.get_generated_length()
                    regex_since = 0
                    # The back-track deleted part of the accepted text.  Judging
                    # the regenerated tail against the old text would roll back
                    # a rewrite of the same sentence as a "repetition".
                    accepted_text = state.get_generated_text()

        # ---------------- final validation sweep --------------------------
        # Validators defer a match that touches the end of the text while the
        # stream is still running, because the next token may extend that word.
        # Now that generation has stopped, that deferral has to be resolved: run
        # one definitive pass and fix whatever it finds.  Without this, a banned
        # phrase at the very end of the response would never be caught.
        #
        # The sweep repeats because fixing one violation can expose another
        # earlier in the rewritten tail, and each pass is bounded by the same
        # per-position budget the live loop uses.
        for _sweep in range(max(1, self.max_consecutive_stops)):
            vio = self._run_validators(state, final=True)
            if not vio:
                break
            time.sleep(0.05)
            fixed = self._perform_backtrack(state, vio, final=True)
            event = {
                "type":  vio.validator_type,
                "index": vio.violation_index,
                "details": vio.details,
                "original_token_string": vio.original_token_string,
                "fixed": fixed,
            }
            self.events.append(event)
            if self.on_ban_event_callback:
                try:
                    self.on_ban_event_callback(event)
                except Exception as e_cb:
                    logger.error(
                        "Error in on_ban_event_callback: %s", e_cb, exc_info=True,
                    )
            if not fixed:
                # Suppressed so the next pass does not report it again; the
                # loop would otherwise spin on an unfixable position.
                self._suppress_violation(vio)
            if last_yield > state.get_generated_length():
                last_yield = state.get_generated_length()

        # ---------------- flush tail --------------------------------------
        if state.get_generated_length() > last_yield:
            residual = _tokens_to_text(state.generated_token_strings[last_yield:])
            if residual:
                emitted = release(residual)
                if emitted:
                    yield self._yield_text_and_callback(emitted)
        tail = release("", final=True)
        if tail:
            yield self._yield_text_and_callback(tail)

    def _generate_chunkwise(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        min_p: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
    ) -> Generator[str, None, None]:
        """
        Stream text while validating the *entire* prompt + generation after
        every chunk, and log per-chunk timing data to CSV so throughput
        decay can be analysed afterwards.
        """
        # ── one-off timing setup ────────────────────────────────────────────
        if not hasattr(self, "_chunk_timings"):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = f"timings_antislop_{ts}.csv"
            self._timings_path = os.getenv("ANTISLOP_TIMINGS", default_path)
            self._chunk_timings: list[tuple[int, int, float, float]] = []  # chunk, ctx_len, api_s, val_s

        chunk_nr = 0
        state = GenerationState(prompt) # already templated upstream
        last_yield_idx = 0                         # first token not yet sent
        scrubber = self._new_scrubber()
        pending_scrub = ""
        # Text at the last accepted chunk, plus the fragments added by continuations
        # whose stop was refused.  Chunk mode has the same repetition loop as stream
        # mode: a backend stop is refused, the model is asked for more, and it
        # answers with a stock phrase.  See _is_repetition.
        accepted_text = ""
        refused_fragments: List[str] = []
        continuation_chunk = False
        next_is_continuation = False
        chunk_restarts = 0

        def release(text: str, final: bool = False) -> Optional[str]:
            nonlocal pending_scrub
            if final:
                pending_scrub += scrubber.flush()
                out, pending_scrub = pending_scrub, ""
            else:
                out = scrubber.feed(text)
            return out or None



        while True:
            # Each iteration starts as a fresh chunk.  The stop handler sets the
            # continuation flag for the chunk it asks for.
            continuation_chunk = next_is_continuation
            next_is_continuation = False

            # hard token limit
            if (
                max_new_tokens is not None
                and state.get_generated_length() >= max_new_tokens
            ):
                logger.info("Reached max_new_tokens.")
                break

            # build prompt for backend
            full_prompt = prompt + state.get_generated_text()
            logger.debug(
                f"Chunk {chunk_nr}: Making API call (prompt_len={len(full_prompt)} chars, "
                f"gen_len={state.get_generated_length()} tokens)"
            )

            # ── API call timing ─────────────────────────────────────────────
            attempt = 0
            while True:
                t0 = time.perf_counter()
                try:
                    chunk = self.api_client.generate_chunk(
                        prompt_text     = full_prompt,
                        max_tokens      = self.chunk_size,
                        top_logprobs    = self.top_logprobs_count,
                        temperature     = temperature,
                        top_p           = top_p,
                        top_k           = top_k,
                        min_p           = min_p,
                        timeout         = self.timeout,
                        stop_sequences  = self.stop_sequences,
                        **self.extra_request_params,
                    )
                    api_sec = time.perf_counter() - t0
                    break  # success
                except requests.exceptions.HTTPError as e:
                    resp_json = {}
                    if e.response is not None:
                        try:
                            resp_json = e.response.json()
                        except Exception:
                            pass
                    # Do not retry context-length overflows; just stop this prompt.
                    if (
                        e.response is not None
                        and e.response.status_code == 400
                        and "maximum context length" in str(resp_json.get("message", "")).lower()
                    ):
                        logger.warning(
                            "Generation aborted for this prompt – the model’s context window was exceeded: %s",
                            str(resp_json.get("message", "")).rstrip()
                        )
                        return  # exit the generator early
                    attempt += 1
                    if attempt <= self.api_retry_attempts:
                        logger.warning(
                            "API HTTPError (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, self.api_retry_attempts, e, self.api_retry_delay_sec
                        )
                        time.sleep(self.api_retry_delay_sec)
                        continue
                    logger.error("API HTTPError after retries: %s", e)
                    raise
                except Exception as e:
                    attempt += 1
                    if attempt <= self.api_retry_attempts:
                        logger.warning(
                            "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, self.api_retry_attempts, e, self.api_retry_delay_sec
                        )
                        time.sleep(self.api_retry_delay_sec)
                        continue
                    logger.error("API call failed after retries: %s", e)
                    raise

            api_sec = time.perf_counter() - t0

            # Strip any trailing end-of-turn marker token(s) BEFORE the chunk
            # enters the generation state, so they can never be flushed to the
            # output (fixes Bug 1).  The chunk may consist *solely* of such
            # token(s); remember whether we saw one so the stop signal below
            # is recognised as a premature end-of-turn emission.
            had_eos_marker = bool(self._sanitize_chunk_eos(chunk))

            if not chunk.token_strings:            # end-of-stream (or all-EOS)
                if chunk.finish_reason:
                    # `produced_tokens=False`: this chunk carried no text, so a
                    # mid-sentence stop here is a no-progress stop and must count
                    # towards the safety valve.  Passing True (as an earlier
                    # version of this fix did) made the valve unreachable and the
                    # loop could spin forever against a backend that keeps
                    # returning empty stop chunks.
                    if self._handle_stop(
                        chunk.finish_reason,
                        had_eos_marker=had_eos_marker,
                        attempt_text=state.get_generated_text(),
                        produced_tokens=False,
                    ):
                        break
                    # premature end-of-turn marker → continue the run
                    continue
                # No text and no reason: this used to end the generation
                # silently, which truncated the record.  Treat it as a failed
                # request and retry the chunk instead.
                logger.warning(
                    "Chunk %d returned no text and no finish_reason; retrying.",
                    chunk_nr,
                )
                self._no_progress_attempts += 1
                if self._no_progress_attempts >= self.max_consecutive_stops:
                    logger.error(
                        "No progress after %d empty chunks; ending generation.",
                        self._no_progress_attempts,
                    )
                    self.finish_reason = "error"
                    break
                continue

            # ── append & validate timing ────────────────────────────────────
            pre_len = state.get_generated_length()
            state.append_chunk(chunk)
            added_text = _tokens_to_text(state.generated_token_strings[pre_len:])

            t1 = time.perf_counter()
            #t = time.time()
            vio = self._run_validators(state)
            #print('val 1st run', time.time() - t)
            if vio:
                fixed = self._perform_backtrack(state, vio)

                event = {
                    "type":  vio.validator_type,
                    "index": vio.violation_index,
                    "details": vio.details,
                    "original_token_string": vio.original_token_string,
                    "fixed": fixed,
                }
                self.events.append(event)

                if self.on_ban_event_callback:
                    try:
                        self.on_ban_event_callback(event)
                    except Exception as e_cb:
                        logger.error(f"Error in on_ban_event_callback: {e_cb}",
                                    exc_info=True)

                if not fixed:
                    self._suppress_violation(vio)
            val_sec = time.perf_counter() - t1

            # record per-chunk stats
            self._chunk_timings.append(
                (chunk_nr, state.get_generated_length(), api_sec, val_sec)
            )
            chunk_nr += 1

            # ── token-throughput bookkeeping ───────────────────────────────
            post_len = state.get_generated_length()
            newly_accepted = post_len - pre_len
            if newly_accepted and self.on_chunk_yielded_callback:
                try:
                    self.on_chunk_yielded_callback("", newly_accepted)
                except Exception as e_cb:
                    logger.error("Throughput callback failed: %s", e_cb, exc_info=True)

            # ── safe-to-stream part ────────────────────────────────────────
            # A *continuation* chunk (one entered because the previous stop was
            # refused) is held back until its own stop is seen: it may be the model
            # repeating itself, and text already yielded cannot be taken back.
            if not continuation_chunk:
                safe_upto = max(0, post_len - self.tail_keep_tokens)
                if safe_upto > last_yield_idx:
                    safe_tokens = state.generated_token_strings[last_yield_idx : safe_upto]
                    last_yield_idx = safe_upto
                    emitted = release(_tokens_to_text(safe_tokens))
                    if emitted:
                        # Token accounting for chunk mode happens in the
                        # on_chunk_yielded_callback("", newly_accepted) call above;
                        # going through _yield_text_and_callback here would count the
                        # same tokens twice.
                        yield emitted

            # stop if backend signalled completion.
            # NOTE: finish_reason="stop" can be premature — Gemma emits its
            # end-of-turn marker mid-response ~20% of the time under sampling
            # (the marker was already stripped above).  _handle_stop() rejects
            # such early stops until the safety-valve limit
            # (max_consecutive_stops) is reached, so generation continues
            # instead of being cut off mid-sentence (fixes Bug 2).
            if chunk.finish_reason:
                if self._handle_stop(
                    chunk.finish_reason,
                    had_eos_marker=had_eos_marker,
                    attempt_text=state.get_generated_text(),
                    produced_tokens=newly_accepted > 0,
                ):
                    if self._resolve_before_stop(
                        state, last_yield_idx,
                        chunk_restarts < self.max_backtrack_restarts
                        and (max_new_tokens is None
                             or state.get_generated_length() < max_new_tokens),
                    ):
                        chunk_restarts += 1
                        accepted_text = state.get_generated_text()
                        continuation_chunk = next_is_continuation = False
                        continue
                    # The stop is genuine; flush whatever the continuation chunk
                    # was holding back.
                    if continuation_chunk:
                        safe_upto = max(0, post_len - self.tail_keep_tokens)
                        if safe_upto > last_yield_idx:
                            safe_tokens = state.generated_token_strings[last_yield_idx : safe_upto]
                            last_yield_idx = safe_upto
                            emitted = release(_tokens_to_text(safe_tokens))
                            if emitted:
                                yield emitted
                    break

                # Rejected stop: a continuation that only repeats what is already
                # there is a no-progress attempt, not a continuation.  Roll it
                # back so it never reaches the output.
                if newly_accepted > 0 and _is_repetition(
                    accepted_text, added_text, refused_fragments
                ):
                    logger.warning(
                        "Chunk %d only repeated the text already generated; "
                        "rolling it back (%d/%d no-progress attempts).",
                        chunk_nr, self._no_progress_attempts + 1,
                        self.max_consecutive_stops,
                    )
                    state.truncate(pre_len)
                    if last_yield_idx > state.get_generated_length():
                        last_yield_idx = state.get_generated_length()
                    self._no_progress_attempts += 1
                    if self._no_progress_attempts >= self.max_consecutive_stops:
                        logger.warning(
                            "Ending turn: the model repeats itself instead of "
                            "continuing (finish_reason=%r).",
                            chunk.finish_reason,
                        )
                        self.finish_reason = chunk.finish_reason
                        break
                    next_is_continuation = True
                    continue

                self._no_progress_attempts = 0
                accepted_text = state.get_generated_text()
                if continuation_chunk and added_text.strip():
                    refused_fragments.append(added_text)
                # rejected early end-of-turn stop → loop back and keep going
                next_is_continuation = True

            continuation_chunk = next_is_continuation
            next_is_continuation = False

        # ── final validation sweep ─────────────────────────────────────────
        # Same reasoning as in stream mode: a match touching the end of the text
        # was deferred while the text was still growing, so it must be resolved
        # definitively now that generation has stopped.
        for _sweep in range(max(1, self.max_consecutive_stops)):
            vio = self._run_validators(state, final=True)
            if not vio:
                break
            fixed = self._perform_backtrack(state, vio, final=True)
            event = {
                "type":  vio.validator_type,
                "index": vio.violation_index,
                "details": vio.details,
                "original_token_string": vio.original_token_string,
                "fixed": fixed,
            }
            self.events.append(event)
            if self.on_ban_event_callback:
                try:
                    self.on_ban_event_callback(event)
                except Exception as e_cb:
                    logger.error(f"Error in on_ban_event_callback: {e_cb}", exc_info=True)
            if not fixed:
                self._suppress_violation(vio)
            if last_yield_idx > state.get_generated_length():
                last_yield_idx = state.get_generated_length()

        # ── flush residual tail ────────────────────────────────────────────
        if state.get_generated_length() > last_yield_idx:
            residual = state.generated_token_strings[last_yield_idx :]
            emitted = release(_tokens_to_text(residual))
            if emitted:
                yield emitted
        tail = release("", final=True)
        if tail:
            yield tail

        # ── write timing CSV ───────────────────────────────────────────────
        if False: # for perf profiling
            try:
                with open(self._timings_path, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["chunk", "ctx_len", "api_sec", "validators_sec"])
                    w.writerows(self._chunk_timings)
                logger.info(f"Timing log written → {self._timings_path} "
                            f"({len(self._chunk_timings)} rows)")
            except Exception as e:
                logger.error(f"Could not write timing CSV: {e}")

        logger.info(
            "Generation finished. Total tokens in state=%d",
            state.get_generated_length(),
        )
