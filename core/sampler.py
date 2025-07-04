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

import logging
import math
import time
import random
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

import tiktoken # Added tiktoken

from api_client.base_client import BaseApiClient
from core.models import ViolationInfo
from state.generation_state import (
    GenerationState,
    _decode_token,
    _tokens_to_text,
)
from validators.base_validator import BaseValidator
from validators.slop_phrase_validator import SlopPhraseValidator
from utils.sampler_helpers import select_tail_tokens
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

    so we need only ASCII space variants – special markers (Ġ▁) are handled
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
        try:
            # If tiktoken_model_name_for_counting is None, it will use "cl100k_base"
            # If it's a model name, it will try that.
            if tiktoken_model_name_for_counting:
                 self.tiktoken_encoding = tiktoken.encoding_for_model(tiktoken_model_name_for_counting)
            else:
                 self.tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            logger.info(f"ApiAntiSlopSampler using tiktoken encoding for '{_tiktoken_encoding_name}' for yielded chunk token counts.")
        except KeyError:
            logger.warning(
                f"Tiktoken model '{tiktoken_model_name_for_counting}' not found. "
                f"Falling back to 'cl100k_base' for yielded chunk token counts."
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
        self.chat_formatter = chat_template_formatter
        self.force_backtrack = bool(config.get("force_backtrack", False))

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
                logger.warning(
                    "tail-prefix filter requested but no chat-template tokenizer available – skipping filter."
                )
                self.filter_tail_banned_prefix_tokens = False
            else:
                cache_key = chat_template_formatter.model_id
                with _BANNED_PREFIX_LOCK:
                    cached = _BANNED_PREFIX_CACHE.get(cache_key)
                    if cached is None:                               # first time ever
                        start = time.time()
                        print('start prefix builder')
                        cached = _build_banned_prefix_set(chat_template_formatter, self.validators)
                        _BANNED_PREFIX_CACHE[cache_key] = cached
                        print('built in', time.time() - start)
                self._banned_prefix_token_strings = cached





        logger.info(
            f"ApiAntiSlopSampler ready — chunk={self.chunk_size}, "
            f"max_new_tokens={self.max_new_tokens}, T={self.temperature}, "
            f"top_p={self.top_p}, top_k={self.top_k}, min_p={self.min_p}, "
            f"tail_keep_tokens={self.tail_keep_tokens}"
        )

        self.request_mode = config.get("generation_params", {}).get("request_mode", "chunk")
        self.regex_interval = int(config.get("regex_validation_interval", 20))

        if self.request_mode == "stream":
            logger.info("Probing backend for streamed logprobs…")
            ok = self._probe_stream_logprobs()
            if not ok:
                raise RuntimeError(
                    "request_mode=stream but backend did not return logprobs in SSE "
                    "events.  Use --request-mode=chunk instead."
                )

    def _probe_stream_logprobs(self) -> bool:
        try:
            probe = next(self.api_client.generate_stream(
                prompt_text="Hello",
                max_tokens=1,
                top_logprobs=self.top_logprobs_count,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                timeout=15,
                stop_sequences=None,
            ))
            return bool(probe.logprobs)
        except Exception as e:
            logger.error("Probe failed: %s", e)
            return False


    def _run_validators(self, state: GenerationState) -> Optional[ViolationInfo]:
        """
        Query every validator and return the violation that starts at the
        **lowest token index**.  If two violations start at the same index
        we keep the one from the validator that appears first in
        self.validators so existing priority ties stay stable.
        """
        earliest: Optional[ViolationInfo] = None
        for v in self.validators:
            vio = v.check(state)
            if not vio:
                continue
            if (earliest is None) or (vio.violation_index < earliest.violation_index):
                earliest = vio
        return earliest

    # ------------------------------------------------------------------ #
    #  Fast sanity-check for a candidate                                 #
    # ------------------------------------------------------------------ #
    def _check_hypothetical_state(
        self,
        state: GenerationState,
        truncate_idx: int,
        alt_token: str,
    ) -> tuple[bool, tuple[str, str] | None]:
        """
        (passes_all, (validator_name, detail_key) | None)

        A violation blocks the candidate IFF the validator’s affected-token
        span includes the *candidate index*.

        For single-token validators (slop phrase, regex char map, …) that means
        `violation_index == truncate_idx`.

        For N-gram validator we treat the span as
            [violation_index, violation_index + len(ngram_tuple)-1].
        """
        if not self.validators:
            return True, None

        # ⸺ build hypothetical state ------------------------
        hypo_state = GenerationState(state.prompt_string)
        hypo_state.generated_token_strings = (
            state.generated_token_strings[:truncate_idx] + [alt_token]
        )

        for v in self.validators:
            vio = v.check(hypo_state)
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
    def _perform_backtrack(self, state: GenerationState, vio: ViolationInfo) -> bool:
        idx         = vio.violation_index
        banned_token   = vio.original_token_string
        lp_list     = state.get_logprobs(idx)
        
        def _abort() -> bool:
            self._suppress_violation(vio)   # ← remember: vio is already in scope
            return False                       # [(token, logp)]

        if not lp_list:
            logger.error("Back-track failed — no logprobs available.")
            return _abort()

        tried_here   = self._tried_alternatives.setdefault(idx, set())
        invert_probs = bool(getattr(self, "gen_config", {}).get("invert_probs", True))

        logger.warning(
            f"Back-tracking @tok={idx} orig='{banned_token}' ban='{vio.details}'."
        )

        # ── utility: validate a single token once per call ───────────────
        validation_cache = {}              # tok_id -> bool
        def _is_valid(tok_id: str) -> bool:
            if tok_id in validation_cache:
                return validation_cache[tok_id]
            ok, _ = self._check_hypothetical_state(state, idx, tok_id)
            validation_cache[tok_id] = ok
            return ok

        # ── build base (token, prob) list after temperature softmax ──────
        logits         = [lp for _, lp in lp_list]
        raw_p          = [math.exp(l) for l in logits]
        tempered       = [p ** (1 / self.temperature) for p in raw_p]
        Z              = sum(tempered)
        base_pairs     = [(tok, pt / Z if Z > 0 else 1 / len(tempered))
                        for (tok, _), pt in zip(lp_list, tempered)
                        if tok != banned_token]                         # banned removed

        if not base_pairs:
            logger.error("Back-track: after removing banned token no candidates left.")
            return _abort()

        # ─────────────────────────────────────────────────────────────────
        #  1. NEXT-TOKEN SELECTION  (honours --force-backtrack)
        # ─────────────────────────────────────────────────────────────────

        def _build_pairs(temp, min_p, top_p, top_k):
            """Return candidate list [(tok, prob)] after all filters."""
            logits  = [lp for _, lp in lp_list]
            raw_p   = [math.exp(l) for l in logits]
            probs_t = [p ** (1.0 / max(temp, 1e-6)) for p in raw_p]
            Z       = sum(probs_t)
            pairs   = [(tok, pt / Z if Z else 1 / len(probs_t))
                    for (tok, _), pt in zip(lp_list, probs_t)
                    if tok != banned_token]

            # min-p filter
            if min_p is not None and pairs:
                floor = min_p * max(pt for _, pt in pairs)
                pairs = [(tok, pt) for tok, pt in pairs if pt >= floor]

            # top-p nucleus
            if top_p is not None and pairs:
                pairs.sort(key=lambda tp: tp[1], reverse=True)
                nucleus, cum = [], 0.0
                for tok, pt in pairs:
                    nucleus.append((tok, pt))
                    cum += pt
                    if cum >= top_p:
                        break
                pairs = nucleus

            # top-k
            if top_k is not None and len(pairs) > top_k:
                pairs = sorted(pairs, key=lambda tp: tp[1], reverse=True)[:top_k]

            return pairs

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

            # Invert only while min_p still active (relax_idx 0 or 1)
            if invert_probs and relax_idx < 2:
                p_vals = [pt for _, pt in pairs]
                p_max, p_min = max(p_vals), min(p_vals)
                pairs = [(tok, (p_max - pt) + p_min) for tok, pt in pairs]
                Z_inv = sum(pt for _, pt in pairs)
                if Z_inv == 0.0:
                    # fall back to a uniform distribution to avoid div by 0
                    uniform = 1.0 / len(pairs) if pairs else 0.0
                    pairs = [(tok, uniform) for tok, _ in pairs]
                else:
                    pairs = [(tok, pt / Z_inv) for tok, pt in pairs]

            valid_pairs = [(tok, pt) for tok, pt in pairs
               if pt > 0 and tok not in tried_here and _is_valid(tok)]
            if valid_pairs:
                break

        if not valid_pairs:
            #logger.error("Back-track: no valid next-token candidates found.")
            return _abort()

        # sample replacement
        tokens, probs = zip(*valid_pairs)
        choice = random.choices(tokens, weights=probs, k=1)[0]


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

        tail_ids: list[str] = []
        if max_tail > 0:
            # re-apply tail-specific filters on **base_pairs** (no inversion!)
            tail_pairs = base_pairs.copy()

            if tail_min_p is not None:
                floor = tail_min_p * max(pt for _, pt in tail_pairs)
                tail_pairs = [(tok, pt) for tok, pt in tail_pairs if pt >= floor]

            if tail_top_k is not None and len(tail_pairs) > tail_top_k:
                tail_pairs = sorted(tail_pairs, key=lambda tp: tp[1], reverse=True)[: tail_top_k]

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
                if tok == banned_token or tok in tried_here:
                    continue
                if not _is_valid(tok):
                    continue
                

                
                #print('adding', tok)
                tail_ids.append(tok)
                if len(tail_ids) >= max_tail:
                    break

        multi_chosen_decoded = [_decode_token(t) for t in tail_ids]

        # ── commit replacement ───────────────────────────────────────────
        tried_here.add(choice)
        state.truncate(idx + 1)
        state.replace_token_string(idx, choice)

        # purge caches beyond idx
        for k in list(self._tried_alternatives):
            if k > idx:
                del self._tried_alternatives[k]

        logger.warning(
            f"    ✓ replacement='{choice}' "
            f"(T={self.temperature}, min_p={self.min_p}, "
            f"top_p={self.top_p}, top_k={self.top_k}, invert={invert_probs})"
        )

        # ── ftpo sample capture (legacy + new multi-fields) ──────────────
        if tail_ids:
            try:
                gen_so_far_tokens = state.generated_token_strings[:idx]
                gen_so_far_text   = _tokens_to_text(gen_so_far_tokens)

                context_chat = (
                    self.chat_formatter.build_prompt(state.prompt_string, gen_so_far_text)
                    if self.chat_formatter is not None
                    else state.prompt_string + gen_so_far_text
                )

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
                    "multi_chosen_raw":      tail_ids,
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

    def _run_validators_skip_regex(self, state):
        earliest = None
        for v in self.validators:
            if v.__class__.__name__ == "RegexValidator":
                continue
            vio = v.check(state)
            if vio and (earliest is None or vio.violation_index < earliest.violation_index):
                earliest = vio
        return earliest


    def generate(self, prompt: str):
        if self.request_mode == "stream":
            yield from self._generate_streamwise(prompt)
        else:
            yield from self._generate_chunkwise(prompt)   # existing logic renamed

    # ------------------------------------------------------------------ #
    #  Streaming generation with proper restart on back-track            #
    # ------------------------------------------------------------------ #
    def _generate_streamwise(self, prompt: str):
        state = GenerationState(prompt)
        last_yield  = 0
        regex_since = 0

        while state.get_generated_length() < self.max_new_tokens:

            restart_stream = False
            natural_end    = False

            remaining = self.max_new_tokens - state.get_generated_length()
            api_prompt = (
                state.get_full_text() if self.chat_formatter is None
                else self.chat_formatter.build_prompt(
                        state.prompt_string,
                        state.get_generated_text())
            )

            stream = self.api_client.generate_stream(
                prompt_text     = api_prompt,
                max_tokens      = remaining,
                top_logprobs    = self.top_logprobs_count,
                temperature     = self.temperature,
                top_p           = self.top_p,
                top_k           = self.top_k,
                min_p           = self.min_p,
                timeout         = self.timeout,
                stop_sequences  = self.stop_sequences,
            )

            for chunk in stream:
                if not chunk.token_strings:
                    if chunk.finish_reason and chunk.finish_reason != "length":
                        natural_end = True
                        break
                    continue

                pre_len = state.get_generated_length()
                state.append_chunk(chunk)
                delta = state.get_generated_length() - pre_len
                regex_since += delta

                if delta and self.on_chunk_yielded_callback:
                    try:
                        # empty text → caller knows this count is “accepted, not yet flushed”
                        self.on_chunk_yielded_callback("", delta)
                    except Exception as e_cb:
                        logger.error("Throughput callback failed: %s", e_cb, exc_info=True)

                run_regex = regex_since >= self.regex_interval
                vio = (
                    self._run_validators(state)
                    if run_regex
                    else self._run_validators_skip_regex(state)
                )

                # ---------- violation handling ------------------------------
                if vio:
                    if vio.validator_type != "regex":
                        vio_full = self._run_validators(state)  # make sure regex ran
                        vio = vio_full or vio

                    self.api_client.cancel_current_stream()

                    fixed = self._perform_backtrack(state, vio)

                    # ---------- NEW: event bookkeeping & callback ----------
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
                    # --------------------------------------------------------

                    if not fixed:
                        self._suppress_violation(vio)

                    restart_stream = True
                    regex_since = 0
                    break


                # ---------- yield safe prefix -------------------------------
                safe_upto = max(0, state.get_generated_length() - self.tail_keep_tokens)
                if safe_upto > last_yield:
                    text_out = _tokens_to_text(
                        state.generated_token_strings[last_yield : safe_upto])
                    if text_out:
                        yield self._yield_text_and_callback(text_out)
                    last_yield = safe_upto

                # ---------- natural termination inside stream --------------
                if chunk.finish_reason and chunk.finish_reason != "length":
                    natural_end = True
                    break

                if run_regex:
                    regex_since = 0

            # ~~~~~~~~~ end of inner for-loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if restart_stream:
                continue                    # repaired → open a new HTTP stream
            if natural_end:
                break                       # server said “done”
            # else: ended by length, outer while will loop if we still want more

        # ---------------- flush tail --------------------------------------
        if state.get_generated_length() > last_yield:
            residual = _tokens_to_text(
                state.generated_token_strings[last_yield:])
            if residual:
                yield self._yield_text_and_callback(residual)



    def _generate_chunkwise(self, prompt: str) -> Generator[str, None, None]:
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
        state = GenerationState(prompt)
        last_yield_idx = 0                         # first token not yet sent

        while True:
            # hard token limit
            if (
                self.max_new_tokens is not None
                and state.get_generated_length() >= self.max_new_tokens
            ):
                logger.info("Reached max_new_tokens.")
                break

            # build prompt for backend
            if self.chat_formatter is not None:
                full_prompt = self.chat_formatter.build_prompt(
                    state.prompt_string, state.get_generated_text()
                )
            else:
                full_prompt = state.get_full_text()

            # ── API call timing ─────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                chunk = self.api_client.generate_chunk(
                    prompt_text     = full_prompt,
                    max_tokens      = self.chunk_size,
                    top_logprobs    = self.top_logprobs_count,
                    temperature     = self.temperature,
                    top_p           = self.top_p,
                    top_k           = self.top_k,
                    min_p           = self.min_p,
                    timeout         = self.timeout,
                    stop_sequences  = self.stop_sequences,
                )
            # ── friendly handling of “context length” overflow ───────────
            except requests.exceptions.HTTPError as e:
                resp_json = {}
                if e.response is not None:
                    try:
                        resp_json = e.response.json()
                    except Exception:
                        pass

                if (
                    e.response is not None
                    and e.response.status_code == 400
                    and "maximum context length" in resp_json.get("message", "").lower()
                ):
                    logger.warning(
                        "Generation aborted for this prompt – the model’s "
                        "context window was exceeded: %s",
                        resp_json.get("message", "").rstrip()
                    )
                else:
                    logger.error("API call failed: %s", e, exc_info=True)
                break   # keep the existing control-flow (skip further chunks)

            except Exception as e:
                logger.error("API call failed: %s", e, exc_info=True)
                break
            api_sec = time.perf_counter() - t0

            if not chunk.token_strings:            # end-of-stream
                break

            # ── append & validate timing ────────────────────────────────────
            pre_len = state.get_generated_length()
            state.append_chunk(chunk)

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
            safe_upto = max(0, post_len - self.tail_keep_tokens)
            if safe_upto > last_yield_idx:
                safe_tokens = state.generated_token_strings[last_yield_idx : safe_upto]
                text_to_yield = _tokens_to_text(safe_tokens)
                if text_to_yield:
                    yield text_to_yield
                last_yield_idx = safe_upto

            # stop if backend signalled completion
            if chunk.finish_reason and chunk.finish_reason != "length":
                logger.info(f"finish_reason={chunk.finish_reason}")
                break

        # ── flush residual tail ────────────────────────────────────────────
        if state.get_generated_length() > last_yield_idx:
            residual = state.generated_token_strings[last_yield_idx :]
            text_to_yield = _tokens_to_text(residual)
            if text_to_yield:
                yield text_to_yield

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
