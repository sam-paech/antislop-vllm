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

logger = logging.getLogger(__name__)


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
        self.tdpo_samples: Dict[str, Dict[str, Any]] = {}


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
        self.max_retries_per_position = back.get("max_retries_per_position", 20)

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
        _do_profile = getattr(self, "profile_backtrack_timing", True)
        _perf_timings = {}
        _perf_start_total = 0.0
        _current_section_start_time = 0.0

        if _do_profile:
            _perf_timings = {
                "total": 0.0, "init_setup": 0.0,
                "is_valid_total": 0.0, "is_valid_calls": 0,
                "build_pairs_total": 0.0, "build_pairs_calls": 0,
                "next_token_selection": 0.0, "tail_selection": 0.0,
                "commit_and_cache": 0.0, "tdpo_capture": 0.0,
            }
            _perf_start_total = time.perf_counter()
            _current_section_start_time = _perf_start_total

        try:
            # ─── Initial setup and checks ────────────────────────────────────
            idx         = vio.violation_index
            banned_id   = vio.original_token_string
            lp_list     = state.get_logprobs(idx)

            def _abort_profiled() -> bool:
                self._suppress_violation(vio)
                return False

            if not lp_list:
                logger.error("Back-track failed — no logprobs available.")
                if _do_profile: _perf_timings["init_setup"] += time.perf_counter() - _current_section_start_time
                return _abort_profiled()

            tried_here   = self._tried_alternatives.setdefault(idx, set())
            invert_probs = bool(getattr(self, "gen_config", {}).get("invert_probs", True))

            logger.warning(
                f"Back-tracking @tok={idx} orig='{banned_id}' ban='{vio.details}'."
            )
            if _do_profile:
                _perf_timings["init_setup"] += time.perf_counter() - _current_section_start_time
                _current_section_start_time = time.perf_counter() # Reset for base_pairs calc + _is_valid setup

            # ── utility: validate a single token once per call ───────────────
            validation_cache = {}
            def _is_valid(tok_id: str) -> bool:
                _is_valid_start_local = 0.0
                if _do_profile: _is_valid_start_local = time.perf_counter()

                if tok_id in validation_cache:
                    if _do_profile:
                        _perf_timings["is_valid_total"] += time.perf_counter() - _is_valid_start_local
                        _perf_timings["is_valid_calls"] += 1
                    return validation_cache[tok_id]
                
                ok, _ = self._check_hypothetical_state(state, idx, tok_id)
                validation_cache[tok_id] = ok

                if _do_profile:
                    _perf_timings["is_valid_total"] += time.perf_counter() - _is_valid_start_local
                    _perf_timings["is_valid_calls"] += 1
                return ok

            # ── build base (token, prob) list after temperature softmax ──────
            logits         = [lp for _, lp in lp_list]
            raw_p          = [math.exp(l) for l in logits]
            tempered       = [p ** (1 / self.temperature) for p in raw_p]
            Z              = sum(tempered)
            base_pairs     = [(tok, pt / Z if Z > 0 else 1 / len(tempered))
                            for (tok, _), pt in zip(lp_list, tempered)
                            if tok != banned_id]

            if not base_pairs:
                logger.error("Back-track: after removing banned token no candidates left.")
                if _do_profile: _perf_timings["init_setup"] += time.perf_counter() - _current_section_start_time
                return _abort_profiled()
            
            if _do_profile:
                _perf_timings["init_setup"] += time.perf_counter() - _current_section_start_time
                _current_section_start_time = time.perf_counter() # Reset for NEXT-TOKEN SELECTION

            # ─────────────────────────────────────────────────────────────────
            #  1. NEXT-TOKEN SELECTION  (honours --force-backtrack)
            # ─────────────────────────────────────────────────────────────────
            def _build_pairs(temp, min_p, top_p, top_k):
                _build_pairs_start_local = 0.0
                if _do_profile: _build_pairs_start_local = time.perf_counter()

                logits_bp  = [lp for _, lp in lp_list]
                raw_p_bp   = [math.exp(l) for l in logits_bp]
                probs_t = [p ** (1.0 / max(temp, 1e-6)) for p in raw_p_bp]
                Z_bp       = sum(probs_t)
                pairs   = [(tok, pt / Z_bp if Z_bp else 1 / len(probs_t))
                        for (tok, _), pt in zip(lp_list, probs_t)
                        if tok != banned_id]

                if min_p is not None and pairs:
                    floor = min_p * max(pt for _, pt in pairs)
                    pairs = [(tok, pt) for tok, pt in pairs if pt >= floor]

                if top_p is not None and pairs:
                    pairs.sort(key=lambda tp: tp[1], reverse=True)
                    nucleus, cum = [], 0.0
                    for tok, pt in pairs:
                        nucleus.append((tok, pt))
                        cum += pt
                        if cum >= top_p:
                            break
                    pairs = nucleus

                if top_k is not None and len(pairs) > top_k:
                    pairs = sorted(pairs, key=lambda tp: tp[1], reverse=True)[:top_k]

                if _do_profile:
                    _perf_timings["build_pairs_total"] += time.perf_counter() - _build_pairs_start_local
                    _perf_timings["build_pairs_calls"] += 1
                return pairs

            attempts = [(self.temperature, self.min_p, self.top_p, self.top_k)]
            if getattr(self, "force_backtrack", False):
                attempts += [
                    (1.0, self.min_p, self.top_p, self.top_k),
                    (1.0, None, self.top_p, self.top_k),
                    (1.0, None, None, self.top_k),
                    (1.0, None, None, None),
                ]

            valid_pairs = []
            for relax_idx, (temp, min_p, top_p, top_k) in enumerate(attempts):
                pairs = _build_pairs(temp, min_p, top_p, top_k)
                if not pairs: continue

                if invert_probs and relax_idx < 2:
                    p_vals = [pt for _, pt in pairs]
                    p_max, p_min = max(p_vals), min(p_vals)
                    pairs = [(tok, (p_max - pt) + p_min) for tok, pt in pairs]
                    Z_inv = sum(pt for _, pt in pairs)
                    pairs = [(tok, pt / Z_inv) for tok, pt in pairs]

                valid_pairs = [(tok, pt) for tok, pt in pairs
                            if tok not in tried_here and _is_valid(tok)]
                if valid_pairs: break

            if not valid_pairs:
                logger.error("Back-track: no valid next-token candidates found.")
                if _do_profile: _perf_timings["next_token_selection"] += time.perf_counter() - _current_section_start_time
                return _abort_profiled()

            tokens, probs = zip(*valid_pairs)
            choice = random.choices(tokens, weights=probs, k=1)[0]
            
            if _do_profile:
                _perf_timings["next_token_selection"] += time.perf_counter() - _current_section_start_time
                _current_section_start_time = time.perf_counter()

            # ─────────────────────────────────────────────────────────────────
            # 2. TAIL-CANDIDATE SELECTION  (independent knobs)
            # ─────────────────────────────────────────────────────────────────
            max_tail   = getattr(self, "max_chosen_tokens", 8)
            tail_min_p = getattr(self, "tail_min_p", 0.0)
            tail_top_k = getattr(self, "tail_top_k", 50)

            tail_ids: list[str] = []
            if max_tail > 0:
                tail_pairs = base_pairs.copy()
                if tail_min_p is not None and tail_pairs:
                    floor = tail_min_p * max(pt for _, pt in tail_pairs)
                    tail_pairs = [(tok, pt) for tok, pt in tail_pairs if pt >= floor]

                if tail_top_k is not None and len(tail_pairs) > tail_top_k:
                    tail_pairs = sorted(tail_pairs, key=lambda tp: tp[1], reverse=True)[: tail_top_k]

                tail_pairs.sort(key=lambda tp: tp[1])
                for tok, _ in tail_pairs:
                    if tok == banned_id or tok in tried_here: continue
                    if not _is_valid(tok): continue
                    tail_ids.append(tok)
                    if len(tail_ids) >= max_tail: break
                if not tail_ids: tail_ids = [choice]
            
            multi_chosen_decoded = [_decode_token(t) for t in tail_ids]
            
            if _do_profile:
                _perf_timings["tail_selection"] += time.perf_counter() - _current_section_start_time
                _current_section_start_time = time.perf_counter()

            # ── commit replacement ───────────────────────────────────────────
            tried_here.add(choice)
            state.truncate(idx + 1)
            state.replace_token_string(idx, choice)

            for k in list(self._tried_alternatives):
                if k > idx: del self._tried_alternatives[k]
            
            if _do_profile:
                _perf_timings["commit_and_cache"] += time.perf_counter() - _current_section_start_time
                _current_section_start_time = time.perf_counter()

            logger.warning(
                f"    ✓ replacement='{choice}' "
                f"(T={self.temperature}, min_p={self.min_p}, "
                f"top_p={self.top_p}, top_k={self.top_k}, invert={invert_probs})"
            )

            # ── TDPO sample capture ──────────────────────────────────────────
            try:
                gen_so_far_tokens = state.generated_token_strings[:idx]
                gen_so_far_text   = _tokens_to_text(gen_so_far_tokens)
                context_chat = (
                    self.chat_formatter.build_prompt(state.prompt_string, gen_so_far_text)
                    if self.chat_formatter is not None
                    else state.prompt_string + gen_so_far_text
                )
                self.tdpo_samples[context_chat] = {
                    "prompt_raw": state.prompt_string, "generation_raw": gen_so_far_text,
                    "context_with_chat_template": context_chat,
                    "chosen_decoded": _decode_token(choice), "rejected_decoded": _decode_token(banned_id),
                    "chosen_raw": choice, "rejected_raw": banned_id,
                    "multi_chosen_decoded": multi_chosen_decoded, "multi_chosen_raw": tail_ids,
                    "multi_rejected_decoded": [_decode_token(banned_id)], "multi_rejected_raw": [banned_id],
                    "validator": {
                        "class": vio.validator_type,
                        "rule": (vio.details.get("phrase") or vio.details.get("ngram_string") or vio.details.get("pattern") or ""),
                        "subtype": (vio.validator_type if vio.validator_type != "ngram"
                                    else ("trigram" if len(vio.details.get("ngram_tuple", [])) == 3 else "bigram")),
                    }, "stats": {},
                }
            except Exception as e_log:
                logger.error(f"TDPO-pair capture failed: {e_log}", exc_info=True)
            
            if _do_profile:
                _perf_timings["tdpo_capture"] += time.perf_counter() - _current_section_start_time
            
            return True

        finally:
            if _do_profile:
                _perf_timings["total"] = time.perf_counter() - _perf_start_total
                
                parts = [f"Total: {_perf_timings['total']:.4f}s"]
                if _perf_timings['init_setup'] > 1e-6: parts.append(f"Setup: {_perf_timings['init_setup']:.4f}s")
                if _perf_timings['build_pairs_calls'] > 0:
                    avg = _perf_timings['build_pairs_total'] / _perf_timings['build_pairs_calls']
                    parts.append(f"BuildPairs(x{_perf_timings['build_pairs_calls']}): {_perf_timings['build_pairs_total']:.4f}s (avg {avg:.4f}s)")
                if _perf_timings['is_valid_calls'] > 0:
                    avg = _perf_timings['is_valid_total'] / _perf_timings['is_valid_calls']
                    parts.append(f"IsValid(x{_perf_timings['is_valid_calls']}): {_perf_timings['is_valid_total']:.4f}s (avg {avg:.4f}s)")
                if _perf_timings['next_token_selection'] > 1e-6: parts.append(f"NextTokenSel: {_perf_timings['next_token_selection']:.4f}s")
                if _perf_timings['tail_selection'] > 1e-6: parts.append(f"TailSel: {_perf_timings['tail_selection']:.4f}s")
                if _perf_timings['commit_and_cache'] > 1e-6: parts.append(f"Commit: {_perf_timings['commit_and_cache']:.4f}s")
                if _perf_timings['tdpo_capture'] > 1e-6: parts.append(f"TDPO: {_perf_timings['tdpo_capture']:.4f}s")
                
                print(f"Backtrack Perf: {' | '.join(parts)}")




    def _suppress_violation(self, vio: ViolationInfo) -> None:
        for v in self.validators:
            v_name = v.__class__.__name__.lower()
            if vio.validator_type in v_name and hasattr(v, "ignore_violation"):
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
            except Exception as e:
                logger.error(f"API call failed: {e}", exc_info=True)
                break
            api_sec = time.perf_counter() - t0

            if not chunk.token_strings:            # end-of-stream
                break

            # ── append & validate timing ────────────────────────────────────
            pre_len = state.get_generated_length()
            state.append_chunk(chunk)

            t1 = time.perf_counter()
            vio = self._run_validators(state)
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
