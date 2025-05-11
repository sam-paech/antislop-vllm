# validators/slop_phrase_validator.py
import logging
from typing import Dict, Optional, Set, Any # Added Any

from .base_validator import BaseValidator
from state.generation_state import GenerationState
from core.models import ViolationInfo
from utils.slop_helpers import detect_disallowed_sequence
import ahocorasick

logger = logging.getLogger(__name__)


class SlopPhraseValidator(BaseValidator):
    """
    Hard-ban validator for an explicit phrase list.
    """

    def __init__(self, slop_phrases_dict: Dict[str, float], app_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        self.slop_phrases_keys: Set[str] = set(slop_phrases_dict.keys())
        self.max_phrase_len = max((len(p) for p in self.slop_phrases_keys), default=0)
        self.min_phrase_len = min((len(p) for p in self.slop_phrases_keys), default=0)
        
        # Determine scan_window_size_config based on app_config if provided
        # This is used to estimate a good window size for scanning.
        # A sensible default if config is not available or chunk_size is missing.
        self.scan_window_base_size = 250 # Default characters to look back beyond max_phrase_len
        if app_config:
            generation_params = app_config.get("generation_params", {})
            chunk_size_tokens = generation_params.get("chunk_size", 50)
            # Estimate characters per chunk (e.g., average 4-5 chars per token)
            estimated_chars_per_chunk = chunk_size_tokens * 5 
            self.scan_window_base_size = max(self.scan_window_base_size, estimated_chars_per_chunk)

        # ── fast matcher (falls back to sliding window on any error) ──────────
        self._aho = None
        try:
            A = ahocorasick.Automaton()
            for p in self.slop_phrases_keys:
                A.add_word(p, p)
            A.make_automaton()
            self._aho = A
            logger.info(f"Aho-Corasick automaton built ({len(self.slop_phrases_keys)} phrases).")
        except Exception as e:
            logger.warning(f"Automaton build failed – reverting to legacy scanner: {e}")

        logger.info(
            "SlopPhraseValidator ready "
            f"(count={len(self.slop_phrases_keys)}, "
            f"min_len={self.min_phrase_len}, max_len={self.max_phrase_len}, "
            f"scan_window_base_size={self.scan_window_base_size})"
        )

    def check(self, state: GenerationState) -> Optional[ViolationInfo]:
        # early exits
        if not self.slop_phrases_keys or state.get_generated_length() == 0 or self.min_phrase_len == 0:
            return None

        generated_text = state.get_generated_text()

        # Window = newest chunk + a safety margin
        scan_window_size = self.max_phrase_len + self.scan_window_base_size
        start_scan_char_offset = max(0, len(generated_text) - scan_window_size)
        text_to_scan = generated_text[start_scan_char_offset:]
        if not text_to_scan.strip():
            return None

        # ------------------------------------------------------------------
        #  1) try Aho-Corasick (O(window))                                  #
        # ------------------------------------------------------------------
        phrase            = None
        rel_pos_in_window = None

        if self._aho is not None:
            for end_idx, found in self._aho.iter(text_to_scan.lower()):
                phrase            = found
                rel_pos_in_window = end_idx - len(found) + 1
                break   # earliest hit wins

        # ------------------------------------------------------------------
        #  2) fallback to legacy sliding-window search                       #
        # ------------------------------------------------------------------
        if phrase is None:
            phrase, rel_pos_in_window = detect_disallowed_sequence(
                text=text_to_scan,
                slop_phrases_keys=self.slop_phrases_keys,
                max_phrase_len=self.max_phrase_len,
                min_phrase_len=self.min_phrase_len,
                check_n_chars_back=len(text_to_scan)
            )

        if not phrase:
            return None

        # Map char-offset → token index
        absolute_char_pos = start_scan_char_offset + rel_pos_in_window
        tok_idx = self._char_to_token_index(state, absolute_char_pos)
        if tok_idx is None:
            logger.warning(f"SlopPhraseValidator: char_pos {absolute_char_pos} → token_idx failed for '{phrase}'.")
            return None
        if self._is_ignored(tok_idx, phrase):
            return None

        snippet_start = max(0, absolute_char_pos - 20)
        snippet_end   = min(len(generated_text), absolute_char_pos + len(phrase) + 20)
        snippet       = generated_text[snippet_start:snippet_end].replace("\n", " ")

        logger.warning(f"Slop violation: '{phrase}' @tok={tok_idx} …{snippet}…")

        return ViolationInfo(
            validator_type="slop_phrase",
            violation_index=tok_idx,
            original_token_string=state.get_token_string_at(tok_idx) or "[UNK]",
            details={
                "phrase": phrase,
                "context": snippet,
                "suppression_detail_key": phrase
            },
        )