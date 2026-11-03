# validators/slop_phrase_validator.py
import logging
from typing import Dict, Optional, Set, Any # Added Any

from .base_validator import BaseValidator
from state.generation_state import GenerationState
from core.models import ViolationInfo
from utils.slop_helpers import detect_disallowed_sequence, fold_for_matching

logger = logging.getLogger(__name__)


class SlopPhraseValidator(BaseValidator):
    """
    Hard-ban validator for an explicit phrase list.
    """
    validator_type = "slop_phrase"

    def __init__(self, slop_phrases_dict: Dict[str, float], app_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        self.slop_phrases_keys: Set[str] = {fold_for_matching(k) for k in slop_phrases_dict.keys()}
        self.max_phrase_len = max((len(p) for p in self.slop_phrases_keys), default=0)
        self.min_phrase_len = min((len(p) for p in self.slop_phrases_keys), default=0)

        # Configurable boundary behavior (default True)
        self.match_requires_word_boundaries: bool = True
        if app_config is not None:
            try:
                # config-example.yaml documents this key under generation_params;
                # it used to be read only at the top level, so setting it there
                # had no effect.
                value = app_config.get("match_requires_word_boundaries")
                if value is None:
                    value = (app_config.get("generation_params") or {}).get(
                        "match_requires_word_boundaries", True)
                self.match_requires_word_boundaries = bool(value)
            except Exception:
                self.match_requires_word_boundaries = True
        
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

        logger.info(
            f"SlopPhraseValidator ready "
            f"(count={len(self.slop_phrases_keys)}, "
            f"min_len={self.min_phrase_len}, max_len={self.max_phrase_len}, "
            f"scan_window_base_size={self.scan_window_base_size}, "
            f"require_word_boundaries={self.match_requires_word_boundaries})"
        )


    def check(
        self, state: GenerationState, *, final: bool = False
    ) -> Optional[ViolationInfo]:
        """Return the earliest banned phrase, or None.

        *final* is False while the text is still growing.  In that case a match
        whose right edge is the end of the text is **provisional** and is not
        reported: the next token may extend the word (``"la"`` → ``"land"``), so
        banning it immediately caused the back-tracker to rewrite a token the
        model was about to disambiguate itself, looping at the same index.
        The final sweep after generation stops passes ``final=True``.
        """
        if not self.slop_phrases_keys or state.get_generated_length() == 0 or self.min_phrase_len == 0:
            return None

        generated_text = state.get_generated_text()

        # Dynamic scan window size based on max_phrase_len and configured base size
        scan_window_size = self.max_phrase_len + self.scan_window_base_size
        
        start_scan_char_offset = max(0, len(generated_text) - scan_window_size)
        # Start the window on a word boundary.  A window that began mid-word
        # treated its first character as a boundary, so the single-word entries
        # fired inside longer words as the window slid ("grim" in "pilgrim",
        # "aria" in "Maria") and each false hit discarded ~50 good tokens.
        while start_scan_char_offset > 0 and (
            generated_text[start_scan_char_offset - 1].isalnum()
            or generated_text[start_scan_char_offset - 1] == "_"
        ):
            start_scan_char_offset -= 1
        text_to_scan = generated_text[start_scan_char_offset:]

        if not text_to_scan.strip(): # Avoid processing if the window is just whitespace
            return None

        phrase, rel_pos_in_scanned_text = detect_disallowed_sequence(
            text=text_to_scan,
            slop_phrases_keys=self.slop_phrases_keys,
            max_phrase_len=self.max_phrase_len,
            min_phrase_len=self.min_phrase_len,
            check_n_chars_back=len(text_to_scan),
            require_word_boundaries=self.match_requires_word_boundaries,
            # A match touching the end of the text is only definitive once
            # generation has stopped.  While it is still running the next token
            # may extend the word and dissolve the boundary.
            allow_end_of_text_match=final,
            skip=lambda cand, pos: self._is_ignored(
                self._char_to_token_index(state, start_scan_char_offset + pos), cand),
        )

        if not phrase:
            return None

        absolute_char_pos = start_scan_char_offset + rel_pos_in_scanned_text
        
        tok_idx = self._char_to_token_index(state, absolute_char_pos)
        if tok_idx is None:
            logger.warning(f"SlopPhraseValidator: Could not map char_pos {absolute_char_pos} to token_idx for phrase '{phrase}'.")
            return None

        if self._is_ignored(tok_idx, phrase):
            return None

        snippet_start = max(0, absolute_char_pos - 20)
        snippet_end = min(len(generated_text), absolute_char_pos + len(phrase) + 20)
        snippet = generated_text[snippet_start:snippet_end].replace("\n", " ")

        logger.warning(
            f"Slop violation: '{phrase}' @tok={tok_idx} (char_pos={absolute_char_pos}) …{snippet}…"
        )

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