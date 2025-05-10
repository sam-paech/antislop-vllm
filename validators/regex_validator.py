# validators/regex_validator.py
import logging
import re
from typing import List, Optional, Tuple

from .base_validator import BaseValidator
from state.generation_state import GenerationState
from core.models import ViolationInfo

logger = logging.getLogger(__name__)


class RegexValidator(BaseValidator):
    """
    Hard-ban validator driven by arbitrary regular-expression patterns.
    """

    def __init__(self, pattern_strings: List[str]) -> None:
        super().__init__()
        flags = re.IGNORECASE | re.MULTILINE | re.DOTALL # Consider if DOTALL is always wanted
        self._compiled: List[Tuple[re.Pattern, str]] = []
        for p_str in pattern_strings:
            try:
                self._compiled.append((re.compile(p_str, flags), p_str))
            except re.error as e:
                logger.error(f"Invalid regex pattern '{p_str}': {e}. Skipping.")
        
        logger.info(f"RegexValidator ready (loaded {len(self._compiled)} patterns)")

    def check(self, state: GenerationState) -> Optional[ViolationInfo]:
        if not self._compiled or state.get_generated_length() == 0:
            return None

        text = state.get_generated_text() # Full decoded text

        earliest_violation: Optional[Tuple[int, re.Match, str]] = None  # (char_start_pos, match_object, raw_pattern_string)

        for compiled_pattern, raw_pattern_str in self._compiled:
            # Iterate over all matches for this pattern in the text
            for match_obj in compiled_pattern.finditer(text):
                match_start_pos = match_obj.start()

                if earliest_violation is None or match_start_pos < earliest_violation[0]:
                    earliest_violation = (match_start_pos, match_obj, raw_pattern_str)
                # If same start position, prefer the one from the pattern that resulted in a longer match string.
                # This is a bit arbitrary; another option is to prefer the pattern listed earlier in the config.
                # Or, simply the first one found at that position.
                # Current logic: if same start, keep the one that matched more text.
                elif match_start_pos == earliest_violation[0] and \
                     len(match_obj.group(0)) > len(earliest_violation[1].group(0)):
                    earliest_violation = (match_start_pos, match_obj, raw_pattern_str)

        if earliest_violation is None:
            return None

        violation_char_pos, match_object, violated_raw_pattern = earliest_violation
        matched_text_segment = match_object.group(0)
        
        tok_idx = self._char_to_token_index(state, violation_char_pos)
        if tok_idx is None:
            logger.warning(f"RegexValidator: Could not map char_pos {violation_char_pos} to token_idx for pattern '{violated_raw_pattern}'.")
            return None

        # The raw pattern string is the detail key for suppression.
        if self._is_ignored(tok_idx, violated_raw_pattern):
            return None

        logger.warning(
            f"Regex violation: /{violated_raw_pattern}/ matched '{matched_text_segment[:100]}...' " # Truncate long matches
            f"@tok={tok_idx} (char_pos={violation_char_pos})"
        )

        return ViolationInfo(
            validator_type="regex",
            violation_index=tok_idx,
            original_token_string=state.get_token_string_at(tok_idx) or "[UNK]",
            details={
                "pattern": violated_raw_pattern,
                "match": matched_text_segment,
                "suppression_detail_key": violated_raw_pattern # Used by BaseValidator._is_ignored
            },
        )