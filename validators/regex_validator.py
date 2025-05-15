# validators/regex_validator.py
import logging
import regex as re
from typing import List, Optional, Tuple, Dict

from .base_validator import BaseValidator
from state.generation_state import GenerationState
from core.models import ViolationInfo

logger = logging.getLogger(__name__)


class RegexValidator(BaseValidator):
    """
    Hard-ban validator driven by an arbitrary list of regular-expression
    patterns.  All patterns are merged into a single master regex so the
    generated text is traversed only once per `check()` call.

    Each original pattern is wrapped in its own *capturing* group:
        (?P<P0>pattern0)|(?P<P1>pattern1)|...
    and a mapping  group-name → raw-pattern  is kept so we still know
    which rule was violated.
    """

    _GROUP_PREFIX = "P"   # group names: P0, P1, ...

    def __init__(self, pattern_strings: List[str]) -> None:
        super().__init__()

        if not pattern_strings:
            logger.info("RegexValidator initialised with 0 patterns.")
            self._big_re: Optional[re.Pattern] = None
            self._group2raw: Dict[str, str] = {}
            return

        flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
        parts, g2raw = [], {}

        for i, p_str in enumerate(pattern_strings):
            try:
                # compile individually once to verify the syntax
                re.compile(p_str, flags)
            except re.error as e:
                logger.error("Invalid regex pattern '%s': %s – skipping.", p_str, e)
                continue

            # Wrap original pattern in a *named* capturing group so we can
            # pinpoint which alternative matched later.  If the pattern
            # itself contains top-level alternations, this still works.
            gname = f"{self._GROUP_PREFIX}{i}"
            parts.append(f"(?P<{gname}>{p_str})")
            g2raw[gname] = p_str

        if not parts:
            logger.error("No valid regex patterns left after sanitising input.")
            self._big_re = None
            self._group2raw = {}
            return

        alternation = "|".join(parts)
        self._big_re = re.compile(alternation, flags)
        self._group2raw = g2raw
        logger.info("RegexValidator ready (loaded %d patterns).", len(g2raw))

    # ------------------------------------------------------------------ #
    #  Fast check                                                        #
    # ------------------------------------------------------------------ #
    def check(self, state: GenerationState) -> Optional[ViolationInfo]:
        if self._big_re is None or state.get_generated_length() == 0:
            return None

        text = state.get_generated_text()
        match_obj = self._big_re.search(text)
        if match_obj is None:
            return None

        # Which sub-pattern fired?
        gname = match_obj.lastgroup                       # e.g. 'P17'
        violated_raw_pattern = self._group2raw.get(gname, "?")

        violation_char_pos = match_obj.start()
        matched_text_segment = match_obj.group(0)

        tok_idx = self._char_to_token_index(state, violation_char_pos)
        if tok_idx is None:
            logger.warning(
                "RegexValidator: Could not map char_pos %d to token index for pattern '%s'.",
                violation_char_pos,
                violated_raw_pattern,
            )
            return None

        if self._is_ignored(tok_idx, violated_raw_pattern):
            return None

        logger.warning(
            "Regex violation: /%s/ matched '%s' @tok=%d (char_pos=%d)",
            violated_raw_pattern,
            matched_text_segment[:100],
            tok_idx,
            violation_char_pos,
        )

        return ViolationInfo(
            validator_type="regex",
            violation_index=tok_idx,
            original_token_string=state.get_token_string_at(tok_idx) or "[UNK]",
            details={
                "pattern": violated_raw_pattern,
                "match": matched_text_segment,
                "suppression_detail_key": violated_raw_pattern,
            },
        )
