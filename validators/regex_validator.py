# validators/regex_validator.py
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import List, Optional, Dict, Tuple, Any

from .base_validator import BaseValidator
from state.generation_state import GenerationState
from core.models import ViolationInfo

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  Global config ― tweak here when necessary
# ────────────────────────────────────────────────────────────────────────────
REGEX_TIMEOUT_SEC: float = 2          # seconds before we abort a search
REGEX_MAX_WORKERS: int   = 16             # keep this many “hot” workers alive
# Create the process pool once – stays alive for the interpreter lifetime
_REGEX_POOL = ProcessPoolExecutor(max_workers=REGEX_MAX_WORKERS)


# --------------------------------------------------------------------------- #
#  Helper run in a separate process                                           #
# --------------------------------------------------------------------------- #
def _regex_search_worker(args: Tuple[str, int, str]) -> Optional[Tuple[str, int, str]]:
    """
    Compile the giant alternation and search the given text.
    Returns (group_name, start_char_idx, matched_segment) or None.
    """
    pattern_str, flags, text = args
    compiled = re.compile(pattern_str, flags)
    m = compiled.search(text)
    if not m:
        return None
    return (m.lastgroup, m.start(), m.group(0))


# --------------------------------------------------------------------------- #
#  Validator                                                                  #
# --------------------------------------------------------------------------- #
class RegexValidator(BaseValidator):
    """
    Hard-ban validator driven by an arbitrary list of regular-expression
    patterns.  All patterns are merged into one master regex so the generated
    text is traversed only once per `check()` call.

    Each original pattern is wrapped in its own *named* capturing group:
        (?P<P0>pattern0)|(?P<P1>pattern1)|...
    Mapping   group-name → raw-pattern   is kept so we still know which rule
    was violated.
    """

    _GROUP_PREFIX = "P"   # group names: P0, P1, …

    def __init__(self, pattern_strings: List[str]) -> None:
        super().__init__()

        if not pattern_strings:
            logger.info("RegexValidator initialised with 0 patterns.")
            self._big_re: Optional[re.Pattern] = None
            self._group2raw: Dict[str, str] = {}
            return

        flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
        parts: List[str] = []
        g2raw: Dict[str, str] = {}

        for i, p_str in enumerate(pattern_strings):
            try:
                # compile individually once to verify the syntax
                re.compile(p_str, flags)
            except re.error as e:
                logger.error("Invalid regex pattern '%s': %s – skipping.", p_str, e)
                continue

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
        #logger.info("RegexValidator ready (loaded %d patterns).", len(g2raw))

    # ---------------------------------------------------------------------- #
    #  Fast check (wrapped in process-pool search with timeout)              #
    # ---------------------------------------------------------------------- #
    def check(self, state: GenerationState) -> Optional[ViolationInfo]:
        if self._big_re is None or state.get_generated_length() == 0:
            return None

        text = state.get_generated_text()

        # ── timing probe ---------------------------------------------------
        t0 = time.perf_counter()

        future = _REGEX_POOL.submit(
            _regex_search_worker,
            (self._big_re.pattern, self._big_re.flags, text)
        )
        try:
            result = future.result(timeout=REGEX_TIMEOUT_SEC)
        except TimeoutError:
            future.cancel()
            print(f"[RegexValidator] search timed out after {REGEX_TIMEOUT_SEC:.2f}s")
            return None

        elapsed = time.perf_counter() - t0
        #print(f"[RegexValidator] search {elapsed:.4f}s (len={len(text)})")

        if result is None:
            return None

        gname, violation_char_pos, matched_text_segment = result
        violated_raw_pattern = self._group2raw.get(gname, "?")

        tok_idx = self._char_to_token_index(state, violation_char_pos)
        if tok_idx is None:
            logger.warning(
                "RegexValidator: Could not map char_pos %d to token index for "
                "pattern '%s'.",
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
