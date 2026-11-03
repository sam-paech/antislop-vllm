# validators/regex_validator.py
import logging
import os
import re
import threading
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


def _default_max_workers() -> int:
    """Workers to keep “hot”.

    Capped at the CPU count: 16 workers on a 4-core board just oversubscribes it, and
    each one is a forked interpreter that outlives a killed parent unless it is in the
    same process group.  ``ANTISLOP_REGEX_MAX_WORKERS`` overrides (the test harness
    pins it to 1).
    """
    raw = os.environ.get("ANTISLOP_REGEX_MAX_WORKERS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "Ignoring ANTISLOP_REGEX_MAX_WORKERS=%r: not an integer.", raw
            )
        else:
            if value >= 1:
                return value
            logger.warning(
                "Ignoring ANTISLOP_REGEX_MAX_WORKERS=%r: must be >= 1.", raw
            )
    return max(1, min(16, os.cpu_count() or 1))


REGEX_MAX_WORKERS: int = _default_max_workers()

# The pool is created on first use, not at import time.  Importing this module used to
# fork 16 processes as a side effect, which made `import validators` expensive and
# meant a test run spawned workers it never used.
_REGEX_POOL: Optional[ProcessPoolExecutor] = None
_REGEX_POOL_LOCK = threading.Lock()


def _get_regex_pool() -> ProcessPoolExecutor:
    global _REGEX_POOL
    pool = _REGEX_POOL
    if pool is not None:
        return pool
    with _REGEX_POOL_LOCK:
        if _REGEX_POOL is None:
            _REGEX_POOL = ProcessPoolExecutor(max_workers=REGEX_MAX_WORKERS)
        return _REGEX_POOL


# --------------------------------------------------------------------------- #
#  Helper run in a separate process                                           #
# --------------------------------------------------------------------------- #
_worker_cache = {}
def _regex_search_worker(args: Tuple[str, int, str]) -> Optional[Tuple[str, int, str]]:
    """
    Compile the giant alternation and search the given text.
    Returns (group_name, start_char_idx, matched_segment) or None.
    """
    pattern_str, flags, text = args
    key = (pattern_str, flags)
    compiled = _worker_cache.get(key)
    if compiled is None:
        compiled = re.compile(pattern_str, flags)
        _worker_cache[key] = compiled
    # Return several matches, earliest first, so the caller can pass over one it
    # has suppressed.  Returning only the first meant a suppressed match hid
    # every later violation for the rest of the request.
    out = []
    for m in compiled.finditer(text):
        out.append((m.lastgroup, m.start(), m.group(0)))
        if len(out) >= 32:
            break
    return out or None


# --------------------------------------------------------------------------- #
#  Validator                                                                  #
# --------------------------------------------------------------------------- #
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


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
    validator_type = "regex"

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

        # Fold typographic apostrophes so `it'?s` also matches "it’s".  The fold
        # is one character for one, so offsets still index the original text.
        text = state.get_generated_text().translate(_APOSTROPHES)

        # ── timing probe ---------------------------------------------------
        t0 = time.perf_counter()

        future = _get_regex_pool().submit(
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

        for gname, violation_char_pos, matched_text_segment in result:
            violated_raw_pattern = self._group2raw.get(gname, "?")
            tok_idx = self._char_to_token_index(state, violation_char_pos)
            if tok_idx is None:
                logger.warning(
                    "RegexValidator: Could not map char_pos %d to token index for "
                    "pattern '%s'.",
                    violation_char_pos,
                    violated_raw_pattern,
                )
                continue
            if not self._is_ignored(tok_idx, violated_raw_pattern):
                break
        else:
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
