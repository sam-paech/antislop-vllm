# utils/slop_helpers.py
import json
import logging
import unicodedata
from typing import Dict, Set, Tuple, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------- #
#  Load slop phrases                                            #
# ------------------------------------------------------------- #
def load_slop_phrases(filepath: str, top_n: Optional[int] = None) -> Dict[str, float]:
    """
    Read a JSON list like [["phrase", score], ...] and return a
    {lowercase_phrase: score} dict.  If *top_n* is supplied, only the
    first N entries of the file are used.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Slop phrases file not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in {filepath}")
        return {}

    if not isinstance(data, list):
        logger.error("Expected JSON root to be a list.")
        return {}

    if top_n is not None and top_n > 0:
        data = data[:top_n]
    elif top_n is not None and top_n <= 0:
        data = []

    phrases: Dict[str, float] = {}
    for item in data:
        if isinstance(item, list) and item and isinstance(item[0], str):
            phrase = item[0].lower()
            score  = float(item[1]) if len(item) > 1 else 0.0
            phrases[phrase] = score
        else:
            logger.warning(f"Skipping malformed item in slop list: {item}")

    logger.info(f"Loaded {len(phrases)} slop phrases from {filepath}.")
    return phrases

# ------------------------------------------------------------- #
#  Unicode-aware helpers                                        #
# ------------------------------------------------------------- #
def _is_word_char(ch: str) -> bool:
    """
    Return True iff *ch* should be considered part of a word in *any*
    writing system:

        • General Category starts with L (Letter), N (Number) or M (Mark)
          – that covers diacritics and CJK ideographs.
    """
    if not ch:
        return False
    cat = unicodedata.category(ch)
    return cat and cat[0] in {"L", "N", "M"}

# ------------------------------------------------------------- #
#  Earliest-hit matcher                                         #
# ------------------------------------------------------------- #
def detect_disallowed_sequence(
    text: str,
    slop_phrases_keys: Set[str],
    max_phrase_len: int,
    min_phrase_len: int,
    check_n_chars_back: int = 1,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Scan the last *check_n_chars_back* characters of *text* for any phrase
    in *slop_phrases_keys*.  Return a tuple *(matched_phrase, start_index)*
    where *start_index* is an **absolute** char offset in *text*.

    The algorithm evaluates **all** start positions and **all** lengths,
    then chooses the match whose start index is the smallest.  When more
    than one phrase begins at that earliest offset, the longest phrase is
    preferred so we block the full expression.

    To trigger a ban, the sequence must not have a word-like character
    (not punctuation or whitespace) directly on either side. That is to say, we
    are not banning disallowed sequences that occur as substrings in longer
    words. The exception is if the banned string is already bookended by
    a non-word character.

    Examples: 
    banned string "cat"
      - won't trigger a ban for "cation"
      - will trigger a ban on "cat[morecat]"
    banned string "cat["
      - *will* trigger a ban on "cat[morecat]", because the banned string
        ends with a non-word character.    

    If nothing is found, returns *(None, None)*.
    """
    if not text or not slop_phrases_keys or max_phrase_len == 0:
        return None, None

    text_lower = text.lower()
    text_len   = len(text_lower)

    win_start  = max(0, text_len - check_n_chars_back)
    window     = text_lower[win_start:]
    win_len    = len(window)

    for start in range(0, win_len - min_phrase_len + 1):
        max_len_here = min(max_phrase_len, win_len - start)

        for length in range(min_phrase_len, max_len_here + 1):
            cand = window[start : start + length]
            if cand not in slop_phrases_keys:
                continue

            global_pos = win_start + start
            right_pos  = global_pos + length

            need_left  = _is_word_char(cand[0])
            need_right = _is_word_char(cand[-1])

            left_ok  = (
                True if not need_left
                else global_pos == 0
                or not _is_word_char(text_lower[global_pos - 1])
            )
            right_ok = (
                True if not need_right
                else right_pos >= text_len
                or not _is_word_char(text_lower[right_pos])
            )

            if left_ok and right_ok:
                return cand, global_pos        # ← EARLY EXIT

    return None, None