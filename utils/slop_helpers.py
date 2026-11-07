# utils/slop_helpers.py
import json
import logging
import unicodedata
from typing import Callable, Dict, Set, Tuple, Optional

# Typographic apostrophes and quotes folded to ASCII before matching.  The ban
# lists are written with ASCII `'`, so "couldn’t help but" (U+2019) never
# matched "couldn't help but".  Every mapping is one character to one character,
# so match offsets still index the original text.
_FOLD_TABLE = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'",
                             "\u201c": '"', "\u201d": '"'})


def fold_for_matching(text: str) -> str:
    """Lower-case and fold quotes **without changing the length** of *text*.

    ``str.lower()`` can lengthen a string ("İ" becomes two characters), which
    shifted every later match offset and blamed the wrong token.
    """
    text = text.translate(_FOLD_TABLE)
    lowered = text.lower()
    if len(lowered) == len(text):
        return lowered
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)

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
    require_word_boundaries: bool = True,
    allow_end_of_text_match: bool = True,
    skip: Optional[Callable[[str, int], bool]] = None,
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

    *allow_end_of_text_match* controls the **provisional** tail of a streamed
    generation.  During decoding, ``text`` is only the prefix produced so far, so
    a match that ends at the very last character has no right-hand neighbour yet.
    Treating that as a boundary made a ban on ``"la"`` fire the moment the model
    had emitted ``" la"`` — even though the next token was about to extend it to
    ``"land"``.  The back-tracker then rewrote the token, the model re-emitted the
    same prefix, and the sampler looped at the same token index forever.  Passing
    ``False`` while generation is still running defers such a match until either
    more text arrives (which decides it) or the final sweep runs with ``True``.

    A match ending in a non-word character needs no lookahead and is never
    deferred, because the ban already demands a non-word character there.

    If nothing is found, returns *(None, None)*.
    """
    if not text or not slop_phrases_keys or max_phrase_len == 0:
        return None, None

    text_lower = fold_for_matching(text)
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

            # *skip* lets the caller pass over a match it has already
            # suppressed and keep looking.  Returning the earliest match
            # unconditionally meant one suppressed phrase hid every later
            # violation in the window.
            if skip is not None and skip(cand, global_pos):
                continue

            if not require_word_boundaries:
                # No boundary checks — accept any substring
                return cand, global_pos

            need_left  = _is_word_char(cand[0])
            need_right = _is_word_char(cand[-1])

            left_ok  = (
                True if not need_left
                else global_pos == 0
                or not _is_word_char(text_lower[global_pos - 1])
            )

            # A match whose right edge is the end of the *current* text is
            # provisional: the next token may extend the word and break the
            # boundary.  Defer it unless this is the final pass.
            at_text_end = right_pos >= text_len
            if need_right and at_text_end and not allow_end_of_text_match:
                continue

            right_ok = (
                True if not need_right
                else at_text_end
                or not _is_word_char(text_lower[right_pos])
            )

            if left_ok and right_ok:
                return cand, global_pos        # ← EARLY EXIT


    return None, None