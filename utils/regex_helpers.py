# utils/regex_helpers.py
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_regex_patterns(filepath: str) -> List[str]:
    """
    Read a JSON file containing a list of regex strings and return them.
    """
    if not filepath:
        return []

    p = Path(filepath)
    if not p.exists():
        logger.error(f"Regex block-list file not found: {filepath}")
        return []

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Regex JSON decode error in {filepath}: {e}")
        return []

    if not isinstance(data, list):
        logger.error("Regex block-list JSON must be a list of strings.")
        return []

    patterns: List[str] = []
    for item in data:
        if isinstance(item, str):
            patterns.append(item)
        else:
            logger.warning(f"Skipping non-string regex entry: {item!r}")

    logger.info(f"Loaded {len(patterns)} regex patterns from {filepath}")
    return patterns
