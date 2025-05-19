# validators/base_validator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple, Any # Added Any

from state.generation_state import GenerationState # Keep relative import
from core.models import ViolationInfo # Keep relative import


class BaseValidator(ABC):
    """
    Common interface for sequence validators.
    """
    validator_type: str  # subclasses must set

    def __init__(self) -> None:
        # (token_index, str(details_key)) ⇒ suppressed
        # The details_key should be a canonical string representation of the specific violation cause.
        self._ignored: Set[Tuple[int, str]] = set()

    @abstractmethod
    def check(self, state: GenerationState) -> Optional[ViolationInfo]:
        raise NotImplementedError

    def ignore_violation(self, vio: ViolationInfo) -> None:
        """
        Permanently suppress further alerts for this (index, details_key) pair.
        The details_key should come from vio.details if it's structured,
        otherwise a string representation of vio.details.
        """
        # Default implementation assumes details can be str() directly.
        # Subclasses like NGramValidator might override this to use a specific key from details.
        if isinstance(vio.details, dict) and "suppression_detail_key" in vio.details:
             detail_key = vio.details["suppression_detail_key"]
        elif isinstance(vio.details, dict) and "phrase" in vio.details: # For SlopPhraseValidator
            detail_key = vio.details["phrase"]
        elif isinstance(vio.details, dict) and "pattern" in vio.details: # For RegexValidator
            detail_key = vio.details["pattern"]
        else:
            detail_key = str(vio.details) # Fallback
        self._ignored.add((vio.violation_index, detail_key))


    def _is_ignored(self, idx: int, details_str_representation: str) -> bool:
        """
        Checks if a violation at a given token index with specific details has been ignored.
        `details_str_representation` should be the same key used in `ignore_violation`.
        """
        return (idx, details_str_representation) in self._ignored

    # Common helper method
    def _char_to_token_index(
        self, state: GenerationState, char_pos: int
    ) -> Optional[int]:
        """
        Delegate to GenerationState’s fast mapper if present; otherwise
        fall back to the original raw-token-length scan.
        """
        if hasattr(state, "char_pos_to_token_index"):
            return state.char_pos_to_token_index(char_pos)

        # legacy fallback – keeps old behaviour for any external callers
        cur = 0
        for idx, raw_tok in enumerate(state.generated_token_strings):
            nxt = cur + len(raw_tok)
            if cur <= char_pos < nxt:
                return idx
            cur = nxt
        return None
