from dataclasses import dataclass, field # Added field
from typing import Any, List, Tuple, Dict, Optional

@dataclass
class ViolationInfo:
    """Holds information about a detected validation violation."""
    validator_type: str
    # Store index relative to the start of generated_token_strings
    violation_index: int
    original_token_string: str # Store the string directly
    details: Any

@dataclass
class ApiChunkResult:
    """Result from an API generate_chunk call (string-based)."""
    generated_text: str # The full text chunk returned by the API
    # List of token strings as determined by the API's logprobs structure
    token_strings: List[str] = field(default_factory=list)
    # Maps relative position index (0 to n-1) in token_strings to top logprobs list
    logprobs: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    finish_reason: Optional[str] = None