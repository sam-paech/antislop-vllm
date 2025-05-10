from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from core.models import ApiChunkResult

class BaseApiClient(ABC):
    @abstractmethod
    def generate_chunk(
        self,
        prompt_text: str,
        max_tokens: int,
        top_logprobs: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        min_p: Optional[float],
        timeout: int,
        stop_sequences: Optional[List[str]],
        **kwargs,
    ) -> ApiChunkResult:
        """
        Generates the next chunk of text using the API.

        Args:
            prompt_text: The text prompt to continue from.
            max_tokens: The maximum number of tokens to generate in this chunk.
            top_logprobs: The number of top logprobs to request.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop_sequences: Optional list of sequences to stop generation at.
            **kwargs: Additional API-specific parameters.

        Returns:
            An ApiChunkResult containing the new token IDs, their logprobs (relative position),
            and the finish reason.
        """
        pass