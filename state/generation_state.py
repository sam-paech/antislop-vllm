import logging
from typing import Dict, List, Optional, Tuple

from core.models import ApiChunkResult

logger = logging.getLogger(__name__)


def _decode_token(token: str) -> str:
    if not token:
        return token

    # 1) GPT-BPE newline
    if token.startswith("Ċ"):
        token = "\n" + token[1:]

    # 2) GPT-BPE / SP leading-space
    if token.startswith("Ġ"):
        token = " " + token[1:]
    elif token.startswith("▁"):
        token = " " + token[1:]

    # 3) Remove *any remaining* marker occurrences inside the token
    token = (
        token.replace("Ċ", "\n")
             .replace("Ġ", " ")
             .replace("▁", " ")
    )

    return token



def _tokens_to_text(tokens: List[str]) -> str:
    """Utility: join & decode a list of token strings."""
    return "".join(_decode_token(tok) for tok in tokens)


class GenerationState:
    """
    Manages the generated sequence using *token strings* (no local tokenizer)
    while offering decoded-text views for prompts and output.
    """

    def __init__(self, prompt_string: str):
        if not isinstance(prompt_string, str):
            raise TypeError("prompt_string must be a string.")
        self.prompt_string: str = prompt_string
        self.generated_token_strings: List[str] = []
        self._decoded_so_far: str = ""      # running decoded view
        self.logprobs_cache: Dict[int, List[Tuple[str, float]]] = {}
        logger.info(
            f"GenerationState initialised. Prompt length (chars): {len(prompt_string)}"
        )

    # ------------------------------------------------------------------ #
    #  Token management                                                   #
    # ------------------------------------------------------------------ #
    def append_chunk(self, chunk_result: ApiChunkResult) -> None:
        if not chunk_result.token_strings:
            logger.warning("append_chunk called with no token strings.")
            return

        start = len(self.generated_token_strings)
        self.generated_token_strings.extend(chunk_result.token_strings)

        for rel_pos, alt_list in chunk_result.logprobs.items():
            abs_idx = start + rel_pos
            if abs_idx < len(self.generated_token_strings):
                self.logprobs_cache[abs_idx] = alt_list
            else:
                logger.warning(
                    f"logprob index {abs_idx} out of bounds "
                    f"(len={len(self.generated_token_strings)})"
                )

    def truncate(self, generated_index: int) -> None:
        if generated_index < 0 or generated_index >= len(self.generated_token_strings):
            return
        logger.info(
            f"Truncating generated strings from index {generated_index}. "
            f"Old length: {len(self.generated_token_strings)}"
        )
        self.generated_token_strings = self.generated_token_strings[:generated_index]
        # rebuild decoded cache so validators stay consistent
        self._decoded_so_far = _tokens_to_text(self.generated_token_strings)

        self.logprobs_cache = {
            k: v for k, v in self.logprobs_cache.items() if k < generated_index
        }

    def replace_token_string(self, generated_index: int, new_token_string: str) -> None:
        if not (0 <= generated_index < len(self.generated_token_strings)):
            logger.error(
                f"replace_token_string: index {generated_index} out of range "
                f"(len={len(self.generated_token_strings)})"
            )
            return
        self.generated_token_strings[generated_index] = new_token_string
        self._decoded_so_far = _tokens_to_text(self.generated_token_strings)

    # ------------------------------------------------------------------ #
    #  Views as decoded text                                              #
    # ------------------------------------------------------------------ #
    def get_generated_text(self) -> str:
        # O(1) – no re-join every call
        return self._decoded_so_far

    def get_full_text(self) -> str:
        return self.prompt_string + self.get_generated_text()

    def get_hypothetical_generated_text(
        self, truncate_index: int, next_token_string: str
    ) -> str:
        if truncate_index < 0:
            truncate_index = 0
        safe_index = min(truncate_index, len(self.generated_token_strings))
        tokens = (
            self.generated_token_strings[:safe_index] + [next_token_string]
        )
        return _tokens_to_text(tokens)

    # ------------------------------------------------------------------ #
    #  Simple helpers                                                     #
    # ------------------------------------------------------------------ #
    def get_generated_length(self) -> int:
        return len(self.generated_token_strings)

    def get_logprobs(self, generated_index: int) -> Optional[List[Tuple[str, float]]]:
        return self.logprobs_cache.get(generated_index)

    def get_token_string_at(self, generated_index: int) -> Optional[str]:
        if 0 <= generated_index < len(self.generated_token_strings):
            return self.generated_token_strings[generated_index]
        return None
