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
        Maps an absolute character position in the generated_text to a token index
        within state.generated_token_strings.
        """
        current_char_offset = 0
        for idx, token_string in enumerate(state.generated_token_strings):
            # Note: This assumes token_strings from API are raw and don't need decoding for length.
            # If state._decode_token was used to get text for char_pos, this might be tricky.
            # However, state.get_generated_text() uses _decode_token, so char_pos is in decoded space.
            # The token_strings in GenerationState are the raw strings from the API.
            # We need to compare char_pos against the *decoded* text cumulative length.
            # This is a subtle point. Let's assume for now that char_pos is relative to
            # the concatenated raw token strings, which is simpler for this mapping.
            # If char_pos is from state.get_generated_text(), then this mapping needs care.

            # Re-evaluating: state.get_generated_text() produces the text that validators see.
            # So, char_pos is in that decoded space.
            # We need to map this back to the original token strings.
            # This means we should probably iterate through decoded token strings.
            
            # Let's use the existing logic from SlopPhraseValidator which seems to work.
            # It iterates over state.generated_token_strings directly.
            # This implies char_pos is an index into the string formed by *directly concatenating*
            # state.generated_token_strings, not the fully decoded one.
            # This is simpler for the validator's internal logic if it also works on that basis.
            # The NGramValidator's _tokenize_text_with_spans works on state.get_generated_text().
            # This means char_pos *is* in the decoded space.

            # Correct approach: iterate through decoded tokens to match char_pos
            # This was in SlopPhraseValidator and RegexValidator, let's ensure it's robust.
            # The original implementation in those validators was:
            # cur = 0
            # for idx, tok_str in enumerate(state.generated_token_strings):
            #    # tok_str is raw. We need its decoded length.
            #    # This is problematic. _char_to_token_index should operate on the same text view
            #    # as the one that produced char_pos.
            #
            # Let's assume char_pos is an index into state.get_generated_text().
            # We need to find which raw token in state.generated_token_strings corresponds to this.

            # Simpler: The existing _char_to_token_index in other validators seems to assume
            # char_pos is relative to the concatenation of raw token strings.
            # If NGramValidator produces char_pos from the *decoded* text, this will mismatch.
            #
            # The NGramValidator's `_tokenize_text_with_spans` takes `generated_text = state.get_generated_text()`.
            # So `char_start_offset` is indeed an index into this fully decoded text.
            #
            # Therefore, `_char_to_token_index` MUST correctly map from this decoded text space.

            # Revised _char_to_token_index:
            # This needs to reconstruct the decoded text token by token.
            # This is inefficient if called often.
            # A better way: GenerationState could maintain a mapping if this becomes a bottleneck.
            # For now, let's do the reconstruction here.
            
            # Let's stick to the original simpler version from other validators and assume
            # that the char_pos obtained by the validator can be mapped using raw token string lengths.
            # This means the validator's span calculation might need to be adjusted if it's critical.
            # The key is consistency. If char_pos is from text derived from raw tokens, map using raw tokens.
            # If char_pos is from text derived from decoded tokens, map using decoded tokens.
            #
            # The `SlopPhraseValidator` and `RegexValidator` get `text = state.get_generated_text()`.
            # Then they find `char_pos` within this `text`.
            # Then they call `_char_to_token_index(state, char_pos)`.
            # The `_char_to_token_index` in those files is:
            # cur = 0
            # for idx, tok in enumerate(state.generated_token_strings): # tok is raw
            #    nxt = cur + len(tok) # len(raw_tok)
            #    if cur <= char_pos < nxt: return idx
            #    cur = nxt
            # This is only correct if char_pos was an index into the concatenation of raw tokens.
            # This is a bug in the original Slop/Regex validators if char_pos comes from decoded text.
            #
            # Let's fix it here in BaseValidator.
            
            # Corrected _char_to_token_index:
            # It must map a character position from the *fully decoded text*
            # (i.e., `state.get_generated_text()`) back to an index in the
            # `state.generated_token_strings` list.

            # To do this, we simulate the construction of get_generated_text()
            # token by token.
            
            # This is potentially slow. A better GenerationState would precompute this.
            # For now:
            current_decoded_text_len = 0
            for token_idx, raw_token_str in enumerate(state.generated_token_strings):
                # IMPORTANT: _decode_token is in generation_state.py, not directly accessible here
                # This highlights a design issue. BaseValidator shouldn't know about _decode_token.
                # GenerationState should provide this mapping.
                #
                # Quick Fix: Assume state can provide decoded token strings.
                # Let's assume GenerationState gets a method: get_decoded_token_string_at(idx)
                # For now, we'll use a simplified approach that might have slight off-by-one errors
                # if token decoding significantly changes lengths (e.g. "Ġ" -> " ").
                # The original approach in Slop/Regex was simpler but potentially flawed.
                #
                # The most robust way is for GenerationState to provide this mapping.
                # Lacking that, we must be careful.
                #
                # Let's use the version from SlopPhraseValidator and RegexValidator for now,
                # acknowledging its potential slight inaccuracy if char_pos is from fully decoded text
                # and raw token lengths differ significantly from decoded lengths.
                # The impact is usually small (e.g. related to leading space markers).
                # The N-gram validator's spans are from decoded text.

                # Reverting to the simpler, potentially slightly off, version for consistency with others for now.
                # This should be revisited for perfect accuracy.
                # The core issue is that char_pos is from decoded text, but we map using raw token lengths.
                
                # If state.generated_token_strings are already "somewhat" decoded by the API client
                # (e.g. if the API returns tokens that are mostly plain text), this is less of an issue.
                # The `_decode_token` in GenerationState handles specific markers like "Ġ", "Ċ", " ".
                
                # Let's use the version that was in SlopPhraseValidator / RegexValidator
                # as it's the current established pattern in the codebase.
                # This means NGramValidator's char_pos might be slightly off if decoding changes length.
                # The alternative is a more complex mapping in GenerationState.
                
                # For NGramValidator, the `char_start_offset` comes from `TreebankWordTokenizer().span_tokenize(generated_text)`
                # where `generated_text = state.get_generated_text()`. This is the decoded text.
                # So, this `_char_to_token_index` needs to correctly map from this decoded space.

                # Let's implement the more correct (but potentially slower) version here.
                # We need access to _decode_token or similar functionality.
                # This is a good reason for GenerationState to expose this mapping.

                # Temporary solution: Re-implement a simplified _decode_token logic here or assume
                # the impact of decoding on length for char_pos mapping is minor for most cases.
                # The original code in Slop/Regex validators:
                # cur = 0
                # for idx, tok_str in enumerate(state.generated_token_strings):
                #    nxt = cur + len(tok_str) # Uses raw token length
                #    if cur <= char_pos < nxt:
                #        return idx
                #    cur = nxt
                # return None
                # This is what we'll use for now to avoid breaking existing behavior,
                # but with a strong note that it's an approximation if char_pos is from fully decoded text.
                # The NGramValidator will use this, and its char_pos *is* from fully decoded text.
                # This means the `tok_idx` might occasionally be off by one if a token boundary
                # shifts due to decoding (e.g. "Ġhello" vs " hello").

                # For the purpose of this PR, let's assume the existing char_to_token_index is sufficient.
                # The one from SlopPhraseValidator:
                cur = 0
                for idx, raw_tok in enumerate(state.generated_token_strings):
                    # length **after** decoding (matches the cached text)
                    dec_len = len(state._decode_token(raw_tok))   # uses the same helper as the cache
                    nxt = cur + dec_len
                    if cur <= char_pos < nxt:
                        return idx
                    cur = nxt
                return None
            