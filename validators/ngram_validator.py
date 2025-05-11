# validators/ngram_validator.py
import logging
import string
from typing import List, Optional, Set, Tuple, Union, Dict, Any
from collections import Counter # Not strictly needed for validator, but good for n-gram concepts

import nltk

from .base_validator import BaseValidator
from state.generation_state import GenerationState
from core.models import ViolationInfo

logger = logging.getLogger(__name__)

# Helper to ensure NLTK data is available
_NLTK_DATA_ENSURED = False
def _ensure_nltk_data():
    global _NLTK_DATA_ENSURED
    if _NLTK_DATA_ENSURED:
        return

    data_to_check_download = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }
    for name, path_check in data_to_check_download.items():
        try:
            nltk.data.find(path_check)
        except (nltk.downloader.DownloadError, LookupError):
            logger.info(f"NLTK '{name}' resource not found. Attempting download...")
            try:
                nltk.download(name, quiet=True)
                logger.info(f"NLTK '{name}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download NLTK '{name}': {e}. NGramValidator might not work correctly.")
                # Depending on strictness, could raise an error here
    _NLTK_DATA_ENSURED = True


# Import NLTK components after ensuring data
try:
    _ensure_nltk_data()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, TreebankWordTokenizer # Treebank for span_tokenize
    from nltk.util import ngrams as nltk_ngrams_util
except ImportError as e:
    logger.error(f"NLTK import error after attempting data download: {e}. NGramValidator will be disabled.")
    # Create dummy classes/functions if NLTK is critical and failed to load, to prevent crashes
    stopwords = None
    TreebankWordTokenizer = None
    nltk_ngrams_util = None

from utils.profile_helpers import profile_zone
class NGramValidator(BaseValidator):
    """
    Hard-ban validator for specified N-grams.
    Can optionally remove stopwords before checking.
    """

    def __init__(
        self,
        banned_ngrams_config: List[Union[str, List[str]]],
        remove_stopwords_flag: bool = True,
        language: str = 'english'
    ) -> None:
        super().__init__()

        if stopwords is None or TreebankWordTokenizer is None or nltk_ngrams_util is None:
            logger.error("NLTK components not available. NGramValidator is disabled.")
            self.banned_ngrams_tuples: Set[Tuple[str, ...]] = set()
            self.min_ngram_len = 0
            self.max_ngram_len = 0
            self.is_disabled = True
            return
        self.is_disabled = False

        self.remove_stopwords_flag = remove_stopwords_flag
        try:
            self.stopwords_set = set(stopwords.words(language)) if self.remove_stopwords_flag else set()
        except Exception as e:
            logger.warning(f"Could not load stopwords for language '{language}': {e}. Stopword removal might be affected.")
            self.stopwords_set = set() # Fallback to no stopwords

        self.punctuation_set = set(string.punctuation)
        self._tokenizer = TreebankWordTokenizer() # For span_tokenize

        self.banned_ngrams_tuples: Set[Tuple[str, ...]] = set()
        self.min_ngram_len = float('inf')
        self.max_ngram_len = 0

        for ngram_item in banned_ngrams_config:
            raw_words: List[str] = []
            if isinstance(ngram_item, str):
                # Tokenize the string, keep only alpha, lowercase
                raw_words = [w.lower() for w in word_tokenize(ngram_item) if w.isalpha()]
            elif isinstance(ngram_item, list):
                raw_words = [str(w).lower() for w in ngram_item if isinstance(w, str) and w.isalpha()]
            else:
                logger.warning(f"Skipping invalid n-gram item in config: {ngram_item!r}")
                continue

            if self.remove_stopwords_flag:
                processed_words = [w for w in raw_words if w not in self.stopwords_set]
            else:
                processed_words = raw_words
            
            if processed_words:
                ngram_tuple = tuple(processed_words)
                self.banned_ngrams_tuples.add(ngram_tuple)
                current_len = len(ngram_tuple)
                self.min_ngram_len = min(self.min_ngram_len, current_len)
                self.max_ngram_len = max(self.max_ngram_len, current_len)

        if not self.banned_ngrams_tuples:
            self.min_ngram_len = 0 # Avoid inf if no ngrams loaded
            logger.info("NGramValidator initialized with no banned n-grams.")
        else:
            logger.info(
                f"NGramValidator ready: {len(self.banned_ngrams_tuples)} unique banned n-grams "
                f"(lengths {self.min_ngram_len}-{self.max_ngram_len}), "
                f"remove_stopwords={self.remove_stopwords_flag}"
            )

    def _tokenize_text_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Tokenizes text and returns list of (token_string, start_char, end_char)."""
        if not text or self._tokenizer is None:
            return []
        
        # TreebankWordTokenizer().span_tokenize gives (start, end) of each token
        spans = list(self._tokenizer.span_tokenize(text))
        tokens_with_spans = [(text[start:end], start, end) for start, end in spans]
        return tokens_with_spans

    def _get_cleaned_words_with_original_indices(
        self, text_tokens_with_spans: List[Tuple[str, int, int]]
    ) -> List[Tuple[str, int]]:
        """
        Cleans tokens (lowercase, optionally remove stopwords/punctuation) and
        maps them to their original index in text_tokens_with_spans.
        Returns list of (cleaned_word, original_token_list_index).
        """
        cleaned_words_map = []
        for original_idx, (word_str, _start_char, _end_char) in enumerate(text_tokens_with_spans):
            lower_word = word_str.lower()

            # Filter out non-alphabetic tokens (punctuation, numbers, etc.)
            # This is a stricter filter than just punctuation_set.
            if not word_str.isalpha(): # Consider if this is too strict or just right
                continue

            if self.remove_stopwords_flag and lower_word in self.stopwords_set:
                continue
            
            # If not removing stopwords, we still skip pure punctuation based on isalpha above.
            # The word itself (lower_word) is kept.
            cleaned_words_map.append((lower_word, original_idx))
        return cleaned_words_map


    @profile_zone()
    def check(self, state: GenerationState) -> Optional[ViolationInfo]:
        if self.is_disabled or not self.banned_ngrams_tuples or state.get_generated_length() == 0 or self.min_ngram_len == 0:
            return None

        generated_text = state.get_generated_text()
        if not generated_text.strip():
            return None

        # 1. Tokenize the full generated text with original character spans
        #    This list contains (raw_token_string, start_char_offset, end_char_offset)
        raw_tokens_with_char_spans = self._tokenize_text_with_spans(generated_text)
        if not raw_tokens_with_char_spans:
            return None

        # 2. Clean these tokens (stopwords, etc.) and keep track of their original index
        #    This list contains (cleaned_word_string, index_in_raw_tokens_with_char_spans)
        cleaned_words_and_their_raw_indices = self._get_cleaned_words_with_original_indices(raw_tokens_with_char_spans)
        
        # Extract just the sequence of cleaned words for n-gram generation
        cleaned_word_sequence = [word for word, _raw_idx in cleaned_words_and_their_raw_indices]

        if len(cleaned_word_sequence) < self.min_ngram_len:
            return None # Not enough cleaned words to form any banned n-gram

        # 3. Iterate through possible n-gram lengths and check for banned sequences
        # We want the *earliest* violation in the text.
        earliest_violation: Optional[Tuple[int, Tuple[str, ...], int]] = None # (char_pos, ngram_tuple, raw_token_idx_of_first_word)

        for n in range(self.min_ngram_len, min(self.max_ngram_len, len(cleaned_word_sequence)) + 1):
            if n == 0: continue # Should not happen if min_ngram_len is set correctly
            
            # Generate n-grams from the cleaned_word_sequence
            # nltk_ngrams_util yields tuples of words
            for i, current_ngram_tuple in enumerate(nltk_ngrams_util(cleaned_word_sequence, n)):
                if current_ngram_tuple in self.banned_ngrams_tuples:
                    # Found a banned n-gram. Now map it back to the original text.
                    # `i` is the starting index of this n-gram in `cleaned_word_sequence`.
                    # The first word of this n-gram is `cleaned_word_sequence[i]`.
                    # Its info (including original index in raw_tokens_with_char_spans) is:
                    _first_cleaned_word, raw_token_list_idx_of_first_word = cleaned_words_and_their_raw_indices[i]
                    
                    # Get the character span of this first raw token
                    _raw_token_str, char_start_offset, _char_end_offset = raw_tokens_with_char_spans[raw_token_list_idx_of_first_word]

                    if earliest_violation is None or char_start_offset < earliest_violation[0]:
                        earliest_violation = (char_start_offset, current_ngram_tuple, raw_token_list_idx_of_first_word)
                    # If same start, prefer longer n-gram (though less critical here than phrase matching)
                    elif char_start_offset == earliest_violation[0] and len(current_ngram_tuple) > len(earliest_violation[1]):
                         earliest_violation = (char_start_offset, current_ngram_tuple, raw_token_list_idx_of_first_word)


        if earliest_violation is None:
            return None

        violation_char_pos, matched_ngram_tuple, _ = earliest_violation
        
        # Convert character position to API token index
        tok_idx = self._char_to_token_index(state, violation_char_pos)
        if tok_idx is None:
            # This can happen if the character position is somehow outside the tokenized spans
            # or if the text has changed in a way that invalidates the mapping.
            logger.error(f"NGramValidator: Could not map char_pos {violation_char_pos} to token_idx for n-gram {' '.join(matched_ngram_tuple)}.")
            return None

        # Check if this specific violation (at this token index, for this n-gram) has been ignored
        # For n-grams, the "details" string should uniquely identify the n-gram itself.
        ngram_detail_str = f"{' '.join(matched_ngram_tuple)} (stopwords_removed={self.remove_stopwords_flag})"
        if self._is_ignored(tok_idx, ngram_detail_str):
            return None

        # Create context snippet
        context_window = 30 # Characters before and after
        snippet_start = max(0, violation_char_pos - context_window)
        # Estimate end of n-gram: find last word of n-gram in cleaned_words_and_their_raw_indices
        # and get its end char pos. This is a bit complex.
        # Simpler: just use length of matched_ngram_tuple words + spaces.
        # For a more accurate end, we'd need to map all words of the n-gram back.
        # For now, a simpler context end:
        estimated_ngram_char_len = sum(len(w) for w in matched_ngram_tuple) + max(0, len(matched_ngram_tuple) -1)
        snippet_end = min(len(generated_text), violation_char_pos + estimated_ngram_char_len + context_window)
        context_snippet = generated_text[snippet_start:snippet_end].replace("\n", " ")


        logger.warning(
            f"NGram violation: '{' '.join(matched_ngram_tuple)}' "
            f"(stopwords_removed={self.remove_stopwords_flag}) "
            f"@tok={tok_idx} (char_pos={violation_char_pos}). Context: ...{context_snippet}..."
        )

        return ViolationInfo(
            validator_type="ngram",
            violation_index=tok_idx,
            original_token_string=state.get_token_string_at(tok_idx) or "[UNK]",
            details={
                "ngram_tuple": list(matched_ngram_tuple), # Convert tuple to list for JSON
                "ngram_string": ' '.join(matched_ngram_tuple),
                "remove_stopwords_active": self.remove_stopwords_flag,
                "context": context_snippet,
                # Store the unique detail string used for suppression
                "suppression_detail_key": ngram_detail_str
            },
        )

    # Override ignore_violation and _is_ignored to use the specific key from details
    def ignore_violation(self, vio: ViolationInfo) -> None:
        if isinstance(vio.details, dict) and "suppression_detail_key" in vio.details:
            detail_key = vio.details["suppression_detail_key"]
            self._ignored.add((vio.violation_index, detail_key))
        else: # Fallback to default behavior if key is missing
            super().ignore_violation(vio)

    def _is_ignored(self, idx: int, details_str_representation: str) -> bool:
        # For NGramValidator, details_str_representation is the pre-computed suppression_detail_key
        return (idx, details_str_representation) in self._ignored