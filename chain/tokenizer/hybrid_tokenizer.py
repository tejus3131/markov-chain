from chain.tokenizer.word_tokerizer import WordTokenizer
from chain.tokenizer.character_tokenizer import CharacterTokenizer
from chain.tokenizer.types import TokenizationResult, BatchTokenizationResult
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Multi-level tokenizer that breaks text into various token formats
    for different prediction contexts with batch processing support.
    """

    def __init__(self, lowercase: bool = True, max_workers: Optional[int] = None):
        """
        Initialize the tokenizer.

        Args:
            lowercase: Whether to convert text to lowercase for word tokenization
            max_workers: Maximum number of workers for parallel processing (None for CPU count)
        """
        self._char_tokenizer = CharacterTokenizer()
        self._word_tokenizer = WordTokenizer(lowercase=lowercase)
        self._max_workers = max_workers or multiprocessing.cpu_count()
        logger.info(
            f"Initialized multi-level Tokenizer with max_workers={self._max_workers}"
        )

    def tokenize_characters(self, text: str) -> List[str]:
        """
        Tokenize text at character level.

        Args:
            text: Input text string

        Returns:
            List of individual characters including spaces

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return self._char_tokenizer.tokenize(text)

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text at word level, preserving punctuation.

        Args:
            text: Input text string

        Returns:
            List of words and punctuation marks

        Raises:
            ValueError: If text is None or not a string
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return self._word_tokenizer.tokenize(text)

    def generate_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from a list of tokens.

        Args:
            tokens: List of tokens (characters or words)
            n: Size of n-gram (must be positive)

        Returns:
            List of n-gram tuples

        Raises:
            ValueError: If n is not positive or tokens is None

        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.generate_ngrams(['a', 'b', 'c'], 2)
            [('a', 'b'), ('b', 'c')]
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        if tokens is None:
            raise ValueError("tokens cannot be None")

        if len(tokens) < n:
            logger.warning(f"Token list too short for {n}-grams: {len(tokens)} < {n}")
            return []

        ngrams = self._generate_ngrams_internal(tokens, n)
        logger.debug(f"Generated {len(ngrams)} {n}-grams")
        return ngrams

    def _generate_ngrams_internal(
        self, tokens: List[str], n: int
    ) -> List[Tuple[str, ...]]:
        """
        Internal method to generate n-grams.

        Args:
            tokens: List of tokens
            n: Size of n-gram

        Returns:
            List of n-gram tuples
        """
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def tokenize_all_levels(
        self, text: str, ngram_sizes: Optional[List[int]] = None
    ) -> TokenizationResult:
        """
        Perform comprehensive tokenization at all levels.

        Args:
            text: Input text string
            ngram_sizes: List of n-gram sizes to generate (default: [2, 3])

        Returns:
            TokenizationResult containing all tokenization results

        Raises:
            ValueError: If text is None or not a string
            ValueError: If ngram_sizes contains non-positive integers

        Examples:
            >>> tokenizer = Tokenizer()
            >>> result = tokenizer.tokenize_all_levels("Hello world", [2, 3])
            >>> isinstance(result, TokenizationResult)
            True
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        if ngram_sizes is None:
            ngram_sizes = [2, 3]

        if not all(isinstance(n, int) and n > 0 for n in ngram_sizes):
            raise ValueError("All n-gram sizes must be positive integers")

        logger.info(
            f"Tokenizing text of length {len(text)} with n-gram sizes: {ngram_sizes}"
        )

        # Character-level tokenization
        char_tokens = self.tokenize_characters(text)

        # Word-level tokenization
        word_tokens = self.tokenize_words(text)

        # Generate character n-grams
        char_ngrams = self._generate_all_ngrams(char_tokens, ngram_sizes, "character")

        # Generate word n-grams
        word_ngrams = self._generate_all_ngrams(word_tokens, ngram_sizes, "word")

        result = TokenizationResult(
            characters=char_tokens,
            words=word_tokens,
            char_ngrams=char_ngrams,
            word_ngrams=word_ngrams,
        )

        logger.info(
            f"Tokenization complete: {len(char_tokens)} chars, {len(word_tokens)} words"
        )
        return result

    def _generate_all_ngrams(
        self, tokens: List[str], ngram_sizes: List[int], token_type: str
    ) -> Dict[int, List[Tuple[str, ...]]]:
        """
        Generate n-grams for all specified sizes.

        Args:
            tokens: List of tokens
            ngram_sizes: List of n-gram sizes
            token_type: Type of tokens (for logging)

        Returns:
            Dictionary mapping n-gram size to list of n-grams
        """
        ngrams_dict = {}

        for n in ngram_sizes:
            if len(tokens) >= n:
                ngrams = self._generate_ngrams_internal(tokens, n)
                ngrams_dict[n] = ngrams
                logger.debug(f"Generated {len(ngrams)} {token_type} {n}-grams")
            else:
                logger.debug(f"Skipping {token_type} {n}-grams: insufficient tokens")

        return ngrams_dict

    def tokenize_batch(
        self,
        texts: List[str],
        ngram_sizes: Optional[List[int]] = None,
        use_processes: bool = False,
        chunk_size: Optional[int] = None,
    ) -> BatchTokenizationResult:
        """
        Tokenize multiple texts in parallel.

        Args:
            texts: List of input text strings
            ngram_sizes: List of n-gram sizes to generate (default: [2, 3])
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            chunk_size: Size of chunks for batch processing (None for automatic)

        Returns:
            BatchTokenizationResult containing results and error information

        Examples:
            >>> tokenizer = Tokenizer()
            >>> texts = ["Hello world", "Python programming"]
            >>> batch_result = tokenizer.tokenize_batch(texts)
            >>> len(batch_result.results) == 2
            True
        """
        if not texts:
            return BatchTokenizationResult(results=[], failed_indices=[], errors={})

        if ngram_sizes is None:
            ngram_sizes = [2, 3]

        logger.info(f"Starting batch tokenization of {len(texts)} texts")

        # Prepare results storage
        results: List[Optional[TokenizationResult]] = [None] * len(texts)
        failed_indices = []
        errors = {}

        # Choose executor type
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        # Calculate chunk size if not provided
        if chunk_size is None:
            chunk_size = max(1, len(texts) // (self._max_workers * 4))

        # Create partial function for easier parallel execution
        tokenize_func = partial(self._tokenize_single, ngram_sizes=ngram_sizes)

        with executor_class(max_workers=self._max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(tokenize_func, text): i for i, text in enumerate(texts)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Failed to tokenize text at index {index}: {str(e)}")
                    failed_indices.append(index)
                    errors[index] = str(e)

        # Filter out None values from failed tokenizations
        valid_results = [r for r in results if r is not None]

        logger.info(
            f"Batch tokenization complete: {len(valid_results)} successful, {len(failed_indices)} failed"
        )

        return BatchTokenizationResult(
            results=valid_results, failed_indices=failed_indices, errors=errors
        )

    def _tokenize_single(self, text: str, ngram_sizes: List[int]) -> TokenizationResult:
        """
        Helper method for parallel tokenization of a single text.

        Args:
            text: Input text string
            ngram_sizes: List of n-gram sizes

        Returns:
            TokenizationResult
        """
        return self.tokenize_all_levels(text, ngram_sizes)

    def tokenize_batch_characters(self, texts: List[str]) -> List[List[str]]:
        """
        Batch tokenize texts at character level.

        Args:
            texts: List of input text strings

        Returns:
            List of character token lists
        """
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self.tokenize_characters, texts))

    def tokenize_batch_words(self, texts: List[str]) -> List[List[str]]:
        """
        Batch tokenize texts at word level.

        Args:
            texts: List of input text strings

        Returns:
            List of word token lists
        """
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(self.tokenize_words, texts))
