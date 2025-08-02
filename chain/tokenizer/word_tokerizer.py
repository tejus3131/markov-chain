from chain.tokenizer.base import TokenizerBase
from typing import List
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)


class WordTokenizer(TokenizerBase):
    """Tokenizer for word-level tokenization."""

    def __init__(self, lowercase: bool = True):
        """
        Initialize word tokenizer.

        Args:
            lowercase: Whether to convert text to lowercase
        """
        self._word_pattern = re.compile(r"\b\w+\b|[^\w\s]")
        self._lowercase = lowercase
        logger.debug(f"Initialized WordTokenizer with lowercase={lowercase}")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text at word level, preserving punctuation.

        Args:
            text: Input text string

        Returns:
            List of words and punctuation marks

        Examples:
            >>> tokenizer = WordTokenizer()
            >>> tokenizer.tokenize("Hello, world!")
            ['hello', ',', 'world', '!']
        """
        if self._lowercase:
            text = text.lower()

        tokens = self._word_pattern.findall(text)
        logger.debug(f"Tokenized into {len(tokens)} words")
        return tokens
