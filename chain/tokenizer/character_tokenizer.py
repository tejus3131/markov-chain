from chain.tokenizer.base import TokenizerBase
from typing import List
import logging

# Configure logging
logger = logging.getLogger(__name__)


class CharacterTokenizer(TokenizerBase):
    """Tokenizer for character-level tokenization."""

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text at character level.

        Args:
            text: Input text string

        Returns:
            List of individual characters including spaces

        Examples:
            >>> tokenizer = CharacterTokenizer()
            >>> tokenizer.tokenize("Hello")
            ['H', 'e', 'l', 'l', 'o']
        """
        logger.debug(f"Tokenizing {len(text)} characters")
        return list(text)
