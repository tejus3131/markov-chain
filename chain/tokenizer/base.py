from abc import ABC, abstractmethod
from typing import List


class TokenizerBase(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Abstract method for tokenization.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        pass
