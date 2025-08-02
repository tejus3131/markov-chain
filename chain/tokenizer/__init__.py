"""
Tokenizer Component for Markov Chain Predictive Text System.

This module provides multi-level tokenization functionality for text processing,
including character-level, word-level, and n-gram tokenization with batch processing support.
"""

from chain.tokenizer.hybrid_tokenizer import Tokenizer
from typing import Optional
from chain.tokenizer.types import TokenizationResult, BatchTokenizationResult


def create_tokenizer(
    lowercase: bool = True, max_workers: Optional[int] = None
) -> Tokenizer:
    """
    Factory function to create a tokenizer instance.

    Args:
        lowercase: Whether to convert text to lowercase for word tokenization
        max_workers: Maximum number of workers for parallel processing

    Returns:
        Configured Tokenizer instance

    Examples:
        >>> tokenizer = create_tokenizer()
        >>> isinstance(tokenizer, Tokenizer)
        True
    """
    return Tokenizer(lowercase=lowercase, max_workers=max_workers)


__all__ = [
    "create_tokenizer",
    "Tokenizer",
    "TokenizationResult",
    "BatchTokenizationResult",
]
