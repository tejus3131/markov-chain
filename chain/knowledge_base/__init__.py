from typing import List, Optional
from chain.knowledge_base.knowledge_base import KnowledgeBase
from chain.knowledge_base.types import (
    KnowledgeBaseStatistics,
    NGramStatistics,
    PredictionCandidate,
    ContextPredictions,
)


def create_knowledge_base(ngram_sizes: Optional[List[int]] = None) -> KnowledgeBase:
    """
    Factory function to create a knowledge base instance.

    Args:
        ngram_sizes: List of n-gram sizes (default: [2, 3])

    Returns:
        Configured KnowledgeBase instance

    Examples:
        >>> kb = create_knowledge_base()
        >>> isinstance(kb, KnowledgeBase)
        True
    """
    if ngram_sizes is None:
        ngram_sizes = [2, 3]

    return KnowledgeBase(ngram_sizes)


__all__ = [
    "create_knowledge_base",
    "KnowledgeBase",
    "KnowledgeBaseStatistics",
    "NGramStatistics",
    "PredictionCandidate",
    "ContextPredictions",
]
