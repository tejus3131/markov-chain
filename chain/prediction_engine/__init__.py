
from chain.prediction_engine.types import (
    PredictionResult,
    PredictionLevel,
    Prediction,
    ContextSuggestionsResult,
    ContextSuggestion,
    TextGenerationConfig,
)
from chain.prediction_engine.engine import PredictionEngine
from typing import List, Optional, Dict, Any


def create_prediction_engine(
    knowledge_base: Any,
    probability_normalizer: Any,
    ngram_sizes: Optional[List[int]] = None,
    backoff_weights: Optional[Dict[int, float]] = None
) -> PredictionEngine:
    """
    Factory function to create a PredictionEngine instance.
    
    Args:
        knowledge_base: KnowledgeBase instance
        probability_normalizer: ProbabilityNormalizer instance
        ngram_sizes: N-gram sizes in order of preference (largest first)
        backoff_weights: Custom backoff weights for each n-gram size
        
    Returns:
        Initialized PredictionEngine instance
    """
    return PredictionEngine(
        knowledge_base=knowledge_base,
        probability_normalizer=probability_normalizer,
        ngram_sizes=ngram_sizes,
        backoff_weights=backoff_weights
    )

__all__ = [
    "create_prediction_engine",
    "PredictionEngine",
    "PredictionResult",
    "PredictionLevel",
    "Prediction",
    "ContextSuggestionsResult",
    "ContextSuggestion",
    "TextGenerationConfig",
]