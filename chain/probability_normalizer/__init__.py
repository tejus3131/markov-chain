from typing import Union
from chain.probability_normalizer.normalizer import ProbabilityNormalizer
from chain.probability_normalizer.types import (
    SmoothingMethod,
    PerplexityResult,
    PredictionResult,
    ProbabilityDistribution,
)


def create_probability_normalizer(
    smoothing_method: Union[SmoothingMethod, str] = SmoothingMethod.LAPLACE,
    alpha: float = 1.0,
) -> ProbabilityNormalizer:
    """
    Factory function to create a ProbabilityNormalizer instance.

    Args:
        smoothing_method: Smoothing technique to use
        alpha: Smoothing parameter for Laplace smoothing

    Returns:
        Instance of ProbabilityNormalizer
    """
    return ProbabilityNormalizer(smoothing_method, alpha)


__all__ = [
    "create_probability_normalizer",
    "ProbabilityNormalizer",
    "SmoothingMethod",
    "PerplexityResult",
    "PredictionResult",
    "ProbabilityDistribution",
]
