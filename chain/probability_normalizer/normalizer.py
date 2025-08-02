
from collections import Counter
from typing import Dict, List, Optional, Union
import math
from chain.probability_normalizer.types import (
    SmoothingMethod,
    ProbabilityDistribution,
    PredictionResult,
    PerplexityResult
)
from chain.knowledge_base.types import (
    ContextPredictions,
    PredictionCandidate
)
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ProbabilityNormalizer:
    """
    Convert raw frequency counts into normalized probabilities.
    
    This class provides various smoothing techniques for handling unseen data
    in predictive text systems, including Laplace and Good-Turing smoothing.
    It's designed to work seamlessly with the tokenizer and knowledge base
    modules, supporting their output types like ContextPredictions and
    PredictionCandidate.
    
    Attributes:
        smoothing_method: The smoothing technique to use
        alpha: Smoothing parameter for Laplace smoothing
        
    Example:
        >>> normalizer = ProbabilityNormalizer(SmoothingMethod.LAPLACE, alpha=1.0)
        >>> counts = Counter({'the': 10, 'a': 5, 'an': 2})
        >>> result = normalizer.normalize_counts(counts)
        >>> print(result.probabilities)
        
        # Working with knowledge base predictions:
        >>> context_predictions = knowledge_base.get_predictions(("hello",), 2)
        >>> normalized = normalizer.normalize_context_predictions(context_predictions)
    """
    
    def __init__(
        self, 
        smoothing_method: Union[SmoothingMethod, str] = SmoothingMethod.LAPLACE,
        alpha: float = 1.0
    ) -> None:
        """
        Initialize probability normalizer.
        
        Args:
            smoothing_method: Smoothing technique to use
            alpha: Smoothing parameter (for Laplace smoothing), must be > 0
            
        Raises:
            ValueError: If alpha is not positive for Laplace smoothing
        """
        if isinstance(smoothing_method, str):
            smoothing_method = SmoothingMethod(smoothing_method)
            
        self.smoothing_method = smoothing_method
        self.alpha = alpha
        
        if self.smoothing_method == SmoothingMethod.LAPLACE and alpha <= 0:
            raise ValueError("Alpha must be positive for Laplace smoothing")
            
        logger.info(
            f"Initialized ProbabilityNormalizer with method={smoothing_method.value}, "
            f"alpha={alpha}"
        )
    
    def normalize_context_predictions(
        self,
        context_predictions: ContextPredictions,
        vocabulary_size: Optional[int] = None
    ) -> ContextPredictions:
        """
        Normalize probabilities in ContextPredictions from knowledge base.
        
        Args:
            context_predictions: ContextPredictions from knowledge base
            vocabulary_size: Size of vocabulary for smoothing (optional)
            
        Returns:
            ContextPredictions with normalized probabilities
        """
        if not context_predictions.candidates:
            logger.warning("No candidates found in context predictions")
            return context_predictions
        
        # Extract counts from candidates
        counts = Counter({
            candidate.token: candidate.count 
            for candidate in context_predictions.candidates
        })
        
        # Normalize using selected method
        distribution = self.normalize_counts(counts, vocabulary_size)
        
        # Create new candidates with normalized probabilities
        normalized_candidates = []
        for candidate in context_predictions.candidates:
            normalized_prob = distribution.probabilities.get(candidate.token, 0.0)
            normalized_candidates.append(
                PredictionCandidate(
                    token=candidate.token,
                    count=candidate.count,
                    probability=normalized_prob
                )
            )
        
        # Sort by probability (descending)
        normalized_candidates.sort(key=lambda x: x.probability, reverse=True)
        
        logger.debug(
            f"Normalized {len(normalized_candidates)} candidates for context "
            f"{context_predictions.context}"
        )
        
        return ContextPredictions(
            context=context_predictions.context,
            candidates=normalized_candidates,
            n_gram_size=context_predictions.n_gram_size,
            level=context_predictions.level
        )

    def normalize_counts(
        self, 
        counts: Counter[str],
        vocabulary_size: Optional[int] = None
    ) -> ProbabilityDistribution:
        """
        Convert frequency counts to probabilities.
        
        Args:
            counts: Counter with frequency data
            vocabulary_size: Size of vocabulary for smoothing (optional)
            
        Returns:
            ProbabilityDistribution containing normalized probabilities
            
        Raises:
            ValueError: If counts is empty
        """
        if not counts:
            logger.warning("Empty counts provided for normalization")
            raise ValueError("Cannot normalize empty counts")
        
        logger.debug(f"Normalizing {len(counts)} items with {self.smoothing_method.value}")
        
        total_count = sum(counts.values())
        
        if self.smoothing_method == SmoothingMethod.LAPLACE:
            probabilities = self._laplace_smoothing(counts, total_count, vocabulary_size)
        elif self.smoothing_method == SmoothingMethod.GOOD_TURING:
            probabilities = self._good_turing_smoothing(counts, total_count)
        else:
            probabilities = self._no_smoothing(counts, total_count)
        
        total_prob = sum(probabilities.values())
        return ProbabilityDistribution(probabilities=probabilities, total_probability=total_prob)
    
    def _no_smoothing(
        self, 
        counts: Counter[str],
        total_count: int
    ) -> Dict[str, float]:
        """
        Simple normalization without smoothing.
        
        Args:
            counts: Frequency counts
            total_count: Total number of occurrences
            
        Returns:
            Normalized probabilities
        """
        logger.debug("Applying no smoothing normalization")
        probabilities = {}
        for item, count in counts.items():
            probabilities[item] = count / total_count
        return probabilities
    
    def _laplace_smoothing(
        self,
        counts: Counter[str],
        total_count: int,
        vocabulary_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Apply Laplace (add-alpha) smoothing.
        
        Args:
            counts: Frequency counts
            total_count: Total number of occurrences
            vocabulary_size: Size of vocabulary
            
        Returns:
            Smoothed probabilities
        """
        if vocabulary_size is None:
            vocabulary_size = len(counts)
            logger.debug(f"Using vocabulary size from counts: {vocabulary_size}")
        
        # Adjusted total count includes smoothing
        adjusted_total = total_count + self.alpha * vocabulary_size
        
        logger.debug(
            f"Applying Laplace smoothing with alpha={self.alpha}, "
            f"vocab_size={vocabulary_size}"
        )
        
        probabilities = {}
        for item, count in counts.items():
            probabilities[item] = (count + self.alpha) / adjusted_total
        
        return probabilities
    
    def _good_turing_smoothing(
        self,
        counts: Counter[str],
        total_count: int
    ) -> Dict[str, float]:
        """
        Apply Good-Turing smoothing (simplified version).
        
        Args:
            counts: Frequency counts
            total_count: Total number of occurrences
            
        Returns:
            Smoothed probabilities
        """
        logger.debug("Applying Good-Turing smoothing")
        
        # Count frequency of frequencies
        freq_of_freq = Counter(counts.values())
        
        probabilities = {}
        for item, count in counts.items():
            if count == 1:
                # Items seen once get special treatment
                n1 = freq_of_freq.get(1, 0)
                adjusted_count = n1 / total_count if n1 > 0 else count
            else:
                # Use standard count for items seen multiple times
                adjusted_count = count
            
            probabilities[item] = adjusted_count / total_count
        
        return probabilities
    
    def combine_probabilities(
        self,
        prob_dists: List[ProbabilityDistribution],
        weights: Optional[List[float]] = None
    ) -> ProbabilityDistribution:
        """
        Combine multiple probability distributions with optional weighting.
        
        Args:
            prob_dists: List of probability distributions
            weights: Optional weights for each distribution
            
        Returns:
            Combined probability distribution
            
        Raises:
            ValueError: If weights don't match distributions or are invalid
        """
        if not prob_dists:
            raise ValueError("No probability distributions provided")
        
        if weights is None:
            weights = [1.0] * len(prob_dists)
        elif len(weights) != len(prob_dists):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of distributions ({len(prob_dists)})"
            )
        
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
            
        weights = [w / total_weight for w in weights]
        
        logger.debug(f"Combining {len(prob_dists)} distributions with weights {weights}")
        
        # Combine distributions
        combined = {}
        all_items = set()
        for dist in prob_dists:
            all_items.update(list(dist.probabilities.keys()))
        
        for item in all_items:
            combined_prob = 0.0
            for i, dist in enumerate(prob_dists):
                prob = dist.probabilities.get(item, 0.0)
                combined_prob += prob * weights[i]
            combined[item] = combined_prob
        
        total_prob = sum(combined.values())
        return ProbabilityDistribution(probabilities=combined, total_probability=total_prob)
    
    def candidates_to_distribution(
        self,
        candidates: List[PredictionCandidate]
    ) -> ProbabilityDistribution:
        """
        Convert list of PredictionCandidates to ProbabilityDistribution.
        
        Args:
            candidates: List of prediction candidates
            
        Returns:
            ProbabilityDistribution from candidates
            
        Raises:
            ValueError: If candidates list is empty
        """
        if not candidates:
            raise ValueError("Cannot create distribution from empty candidates")
        
        probabilities = {
            candidate.token: candidate.probability 
            for candidate in candidates
        }
        total_prob = sum(probabilities.values())
        
        return ProbabilityDistribution(
            probabilities=probabilities,
            total_probability=total_prob
        )

    def distribution_to_candidates(
        self,
        distribution: ProbabilityDistribution,
        counts: Optional[Counter[str]] = None
    ) -> List[PredictionCandidate]:
        """
        Convert ProbabilityDistribution to list of PredictionCandidates.
        
        Args:
            distribution: Probability distribution
            counts: Optional frequency counts (defaults to 1 for all)
            
        Returns:
            List of PredictionCandidates
        """
        if counts is None:
            counts = Counter({token: 1 for token in distribution.probabilities})
        
        candidates = []
        for token, probability in distribution.probabilities.items():
            count = counts.get(token, 1)
            candidates.append(
                PredictionCandidate(
                    token=token,
                    count=count,
                    probability=probability
                )
            )
        
        # Sort by probability (descending)
        candidates.sort(key=lambda x: x.probability, reverse=True)
        
        return candidates

    def get_top_candidates(
        self,
        candidates: List[PredictionCandidate],
        top_k: int = 5
    ) -> List[PredictionCandidate]:
        """
        Get top K candidates sorted by probability.
        
        Args:
            candidates: List of prediction candidates
            top_k: Number of top candidates to return
            
        Returns:
            List of top PredictionCandidates
            
        Raises:
            ValueError: If top_k is not positive
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Sort by probability (descending) then by count (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (x.probability, x.count),
            reverse=True
        )
        
        top_candidates = sorted_candidates[:top_k]
        logger.debug(f"Retrieved top {len(top_candidates)} candidates")
        
        return top_candidates

    def get_top_predictions(
        self,
        distribution: ProbabilityDistribution,
        top_k: int = 5
    ) -> PredictionResult:
        """
        Get top K predictions sorted by probability.
        
        Args:
            distribution: Probability distribution
            top_k: Number of top predictions to return
            
        Returns:
            PredictionResult with top predictions
            
        Raises:
            ValueError: If top_k is not positive
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
            
        sorted_items = sorted(
            distribution.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        predictions = sorted_items[:top_k]
        logger.debug(f"Retrieved top {len(predictions)} predictions")
        
        return PredictionResult(predictions=predictions, top_k=top_k)
    
    def calculate_candidate_perplexity(
        self,
        candidates: List[PredictionCandidate]
    ) -> PerplexityResult:
        """
        Calculate perplexity from a list of prediction candidates.
        
        Args:
            candidates: List of prediction candidates
            
        Returns:
            PerplexityResult with perplexity score
            
        Raises:
            ValueError: If candidates list is empty
        """
        if not candidates:
            raise ValueError("Cannot calculate perplexity for empty candidates")
        
        probabilities = [candidate.probability for candidate in candidates]
        return self.calculate_perplexity(probabilities)

    def calculate_perplexity(
        self,
        probabilities: List[float]
    ) -> PerplexityResult:
        """
        Calculate perplexity of a sequence of probabilities.
        
        Perplexity measures how well a probability model predicts a sample.
        Lower perplexity indicates better prediction.
        
        Args:
            probabilities: List of probabilities
            
        Returns:
            PerplexityResult with perplexity score
            
        Raises:
            ValueError: If probabilities list is empty
        """
        if not probabilities:
            raise ValueError("Cannot calculate perplexity for empty sequence")
        
        if any(p <= 0 for p in probabilities):
            logger.warning("Zero or negative probability detected, returning infinity")
            return PerplexityResult(
                perplexity=float('inf'),
                sequence_length=len(probabilities)
            )
        
        log_prob_sum = sum(math.log(p) for p in probabilities)
        avg_log_prob = log_prob_sum / len(probabilities)
        perplexity = math.exp(-avg_log_prob)
        
        logger.debug(f"Calculated perplexity: {perplexity:.4f}")
        
        return PerplexityResult(
            perplexity=perplexity,
            sequence_length=len(probabilities)
        )
    
    def calculate_candidate_entropy(
        self,
        candidates: List[PredictionCandidate]
    ) -> float:
        """
        Calculate entropy from a list of prediction candidates.
        
        Args:
            candidates: List of prediction candidates
            
        Returns:
            Entropy in bits
        """
        distribution = self.candidates_to_distribution(candidates)
        return self.calculate_entropy(distribution)

    def calculate_entropy(
        self,
        distribution: ProbabilityDistribution
    ) -> float:
        """
        Calculate entropy of a probability distribution.
        
        Entropy measures the average amount of information contained in each
        outcome. Higher entropy indicates more uncertainty.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Entropy in bits
        """
        entropy = 0.0
        for prob in distribution.probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        logger.debug(f"Calculated entropy: {entropy:.4f} bits")
        
        return entropy

    def normalize_multiple_contexts(
        self,
        context_predictions_list: List[ContextPredictions],
        vocabulary_size: Optional[int] = None
    ) -> List[ContextPredictions]:
        """
        Normalize probabilities for multiple context predictions.
        
        Args:
            context_predictions_list: List of ContextPredictions from knowledge base
            vocabulary_size: Size of vocabulary for smoothing (optional)
            
        Returns:
            List of ContextPredictions with normalized probabilities
        """
        normalized_list = []
        for context_predictions in context_predictions_list:
            normalized = self.normalize_context_predictions(
                context_predictions, vocabulary_size
            )
            normalized_list.append(normalized)
        
        logger.debug(f"Normalized {len(normalized_list)} context predictions")
        return normalized_list

    def combine_context_predictions(
        self,
        context_predictions_list: List[ContextPredictions],
        weights: Optional[List[float]] = None
    ) -> ContextPredictions:
        """
        Combine multiple context predictions into a single prediction.
        
        This method is useful for combining predictions from different n-gram sizes
        or different levels (word/char).
        
        Args:
            context_predictions_list: List of ContextPredictions to combine
            weights: Optional weights for each prediction context
            
        Returns:
            Combined ContextPredictions
            
        Raises:
            ValueError: If inputs are invalid or inconsistent
        """
        if not context_predictions_list:
            raise ValueError("No context predictions provided")
        
        # Verify all contexts are from the same level
        first_level = context_predictions_list[0].level
        if not all(cp.level == first_level for cp in context_predictions_list):
            raise ValueError("All context predictions must be from the same level")
        
        # Convert each to probability distributions
        distributions = []
        for cp in context_predictions_list:
            if cp.candidates:
                dist = self.candidates_to_distribution(cp.candidates)
                distributions.append(dist)
        
        if not distributions:
            # Return empty prediction if no valid candidates
            return ContextPredictions(
                context=context_predictions_list[0].context,
                candidates=[],
                n_gram_size=context_predictions_list[0].n_gram_size,
                level=first_level
            )
        
        # Combine distributions
        combined_dist = self.combine_probabilities(distributions, weights)
        
        # Convert back to candidates (using count=1 as default)
        combined_candidates = self.distribution_to_candidates(combined_dist)
        
        # Use the first context as the representative context
        representative_context = context_predictions_list[0]
        
        return ContextPredictions(
            context=representative_context.context,
            candidates=combined_candidates,
            n_gram_size=representative_context.n_gram_size,
            level=first_level
        )
