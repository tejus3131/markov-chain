from typing import Dict, List, Optional, Tuple
from chain.prediction_engine.types import (
    PredictionResult,
    Prediction,
    ContextSuggestion,
    ContextSuggestionsResult,
    TextGenerationConfig,
    PredictionLevel,
)
from chain.knowledge_base.knowledge_base import KnowledgeBase
from chain.knowledge_base.types import ContextPredictions, PredictionCandidate
from chain.probability_normalizer.normalizer import ProbabilityNormalizer
from chain.tokenizer.hybrid_tokenizer import Tokenizer
from chain.tokenizer.types import TokenizationResult
import logging
import random
import math

# Configure logging
logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Main prediction engine that uses knowledge base and probability normalizer
    to predict next words or characters using n-gram backoff strategy.

    Attributes:
        kb: Knowledge base instance containing n-gram statistics
        normalizer: Probability normalizer for converting counts to probabilities
        tokenizer: Tokenizer for processing input text
        ngram_sizes: List of n-gram sizes to use (largest first)
        backoff_weights: Weights for combining different n-gram predictions
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        probability_normalizer: ProbabilityNormalizer,
        tokenizer: Optional[Tokenizer] = None,
        ngram_sizes: Optional[List[int]] = None,
        backoff_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Initialize prediction engine.

        Args:
            knowledge_base: KnowledgeBase instance
            probability_normalizer: ProbabilityNormalizer instance
            tokenizer: Optional Tokenizer instance for text processing
            ngram_sizes: N-gram sizes in order of preference (largest first)
            backoff_weights: Custom backoff weights for each n-gram size

        Raises:
            ValueError: If ngram_sizes contains invalid values
        """
        self.kb = knowledge_base
        self.normalizer = probability_normalizer
        self.tokenizer = tokenizer or Tokenizer()

        # Set default n-gram sizes if not provided
        if ngram_sizes is None:
            ngram_sizes = [7, 5, 3, 2]

        # Validate n-gram sizes
        if not all(n >= 2 for n in ngram_sizes):
            raise ValueError("All n-gram sizes must be at least 2")

        self.ngram_sizes = sorted(ngram_sizes, reverse=True)

        # Set default backoff weights if not provided
        if backoff_weights is None:
            backoff_weights = {7: 1.0, 5: 0.8, 3: 0.6, 2: 0.4}
        self.backoff_weights = backoff_weights

        logger.info(
            f"PredictionEngine initialized with n-gram sizes: {self.ngram_sizes}"
        )

    def predict_from_text(
        self,
        text: str,
        level: PredictionLevel = PredictionLevel.WORD,
        top_k: int = 5,
        combine_ngrams: bool = True
    ) -> PredictionResult:
        """
        Predict next token from input text using tokenizer.
        
        Args:
            text: Input text to use as context
            level: Prediction level (word or character)
            top_k: Number of top predictions to return
            combine_ngrams: Whether to combine predictions from multiple n-grams
            
        Returns:
            PredictionResult containing predictions and metadata
        """
        if not text:
            logger.warning("Empty text provided for prediction")
            return PredictionResult(predictions=[], context_used=None, ngram_size=None)
        
        # Tokenize the input text
        tokenization_result = self.tokenizer.tokenize_all_levels(text)
        
        if level == PredictionLevel.WORD:
            context = tokenization_result.words
        else:
            context = tokenization_result.characters
        
        if level == PredictionLevel.WORD:
            return self.predict_next_word(context, top_k, combine_ngrams)
        else:
            return self.predict_next_character(context, top_k, combine_ngrams)

    def predict_next_word(
        self, context: List[str], top_k: int = 5, combine_ngrams: bool = True
    ) -> PredictionResult:
        """
        Predict next word given a context of previous words.

        Args:
            context: List of previous words as context
            top_k: Number of top predictions to return
            combine_ngrams: Whether to combine predictions from multiple n-grams

        Returns:
            PredictionResult containing predictions and metadata

        Example:
            >>> engine.predict_next_word(["the", "quick"], top_k=3)
            PredictionResult(predictions=[...], context_used=("the", "quick"), ngram_size=3)
        """
        logger.debug(
            f"Predicting next word for context: {context[-5:]}"
        )  # Log last 5 words

        if not context:
            logger.warning("Empty context provided for word prediction")
            return PredictionResult(predictions=[], context_used=None, ngram_size=None)

        # Convert to lowercase for consistency
        context = [word.lower() for word in context]

        if combine_ngrams:
            predictions = self._predict_with_combination(
                context, top_k, PredictionLevel.WORD
            )
        else:
            predictions = self._predict_with_backoff(
                context, top_k, PredictionLevel.WORD
            )

        return predictions

    def predict_next_character(
        self, context: List[str], top_k: int = 5, combine_ngrams: bool = True
    ) -> PredictionResult:
        """
        Predict next character given a context of previous characters.

        Args:
            context: List of previous characters as context
            top_k: Number of top predictions to return
            combine_ngrams: Whether to combine predictions from multiple n-grams

        Returns:
            PredictionResult containing predictions and metadata
        """
        logger.debug(f"Predicting next character for context: {''.join(context[-10:])}")

        if not context:
            logger.warning("Empty context provided for character prediction")
            return PredictionResult(predictions=[], context_used=None, ngram_size=None)

        if combine_ngrams:
            predictions = self._predict_with_combination(
                context, top_k, PredictionLevel.CHAR
            )
        else:
            predictions = self._predict_with_backoff(
                context, top_k, PredictionLevel.CHAR
            )

        return predictions

    def _predict_with_backoff(
        self, context: List[str], top_k: int, level: PredictionLevel
    ) -> PredictionResult:
        """
        Predict using n-gram backoff strategy (use longest available n-gram).

        Args:
            context: Context tokens
            top_k: Number of predictions
            level: 'word' or 'char' prediction level

        Returns:
            PredictionResult with predictions from the longest matching n-gram
        """
        for n in self.ngram_sizes:
            if len(context) >= n - 1:
                # Use last (n-1) tokens as context
                ngram_context = tuple(context[-(n - 1) :])

                # Get predictions using knowledge base
                context_predictions = self.kb.get_predictions(
                    ngram_context, 
                    n, 
                    level.value, 
                    top_k=top_k
                )

                if context_predictions.candidates:
                    logger.debug(f"Found {len(context_predictions.candidates)} predictions using {n}-gram")
                    
                    # Normalize probabilities
                    normalized_predictions = self.normalizer.normalize_context_predictions(
                        context_predictions
                    )
                    
                    # Convert to engine prediction format
                    predictions = [
                        Prediction(token=candidate.token, probability=candidate.probability)
                        for candidate in normalized_predictions.candidates[:top_k]
                    ]

                    return PredictionResult(
                        predictions=predictions,
                        context_used=ngram_context,
                        ngram_size=n,
                    )

        logger.warning("No predictions found using backoff strategy")
        return PredictionResult(predictions=[], context_used=None, ngram_size=None)

    def _predict_with_combination(
        self, context: List[str], top_k: int, level: PredictionLevel
    ) -> PredictionResult:
        """
        Predict by combining probabilities from multiple n-grams.

        Args:
            context: Context tokens
            top_k: Number of predictions
            level: 'word' or 'char' prediction level

        Returns:
            PredictionResult with combined predictions from multiple n-grams
        """
        context_predictions_list = []
        weights = []
        contexts_used = []

        for n in self.ngram_sizes:
            if len(context) >= n - 1:
                # Use last (n-1) tokens as context
                ngram_context = tuple(context[-(n - 1) :])

                # Get predictions using knowledge base
                context_predictions = self.kb.get_predictions(
                    ngram_context, 
                    n, 
                    level.value
                )

                if context_predictions.candidates:
                    context_predictions_list.append(context_predictions)
                    weights.append(self.backoff_weights.get(n, 0.1))
                    contexts_used.append((ngram_context, n))

        if not context_predictions_list:
            logger.warning("No predictions found for combination")
            return PredictionResult(predictions=[], context_used=None, ngram_size=None)

        logger.debug(f"Combining {len(context_predictions_list)} n-gram predictions")

        # Combine predictions using normalizer
        combined_predictions = self.normalizer.combine_context_predictions(
            context_predictions_list, weights
        )
        
        # Get top candidates and convert to engine format
        top_candidates = self.normalizer.get_top_candidates(
            combined_predictions.candidates, top_k
        )
        
        predictions = [
            Prediction(token=candidate.token, probability=candidate.probability)
            for candidate in top_candidates
        ]

        # Use the longest context as the primary context
        primary_context = contexts_used[0][0] if contexts_used else None

        return PredictionResult(
            predictions=predictions,
            context_used=primary_context,
            ngram_size=None,  # Multiple n-grams used
        )

    def complete_word(
        self, partial_word: str, top_k: int = 5, max_completion_length: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Complete a partially typed word using character-level prediction.

        Args:
            partial_word: Partially typed word
            top_k: Number of completions to return
            max_completion_length: Maximum length for word completion

        Returns:
            List of (completion, probability) tuples

        Example:
            >>> engine.complete_word("hel", top_k=3)
            [("hello", 0.8), ("help", 0.6), ("helicopter", 0.4)]
        """
        if not partial_word:
            logger.warning("Empty partial word provided")
            return []

        logger.debug(f"Completing word: '{partial_word}'")

        partial_word = partial_word.lower()
        completions: Dict[str, float] = {}

        # Word boundary characters
        word_boundaries = {" ", ".", ",", "!", "?", ";", ":", "\n"}

        # Use tokenizer to get character tokens
        char_tokens = self.tokenizer.tokenize_characters(partial_word)

        # Generate multiple possible completions
        current_tokens = char_tokens.copy()

        for _ in range(max_completion_length):
            result = self.predict_next_character(current_tokens, top_k=3)

            if not result.predictions:
                break

            # Try each predicted character
            for prediction in result.predictions:
                char = prediction.token
                prob = prediction.probability
                potential_completion = ''.join(current_tokens + [char])

                # Stop if we hit a word boundary
                if char in word_boundaries:
                    word_completion = ''.join(current_tokens)
                    if word_completion not in completions:
                        completions[word_completion] = prob
                    break

                # Continue building the word
                if len(potential_completion) <= len(partial_word) + 15:
                    if potential_completion not in completions:
                        completions[potential_completion] = prob

        # Sort completions by probability
        sorted_completions = sorted(
            completions.items(), key=lambda x: x[1], reverse=True
        )

        logger.info(
            f"Generated {len(sorted_completions)} completions for '{partial_word}'"
        )
        return sorted_completions[:top_k]

    def generate_text(self, config: TextGenerationConfig) -> str:
        """
        Generate text starting from a seed using the Markov chain.

        Args:
            config: Text generation configuration

        Returns:
            Generated text string

        Example:
            >>> config = TextGenerationConfig(seed=["Once", "upon"], max_length=20)
            >>> engine.generate_text(config)
            "Once upon a time there was a..."
        """
        logger.info(f"Generating text with seed: {config.seed[:3]}...")

        # config.seed is already a List[str] according to TextGenerationConfig
        generated = config.seed.copy()
        sentence_endings = {".", "!", "?"}

        for i in range(config.max_length):
            # Get predictions for current context
            context = generated[-10:]  # Use last 10 words as context
            result = self.predict_next_word(context, top_k=10)

            if not result.predictions:
                logger.debug("No more predictions available, stopping generation")
                break

            # Convert to format expected by temperature scaling
            predictions = [
                (pred.token, pred.probability) for pred in result.predictions
            ]

            # Apply temperature scaling
            if config.temperature != 1.0:
                predictions = self._apply_temperature(predictions, config.temperature)

            # Sample from predictions
            words, probs = zip(*predictions)
            next_word = random.choices(words, weights=probs)[0]

            generated.append(next_word)

            # Stop at sentence endings if configured
            if config.stop_on_punctuation and next_word in sentence_endings:
                logger.debug(f"Stopping at sentence ending: '{next_word}'")
                break

        generated_text = " ".join(generated)
        logger.info(f"Generated {len(generated)} words")
        return generated_text

    def generate_from_text(
        self, 
        seed_text: str, 
        max_length: int = 50,
        temperature: float = 1.0,
        stop_on_punctuation: bool = True
    ) -> str:
        """
        Generate text starting from seed text (convenience method).
        
        Args:
            seed_text: Input text to use as seed
            max_length: Maximum number of words to generate
            temperature: Controls randomness
            stop_on_punctuation: Stop generation on sentence endings
            
        Returns:
            Generated text string
        """
        # Tokenize the seed text to get word tokens
        tokenization_result = self.tokenizer.tokenize_all_levels(seed_text)
        seed_words = tokenization_result.words
        
        if not seed_words:
            logger.warning("No words found in seed text")
            return seed_text
        
        # Create configuration
        config = TextGenerationConfig(
            seed=seed_words,
            max_length=max_length,
            temperature=temperature,
            stop_on_punctuation=stop_on_punctuation
        )
        
        return self.generate_text(config)

    def _apply_temperature(
        self, predictions: List[Tuple[str, float]], temperature: float
    ) -> List[Tuple[str, float]]:
        """
        Apply temperature scaling to predictions.

        Args:
            predictions: List of (word, probability) tuples
            temperature: Temperature parameter

        Returns:
            Temperature-scaled predictions
        """
        if temperature <= 0:
            temperature = 1e-10

        # Apply temperature scaling
        scaled_probs = []
        for word, prob in predictions:
            if prob > 0:
                scaled_prob = math.exp(math.log(prob) / temperature)
                scaled_probs.append((word, scaled_prob))

        # Renormalize
        total_prob = sum(prob for _, prob in scaled_probs)
        if total_prob > 0:
            scaled_probs = [(word, prob / total_prob) for word, prob in scaled_probs]

        return scaled_probs

    def get_context_suggestions(
        self, context: List[str], level: PredictionLevel = PredictionLevel.WORD
    ) -> ContextSuggestionsResult:
        """
        Get detailed suggestions for debugging and analysis.

        Args:
            context: Input context
            level: 'word' or 'char' prediction level

        Returns:
            ContextSuggestionsResult with suggestions from each n-gram level

        Example:
            >>> result = engine.get_context_suggestions(["the", "quick"], PredictionLevel.WORD)
            >>> print(result.suggestions["3-gram"].predictions)
        """
        logger.debug(f"Getting context suggestions for level: {level}")

        suggestions = {}

        for n in self.ngram_sizes:
            if len(context) >= n - 1:
                ngram_context = tuple(context[-(n - 1) :])

                # Get predictions using knowledge base
                context_predictions = self.kb.get_predictions(
                    ngram_context, 
                    n, 
                    level.value, 
                    top_k=5
                )

                if context_predictions.candidates:
                    # Normalize probabilities
                    normalized_predictions = self.normalizer.normalize_context_predictions(
                        context_predictions
                    )
                    
                    # Convert to engine prediction format
                    predictions = [
                        Prediction(token=candidate.token, probability=candidate.probability)
                        for candidate in normalized_predictions.candidates[:5]
                    ]

                    # Calculate total count from candidates
                    total_count = sum(candidate.count for candidate in context_predictions.candidates)

                    suggestion = ContextSuggestion(
                        context=ngram_context,
                        predictions=predictions,
                        total_count=total_count,
                        ngram_size=n,
                    )

                    suggestions[f"{n}-gram"] = suggestion

        return ContextSuggestionsResult(suggestions=suggestions, level=level)

    def train_from_text(self, text: str, ngram_sizes: Optional[List[int]] = None) -> None:
        """
        Train the knowledge base from raw text using the tokenizer.
        
        Args:
            text: Raw text to train on
            ngram_sizes: N-gram sizes to use for tokenization
        """
        if not text:
            logger.warning("Empty text provided for training")
            return
        
        # Use engine's ngram_sizes if not specified
        if ngram_sizes is None:
            ngram_sizes = self.ngram_sizes
        
        # Tokenize the text
        tokenization_result = self.tokenizer.tokenize_all_levels(text, ngram_sizes)
        
        # Add to knowledge base
        self.kb.add_training_data(tokenization_result)
        
        logger.info(f"Trained on text of length {len(text)}")

    def train_from_texts(self, texts: List[str], ngram_sizes: Optional[List[int]] = None) -> None:
        """
        Train the knowledge base from multiple texts using batch tokenization.
        
        Args:
            texts: List of raw texts to train on
            ngram_sizes: N-gram sizes to use for tokenization
        """
        if not texts:
            logger.warning("Empty texts list provided for training")
            return
        
        # Use engine's ngram_sizes if not specified
        if ngram_sizes is None:
            ngram_sizes = self.ngram_sizes
        
        # Batch tokenize the texts
        batch_result = self.tokenizer.tokenize_batch(texts, ngram_sizes)
        
        # Add to knowledge base
        self.kb.add_batch_tokenization_result(batch_result)
        
        logger.info(f"Trained on {len(texts)} texts")

    def get_prediction_confidence(
        self, 
        context: List[str], 
        target: str, 
        level: PredictionLevel = PredictionLevel.WORD
    ) -> float:
        """
        Get confidence score for a specific prediction.
        
        Args:
            context: Context tokens
            target: Target token to evaluate
            level: Prediction level
            
        Returns:
            Confidence score (probability) for the target token
        """
        result = self.predict_next_word(context) if level == PredictionLevel.WORD else self.predict_next_character(context)
        
        for prediction in result.predictions:
            if prediction.token == target:
                return prediction.probability
        
        return 0.0

    def evaluate_perplexity(
        self, 
        test_texts: List[str], 
        level: PredictionLevel = PredictionLevel.WORD
    ) -> float:
        """
        Evaluate model perplexity on test texts.
        
        Args:
            test_texts: List of test texts
            level: Prediction level
            
        Returns:
            Average perplexity across all test texts
        """
        perplexities = []
        
        for text in test_texts:
            tokenization_result = self.tokenizer.tokenize_all_levels(text)
            tokens = tokenization_result.words if level == PredictionLevel.WORD else tokenization_result.characters
            
            if len(tokens) < 2:
                continue
            
            probabilities = []
            for i in range(1, len(tokens)):
                context = tokens[:i]
                target = tokens[i]
                prob = self.get_prediction_confidence(context, target, level)
                if prob > 0:
                    probabilities.append(prob)
            
            if probabilities:
                perplexity_result = self.normalizer.calculate_perplexity(probabilities)
                perplexities.append(perplexity_result.perplexity)
        
        return sum(perplexities) / len(perplexities) if perplexities else float('inf')
