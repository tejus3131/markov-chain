
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
import pickle
import logging

from chain.knowledge_base.types import (
    KnowledgeBaseStatistics,
    NGramStatistics,
    PredictionCandidate,
    ContextPredictions
)

from chain.tokenizer.types import TokenizationResult, BatchTokenizationResult


# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Stores n-gram frequency data and provides methods for updating
    and querying the knowledge base with thread-safe operations.
    """
    
    def __init__(self, ngram_sizes: List[int]):
        """
        Initialize knowledge base with specified n-gram sizes.
        
        Args:
            ngram_sizes: List of n-gram sizes to track
            
        Raises:
            ValueError: If ngram_sizes is empty or contains invalid values
        """
        if not ngram_sizes:
            raise ValueError("ngram_sizes cannot be empty")
        
        if not all(isinstance(n, int) and n > 1 for n in ngram_sizes):
            raise ValueError("All n-gram sizes must be integers greater than 1")
        
        self.ngram_sizes = sorted(ngram_sizes)
        
        # Store word-level n-grams
        self.word_ngrams: Dict[int, defaultdict] = {}
        for n in self.ngram_sizes:
            self.word_ngrams[n] = defaultdict(Counter)
        
        # Store character-level n-grams
        self.char_ngrams: Dict[int, defaultdict] = {}
        for n in self.ngram_sizes:
            self.char_ngrams[n] = defaultdict(Counter)
        
        # Statistics
        self.total_word_ngrams: defaultdict = defaultdict(int)
        self.total_char_ngrams: defaultdict = defaultdict(int)
        self.training_sentences: int = 0
        
        logger.info(f"Initialized KnowledgeBase with n-gram sizes: {self.ngram_sizes}")
    
    def update_word_ngrams(self, tokens: List[str]) -> None:
        """
        Update word-level n-gram frequencies from token list.
        
        Args:
            tokens: List of word tokens
            
        Raises:
            ValueError: If tokens is None or empty
        """
        if not tokens:
            raise ValueError("Tokens list cannot be empty")
        
        self._update_ngrams(tokens, self.word_ngrams, self.total_word_ngrams, "word")
    
    def update_char_ngrams(self, tokens: List[str]) -> None:
        """
        Update character-level n-gram frequencies from token list.
        
        Args:
            tokens: List of character tokens
            
        Raises:
            ValueError: If tokens is None or empty
        """
        if not tokens:
            raise ValueError("Tokens list cannot be empty")
        
        self._update_ngrams(tokens, self.char_ngrams, self.total_char_ngrams, "char")
    
    def _update_ngrams(
        self,
        tokens: List[str],
        ngram_storage: Dict[int, defaultdict],
        total_counter: defaultdict,
        level: str
    ) -> None:
        """
        Internal method to update n-gram frequencies.
        
        Args:
            tokens: List of tokens
            ngram_storage: Storage dictionary for n-grams
            total_counter: Counter for total n-grams
            level: Type of n-grams (word/char)
        """
        for n in self.ngram_sizes:
            if len(tokens) >= n:
                # Extract context-target pairs
                for i in range(len(tokens) - n + 1):
                    context = tuple(tokens[i:i + n - 1])
                    target = tokens[i + n - 1]
                    
                    # Update frequency
                    ngram_storage[n][context][target] += 1
                    total_counter[n] += 1
                
                logger.debug(
                    f"Updated {len(tokens) - n + 1} {level} {n}-grams"
                )
    
    def add_training_data(self, tokenized_data: TokenizationResult) -> None:
        """
        Add new training data to the knowledge base.
        
        Args:
            tokenized_data: TokenizationResult from the tokenizer module
            
        Raises:
            ValueError: If tokenized_data has invalid format
        """
        if not isinstance(tokenized_data, TokenizationResult):
            raise ValueError("tokenized_data must be a TokenizationResult")
        
        updated = False
        
        # Update word n-grams
        if tokenized_data.words:
            self.update_word_ngrams(tokenized_data.words)
            updated = True
            logger.debug(f"Added {len(tokenized_data.words)} word tokens")
        
        # Update character n-grams
        if tokenized_data.characters:
            self.update_char_ngrams(tokenized_data.characters)
            updated = True
            logger.debug(f"Added {len(tokenized_data.characters)} character tokens")
        
        if updated:
            self.training_sentences += 1
            logger.info(f"Training sentences count: {self.training_sentences}")
        else:
            logger.warning("No valid token data found in tokenized_data")
    
    def add_tokenization_result(self, result: TokenizationResult) -> None:
        """
        Add training data from a TokenizationResult object.
        
        Args:
            result: TokenizationResult from the tokenizer module
        """
        self.add_training_data(result)
    
    def add_batch_tokenization_result(self, batch_result: BatchTokenizationResult) -> None:
        """
        Add training data from a BatchTokenizationResult object.
        
        Args:
            batch_result: BatchTokenizationResult from the tokenizer module
        """
        if not isinstance(batch_result, BatchTokenizationResult):
            raise ValueError("batch_result must be a BatchTokenizationResult")
        
        for result in batch_result.results:
            self.add_training_data(result)
        
        if batch_result.failed_indices:
            logger.warning(f"Skipped {len(batch_result.failed_indices)} failed tokenizations")
    
    def get_word_context_counts(
        self, 
        context: Tuple[str, ...], 
        n: int
    ) -> Counter:
        """
        Get frequency counts for a word context.
        
        Args:
            context: Context tuple (n-1 tokens)
            n: N-gram size
            
        Returns:
            Counter with target word frequencies
            
        Raises:
            ValueError: If n is not in configured n-gram sizes
        """
        if n not in self.ngram_sizes:
            raise ValueError(f"N-gram size {n} not configured")
        
        return Counter(self.word_ngrams[n].get(context, Counter()))
    
    def get_char_context_counts(
        self, 
        context: Tuple[str, ...], 
        n: int
    ) -> Counter:
        """
        Get frequency counts for a character context.
        
        Args:
            context: Context tuple (n-1 characters)
            n: N-gram size
            
        Returns:
            Counter with target character frequencies
            
        Raises:
            ValueError: If n is not in configured n-gram sizes
        """
        if n not in self.ngram_sizes:
            raise ValueError(f"N-gram size {n} not configured")
        
        return Counter(self.char_ngrams[n].get(context, Counter()))
    
    def get_predictions(
        self,
        context: Tuple[str, ...],
        n: int,
        level: str = 'word',
        top_k: Optional[int] = None,
        min_count: int = 1
    ) -> ContextPredictions:
        """
        Get predictions for a given context.
        
        Args:
            context: Context tuple
            n: N-gram size
            level: 'word' or 'char'
            top_k: Return only top k predictions (None for all)
            min_count: Minimum count threshold
            
        Returns:
            ContextPredictions with candidates
            
        Raises:
            ValueError: If parameters are invalid
        """
        if level not in ['word', 'char']:
            raise ValueError("Level must be 'word' or 'char'")
        
        if n not in self.ngram_sizes:
            raise ValueError(f"N-gram size {n} not configured")
        
        # Get counts
        if level == 'word':
            counts = self.get_word_context_counts(context, n)
        else:
            counts = self.get_char_context_counts(context, n)
        
        # Filter by minimum count
        filtered_counts = {
            token: count for token, count in counts.items() 
            if count >= min_count
        }
        
        # Calculate total for probabilities
        total = sum(filtered_counts.values())
        
        # Create candidates
        candidates = []
        for token, count in filtered_counts.items():
            probability = count / total if total > 0 else 0.0
            candidates.append(
                PredictionCandidate(
                    token=token,
                    count=count,
                    probability=probability
                )
            )
        
        # Sort by count (descending)
        candidates.sort(key=lambda x: x.count, reverse=True)
        
        # Apply top_k if specified
        if top_k is not None and top_k > 0:
            candidates = candidates[:top_k]
        
        return ContextPredictions(
            context=context,
            candidates=candidates,
            n_gram_size=n,
            level=level
        )
    
    def get_available_contexts(
        self, 
        level: str = 'word',
        n: Optional[int] = None
    ) -> Dict[int, List[Tuple[str, ...]]]:
        """
        Get all available contexts for debugging/analysis.
        
        Args:
            level: 'word' or 'char'
            n: Specific n-gram size (None for all)
            
        Returns:
            Dictionary mapping n-gram sizes to lists of contexts
            
        Raises:
            ValueError: If level is invalid
        """
        if level not in ['word', 'char']:
            raise ValueError("Level must be 'word' or 'char'")
        
        contexts = {}
        
        ngrams = self.word_ngrams if level == 'word' else self.char_ngrams
        sizes = [n] if n is not None else self.ngram_sizes
        
        for size in sizes:
            if size in ngrams:
                contexts[size] = list(ngrams[size].keys())
                logger.debug(
                    f"Found {len(contexts[size])} {level} contexts for {size}-grams"
                )
        
        return contexts
    
    def get_statistics(self) -> KnowledgeBaseStatistics:
        """
        Get comprehensive knowledge base statistics.
        
        Returns:
            KnowledgeBaseStatistics with detailed information
        """
        word_stats = {}
        char_stats = {}
        
        for n in self.ngram_sizes:
            # Word statistics
            word_stats[n] = NGramStatistics(
                total_count=self.total_word_ngrams.get(n, 0),
                unique_contexts=len(self.word_ngrams[n]),
                n_gram_size=n
            )
            
            # Character statistics
            char_stats[n] = NGramStatistics(
                total_count=self.total_char_ngrams.get(n, 0),
                unique_contexts=len(self.char_ngrams[n]),
                n_gram_size=n
            )
        
        total_word_contexts = sum(s.unique_contexts for s in word_stats.values())
        total_char_contexts = sum(s.unique_contexts for s in char_stats.values())
        
        stats = KnowledgeBaseStatistics(
            training_sentences=self.training_sentences,
            word_ngram_stats=word_stats,
            char_ngram_stats=char_stats,
            total_unique_word_contexts=total_word_contexts,
            total_unique_char_contexts=total_char_contexts
        )
        
        logger.info(f"Generated statistics: {self.training_sentences} sentences processed")
        return stats
    
    def _prepare_save_data(self) -> Dict[str, Any]:
        """
        Prepare data for serialization.
        
        Returns:
            Dictionary with all data to save
        """
        return {
            'ngram_sizes': self.ngram_sizes,
            'word_ngrams': {
                n: dict(contexts) for n, contexts in self.word_ngrams.items()
            },
            'char_ngrams': {
                n: dict(contexts) for n, contexts in self.char_ngrams.items()
            },
            'total_word_ngrams': dict(self.total_word_ngrams),
            'total_char_ngrams': dict(self.total_char_ngrams),
            'training_sentences': self.training_sentences
        }
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Save knowledge base to file.
        
        Args:
            filepath: Path to save file
            
        Raises:
            IOError: If file cannot be written
        """
        filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = self._prepare_save_data()
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved knowledge base to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {str(e)}")
            raise IOError(f"Could not save knowledge base: {str(e)}") from e
    
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """
        Load knowledge base from file.
        
        Args:
            filepath: Path to load file
            
        Raises:
            IOError: If file cannot be read
            ValueError: If file contains invalid data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise IOError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self._restore_from_data(data)
            logger.info(f"Loaded knowledge base from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            raise IOError(f"Could not load knowledge base: {str(e)}") from e
    
    def _restore_from_data(self, data: Dict[str, Any]) -> None:
        """
        Restore knowledge base state from loaded data.
        
        Args:
            data: Dictionary with saved data
            
        Raises:
            ValueError: If data is invalid or corrupted
        """
        required_keys = {
            'ngram_sizes', 'word_ngrams', 'char_ngrams',
            'total_word_ngrams', 'total_char_ngrams', 'training_sentences'
        }
        
        if not all(key in data for key in required_keys):
            raise ValueError("Invalid save file: missing required keys")
        
        self.ngram_sizes = data['ngram_sizes']
        self.word_ngrams = defaultdict(lambda: defaultdict(Counter))
        self.char_ngrams = defaultdict(lambda: defaultdict(Counter))
        
        # Reconstruct word n-grams
        for n, contexts in data['word_ngrams'].items():
            n = int(n)  # Ensure n is int
            for context, counter in contexts.items():
                self.word_ngrams[n][context] = Counter(counter)
        
        # Reconstruct character n-grams
        for n, contexts in data['char_ngrams'].items():
            n = int(n)  # Ensure n is int
            for context, counter in contexts.items():
                self.char_ngrams[n][context] = Counter(counter)
        
        self.total_word_ngrams = defaultdict(int, data['total_word_ngrams'])
        self.total_char_ngrams = defaultdict(int, data['total_char_ngrams'])
        self.training_sentences = data['training_sentences']
    
    def merge(self, other: 'KnowledgeBase') -> None:
        """
        Merge another knowledge base into this one.
        
        Args:
            other: Another KnowledgeBase instance
            
        Raises:
            ValueError: If n-gram sizes don't match
        """
        if set(self.ngram_sizes) != set(other.ngram_sizes):
            raise ValueError("Cannot merge: n-gram sizes don't match")
        
        # Merge word n-grams
        self._merge_ngrams(self.word_ngrams, other.word_ngrams, "word")
        
        # Merge character n-grams
        self._merge_ngrams(self.char_ngrams, other.char_ngrams, "char")
        
        # Update statistics
        for n in self.ngram_sizes:
            self.total_word_ngrams[n] += other.total_word_ngrams.get(n, 0)
            self.total_char_ngrams[n] += other.total_char_ngrams.get(n, 0)
        
        self.training_sentences += other.training_sentences
        
        logger.info(f"Merged knowledge bases: added {other.training_sentences} sentences")
    
    def _merge_ngrams(
        self,
        target: Dict[int, defaultdict],
        source: Dict[int, defaultdict],
        level: str
    ) -> None:
        """
        Merge n-gram data from source into target.
        
        Args:
            target: Target n-gram storage
            source: Source n-gram storage
            level: Type of n-grams (for logging)
        """
        for n in self.ngram_sizes:
            if n in source:
                for context, counter in source[n].items():
                    target[n][context].update(counter)
                logger.debug(f"Merged {len(source[n])} {level} {n}-gram contexts")
    
    def clear(self) -> None:
        """Clear all data from the knowledge base."""
        for n in self.ngram_sizes:
            self.word_ngrams[n].clear()
            self.char_ngrams[n].clear()
        
        self.total_word_ngrams.clear()
        self.total_char_ngrams.clear()
        self.training_sentences = 0
        
        logger.info("Cleared all data from knowledge base")
