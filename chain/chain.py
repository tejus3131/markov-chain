"""
Main Markov Chain Predictive Text System
Integrates all components into a complete working system
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Import all components with proper types
from chain.tokenizer.hybrid_tokenizer import Tokenizer
from chain.knowledge_base.knowledge_base import KnowledgeBase
from chain.probability_normalizer.normalizer import ProbabilityNormalizer, SmoothingMethod
from chain.prediction_engine.engine import PredictionEngine
from chain.prediction_engine.types import PredictionLevel, TextGenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarkovChainTextPredictor:
    """
    Complete Markov Chain-based predictive text system that integrates
    all components for a full-featured text prediction experience.
    """
    
    def __init__(self, 
                 ngram_sizes: List[int] = [2, 3, 5, 7], 
                 smoothing_method: str = 'laplace',
                 smoothing_alpha: float = 0.1):
        """
        Initialize the complete Markov Chain system.
        
        Args:
            ngram_sizes: List of n-gram sizes to use
            smoothing_method: Smoothing technique ('laplace', 'good_turing', 'none')
            smoothing_alpha: Smoothing parameter
        """
        print("Initializing Markov Chain Text Predictor...")

        # Initialize core components
        self.tokenizer = Tokenizer()
        self.knowledge_base = KnowledgeBase(ngram_sizes)
        
        # Convert string to SmoothingMethod enum
        try:
            smoothing_enum = SmoothingMethod(smoothing_method)
        except ValueError:
            logger.warning(f"Unknown smoothing method '{smoothing_method}', using 'laplace'")
            smoothing_enum = SmoothingMethod.LAPLACE
            
        self.probability_normalizer = ProbabilityNormalizer(
            smoothing_method=smoothing_enum, 
            alpha=smoothing_alpha
        )
        
        self.prediction_engine = PredictionEngine(
            self.knowledge_base, 
            self.probability_normalizer,
            self.tokenizer,
            ngram_sizes
        )

        self.ngram_sizes = ngram_sizes
        self.is_initialized = True
        print("✓ System initialized successfully!")
    
    def train(self, texts: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Train the system on a list of texts.
        
        Args:
            texts: List of training texts
            show_progress: Whether to show training progress
            
        Returns:
            Training statistics and results
        """
        if not texts:
            logger.warning("No texts provided for training")
            return {"success": False, "message": "No texts provided"}
        
        logger.info(f"Training on {len(texts)} texts...")
        
        try:
            # Use the prediction engine's batch training method
            self.prediction_engine.train_from_texts(texts, self.ngram_sizes)
            
            # Get training statistics
            stats = self.knowledge_base.get_statistics()
            
            result = {
                "success": True,
                "texts_processed": len(texts),
                "training_sentences": stats.training_sentences,
                "total_word_contexts": stats.total_unique_word_contexts,
                "total_char_contexts": stats.total_unique_char_contexts,
                "ngram_stats": {
                    "word": {str(k): {"total": v.total_count, "unique": v.unique_contexts} 
                            for k, v in stats.word_ngram_stats.items()},
                    "char": {str(k): {"total": v.total_count, "unique": v.unique_contexts} 
                            for k, v in stats.char_ngram_stats.items()}
                }
            }
            
            if show_progress:
                print(f"✓ Training completed! Processed {result['training_sentences']} sentences")
                print(f"  - Word contexts: {result['total_word_contexts']}")
                print(f"  - Character contexts: {result['total_char_contexts']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def predict_next_word(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Predict the next word based on input text.
        
        Args:
            text: Input text context
            top_k: Number of predictions to return
            
        Returns:
            Word predictions with probabilities
        """
        try:
            result = self.prediction_engine.predict_from_text(
                text, PredictionLevel.WORD, top_k=top_k
            )
            
            predictions = [
                {"word": pred.token, "probability": pred.probability}
                for pred in result.predictions
            ]
            
            return {
                "success": True,
                "predictions": predictions,
                "context_used": list(result.context_used) if result.context_used else None,
                "ngram_size": result.ngram_size
            }
            
        except Exception as e:
            logger.error(f"Word prediction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def predict_next_character(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Predict the next character based on input text.
        
        Args:
            text: Input text context
            top_k: Number of predictions to return
            
        Returns:
            Character predictions with probabilities
        """
        try:
            result = self.prediction_engine.predict_from_text(
                text, PredictionLevel.CHAR, top_k=top_k
            )
            
            predictions = [
                {"character": pred.token, "probability": pred.probability}
                for pred in result.predictions
            ]
            
            return {
                "success": True,
                "predictions": predictions,
                "context_used": list(result.context_used) if result.context_used else None,
                "ngram_size": result.ngram_size
            }
            
        except Exception as e:
            logger.error(f"Character prediction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def complete_word(self, partial_word: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete a partially typed word.
        
        Args:
            partial_word: Partial word to complete
            top_k: Number of completions to return
            
        Returns:
            Word completions with probabilities
        """
        try:
            completions = self.prediction_engine.complete_word(partial_word, top_k)
            
            return {
                "success": True,
                "partial_word": partial_word,
                "completions": [
                    {"word": word, "probability": prob}
                    for word, prob in completions
                ]
            }
            
        except Exception as e:
            logger.error(f"Word completion failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_text(self, 
                     seed_text: str, 
                     max_length: int = 20,
                     temperature: float = 1.0,
                     stop_on_punctuation: bool = True) -> Dict[str, Any]:
        """
        Generate text continuation from seed text.
        
        Args:
            seed_text: Starting text
            max_length: Maximum words to generate
            temperature: Randomness control (higher = more random)
            stop_on_punctuation: Stop at sentence endings
            
        Returns:
            Generated text and metadata
        """
        try:
            generated = self.prediction_engine.generate_from_text(
                seed_text, max_length, temperature, stop_on_punctuation
            )
            
            return {
                "success": True,
                "seed_text": seed_text,
                "generated_text": generated,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "stop_on_punctuation": stop_on_punctuation
                }
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a single text to return all predictions and statistics.
        
        Args:
            text: Input text to process
            
        Returns:
            Comprehensive analysis of the text
        """
        try:
            # Tokenize the text
            tokenization_result = self.tokenizer.tokenize_all_levels(text)
            
            # Get word and character predictions
            word_predictions = self.predict_next_word(text, top_k=5)
            char_predictions = self.predict_next_character(text, top_k=5)

            # For word completions, try to extract the last partial word
            words = tokenization_result.words
            partial_word = ""
            if text and not text.endswith(' '):
                # If text doesn't end with space, last word might be partial
                if words:
                    partial_word = words[-1]
                else:
                    # If no words detected, the entire text might be a partial word
                    partial_word = text.strip()
            
            word_completions = self.complete_word(partial_word, top_k=5)
            text_completions = self.generate_text(text, max_length=10)
            
            # Get context suggestions
            if words:
                word_suggestions = self.prediction_engine.get_context_suggestions(
                    words[-3:], PredictionLevel.WORD  # Use last 3 words for context
                )
            else:
                word_suggestions = None
            
            return {
                "success": True,
                "text": text,
                "tokenization": {
                    "words": tokenization_result.words,
                    "characters": tokenization_result.characters,
                    "word_count": len(tokenization_result.words),
                    "char_count": len(tokenization_result.characters)
                },
                "word_predictions": word_predictions,
                "character_predictions": char_predictions,
                "word_completions": word_completions,
                "text_completions": text_completions,
                "suggestions": {
                    str(ngram): {
                        "context": list(suggestion.context),
                        "predictions": [
                            {"token": pred.token, "probability": pred.probability}
                            for pred in suggestion.predictions
                        ],
                        "total_count": suggestion.total_count
                    }
                    for ngram, suggestion in (word_suggestions.suggestions.items() if word_suggestions else {})
                }
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def add_text(self, text: str) -> bool:
        """
        Add new text to the knowledge base and retrain the model.
        
        Args:
            text: New text to learn from
            
        Returns:
            Success status
        """
        try:
            self.prediction_engine.train_from_text(text)
            logger.info(f"Successfully added text of length {len(text)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add text: {str(e)}")
            return False
    
    def save_model(self, filepath: Optional[str] = None) -> bool:
        """
        Save the current model state to a file.
        
        Args:
            filepath: Path to save the model (default: auto-generated)
            
        Returns:
            Success status
        """
        try:
            if filepath is None:
                filepath = "markov_model.pkl"
            
            # Save the knowledge base (main state)
            self.knowledge_base.save_to_file(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
        
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a model state from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Success status
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Model file not found: {filepath}")
                return False
            
            # Load the knowledge base
            self.knowledge_base.load_from_file(filepath)
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current model statistics.
        
        Returns:
            Model statistics and information
        """
        try:
            stats = self.knowledge_base.get_statistics()
            
            return {
                "training_sentences": stats.training_sentences,
                "total_word_contexts": stats.total_unique_word_contexts,
                "total_char_contexts": stats.total_unique_char_contexts,
                "ngram_sizes": self.ngram_sizes,
                "word_ngram_stats": {
                    str(k): {"total_count": v.total_count, "unique_contexts": v.unique_contexts}
                    for k, v in stats.word_ngram_stats.items()
                },
                "char_ngram_stats": {
                    str(k): {"total_count": v.total_count, "unique_contexts": v.unique_contexts}
                    for k, v in stats.char_ngram_stats.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}


# Simple usage example and demo
def demo():
    """
    Demonstration of the Markov Chain Text Predictor.
    """
    print("🚀 Markov Chain Text Predictor Demo")
    print("=" * 40)
    
    # Initialize the predictor
    predictor = MarkovChainTextPredictor(
        ngram_sizes=[2, 3, 4],
        smoothing_method='laplace',
        smoothing_alpha=0.1
    )
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog jumps over the lazy fox.",
        "The lazy dog sleeps under the warm sun.",
        "Brown foxes are quick and clever animals.",
        "Dogs and foxes are both mammals that live in many places.",
        "The sun shines brightly on a warm summer day.",
        "Summer days are perfect for outdoor activities like jumping and running."
    ]
    
    print("\n📚 Training the model...")
    train_result = predictor.train(training_texts, show_progress=True)
    
    if train_result["success"]:
        print("\n🔮 Testing predictions...")
        
        # Test word prediction
        print("\n1. Word Prediction:")
        word_result = predictor.predict_next_word("The quick", top_k=3)
        if word_result["success"]:
            print(f"   Context: 'The quick'")
            for pred in word_result["predictions"]:
                print(f"   → '{pred['word']}' (probability: {pred['probability']:.3f})")
        
        # Test character prediction
        print("\n2. Character Prediction:")
        char_result = predictor.predict_next_character("The qui", top_k=3)
        if char_result["success"]:
            print(f"   Context: 'The qui'")
            for pred in char_result["predictions"]:
                char_display = repr(pred['character']) if pred['character'] in [' ', '\n', '\t'] else pred['character']
                print(f"   → {char_display} (probability: {pred['probability']:.3f})")
        
        # Test word completion
        print("\n3. Word Completion:")
        completion_result = predictor.complete_word("qui", top_k=3)
        if completion_result["success"]:
            print(f"   Partial: 'qui'")
            for comp in completion_result["completions"]:
                print(f"   → '{comp['word']}' (probability: {comp['probability']:.3f})")
        
        # Test text generation
        print("\n4. Text Generation:")
        gen_result = predictor.generate_text("The quick", max_length=100)
        if gen_result["success"]:
            print(f"   Seed: 'The quick'")
            print(f"   Generated: '{gen_result['generated_text']}'")
        
        # Test process_text
        print("\n5. Process Text (Comprehensive Analysis):")
        process_result = predictor.process_text("The quick b")
        if process_result["success"]:
            print(f"   Text: '{process_result['text']}'")
            print(f"   Tokenization:")
            print(f"      Words: {process_result['tokenization']['words']}")
            print(f"      Characters: {process_result['tokenization']['characters']}")
            print(f"      Word Count: {process_result['tokenization']['word_count']}")
            print(f"      Char Count: {process_result['tokenization']['char_count']}")
            print(f"   Word Predictions:")
            for pred in process_result["word_predictions"]["predictions"]:
                print(f"      → '{pred['word']}' (probability: {pred['probability']:.3f})")
            print(f"   Character Predictions:")
            for pred in process_result["character_predictions"]["predictions"]:
                char_display = repr(pred['character']) if pred['character'] in [' ', '\n', '\t'] else pred['character']
                print(f"      → {char_display} (probability: {pred['probability']:.3f})")
            print(f"   Word Completions:")
            for comp in process_result["word_completions"]["completions"]:
                print(f"      → '{comp['word']}' (probability: {comp['probability']:.3f})")
            print(f"   Text Completions: '{process_result['text_completions']['generated_text']}'")
            print(f"   Suggestions:")
            for ngram, suggestion in process_result["suggestions"].items(): 
                print(f"      N-gram: {ngram}")
                print(f"         Context: {suggestion['context']}")
                for pred in suggestion["predictions"]:
                    print(f"         → '{pred['token']}' (probability: {pred['probability']:.3f})")
                print(f"         Total Count: {suggestion['total_count']}")
        # Save the model
        print("\n💾 Saving the model...")
        if predictor.save_model("markov_model.pkl"):
            print("✓ Model saved successfully!")
        else:
            print("❌ Failed to save model")

        # Load the model
        print("\n📂 Loading the model...")
        if predictor.load_model("markov_model.pkl"):
            print("✓ Model loaded successfully!")
        else:
            print("❌ Failed to load model")
        
        # Show statistics
        print("\n📊 Model Statistics:")
        stats = predictor.get_stats()
        print(f"   Training sentences: {stats['training_sentences']}")
        print(f"   Word contexts: {stats['total_word_contexts']}")
        print(f"   Character contexts: {stats['total_char_contexts']}")
        
    else:
        print(f"❌ Training failed: {train_result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    demo()