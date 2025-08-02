
from enum import Enum
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator

class PredictionLevel(str, Enum):
    """Enumeration for prediction levels."""
    WORD = "word"
    CHAR = "char"


class Prediction(BaseModel):
    """Model representing a single prediction."""
    token: str = Field(..., description="Predicted token (word or character)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of the token")
    
    class Config:
        frozen = True


class PredictionResult(BaseModel):
    """Model representing prediction results."""
    predictions: List[Prediction] = Field(default_factory=list, description="List of predictions")
    context_used: Optional[Tuple[str, ...]] = Field(None, description="Context used for prediction")
    ngram_size: Optional[int] = Field(None, description="N-gram size used")
    
    @field_validator('predictions')
    def validate_predictions(cls, v: List[Prediction]) -> List[Prediction]:
        """Ensure predictions are sorted by probability."""
        return sorted(v, key=lambda x: x.probability, reverse=True)


class ContextSuggestion(BaseModel):
    """Model representing suggestions for a specific n-gram level."""
    context: Tuple[str, ...] = Field(..., description="N-gram context")
    predictions: List[Prediction] = Field(default_factory=list, description="Top predictions")
    total_count: int = Field(..., ge=0, description="Total count of observations")
    ngram_size: int = Field(..., ge=2, description="Size of the n-gram")


class ContextSuggestionsResult(BaseModel):
    """Model representing all context suggestions."""
    suggestions: Dict[str, ContextSuggestion] = Field(
        default_factory=dict, 
        description="Suggestions from each n-gram level"
    )
    level: PredictionLevel = Field(..., description="Prediction level used")


class TextGenerationConfig(BaseModel):
    """Configuration for text generation."""
    seed: List[str] = Field(..., min_length=1, description="Initial words to start generation")
    max_length: int = Field(50, ge=1, le=1000, description="Maximum number of words to generate")
    temperature: float = Field(1.0, gt=0.0, le=10.0, description="Controls randomness")
    stop_on_punctuation: bool = Field(True, description="Stop generation on sentence endings")
