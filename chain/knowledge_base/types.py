from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import List, Dict, Tuple


class NGramStatistics(BaseModel):
    """Model for n-gram statistics."""

    model_config = ConfigDict(frozen=True)

    total_count: int = Field(..., description="Total number of n-grams processed")
    unique_contexts: int = Field(..., description="Number of unique contexts")
    n_gram_size: int = Field(..., description="Size of the n-gram")

    @field_validator("total_count", "unique_contexts", "n_gram_size")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Validate that counts are non-negative."""
        if v < 0:
            raise ValueError("Count values must be non-negative")
        return v


class KnowledgeBaseStatistics(BaseModel):
    """Model for comprehensive knowledge base statistics."""

    model_config = ConfigDict(frozen=True)

    training_sentences: int = Field(
        default=0, description="Number of sentences used for training"
    )
    word_ngram_stats: Dict[int, NGramStatistics] = Field(
        default_factory=dict, description="Statistics for word n-grams by size"
    )
    char_ngram_stats: Dict[int, NGramStatistics] = Field(
        default_factory=dict, description="Statistics for character n-grams by size"
    )
    total_unique_word_contexts: int = Field(
        default=0, description="Total unique word contexts across all n-gram sizes"
    )
    total_unique_char_contexts: int = Field(
        default=0, description="Total unique character contexts across all n-gram sizes"
    )


class PredictionCandidate(BaseModel):
    """Model for a prediction candidate."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(..., description="The predicted token")
    count: int = Field(..., description="Frequency count")
    probability: float = Field(..., description="Probability of the token")

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: int) -> int:
        """Validate count is positive."""
        if v <= 0:
            raise ValueError("Count must be positive")
        return v

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v


class ContextPredictions(BaseModel):
    """Model for predictions from a specific context."""

    model_config = ConfigDict(frozen=True)

    context: Tuple[str, ...] = Field(..., description="The context tuple")
    candidates: List[PredictionCandidate] = Field(
        default_factory=list, description="List of prediction candidates"
    )
    n_gram_size: int = Field(..., description="Size of the n-gram")
    level: str = Field(..., description="Level of prediction (word/char)")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate prediction level."""
        if v not in ["word", "char"]:
            raise ValueError("Level must be 'word' or 'char'")
        return v

    @model_validator(mode="after")
    def validate_context_size(self) -> "ContextPredictions":
        """Validate context size matches n-gram size."""
        if len(self.context) != self.n_gram_size - 1:
            raise ValueError(
                f"Context size {len(self.context)} doesn't match "
                f"n-gram size {self.n_gram_size}"
            )
        return self
