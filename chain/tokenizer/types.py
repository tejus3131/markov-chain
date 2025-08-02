from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Tuple


class TokenizationResult(BaseModel):
    """Model for tokenization results."""

    model_config = ConfigDict(frozen=True)

    characters: List[str] = Field(
        ..., description="List of individual characters including spaces"
    )
    words: List[str] = Field(..., description="List of words and punctuation marks")
    char_ngrams: Dict[int, List[Tuple[str, ...]]] = Field(
        default_factory=dict, description="Character n-grams indexed by n-gram size"
    )
    word_ngrams: Dict[int, List[Tuple[str, ...]]] = Field(
        default_factory=dict, description="Word n-grams indexed by n-gram size"
    )

    @field_validator("characters", "words")
    @classmethod
    def validate_non_empty(cls, v: List[str], info) -> List[str]:
        """Validate that token lists are not None."""
        if v is None:
            raise ValueError(f"{info.field_name} cannot be None")
        return v


class BatchTokenizationResult(BaseModel):
    """Model for batch tokenization results."""

    model_config = ConfigDict(frozen=True)

    results: List[TokenizationResult] = Field(
        ..., description="List of tokenization results for each input text"
    )
    failed_indices: List[int] = Field(
        default_factory=list, description="Indices of texts that failed to tokenize"
    )
    errors: Dict[int, str] = Field(
        default_factory=dict, description="Error messages for failed tokenizations"
    )
