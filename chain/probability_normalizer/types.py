
from enum import Enum
from typing import Dict, List, Tuple, Annotated
from pydantic import BaseModel, Field, field_validator 
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SmoothingMethod(str, Enum):
    """Supported smoothing methods for probability normalization."""
    
    LAPLACE = "laplace"
    GOOD_TURING = "good_turing"
    NONE = "none"

class ProbabilityDistribution(BaseModel):
    """Model representing a probability distribution."""
    
    probabilities: Dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        ..., 
        description="Mapping of items to their probabilities"
    )
    total_probability: Annotated[float, Field(ge=0.0, le=1.01)] = Field(
        1.0,
        description="Sum of all probabilities (should be ~1.0)"
    )
    
    @field_validator('total_probability', mode='after')
    def _validate_total_probability(cls, v, info):
        """Ensure probabilities sum to approximately 1.0."""
        if info.data.get('probabilities'):
            total = sum(info.data['probabilities'].values())
            if not (0.0 <= total <= 1.01):
                logger.warning(f"Probabilities sum to {total}, expected to be <= 1.0")
            return total
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        frozen = True
        extra = "forbid"

class PredictionResult(BaseModel):
    """Model for top prediction results."""
    
    predictions: List[Tuple[str, Annotated[float, Field(ge=0.0, le=1.0)]]] = Field(
        ...,
        description="List of (item, probability) tuples"
    )
    top_k: Annotated[int, Field(ge=1)] = Field(..., description="Number of predictions")
    
    class Config:
        """Pydantic model configuration."""
        
        frozen = True
        extra = "forbid"


class PerplexityResult(BaseModel):
    """Model for perplexity calculation results."""
    
    perplexity: float = Field(..., description="Perplexity score")
    sequence_length: int = Field(..., description="Length of input sequence")
    
    @field_validator('perplexity')
    def _validate_perplexity(cls, v):
        """Ensure perplexity is positive."""
        if v <= 0 and v != float('inf'):
            raise ValueError("Perplexity must be positive or infinity")
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        frozen = True
        extra = "forbid"
