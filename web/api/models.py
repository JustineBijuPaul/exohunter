"""
Pydantic models for ExoHunter API.

This module defines the request and response models for the FastAPI endpoints.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    features: List[float] = Field(
        ..., 
        description="List of numerical features for exoplanet classification",
        min_items=1
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that features are finite numbers."""
        if not all(np.isfinite(f) for f in v):
            raise ValueError("All features must be finite numbers")
        return v


class LightCurveRequest(BaseModel):
    """Request model for light curve prediction."""
    
    time: List[float] = Field(
        ...,
        description="Time series values",
        min_items=10
    )
    flux: List[float] = Field(
        ...,
        description="Flux values corresponding to time points",
        min_items=10
    )
    
    @validator('time', 'flux')
    def validate_arrays(cls, v):
        """Validate that arrays contain finite numbers."""
        if not all(np.isfinite(f) for f in v):
            raise ValueError("All values must be finite numbers")
        return v
    
    @validator('flux')
    def validate_matching_lengths(cls, v, values):
        """Validate that time and flux arrays have the same length."""
        if 'time' in values and len(v) != len(values['time']):
            raise ValueError("Time and flux arrays must have the same length")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints."""
    
    predicted_label: str = Field(
        ...,
        description="Predicted class label (e.g., 'CANDIDATE', 'FALSE POSITIVE', 'CONFIRMED')"
    )
    probability: float = Field(
        ...,
        description="Probability of the predicted class",
        ge=0.0,
        le=1.0
    )
    confidence: str = Field(
        ...,
        description="Confidence level (HIGH, MEDIUM, LOW)"
    )
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all classes"
    )
    model_version: str = Field(
        ...,
        description="Version or type of model used for prediction"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(
        ...,
        description="Service status"
    )
    timestamp: str = Field(
        ...,
        description="Current timestamp"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    version: str = Field(
        ...,
        description="API version"
    )


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics endpoint."""
    
    model_type: str = Field(
        ...,
        description="Type of model (e.g., 'ensemble', 'xgboost', 'mlp')"
    )
    training_accuracy: Optional[float] = Field(
        None,
        description="Training accuracy if available",
        ge=0.0,
        le=1.0
    )
    validation_accuracy: Optional[float] = Field(
        None,
        description="Validation accuracy if available",
        ge=0.0,
        le=1.0
    )
    cross_validation_score: Optional[float] = Field(
        None,
        description="Cross-validation score if available",
        ge=0.0,
        le=1.0
    )
    feature_count: Optional[int] = Field(
        None,
        description="Number of features the model expects",
        ge=1
    )
    classes: List[str] = Field(
        ...,
        description="List of class labels the model can predict"
    )
    last_updated: Optional[str] = Field(
        None,
        description="Timestamp when the model was last updated"
    )


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    timestamp: str = Field(
        ...,
        description="Error timestamp"
    )
