from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    predictions: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str

class ModelMetricsResponse(BaseModel):
    classes: List[str]
    feature_count: int
    last_updated: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str