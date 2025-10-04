from fastapi import Depends, HTTPException
from .models import PredictionRequest

def validate_prediction_request(request: PredictionRequest):
    if not request.features or len(request.features) != 8:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: Features must be a list of 8 numerical values."
        )
    return request