from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import numpy as np
from ..models import PredictionRequest, PredictionResponse
from ..dependencies import get_model

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(request: PredictionRequest):
    model = get_model()
    features = np.array(request.features).reshape(1, -1)

    try:
        predictions = {}
        for model_name, loaded_model in model.items():
            predicted_class = loaded_model.predict(features)
            predictions[model_name] = predicted_class[0]

        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))