from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    predictions: Dict[str, str]
    confidence: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str
    models_available: list

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

loaded_models: Dict[str, Any] = {}
scaler = None
selected_features = []
API_VERSION = "1.0.0"

app = FastAPI(
    title="ExoHunter API",
    description="Production API for exoplanet classification using machine learning",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_models():
    global loaded_models, scaler, selected_features
    
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info(f"API file location: {Path(__file__).absolute()}")
    
    # Try multiple possible paths for the models
    possible_paths = [
        Path(__file__).parent.parent.parent / "models" / "trained_models",
        Path(__file__).parent.parent.parent / "models" / "optimized_training_results" / "trained_models",
        Path(__file__).parent.parent / "models" / "trained_models",
        Path("models") / "trained_models",
        Path("models") / "optimized_training_results" / "trained_models"
    ]
    
    models_dir = None
    for path in possible_paths:
        logger.info(f"Checking path: {path.absolute()}")
        if path.exists():
            models_dir = path
            logger.info(f"✅ Found models directory: {models_dir.absolute()}")
            break
        else:
            logger.info(f"❌ Path does not exist: {path.absolute()}")
    
    if not models_dir:
        logger.error("Could not find models directory")
        return
    
    try:
        model_files = {
            "Extra Trees": "extra_trees_20251004_155128.joblib",
            "LightGBM": "lightgbm_20251004_155128.joblib",
            "Optimized Random Forest": "optimized_rf_20251004_155128.joblib",
            "Optimized XGBoost": "optimized_xgb_20251004_155128.joblib"
        }
        
        # Load scaler separately
        scaler_path = models_dir / "scaler_20251004_155128.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"✅ Loaded scaler successfully")
        
        # Load selected features
        features_paths = [
            models_dir.parent / "selected_features_20251004_155128.json",
            models_dir.parent.parent / "selected_features_20251004_155128.json",
            Path("models") / "selected_features_20251004_155128.json"
        ]
        
        for features_path in features_paths:
            if features_path.exists():
                with open(features_path, 'r') as f:
                    selected_features = json.load(f)
                logger.info(f"✅ Loaded {len(selected_features)} selected features from {features_path}")
                break
        else:
            logger.warning("⚠️ Could not find selected features file")
        
        # Load models
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                loaded_models[name] = joblib.load(model_path)
                logger.info(f"✅ Loaded {name} model successfully")
            else:
                logger.warning(f"⚠️ {name} model file not found at {model_path}")
        
        logger.info(f"Total models loaded: {len(loaded_models)}")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting ExoHunter API...")
    load_models()
    logger.info("ExoHunter API startup complete")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if len(loaded_models) > 0 else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=len(loaded_models) > 0,
        version=API_VERSION,
        models_available=list(loaded_models.keys())
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(request: PredictionRequest):
    try:
        if len(loaded_models) == 0:
            raise HTTPException(status_code=503, detail="No models loaded")
        
        features = np.array(request.features)
        logger.info(f"Received features: {features.shape}")
        
        # Ensure we have the right number of features
        if len(selected_features) > 0 and len(features) != len(selected_features):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(selected_features)} features, got {len(features)}"
            )
        
        # Scale features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            logger.info("Features scaled successfully")
        else:
            features_scaled = features.reshape(1, -1)
            logger.warning("No scaler available, using raw features")
        
        predictions = {}
        confidences = {}
        
        # Define label mapping for encoded predictions
        label_mapping = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}
        
        for model_name, model in loaded_models.items():
            try:
                predicted_class = model.predict(features_scaled)
                
                # Convert numeric predictions to class names for XGBoost and LightGBM
                if 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
                    # These models return encoded labels, convert to class names
                    prediction_value = predicted_class[0]
                    if isinstance(prediction_value, (int, np.int32, np.int64)):
                        predictions[model_name] = label_mapping.get(int(prediction_value), f"Unknown({prediction_value})")
                    else:
                        predictions[model_name] = str(prediction_value)
                else:
                    # Random Forest and Extra Trees return string labels directly
                    predictions[model_name] = str(predicted_class[0])
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)
                    # Convert to percentage
                    confidence_score = float(np.max(proba)) * 100
                    confidences[model_name] = confidence_score
                
                logger.info(f"{model_name} prediction: {predictions[model_name]}")
            except Exception as e:
                logger.error(f"Error with model {model_name}: {str(e)}")
                predictions[model_name] = "Error"
        
        return PredictionResponse(
            predictions=predictions,
            confidence=confidences if confidences else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )