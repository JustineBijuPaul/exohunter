from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import json
import time
from functools import lru_cache

import logging
from logging.handlers import RotatingFileHandler
import sys

# Configure structured logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create handler with UTF-8 encoding for console
class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode on Windows."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Remove emoji characters for Windows console compatibility
            msg = msg.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARN]')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_dir / 'api.log',
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        SafeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of 15 numerical features for exoplanet classification",
        min_items=15,
        max_items=15
    )
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 15:
            raise ValueError(f'Expected exactly 15 features, got {len(v)}')
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError('Features cannot contain NaN or Inf values')
        return v

class BatchPredictionRequest(BaseModel):
    batch_features: List[List[float]] = Field(
        ...,
        description="List of feature sets for batch prediction",
        max_items=100  # Limit batch size
    )
    
    @validator('batch_features')
    def validate_batch(cls, v):
        for idx, features in enumerate(v):
            if len(features) != 15:
                raise ValueError(f'Row {idx}: Expected exactly 15 features, got {len(features)}')
            if any(not isinstance(x, (int, float)) for x in features):
                raise ValueError(f'Row {idx}: All features must be numeric')
            if any(np.isnan(x) or np.isinf(x) for x in features):
                raise ValueError(f'Row {idx}: Features cannot contain NaN or Inf values')
        return v

class PredictionResponse(BaseModel):
    predictions: Dict[str, str] = Field(..., description="Predictions from each model")
    confidence: Optional[Dict[str, float]] = Field(None, description="Confidence scores (0-100%)")
    ensemble_prediction: Optional[str] = Field(None, description="Majority vote prediction")
    ensemble_confidence: Optional[float] = Field(None, description="Agreement percentage")
    processing_time_ms: Optional[float] = Field(None, description="Time taken for prediction")

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_predictions: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str
    models_available: List[str]
    total_predictions: int = Field(default=0, description="Total predictions made")
    uptime_seconds: float = Field(default=0.0, description="API uptime in seconds")

class ModelMetricsResponse(BaseModel):
    model_name: str
    training_metrics: Dict[str, Any]
    feature_count: int
    classes: List[str]
    model_type: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Global state
loaded_models: Dict[str, Any] = {}
scaler = None
selected_features = []
training_metrics = {}
API_VERSION = "2.0.0"
START_TIME = datetime.now()
prediction_counter = 0

# Feature names for documentation
FEATURE_NAMES = [
    'transit_depth', 'planet_radius', 'koi_teq', 'koi_insol', 'stellar_teff',
    'stellar_radius', 'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
    'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter', 'transit_duration', 'st_dist'
]

# Label mapping
LABEL_MAPPING = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}

app = FastAPI(
    title="ExoHunter API",
    description="Production API for exoplanet classification using optimized machine learning models",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check and system status"},
        {"name": "prediction", "description": "Model prediction endpoints"},
        {"name": "models", "description": "Model information and metrics"},
    ]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to ms
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_load_model(model_path: Path):
    """Safely load a model with fallback methods for compatibility."""
    try:
        # Method 1: Standard joblib load
        return joblib.load(model_path)
    except (KeyError, Exception) as e:
        logger.warning(f"Standard load failed for {model_path.name}: {str(e)[:100]}")
        try:
            # Method 2: Load with pickle directly
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            logger.error(f"All loading methods failed for {model_path.name}")
            raise ValueError(f"Could not load model: {str(e)} | Alternative: {str(e2)}")

def load_models():
    global loaded_models, scaler, selected_features, training_metrics
    
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
            logger.info(f"[OK] Found models directory: {models_dir.absolute()}")
            break
        else:
            logger.info(f"[SKIP] Path does not exist: {path.absolute()}")
    
    if not models_dir:
        logger.error("[ERROR] Could not find models directory")
        return
    
    try:
        model_files = {
            "Extra Trees": "extra_trees_20251005_200911.joblib",
            "LightGBM": "lightgbm_20251005_200911.joblib",
            "Optimized Random Forest": "optimized_rf_20251005_200911.joblib",
            "Optimized XGBoost": "optimized_xgb_20251005_200911.joblib"
        }
        
        # Load scaler separately
        scaler_path = models_dir / "scaler_20251005_200911.joblib"
        if scaler_path.exists():
            try:
                scaler = safe_load_model(scaler_path)
                logger.info("[OK] Loaded scaler successfully")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load scaler: {str(e)[:100]}")
                logger.warning("[WARN] Predictions will use unscaled features")
        
        # Load selected features
        features_paths = [
            models_dir.parent / "selected_features_20251005_200911.json",
            models_dir.parent.parent / "selected_features_20251005_200911.json",
            Path("models") / "selected_features_20251005_200911.json"
        ]
        
        for features_path in features_paths:
            if features_path.exists():
                with open(features_path, 'r') as f:
                    selected_features = json.load(f)
                logger.info(f"[OK] Loaded {len(selected_features)} selected features from {features_path}")
                break
        else:
            logger.warning("[WARN] Could not find selected features file")
        
        # Load training metrics
        metrics_paths = [
            models_dir.parent / "training_results_20251005_200911.json",
            models_dir.parent.parent / "training_results_20251005_200911.json",
            Path("models") / "training_results_20251005_200911.json"
        ]
        
        for metrics_path in metrics_paths:
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    training_metrics = json.load(f)
                logger.info(f"[OK] Loaded training metrics from {metrics_path}")
                break
        else:
            logger.warning("[WARN] Could not find training metrics file")
        
        # Load models
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    loaded_models[name] = safe_load_model(model_path)
                    logger.info(f"[OK] Loaded {name} model successfully")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load {name}: {str(e)[:100]}")
            else:
                logger.warning(f"[WARN] {name} model file not found at {model_path}")
        
        logger.info(f"[INFO] Total models loaded: {len(loaded_models)}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error loading models: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting ExoHunter API...")
    load_models()
    logger.info("ExoHunter API startup complete")

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check API health status and get system information."""
    uptime = (datetime.now() - START_TIME).total_seconds()
    return HealthResponse(
        status="healthy" if len(loaded_models) > 0 else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=len(loaded_models) > 0,
        version=API_VERSION,
        models_available=list(loaded_models.keys()),
        total_predictions=prediction_counter,
        uptime_seconds=uptime
    )

def calculate_ensemble_prediction(predictions: Dict[str, str]) -> tuple[str, float]:
    """Calculate ensemble prediction using majority voting."""
    if not predictions:
        return "UNKNOWN", 0.0
    
    # Count votes
    vote_counts = {}
    for pred in predictions.values():
        if pred != "Error":
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
    
    if not vote_counts:
        return "UNKNOWN", 0.0
    
    # Get majority prediction
    ensemble_pred = max(vote_counts, key=vote_counts.get)
    total_votes = sum(vote_counts.values())
    confidence = (vote_counts[ensemble_pred] / total_votes) * 100
    
    return ensemble_pred, confidence

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_exoplanet(request: PredictionRequest):
    """Predict exoplanet classification from features.
    
    Expects 15 numerical features in the following order:
    1. transit_depth (ppm)
    2. planet_radius (Earth radii)
    3. koi_teq (K)
    4. koi_insol (Earth flux)
    5. stellar_teff (K)
    6. stellar_radius (Solar radii)
    7. koi_smass (Solar masses)
    8. koi_slogg (log g)
    9. koi_count
    10. koi_num_transits
    11. koi_max_sngle_ev
    12. koi_max_mult_ev
    13. impact_parameter
    14. transit_duration (hours)
    15. st_dist (parsecs)
    """
    global prediction_counter
    start_time = time.time()
    
    try:
        if len(loaded_models) == 0:
            raise HTTPException(status_code=503, detail="No models loaded")
        
        features = np.array(request.features)
        logger.info(f"Received features: {features.shape}")
        prediction_counter += 1
        
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
        
        for model_name, model in loaded_models.items():
            try:
                predicted_class = model.predict(features_scaled)
                
                # Convert numeric predictions to class names for XGBoost and LightGBM
                if 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
                    # These models return encoded labels, convert to class names
                    prediction_value = predicted_class[0]
                    if isinstance(prediction_value, (int, np.int32, np.int64)):
                        predictions[model_name] = LABEL_MAPPING.get(int(prediction_value), f"Unknown({prediction_value})")
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
                    confidences[model_name] = round(confidence_score, 2)
                
                logger.info(f"{model_name} prediction: {predictions[model_name]}")
            except Exception as e:
                logger.error(f"Error with model {model_name}: {str(e)}", exc_info=True)
                predictions[model_name] = "Error"
        
        # Calculate ensemble prediction
        ensemble_pred, ensemble_conf = calculate_ensemble_prediction(predictions)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            predictions=predictions,
            confidence=confidences if confidences else None,
            ensemble_prediction=ensemble_pred,
            ensemble_confidence=round(ensemble_conf, 2),
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint for multiple exoplanet classifications.
    
    Accepts up to 100 feature sets and returns predictions for all of them.
    Each feature set should contain 15 numerical features.
    """
    global prediction_counter
    start_time = time.time()
    
    try:
        if len(loaded_models) == 0:
            raise HTTPException(status_code=503, detail="No models loaded")
        
        results = []
        
        for features in request.batch_features:
            pred_start = time.time()
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            if scaler is not None:
                features_scaled = scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            predictions = {}
            confidences = {}
            
            # Make predictions with each model
            for model_name, model in loaded_models.items():
                try:
                    predicted_class = model.predict(features_scaled)
                    
                    if 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
                        prediction_value = predicted_class[0]
                        if isinstance(prediction_value, (int, np.int32, np.int64)):
                            predictions[model_name] = LABEL_MAPPING.get(int(prediction_value), f"Unknown({prediction_value})")
                        else:
                            predictions[model_name] = str(prediction_value)
                    else:
                        predictions[model_name] = str(predicted_class[0])
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)
                        confidence_score = float(np.max(proba)) * 100
                        confidences[model_name] = round(confidence_score, 2)
                
                except Exception as e:
                    logger.error(f"Batch prediction error with {model_name}: {str(e)}")
                    predictions[model_name] = "Error"
            
            ensemble_pred, ensemble_conf = calculate_ensemble_prediction(predictions)
            pred_time = (time.time() - pred_start) * 1000
            
            results.append(PredictionResponse(
                predictions=predictions,
                confidence=confidences if confidences else None,
                ensemble_prediction=ensemble_pred,
                ensemble_confidence=round(ensemble_conf, 2),
                processing_time_ms=round(pred_time, 2)
            ))
            
            prediction_counter += 1
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            results=results,
            total_predictions=len(results),
            processing_time_ms=round(total_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/metrics", tags=["models"])
async def get_model_metrics():
    """Get training metrics for all loaded models.
    
    Returns performance metrics from the training phase including accuracy,
    precision, recall, and F1-scores for each model.
    """
    if len(loaded_models) == 0:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    metrics_response = []
    
    for model_name in loaded_models.keys():
        # Map internal model names to training results keys
        training_key = model_name.lower().replace(" ", "_")
        
        if training_key in training_metrics:
            model_metrics = training_metrics[training_key]
            
            metrics_response.append(ModelMetricsResponse(
                model_name=model_name,
                training_metrics={
                    "cv_mean": round(model_metrics.get("cv_mean", 0), 4),
                    "cv_std": round(model_metrics.get("cv_std", 0), 4),
                    "test_accuracy": round(model_metrics.get("test_accuracy", 0), 4),
                    "classification_report": model_metrics.get("classification_report", {})
                },
                feature_count=len(selected_features) if selected_features else 15,
                classes=["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
                model_type=model_name
            ))
    
    if not metrics_response:
        return JSONResponse(
            status_code=200,
            content={
                "message": "Models loaded but metrics not available",
                "models": list(loaded_models.keys())
            }
        )
    
    return metrics_response

@app.get("/models/features", tags=["models"])
async def get_feature_names():
    """Get the list of feature names expected by the models.
    
    Returns the exact order and names of features required for prediction.
    """
    return {
        "feature_count": len(FEATURE_NAMES),
        "features": [
            {
                "index": i,
                "name": name,
                "description": f"Feature {i+1}: {name}"
            }
            for i, name in enumerate(FEATURE_NAMES)
        ]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )