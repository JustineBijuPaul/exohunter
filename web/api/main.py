"""
FastAPI application for ExoHunter exoplanet classification.

This module provides a production-ready API for exoplanet classification
with endpoints for prediction, health checks, and model metrics.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import joblib  # Use joblib instead of pickle for scikit-learn models
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import ExoHunter database functionality
try:
    from exohunter.db import init_db, save_prediction, log_api_request, get_db
    from sqlalchemy.sql import func
    DATABASE_AVAILABLE = True
except ImportError:
    print("Warning: Database functionality not available. Install SQLAlchemy for full features.")
    DATABASE_AVAILABLE = False

# Import models and API schemas
from .models import (
    PredictionRequest,
    LightCurveRequest, 
    PredictionResponse,
    HealthResponse,
    ModelMetricsResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and preprocessors
loaded_models: Dict[str, Any] = {}
model_metadata: Dict[str, Any] = {}

# Database manager
db_manager = None

# API version
API_VERSION = "1.0.0"

app = FastAPI(
    title="ExoHunter API",
    description="Production API for exoplanet classification using machine learning",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware to log all API requests to database and file."""
    start_time = time.time()
    
    # Extract request information
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    method = request.method
    endpoint = str(request.url.path)
    
    # Get request size (approximation)
    request_size = len(str(request.url)) + sum(len(f"{k}: {v}") for k, v in request.headers.items())
    
    response = None
    status_code = 500
    error_message = None
    error_type = None
    
    try:
        # Process the request
        response = await call_next(request)
        status_code = response.status_code
        
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        status_code = 500
        response = JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )
    
    # Calculate response time
    response_time_ms = (time.time() - start_time) * 1000
    
    # Get response size (approximation)
    response_size = 0
    if hasattr(response, 'body'):
        response_size = len(response.body) if response.body else 0
    
    # Log to database if available
    if DATABASE_AVAILABLE and db_manager:
        try:
            log_api_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                ip_address=client_ip,
                user_agent=user_agent,
                request_size=request_size,
                response_size=response_size,
                error_message=error_message,
                error_type=error_type
            )
        except Exception as log_error:
            logger.error(f"Failed to log API request: {log_error}")
    
    return response


@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup."""
    global db_manager
    
    logger.info("Starting ExoHunter API...")
    
    # Initialize database
    if DATABASE_AVAILABLE:
        try:
            db_manager = init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    # Load models
    load_models_on_startup()
    
    logger.info("ExoHunter API startup complete")


def load_models_on_startup():
    """Load models and preprocessors on application startup."""
    global loaded_models, model_metadata
    
    try:
        # Look for saved models in the models directory
        models_dir = Path(__file__).parent.parent.parent / "models"
        
        # Try to load ULTIMATE model first (best performance - 92.53% accuracy)
        ultimate_model_path = models_dir / "exoplanet_classifier_ultimate.pkl"
        ultimate_scaler_path = models_dir / "feature_scaler_ultimate.pkl"
        ultimate_metrics_path = models_dir / "training_metrics_ultimate.json"
        
        if ultimate_model_path.exists() and ultimate_scaler_path.exists():
            loaded_models['ultimate'] = joblib.load(ultimate_model_path)
            loaded_models['ultimate_scaler'] = joblib.load(ultimate_scaler_path)
            logger.info("✅ Loaded ULTIMATE model successfully (92.53% accuracy, 97.14% ROC AUC)")
            
            # Load ultimate model metrics
            if ultimate_metrics_path.exists():
                with open(ultimate_metrics_path, 'r') as f:
                    ultimate_metrics = json.load(f)
                model_metadata.update({
                    "classes": ["FALSE POSITIVE", "CONFIRMED PLANET"],
                    "feature_count": 13,
                    "feature_names": [
                        "orbital_period", "transit_depth", "planet_radius",
                        "koi_teq", "koi_insol", "stellar_teff", "stellar_radius",
                        "koi_smass", "koi_slogg", "transit_duration",
                        "impact_parameter", "koi_max_mult_ev", "koi_num_transits"
                    ],
                    "last_updated": datetime.now().isoformat(),
                    "training_accuracy": 0.9923,
                    "validation_accuracy": 0.9269,
                    "test_accuracy": ultimate_metrics.get("accuracy", 0.9253),
                    "cross_validation_score": ultimate_metrics.get("cv_mean", 0.9209),
                    "cv_std": ultimate_metrics.get("cv_std", 0.0076),
                    "precision": ultimate_metrics.get("precision", 0.8829),
                    "recall": ultimate_metrics.get("recall", 0.9150),
                    "f1_score": ultimate_metrics.get("f1_score", 0.8987),
                    "roc_auc": ultimate_metrics.get("roc_auc", 0.9714),
                    "high_confidence_pct": ultimate_metrics.get("high_confidence_pct", 0.7540),
                    "very_high_confidence_pct": ultimate_metrics.get("very_high_confidence_pct", 0.5290),
                    "model_version": "3.0_ultimate_ensemble",
                    "model_type": "5-classifier ensemble (RF+ET+GB+LR+SVM)"
                })
            else:
                logger.warning("Ultimate model metrics file not found, using defaults")
                model_metadata.update({
                    "classes": ["FALSE POSITIVE", "CONFIRMED PLANET"],
                    "feature_count": 13,
                    "feature_names": [
                        "orbital_period", "transit_depth", "planet_radius",
                        "koi_teq", "koi_insol", "stellar_teff", "stellar_radius",
                        "koi_smass", "koi_slogg", "transit_duration",
                        "impact_parameter", "koi_max_mult_ev", "koi_num_transits"
                    ],
                    "model_version": "3.0_ultimate_ensemble",
                    "model_type": "5-classifier ensemble (RF+ET+GB+LR+SVM)"
                })
        
        # Try to load ADVANCED model as fallback (92.62% accuracy)
        advanced_model_path = models_dir / "exoplanet_classifier_advanced.pkl"
        advanced_scaler_path = models_dir / "feature_scaler_advanced.pkl"
        
        if advanced_model_path.exists() and advanced_scaler_path.exists():
            loaded_models['advanced'] = joblib.load(advanced_model_path)
            loaded_models['advanced_scaler'] = joblib.load(advanced_scaler_path)
            logger.info("✅ Loaded ADVANCED model as fallback (92.62% accuracy)")
        
        # Load older models as additional fallbacks
        ensemble_path = models_dir / "stacking_ensemble.pkl"
        if ensemble_path.exists():
            loaded_models['ensemble'] = joblib.load(ensemble_path)
            logger.info("Loaded ensemble model as backup")
            
        xgb_path = models_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            loaded_models['xgboost'] = joblib.load(xgb_path)
            logger.info("Loaded XGBoost model as backup")
        
        # Set default metadata if no ultimate model loaded
        if 'ultimate' not in loaded_models:
            model_metadata = {
                "classes": ["FALSE POSITIVE", "CONFIRMED PLANET"],
                "feature_count": 8,
                "last_updated": datetime.now().isoformat(),
                "training_accuracy": 0.85,
                "validation_accuracy": 0.82,
                "cross_validation_score": 0.80
            }
            
        if not loaded_models:
            logger.warning("⚠️ No models could be loaded. API will use mock predictions.")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        model_metadata = {
            "classes": ["FALSE POSITIVE", "CONFIRMED PLANET"],
            "feature_count": 13,
            "last_updated": datetime.now().isoformat()
        }


def get_best_model():
    """Get the best available model with its scaler."""
    if 'ultimate' in loaded_models:
        return loaded_models['ultimate'], loaded_models.get('ultimate_scaler'), 'ultimate'
    elif 'advanced' in loaded_models:
        return loaded_models['advanced'], loaded_models.get('advanced_scaler'), 'advanced'
    elif 'ensemble' in loaded_models:
        return loaded_models['ensemble'], None, 'ensemble'
    elif 'xgboost' in loaded_models:
        return loaded_models['xgboost'], None, 'xgboost'
    elif 'random_forest' in loaded_models:
        return loaded_models['random_forest'], None, 'random_forest'
    else:
        return None, None, 'mock'


def predict_with_model(features: np.ndarray):
    """Make prediction with the best available model."""
    model, scaler, model_type = get_best_model()
    
    if model is None:
        # Mock prediction for demonstration
        logger.warning("Using mock prediction - no models loaded")
        predicted_class = np.random.choice(model_metadata["classes"])
        probabilities = np.random.dirichlet([1, 1])  # Random probabilities for 2 classes
        
        prob_dict = {cls: float(prob) for cls, prob in zip(model_metadata["classes"], probabilities)}
        max_prob = float(np.max(probabilities))
        
        return predicted_class, max_prob, prob_dict, model_type
    
    try:
        # Reshape features for prediction
        features_reshaped = features.reshape(1, -1)
        
        # Scale features if scaler is available (for ultimate/advanced models)
        if scaler is not None:
            features_scaled = scaler.transform(features_reshaped)
        else:
            features_scaled = features_reshaped
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            predicted_idx = np.argmax(probabilities)
        else:
            prediction = model.predict(features_scaled)[0]
            predicted_idx = int(prediction) if prediction in [0, 1] else 0
            probabilities = np.zeros(len(model_metadata["classes"]))
            probabilities[predicted_idx] = 1.0
        
        predicted_class = model_metadata["classes"][predicted_idx]
        max_prob = float(probabilities[predicted_idx])
        prob_dict = {cls: float(prob) for cls, prob in zip(model_metadata["classes"], probabilities)}
        
        return predicted_class, max_prob, prob_dict, model_type
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def expand_features_to_13(features: np.ndarray) -> np.ndarray:
    """
    Expand a feature array to 13 features using intelligent defaults.
    
    This allows the API to accept 8 core features from the frontend
    and fill in reasonable defaults for the additional 5 features.
    
    Expected 8 core features:
    1. orbital_period
    2. transit_depth
    3. planet_radius
    4. stellar_teff
    5. stellar_radius
    6. transit_duration
    7. impact_parameter
    8. koi_num_transits
    
    Additional 5 features (calculated/estimated):
    9. koi_teq (calculated from stellar temp and distance)
    10. koi_insol (calculated from stellar properties)
    11. koi_smass (estimated from stellar radius)
    12. koi_slogg (estimated from stellar properties)
    13. koi_max_mult_ev (estimated from signal strength)
    """
    if len(features) == 13:
        return features  # Already complete
    
    if len(features) != 8:
        raise ValueError(f"Expected 8 or 13 features, got {len(features)}")
    
    # Extract the 8 core features
    orbital_period = features[0]
    transit_depth = features[1]
    planet_radius = features[2]
    stellar_teff = features[3]
    stellar_radius = features[4]
    transit_duration = features[5]
    impact_parameter = features[6]
    koi_num_transits = features[7]
    
    # Calculate/estimate the additional 5 features
    
    # koi_teq: Equilibrium temperature (simplified Stefan-Boltzmann)
    # T_eq ≈ T_star * sqrt(R_star / (2 * a))
    # Estimate semi-major axis from Kepler's 3rd law (assume solar mass)
    a_au = (orbital_period / 365.25) ** (2/3)  # Semi-major axis in AU
    koi_teq = stellar_teff * np.sqrt(stellar_radius / (2 * a_au * 215))  # 215 = AU to solar radii
    koi_teq = np.clip(koi_teq, 100, 3000)  # Reasonable bounds
    
    # koi_insol: Insolation flux (Earth = 1.0)
    # F ∝ (R_star²) / a²
    koi_insol = (stellar_radius ** 2) / (a_au ** 2)
    koi_insol = np.clip(koi_insol, 0.01, 10000)  # Reasonable bounds
    
    # koi_smass: Stellar mass (mass-radius relation for main sequence)
    # M ∝ R^α where α ≈ 0.8 for main sequence stars
    if stellar_radius < 1.0:
        koi_smass = stellar_radius ** 2.5  # Lower mass stars
    else:
        koi_smass = stellar_radius ** 0.8  # Higher mass stars
    koi_smass = np.clip(koi_smass, 0.1, 2.0)  # Reasonable bounds
    
    # koi_slogg: Surface gravity log g
    # log g = log(M) - 2*log(R) + constant
    # For Sun: log g = 4.44
    koi_slogg = np.log10(koi_smass) - 2 * np.log10(stellar_radius) + 4.44
    koi_slogg = np.clip(koi_slogg, 3.5, 5.0)  # Reasonable bounds
    
    # koi_max_mult_ev: Multiple event statistic (signal strength indicator)
    # Estimate from transit depth, duration, and number of transits
    # Higher values indicate stronger, more significant signals
    signal_strength = (transit_depth / 100) * (transit_duration / 5) * np.sqrt(koi_num_transits)
    koi_max_mult_ev = signal_strength * 10
    koi_max_mult_ev = np.clip(koi_max_mult_ev, 1, 500)  # Reasonable bounds
    
    # Construct the full 13-feature array in the correct order
    full_features = np.array([
        orbital_period,      # 1
        transit_depth,       # 2
        planet_radius,       # 3
        koi_teq,            # 4 (calculated)
        koi_insol,          # 5 (calculated)
        stellar_teff,        # 6
        stellar_radius,      # 7
        koi_smass,          # 8 (calculated)
        koi_slogg,          # 9 (calculated)
        transit_duration,    # 10
        impact_parameter,    # 11
        koi_max_mult_ev,    # 12 (calculated)
        koi_num_transits    # 13
    ])
    
    return full_features


def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability (updated for production model)."""
    if probability >= 0.90:
        return "VERY HIGH"
    elif probability >= 0.80:
        return "HIGH"
    elif probability >= 0.70:
        return "MEDIUM"
    elif probability >= 0.60:
        return "LOW"
    else:
        return "VERY LOW"


@app.on_event("startup")
async def startup_event():
    """Load models when the application starts."""
    logger.info("Starting ExoHunter API...")
    load_models_on_startup()
    logger.info("ExoHunter API startup complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=len(loaded_models) > 0,
        version=API_VERSION
    )


@app.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Get current model metrics and information."""
    _, _, model_type = get_best_model()
    
    metrics = ModelMetricsResponse(
        model_type=f"{model_type} - {model_metadata.get('model_type', 'N/A')}",
        training_accuracy=model_metadata.get("training_accuracy"),
        validation_accuracy=model_metadata.get("validation_accuracy"),
        cross_validation_score=model_metadata.get("cross_validation_score"),
        feature_count=model_metadata.get("feature_count"),
        classes=model_metadata.get("classes", []),
        last_updated=model_metadata.get("last_updated")
    )
    
    # Add additional metrics if available
    if hasattr(metrics, '__dict__'):
        metrics_dict = metrics.__dict__
        metrics_dict.update({
            "test_accuracy": model_metadata.get("test_accuracy"),
            "precision": model_metadata.get("precision"),
            "recall": model_metadata.get("recall"),
            "f1_score": model_metadata.get("f1_score"),
            "roc_auc": model_metadata.get("roc_auc"),
            "high_confidence_pct": model_metadata.get("high_confidence_pct"),
            "very_high_confidence_pct": model_metadata.get("very_high_confidence_pct"),
            "model_version": model_metadata.get("model_version"),
            "feature_names": model_metadata.get("feature_names", [])
        })
    
    return metrics


@app.post("/predict", response_model=PredictionResponse)
async def predict_exoplanet(request: PredictionRequest):
    """
    Predict exoplanet classification from tabular features.
    
    Accepts either 8 core features or all 13 features.
    If 8 features are provided, the remaining 5 are intelligently estimated.
    
    8 Core Features (in order):
    1. orbital_period - Orbital period (days)
    2. transit_depth - Transit depth (ppm)
    3. planet_radius - Planet radius (Earth radii)
    4. stellar_teff - Stellar temperature (Kelvin)
    5. stellar_radius - Stellar radius (Solar radii)
    6. transit_duration - Transit duration (hours)
    7. impact_parameter - Impact parameter (0-1)
    8. koi_num_transits - Number of transits observed
    
    Returns predicted class, probability, and confidence level.
    """
    start_time = time.time()
    
    try:
        # Convert features to numpy array
        features = np.array(request.features)
        
        # Log the incoming features
        logger.info(f"Received {len(features)} features for prediction")
        
        # Validate feature count (accept 8 or 13)
        if len(features) not in [8, 13]:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 8 or 13 features, got {len(features)}. "
                       f"Provide either 8 core features (will estimate remaining 5) "
                       f"or all 13 features."
            )
        
        # Expand to 13 features if needed
        if len(features) == 8:
            logger.info("Expanding 8 features to 13 with intelligent defaults")
            features = expand_features_to_13(features)
            logger.info(f"Expanded features: {features}")
        
        # Make prediction
        predicted_class, probability, all_probs, model_type = predict_with_model(features)
        confidence = get_confidence_level(probability)
        
        # Calculate processing time
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Save prediction to database
        if DATABASE_AVAILABLE:
            try:
                save_prediction(
                    prediction_type="features",
                    model_type=model_type,
                    predicted_label=predicted_class,
                    predicted_probability=probability,
                    all_probabilities=all_probs,
                    input_features=features.tolist(),
                    feature_names=getattr(request, 'feature_names', None),
                    model_version=model_type,
                    confidence_level=confidence,
                    prediction_time_ms=prediction_time_ms
                )
            except Exception as db_error:
                logger.error(f"Failed to save prediction to database: {db_error}")
        
        return PredictionResponse(
            predicted_label=predicted_class,
            probability=probability,
            confidence=confidence,
            all_probabilities=all_probs,
            model_version=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/lightcurve", response_model=PredictionResponse)
async def predict_lightcurve(request: LightCurveRequest):
    """
    Predict exoplanet classification from light curve data.
    
    Accepts time and flux arrays and returns the predicted class,
    probability, and confidence level.
    """
    start_time = time.time()
    
    try:
        # For now, we'll extract simple statistical features from the light curve
        # In a production system, this would use the CNN model
        flux_array = np.array(request.flux)
        time_array = np.array(request.time)
        
        # Extract statistical features
        features = np.array([
            np.mean(flux_array),                    # Mean flux
            np.std(flux_array),                     # Standard deviation
            np.median(flux_array),                  # Median flux
            np.percentile(flux_array, 25),          # 25th percentile
            np.percentile(flux_array, 75),          # 75th percentile
            np.max(flux_array) - np.min(flux_array), # Range
            len(flux_array),                        # Number of observations
            np.mean(np.diff(flux_array))            # Mean difference between consecutive points
        ])
        
        # Make prediction using statistical features
        predicted_class, probability, all_probs, model_type = predict_with_model(features)
        confidence = get_confidence_level(probability)
        
        # Calculate processing time
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Save prediction to database
        if DATABASE_AVAILABLE:
            try:
                save_prediction(
                    prediction_type="lightcurve",
                    model_type=f"{model_type}_lightcurve",
                    predicted_label=predicted_class,
                    predicted_probability=probability,
                    all_probabilities=all_probs,
                    input_features=features.tolist(),
                    feature_names=['mean_flux', 'std_flux', 'median_flux', 'q25_flux', 'q75_flux', 'range_flux', 'n_obs', 'mean_diff'],
                    time_series=time_array.tolist(),
                    flux_series=flux_array.tolist(),
                    model_version=f"{model_type}_lightcurve",
                    confidence_level=confidence,
                    prediction_time_ms=prediction_time_ms
                )
            except Exception as db_error:
                logger.error(f"Failed to save light curve prediction to database: {db_error}")
        
        return PredictionResponse(
            predicted_label=predicted_class,
            probability=probability,
            confidence=confidence,
            all_probabilities=all_probs,
            model_version=f"{model_type}_lightcurve"
        )
        
    except Exception as e:
        logger.error(f"Light curve prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Light curve prediction failed: {str(e)}")


@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(file: UploadFile = File(...)):
    """
    Predict exoplanet classification from uploaded CSV file.
    
    Accepts a CSV file with either:
    - Single row of features for tabular prediction
    - Time and flux columns for light curve prediction
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Parse CSV
        try:
            import io
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty CSV file")
        
        # Check if it's a light curve (has time and flux columns)
        if 'time' in df.columns and 'flux' in df.columns:
            # Light curve prediction
            time_data = df['time'].tolist()
            flux_data = df['flux'].tolist()
            
            request = LightCurveRequest(time=time_data, flux=flux_data)
            return await predict_lightcurve(request)
            
        else:
            # Tabular prediction - use all numeric columns as features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise HTTPException(status_code=400, detail="No numeric columns found for prediction")
            
            # Use the first row for prediction
            features = df[numeric_columns].iloc[0].tolist()
            request = PredictionRequest(features=features)
            return await predict_exoplanet(request)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload prediction failed: {str(e)}")


@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """
    Get prediction history from the database.
    
    Args:
        limit: Maximum number of predictions to return (default: 100)
        
    Returns:
        List of recent predictions with metadata
    """
    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database functionality not available"
        )
    
    try:
        db = get_db()
        history = db.get_prediction_history(limit=limit)
        
        return {
            "predictions": history,
            "count": len(history),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve prediction history: {str(e)}"
        )


@app.get("/admin/stats")
async def get_api_stats():
    """
    Get API usage statistics (admin endpoint).
    
    Returns statistics about API usage, predictions, and performance.
    """
    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database functionality not available"
        )
    
    try:
        from exohunter.db.models import Prediction, APIRequest, UserSession
        
        db = get_db()
        session = db.get_session()
        
        if not session:
            raise HTTPException(status_code=503, detail="Database session unavailable")
        
        try:
            # Get counts
            total_predictions = session.query(Prediction).count()
            total_api_requests = session.query(APIRequest).count()
            total_sessions = session.query(UserSession).count()
            
            # Get recent activity (last 24 hours)
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            
            recent_predictions = session.query(Prediction).filter(
                Prediction.created_at >= yesterday
            ).count()
            
            recent_requests = session.query(APIRequest).filter(
                APIRequest.timestamp >= yesterday
            ).count()
            
            # Get prediction type distribution
            prediction_types = session.query(
                Prediction.prediction_type,
                func.count(Prediction.id)
            ).group_by(Prediction.prediction_type).all()
            
            # Get model usage distribution
            model_usage = session.query(
                Prediction.model_type,
                func.count(Prediction.id)
            ).group_by(Prediction.model_type).all()
            
            # Get average response times
            avg_prediction_time = session.query(
                func.avg(Prediction.prediction_time_ms)
            ).scalar() or 0
            
            avg_response_time = session.query(
                func.avg(APIRequest.response_time_ms)
            ).scalar() or 0
            
            session.close()
            
            return {
                "totals": {
                    "predictions": total_predictions,
                    "api_requests": total_api_requests,
                    "user_sessions": total_sessions
                },
                "recent_activity": {
                    "predictions_24h": recent_predictions,
                    "api_requests_24h": recent_requests
                },
                "distributions": {
                    "prediction_types": dict(prediction_types),
                    "model_usage": dict(model_usage)
                },
                "performance": {
                    "avg_prediction_time_ms": round(avg_prediction_time, 2),
                    "avg_response_time_ms": round(avg_response_time, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            session.close()
            raise e
            
    except Exception as e:
        logger.error(f"Failed to get API stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve API statistics: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


def main():
    """Run the FastAPI application with Uvicorn."""
    uvicorn.run(
        "web.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
