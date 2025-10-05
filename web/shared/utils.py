"""
Shared utilities for ExoHunter web applications.
Common functions for model loading, predictions, and data validation.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants
FEATURE_NAMES = [
    'transit_depth', 'planet_radius', 'koi_teq', 'koi_insol', 'stellar_teff',
    'stellar_radius', 'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
    'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter', 'transit_duration', 'st_dist'
]

LABEL_MAPPING = {
    0: 'CANDIDATE',
    1: 'CONFIRMED',
    2: 'FALSE POSITIVE'
}

FEATURE_INFO = {
    'transit_depth': {'description': 'Transit depth (ppm)', 'min': 0, 'max': 100000, 'default': 1000, 'unit': 'ppm'},
    'planet_radius': {'description': 'Planet radius (Earth radii)', 'min': 0.1, 'max': 50, 'default': 2.5, 'unit': 'R⊕'},
    'koi_teq': {'description': 'Equilibrium temperature (K)', 'min': 200, 'max': 3000, 'default': 800, 'unit': 'K'},
    'koi_insol': {'description': 'Insolation flux (Earth flux)', 'min': 0.1, 'max': 10000, 'default': 100, 'unit': 'F⊕'},
    'stellar_teff': {'description': 'Stellar effective temperature (K)', 'min': 3000, 'max': 8000, 'default': 5800, 'unit': 'K'},
    'stellar_radius': {'description': 'Stellar radius (Solar radii)', 'min': 0.1, 'max': 5, 'default': 1.0, 'unit': 'R☉'},
    'koi_smass': {'description': 'Stellar mass (Solar masses)', 'min': 0.1, 'max': 3, 'default': 1.0, 'unit': 'M☉'},
    'koi_slogg': {'description': 'Stellar surface gravity (log g)', 'min': 3.5, 'max': 5.0, 'default': 4.4, 'unit': 'log g'},
    'koi_count': {'description': 'Number of KOIs in system', 'min': 1, 'max': 10, 'default': 1, 'unit': 'count'},
    'koi_num_transits': {'description': 'Number of transits observed', 'min': 1, 'max': 1000, 'default': 50, 'unit': 'count'},
    'koi_max_sngle_ev': {'description': 'Maximum single event statistic', 'min': 1, 'max': 1000, 'default': 10, 'unit': 'σ'},
    'koi_max_mult_ev': {'description': 'Maximum multiple event statistic', 'min': 1, 'max': 5000, 'default': 50, 'unit': 'σ'},
    'impact_parameter': {'description': 'Impact parameter', 'min': 0, 'max': 2, 'default': 0.5, 'unit': 'b'},
    'transit_duration': {'description': 'Transit duration (hours)', 'min': 0.1, 'max': 20, 'default': 3.0, 'unit': 'hours'},
    'st_dist': {'description': 'Stellar distance (parsecs)', 'min': 1, 'max': 5000, 'default': 500, 'unit': 'pc'}
}


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass


class FeatureValidationError(Exception):
    """Custom exception for feature validation errors."""
    pass


def find_models_directory() -> Optional[Path]:
    """
    Find the models directory from multiple possible paths.
    
    Returns:
        Path to models directory or None if not found.
    """
    possible_paths = [
        Path(__file__).parent.parent.parent / "models" / "trained_models",
        Path(__file__).parent.parent.parent / "models" / "optimized_training_results" / "trained_models",
        Path(__file__).parent.parent / "models" / "trained_models",
        Path("models") / "trained_models",
        Path("models") / "optimized_training_results" / "trained_models"
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"✅ Found models directory: {path.absolute()}")
            return path
    
    logger.error("❌ Could not find models directory")
    return None


def load_trained_models() -> Tuple[Dict[str, Any], Any, List[str], Dict[str, Any]]:
    """
    Load all trained models, scaler, features, and metrics.
    
    Returns:
        Tuple of (models_dict, scaler, selected_features, training_metrics)
    
    Raises:
        ModelLoadError: If models cannot be loaded.
    """
    models = {}
    scaler = None
    selected_features = []
    training_metrics = {}
    
    models_dir = find_models_directory()
    if not models_dir:
        raise ModelLoadError("Could not find models directory")
    
    try:
        model_files = {
            "Extra Trees": "extra_trees_20251004_155128.joblib",
            "LightGBM": "lightgbm_20251004_155128.joblib",
            "Optimized Random Forest": "optimized_rf_20251004_155128.joblib",
            "Optimized XGBoost": "optimized_xgb_20251004_155128.joblib"
        }
        
        # Load scaler
        scaler_path = models_dir / "scaler_20251004_155128.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("✅ Loaded scaler successfully")
        else:
            logger.warning("⚠️ Scaler not found, predictions will use unscaled features")
        
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
                logger.info(f"✅ Loaded {len(selected_features)} features")
                break
        
        if not selected_features:
            logger.warning("⚠️ Using default feature list")
            selected_features = FEATURE_NAMES
        
        # Load training metrics
        metrics_paths = [
            models_dir.parent / "training_results_20251004_155128.json",
            models_dir.parent.parent / "training_results_20251004_155128.json",
            Path("models") / "training_results_20251004_155128.json"
        ]
        
        for metrics_path in metrics_paths:
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    training_metrics = json.load(f)
                logger.info(f"✅ Loaded training metrics")
                break
        
        # Load models
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                models[name] = joblib.load(model_path)
                logger.info(f"✅ Loaded {name}")
            else:
                logger.warning(f"⚠️ {name} not found at {model_path}")
        
        if not models:
            raise ModelLoadError("No models could be loaded")
        
        logger.info(f"📊 Successfully loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise ModelLoadError(f"Failed to load models: {str(e)}")
    
    return models, scaler, selected_features, training_metrics


def validate_features(features: List[float]) -> bool:
    """
    Validate feature inputs.
    
    Args:
        features: List of feature values
    
    Returns:
        True if valid
    
    Raises:
        FeatureValidationError: If validation fails
    """
    if not features:
        raise FeatureValidationError("Features list is empty")
    
    if len(features) != len(FEATURE_NAMES):
        raise FeatureValidationError(
            f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
        )
    
    for i, value in enumerate(features):
        if not isinstance(value, (int, float, np.number)):
            raise FeatureValidationError(
                f"Feature {i} ({FEATURE_NAMES[i]}): must be numeric, got {type(value)}"
            )
        
        if np.isnan(value) or np.isinf(value):
            raise FeatureValidationError(
                f"Feature {i} ({FEATURE_NAMES[i]}): contains NaN or Inf"
            )
        
        # Check ranges
        feature_name = FEATURE_NAMES[i]
        if feature_name in FEATURE_INFO:
            min_val = FEATURE_INFO[feature_name]['min']
            max_val = FEATURE_INFO[feature_name]['max']
            
            if value < min_val or value > max_val:
                logger.warning(
                    f"Feature {feature_name} value {value} outside expected range [{min_val}, {max_val}]"
                )
    
    return True


def predict_with_model(
    features: List[float],
    model: Any,
    model_name: str,
    scaler: Optional[Any] = None
) -> Tuple[str, Optional[float]]:
    """
    Make prediction with a single model.
    
    Args:
        features: List of feature values
        model: Trained model
        model_name: Name of the model
        scaler: Optional scaler for feature preprocessing
    
    Returns:
        Tuple of (prediction_label, confidence_score)
    """
    try:
        # Prepare features
        features_array = np.array(features).reshape(1, -1)
        
        # Scale if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Get prediction
        predicted_class = model.predict(features_scaled)
        
        # Convert numeric predictions to labels
        if 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
            prediction_value = predicted_class[0]
            if isinstance(prediction_value, (int, np.integer)):
                prediction = LABEL_MAPPING.get(
                    int(prediction_value),
                    f"Unknown({prediction_value})"
                )
            else:
                prediction = str(prediction_value)
        else:
            prediction = str(predicted_class[0])
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)
            confidence = float(np.max(proba)) * 100
        
        return prediction, confidence
    
    except Exception as e:
        logger.error(f"Prediction error with {model_name}: {str(e)}")
        return "Error", None


def calculate_ensemble_prediction(predictions: Dict[str, str]) -> Tuple[str, float]:
    """
    Calculate ensemble prediction using majority voting.
    
    Args:
        predictions: Dictionary of model predictions
    
    Returns:
        Tuple of (ensemble_prediction, confidence_percentage)
    """
    if not predictions:
        return "UNKNOWN", 0.0
    
    # Count votes (excluding errors)
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


def format_prediction_response(
    predictions: Dict[str, str],
    confidences: Dict[str, float],
    processing_time_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Format prediction results into a standardized response.
    
    Args:
        predictions: Dictionary of model predictions
        confidences: Dictionary of confidence scores
        processing_time_ms: Optional processing time in milliseconds
    
    Returns:
        Formatted response dictionary
    """
    ensemble_pred, ensemble_conf = calculate_ensemble_prediction(predictions)
    
    response = {
        'predictions': predictions,
        'confidence': confidences,
        'ensemble_prediction': ensemble_pred,
        'ensemble_confidence': round(ensemble_conf, 2),
    }
    
    if processing_time_ms is not None:
        response['processing_time_ms'] = round(processing_time_ms, 2)
    
    return response


def get_feature_descriptions() -> List[Dict[str, Any]]:
    """
    Get feature descriptions with metadata.
    
    Returns:
        List of feature information dictionaries
    """
    return [
        {
            'index': i,
            'name': name,
            'description': FEATURE_INFO[name]['description'],
            'unit': FEATURE_INFO[name].get('unit', ''),
            'min': FEATURE_INFO[name]['min'],
            'max': FEATURE_INFO[name]['max'],
            'default': FEATURE_INFO[name]['default']
        }
        for i, name in enumerate(FEATURE_NAMES)
    ]


def get_model_performance_summary() -> Dict[str, Dict[str, str]]:
    """
    Get summary of model performance metrics.
    
    Returns:
        Dictionary of model performance data
    """
    return {
        "Optimized XGBoost": {
            "accuracy": "82.5%",
            "f1_score": "82.2%",
            "precision": "82.3%",
            "recall": "82.1%"
        },
        "LightGBM": {
            "accuracy": "82.1%",
            "f1_score": "81.9%",
            "precision": "82.0%",
            "recall": "82.1%"
        },
        "Optimized Random Forest": {
            "accuracy": "81.4%",
            "f1_score": "81.0%",
            "precision": "81.2%",
            "recall": "80.9%"
        },
        "Extra Trees": {
            "accuracy": "74.7%",
            "f1_score": "74.3%",
            "precision": "74.5%",
            "recall": "74.6%"
        }
    }
