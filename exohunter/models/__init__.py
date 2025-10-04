"""
Machine learning models for exoplanet classification.

This package contains both traditional machine learning models and deep learning
implementations for exoplanet detection and classification.
"""

# Import baseline models
try:
    from .train_baseline import train_random_forest, train_xgboost
except ImportError:
    # Baseline model dependencies not available
    pass

# Import advanced models
try:
    from .advanced import TabularMLP, make_lightcurve_cnn
except ImportError:
    # TensorFlow not available
    pass

# Import ensemble models
try:
    from .ensemble import (
        MLPWrapper, 
        XGBWrapper, 
        build_stacking_ensemble, 
        train_ensemble_suite
    )
except ImportError:
    # Ensemble dependencies not available
    pass

# Import evaluation utilities
try:
    from .evaluate import evaluate_model, compare_models, calculate_metrics
except ImportError:
    # Evaluation dependencies not available
    pass
