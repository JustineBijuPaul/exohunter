from pathlib import Path

# Configuration settings for the application
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    API_URL = "http://localhost:8000"  # URL for the FastAPI backend
    MODEL_PATH = BASE_DIR / "models" / "trained_models"  # Path to the trained models directory
    SCALER_PATH = MODEL_PATH / "scaler_20251004_155128.joblib"
    EXTRA_TREES_MODEL_PATH = MODEL_PATH / "extra_trees_20251004_155128.joblib"
    LIGHTGBM_MODEL_PATH = MODEL_PATH / "lightgbm_20251004_155128.joblib"
    OPTIMIZED_RF_MODEL_PATH = MODEL_PATH / "optimized_rf_20251004_155128.joblib"
    OPTIMIZED_XGB_MODEL_PATH = MODEL_PATH / "optimized_xgb_20251004_155128.joblib"