"""
Ultimate Ensemble Model API Integration
=======================================
Adds /predict/ensemble endpoint to FastAPI for the ultimate ensemble model
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class AdvancedFeatureEngineering:
    """Advanced feature engineering for exoplanet data - copied from training script"""
    
    def __init__(self):
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features - EXACT COPY from train_ultimate_ensemble.py"""
        df = df.copy()
        features_created = 0
        
        # Original features
        base_features = ['orbital_period', 'transit_depth', 'planet_radius', 
                        'koi_teq', 'koi_insol', 'stellar_teff', 'stellar_radius',
                        'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
                        'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter',
                        'transit_duration', 'koi_dor', 'st_tmag', 'st_logg', 
                        'st_dist', 'st_mass']
        
        # 1. Ratio features (physical relationships)
        if 'planet_radius' in df.columns and 'stellar_radius' in df.columns:
            df['radius_ratio'] = df['planet_radius'] / (df['stellar_radius'] + 1e-10)
            features_created += 1
        
        if 'transit_depth' in df.columns and 'planet_radius' in df.columns:
            df['depth_radius_ratio'] = df['transit_depth'] / (df['planet_radius']**2 + 1e-10)
            features_created += 1
        
        if 'orbital_period' in df.columns and 'transit_duration' in df.columns:
            df['period_duration_ratio'] = df['orbital_period'] / (df['transit_duration'] + 1e-10)
            features_created += 1
        
        if 'koi_teq' in df.columns and 'stellar_teff' in df.columns:
            df['temp_ratio'] = df['koi_teq'] / (df['stellar_teff'] + 1e-10)
            features_created += 1
        
        # 2. Derived physical quantities
        if 'orbital_period' in df.columns and 'koi_smass' in df.columns:
            df['semi_major_axis'] = (df['orbital_period'] * df['koi_smass']**(1/3)) ** (2/3)
            features_created += 1
        
        if 'planet_radius' in df.columns and 'koi_teq' in df.columns:
            df['planet_energy'] = df['planet_radius']**2 * df['koi_teq']**4
            features_created += 1
        
        # 3. Detection quality features
        if 'koi_max_sngle_ev' in df.columns and 'koi_max_mult_ev' in df.columns:
            df['snr_ratio'] = df['koi_max_mult_ev'] / (df['koi_max_sngle_ev'] + 1e-10)
            df['snr_product'] = df['koi_max_sngle_ev'] * df['koi_max_mult_ev']
            df['snr_diff'] = df['koi_max_mult_ev'] - df['koi_max_sngle_ev']
            features_created += 3
        
        if 'koi_num_transits' in df.columns and 'orbital_period' in df.columns:
            df['transits_per_day'] = df['koi_num_transits'] / (df['orbital_period'] + 1e-10)
            features_created += 1
        
        # 4. Polynomial features for key variables
        for feat in ['transit_depth', 'planet_radius', 'orbital_period']:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
                df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
                df[f'{feat}_log'] = np.log1p(np.abs(df[feat]))
                features_created += 3
        
        # 5. Interaction features
        if 'impact_parameter' in df.columns and 'transit_duration' in df.columns:
            df['impact_duration'] = df['impact_parameter'] * df['transit_duration']
            features_created += 1
        
        if 'koi_insol' in df.columns and 'planet_radius' in df.columns:
            df['habitability_index'] = df['koi_insol'] / (df['planet_radius']**2 + 1e-10)
            features_created += 1
        
        # 6. Statistical features - normalize ALL numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['disposition_encoded'] and not col.endswith('_normalized'):
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
        
        return df


class UltimateEnsemblePredictor:
    """Wrapper for ultimate ensemble model predictions"""
    
    def __init__(self, model_dir: Path):
        """Load all models and preprocessors"""
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.feature_engineer = AdvancedFeatureEngineering()
        
        self._load_models()
    
    def _load_models(self):
        """Load all ensemble models"""
        print("Loading ultimate ensemble models...")
        
        try:
            # Load tree-based models
            self.models['xgboost'] = joblib.load(self.model_dir / 'ultimate_ensemble_xgboost.pkl')
            self.models['lightgbm'] = joblib.load(self.model_dir / 'ultimate_ensemble_lightgbm.pkl')
            self.models['catboost'] = joblib.load(self.model_dir / 'ultimate_ensemble_catboost.pkl')
            self.models['random_forest'] = joblib.load(self.model_dir / 'ultimate_ensemble_random_forest.pkl')
            self.models['extra_trees'] = joblib.load(self.model_dir / 'ultimate_ensemble_extra_trees.pkl')
            
            # Load deep learning model
            try:
                from tensorflow import keras
                self.models['deep_nn'] = keras.models.load_model(self.model_dir / 'ultimate_ensemble_deep_nn.h5')
                print("✓ Loaded deep neural network")
            except Exception as e:
                print(f"⚠️  Deep NN not loaded: {e}")
            
            # Load preprocessing objects
            self.scaler = joblib.load(self.model_dir / 'ultimate_ensemble_scaler.pkl')
            self.encoder = joblib.load(self.model_dir / 'ultimate_ensemble_encoder.pkl')
            
            # Load feature names
            feature_names_path = self.model_dir / 'ultimate_ensemble_feature_names.pkl'
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                print(f"✓ Loaded {len(self.feature_names)} feature names")
            
            print(f"✓ Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def preprocess_input(self, data: Dict[str, float]) -> np.ndarray:
        """Preprocess input data to match training format"""
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Fill missing base features
        numeric_cols = ['orbital_period', 'transit_depth', 'planet_radius', 
                       'koi_teq', 'koi_insol', 'stellar_teff', 'stellar_radius',
                       'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
                       'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter',
                       'transit_duration', 'koi_dor', 'st_tmag', 'st_logg', 
                       'st_dist', 'st_mass']
        
        # Keep only available columns
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        # Fill missing values with 0 (median will be computed during feature engineering)
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df)
        
        # Select features using saved feature names
        if self.feature_names is not None:
            # Add missing features as zeros
            for feat in self.feature_names:
                if feat not in df_features.columns:
                    df_features[feat] = 0
            X = df_features[self.feature_names]
        else:
            # Fallback: select all numeric columns
            feature_cols = [col for col in df_features.columns 
                           if df_features[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
            X = df_features[feature_cols]
        
        # Fill any remaining NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X.values
    
    def predict(self, data: Dict[str, float], use_xgboost_only: bool = False) -> Dict[str, Any]:
        """Make prediction with confidence scores"""
        
        # Preprocess input
        X = self.preprocess_input(data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        if use_xgboost_only and 'xgboost' in self.models:
            # Use only XGBoost (best performer at 78.22%)
            probs = self.models['xgboost'].predict_proba(X_scaled)[0]
            model_used = "XGBoost"
        else:
            # Ensemble predictions
            all_probs = []
            for name, model in self.models.items():
                if name == 'deep_nn':
                    # Skip Deep NN as it performs poorly (18.81%)
                    continue
                probs_model = model.predict_proba(X_scaled)[0]
                all_probs.append(probs_model)
            
            # Average predictions
            probs = np.mean(all_probs, axis=0)
            model_used = f"Ensemble ({len(all_probs)} models)"
        
        # Get prediction
        pred_idx = np.argmax(probs)
        prediction = self.encoder.classes_[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Create probability dict
        probabilities = {
            str(cls): float(prob) 
            for cls, prob in zip(self.encoder.classes_, probs)
        }
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": model_used,
            "all_probabilities": probabilities
        }
    
    def predict_batch(self, data_list: List[Dict[str, float]], use_xgboost_only: bool = False) -> List[Dict[str, Any]]:
        """Make predictions for multiple samples"""
        return [self.predict(data, use_xgboost_only) for data in data_list]


# Endpoint code to add to main.py
ENDPOINT_CODE = '''
# Ultimate Ensemble Model Endpoint
from pathlib import Path
import sys

# Import predictor
sys.path.append(str(Path(__file__).parent.parent.parent))
from web.api.ensemble_integration import UltimateEnsemblePredictor

# Global predictor instance
ultimate_predictor = None

def load_ultimate_ensemble():
    """Load ultimate ensemble model"""
    global ultimate_predictor
    try:
        models_dir = Path(__file__).parent.parent.parent / "models"
        ultimate_predictor = UltimateEnsemblePredictor(models_dir)
        logger.info("✅ Ultimate ensemble loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load ultimate ensemble: {e}")

@app.post("/predict/ensemble", response_model=PredictionResponse, tags=["Predictions"])
async def predict_ensemble(
    request: PredictionRequest,
    use_xgboost_only: bool = False
) -> PredictionResponse:
    """
    Predict exoplanet disposition using ultimate ensemble model.
    
    Args:
        request: Prediction request with features
        use_xgboost_only: If True, use only XGBoost model (best individual performer)
    
    Returns:
        Prediction response with class, confidence, and probabilities
    """
    try:
        if ultimate_predictor is None:
            raise HTTPException(status_code=503, detail="Ultimate ensemble model not loaded")
        
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        result = ultimate_predictor.predict(input_data, use_xgboost_only)
        
        # Format response
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version="ultimate_ensemble_v1",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=0.0  # Can add timing if needed
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ensemble/batch", tags=["Predictions"])
async def predict_ensemble_batch(
    requests: List[PredictionRequest],
    use_xgboost_only: bool = False
) -> List[PredictionResponse]:
    """
    Batch prediction endpoint for multiple samples.
    
    Args:
        requests: List of prediction requests
        use_xgboost_only: If True, use only XGBoost model
    
    Returns:
        List of prediction responses
    """
    try:
        if ultimate_predictor is None:
            raise HTTPException(status_code=503, detail="Ultimate ensemble model not loaded")
        
        # Convert requests to list of dicts
        input_data = [req.dict() for req in requests]
        
        # Make predictions
        results = ultimate_predictor.predict_batch(input_data, use_xgboost_only)
        
        # Format responses
        responses = []
        for result in results:
            responses.append(PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                model_version="ultimate_ensemble_v1",
                timestamp=datetime.now().isoformat(),
                processing_time_ms=0.0
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add to startup event
@app.on_event("startup")
async def startup_ultimate_ensemble():
    """Load ultimate ensemble on startup"""
    load_ultimate_ensemble()
'''

if __name__ == "__main__":
    # Test the predictor
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    try:
        predictor = UltimateEnsemblePredictor(models_dir)
        
        # Test prediction
        sample_data = {
            'orbital_period': 7.0,
            'transit_depth': 81.6,
            'planet_radius': 0.92,
            'koi_teq': 974.0,
            'koi_insol': 212.66,
            'stellar_teff': 5977.0,
            'stellar_radius': 1.022,
            'koi_smass': 1.133,
            'koi_slogg': 4.473,
            'koi_count': 1.0,
            'koi_num_transits': 192.0,
            'koi_max_sngle_ev': 3.149819,
            'koi_max_mult_ev': 8.076329,
            'impact_parameter': 0.098,
            'transit_duration': 2.853,
            'koi_dor': 18.9,
            'st_tmag': 0.0,
            'st_logg': 0.0,
            'st_dist': 0.0,
            'st_mass': 0.0
        }
        
        print("\n" + "="*70)
        print("TESTING ULTIMATE ENSEMBLE PREDICTOR")
        print("="*70)
        
        # Test ensemble
        result_ensemble = predictor.predict(sample_data, use_xgboost_only=False)
        print(f"\n🔮 Ensemble Prediction:")
        print(f"   Class: {result_ensemble['prediction']}")
        print(f"   Confidence: {result_ensemble['confidence']:.2%}")
        print(f"   Model: {result_ensemble['model_used']}")
        print(f"   Probabilities:")
        for cls, prob in result_ensemble['probabilities'].items():
            print(f"      {cls}: {prob:.2%}")
        
        # Test XGBoost only
        result_xgb = predictor.predict(sample_data, use_xgboost_only=True)
        print(f"\n⭐ XGBoost Only Prediction:")
        print(f"   Class: {result_xgb['prediction']}")
        print(f"   Confidence: {result_xgb['confidence']:.2%}")
        print(f"   Probabilities:")
        for cls, prob in result_xgb['probabilities'].items():
            print(f"      {cls}: {prob:.2%}")
        
        print("\n✅ Predictor test successful!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
