import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any

# Feature names for the optimized models
FEATURE_NAMES = [
    'transit_depth', 'planet_radius', 'koi_teq', 'koi_insol', 'stellar_teff',
    'stellar_radius', 'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
    'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter', 'transit_duration', 'st_dist'
]

# Feature descriptions and typical ranges
FEATURE_INFO = {
    'transit_depth': {'description': 'Transit depth (ppm)', 'min': 0, 'max': 100000, 'default': 1000},
    'planet_radius': {'description': 'Planet radius (Earth radii)', 'min': 0.1, 'max': 50, 'default': 2.5},
    'koi_teq': {'description': 'Equilibrium temperature (K)', 'min': 200, 'max': 3000, 'default': 800},
    'koi_insol': {'description': 'Insolation flux (Earth flux)', 'min': 0.1, 'max': 10000, 'default': 100},
    'stellar_teff': {'description': 'Stellar effective temperature (K)', 'min': 3000, 'max': 8000, 'default': 5800},
    'stellar_radius': {'description': 'Stellar radius (Solar radii)', 'min': 0.1, 'max': 5, 'default': 1.0},
    'koi_smass': {'description': 'Stellar mass (Solar masses)', 'min': 0.1, 'max': 3, 'default': 1.0},
    'koi_slogg': {'description': 'Stellar surface gravity (log g)', 'min': 3.5, 'max': 5.0, 'default': 4.4},
    'koi_count': {'description': 'Number of KOIs in system', 'min': 1, 'max': 10, 'default': 1},
    'koi_num_transits': {'description': 'Number of transits observed', 'min': 1, 'max': 1000, 'default': 50},
    'koi_max_sngle_ev': {'description': 'Maximum single event statistic', 'min': 1, 'max': 1000, 'default': 10},
    'koi_max_mult_ev': {'description': 'Maximum multiple event statistic', 'min': 1, 'max': 5000, 'default': 50},
    'impact_parameter': {'description': 'Impact parameter', 'min': 0, 'max': 2, 'default': 0.5},
    'transit_duration': {'description': 'Transit duration (hours)', 'min': 0.1, 'max': 20, 'default': 3.0},
    'st_dist': {'description': 'Stellar distance (parsecs)', 'min': 1, 'max': 5000, 'default': 500}
}

# Sample data for quick testing
SAMPLE_DATA = {
    "Confirmed Exoplanet": [
        1500, 2.5, 800, 150, 5800, 1.0, 1.0, 4.4, 1, 100, 15, 75, 0.3, 3.5, 400
    ],
    "Candidate": [
        800, 1.8, 600, 50, 5200, 0.8, 0.9, 4.5, 2, 30, 8, 25, 0.7, 2.8, 300
    ],
    "False Positive": [
        5000, 8.0, 1200, 500, 6200, 1.2, 1.1, 4.3, 1, 200, 50, 300, 1.2, 5.0, 600
    ]
}

@st.cache_resource
def load_models():
    """Load trained models and preprocessors."""
    models = {}
    scaler = None
    selected_features = []
    
    # Try multiple possible paths for the models
    possible_paths = [
        Path(__file__).parent.parent.parent / "models" / "trained_models",
        Path(__file__).parent.parent / "models" / "trained_models",
        Path("models") / "trained_models",
        Path("../models/trained_models"),
        Path("../../models/trained_models")
    ]
    
    models_dir = None
    for path in possible_paths:
        if path.exists():
            models_dir = path
            st.sidebar.success(f"✅ Found models in: {models_dir}")
            break
    
    if not models_dir:
        st.sidebar.error("❌ Could not find models directory")
        return models, scaler, selected_features
    
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
            st.sidebar.success("✅ Loaded scaler")
        
        # Load selected features
        features_paths = [
            models_dir.parent / "selected_features_20251004_155128.json",
            Path("models") / "selected_features_20251004_155128.json"
        ]
        
        for features_path in features_paths:
            if features_path.exists():
                with open(features_path, 'r') as f:
                    selected_features = json.load(f)
                st.sidebar.success(f"✅ Loaded {len(selected_features)} features")
                break
        
        # Load models
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                models[name] = joblib.load(model_path)
                st.sidebar.success(f"✅ Loaded {name}")
            else:
                st.sidebar.warning(f"⚠️ {name} not found")
        
        st.sidebar.info(f"📊 Total models loaded: {len(models)}")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
    
    return models, scaler, selected_features

def make_predictions(features, models, scaler):
    """Make predictions using loaded models."""
    predictions = {}
    confidences = {}
    
    # Prepare features
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features if scaler is available
    if scaler is not None:
        features_scaled = scaler.transform(features_array)
    else:
        features_scaled = features_array
    
    # Define label mapping for encoded predictions
    label_mapping = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}
    
    # Make predictions with each model
    for model_name, model in models.items():
        try:
            # Get prediction
            predicted_class = model.predict(features_scaled)
            
            # Convert numeric predictions to class names for XGBoost and LightGBM
            if 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
                # These models return encoded labels, convert to class names
                prediction_value = predicted_class[0]
                if isinstance(prediction_value, (int, np.integer)):
                    predictions[model_name] = label_mapping.get(int(prediction_value), f"Unknown({prediction_value})")
                else:
                    predictions[model_name] = str(prediction_value)
            else:
                # Random Forest and Extra Trees return string labels directly
                predictions[model_name] = str(predicted_class[0])
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)
                # Convert to percentage
                confidence_score = float(np.max(proba)) * 100
                confidences[model_name] = confidence_score
            
        except Exception as e:
            st.error(f"Error with {model_name}: {str(e)}")
            predictions[model_name] = "Error"
    
    return predictions, confidences

def main():
    st.set_page_config(
        page_title="ExoHunter - Standalone Exoplanet Classifier",
        page_icon="🌌",
        layout="wide"
    )
    
    st.title("🌌 ExoHunter - Exoplanet Classification")
    st.markdown("*Classify exoplanets using optimized machine learning models*")
    st.markdown("**Standalone Version** - Models loaded directly (no API required)")
    
    # Load models
    models, scaler, selected_features = load_models()
    
    if not models:
        st.error("❌ No models could be loaded. Please check the models directory.")
        st.info("💡 Make sure the trained models are in the correct location.")
        return
    
    # Sidebar for sample data
    with st.sidebar:
        st.header("📋 Quick Test Samples")
        st.markdown("Use these sample data to quickly test the models:")
        
        for sample_name, sample_values in SAMPLE_DATA.items():
            if st.button(f"Load {sample_name}", key=f"load_{sample_name}"):
                for i, feature in enumerate(FEATURE_NAMES):
                    st.session_state[f"feature_{i}"] = sample_values[i]
                st.success(f"Loaded {sample_name} sample data!")
                st.rerun()
    
    st.header("🎯 Exoplanet Feature Input")
    
    # Feature input form
    features = []
    
    # Initialize session state for features if not exists
    for i, feature_name in enumerate(FEATURE_NAMES):
        if f"feature_{i}" not in st.session_state:
            st.session_state[f"feature_{i}"] = float(FEATURE_INFO[feature_name]['default'])
    
    # Create input fields in a nice layout
    col1, col2, col3 = st.columns(3)
    
    for i, feature_name in enumerate(FEATURE_NAMES):
        info = FEATURE_INFO[feature_name]
        col = [col1, col2, col3][i % 3]
        
        with col:
            value = st.number_input(
                label=f"{feature_name}",
                help=info['description'],
                min_value=float(info['min']),
                max_value=float(info['max']),
                value=float(st.session_state[f"feature_{i}"]),  # Ensure float type
                key=f"feature_{i}",
                format="%.3f"
            )
            features.append(value)
    
    st.markdown("---")
    
    # Prediction section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Classify Exoplanet", type="primary"):
            with st.spinner("🔍 Running classification models..."):
                predictions, confidences = make_predictions(features, models, scaler)
            
            if predictions:
                st.success("✅ Classification Complete!")
                
                # Display results
                st.header("📊 Classification Results")
                
                # Create results dataframe
                results_data = []
                for model_name, prediction in predictions.items():
                    confidence = confidences.get(model_name, "N/A")
                    if isinstance(confidence, float):
                        confidence = f"{confidence:.1f}%"  # Format as percentage with 1 decimal
                    results_data.append({
                        "Model": model_name,
                        "Prediction": prediction,
                        "Confidence": confidence
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Ensemble prediction (most common prediction)
                if len(predictions) > 1:
                    prediction_counts = {}
                    for pred in predictions.values():
                        if pred != "Error":
                            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                    
                    if prediction_counts:
                        ensemble_prediction = max(prediction_counts, key=prediction_counts.get)
                        ensemble_count = prediction_counts[ensemble_prediction]
                        total_valid = sum(prediction_counts.values())
                        
                        st.markdown("### 🎯 Ensemble Prediction")
                        st.markdown(f"**{ensemble_prediction}** ({ensemble_count}/{total_valid} models agree)")
                        
                        # Progress bar for agreement
                        agreement_ratio = ensemble_count / total_valid if total_valid > 0 else 0
                        st.progress(agreement_ratio)
                        
                        # Interpretation
                        if agreement_ratio >= 0.75:
                            st.success("🎯 High model agreement - reliable prediction")
                        elif agreement_ratio >= 0.5:
                            st.warning("⚠️ Moderate model agreement - consider additional analysis")
                        else:
                            st.error("❌ Low model agreement - prediction uncertain")
                        
                        # Show prediction distribution
                        st.markdown("#### Prediction Distribution")
                        for pred, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / total_valid) * 100
                            st.markdown(f"- **{pred}**: {count} models ({percentage:.1f}%)")
                
                # Feature importance display
                with st.expander("📈 Feature Values Summary"):
                    feature_df = pd.DataFrame({
                        'Feature': FEATURE_NAMES,
                        'Value': features,
                        'Description': [FEATURE_INFO[f]['description'] for f in FEATURE_NAMES]
                    })
                    st.dataframe(feature_df, use_container_width=True)
                
                # Model performance info
                with st.expander("🔬 Model Information"):
                    st.markdown("**Model Performance (on test data):**")
                    model_performance = {
                        "Optimized XGBoost": "82.5% accuracy",
                        "LightGBM": "82.1% accuracy", 
                        "Optimized Random Forest": "81.4% accuracy",
                        "Extra Trees": "74.7% accuracy"
                    }
                    
                    for model, performance in model_performance.items():
                        st.markdown(f"- **{model}**: {performance}")
            else:
                st.error("❌ No predictions could be made")

if __name__ == "__main__":
    main()