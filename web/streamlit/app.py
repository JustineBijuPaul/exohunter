import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sklearn

# Display version info for debugging
st.sidebar.caption(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
st.sidebar.caption(f"scikit-learn: {sklearn.__version__}")

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

def safe_load_model(model_path: Path):
    """Safely load a model with fallback methods for compatibility."""
    try:
        # Method 1: Standard joblib load
        return joblib.load(model_path)
    except (KeyError, pickle.UnpicklingError) as e:
        st.sidebar.warning(f"Standard load failed: {str(e)[:50]}... Trying alternative method")
        try:
            # Method 2: Load with pickle directly
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            st.sidebar.error(f"Alternative load also failed: {str(e2)[:50]}")
            # If all methods fail, provide helpful error
            st.sidebar.error(f"All loading methods failed")
            error_msg = f"Could not load model from {model_path.name}. "
            error_msg += f"This may be due to Python version or scikit-learn version mismatch. "
            error_msg += f"Original error: {str(e)}"
            raise ValueError(error_msg)

@st.cache_resource
def load_models():
    """Load trained models and preprocessors."""
    models = {}
    scaler = None
    selected_features = []
    
    # Initialize session state for prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
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
        st.sidebar.info("💡 Please ensure models are in the correct location")
        return models, scaler, selected_features
    
    try:
        model_files = {
            "Extra Trees": "extra_trees_20251005_200911.joblib",
            "LightGBM": "lightgbm_20251005_200911.joblib", 
            "Optimized Random Forest": "optimized_rf_20251005_200911.joblib",
            "Optimized XGBoost": "optimized_xgb_20251005_200911.joblib"
        }
        
        # Load scaler
        scaler_path = models_dir / "scaler_20251005_200911.joblib"
        if scaler_path.exists():
            try:
                scaler = safe_load_model(scaler_path)
                st.sidebar.success("✅ Loaded scaler")
            except Exception as e:
                st.sidebar.error(f"Failed to load scaler: {str(e)[:100]}")
                st.sidebar.info("Predictions will use unscaled features")
        
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
                try:
                    models[name] = safe_load_model(model_path)
                    st.sidebar.success(f"✅ Loaded {name}")
                except Exception as e:
                    st.sidebar.error(f"⚠️ Failed to load {name}: {str(e)[:50]}")
            else:
                st.sidebar.warning(f"⚠️ {name} not found")
        
        st.sidebar.info(f"📊 Total models loaded: {len(models)}")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.sidebar.error(f"❌ Error loading models")
        st.sidebar.code(error_details, language="python")
    
    return models, scaler, selected_features

def make_predictions(features, models, scaler):
    """Make predictions using loaded models."""
    import time
    start_time = time.time()
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
    
    processing_time = (time.time() - start_time) * 1000  # ms
    return predictions, confidences, processing_time

def main():
    st.set_page_config(
        page_title="ExoHunter - Exoplanet Classifier",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.75rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🌌 ExoHunter</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Exoplanet Classification System</p>', unsafe_allow_html=True)
    
    # Load models
    models, scaler, selected_features = load_models()
    
    if not models:
        st.error("❌ No models could be loaded. Please check the models directory.")
        st.info("💡 Make sure the trained models are in the correct location.")
        with st.expander("🔍 Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            1. Models not found in expected directory
            2. Model files are corrupted
            3. Missing dependencies (joblib, sklearn)
            
            **Solution:**
            - Verify models exist in `models/trained_models/`
            - Re-run training script: `python scripts/optimized_training.py`
            - Check logs for detailed error messages
            """)
        return
    
    # Display model information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Loaded", len(models))
    with col2:
        st.metric("Features", len(FEATURE_NAMES))
    with col3:
        st.metric("Predictions Made", len(st.session_state.get('prediction_history', [])))
    with col4:
        st.metric("Classes", 3)
    
    st.markdown("---")
    
    # Handle sample data loading BEFORE creating widgets
    if 'pending_sample_load' in st.session_state and st.session_state.pending_sample_load:
        sample_name = st.session_state.pending_sample_load
        if sample_name in SAMPLE_DATA:
            sample_values = SAMPLE_DATA[sample_name]
            for i in range(len(FEATURE_NAMES)):
                st.session_state[f"feature_{i}"] = float(sample_values[i])
        st.session_state.pending_sample_load = None
    
    # Initialize session state for features if not exists
    for i, feature_name in enumerate(FEATURE_NAMES):
        if f"feature_{i}" not in st.session_state:
            st.session_state[f"feature_{i}"] = float(FEATURE_INFO[feature_name]['default'])
    
    # Create tabs for different input methods
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Manual Input", "📄 CSV Upload", "📋 Quick Samples", "📊 History"])
    
    # Tab 1: Manual Input
    with tab1:
        st.header("🎯 Enter Exoplanet Features")
        
        features = []
        
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
                    key=f"feature_{i}",
                    format="%.3f"
                )
                features.append(value)
        
        st.markdown("---")
        
        if st.button("🚀 Classify Exoplanet", key="classify_manual", type="primary"):
            with st.spinner("🔍 Running classification models..."):
                predictions, confidences, proc_time = make_predictions(features, models, scaler)
            
            display_prediction_results(predictions, confidences, proc_time, features)
    
    # Tab 2: CSV Upload
    with tab2:
        st.header("📄 Upload CSV for Batch Prediction")
        st.markdown("Upload a CSV file with exoplanet features. Each row should contain 15 features.")
        
        # Download template
        template_df = pd.DataFrame([FEATURE_INFO[f]['default'] for f in FEATURE_NAMES]).T
        template_df.columns = FEATURE_NAMES
        csv_template = template_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="exoplanet_features_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head())
                
                if st.button("🚀 Predict All", key="classify_batch", type="primary"):
                    if len(df.columns) != 15:
                        st.error(f"❌ Expected 15 columns, got {len(df.columns)}")
                    else:
                        progress_bar = st.progress(0)
                        results_list = []
                        
                        for idx, row in df.iterrows():
                            features = row.values.tolist()
                            predictions, confidences, proc_time = make_predictions(features, models, scaler)
                            
                            # Get ensemble prediction
                            prediction_counts = {}
                            for pred in predictions.values():
                                if pred != "Error":
                                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                            
                            if prediction_counts:
                                ensemble_pred = max(prediction_counts, key=prediction_counts.get)
                            else:
                                ensemble_pred = "Unknown"
                            
                            results_list.append({
                                'Row': idx + 1,
                                'Ensemble_Prediction': ensemble_pred,
                                **predictions
                            })
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        results_df = pd.DataFrame(results_list)
                        st.success("✅ Batch prediction complete!")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="� Download Results",
                            data=csv_results,
                            file_name="exoplanet_predictions.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Tab 3: Quick Samples
    with tab3:
        st.header("📋 Quick Test Samples")
        st.markdown("Use these pre-configured samples to quickly test the models:")
        
        cols = st.columns(3)
        
        for idx, (sample_name, sample_values) in enumerate(SAMPLE_DATA.items()):
            with cols[idx]:
                st.subheader(sample_name)
                if st.button(f"Load {sample_name}", key=f"load_{sample_name}"):
                    st.session_state.pending_sample_load = sample_name
                    st.rerun()
                
                if st.button(f"Predict {sample_name}", key=f"predict_{sample_name}", type="primary"):
                    with st.spinner("🔍 Classifying..."):
                        predictions, confidences, proc_time = make_predictions(sample_values, models, scaler)
                    display_prediction_results(predictions, confidences, proc_time, sample_values, sample_name)
    
    # Tab 4: History
    with tab4:
        st.header("📊 Prediction History")
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Visualize prediction distribution
            if len(history_df) > 0:
                st.subheader("📈 Prediction Distribution")
                pred_counts = history_df['Ensemble_Prediction'].value_counts()
                fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                           title="Distribution of Predictions")
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions yet. Make some predictions to see history!")

def display_prediction_results(predictions, confidences, processing_time, features, sample_name=None):
    """Display prediction results with visualizations."""
    st.success("✅ Classification Complete!")
    
    # Processing time
    st.info(f"⏱️ Processing Time: {processing_time:.2f} ms")
    
    st.header("📊 Classification Results")
                
    # Create results dataframe
    results_data = []
    for model_name, prediction in predictions.items():
        confidence = confidences.get(model_name, "N/A")
        if isinstance(confidence, (int, float)):
            confidence_str = f"{confidence:.1f}%"
            confidence_val = confidence
        else:
            confidence_str = "N/A"
            confidence_val = 0
        results_data.append({
            "Model": model_name,
            "Prediction": prediction,
            "Confidence": confidence_str,
            "_confidence_val": confidence_val
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Display results table
    display_df = results_df[["Model", "Prediction", "Confidence"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Ensemble prediction
    prediction_counts = {}
    for pred in predictions.values():
        if pred != "Error":
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    if prediction_counts:
        ensemble_prediction = max(prediction_counts, key=prediction_counts.get)
        ensemble_count = prediction_counts[ensemble_prediction]
        total_valid = sum(prediction_counts.values())
        agreement_ratio = ensemble_count / total_valid if total_valid > 0 else 0
        
        # Visualize confidence with gauge chart
        st.markdown("### 🎯 Ensemble Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"## **{ensemble_prediction}**")
            st.markdown(f"**Agreement:** {ensemble_count}/{total_valid} models ({agreement_ratio*100:.1f}%)")
            
            if agreement_ratio >= 0.75:
                st.success("🎯 High agreement - reliable prediction")
            elif agreement_ratio >= 0.5:
                st.warning("⚠️ Moderate agreement")
            else:
                st.error("❌ Low agreement - uncertain")
        
        with col2:
            # Create gauge chart for confidence
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=agreement_ratio * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Model Agreement"},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "lightgreen"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction distribution chart
        st.markdown("#### 📊 Prediction Distribution")
        pred_df = pd.DataFrame(list(prediction_counts.items()), columns=['Prediction', 'Count'])
        fig = px.bar(pred_df, x='Prediction', y='Count', 
                    title="Model Votes",
                    color='Prediction',
                    color_discrete_map={
                        'CONFIRMED': '#2ecc71',
                        'CANDIDATE': '#f39c12',
                        'FALSE POSITIVE': '#e74c3c'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Add to history
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        history_entry = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Sample': sample_name if sample_name else "Manual Input",
            'Ensemble_Prediction': ensemble_prediction,
            'Agreement': f"{agreement_ratio*100:.1f}%",
            'Processing_Time_ms': f"{processing_time:.2f}"
        }
        st.session_state.prediction_history.append(history_entry)
    
    # Confidence visualization
    if confidences:
        st.markdown("#### 🎲 Model Confidence Scores")
        conf_df = pd.DataFrame([
            {'Model': k, 'Confidence': v} 
            for k, v in confidences.items()
        ])
        fig = px.bar(conf_df, x='Model', y='Confidence',
                    title="Confidence by Model",
                    color='Confidence',
                    color_continuous_scale='Blues')
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature values summary
    with st.expander("📈 Feature Values Summary"):
        feature_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Value': features,
            'Description': [FEATURE_INFO[f]['description'] for f in FEATURE_NAMES]
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    # Model performance info
    with st.expander("🔬 Model Performance Information"):
        st.markdown("**Model Performance (on test data):**")
        model_performance = {
            "Optimized XGBoost": {"accuracy": "82.5%", "f1": "82.2%"},
            "LightGBM": {"accuracy": "82.1%", "f1": "81.9%"},
            "Optimized Random Forest": {"accuracy": "81.4%", "f1": "81.0%"},
            "Extra Trees": {"accuracy": "74.7%", "f1": "74.3%"}
        }
        
        perf_df = pd.DataFrame([
            {'Model': k, 'Accuracy': v['accuracy'], 'F1-Score': v['f1']}
            for k, v in model_performance.items()
        ])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()