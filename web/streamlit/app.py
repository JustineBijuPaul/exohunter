"""
ExoHunter Streamlit Demo App
============================

A comprehensive demo application for exoplanet classification using machine learning.
This app provides an interactive interface for dataset upload, model predictions,
and evaluation metrics visualization.

Usage:
    streamlit run web/streamlit/app.py

Author: ExoHunter Team
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import joblib
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from exohunter.models.ensemble import train_ensemble_suite
    from exohunter.models.evaluate import evaluate_model
    from exohunter.data.labels import map_labels
except ImportError as e:
    st.error(f"Could not import ExoHunter modules: {e}")
    st.info("Please ensure you're running from the project root directory")

# Configure Streamlit page
st.set_page_config(
    page_title="ExoHunter Demo",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3c72;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2a5298;
    }
    .stAlert > div {
        background-color: #e3f2fd;
        color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

def load_models():
    """Load available pre-trained models."""
    models = {}
    models_dir = project_root / "models"
    
    if models_dir.exists():
        # Try to load different model types
        model_files = {
            "Ensemble": "stacking_ensemble.pkl",
            "XGBoost": "xgboost_model.pkl", 
            "Random Forest": "random_forest_model.pkl"
        }
        
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        models[name] = pickle.load(f)
                except Exception as e:
                    st.warning(f"Could not load {name} model: {e}")
    
    return models

def generate_sample_data(n_samples=1000):
    """Generate synthetic exoplanet data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic exoplanet features
    data = {
        'period': np.random.lognormal(1, 1, n_samples),  # Orbital period (days)
        'radius': np.random.lognormal(0, 0.5, n_samples),  # Planet radius (Earth radii)
        'temperature': np.random.normal(5800, 800, n_samples),  # Stellar temperature (K)
        'magnitude': np.random.normal(12, 2, n_samples),  # Stellar magnitude
        'snr': np.random.gamma(2, 5, n_samples),  # Signal-to-noise ratio
        'duration': np.random.gamma(2, 1.5, n_samples),  # Transit duration (hours)
        'depth': np.random.gamma(1, 0.01, n_samples),  # Transit depth
        'impact': np.random.uniform(0, 1, n_samples)  # Impact parameter
    }
    
    # Generate labels based on features (simplified logic)
    labels = []
    for i in range(n_samples):
        if data['snr'][i] > 15 and data['depth'][i] > 0.005 and data['radius'][i] > 0.5:
            if np.random.random() > 0.3:
                labels.append('CANDIDATE')
            else:
                labels.append('CONFIRMED')
        else:
            labels.append('FALSE POSITIVE')
    
    data['label'] = labels
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess uploaded data for model training/prediction."""
    if 'label' in df.columns:
        # Training data with labels
        X = df.drop('label', axis=1)
        y = df['label']
        return X, y
    else:
        # Prediction data without labels
        return df, None

def train_model_on_data(X, y, model_type="XGBoost"):
    """Train a model on the uploaded data."""
    if model_type == "XGBoost":
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        # Use XGBoost for this demo
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == "Ensemble":
        # For demo purposes, use a simple voting ensemble
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from xgboost import XGBClassifier
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        xgb = XGBClassifier(n_estimators=50, random_state=42)
        
        model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
    
    # Train the model
    model.fit(X, y)
    return model

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """Evaluate model predictions and return metrics."""
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        results['probabilities'] = y_pred_proba
    
    return results

def plot_confusion_matrix(cm, labels):
    """Create an interactive confusion matrix plot."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        width=500,
        height=400
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance if available."""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'estimators_'):
            # For ensemble models, average the importance
            importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        else:
            return None
        
        fig = px.bar(
            x=importance,
            y=feature_names,
            orientation='h',
            title="Feature Importance",
            labels={'x': 'Importance', 'y': 'Features'}
        )
        fig.update_layout(height=400)
        return fig
    except:
        return None

def plot_data_distribution(df):
    """Create distribution plots for the dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # Show first 4 numeric columns
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=numeric_cols
    )
    
    for i, col in enumerate(numeric_cols):
        row = i // 2 + 1
        col_idx = i % 2 + 1
        
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title="Feature Distributions",
        height=500
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌟 ExoHunter Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Exoplanet Classification Prototype")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Demo with Sample Data", "Upload Your Dataset", "Model Comparison"]
    )
    
    if mode == "Demo with Sample Data":
        st.sidebar.markdown("---")
        n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000, 100)
        
        if st.sidebar.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                st.session_state.data = generate_sample_data(n_samples)
                st.success(f"Generated {n_samples} sample exoplanet observations!")
    
    elif mode == "Upload Your Dataset":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with exoplanet features"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("Dataset uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Main content area
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Dataset overview
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - (1 if 'label' in df.columns else 0))
        with col3:
            if 'label' in df.columns:
                st.metric("Classes", df['label'].nunique())
            else:
                st.metric("Classes", "N/A")
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data distributions
        st.subheader("📈 Feature Distributions")
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            fig_dist = plot_data_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Label distribution (if available)
        if 'label' in df.columns:
            st.subheader("🏷️ Class Distribution")
            label_counts = df['label'].value_counts()
            fig_labels = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Distribution of Classes"
            )
            st.plotly_chart(fig_labels, use_container_width=True)
        
        # Model training/prediction section
        st.header("🤖 Model Training & Prediction")
        
        if 'label' in df.columns:
            # Training mode
            col1, col2 = st.columns([2, 1])
            
            with col2:
                model_type = st.selectbox(
                    "Select Model Type",
                    ["XGBoost", "Random Forest", "Ensemble"]
                )
                
                test_size = st.slider("Test Split (%)", 10, 50, 20) / 100
                
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            X, y = preprocess_data(df)
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, stratify=y
                            )
                            
                            # Train model
                            model = train_model_on_data(X_train, y_train, model_type)
                            st.session_state.model = model
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                            
                            # Evaluate
                            results = evaluate_predictions(y_test, y_pred, y_pred_proba)
                            st.session_state.evaluation_results = results
                            st.session_state.predictions = {
                                'y_test': y_test,
                                'y_pred': y_pred,
                                'X_test': X_test
                            }
                            
                            st.success(f"Model trained successfully! Accuracy: {results['accuracy']:.3f}")
                            
                        except Exception as e:
                            st.error(f"Error training model: {e}")
            
            with col1:
                if st.session_state.evaluation_results is not None:
                    results = st.session_state.evaluation_results
                    
                    # Metrics
                    st.subheader("📊 Model Performance")
                    col1a, col1b, col1c = st.columns(3)
                    
                    with col1a:
                        st.metric("Accuracy", f"{results['accuracy']:.3f}")
                    with col1b:
                        precision = np.mean([results['classification_report'][label]['precision'] 
                                           for label in results['classification_report'] 
                                           if label not in ['accuracy', 'macro avg', 'weighted avg']])
                        st.metric("Avg Precision", f"{precision:.3f}")
                    with col1c:
                        recall = np.mean([results['classification_report'][label]['recall'] 
                                        for label in results['classification_report'] 
                                        if label not in ['accuracy', 'macro avg', 'weighted avg']])
                        st.metric("Avg Recall", f"{recall:.3f}")
                    
                    # Confusion Matrix
                    st.subheader("🔀 Confusion Matrix")
                    labels = list(set(st.session_state.predictions['y_test']))
                    fig_cm = plot_confusion_matrix(results['confusion_matrix'], labels)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Feature importance
                    if st.session_state.model is not None:
                        st.subheader("🎯 Feature Importance")
                        fig_importance = plot_feature_importance(st.session_state.model, X.columns)
                        if fig_importance:
                            st.plotly_chart(fig_importance, use_container_width=True)
                        else:
                            st.info("Feature importance not available for this model type")
        
        else:
            # Prediction mode (no labels in data)
            st.info("No labels detected in dataset. Upload a pre-trained model or add labels for training.")
            
            # Load pre-trained models
            models = load_models()
            if models:
                model_name = st.selectbox("Select Pre-trained Model", list(models.keys()))
                
                if st.button("Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        try:
                            model = models[model_name]
                            X, _ = preprocess_data(df)
                            predictions = model.predict(X)
                            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                            
                            # Add predictions to dataframe
                            df_with_predictions = df.copy()
                            df_with_predictions['Predicted_Label'] = predictions
                            
                            if probabilities is not None:
                                for i, class_name in enumerate(model.classes_):
                                    df_with_predictions[f'Prob_{class_name}'] = probabilities[:, i]
                            
                            st.subheader("🔮 Predictions")
                            st.dataframe(df_with_predictions.head(20), use_container_width=True)
                            
                            # Prediction distribution
                            pred_counts = pd.Series(predictions).value_counts()
                            fig_pred = px.pie(
                                values=pred_counts.values,
                                names=pred_counts.index,
                                title="Predicted Class Distribution"
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {e}")
            else:
                st.warning("No pre-trained models found. Please train a model first.")
    
    elif mode == "Model Comparison":
        st.header("🏆 Model Comparison")
        st.info("This feature compares different model types on the same dataset.")
        
        if st.button("Run Model Comparison"):
            with st.spinner("Generating data and training models..."):
                # Generate comparison data
                data = generate_sample_data(1000)
                X, y = preprocess_data(data)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                models = {}
                results = {}
                
                # Train different models
                for model_type in ["Random Forest", "XGBoost"]:
                    try:
                        model = train_model_on_data(X_train, y_train, model_type)
                        models[model_type] = model
                        
                        y_pred = model.predict(X_test)
                        results[model_type] = evaluate_predictions(y_test, y_pred)
                    except Exception as e:
                        st.warning(f"Could not train {model_type}: {e}")
                
                # Display comparison
                if results:
                    comparison_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [results[model]['accuracy'] for model in results],
                        'Precision': [np.mean([results[model]['classification_report'][label]['precision'] 
                                             for label in results[model]['classification_report'] 
                                             if label not in ['accuracy', 'macro avg', 'weighted avg']]) 
                                    for model in results],
                        'Recall': [np.mean([results[model]['classification_report'][label]['recall'] 
                                          for label in results[model]['classification_report'] 
                                          if label not in ['accuracy', 'macro avg', 'weighted avg']]) 
                                 for model in results]
                    })
                    
                    st.subheader("📊 Model Comparison Results")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Comparison chart
                    fig_comparison = px.bar(
                        comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                        x='Model',
                        y='Score',
                        color='Metric',
                        title="Model Performance Comparison",
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
    
    else:
        # Welcome screen
        st.header("🚀 Welcome to ExoHunter Demo")
        st.markdown("""
        This interactive demo allows you to:
        
        1. **📊 Explore Data**: Upload your own exoplanet datasets or use generated sample data
        2. **🤖 Train Models**: Train machine learning models on your data
        3. **🔮 Make Predictions**: Classify exoplanets as candidates, confirmed, or false positives
        4. **📈 Visualize Results**: See confusion matrices, feature importance, and performance metrics
        5. **🏆 Compare Models**: Evaluate different algorithms side-by-side
        
        ### Getting Started:
        - Select a mode from the sidebar
        - Upload your data or generate sample data
        - Train a model and explore the results!
        
        ### Supported Features:
        - **Period**: Orbital period (days)
        - **Radius**: Planet radius (Earth radii)  
        - **Temperature**: Stellar temperature (K)
        - **Magnitude**: Stellar magnitude
        - **SNR**: Signal-to-noise ratio
        - **Duration**: Transit duration (hours)
        - **Depth**: Transit depth
        - **Impact**: Impact parameter
        """)
        
        # Quick start button
        if st.button("🎲 Quick Start with Sample Data", type="primary"):
            st.session_state.data = generate_sample_data(1000)
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "🌟 **ExoHunter Demo** | Built with Streamlit | "
        "For the full experience, try the React frontend at `web/frontend/`"
    )

if __name__ == "__main__":
    main()
