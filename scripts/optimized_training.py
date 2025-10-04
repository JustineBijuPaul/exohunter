"""
Optimized Model Training with Cleaned Data

This script trains improved models using the cleaned and feature-engineered datasets
to achieve better accuracy and efficiency in exoplanet classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline

# XGBoost and other advanced models
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the cleaned dataset for training."""
    print(f"Loading cleaned data from {data_path}...")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    # Prepare features and target
    target_col = 'disposition'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Select relevant features for training
    feature_cols = [
        'orbital_period', 'transit_depth', 'planet_radius', 'koi_teq', 'koi_insol',
        'stellar_teff', 'stellar_radius', 'koi_smass', 'koi_slogg', 'koi_count',
        'koi_num_transits', 'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter',
        'transit_duration', 'koi_dor', 'st_tmag', 'st_logg', 'st_dist',
        'ra', 'dec', 'orbital_period_log', 'stellar_density'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Available features: {len(available_features)}")
    
    # Prepare feature matrix
    X = df[available_features].copy()
    
    # Handle any remaining missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col].fillna(X[col].median(), inplace=True)
    
    # Prepare target variable
    y = df[target_col].copy()
    
    # Simplify target classes for better training
    # Group similar classes together
    class_mapping = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE', 
        'FALSE POSITIVE': 'FALSE POSITIVE',
        'FP': 'FALSE POSITIVE',
        'PC': 'CANDIDATE',
        'CP': 'CANDIDATE',
        'KP': 'CANDIDATE',
        'APC': 'CANDIDATE',
        'FA': 'FALSE POSITIVE'
    }
    
    y = y.map(class_mapping)
    
    print(f"\\nClass distribution after mapping:")
    print(y.value_counts())
    print(f"\\nFinal dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y

def feature_selection_and_engineering(X: pd.DataFrame, y: pd.Series, k_features: int = 15) -> Tuple[pd.DataFrame, List[str]]:
    """Perform advanced feature selection and engineering."""
    print(f"\\nPerforming feature selection...")
    
    # Remove features with low variance
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = pd.DataFrame(
        variance_selector.fit_transform(X),
        columns=X.columns[variance_selector.get_support()],
        index=X.index
    )
    
    print(f"After variance thresholding: {X_variance.shape[1]} features")
    
    # Encode target for feature selection
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Select K best features using F-test
    selector = SelectKBest(score_func=f_classif, k=min(k_features, X_variance.shape[1]))
    X_selected = pd.DataFrame(
        selector.fit_transform(X_variance, y_encoded),
        columns=X_variance.columns[selector.get_support()],
        index=X.index
    )
    
    selected_features = list(X_selected.columns)
    print(f"Selected {len(selected_features)} best features: {selected_features}")
    
    # Feature importance scores
    feature_scores = pd.DataFrame({
        'feature': X_variance.columns[selector.get_support()],
        'score': selector.scores_[selector.get_support()]
    }).sort_values('score', ascending=False)
    
    print(f"\\nTop 10 feature importance scores:")
    print(feature_scores.head(10))
    
    return X_selected, selected_features

def create_optimized_models() -> Dict[str, Any]:
    """Create optimized model configurations."""
    models = {}
    
    # Optimized Random Forest
    models['optimized_rf'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Extra Trees (faster than RF)
    models['extra_trees'] = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=False,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    if XGBOOST_AVAILABLE:
        # Optimized XGBoost
        models['optimized_xgb'] = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1
        )
    
    if LIGHTGBM_AVAILABLE:
        # LightGBM (very efficient)
        models['lightgbm'] = LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
    
    return models

def hyperparameter_tuning(model, X_train, y_train, param_grid: Dict) -> Any:
    """Perform hyperparameter tuning using GridSearchCV."""
    print(f"\\nTuning hyperparameters for {type(model).__name__}...")
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train and evaluate all models."""
    print(f"\\n{'='*60}")
    print("TRAINING AND EVALUATION")
    print(f"{'='*60}")
    
    # Encode labels for models that need numeric targets
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Also split encoded labels
    _, _, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Create models
    models = create_optimized_models()
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\\n{'='*40}")
        print(f"Training {name}")
        print(f"{'='*40}")
        
        # Use encoded labels for XGBoost and LightGBM, string labels for others
        if 'xgb' in name.lower() or 'lightgbm' in name.lower():
            y_train_use = y_train_encoded
            y_test_use = y_test_encoded
        else:
            y_train_use = y_train
            y_test_use = y_test
        
        # Train model
        model.fit(X_train_scaled, y_train_use)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train_use, cv=5, scoring='accuracy')
        
        # Test predictions
        y_pred = model.predict(X_test_scaled)
        
        # Convert predictions back to string labels if needed
        if 'xgb' in name.lower() or 'lightgbm' in name.lower():
            y_pred_strings = label_encoder.inverse_transform(y_pred)
            test_accuracy = accuracy_score(y_test, y_pred_strings)
            classification_rep = classification_report(y_test, y_pred_strings, output_dict=True)
            print_report = classification_report(y_test, y_pred_strings)
        else:
            test_accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            print_report = classification_report(y_test, y_pred)
        
        # Store results
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'classification_report': classification_rep
        }
        
        trained_models[name] = model
        
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"\\nClassification Report:")
        print(print_report)
    
    return results, trained_models, scaler, (X_test_scaled, y_test), label_encoder

def create_ensemble_model(trained_models: Dict, X_test, y_test, label_encoder) -> Dict[str, float]:
    """Create and evaluate ensemble predictions."""
    print(f"\\n{'='*40}")
    print("ENSEMBLE PREDICTIONS")
    print(f"{'='*40}")
    
    # Get predictions from all models
    predictions = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        # Convert predictions to string labels if needed
        if 'xgb' in name.lower() or 'lightgbm' in name.lower():
            predictions[name] = label_encoder.inverse_transform(y_pred)
        else:
            predictions[name] = y_pred
    
    # Simple voting ensemble
    pred_df = pd.DataFrame(predictions)
    
    # Majority voting
    ensemble_pred = pred_df.mode(axis=1)[0]
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble (Majority Voting) accuracy: {ensemble_accuracy:.4f}")
    print(f"\\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    
    return {'ensemble_accuracy': ensemble_accuracy}

def save_results_and_models(results: Dict, trained_models: Dict, scaler, selected_features: List[str]):
    """Save training results and models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("models") / "optimized_training_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = results_dir / f"training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save models
    models_dir = results_dir / "trained_models"
    models_dir.mkdir(exist_ok=True)
    
    for name, model in trained_models.items():
        model_file = models_dir / f"{name}_{timestamp}.joblib"
        joblib.dump(model, model_file)
        print(f"Saved {name} to {model_file}")
    
    # Save scaler
    scaler_file = models_dir / f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_file)
    
    # Save selected features
    features_file = results_dir / f"selected_features_{timestamp}.json"
    with open(features_file, 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    print(f"\\nAll results saved to {results_dir}")
    
    return results_dir

def plot_results(results: Dict):
    """Plot training results comparison."""
    model_names = list(results.keys())
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    test_scores = [results[name]['test_accuracy'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # CV scores
    ax1.bar(model_names, cv_scores)
    ax1.set_title('Cross-Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # Test scores
    ax2.bar(model_names, test_scores)
    ax2.set_title('Test Set Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("models") / "optimized_training_results" / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")
    
    plt.show()

def main():
    """Main training pipeline."""
    print("OPTIMIZED MODEL TRAINING WITH CLEANED DATA")
    print("="*60)
    
    # Load cleaned data
    data_path = "data/cleaned_training_data.csv"
    X, y = load_and_prepare_data(data_path)
    
    # Feature selection
    X_selected, selected_features = feature_selection_and_engineering(X, y, k_features=15)
    
    # Train and evaluate models
    results, trained_models, scaler, test_data, label_encoder = train_and_evaluate_models(X_selected, y)
    
    # Create ensemble predictions
    ensemble_results = create_ensemble_model(trained_models, test_data[0], test_data[1], label_encoder)
    results['ensemble'] = ensemble_results
    
    # Save everything
    results_dir = save_results_and_models(results, trained_models, scaler, selected_features)
    
    # Plot results
    plot_results({k: v for k, v in results.items() if k != 'ensemble'})
    
    # Print summary
    print(f"\\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    best_model = max(results.items(), key=lambda x: x[1].get('test_accuracy', 0) if isinstance(x[1], dict) else 0)
    print(f"Best model: {best_model[0]} with test accuracy: {best_model[1]['test_accuracy']:.4f}")
    print(f"Ensemble accuracy: {ensemble_results['ensemble_accuracy']:.4f}")
    print(f"\\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main()