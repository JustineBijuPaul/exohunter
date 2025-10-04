"""
Ensemble methods for combining multiple exoplanet classification models.

This module provides ensemble techniques including voting and stacking to combine
predictions from Random Forest, XGBoost, and MLP models for improved robustness
and performance.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

# Import our models
try:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    TRADITIONAL_MODELS_AVAILABLE = True
except ImportError:
    TRADITIONAL_MODELS_AVAILABLE = False
    warnings.warn("Traditional ML models not available")

try:
    from .advanced import TabularMLP
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    warnings.warn("Deep learning models not available")


class XGBWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for XGBoost to handle string labels properly.
    """
    
    def __init__(self, **xgb_kwargs):
        """Initialize XGB wrapper."""
        self.xgb_kwargs = xgb_kwargs
        self.model = None
        self.label_encoder = None
        self.classes_ = None
        
    def _more_tags(self):
        return {'_xfail_checks': {'check_parameters_default_constructible'}}
    
    @property
    def _estimator_type(self):
        return "classifier"
        
    def fit(self, X, y):
        """Fit the XGBoost model with label encoding."""
        if not TRADITIONAL_MODELS_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        # Convert to arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Encode string labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Train XGBoost
        self.model = XGBClassifier(**self.xgb_kwargs)
        self.model.fit(X, y_encoded)
        
        return self
    
    def predict(self, X):
        """Make predictions and decode labels."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if hasattr(X, 'values'):
            X = X.values
        
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if hasattr(X, 'values'):
            X = X.values
        
        return self.model.predict_proba(X)


class MLPWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for TabularMLP to use in ensemble methods.
    """
    
    def __init__(self, **mlp_kwargs):
        """Initialize MLP wrapper with parameters for TabularMLP."""
        self.mlp_kwargs = mlp_kwargs
        self.model = None
        self.classes_ = None
        
    def _more_tags(self):
        return {'_xfail_checks': {'check_parameters_default_constructible'}}
    
    @property
    def _estimator_type(self):
        return "classifier"
        
    def fit(self, X, y):
        """Fit the MLP model."""
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TabularMLP not available")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        
        # Initialize MLP
        input_dim = X.shape[1]
        num_classes = len(self.classes_)
        
        self.model = TabularMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            **self.mlp_kwargs
        )
        
        # Train the model
        self.model.fit(X, y, verbose=0)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Scale features and get probabilities
        X_scaled = self.model.scaler.transform(X)
        probabilities = self.model.model.predict(X_scaled)
        
        return probabilities


def create_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create base models for ensemble methods.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of base models
    """
    base_models = {}
    
    if TRADITIONAL_MODELS_AVAILABLE:
        # Random Forest
        base_models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # XGBoost (with wrapper for string labels)
        base_models['xgb'] = XGBWrapper(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            verbosity=0
        )
    
    if DEEP_LEARNING_AVAILABLE:
        # MLP (wrapped for sklearn compatibility)
        base_models['mlp'] = MLPWrapper(
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            activation='relu'
        )
    
    return base_models


def build_voting_ensemble(
    base_models: Dict[str, Any],
    voting: str = 'soft'
) -> VotingClassifier:
    """
    Build a voting ensemble from base models.
    
    Args:
        base_models: Dictionary of base models
        voting: 'hard' or 'soft' voting
        
    Returns:
        Configured VotingClassifier
    """
    estimators = [(name, model) for name, model in base_models.items()]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting
    )
    
    return ensemble


def build_stacking_ensemble(
    base_models: Dict[str, Any],
    meta_model: Optional[Any] = None,
    cv: int = 5,
    random_state: int = 42
) -> StackingClassifier:
    """
    Build a stacking ensemble with a meta-model.
    
    This function creates a stacking ensemble where base models make predictions
    on cross-validation folds, and a meta-model learns to combine these predictions.
    
    Args:
        base_models: Dictionary of base models
        meta_model: Meta-learner model (default: LogisticRegression)
        cv: Number of cross-validation folds for stacking
        random_state: Random seed for reproducibility
        
    Returns:
        Configured StackingClassifier
        
    Example:
        >>> base_models = create_base_models()
        >>> meta_model = LogisticRegression()
        >>> ensemble = build_stacking_ensemble(base_models, meta_model)
    """
    if meta_model is None:
        meta_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return ensemble


def cross_val_score_ensemble(
    ensemble: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv: int = 5,
    scoring: str = 'accuracy',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform cross-validation scoring for ensemble models.
    
    This function provides stable metrics estimation using stratified cross-validation
    to ensure reliable performance assessment of ensemble models.
    
    Args:
        ensemble: Ensemble model to evaluate
        X: Feature matrix
        y: Target labels
        cv: Number of cross-validation folds
        scoring: Scoring metric ('accuracy', 'f1_macro', etc.)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with cross-validation results and statistics
        
    Example:
        >>> ensemble = build_stacking_ensemble(base_models)
        >>> results = cross_val_score_ensemble(ensemble, X, y, cv=5)
        >>> print(f"Mean CV accuracy: {results['mean']:.4f} ± {results['std']:.4f}")
    """
    # Convert to numpy if needed
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        ensemble, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1
    )
    
    results = {
        'scores': cv_scores.tolist(),
        'mean': float(np.mean(cv_scores)),
        'std': float(np.std(cv_scores)),
        'min': float(np.min(cv_scores)),
        'max': float(np.max(cv_scores)),
        'scoring': scoring,
        'cv_folds': cv
    }
    
    return results


def train_ensemble_suite(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output_dir: Union[str, Path] = "ensemble_models",
    random_state: int = 42,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Train and evaluate a complete ensemble suite.
    
    This function trains individual base models, voting ensemble, and stacking
    ensemble, then evaluates all models and saves the results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save models and results
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with all results and model paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Training ensemble suite...")
    
    # Convert to DataFrames if needed
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
    
    results = {
        'base_models': {},
        'ensembles': {},
        'cv_results': {},
        'test_results': {},
        'model_paths': {}
    }
    
    # 1. Create and train base models
    print("📦 Training base models...")
    base_models = create_base_models(random_state)
    
    for name, model in base_models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_results = cross_val_score_ensemble(
                model, X_train, y_train, cv=cv_folds, random_state=random_state
            )
            results['cv_results'][name] = cv_results
            
            # Test evaluation
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            results['test_results'][name] = {
                'accuracy': float(test_accuracy),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Save model
            model_path = output_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            results['model_paths'][name] = str(model_path)
            
            print(f"    ✅ {name}: CV = {cv_results['mean']:.4f} ± {cv_results['std']:.4f}, "
                  f"Test = {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"    ❌ {name}: Failed to train - {str(e)}")
            continue
    
    # 2. Build and train voting ensemble
    print("🗳️  Training voting ensemble...")
    try:
        voting_ensemble = build_voting_ensemble(base_models, voting='soft')
        voting_ensemble.fit(X_train, y_train)
        
        # Cross-validation
        cv_results = cross_val_score_ensemble(
            voting_ensemble, X_train, y_train, cv=cv_folds, random_state=random_state
        )
        results['cv_results']['voting'] = cv_results
        
        # Test evaluation
        y_pred = voting_ensemble.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        results['test_results']['voting'] = {
            'accuracy': float(test_accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model
        voting_path = output_dir / "voting_ensemble.joblib"
        joblib.dump(voting_ensemble, voting_path)
        results['model_paths']['voting'] = str(voting_path)
        
        print(f"    ✅ Voting: CV = {cv_results['mean']:.4f} ± {cv_results['std']:.4f}, "
              f"Test = {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"    ❌ Voting ensemble failed: {str(e)}")
    
    # 3. Build and train stacking ensemble
    print("📚 Training stacking ensemble...")
    try:
        stacking_ensemble = build_stacking_ensemble(
            base_models, cv=cv_folds, random_state=random_state
        )
        stacking_ensemble.fit(X_train, y_train)
        
        # Cross-validation
        cv_results = cross_val_score_ensemble(
            stacking_ensemble, X_train, y_train, cv=cv_folds, random_state=random_state
        )
        results['cv_results']['stacking'] = cv_results
        
        # Test evaluation
        y_pred = stacking_ensemble.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        results['test_results']['stacking'] = {
            'accuracy': float(test_accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model
        stacking_path = output_dir / "stacking_ensemble.joblib"
        joblib.dump(stacking_ensemble, stacking_path)
        results['model_paths']['stacking'] = str(stacking_path)
        
        print(f"    ✅ Stacking: CV = {cv_results['mean']:.4f} ± {cv_results['std']:.4f}, "
              f"Test = {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"    ❌ Stacking ensemble failed: {str(e)}")
    
    # 4. Find best model
    test_accuracies = {name: results['test_results'][name]['accuracy'] 
                      for name in results['test_results']}
    best_model_name = max(test_accuracies, key=test_accuracies.get)
    best_accuracy = test_accuracies[best_model_name]
    
    results['best_model'] = {
        'name': best_model_name,
        'accuracy': best_accuracy,
        'path': results['model_paths'].get(best_model_name)
    }
    
    # 5. Save complete results
    results_path = output_dir / "ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    results['results_path'] = str(results_path)
    
    print(f"\n🎉 Ensemble training completed!")
    print(f"📊 Best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    print(f"💾 Results saved to: {output_dir}")
    
    return results


def load_ensemble(model_path: Union[str, Path]) -> Any:
    """
    Load a saved ensemble model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded ensemble model
    """
    return joblib.load(model_path)


def predict_with_ensemble(
    ensemble: Any,
    X: Union[pd.DataFrame, np.ndarray],
    return_probabilities: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions with an ensemble model.
    
    Args:
        ensemble: Trained ensemble model
        X: Features to predict
        return_probabilities: Whether to return prediction probabilities
        
    Returns:
        Predictions (and probabilities if requested)
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    predictions = ensemble.predict(X)
    
    if return_probabilities:
        if hasattr(ensemble, 'predict_proba'):
            probabilities = ensemble.predict_proba(X)
            return predictions, probabilities
        else:
            warnings.warn("Model does not support probability prediction")
            return predictions, None
    
    return predictions


def ensemble_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of ensemble training results.
    
    Args:
        results: Results dictionary from train_ensemble_suite
    """
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING SUMMARY")
    print("="*60)
    
    # Cross-validation results
    print("\n📊 Cross-Validation Results:")
    print("-"*40)
    for model_name, cv_results in results['cv_results'].items():
        print(f"{model_name:>12}: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    
    # Test results
    print("\n🎯 Test Set Results:")
    print("-"*40)
    for model_name, test_results in results['test_results'].items():
        print(f"{model_name:>12}: {test_results['accuracy']:.4f}")
    
    # Best model
    print(f"\n🏆 Best Model: {results['best_model']['name']} "
          f"(accuracy: {results['best_model']['accuracy']:.4f})")
    
    # Model paths
    print(f"\n💾 Saved Models:")
    print("-"*40)
    for model_name, path in results['model_paths'].items():
        print(f"{model_name:>12}: {path}")


# Example usage
if __name__ == "__main__":
    print("Ensemble module example usage:")
    print("""
    # Train ensemble suite
    from exohunter.models.ensemble import train_ensemble_suite, ensemble_summary
    
    results = train_ensemble_suite(
        X_train, y_train, X_test, y_test,
        output_dir='ensemble_models',
        cv_folds=5
    )
    
    ensemble_summary(results)
    
    # Load and use best model
    from exohunter.models.ensemble import load_ensemble, predict_with_ensemble
    
    best_model = load_ensemble(results['best_model']['path'])
    predictions = predict_with_ensemble(best_model, X_new)
    """)
