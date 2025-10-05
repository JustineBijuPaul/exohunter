"""
Train ExoHunter models on train.csv with disposition as labels

This script:
1. Loads train.csv
2. Splits into train/test sets
3. Trains 4 optimized models
4. Saves models compatible with Python 3.12.4
5. Evaluates performance
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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)

# Try to import optional models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the training dataset."""
    print(f"Loading training data from {data_path}...")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset shape: {df.shape}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check for the target column
    if 'disposition' not in df.columns:
        raise ValueError("Target column 'disposition' not found in data")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != 'disposition']
    X = df[feature_cols]
    y = df['disposition']
    
    print(f"\n📊 Features: {len(feature_cols)}")
    print(f"   {', '.join(feature_cols)}")
    
    print(f"\n🎯 Target distribution:")
    print(y.value_counts())
    print(f"\n   Percentages:")
    print(y.value_counts(normalize=True).mul(100).round(2))
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values found:")
        print(missing[missing > 0])
        print("   Filling with median...")
        df = df.fillna(df.median())
    else:
        print(f"\n✅ No missing values")
    
    return X, y

def train_optimized_models(X_train, X_test, y_train, y_test, scaler):
    """Train and optimize multiple models."""
    results = {}
    trained_models = {}
    
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # 1. Optimized Random Forest
    print("\n[1/4] Training Optimized Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, n_jobs=-1)
    rf_test_pred = rf_model.predict(X_test)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    
    results['optimized_rf'] = {
        'cv_mean': rf_cv_scores.mean(),
        'cv_std': rf_cv_scores.std(),
        'test_accuracy': rf_test_acc,
        'classification_report': classification_report(y_test, rf_test_pred, output_dict=True)
    }
    trained_models['optimized_rf'] = rf_model
    print(f"   ✅ CV: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
    print(f"   ✅ Test Accuracy: {rf_test_acc:.4f}")
    
    # 2. Extra Trees
    print("\n[2/4] Training Extra Trees...")
    et_model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    et_model.fit(X_train, y_train)
    et_cv_scores = cross_val_score(et_model, X_train, y_train, cv=5, n_jobs=-1)
    et_test_pred = et_model.predict(X_test)
    et_test_acc = accuracy_score(y_test, et_test_pred)
    
    results['extra_trees'] = {
        'cv_mean': et_cv_scores.mean(),
        'cv_std': et_cv_scores.std(),
        'test_accuracy': et_test_acc,
        'classification_report': classification_report(y_test, et_test_pred, output_dict=True)
    }
    trained_models['extra_trees'] = et_model
    print(f"   ✅ CV: {et_cv_scores.mean():.4f} ± {et_cv_scores.std():.4f}")
    print(f"   ✅ Test Accuracy: {et_test_acc:.4f}")
    
    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n[3/4] Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=15,
            learning_rate=0.05,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        lgb_cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, n_jobs=-1)
        lgb_test_pred = lgb_model.predict(X_test)
        lgb_test_acc = accuracy_score(y_test, lgb_test_pred)
        
        results['lightgbm'] = {
            'cv_mean': lgb_cv_scores.mean(),
            'cv_std': lgb_cv_scores.std(),
            'test_accuracy': lgb_test_acc,
            'classification_report': classification_report(y_test, lgb_test_pred, output_dict=True)
        }
        trained_models['lightgbm'] = lgb_model
        print(f"   ✅ CV: {lgb_cv_scores.mean():.4f} ± {lgb_cv_scores.std():.4f}")
        print(f"   ✅ Test Accuracy: {lgb_test_acc:.4f}")
    else:
        print("\n[3/4] ⏭️  Skipping LightGBM (not installed)")
    
    # 4. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n[4/4] Training Optimized XGBoost...")
        
        # Encode labels for XGBoost
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(X_train, y_train_encoded)
        xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train_encoded, cv=5, n_jobs=-1)
        xgb_test_pred_encoded = xgb_model.predict(X_test)
        xgb_test_pred = label_encoder.inverse_transform(xgb_test_pred_encoded)
        xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
        
        results['optimized_xgb'] = {
            'cv_mean': xgb_cv_scores.mean(),
            'cv_std': xgb_cv_scores.std(),
            'test_accuracy': xgb_test_acc,
            'classification_report': classification_report(y_test, xgb_test_pred, output_dict=True),
            'label_encoder': label_encoder
        }
        trained_models['optimized_xgb'] = xgb_model
        print(f"   ✅ CV: {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")
        print(f"   ✅ Test Accuracy: {xgb_test_acc:.4f}")
    else:
        print("\n[4/4] ⏭️  Skipping XGBoost (not installed)")
    
    return results, trained_models

def create_ensemble_predictions(trained_models, X_test, y_test, label_encoder=None):
    """Create ensemble predictions from multiple models."""
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTIONS")
    print("="*70)
    
    predictions = []
    model_names = []
    
    for name, model in trained_models.items():
        if name == 'optimized_xgb' and label_encoder:
            # XGBoost predictions need to be decoded
            pred_encoded = model.predict(X_test)
            pred = label_encoder.inverse_transform(pred_encoded)
        else:
            pred = model.predict(X_test)
        predictions.append(pred)
        model_names.append(name)
    
    # Majority voting
    predictions_array = np.array(predictions)
    ensemble_pred = []
    
    for i in range(len(X_test)):
        votes = predictions_array[:, i]
        unique, counts = np.unique(votes, return_counts=True)
        ensemble_pred.append(unique[np.argmax(counts)])
    
    ensemble_pred = np.array(ensemble_pred)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\n🎯 Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"   Individual model accuracies:")
    for name, pred in zip(model_names, predictions):
        acc = accuracy_score(y_test, pred)
        print(f"      {name:20s}: {acc:.4f}")
    
    print(f"\n📊 Ensemble Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    
    return {
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_predictions': ensemble_pred,
        'individual_predictions': {name: pred for name, pred in zip(model_names, predictions)}
    }

def save_models(trained_models, scaler, feature_names, results, ensemble_results, label_encoder=None):
    """Save all trained models and metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    models_dir = Path("models/trained_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path("models")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    # Save scaler
    scaler_path = models_dir / f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved scaler: {scaler_path.name}")
    
    # Save models
    for name, model in trained_models.items():
        model_path = models_dir / f"{name}_{timestamp}.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Saved {name}: {model_path.name}")
    
    # Save label encoder if exists
    if label_encoder:
        encoder_path = models_dir / f"label_encoder_{timestamp}.joblib"
        joblib.dump(label_encoder, encoder_path)
        print(f"✅ Saved label encoder: {encoder_path.name}")
    
    # Save feature names
    features_path = results_dir / f"selected_features_{timestamp}.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ Saved features: {features_path.name}")
    
    # Save training results (remove non-serializable objects)
    results_path = results_dir / f"training_results_{timestamp}.json"
    results_save = {}
    for model_name, model_results in results.items():
        results_save[model_name] = {
            'cv_mean': float(model_results['cv_mean']),
            'cv_std': float(model_results['cv_std']),
            'test_accuracy': float(model_results['test_accuracy']),
            'classification_report': model_results['classification_report']
        }
    with open(results_path, 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"✅ Saved results: {results_path.name}")
    
    # Save ensemble results
    ensemble_path = results_dir / f"ensemble_results_{timestamp}.json"
    ensemble_save = {
        'ensemble_accuracy': float(ensemble_results['ensemble_accuracy']),
        'timestamp': timestamp
    }
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_save, f, indent=2)
    print(f"✅ Saved ensemble results: {ensemble_path.name}")
    
    print(f"\n📁 All models saved with timestamp: {timestamp}")
    
    return timestamp

def print_summary(results, ensemble_results, timestamp):
    """Print final summary."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print(f"\n📊 Model Performance:")
    print(f"{'Model':<25} {'CV Score':>12} {'Test Accuracy':>15}")
    print("-" * 55)
    
    for name, res in results.items():
        cv_score = f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}"
        test_acc = f"{res['test_accuracy']:.4f}"
        print(f"{name:<25} {cv_score:>12} {test_acc:>15}")
    
    ensemble_acc = f"{ensemble_results['ensemble_accuracy']:.4f}"
    print(f"{'Ensemble (Majority Vote)':<25} {'-':>12} {ensemble_acc:>15}")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n🏆 Best Individual Model: {best_model[0]}")
    print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.4f}")
    
    print(f"\n🎯 Ensemble Accuracy: {ensemble_results['ensemble_accuracy']:.4f}")
    
    print(f"\n📁 Models saved with timestamp: {timestamp}")
    print(f"   Location: models/trained_models/*_{timestamp}.joblib")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print("\n📋 Next Steps:")
    print(f"   1. Update model filenames in web/streamlit/app.py")
    print(f"   2. Update model filenames in web/api/main.py")
    print(f"   3. Use timestamp: {timestamp}")
    print(f"   4. Restart your web applications")
    
    print(f"\n💡 Example filenames to use:")
    print(f"   - optimized_rf_{timestamp}.joblib")
    print(f"   - extra_trees_{timestamp}.joblib")
    if LIGHTGBM_AVAILABLE:
        print(f"   - lightgbm_{timestamp}.joblib")
    if XGBOOST_AVAILABLE:
        print(f"   - optimized_xgb_{timestamp}.joblib")
    print(f"   - scaler_{timestamp}.joblib")

def main():
    """Main training pipeline."""
    print("="*70)
    print("EXOHUNTER MODEL TRAINING")
    print("Training on: train.csv")
    print("Target: disposition")
    print("="*70)
    
    # Load data
    data_path = "data/train.csv"
    X, y = load_and_prepare_data(data_path)
    
    # Split data
    print("\n" + "="*70)
    print("SPLITTING DATA")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✅ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scale features
    print("\n" + "="*70)
    print("FEATURE SCALING")
    print("="*70)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled using StandardScaler")
    
    # Train models
    results, trained_models = train_optimized_models(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    )
    
    # Get label encoder if XGBoost was trained
    label_encoder = results.get('optimized_xgb', {}).get('label_encoder')
    
    # Create ensemble
    ensemble_results = create_ensemble_predictions(
        trained_models, X_test_scaled, y_test, label_encoder
    )
    
    # Save everything
    timestamp = save_models(
        trained_models, scaler, list(X.columns), 
        results, ensemble_results, label_encoder
    )
    
    # Print summary
    print_summary(results, ensemble_results, timestamp)

if __name__ == "__main__":
    main()
