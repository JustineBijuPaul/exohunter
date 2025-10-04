"""
Hyperparameter Tuning with Optuna
=================================
Optimize XGBoost, LightGBM, and CatBoost hyperparameters using Bayesian optimization
"""

import pandas as pd
import numpy as np
import joblib
import optuna
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import sys

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import feature engineering
sys.path.append(str(Path(__file__).parent))
from train_ultimate_ensemble import AdvancedFeatureEngineering


def load_and_prepare_data():
    """Load and prepare data for tuning"""
    print("Loading data...")
    
    # Load data
    df = pd.read_csv('notebooks/datasets/exoplanets_combined.csv')
    
    # Map dispositions
    disposition_mapping = {
        'FALSE POSITIVE': 'FALSE POSITIVE',
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'NOT DISPOSITIONED': 'CANDIDATE',
        'Confirmed Planet': 'CONFIRMED',
        'False Positive': 'FALSE POSITIVE',
        'Candidate': 'CANDIDATE'
    }
    df['disposition'] = df['disposition'].map(disposition_mapping)
    df = df[df['disposition'].notna()].copy()
    
    print(f"Loaded {len(df)} samples")
    
    # Base numeric columns
    numeric_cols = ['orbital_period', 'transit_depth', 'planet_radius', 
                   'koi_teq', 'koi_insol', 'stellar_teff', 'stellar_radius',
                   'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
                   'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter',
                   'transit_duration', 'koi_dor', 'st_tmag', 'st_logg', 
                   'st_dist', 'st_mass']
    
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Fill missing values
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Engineer features
    fe = AdvancedFeatureEngineering()
    df = fe.engineer_features(df)
    
    # Select features
    feature_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    feature_cols = [col for col in feature_cols if col != 'disposition_encoded']
    
    X = df[feature_cols].values
    y = df['disposition'].values
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Features: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    return X, y_encoded, le


def objective_xgboost(trial, X, y):
    """Optuna objective for XGBoost"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    model = xgb.XGBClassifier(**params)
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()


def objective_lightgbm(trial, X, y):
    """Optuna objective for LightGBM"""
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    
    return scores.mean()


def objective_catboost(trial, X, y):
    """Optuna objective for CatBoost"""
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }
    
    model = cb.CatBoostClassifier(**params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=1)
    
    return scores.mean()


def tune_model(model_name, objective_func, X, y, n_trials=50):
    """Tune a model using Optuna"""
    print(f"\n{'='*80}")
    print(f"TUNING {model_name}")
    print(f"{'='*80}")
    print(f"Trials: {n_trials}")
    print(f"CV Folds: 5")
    print(f"Metric: F1-weighted")
    
    study = optuna.create_study(direction='maximize', study_name=f'{model_name}_tuning')
    study.optimize(lambda trial: objective_func(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✓ Best F1 Score: {study.best_value:.4f}")
    print(f"✓ Best Parameters:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    
    return study


def train_and_evaluate_best_models(X, y, le, best_params):
    """Train models with best parameters and evaluate"""
    print(f"\n{'='*80}")
    print("TRAINING OPTIMIZED MODELS")
    print(f"{'='*80}")
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import classification_report, accuracy_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Train XGBoost
    print("\n📊 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**best_params['XGBoost'], random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    results['XGBoost'] = {'accuracy': xgb_acc, 'f1_score': xgb_f1, 'model': xgb_model}
    print(f"   Accuracy: {xgb_acc:.4f}")
    print(f"   F1-Score: {xgb_f1:.4f}")
    
    # Train LightGBM
    print("\n📊 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**best_params['LightGBM'], random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train_scaled, y_train)
    lgb_pred = lgb_model.predict(X_test_scaled)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred, average='weighted')
    results['LightGBM'] = {'accuracy': lgb_acc, 'f1_score': lgb_f1, 'model': lgb_model}
    print(f"   Accuracy: {lgb_acc:.4f}")
    print(f"   F1-Score: {lgb_f1:.4f}")
    
    # Train CatBoost
    print("\n📊 Training CatBoost...")
    cat_model = cb.CatBoostClassifier(**best_params['CatBoost'], random_seed=42, verbose=False)
    cat_model.fit(X_train_scaled, y_train)
    cat_pred = cat_model.predict(X_test_scaled)
    cat_acc = accuracy_score(y_test, cat_pred)
    cat_f1 = f1_score(y_test, cat_pred, average='weighted')
    results['CatBoost'] = {'accuracy': cat_acc, 'f1_score': cat_f1, 'model': cat_model}
    print(f"   Accuracy: {cat_acc:.4f}")
    print(f"   F1-Score: {cat_f1:.4f}")
    
    # Save models
    print(f"\n💾 Saving optimized models...")
    models_dir = Path('models')
    joblib.dump(xgb_model, models_dir / 'xgboost_optimized.pkl')
    joblib.dump(lgb_model, models_dir / 'lightgbm_optimized.pkl')
    joblib.dump(cat_model, models_dir / 'catboost_optimized.pkl')
    joblib.dump(scaler, models_dir / 'scaler_optimized.pkl')
    joblib.dump(le, models_dir / 'encoder_optimized.pkl')
    print("✓ Models saved")
    
    return results


def main():
    """Main tuning pipeline"""
    print("="*80)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    X, y, le = load_and_prepare_data()
    
    # Tune models (use fewer trials for speed, increase for production)
    n_trials = 30  # Increase to 100+ for production
    
    studies = {}
    studies['XGBoost'] = tune_model('XGBoost', objective_xgboost, X, y, n_trials)
    studies['LightGBM'] = tune_model('LightGBM', objective_lightgbm, X, y, n_trials)
    studies['CatBoost'] = tune_model('CatBoost', objective_catboost, X, y, n_trials)
    
    # Collect best parameters
    best_params = {
        'XGBoost': studies['XGBoost'].best_params,
        'LightGBM': studies['LightGBM'].best_params,
        'CatBoost': studies['CatBoost'].best_params
    }
    
    # Save tuning results
    tuning_results = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': n_trials,
        'best_scores': {
            'XGBoost': studies['XGBoost'].best_value,
            'LightGBM': studies['LightGBM'].best_value,
            'CatBoost': studies['CatBoost'].best_value
        },
        'best_parameters': best_params
    }
    
    with open('models/tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"\n✓ Saved tuning results to models/tuning_results.json")
    
    # Train final models with best parameters
    results = train_and_evaluate_best_models(X, y, le, best_params)
    
    # Summary
    print(f"\n{'='*80}")
    print("TUNING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    print(f"\nBest CV Scores (F1-weighted):")
    for model, study in studies.items():
        print(f"   {model}: {study.best_value:.4f}")
    
    print(f"\nTest Set Performance:")
    for model, res in results.items():
        print(f"   {model}:")
        print(f"      Accuracy: {res['accuracy']:.4f}")
        print(f"      F1-Score: {res['f1_score']:.4f}")
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
