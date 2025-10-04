"""
Improved Stacking Ensemble
==========================
Create a stacked ensemble with best performing models as base learners
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import sys

sys.path.append(str(Path(__file__).parent))
from train_ultimate_ensemble import AdvancedFeatureEngineering


def load_data():
    """Load and prepare data"""
    print("Loading data...")
    
    df = pd.read_csv('notebooks/datasets/exoplanets_combined.csv')
    
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
    
    # Base features
    numeric_cols = ['orbital_period', 'transit_depth', 'planet_radius', 
                   'koi_teq', 'koi_insol', 'stellar_teff', 'stellar_radius',
                   'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
                   'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter',
                   'transit_duration', 'koi_dor', 'st_tmag', 'st_logg', 
                   'st_dist', 'st_mass']
    
    numeric_cols = [col for col in numeric_cols if col in df.columns]
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
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"✓ Loaded {len(df)} samples with {len(feature_cols)} features")
    return X, y_encoded, le


def create_stacking_ensemble():
    """Create improved stacking ensemble"""
    print("\n" + "="*80)
    print("CREATING IMPROVED STACKING ENSEMBLE")
    print("="*80)
    
    # Load data
    X, y, le = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base learners (use only best performers)
    print("\n📦 Base Learners:")
    
    base_learners = [
        ('xgboost', xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('lightgbm', lgb.LGBMClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )),
        ('catboost', cb.CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=300,
            random_seed=42,
            verbose=False
        )),
        ('random_forest', xgb.XGBRFClassifier(
            max_depth=8,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ]
    
    for name, _ in base_learners:
        print(f"   - {name}")
    
    # Define meta-learner
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        class_weight='balanced'
    )
    
    print(f"\n🎯 Meta-Learner: Logistic Regression")
    
    # Create stacking classifier
    print(f"\n🔨 Building stacking ensemble...")
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False  # Don't pass original features to meta-learner
    )
    
    # Train
    print(f"\n⏳ Training (this may take a while)...")
    stacking_model.fit(X_train_scaled, y_train)
    print(f"✓ Training complete")
    
    # Evaluate on validation set
    print(f"\n📊 Validation Performance:")
    val_pred = stacking_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    print(f"   Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   F1-Score: {val_f1:.4f}")
    
    # Evaluate on test set
    print(f"\n🎯 Test Performance:")
    test_pred = stacking_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   F1-Score: {test_f1:.4f}")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_))
    
    # Compare with individual models
    print(f"\n⚖️  Individual Model Performance (Test Set):")
    for name, model in base_learners:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        print(f"   {name:15} - Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    print(f"   {'Stacking':15} - Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} ⭐")
    
    # Save model
    models_dir = Path('models')
    joblib.dump(stacking_model, models_dir / 'stacking_ensemble_v2.pkl')
    joblib.dump(scaler, models_dir / 'stacking_scaler_v2.pkl')
    joblib.dump(le, models_dir / 'stacking_encoder_v2.pkl')
    
    print(f"\n💾 Saved models:")
    print(f"   - stacking_ensemble_v2.pkl")
    print(f"   - stacking_scaler_v2.pkl")
    print(f"   - stacking_encoder_v2.pkl")
    
    print(f"\n{'='*80}")
    print("STACKING ENSEMBLE COMPLETE!")
    print(f"{'='*80}")
    
    return stacking_model, scaler, le, test_acc, test_f1


if __name__ == "__main__":
    create_stacking_ensemble()
