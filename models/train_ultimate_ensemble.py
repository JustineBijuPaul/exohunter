#!/usr/bin/env python3
"""
Ultimate Ensemble Model for Exoplanet Classification
Combines:
- Deep Neural Networks (TensorFlow)
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Extra Trees
- Advanced feature engineering
- Stacking ensemble

This is the BEST model possible for exoplanet classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available")

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               VotingClassifier, StackingClassifier)
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, log_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, mutual_info_classif

# Gradient Boosting
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available. Install: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠ CatBoost not available. Install: pip install catboost")

import joblib


class AdvancedFeatureEngineering:
    """Advanced feature engineering for exoplanet data."""
    
    def __init__(self):
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features."""
        print("\n🔧 Engineering advanced features...")
        
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
        
        # 6. Statistical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['disposition_encoded']:
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
        
        print(f"✓ Created {features_created} engineered features")
        
        return df


class DeepNeuralNetwork:
    """Deep Neural Network for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self) -> models.Model:
        """Build deep neural network."""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            
            # First block
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Second block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Fourth block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the DNN."""
        self.model = self.build_model()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7
        )
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict(X, verbose=0)


class UltimateEnsemble:
    """Ultimate ensemble combining all best models."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.feature_engineer = AdvancedFeatureEngineering()
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.num_classes = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and clean data."""
        print("\n📊 Preparing data...")
        
        # Map disposition labels
        disposition_map = {
            'CONFIRMED': 'CONFIRMED',
            'FALSE POSITIVE': 'FALSE POSITIVE',
            'CANDIDATE': 'CANDIDATE',
            'PC': 'CANDIDATE',
            'FP': 'FALSE POSITIVE',
            'CP': 'CONFIRMED',
            'KP': 'CONFIRMED',
            'APC': 'CANDIDATE',
            'FA': 'FALSE POSITIVE'
        }
        
        df = df.copy()
        df['disposition'] = df['disposition'].map(disposition_map)
        df = df.dropna(subset=['disposition'])
        
        print(f"✓ Cleaned disposition labels")
        print(f"  Classes: {df['disposition'].value_counts().to_dict()}")
        
        # Select numeric features
        numeric_cols = ['orbital_period', 'transit_depth', 'planet_radius', 
                       'koi_teq', 'koi_insol', 'stellar_teff', 'stellar_radius',
                       'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
                       'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter',
                       'transit_duration', 'koi_dor', 'st_tmag', 'st_logg', 
                       'st_dist', 'st_mass']
        
        # Keep only available columns
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        # Fill missing values
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df)
        
        # Select all numeric columns (including engineered)
        feature_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        feature_cols = [col for col in feature_cols if col != 'disposition_encoded']
        
        X = df[feature_cols].values
        y = df['disposition'].values
        
        self.feature_names = feature_cols
        
        print(f"✓ Prepared {len(feature_cols)} features from {len(df)} samples")
        
        return X, y
    
    def train(self, df: pd.DataFrame):
        """Train the ultimate ensemble."""
        print("="*80)
        print("🚀 TRAINING ULTIMATE ENSEMBLE MODEL")
        print("="*80)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"\n✓ Classes: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=self.random_state, stratify=y_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\n✓ Data split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models
        print("\n" + "="*80)
        print("📚 Training Individual Models")
        print("="*80)
        
        # 1. XGBoost
        print("\n1️⃣  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train_scaled, y_train,
                     eval_set=[(X_val_scaled, y_val)],
                     verbose=False)
        self.models['xgboost'] = xgb_model
        val_score = accuracy_score(y_val, xgb_model.predict(X_val_scaled))
        print(f"   ✓ XGBoost validation accuracy: {val_score:.4f}")
        
        # 2. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n2️⃣  Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train,
                         eval_set=[(X_val_scaled, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            self.models['lightgbm'] = lgb_model
            val_score = accuracy_score(y_val, lgb_model.predict(X_val_scaled))
            print(f"   ✓ LightGBM validation accuracy: {val_score:.4f}")
        
        # 3. CatBoost
        if CATBOOST_AVAILABLE:
            print("\n3️⃣  Training CatBoost...")
            cat_model = cb.CatBoostClassifier(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_state=self.random_state,
                verbose=False
            )
            cat_model.fit(X_train_scaled, y_train,
                         eval_set=(X_val_scaled, y_val),
                         early_stopping_rounds=50)
            self.models['catboost'] = cat_model
            val_score = accuracy_score(y_val, cat_model.predict(X_val_scaled))
            print(f"   ✓ CatBoost validation accuracy: {val_score:.4f}")
        
        # 4. Random Forest
        print("\n4️⃣  Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        val_score = accuracy_score(y_val, rf_model.predict(X_val_scaled))
        print(f"   ✓ Random Forest validation accuracy: {val_score:.4f}")
        
        # 5. Extra Trees
        print("\n5️⃣  Training Extra Trees...")
        et_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        et_model.fit(X_train_scaled, y_train)
        self.models['extra_trees'] = et_model
        val_score = accuracy_score(y_val, et_model.predict(X_val_scaled))
        print(f"   ✓ Extra Trees validation accuracy: {val_score:.4f}")
        
        # 6. Deep Neural Network
        if TENSORFLOW_AVAILABLE:
            print("\n6️⃣  Training Deep Neural Network...")
            dnn = DeepNeuralNetwork(input_dim=X_train_scaled.shape[1], num_classes=self.num_classes)
            dnn.train(X_train_scaled, y_train, X_val_scaled, y_val, epochs=100)
            self.models['deep_nn'] = dnn
            val_preds = np.argmax(dnn.predict_proba(X_val_scaled), axis=1)
            val_score = accuracy_score(y_val, val_preds)
            print(f"   ✓ Deep NN validation accuracy: {val_score:.4f}")
        
        # Create ensemble predictions
        print("\n" + "="*80)
        print("🎯 Creating Ensemble Predictions")
        print("="*80)
        
        # Get predictions from all models
        test_preds = []
        for name, model in self.models.items():
            if name == 'deep_nn':
                probs = model.predict_proba(X_test_scaled)
            else:
                probs = model.predict_proba(X_test_scaled)
            test_preds.append(probs)
        
        # Average predictions
        ensemble_probs = np.mean(test_preds, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Evaluate
        test_accuracy = accuracy_score(y_test, ensemble_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, ensemble_preds, average='weighted')
        
        print(f"\n🏆 ENSEMBLE RESULTS:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, ensemble_preds, 
                                   target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_preds)
        
        # Save results
        self.save_results(cm, test_accuracy, precision, recall, f1, X_test_scaled, y_test)
        
        return {
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_results(self, cm, accuracy, precision, recall, f1, X_test, y_test):
        """Save models and results."""
        print("\n💾 Saving models and results...")
        
        # Save all models
        for name, model in self.models.items():
            if name == 'deep_nn':
                model.model.save(f'models/ultimate_ensemble_{name}.h5')
            else:
                joblib.dump(model, f'models/ultimate_ensemble_{name}.pkl')
        
        # Save scaler and encoder
        joblib.dump(self.scaler, 'models/ultimate_ensemble_scaler.pkl')
        joblib.dump(self.label_encoder, 'models/ultimate_ensemble_encoder.pkl')
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'num_models': len(self.models),
            'feature_count': len(self.feature_names),
            'classes': self.label_encoder.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/ultimate_ensemble_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Ultimate Ensemble - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/ultimate_ensemble_confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved {len(self.models)} models")
        print(f"✓ Saved metrics to ultimate_ensemble_metrics.json")
        print(f"✓ Saved confusion matrix")


def main():
    """Main training function."""
    print("="*80)
    print("🌟 ULTIMATE EXOPLANET ENSEMBLE CLASSIFIER")
    print("="*80)
    
    # Load data
    data_path = Path('notebooks/datasets/exoplanets_combined.csv')
    print(f"\n📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Train ensemble
    ensemble = UltimateEnsemble(random_state=42)
    results = ensemble.train(df)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n🎯 Final Test Accuracy: {results['test_accuracy']:.2%}")
    print(f"📊 Models trained: {len(ensemble.models)}")
    print(f"🔧 Features used: {len(ensemble.feature_names)}")
    
    print("\n📁 Saved files:")
    print("   - ultimate_ensemble_*.pkl (model files)")
    print("   - ultimate_ensemble_metrics.json")
    print("   - ultimate_ensemble_confusion_matrix.png")
    
    print("\n🚀 Ready for production use!")


if __name__ == "__main__":
    main()
