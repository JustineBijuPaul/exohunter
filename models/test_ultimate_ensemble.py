"""
Test Ultimate Ensemble Model on Test Dataset
============================================
Tests the trained ultimate ensemble model on test.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import feature engineering from training script
import sys
sys.path.append(str(Path(__file__).parent))


class AdvancedFeatureEngineering:
    """Advanced feature engineering for exoplanet classification - matches training exactly"""
    
    def __init__(self):
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features - EXACT COPY from train_ultimate_ensemble.py"""
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
        
        # 6. Statistical features - CRITICAL: normalize ALL numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['disposition_encoded'] and not col.endswith('_normalized'):
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
        
        print(f"✓ Created {features_created} base engineered features")
        print(f"✓ Total features after normalization: {len(df.select_dtypes(include=[np.number]).columns)}")
        
        return df


def load_models(model_dir: Path):
    """Load all trained models and preprocessing objects"""
    print("Loading models...")
    
    models = {}
    
    # Load tree-based models
    models['xgboost'] = joblib.load(model_dir / 'ultimate_ensemble_xgboost.pkl')
    models['lightgbm'] = joblib.load(model_dir / 'ultimate_ensemble_lightgbm.pkl')
    models['catboost'] = joblib.load(model_dir / 'ultimate_ensemble_catboost.pkl')
    models['random_forest'] = joblib.load(model_dir / 'ultimate_ensemble_random_forest.pkl')
    models['extra_trees'] = joblib.load(model_dir / 'ultimate_ensemble_extra_trees.pkl')
    
    # Load deep learning model
    from tensorflow import keras
    models['deep_nn'] = keras.models.load_model(model_dir / 'ultimate_ensemble_deep_nn.h5')
    
    # Load preprocessing objects
    scaler = joblib.load(model_dir / 'ultimate_ensemble_scaler.pkl')
    encoder = joblib.load(model_dir / 'ultimate_ensemble_encoder.pkl')
    
    # Load feature names if available
    feature_names_path = model_dir / 'ultimate_ensemble_feature_names.pkl'
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
        print(f"✓ Loaded feature names: {len(feature_names)} features")
    else:
        feature_names = None
        print("⚠️  No feature names file found")
    
    print("✓ All models loaded successfully!")
    return models, scaler, encoder, feature_names


def prepare_test_data(csv_path: Path, feature_names=None):
    """Load and prepare test data"""
    print(f"\nLoading test data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Original test data shape: {df.shape}")
    
    # Map disposition labels
    disposition_mapping = {
        'FALSE POSITIVE': 'FALSE POSITIVE',
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'NOT DISPOSITIONED': 'CANDIDATE',
        'Confirmed Planet': 'CONFIRMED',
        'False Positive': 'FALSE POSITIVE',
        'Candidate': 'CANDIDATE'
    }
    
    # Clean disposition column
    if 'disposition' in df.columns:
        df['disposition_clean'] = df['disposition'].map(disposition_mapping)
        # Remove rows with missing dispositions
        df = df[df['disposition_clean'].notna()].copy()
        print(f"After cleaning: {df.shape}")
        print(f"Class distribution:\n{df['disposition_clean'].value_counts()}")
        has_labels = True
    else:
        print("⚠️  No 'disposition' column found - predictions only mode")
        has_labels = False
    
    # Feature engineering
    print("\nCreating features...")
    
    # First, fill missing values in base numeric columns (use only the 20 features from training)
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
        df[col] = df[col].fillna(df[col].median())
    
    # Engineer features
    fe = AdvancedFeatureEngineering()
    df_features = fe.engineer_features(df)
    
    # If we have saved feature names, use them to select exact columns
    if feature_names is not None:
        print(f"Using saved feature names: {len(feature_names)} features")
        # Make sure all required features exist
        missing_features = []
        for feat in feature_names:
            if feat not in df_features.columns:
                missing_features.append(feat)
                df_features[feat] = 0  # Add missing features as zeros
        
        if missing_features:
            print(f"⚠️  Added {len(missing_features)} missing features as zeros")
        
        X = df_features[feature_names]
    else:
        # Fallback: select all numeric columns
        feature_cols = [col for col in df_features.columns 
                       if df_features[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        exclude_cols = ['disposition_encoded', 'disposition', 'disposition_clean']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        X = df_features[feature_cols]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Missing values before filling: {X.isnull().sum().sum()}")
    
    # Fill missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Missing values after filling: {X.isnull().sum().sum()}")
    
    if has_labels:
        y = df['disposition_clean'].values
        return X, y, df
    else:
        return X, None, df


def make_predictions(models, scaler, X):
    """Make predictions using ensemble"""
    print("\nMaking predictions...")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions from each model
    all_probs = []
    
    for name, model in models.items():
        print(f"  - {name}...")
        if name == 'deep_nn':
            # Keras model uses predict() not predict_proba()
            probs = model.predict(X_scaled, verbose=0)
        else:
            probs = model.predict_proba(X_scaled)
        all_probs.append(probs)
    
    # Ensemble averaging
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    print("✓ Predictions complete!")
    
    return ensemble_preds, ensemble_probs, all_probs


def evaluate_predictions(y_true, y_pred, encoder, output_dir: Path):
    """Evaluate predictions and create visualizations"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 70)
    report = classification_report(
        y_true, y_pred,
        target_names=encoder.classes_,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Ultimate Ensemble - Test Set Confusion Matrix', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = output_dir / 'test_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {cm_path}")
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classes': encoder.classes_.tolist(),
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(y_true)
    }
    
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    return metrics


def save_predictions(df, predictions, probs, encoder, output_dir: Path):
    """Save predictions to CSV"""
    print("\nSaving predictions...")
    
    # Create results dataframe
    results = df.copy()
    results['predicted_disposition'] = encoder.inverse_transform(predictions)
    
    # Add probability columns
    for i, class_name in enumerate(encoder.classes_):
        results[f'prob_{class_name}'] = probs[:, i]
    
    # Add prediction confidence
    results['confidence'] = np.max(probs, axis=1)
    
    # Save to CSV
    output_path = output_dir / 'test_predictions.csv'
    results.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")
    
    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Predicted distribution:")
    for class_name in encoder.classes_:
        count = (results['predicted_disposition'] == class_name).sum()
        pct = count / len(results) * 100
        print(f"    {class_name}: {count} ({pct:.2f}%)")
    
    print(f"\n  Average confidence: {results['confidence'].mean():.4f}")
    print(f"  Min confidence: {results['confidence'].min():.4f}")
    print(f"  Max confidence: {results['confidence'].max():.4f}")
    
    return results


def analyze_individual_models(all_probs, y_true, encoder):
    """Compare individual model performance"""
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL COMPARISON")
    print("="*70)
    
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Extra Trees', 'Deep NN']
    
    for i, (name, probs) in enumerate(zip(model_names, all_probs)):
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted')
        print(f"{name:15} - Accuracy: {acc:.4f} ({acc*100:.2f}%) | F1: {f1:.4f}")


def main():
    """Main testing function"""
    print("="*70)
    print("ULTIMATE ENSEMBLE MODEL - TEST EVALUATION")
    print("="*70)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / 'models'
    test_csv = project_dir / 'notebooks' / 'datasets' / 'test.csv'
    
    # Load models
    models, scaler, encoder, feature_names = load_models(model_dir)
    
    # Prepare test data
    X_test, y_test, df_test = prepare_test_data(test_csv, feature_names)
    
    # Make predictions
    predictions, probs, all_probs = make_predictions(models, scaler, X_test)
    
    # Decode predictions
    predictions_decoded = encoder.inverse_transform(predictions)
    
    # Evaluate if labels available
    if y_test is not None:
        # Encode true labels
        y_test_encoded = encoder.transform(y_test)
        
        # Evaluate
        metrics = evaluate_predictions(y_test_encoded, predictions, encoder, model_dir)
        
        # Analyze individual models
        analyze_individual_models(all_probs, y_test_encoded, encoder)
    else:
        print("\n⚠️  No labels available - skipping evaluation")
    
    # Save predictions
    results = save_predictions(df_test, predictions, probs, encoder, model_dir)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - Predictions: {model_dir / 'test_predictions.csv'}")
    if y_test is not None:
        print(f"  - Confusion Matrix: {model_dir / 'test_confusion_matrix.png'}")
        print(f"  - Metrics: {model_dir / 'test_metrics.json'}")


if __name__ == '__main__':
    main()
