"""
Generate fresh predictions using newly trained models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

def load_latest_models():
    """Load the most recently trained models"""
    models_dir = Path("models/trained_models")
    
    # Find all model files
    all_files = list(models_dir.glob("*.joblib"))
    if not all_files:
        raise FileNotFoundError("No model files found in models/trained_models/")
    
    # Group by model type
    model_types = ["optimized_rf", "extra_trees", "lightgbm", "optimized_xgb", "scaler"]
    models = {}
    
    for model_type in model_types:
        matching_files = [f for f in all_files if model_type in f.name]
        if matching_files:
            # Get the most recent file
            latest = max(matching_files, key=lambda x: x.stat().st_mtime)
            print(f"Loading {model_type}: {latest.name}")
            models[model_type] = joblib.load(latest)
        else:
            print(f"Warning: No {model_type} model found")
    
    return models

def make_predictions(data_path, models):
    """Make predictions on the dataset"""
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df)} samples")
    
    # Feature names used by the models
    feature_names = [
        'transit_depth', 'planet_radius', 'koi_teq', 'koi_insol', 'stellar_teff',
        'stellar_radius', 'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
        'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter', 'transit_duration', 'st_dist'
    ]
    
    # Extract features
    X = df[feature_names].values
    print(f"Features shape: {X.shape}")
    
    # Scale features if scaler available
    if 'scaler' in models:
        print("Scaling features...")
        X_scaled = models['scaler'].transform(X)
    else:
        print("Warning: No scaler found, using unscaled features")
        X_scaled = X
    
    # Make predictions with each model
    results = {'Row': range(1, len(df) + 1)}
    
    model_names = {
        'extra_trees': 'Extra Trees',
        'lightgbm': 'LightGBM',
        'optimized_rf': 'Optimized Random Forest',
        'optimized_xgb': 'Optimized XGBoost'
    }
    
    predictions_list = []
    
    for model_key, model_name in model_names.items():
        if model_key in models:
            print(f"Predicting with {model_name}...")
            preds = models[model_key].predict(X_scaled)
            results[model_name] = preds
            predictions_list.append(preds)
        else:
            print(f"Skipping {model_name} (not loaded)")
    
    # Ensemble prediction (majority vote)
    if predictions_list:
        print("Computing ensemble predictions...")
        predictions_array = np.array(predictions_list)
        ensemble_preds = []
        
        for i in range(len(df)):
            # Get predictions from all models for this sample
            sample_preds = predictions_array[:, i]
            # Count votes
            unique, counts = np.unique(sample_preds, return_counts=True)
            # Majority vote
            ensemble_preds.append(unique[np.argmax(counts)])
        
        results['Ensemble_Prediction'] = ensemble_preds
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put Ensemble first
    cols = ['Row', 'Ensemble_Prediction'] + [col for col in results_df.columns if col not in ['Row', 'Ensemble_Prediction']]
    results_df = results_df[cols]
    
    return results_df

def main():
    print("="*70)
    print("BATCH PREDICTION WITH LATEST MODELS")
    print("="*70)
    
    # Load models
    try:
        models = load_latest_models()
        print(f"\n✅ Loaded {len(models)} models")
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        return
    
    # Make predictions
    data_path = Path("data/exoplanets_combined_cleaned.csv")
    
    try:
        results_df = make_predictions(data_path, models)
        
        # Save results
        output_path = Path("data/results_new.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved predictions to: {output_path}")
        
        # Show sample
        print("\nFirst 10 predictions:")
        print(results_df.head(10).to_string())
        
        # Show prediction distribution
        if 'Ensemble_Prediction' in results_df.columns:
            print("\nPrediction distribution:")
            print(results_df['Ensemble_Prediction'].value_counts())
        
        print("\n" + "="*70)
        print("PREDICTION COMPLETE!")
        print("="*70)
        print(f"\nTotal samples predicted: {len(results_df)}")
        print(f"Output file: {output_path}")
        print("\nTo calculate accuracy, run:")
        print("  python scripts/calculate_accuracy.py")
        
    except Exception as e:
        print(f"\n❌ Error making predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
