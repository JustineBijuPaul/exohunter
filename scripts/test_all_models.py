"""
Test all models on train.csv dataset and calculate comprehensive accuracy
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("🎯 TESTING ALL MODELS ON TRAIN.CSV DATASET")
print("=" * 80)

# Configuration
TIMESTAMP = "20251005_200911"
MODELS_DIR = project_root / "models" / "trained_models"
DATA_FILE = project_root / "data" / "train.csv"

print(f"\nConfiguration:")
print(f"  Models: {TIMESTAMP}")
print(f"  Dataset: {DATA_FILE.name}")
print(f"  Models Directory: {MODELS_DIR}")

# ============================================================================
# Load Dataset
# ============================================================================
print("\n" + "=" * 80)
print("📊 LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")

# Features used in training
features = [
    'transit_depth', 'planet_radius', 'koi_teq', 'koi_insol', 'stellar_teff',
    'stellar_radius', 'koi_smass', 'koi_slogg', 'koi_count', 'koi_num_transits',
    'koi_max_sngle_ev', 'koi_max_mult_ev', 'impact_parameter', 'transit_duration', 'st_dist'
]

# Check if all features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    sys.exit(1)

print(f"✅ All {len(features)} features present")

# Extract features and target
X = df[features]
y = df['disposition']

print(f"\nClass distribution:")
class_counts = y.value_counts()
for cls, count in class_counts.items():
    pct = (count / len(y)) * 100
    print(f"  {cls:20s}: {count:6,} ({pct:5.2f}%)")

# ============================================================================
# Load Models
# ============================================================================
print("\n" + "=" * 80)
print("🔧 LOADING MODELS")
print("=" * 80)

models = {}
model_files = {
    "Optimized Random Forest": f"optimized_rf_{TIMESTAMP}.joblib",
    "Extra Trees": f"extra_trees_{TIMESTAMP}.joblib",
    "LightGBM": f"lightgbm_{TIMESTAMP}.joblib",
    "Optimized XGBoost": f"optimized_xgb_{TIMESTAMP}.joblib"
}

# Load scaler
scaler_path = MODELS_DIR / f"scaler_{TIMESTAMP}.joblib"
print(f"\nLoading scaler...")
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded scaler: {type(scaler).__name__}")
else:
    print(f"❌ Scaler not found: {scaler_path}")
    sys.exit(1)

# Scale features
print(f"Scaling features...")
X_scaled = scaler.transform(X)
print(f"✅ Features scaled")

# Load models
print(f"\nLoading ML models...")
for name, filename in model_files.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        try:
            models[name] = joblib.load(model_path)
            print(f"✅ Loaded: {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {str(e)}")
    else:
        print(f"❌ Not found: {name}")

print(f"\n✅ Total models loaded: {len(models)}/4")

if len(models) == 0:
    print("❌ No models loaded. Exiting.")
    sys.exit(1)

# ============================================================================
# Test Individual Models
# ============================================================================
print("\n" + "=" * 80)
print("🧪 TESTING INDIVIDUAL MODELS")
print("=" * 80)

individual_predictions = {}
individual_results = {}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Make predictions
        y_pred = model.predict(X_scaled)
        individual_predictions[model_name] = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Store results
        individual_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\n📈 Per-Class Performance:")
        unique_classes = sorted(y.unique())
        for cls in unique_classes:
            cls_mask = y == cls
            cls_acc = accuracy_score(y[cls_mask], y_pred[cls_mask])
            cls_count = cls_mask.sum()
            print(f"  {cls:20s}: {cls_acc:.4f} ({cls_acc*100:.2f}%) - {cls_count:,} samples")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=unique_classes)
        print(f"\n🔢 Confusion Matrix:")
        print(f"{'':20s} | " + " | ".join(f"{cls[:15]:15s}" for cls in unique_classes))
        print("-" * (20 + len(unique_classes) * 18))
        for i, cls in enumerate(unique_classes):
            print(f"{cls:20s} | " + " | ".join(f"{cm[i,j]:15,}" for j in range(len(unique_classes))))
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Test Ensemble (Majority Voting)
# ============================================================================
print("\n" + "=" * 80)
print("🎯 TESTING ENSEMBLE (MAJORITY VOTING)")
print("=" * 80)

if len(individual_predictions) >= 2:
    # Create DataFrame of predictions for voting
    pred_df = pd.DataFrame(individual_predictions)
    
    # Majority voting
    ensemble_pred = pred_df.mode(axis=1)[0]
    
    # Calculate metrics
    accuracy = accuracy_score(y, ensemble_pred)
    precision = precision_score(y, ensemble_pred, average='weighted', zero_division=0)
    recall = recall_score(y, ensemble_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, ensemble_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Ensemble Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"\n📈 Ensemble Per-Class Performance:")
    unique_classes = sorted(y.unique())
    for cls in unique_classes:
        cls_mask = y == cls
        cls_acc = accuracy_score(y[cls_mask], ensemble_pred[cls_mask])
        cls_count = cls_mask.sum()
        print(f"  {cls:20s}: {cls_acc:.4f} ({cls_acc*100:.2f}%) - {cls_count:,} samples")
    
    # Confusion matrix
    cm = confusion_matrix(y, ensemble_pred, labels=unique_classes)
    print(f"\n🔢 Ensemble Confusion Matrix:")
    print(f"{'':20s} | " + " | ".join(f"{cls[:15]:15s}" for cls in unique_classes))
    print("-" * (20 + len(unique_classes) * 18))
    for i, cls in enumerate(unique_classes):
        print(f"{cls:20s} | " + " | ".join(f"{cm[i,j]:15,}" for j in range(len(unique_classes))))
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y, ensemble_pred, digits=4))
    
    # Model agreement analysis
    print(f"\n🤝 Model Agreement Analysis:")
    all_agree = (pred_df.nunique(axis=1) == 1).sum()
    print(f"  All models agree:      {all_agree:,}/{len(pred_df):,} ({all_agree/len(pred_df)*100:.2f}%)")
    
    majority_3 = (pred_df.nunique(axis=1) == 2).sum()
    print(f"  3+ models agree:       {majority_3:,}/{len(pred_df):,} ({majority_3/len(pred_df)*100:.2f}%)")
    
    high_disagree = (pred_df.nunique(axis=1) >= 3).sum()
    print(f"  High disagreement:     {high_disagree:,}/{len(pred_df):,} ({high_disagree/len(pred_df)*100:.2f}%)")
    
else:
    print("⚠️ Not enough models for ensemble (need at least 2)")

# ============================================================================
# Summary Comparison
# ============================================================================
print("\n" + "=" * 80)
print("📊 ACCURACY SUMMARY - ALL MODELS")
print("=" * 80)

print("\n{:30s} | {:>10s} | {:>10s} | {:>10s} | {:>10s}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
))
print("-" * 80)

# Sort by accuracy
sorted_models = sorted(individual_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for model_name, metrics in sorted_models:
    print("{:30s} | {:>9.2f}% | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(
        model_name,
        metrics['accuracy'] * 100,
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ))

if len(individual_predictions) >= 2:
    print("-" * 80)
    print("{:30s} | {:>9.2f}% | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(
        "🏆 ENSEMBLE (Majority Vote)",
        accuracy * 100,
        precision,
        recall,
        f1
    ))

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("🎯 FINAL SUMMARY")
print("=" * 80)

print(f"\n📊 Dataset:")
print(f"  Total samples: {len(df):,}")
print(f"  Features: {len(features)}")
print(f"  Classes: {len(unique_classes)}")

print(f"\n🔧 Models tested: {len(models)}/4")

if sorted_models:
    best_model = sorted_models[0]
    print(f"\n🏆 Best Individual Model:")
    print(f"  Name: {best_model[0]}")
    print(f"  Accuracy: {best_model[1]['accuracy']*100:.2f}%")

if len(individual_predictions) >= 2:
    print(f"\n🎯 Ensemble Accuracy: {accuracy*100:.2f}%")
    
    # Compare with best individual
    if accuracy > best_model[1]['accuracy']:
        improvement = (accuracy - best_model[1]['accuracy']) * 100
        print(f"  ✅ Ensemble is {improvement:.2f}% better than best individual model")
    else:
        diff = (best_model[1]['accuracy'] - accuracy) * 100
        print(f"  ⚠️ Best individual model is {diff:.2f}% better than ensemble")

print("\n" + "=" * 80)
print("✅ TESTING COMPLETE!")
print("=" * 80)
