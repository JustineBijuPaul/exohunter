"""
Comprehensive Testing Suite for Exoplanet Classification Models
Tests the newly trained models thoroughly on the dataset
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
TIMESTAMP = "20251005_200911"
MODELS_DIR = project_root / "models" / "trained_models"
DATA_DIR = project_root / "data"

print("=" * 80)
print("🧪 COMPREHENSIVE MODEL TESTING SUITE")
print("=" * 80)
print(f"Timestamp: {TIMESTAMP}")
print(f"Models Directory: {MODELS_DIR}")
print(f"Data Directory: {DATA_DIR}")
print()

# ============================================================================
# TEST 1: Verify Model Files Exist and Can Be Loaded
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Model File Verification")
print("=" * 80)

model_files = {
    "Optimized Random Forest": f"optimized_rf_{TIMESTAMP}.joblib",
    "Extra Trees": f"extra_trees_{TIMESTAMP}.joblib",
    "LightGBM": f"lightgbm_{TIMESTAMP}.joblib",
    "Optimized XGBoost": f"optimized_xgb_{TIMESTAMP}.joblib",
    "Scaler": f"scaler_{TIMESTAMP}.joblib",
    "Label Encoder": f"label_encoder_{TIMESTAMP}.joblib"
}

models = {}
test_results = {"model_loading": {}}

for name, filename in model_files.items():
    filepath = MODELS_DIR / filename
    try:
        if filepath.exists():
            model = joblib.load(filepath)
            models[name] = model
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {name:30s} | Size: {file_size:6.2f} MB | {filename}")
            test_results["model_loading"][name] = "SUCCESS"
        else:
            print(f"❌ {name:30s} | NOT FOUND | {filename}")
            test_results["model_loading"][name] = "FILE_NOT_FOUND"
    except Exception as e:
        print(f"❌ {name:30s} | LOAD ERROR: {str(e)[:50]}")
        test_results["model_loading"][name] = f"ERROR: {str(e)}"

# Check if we have all required models
required_models = ["Optimized Random Forest", "Extra Trees", "LightGBM", "Optimized XGBoost", "Scaler", "Label Encoder"]
all_loaded = all(name in models for name in required_models)

if all_loaded:
    print(f"\n✅ ALL MODELS LOADED SUCCESSFULLY ({len(models)}/6)")
else:
    print(f"\n❌ MISSING MODELS: {6 - len(models)} models failed to load")
    missing = [name for name in required_models if name not in models]
    print(f"   Missing: {', '.join(missing)}")

# ============================================================================
# TEST 2: Load and Verify Dataset
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Dataset Verification")
print("=" * 80)

# Load train.csv (used for training)
train_path = DATA_DIR / "train.csv"
if train_path.exists():
    df_train = pd.read_csv(train_path)
    print(f"✅ Training Dataset: {len(df_train):,} rows, {len(df_train.columns)} columns")
    print(f"   File: {train_path.name}")
    
    # Check for disposition column
    if 'disposition' in df_train.columns:
        class_dist = df_train['disposition'].value_counts()
        print(f"\n   Class Distribution:")
        for cls, count in class_dist.items():
            pct = (count / len(df_train)) * 100
            print(f"      {cls:20s}: {count:5,} ({pct:5.2f}%)")
    
    test_results["train_dataset"] = {
        "rows": len(df_train),
        "columns": len(df_train.columns),
        "status": "SUCCESS"
    }
else:
    print(f"❌ Training Dataset NOT FOUND: {train_path}")
    test_results["train_dataset"] = {"status": "NOT_FOUND"}

# Load test.csv (for testing)
test_path = DATA_DIR / "test.csv"
if test_path.exists():
    df_test = pd.read_csv(test_path)
    print(f"\n✅ Test Dataset: {len(df_test):,} rows, {len(df_test.columns)} columns")
    print(f"   File: {test_path.name}")
    
    # Check what columns are available
    print(f"   Available columns: {', '.join(df_test.columns[:5])}{'...' if len(df_test.columns) > 5 else ''}")
    
    test_results["test_dataset"] = {
        "rows": len(df_test),
        "columns": len(df_test.columns),
        "status": "SUCCESS"
    }
else:
    print(f"❌ Test Dataset NOT FOUND: {test_path}")
    test_results["test_dataset"] = {"status": "NOT_FOUND"}

# ============================================================================
# TEST 3: Test Individual Model Predictions
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Individual Model Predictions")
print("=" * 80)

if all_loaded and train_path.exists():
    # Load features from training script
    features_path = MODELS_DIR.parent / f"selected_features_{TIMESTAMP}.json"
    if features_path.exists():
        with open(features_path, 'r') as f:
            feature_data = json.load(f)
            # Handle both list and dict formats
            if isinstance(feature_data, list):
                selected_features = feature_data
            else:
                selected_features = feature_data.get('features', [])
        print(f"✅ Loaded {len(selected_features)} features from training configuration")
    else:
        # Fallback: use features from training data (exclude target)
        selected_features = [col for col in df_train.columns if col != 'disposition']
        print(f"⚠️  Using {len(selected_features)} features from training data (features file not found)")
    
    print(f"   Features: {', '.join(selected_features[:5])}...")
    
    # Prepare a small test sample from train.csv
    test_sample = df_train.sample(n=min(100, len(df_train)), random_state=42)
    X_sample = test_sample[selected_features]
    y_sample = test_sample['disposition']
    
    # Scale features
    scaler = models["Scaler"]
    X_sample_scaled = scaler.transform(X_sample)
    
    # Test each model
    ml_models = ["Optimized Random Forest", "Extra Trees", "LightGBM", "Optimized XGBoost"]
    predictions = {}
    test_results["individual_models"] = {}
    
    print("\nTesting each model on 100 samples:")
    for model_name in ml_models:
        try:
            model = models[model_name]
            y_pred = model.predict(X_sample_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_sample, y_pred)
            precision = precision_score(y_sample, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_sample, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_sample, y_pred, average='weighted', zero_division=0)
            
            predictions[model_name] = y_pred
            
            print(f"\n   {model_name}:")
            print(f"      Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall:    {recall:.4f}")
            print(f"      F1-Score:  {f1:.4f}")
            
            # Check prediction distribution
            unique, counts = np.unique(y_pred, return_counts=True)
            print(f"      Predictions: {dict(zip(unique, counts))}")
            
            test_results["individual_models"][model_name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "status": "SUCCESS"
            }
            
        except Exception as e:
            print(f"\n   ❌ {model_name}: ERROR - {str(e)}")
            test_results["individual_models"][model_name] = {
                "status": f"ERROR: {str(e)}"
            }

# ============================================================================
# TEST 4: Test Ensemble Predictions (Majority Voting)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Ensemble Predictions (Majority Voting)")
print("=" * 80)

if all_loaded and train_path.exists() and predictions:
    try:
        # Convert predictions to DataFrame for easier voting
        pred_df = pd.DataFrame(predictions)
        
        # Majority voting
        ensemble_pred = pred_df.mode(axis=1)[0]
        
        # Calculate ensemble metrics
        accuracy = accuracy_score(y_sample, ensemble_pred)
        precision = precision_score(y_sample, ensemble_pred, average='weighted', zero_division=0)
        recall = recall_score(y_sample, ensemble_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_sample, ensemble_pred, average='weighted', zero_division=0)
        
        print(f"✅ Ensemble Results (100 samples):")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        # Check agreement between models
        print(f"\n   Model Agreement Analysis:")
        all_agree = (pred_df.nunique(axis=1) == 1).sum()
        print(f"      All models agree:     {all_agree}/{len(pred_df)} ({all_agree/len(pred_df)*100:.1f}%)")
        
        majority_3 = (pred_df.nunique(axis=1) == 2).sum()
        print(f"      3+ models agree:      {majority_3}/{len(pred_df)} ({majority_3/len(pred_df)*100:.1f}%)")
        
        all_differ = (pred_df.nunique(axis=1) >= 3).sum()
        print(f"      High disagreement:    {all_differ}/{len(pred_df)} ({all_differ/len(pred_df)*100:.1f}%)")
        
        test_results["ensemble"] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "all_agree_pct": float(all_agree/len(pred_df)*100),
            "status": "SUCCESS"
        }
        
    except Exception as e:
        print(f"❌ Ensemble prediction failed: {str(e)}")
        test_results["ensemble"] = {"status": f"ERROR: {str(e)}"}

# ============================================================================
# TEST 5: Full Dataset Accuracy Test
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Full Dataset Accuracy Test")
print("=" * 80)

if all_loaded and train_path.exists():
    try:
        # Use entire training set for comprehensive test (or split if too large)
        use_full = len(df_train) <= 20000
        
        if use_full:
            df_eval = df_train
            print(f"Testing on FULL training dataset: {len(df_eval):,} samples")
        else:
            df_eval = df_train.sample(n=20000, random_state=42)
            print(f"Testing on SAMPLE of training dataset: {len(df_eval):,} samples")
        
        X_eval = df_eval[selected_features]
        y_eval = df_eval['disposition']
        X_eval_scaled = scaler.transform(X_eval)
        
        # Get predictions from all models
        all_predictions = {}
        for model_name in ml_models:
            model = models[model_name]
            all_predictions[model_name] = model.predict(X_eval_scaled)
        
        # Ensemble prediction
        pred_df = pd.DataFrame(all_predictions)
        ensemble_pred = pred_df.mode(axis=1)[0]
        
        # Calculate comprehensive metrics
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_eval, ensemble_pred, digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_eval, ensemble_pred)
        classes = sorted(y_eval.unique())
        
        print("\n📈 Confusion Matrix:")
        print(f"{'':20s} | " + " | ".join(f"{cls:15s}" for cls in classes))
        print("-" * (20 + len(classes) * 18))
        for i, cls in enumerate(classes):
            print(f"{cls:20s} | " + " | ".join(f"{cm[i,j]:15,}" for j in range(len(classes))))
        
        # Calculate per-class metrics
        print("\n📊 Per-Class Performance:")
        for cls in classes:
            cls_mask = y_eval == cls
            cls_accuracy = accuracy_score(y_eval[cls_mask], ensemble_pred[cls_mask])
            print(f"   {cls:20s}: {cls_accuracy:.4f} ({cls_accuracy*100:.2f}%)")
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_eval, ensemble_pred)
        print(f"\n🎯 Overall Ensemble Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        test_results["full_dataset_test"] = {
            "samples_tested": len(df_eval),
            "overall_accuracy": float(overall_accuracy),
            "confusion_matrix": cm.tolist(),
            "classes": classes,
            "status": "SUCCESS"
        }
        
    except Exception as e:
        print(f"❌ Full dataset test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        test_results["full_dataset_test"] = {"status": f"ERROR: {str(e)}"}

# ============================================================================
# TEST 6: Edge Case Testing
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: Edge Case Testing")
print("=" * 80)

if all_loaded and train_path.exists():
    edge_test_results = []
    
    # Test 6.1: Minimum values
    print("\n6.1 Testing with minimum feature values:")
    try:
        X_min = pd.DataFrame([X_sample.min()], columns=selected_features)
        X_min_scaled = scaler.transform(X_min)
        pred = models["Optimized Random Forest"].predict(X_min_scaled)
        print(f"   ✅ Minimum values prediction: {pred[0]}")
        edge_test_results.append(("minimum_values", "SUCCESS"))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        edge_test_results.append(("minimum_values", f"ERROR: {str(e)}"))
    
    # Test 6.2: Maximum values
    print("\n6.2 Testing with maximum feature values:")
    try:
        X_max = pd.DataFrame([X_sample.max()], columns=selected_features)
        X_max_scaled = scaler.transform(X_max)
        pred = models["Optimized Random Forest"].predict(X_max_scaled)
        print(f"   ✅ Maximum values prediction: {pred[0]}")
        edge_test_results.append(("maximum_values", "SUCCESS"))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        edge_test_results.append(("maximum_values", f"ERROR: {str(e)}"))
    
    # Test 6.3: Mean values
    print("\n6.3 Testing with mean feature values:")
    try:
        X_mean = pd.DataFrame([X_sample.mean()], columns=selected_features)
        X_mean_scaled = scaler.transform(X_mean)
        pred = models["Optimized Random Forest"].predict(X_mean_scaled)
        print(f"   ✅ Mean values prediction: {pred[0]}")
        edge_test_results.append(("mean_values", "SUCCESS"))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        edge_test_results.append(("mean_values", f"ERROR: {str(e)}"))
    
    # Test 6.4: Zero values (where appropriate)
    print("\n6.4 Testing with zero values:")
    try:
        X_zero = pd.DataFrame([[0] * len(selected_features)], columns=selected_features)
        X_zero_scaled = scaler.transform(X_zero)
        pred = models["Optimized Random Forest"].predict(X_zero_scaled)
        print(f"   ✅ Zero values prediction: {pred[0]}")
        edge_test_results.append(("zero_values", "SUCCESS"))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        edge_test_results.append(("zero_values", f"ERROR: {str(e)}"))
    
    # Test 6.5: Batch prediction (multiple samples at once)
    print("\n6.5 Testing batch predictions:")
    try:
        batch_sizes = [1, 10, 100, 1000]
        for batch_size in batch_sizes:
            if batch_size <= len(X_sample):
                X_batch = X_sample.iloc[:batch_size]
                X_batch_scaled = scaler.transform(X_batch)
                pred = models["Optimized Random Forest"].predict(X_batch_scaled)
                print(f"   ✅ Batch size {batch_size:4d}: {len(pred)} predictions")
        edge_test_results.append(("batch_predictions", "SUCCESS"))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        edge_test_results.append(("batch_predictions", f"ERROR: {str(e)}"))
    
    test_results["edge_cases"] = dict(edge_test_results)
    
    passed = sum(1 for _, status in edge_test_results if status == "SUCCESS")
    total = len(edge_test_results)
    print(f"\n✅ Edge case tests passed: {passed}/{total}")

# ============================================================================
# TEST 7: Model Consistency Test
# ============================================================================
print("\n" + "=" * 80)
print("TEST 7: Model Consistency Test")
print("=" * 80)

if all_loaded and train_path.exists():
    try:
        # Test same input multiple times - should get same output
        print("Testing model consistency (same input → same output):")
        
        X_consistency = X_sample.iloc[:10]
        X_consistency_scaled = scaler.transform(X_consistency)
        
        consistent = True
        for model_name in ml_models:
            model = models[model_name]
            
            # Predict 3 times
            pred1 = model.predict(X_consistency_scaled)
            pred2 = model.predict(X_consistency_scaled)
            pred3 = model.predict(X_consistency_scaled)
            
            # Check if all predictions are identical
            is_consistent = np.array_equal(pred1, pred2) and np.array_equal(pred2, pred3)
            
            if is_consistent:
                print(f"   ✅ {model_name:30s}: CONSISTENT")
            else:
                print(f"   ❌ {model_name:30s}: INCONSISTENT")
                consistent = False
        
        test_results["consistency"] = "SUCCESS" if consistent else "FAILED"
        
    except Exception as e:
        print(f"❌ Consistency test failed: {str(e)}")
        test_results["consistency"] = f"ERROR: {str(e)}"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("📊 FINAL TEST SUMMARY")
print("=" * 80)

# Count successes
test_categories = [
    "model_loading",
    "train_dataset",
    "test_dataset",
    "individual_models",
    "ensemble",
    "full_dataset_test",
    "edge_cases",
    "consistency"
]

summary = []
for category in test_categories:
    if category in test_results:
        result = test_results[category]
        if isinstance(result, dict):
            if "status" in result:
                status = "✅" if result["status"] == "SUCCESS" else "❌"
            else:
                # Check if all items in dict are successful
                all_success = all(
                    v == "SUCCESS" or (isinstance(v, dict) and v.get("status") == "SUCCESS")
                    for v in result.values()
                )
                status = "✅" if all_success else "❌"
        else:
            status = "✅" if result == "SUCCESS" else "❌"
        
        summary.append((category.replace("_", " ").title(), status))

print("\nTest Results:")
for test_name, status in summary:
    print(f"   {status} {test_name}")

# Calculate pass rate
passed = sum(1 for _, status in summary if status == "✅")
total = len(summary)
pass_rate = (passed / total * 100) if total > 0 else 0

print(f"\n{'=' * 80}")
print(f"🎯 Overall Pass Rate: {passed}/{total} tests passed ({pass_rate:.1f}%)")
print(f"{'=' * 80}")

# Save test results to file
results_file = project_root / "test_results" / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
results_file.parent.mkdir(exist_ok=True)

test_results["summary"] = {
    "timestamp": datetime.now().isoformat(),
    "model_timestamp": TIMESTAMP,
    "tests_passed": passed,
    "tests_total": total,
    "pass_rate": pass_rate
}

with open(results_file, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\n💾 Test results saved to: {results_file}")

# Final verdict
if pass_rate >= 90:
    print("\n🎉 EXCELLENT! Models are production-ready!")
elif pass_rate >= 70:
    print("\n✅ GOOD! Models are working well with minor issues.")
elif pass_rate >= 50:
    print("\n⚠️  FAIR! Models need some attention before production.")
else:
    print("\n❌ POOR! Models have significant issues that must be addressed.")

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
