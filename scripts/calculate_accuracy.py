"""
Calculate accuracy by comparing predictions with actual labels
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_accuracy():
    """Compare predictions in results.csv with actual labels in exoplanets_combined_cleaned.csv"""
    
    print("="*70)
    print("PREDICTION ACCURACY ANALYSIS")
    print("="*70)
    
    # Load the actual data
    data_path = Path("data/exoplanets_combined_cleaned.csv")
    actual_df = pd.read_csv(data_path, low_memory=False)
    print(f"\n✅ Loaded actual data: {len(actual_df)} rows")
    
    # Load the predictions
    results_path = Path("data/results.csv")
    results_df = pd.read_csv(results_path)
    print(f"✅ Loaded predictions: {len(results_df)} rows")
    
    # Match the lengths - use only rows that have predictions
    n_samples = min(len(actual_df), len(results_df))
    print(f"\n📊 Comparing {n_samples} samples")
    
    # Get the actual labels
    actual_labels = actual_df['disposition'].head(n_samples).values
    
    # Get ensemble predictions
    ensemble_predictions = results_df['Ensemble_Prediction'].head(n_samples).values
    
    # Get individual model predictions
    model_names = ['Extra Trees', 'LightGBM', 'Optimized Random Forest', 'Optimized XGBoost']
    
    print("\n" + "="*70)
    print("ACCURACY RESULTS")
    print("="*70)
    
    # Calculate ensemble accuracy
    ensemble_correct = np.sum(actual_labels == ensemble_predictions)
    ensemble_accuracy = (ensemble_correct / len(actual_labels)) * 100
    
    print(f"\n🎯 ENSEMBLE MODEL:")
    print(f"   Correct predictions: {ensemble_correct}/{len(actual_labels)}")
    print(f"   Accuracy: {ensemble_accuracy:.2f}%")
    
    # Calculate individual model accuracies
    print(f"\n📊 INDIVIDUAL MODELS:")
    model_accuracies = {}
    
    for model_name in model_names:
        predictions = results_df[model_name].head(n_samples).values
        correct = np.sum(actual_labels == predictions)
        accuracy = (correct / len(actual_labels)) * 100
        model_accuracies[model_name] = accuracy
        print(f"   {model_name:30s}: {correct:3d}/{len(actual_labels)} = {accuracy:.2f}%")
    
    # Find best and worst models
    best_model = max(model_accuracies.items(), key=lambda x: x[1])
    worst_model = min(model_accuracies.items(), key=lambda x: x[1])
    
    print(f"\n🏆 BEST MODEL: {best_model[0]} ({best_model[1]:.2f}%)")
    print(f"⚠️  WORST MODEL: {worst_model[0]} ({worst_model[1]:.2f}%)")
    
    # Class-wise accuracy analysis
    print("\n" + "="*70)
    print("CLASS-WISE ACCURACY (Ensemble)")
    print("="*70)
    
    classes = np.unique(actual_labels)
    for cls in classes:
        cls_mask = actual_labels == cls
        cls_actual = actual_labels[cls_mask]
        cls_predictions = ensemble_predictions[cls_mask]
        cls_correct = np.sum(cls_actual == cls_predictions)
        cls_accuracy = (cls_correct / len(cls_actual)) * 100
        
        print(f"\n   {cls:15s}:")
        print(f"   - Total samples: {len(cls_actual)}")
        print(f"   - Correct: {cls_correct}")
        print(f"   - Accuracy: {cls_accuracy:.2f}%")
    
    # Confusion matrix analysis
    print("\n" + "="*70)
    print("ENSEMBLE CONFUSION ANALYSIS")
    print("="*70)
    
    misclassified = actual_labels != ensemble_predictions
    misclassified_count = np.sum(misclassified)
    
    print(f"\n❌ Total misclassified: {misclassified_count}/{len(actual_labels)}")
    
    if misclassified_count > 0:
        print(f"\nMisclassification breakdown:")
        for i, (actual, predicted) in enumerate(zip(actual_labels[misclassified], 
                                                      ensemble_predictions[misclassified])):
            row_num = np.where(misclassified)[0][i] + 1
            print(f"   Row {row_num:3d}: Actual={actual:15s} → Predicted={predicted:15s}")
    
    # Model agreement analysis
    print("\n" + "="*70)
    print("MODEL AGREEMENT ANALYSIS")
    print("="*70)
    
    # Count how many models agree on each prediction
    agreement_counts = []
    for idx in range(n_samples):
        predictions = [results_df[model_name].iloc[idx] for model_name in model_names]
        # Count most common prediction
        unique, counts = np.unique(predictions, return_counts=True)
        max_agreement = max(counts)
        agreement_counts.append(max_agreement)
    
    agreement_counts = np.array(agreement_counts)
    
    print(f"\nPredictions with unanimous agreement (4/4 models): {np.sum(agreement_counts == 4)}")
    print(f"Predictions with 3/4 models agreeing: {np.sum(agreement_counts == 3)}")
    print(f"Predictions with 2/4 models agreeing: {np.sum(agreement_counts == 2)}")
    
    # Calculate accuracy for unanimous predictions
    unanimous_mask = agreement_counts == 4
    if np.sum(unanimous_mask) > 0:
        unanimous_correct = np.sum(actual_labels[unanimous_mask] == ensemble_predictions[unanimous_mask])
        unanimous_accuracy = (unanimous_correct / np.sum(unanimous_mask)) * 100
        print(f"\n✨ Accuracy when all models agree: {unanimous_accuracy:.2f}%")
    
    # Save detailed results
    print("\n" + "="*70)
    print("SAVING DETAILED ANALYSIS")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Row': range(1, len(actual_labels) + 1),
        'Actual': actual_labels,
        'Ensemble_Prediction': ensemble_predictions,
        'Correct': actual_labels == ensemble_predictions,
        'Extra_Trees': results_df['Extra Trees'].head(n_samples),
        'LightGBM': results_df['LightGBM'].head(n_samples),
        'Optimized_RF': results_df['Optimized Random Forest'].head(n_samples),
        'Optimized_XGB': results_df['Optimized XGBoost'].head(n_samples),
        'Models_Agreement': agreement_counts
    })
    
    output_path = Path("data/accuracy_analysis.csv")
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved detailed comparison to: {output_path}")
    
    # Create summary report
    summary_path = Path("data/accuracy_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PREDICTION ACCURACY SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total predictions: {len(actual_labels)}\n\n")
        f.write(f"ENSEMBLE ACCURACY: {ensemble_accuracy:.2f}%\n")
        f.write(f"Correct: {ensemble_correct}/{len(actual_labels)}\n\n")
        f.write("INDIVIDUAL MODEL ACCURACIES:\n")
        for model_name, accuracy in sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model_name:30s}: {accuracy:.2f}%\n")
        f.write(f"\nBest Model: {best_model[0]} ({best_model[1]:.2f}%)\n")
        f.write(f"Worst Model: {worst_model[0]} ({worst_model[1]:.2f}%)\n")
        f.write(f"\nMisclassified samples: {misclassified_count}\n")
        f.write(f"Unanimous predictions: {np.sum(unanimous_mask)}\n")
        if np.sum(unanimous_mask) > 0:
            f.write(f"Unanimous accuracy: {unanimous_accuracy:.2f}%\n")
    
    print(f"✅ Saved summary report to: {summary_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    return {
        'ensemble_accuracy': ensemble_accuracy,
        'model_accuracies': model_accuracies,
        'total_samples': len(actual_labels),
        'correct_predictions': ensemble_correct,
        'misclassified': misclassified_count
    }

if __name__ == "__main__":
    results = calculate_accuracy()
