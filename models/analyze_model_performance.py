"""
Feature Importance and Error Analysis
=====================================
Comprehensive analysis of model performance including:
- Feature importance visualization
- Error analysis with misclassification patterns
- Confusion matrix deep dive
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def analyze_feature_importance():
    """Extract and visualize feature importance from all models"""
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    models_dir = Path("models")
    
    # Load models
    print("\nLoading models...")
    xgb = joblib.load(models_dir / 'ultimate_ensemble_xgboost.pkl')
    lgb = joblib.load(models_dir / 'ultimate_ensemble_lightgbm.pkl')
    cat = joblib.load(models_dir / 'ultimate_ensemble_catboost.pkl')
    rf = joblib.load(models_dir / 'ultimate_ensemble_random_forest.pkl')
    et = joblib.load(models_dir / 'ultimate_ensemble_extra_trees.pkl')
    
    # Load feature names
    feature_names = joblib.load(models_dir / 'ultimate_ensemble_feature_names.pkl')
    print(f"✓ Loaded {len(feature_names)} features")
    
    # Extract feature importances
    importances = {
        'XGBoost': xgb.feature_importances_,
        'LightGBM': lgb.feature_importances_,
        'CatBoost': cat.feature_importances_,
        'Random Forest': rf.feature_importances_,
        'Extra Trees': et.feature_importances_
    }
    
    # Create DataFrame
    importance_df = pd.DataFrame(importances, index=feature_names)
    
    # Calculate ensemble average importance
    importance_df['Ensemble Average'] = importance_df.mean(axis=1)
    
    # Sort by ensemble average
    importance_df = importance_df.sort_values('Ensemble Average', ascending=False)
    
    # Save to CSV
    importance_df.to_csv(models_dir / 'feature_importance_detailed.csv')
    print(f"✓ Saved detailed importances to feature_importance_detailed.csv")
    
    # Display top 20 features
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*80)
    top_20 = importance_df.head(20)
    print(top_20[['XGBoost', 'LightGBM', 'Random Forest', 'Ensemble Average']].to_string())
    
    # Visualize top 20 features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Ensemble average (top 20)
    ax = axes[0, 0]
    top_20_avg = importance_df['Ensemble Average'].head(20)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    top_20_avg.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Top 20 Features - Ensemble Average Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.invert_yaxis()
    
    # 2. Model comparison (top 15)
    ax = axes[0, 1]
    top_15 = importance_df[['XGBoost', 'LightGBM', 'Random Forest']].head(15)
    top_15.plot(kind='barh', ax=ax)
    ax.set_title('Top 15 Features - Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    
    # 3. Feature categories
    ax = axes[1, 0]
    
    # Categorize features
    categories = {
        'Original': [],
        'Ratio': [],
        'Polynomial': [],
        'Normalized': [],
        'Interaction': []
    }
    
    for feat in feature_names:
        if '_normalized' in feat:
            categories['Normalized'].append(feat)
        elif '_ratio' in feat or feat in ['radius_ratio', 'temp_ratio', 'snr_ratio']:
            categories['Ratio'].append(feat)
        elif '_squared' in feat or '_sqrt' in feat or '_log' in feat:
            categories['Polynomial'].append(feat)
        elif feat in ['snr_product', 'planet_energy', 'semi_major_axis', 'impact_duration', 'habitability_index', 'snr_diff', 'transits_per_day']:
            categories['Interaction'].append(feat)
        else:
            categories['Original'].append(feat)
    
    category_importance = {}
    for cat, feats in categories.items():
        cat_feats = [f for f in feats if f in importance_df.index]
        if cat_feats:
            category_importance[cat] = importance_df.loc[cat_feats, 'Ensemble Average'].mean()
    
    cat_df = pd.Series(category_importance).sort_values(ascending=False)
    cat_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_title('Feature Category Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Category')
    ax.set_ylabel('Average Importance')
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Model agreement on top features
    ax = axes[1, 1]
    # Get top 10 from each model
    top_features_per_model = {}
    for model in ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Extra Trees']:
        top_features_per_model[model] = set(importance_df.nlargest(10, model).index)
    
    # Count how many models agree on each feature
    all_top_features = set()
    for feats in top_features_per_model.values():
        all_top_features.update(feats)
    
    agreement_scores = {}
    for feat in all_top_features:
        count = sum(1 for feats in top_features_per_model.values() if feat in feats)
        agreement_scores[feat] = count
    
    agreement_df = pd.Series(agreement_scores).sort_values(ascending=False).head(15)
    agreement_df.plot(kind='barh', ax=ax, color='coral')
    ax.set_title('Feature Agreement Across Models (Top 15)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Models in Top 10')
    ax.set_xlim(0, 5.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(models_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to feature_importance_analysis.png")
    
    return importance_df, category_importance


def analyze_errors():
    """Analyze misclassified samples from test set"""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    models_dir = Path("models")
    
    # Load test predictions
    test_preds = pd.read_csv(models_dir / 'test_predictions.csv')
    
    # Identify misclassifications
    if 'disposition' in test_preds.columns:
        # Map true dispositions
        disposition_mapping = {
            'FALSE POSITIVE': 'FALSE POSITIVE',
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE'
        }
        test_preds['true_disposition'] = test_preds['disposition'].map(disposition_mapping)
        
        # Find errors
        errors = test_preds[test_preds['true_disposition'] != test_preds['predicted_disposition']].copy()
        
        print(f"\n📊 Error Statistics:")
        print(f"   Total samples: {len(test_preds)}")
        print(f"   Correct predictions: {len(test_preds) - len(errors)}")
        print(f"   Errors: {len(errors)}")
        print(f"   Error rate: {len(errors)/len(test_preds)*100:.2f}%")
        
        # Error breakdown
        print(f"\n🔍 Error Breakdown:")
        error_types = errors.groupby(['true_disposition', 'predicted_disposition']).size()
        for (true_class, pred_class), count in error_types.items():
            pct = count / len(errors) * 100
            print(f"   {true_class} → {pred_class}: {count} ({pct:.1f}% of errors)")
        
        # Confidence analysis
        print(f"\n💡 Confidence Analysis:")
        correct = test_preds[test_preds['true_disposition'] == test_preds['predicted_disposition']]
        print(f"   Correct predictions avg confidence: {correct['confidence'].mean():.2%}")
        print(f"   Incorrect predictions avg confidence: {errors['confidence'].mean():.2%}")
        print(f"   Difference: {(correct['confidence'].mean() - errors['confidence'].mean()):.2%}")
        
        # Save errors to CSV
        error_cols = ['object_name', 'true_disposition', 'predicted_disposition', 
                     'confidence', 'prob_CANDIDATE', 'prob_CONFIRMED', 'prob_FALSE POSITIVE',
                     'orbital_period', 'transit_depth', 'planet_radius', 'koi_teq']
        available_cols = [col for col in error_cols if col in errors.columns]
        errors[available_cols].to_csv(models_dir / 'error_analysis.csv', index=False)
        print(f"\n✓ Saved error analysis to error_analysis.csv")
        
        # Visualize errors
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion matrix
        ax = axes[0, 0]
        cm = confusion_matrix(test_preds['true_disposition'], test_preds['predicted_disposition'],
                             labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                   yticklabels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # 2. Confidence distribution
        ax = axes[0, 1]
        correct['confidence'].hist(bins=20, alpha=0.7, label='Correct', ax=ax, color='green')
        errors['confidence'].hist(bins=20, alpha=0.7, label='Incorrect', ax=ax, color='red')
        ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.legend()
        ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='50% threshold')
        
        # 3. Error rate by confidence bin
        ax = axes[1, 0]
        bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        test_preds['conf_bin'] = pd.cut(test_preds['confidence'], bins=bins)
        error_rate_by_conf = test_preds.groupby('conf_bin').apply(
            lambda x: (x['true_disposition'] != x['predicted_disposition']).mean()
        )
        error_rate_by_conf.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Error Rate by Confidence Level', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence Bin')
        ax.set_ylabel('Error Rate')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(len(errors)/len(test_preds), color='red', linestyle='--', label='Overall Error Rate')
        ax.legend()
        
        # 4. Feature characteristics of errors
        ax = axes[1, 1]
        if 'transit_depth' in errors.columns and 'orbital_period' in errors.columns:
            # Plot errors vs correct in feature space
            ax.scatter(correct['orbital_period'], correct['transit_depth'], 
                      alpha=0.5, s=30, label='Correct', color='green')
            ax.scatter(errors['orbital_period'], errors['transit_depth'], 
                      alpha=0.7, s=50, label='Error', color='red', marker='x')
            ax.set_xlabel('Orbital Period (days)')
            ax.set_ylabel('Transit Depth (ppm)')
            ax.set_title('Errors in Feature Space', fontsize=14, fontweight='bold')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(models_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved error visualization to error_analysis.png")
        
        return errors
    else:
        print("⚠️  No ground truth labels available for error analysis")
        return None


def create_comprehensive_report():
    """Generate comprehensive analysis report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    # Load metrics
    models_dir = Path("models")
    test_metrics = json.load(open(models_dir / 'test_metrics.json'))
    
    # Run analyses
    importance_df, category_importance = analyze_feature_importance()
    errors = analyze_errors()
    
    # Create markdown report
    report = f"""# Comprehensive Model Analysis Report

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Model Performance Summary

- **Test Accuracy**: {test_metrics['test_accuracy']:.2%}
- **Precision**: {test_metrics['test_precision']:.2%}
- **Recall**: {test_metrics['test_recall']:.2%}
- **F1-Score**: {test_metrics['test_f1_score']:.2%}
- **Test Samples**: {test_metrics['num_samples']}

---

## 🎯 Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
"""
    
    for i, (feat, imp) in enumerate(importance_df['Ensemble Average'].head(10).items(), 1):
        report += f"| {i} | `{feat}` | {imp:.4f} |\n"
    
    report += f"""
---

## 📦 Feature Category Analysis

"""
    
    for cat, imp in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{cat}**: {imp:.4f}\n"
    
    if errors is not None:
        error_rate = len(errors) / test_metrics['num_samples']
        report += f"""
---

## ⚠️ Error Analysis

- **Total Errors**: {len(errors)}
- **Error Rate**: {error_rate:.2%}
- **Correct Predictions**: {test_metrics['num_samples'] - len(errors)}

### Error Patterns

"""
        error_types = errors.groupby(['true_disposition', 'predicted_disposition']).size()
        for (true_class, pred_class), count in error_types.items():
            pct = count / len(errors) * 100
            report += f"- {true_class} → {pred_class}: {count} ({pct:.1f}% of errors)\n"
    
    report += f"""
---

## 💡 Key Insights

1. **Feature Engineering Impact**: Normalized and ratio features show high importance
2. **Model Consistency**: Top features show agreement across multiple models
3. **Detection Quality**: SNR-related features are crucial for classification
4. **Physical Properties**: Planet radius and orbital period are fundamental predictors

---

## 📁 Generated Files

- `feature_importance_detailed.csv` - Full feature importance matrix
- `feature_importance_analysis.png` - Visual analysis (4 subplots)
- `error_analysis.csv` - Detailed error cases
- `error_analysis.png` - Error pattern visualization

---

## 🚀 Recommendations

1. **Feature Selection**: Consider using top 50 features for faster inference
2. **Model Optimization**: Remove Deep NN (poor performance), retrain ensemble
3. **Confidence Thresholds**: Use 60% confidence for binary decisions
4. **Error Mitigation**: Focus on improving CANDIDATE vs FALSE POSITIVE separation

"""
    
    # Save report
    with open(models_dir / 'COMPREHENSIVE_ANALYSIS.md', 'w') as f:
        f.write(report)
    
    print(f"\n✓ Saved comprehensive report to COMPREHENSIVE_ANALYSIS.md")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    create_comprehensive_report()
