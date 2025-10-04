"""
Model evaluation utilities for exoplanet classification.

This module provides comprehensive evaluation functions including metrics calculation,
confusion matrices, ROC curves, calibration plots, and report generation for both
traditional ML models and deep learning models.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelBinarizer
import warnings

# Optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    class_names: Optional[List[str]] = None,
    output_dir: Union[str, Path] = "evaluation_results",
    model_name: str = "model",
    save_plots: bool = True,
    show_plots: bool = False,
    include_calibration: bool = True,
    include_precision_recall: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a classification model.
    
    This function evaluates a trained model and generates:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix plot
    - ROC curves for each class
    - Precision-recall curves (optional)
    - Calibration curves (optional)
    - Classification report JSON
    
    Args:
        model: Trained model with predict() and predict_proba() methods
        X_test: Test features
        y_test: True test labels
        class_names: Names of classes (inferred if None)
        output_dir: Directory to save evaluation artifacts
        model_name: Name for the model (used in filenames)
        save_plots: Whether to save plot figures
        show_plots: Whether to display plots
        include_calibration: Whether to generate calibration plots
        include_precision_recall: Whether to generate precision-recall curves
        
    Returns:
        Dictionary containing all evaluation metrics and paths to saved artifacts
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> results = evaluate_model(model, X_test, y_test, 
        ...                         class_names=['confirmed', 'candidate', 'false_positive'])
        >>> print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert inputs to numpy arrays if needed
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Get predictions
    try:
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
    except Exception as e:
        # Handle TensorFlow/Keras models
        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
            # TabularMLP case
            y_pred = model.predict(pd.DataFrame(X_test))
            y_proba = model.model.predict(model.scaler.transform(X_test))
        elif hasattr(model, 'predict') and 'tensorflow' in str(type(model)):
            # Direct Keras model
            y_proba = model.predict(X_test)
            y_pred = np.argmax(y_proba, axis=1)
            # Convert numeric predictions back to class names if needed
            if class_names and len(class_names) == y_proba.shape[1]:
                y_pred = np.array([class_names[i] for i in y_pred])
        else:
            raise e
    
    # Infer class names if not provided
    if class_names is None:
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        class_names = [str(label) for label in sorted(unique_labels)]
    
    # Calculate basic metrics
    metrics = calculate_metrics(y_test, y_pred, class_names)
    
    # Generate and save confusion matrix
    cm_path = None
    if save_plots:
        cm_path = plot_confusion_matrix(
            y_test, y_pred, class_names, 
            save_path=output_dir / f"{model_name}_confusion_matrix.png",
            show=show_plots
        )
    
    # Generate ROC curves if probabilities are available
    roc_path = None
    if y_proba is not None and save_plots:
        roc_path = plot_roc_curves(
            y_test, y_proba, class_names,
            save_path=output_dir / f"{model_name}_roc_curves.png",
            show=show_plots
        )
    
    # Generate precision-recall curves
    pr_path = None
    if y_proba is not None and include_precision_recall and save_plots:
        pr_path = plot_precision_recall_curves(
            y_test, y_proba, class_names,
            save_path=output_dir / f"{model_name}_precision_recall.png",
            show=show_plots
        )
    
    # Generate calibration curves
    cal_path = None
    if y_proba is not None and include_calibration and save_plots:
        cal_path = plot_calibration_curves(
            y_test, y_proba, class_names,
            save_path=output_dir / f"{model_name}_calibration.png",
            show=show_plots
        )
    
    # Calculate class-wise metrics
    class_metrics = calculate_class_metrics(y_test, y_pred, y_proba, class_names)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Save classification report as JSON
    report_path = output_dir / f"{model_name}_classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Compile results
    results = {
        'metrics': metrics,
        'class_metrics': class_metrics,
        'classification_report': report,
        'artifacts': {
            'confusion_matrix': str(cm_path) if cm_path else None,
            'roc_curves': str(roc_path) if roc_path else None,
            'precision_recall': str(pr_path) if pr_path else None,
            'calibration': str(cal_path) if cal_path else None,
            'report_json': str(report_path)
        },
        'predictions': {
            'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
            'y_proba': y_proba.tolist() if y_proba is not None else None
        }
    }
    
    # Save complete results
    results_path = output_dir / f"{model_name}_evaluation_results.json"
    with open(results_path, 'w') as f:
        # Create a copy without predictions for cleaner JSON
        results_summary = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(results_summary, f, indent=2, default=str)
    
    results['artifacts']['summary'] = str(results_path)
    
    print(f"✅ Evaluation completed for {model_name}")
    print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
    print(f"📈 Macro F1: {metrics['macro_f1']:.4f}")
    print(f"📋 Artifacts saved to: {output_dir}")
    
    return results


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0)
    }


def calculate_class_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray],
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Calculate per-class metrics including calibration if probabilities available."""
    class_metrics = {}
    
    # Get per-class precision, recall, f1
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=class_names)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=class_names)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=class_names)
    
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        }
        
        # Add calibration metrics if probabilities available
        if y_proba is not None:
            try:
                # Convert to binary problem for this class
                y_binary = (y_true == class_name).astype(int)
                y_prob_binary = y_proba[:, i] if y_proba.shape[1] > i else np.zeros(len(y_true))
                
                # Brier score (lower is better)
                brier = brier_score_loss(y_binary, y_prob_binary)
                class_metrics[class_name]['brier_score'] = float(brier)
                
                # Average precision for this class
                if np.sum(y_binary) > 0:  # Only if class exists in test set
                    avg_precision = average_precision_score(y_binary, y_prob_binary)
                    class_metrics[class_name]['average_precision'] = float(avg_precision)
                
            except Exception as e:
                warnings.warn(f"Could not calculate calibration metrics for class {class_name}: {e}")
    
    return class_metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
) -> Optional[Path]:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
) -> Optional[Path]:
    """Plot ROC curves for each class (one-vs-rest)."""
    # Convert labels to binary format
    lb = LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)
    
    if len(class_names) == 2:
        # Binary classification
        y_true_binary = y_true_binary.ravel()
        y_proba_binary = y_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
    else:
        # Multi-class classification
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            if i < y_proba.shape[1]:
                # One-vs-rest for this class
                y_binary = (y_true == class_name).astype(int)
                y_prob = y_proba[:, i]
                
                fpr, tpr, _ = roc_curve(y_binary, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 ROC curves saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
) -> Optional[Path]:
    """Plot precision-recall curves for each class."""
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        if i < y_proba.shape[1]:
            # One-vs-rest for this class
            y_binary = (y_true == class_name).astype(int)
            y_prob = y_proba[:, i]
            
            if np.sum(y_binary) > 0:  # Only if class exists in test set
                precision, recall, _ = precision_recall_curve(y_binary, y_prob)
                avg_precision = average_precision_score(y_binary, y_prob)
                
                plt.plot(recall, precision, linewidth=2,
                        label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Precision-recall curves saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def plot_calibration_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False,
    n_bins: int = 10
) -> Optional[Path]:
    """Plot calibration curves for each class."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        if i < y_proba.shape[1]:
            # One-vs-rest for this class
            y_binary = (y_true == class_name).astype(int)
            y_prob = y_proba[:, i]
            
            if np.sum(y_binary) > 0:  # Only if class exists in test set
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_binary, y_prob, n_bins=n_bins
                    )
                    
                    axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", 
                               linewidth=2, label=f'{class_name}')
                    axes[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    axes[i].set_xlabel('Mean Predicted Probability')
                    axes[i].set_ylabel('Fraction of Positives')
                    axes[i].set_title(f'Calibration - {class_name}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Calibration - {class_name}')
            else:
                axes[i].text(0.5, 0.5, 'No positive samples', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Calibration - {class_name}')
    
    plt.suptitle('Calibration Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Calibration curves saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return save_path


def compare_models(
    results_list: List[Dict[str, Any]],
    model_names: List[str],
    output_dir: Union[str, Path] = "model_comparison",
    save_plot: bool = True,
    show_plot: bool = False
) -> Dict[str, Any]:
    """
    Compare multiple model evaluation results.
    
    Args:
        results_list: List of evaluation results from evaluate_model()
        model_names: Names of the models being compared
        output_dir: Directory to save comparison artifacts
        save_plot: Whether to save comparison plots
        show_plot: Whether to display plots
        
    Returns:
        Dictionary with comparison metrics and plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    metrics_df = pd.DataFrame([
        {
            'model': name,
            **results['metrics']
        }
        for results, name in zip(results_list, model_names)
    ])
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(metrics_df['model'], metrics_df['accuracy'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1 Score comparison
    axes[0, 1].bar(metrics_df['model'], metrics_df['macro_f1'])
    axes[0, 1].set_title('Macro F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision comparison
    axes[1, 0].bar(metrics_df['model'], metrics_df['macro_precision'])
    axes[1, 0].set_title('Macro Precision Comparison')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Recall comparison
    axes[1, 1].bar(metrics_df['model'], metrics_df['macro_recall'])
    axes[1, 1].set_title('Macro Recall Comparison')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = None
    if save_plot:
        comparison_path = output_dir / "model_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison saved to: {comparison_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Save comparison data
    comparison_data = {
        'metrics_comparison': metrics_df.to_dict('records'),
        'best_model': {
            'by_accuracy': model_names[metrics_df['accuracy'].idxmax()],
            'by_f1': model_names[metrics_df['macro_f1'].idxmax()],
            'by_precision': model_names[metrics_df['macro_precision'].idxmax()],
            'by_recall': model_names[metrics_df['macro_recall'].idxmax()]
        }
    }
    
    comparison_json_path = output_dir / "model_comparison.json"
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    return {
        'comparison_data': comparison_data,
        'metrics_df': metrics_df,
        'artifacts': {
            'comparison_plot': str(comparison_path) if comparison_path else None,
            'comparison_data': str(comparison_json_path)
        }
    }


# Example usage function
def example_evaluation():
    """Example of how to use the evaluation functions."""
    print("Example evaluation usage:")
    print("""
    # Basic evaluation
    from exohunter.models.evaluate import evaluate_model
    
    # Train your model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=['confirmed', 'candidate', 'false_positive'],
        output_dir='evaluation_results',
        model_name='random_forest',
        include_calibration=True,
        include_precision_recall=True
    )
    
    # Compare multiple models
    from exohunter.models.evaluate import compare_models
    
    results_list = [rf_results, xgb_results, mlp_results]
    model_names = ['RandomForest', 'XGBoost', 'MLP']
    
    comparison = compare_models(
        results_list=results_list,
        model_names=model_names,
        output_dir='model_comparison'
    )
    """)


if __name__ == "__main__":
    example_evaluation()
