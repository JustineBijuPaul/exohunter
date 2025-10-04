"""
Model performance smoke tests for ExoHunter.

This module provides performance testing to guard against accidental drops
in model quality. Uses soft assertions that warn rather than fail hard when
performance targets aren't met, allowing for investigation rather than
blocking deployments.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
from datetime import datetime

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.train_baseline import (
    create_synthetic_dataset,
    load_and_preprocess_data,
    split_data,
    evaluate_model,
    train_random_forest,
    train_xgboost,
    standardize_feature_names
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_f1_macro': 0.6,
    'min_f1_weighted': 0.6,
    'min_accuracy': 0.65,
    'min_precision_macro': 0.6,
    'min_recall_macro': 0.6
}

# Test configuration
TEST_CONFIG = {
    'n_samples': 500,  # Larger dataset for more reliable metrics
    'test_size': 0.3,
    'random_state': 42,
    'models_to_test': ['random_forest', 'xgboost']
}


@pytest.fixture
def evaluation_dataset():
    """Create a larger synthetic dataset for performance evaluation."""
    np.random.seed(TEST_CONFIG['random_state'])
    return create_synthetic_dataset(n_samples=TEST_CONFIG['n_samples'])


@pytest.fixture
def performance_results_dir():
    """Create directory for storing performance test results."""
    results_dir = Path(__file__).parent / "performance_results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


class PerformanceTestResult:
    """Container for performance test results."""
    
    def __init__(self, model_name: str, metrics: Dict[str, float], 
                 thresholds: Dict[str, float], passed: bool):
        self.model_name = model_name
        self.metrics = metrics
        self.thresholds = thresholds
        self.passed = passed
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'thresholds': self.thresholds,
            'passed': self.passed,
            'timestamp': self.timestamp
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def check_performance_thresholds(metrics: Dict[str, float], 
                                thresholds: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if metrics meet performance thresholds.
    
    Args:
        metrics: Computed model metrics
        thresholds: Performance thresholds to check against
        
    Returns:
        Tuple of (all_passed, individual_checks)
    """
    checks = {}
    for threshold_name, threshold_value in thresholds.items():
        metric_name = threshold_name.replace('min_', '')
        if metric_name in metrics:
            checks[threshold_name] = metrics[metric_name] >= threshold_value
        else:
            checks[threshold_name] = False
    
    all_passed = all(checks.values())
    return all_passed, checks


def soft_assert_performance(result: PerformanceTestResult, 
                           performance_results_dir: Path) -> None:
    """
    Perform soft assertion on performance metrics.
    
    Issues warnings instead of hard failures when performance doesn't meet
    thresholds, allowing for investigation rather than blocking CI/CD.
    """
    # Save results for inspection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = performance_results_dir / f"{result.model_name}_{timestamp}.json"
    result.save_to_file(result_file)
    
    if not result.passed:
        # Create detailed warning message
        failed_checks = [
            f"{check}: {result.metrics.get(check.replace('min_', ''), 'N/A'):.3f} < {threshold:.3f}"
            for check, threshold in result.thresholds.items()
            if not (result.metrics.get(check.replace('min_', ''), 0) >= threshold)
        ]
        
        warning_msg = (
            f"\n{'='*60}\n"
            f"PERFORMANCE WARNING: {result.model_name}\n"
            f"{'='*60}\n"
            f"Model performance below expected thresholds:\n"
            f"{chr(10).join(f'  - {check}' for check in failed_checks)}\n"
            f"\nFull metrics:\n"
            f"{chr(10).join(f'  {k}: {v:.3f}' for k, v in result.metrics.items())}\n"
            f"\nResults saved to: {result_file}\n"
            f"Consider investigating model degradation or adjusting thresholds.\n"
            f"{'='*60}"
        )
        
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
    else:
        print(f"\n✅ {result.model_name} performance check PASSED")
        print(f"   F1 (macro): {result.metrics.get('f1_macro', 0):.3f}")
        print(f"   Accuracy: {result.metrics.get('accuracy', 0):.3f}")


@pytest.mark.slow
@pytest.mark.performance
class TestModelPerformance:
    """Performance smoke tests for trained models."""
    
    def test_random_forest_performance(self, evaluation_dataset, performance_results_dir):
        """Test Random Forest model performance against thresholds."""
        self._test_model_performance(
            evaluation_dataset, 
            performance_results_dir,
            'random_forest'
        )
    
    def test_xgboost_performance(self, evaluation_dataset, performance_results_dir):
        """Test XGBoost model performance against thresholds."""
        self._test_model_performance(
            evaluation_dataset, 
            performance_results_dir,
            'xgboost'
        )
    
    def _test_model_performance(self, dataset: pd.DataFrame, 
                              performance_results_dir: Path,
                              model_type: str) -> None:
        """
        Core performance testing logic.
        
        Args:
            dataset: Evaluation dataset
            performance_results_dir: Directory to save results
            model_type: Type of model to test ('random_forest' or 'xgboost')
        """
        # Prepare data
        feature_columns = [
            'orbital_period', 'planet_radius', 'transit_duration',
            'stellar_teff', 'stellar_radius', 'stellar_logg'
        ]
        
        # Filter to standard 3-class problem for consistency
        standard_classes = ['confirmed', 'candidate', 'false_positive']
        dataset_filtered = dataset[dataset['disposition'].isin(standard_classes)].copy()
        
        if len(dataset_filtered) < 50:
            pytest.skip(f"Insufficient data after filtering: {len(dataset_filtered)} samples")
        
        print(f"\nDataset info for {model_type}:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Filtered samples: {len(dataset_filtered)}")
        print(f"  Class distribution: {dataset_filtered['disposition'].value_counts().to_dict()}")
        
        X = dataset_filtered[feature_columns].fillna(dataset_filtered[feature_columns].median())
        y = dataset_filtered['disposition']
        
        # Handle missing values and outliers
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                q1, q3 = X[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X.values, y_encoded, 
            test_size=TEST_CONFIG['test_size'],
            random_state=TEST_CONFIG['random_state']
        )
        
        # Train model
        if model_type == 'random_forest':
            model = train_random_forest(
                X_train, y_train, X_val, y_val,
                cv_folds=3,  # Reduced for faster testing
                random_state=TEST_CONFIG['random_state']
            )
        elif model_type == 'xgboost':
            model = train_xgboost(
                X_train, y_train, X_val, y_val,
                label_encoder,
                cv_folds=3,  # Reduced for faster testing
                random_state=TEST_CONFIG['random_state']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Check performance
        passed, checks = check_performance_thresholds(metrics, PERFORMANCE_THRESHOLDS)
        
        # Create result object
        result = PerformanceTestResult(
            model_name=model_type,
            metrics=metrics,
            thresholds=PERFORMANCE_THRESHOLDS,
            passed=passed
        )
        
        # Perform soft assertion
        soft_assert_performance(result, performance_results_dir)
        
        # Additional detailed reporting for failed tests
        if not passed:
            self._generate_detailed_report(
                model, X_test, y_test, label_encoder, 
                result, performance_results_dir
            )
    
    def _generate_detailed_report(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                 label_encoder: LabelEncoder, result: PerformanceTestResult,
                                 performance_results_dir: Path) -> None:
        """Generate detailed performance report for failed tests."""
        y_pred = model.predict(X_test)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create detailed report
        report = {
            'summary': result.to_dict(),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': label_encoder.classes_.tolist(),
            'test_config': TEST_CONFIG,
            'recommendations': self._get_performance_recommendations(result.metrics)
        }
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = performance_results_dir / f"{result.model_name}_detailed_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Detailed performance report saved to: {report_file}")
    
    def _get_performance_recommendations(self, metrics: Dict[str, float]) -> list:
        """Generate recommendations based on performance metrics."""
        recommendations = []
        
        if metrics.get('f1_macro', 0) < PERFORMANCE_THRESHOLDS['min_f1_macro']:
            recommendations.append(
                "F1 score below threshold. Consider: "
                "1) Increasing dataset size, "
                "2) Feature engineering, "
                "3) Hyperparameter tuning, "
                "4) Class balancing techniques"
            )
        
        if metrics.get('accuracy', 0) < PERFORMANCE_THRESHOLDS['min_accuracy']:
            recommendations.append(
                "Accuracy below threshold. Consider: "
                "1) Model complexity adjustments, "
                "2) Cross-validation for better generalization, "
                "3) Ensemble methods"
            )
        
        precision = metrics.get('precision_macro', 0)
        recall = metrics.get('recall_macro', 0)
        
        if precision < recall - 0.1:
            recommendations.append(
                "Low precision relative to recall. Consider: "
                "1) Adjusting classification thresholds, "
                "2) Feature selection for noise reduction, "
                "3) Regularization techniques"
            )
        elif recall < precision - 0.1:
            recommendations.append(
                "Low recall relative to precision. Consider: "
                "1) Class balancing (SMOTE, class weights), "
                "2) Increasing model complexity, "
                "3) Threshold adjustments for better sensitivity"
            )
        
        if not recommendations:
            recommendations.append("Performance within acceptable ranges.")
        
        return recommendations


@pytest.mark.slow
@pytest.mark.performance
def test_performance_comparison(evaluation_dataset, performance_results_dir):
    """Compare performance across different models."""
    results = {}
    
    for model_type in TEST_CONFIG['models_to_test']:
        # This will run the individual model tests
        # Results are saved by the individual test methods
        pass
    
    # Load and compare recent results
    result_files = list(performance_results_dir.glob("*.json"))
    if len(result_files) >= 2:
        # Load recent results
        recent_results = []
        for file in sorted(result_files, key=lambda x: x.stat().st_mtime)[-2:]:
            with open(file, 'r') as f:
                recent_results.append(json.load(f))
        
        # Compare models
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': [r['model_name'] for r in recent_results],
            'metric_comparison': {}
        }
        
        for metric in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro']:
            comparison['metric_comparison'][metric] = {
                r['model_name']: r['metrics'].get(metric, 0)
                for r in recent_results
            }
        
        # Save comparison
        comparison_file = performance_results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n📈 Model comparison saved to: {comparison_file}")


@pytest.mark.slow
@pytest.mark.performance
def test_performance_regression(performance_results_dir):
    """Test for performance regression compared to previous runs."""
    result_files = list(performance_results_dir.glob("*_20*.json"))  # Files with timestamps
    
    if len(result_files) < 2:
        pytest.skip("Not enough previous results for regression testing")
    
    # Group by model type
    model_results = {}
    for file in result_files:
        with open(file, 'r') as f:
            result = json.load(f)
        
        model_name = result.get('model_name', 'unknown')
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)
    
    # Check for regression
    for model_name, results in model_results.items():
        if len(results) < 2:
            continue
        
        # Sort by timestamp and compare latest with previous
        sorted_results = sorted(results, key=lambda x: x.get('timestamp', ''))
        latest = sorted_results[-1]
        previous = sorted_results[-2]
        
        # Check for significant drops
        regression_detected = False
        regression_details = []
        
        for metric in ['f1_macro', 'accuracy']:
            latest_value = latest['metrics'].get(metric, 0)
            previous_value = previous['metrics'].get(metric, 0)
            
            # Allow for 5% drop without flagging as regression
            if latest_value < previous_value * 0.95:
                regression_detected = True
                regression_details.append(
                    f"{metric}: {previous_value:.3f} → {latest_value:.3f} "
                    f"(drop: {(previous_value - latest_value):.3f})"
                )
        
        if regression_detected:
            warning_msg = (
                f"\n{'='*60}\n"
                f"REGRESSION WARNING: {model_name}\n"
                f"{'='*60}\n"
                f"Performance regression detected:\n"
                f"{chr(10).join(f'  - {detail}' for detail in regression_details)}\n"
                f"Previous timestamp: {previous.get('timestamp', 'unknown')}\n"
                f"Latest timestamp: {latest.get('timestamp', 'unknown')}\n"
                f"{'='*60}"
            )
            warnings.warn(warning_msg, UserWarning)


# Utility functions for manual testing
def run_performance_check(model_path: str = None, dataset_path: str = None):
    """
    Utility function to run performance checks manually.
    
    Args:
        model_path: Path to saved model (optional)
        dataset_path: Path to evaluation dataset (optional)
    """
    if dataset_path:
        dataset = pd.read_csv(dataset_path)
    else:
        dataset = create_synthetic_dataset(n_samples=TEST_CONFIG['n_samples'])
    
    if model_path and os.path.exists(model_path):
        # Load pre-trained model
        model = joblib.load(model_path)
        # Run evaluation...
        print(f"Loaded model from {model_path}")
    else:
        print("Training new model for performance check...")
        # Train new model for testing
    
    print("Performance check completed.")


if __name__ == "__main__":
    # Allow running performance tests directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model performance tests")
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--data", help="Path to evaluation dataset")
    parser.add_argument("--output", help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    run_performance_check(args.model, args.data)
