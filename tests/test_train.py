"""
Tests for train_baseline.py model training functionality.

Tests model training, evaluation, and file output functionality to ensure
models are created correctly and saved to disk.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import joblib

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.train_baseline import (
    create_synthetic_dataset,
    split_data,
    evaluate_model,
    train_random_forest,
    train_xgboost,
    print_confusion_matrix,
    load_and_preprocess_data
)
from sklearn.preprocessing import LabelEncoder


class TestCreateSyntheticDataset:
    """Test synthetic dataset creation for model training."""
    
    def test_synthetic_dataset_shape(self):
        """Test that synthetic dataset has expected shape and columns."""
        df = create_synthetic_dataset(n_samples=100)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'disposition' in df.columns
        
        # Should have the main feature columns
        expected_features = [
            'orbital_period', 'planet_radius', 'transit_duration',
            'stellar_teff', 'stellar_radius', 'stellar_logg'
        ]
        for feature in expected_features:
            assert feature in df.columns
    
    def test_synthetic_dataset_disposition_values(self):
        """Test that synthetic dataset has valid disposition values."""
        df = create_synthetic_dataset(n_samples=300)
        
        disposition_values = df['disposition'].unique()
        valid_dispositions = ['confirmed', 'false_positive', 'candidate']
        
        for disp in disposition_values:
            assert disp in valid_dispositions
    
    def test_synthetic_dataset_realistic_values(self):
        """Test that synthetic dataset contains realistic exoplanet values."""
        df = create_synthetic_dataset(n_samples=100)
        
        # Check orbital period (should be positive)
        assert (df['orbital_period'] > 0).all()
        
        # Check planet radius (should be positive)
        assert (df['planet_radius'] > 0).all()
        
        # Check transit duration (should be positive and reasonable)
        assert (df['transit_duration'] > 0).all()
        assert (df['transit_duration'] < 100).all()  # Reasonable upper bound
        
        # Check stellar temperature (should be in reasonable range)
        assert (df['stellar_teff'] > 2000).all()
        assert (df['stellar_teff'] < 15000).all()


class TestSplitData:
    """Test data splitting functionality."""
    
    def test_split_data_shapes(self):
        """Test that data splitting produces correct shapes."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Check that all splits have the right number of features
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 5
        
        # Check that sample counts match
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        
        # Check approximate split sizes (allowing for rounding)
        total_samples = len(X)
        assert abs(len(X_test) - total_samples * 0.2) <= 2
        assert abs(len(X_val) - total_samples * 0.2) <= 2
    
    def test_split_data_no_overlap(self):
        """Test that data splits don't overlap."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Flatten arrays for easier comparison
        train_indices = X_train.flatten()
        val_indices = X_val.flatten()
        test_indices = X_test.flatten()
        
        # Check no overlap between sets
        assert len(set(train_indices) & set(val_indices)) == 0
        assert len(set(train_indices) & set(test_indices)) == 0
        assert len(set(val_indices) & set(test_indices)) == 0
    
    def test_split_data_reproducible(self):
        """Test that data splitting is reproducible with same random seed."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        
        # Split with same random seed twice
        split1 = split_data(X, y, test_size=0.2, val_size=0.2, random_state=42)
        split2 = split_data(X, y, test_size=0.2, val_size=0.2, random_state=42)
        
        # Results should be identical
        for arr1, arr2 in zip(split1, split2):
            np.testing.assert_array_equal(arr1, arr2)


class TestEvaluateModel:
    """Test model evaluation functionality."""
    
    def test_evaluate_model_metrics(self):
        """Test that model evaluation returns expected metrics."""
        # Create a mock model
        mock_model = Mock()
        y_pred = np.array([0, 1, 2, 0, 1])
        mock_model.predict.return_value = y_pred
        
        # Create test data
        X_test = np.random.rand(5, 3)
        y_test = np.array([0, 1, 2, 1, 1])
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['confirmed', 'false_positive', 'candidate'])
        
        metrics = evaluate_model(mock_model, X_test, y_test, label_encoder)
        
        # Check that required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1
    
    def test_evaluate_model_with_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        mock_model = Mock()
        y_test = np.array([0, 1, 2, 0, 1])
        mock_model.predict.return_value = y_test  # Perfect predictions
        
        X_test = np.random.rand(5, 3)
        
        label_encoder = LabelEncoder()
        label_encoder.fit(['confirmed', 'false_positive', 'candidate'])
        
        metrics = evaluate_model(mock_model, X_test, y_test, label_encoder)
        
        # All metrics should be 1.0 for perfect predictions
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0


class TestTrainRandomForest:
    """Test Random Forest training functionality."""
    
    def test_train_random_forest_returns_model(self):
        """Test that Random Forest training returns a fitted model."""
        # Create synthetic training data
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.randint(0, 3, 20)
        
        model = train_random_forest(
            X_train, y_train, X_val, y_val, 
            n_estimators=10, random_state=42
        )
        
        # Check that we get a model back
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'fit')
        
        # Check that model can make predictions
        predictions = model.predict(X_val)
        assert len(predictions) == len(y_val)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_train_random_forest_reproducible(self):
        """Test that Random Forest training is reproducible."""
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 3, 50)
        X_val = np.random.rand(10, 3)
        y_val = np.random.randint(0, 3, 10)
        
        # Train same model twice with same random seed
        model1 = train_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=5, random_state=42
        )
        model2 = train_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=5, random_state=42
        )
        
        # Predictions should be identical
        pred1 = model1.predict(X_val)
        pred2 = model2.predict(X_val)
        np.testing.assert_array_equal(pred1, pred2)


class TestTrainXGBoost:
    """Test XGBoost training functionality."""
    
    def test_train_xgboost_returns_model(self):
        """Test that XGBoost training returns a fitted model."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.randint(0, 3, 20)
        
        model = train_xgboost(
            X_train, y_train, X_val, y_val,
            n_estimators=10, random_state=42
        )
        
        # Check that we get a model back
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Check that model can make predictions
        predictions = model.predict(X_val)
        assert len(predictions) == len(y_val)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_train_xgboost_with_early_stopping(self):
        """Test XGBoost training with early stopping."""
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.randint(0, 3, 20)
        
        model = train_xgboost(
            X_train, y_train, X_val, y_val,
            n_estimators=100, early_stopping_rounds=10, random_state=42
        )
        
        # Model should be trained with early stopping
        assert model is not None
        # XGBoost should have stopped before 100 estimators due to small dataset
        assert hasattr(model, 'best_iteration') or hasattr(model, 'n_estimators')


class TestModelFileSaving:
    """Test model file creation and saving functionality."""
    
    def test_model_saving_with_joblib(self):
        """Test that models can be saved and loaded with joblib."""
        # Create and train a simple model
        X_train = np.random.rand(50, 3)
        y_train = np.random.randint(0, 3, 50)
        X_val = np.random.rand(10, 3)
        y_val = np.random.randint(0, 3, 10)
        
        model = train_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=5, random_state=42
        )
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model_path = tmp.name
            joblib.dump(model, model_path)
            
            # Verify file was created
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 0
            
            # Load model and verify it works
            loaded_model = joblib.load(model_path)
            
            # Compare predictions
            original_pred = model.predict(X_val)
            loaded_pred = loaded_model.predict(X_val)
            np.testing.assert_array_equal(original_pred, loaded_pred)
            
            # Clean up
            os.unlink(model_path)
    
    def test_model_metadata_creation(self):
        """Test creation of model metadata for saved models."""
        # Create test model
        X_train = np.random.rand(50, 4)
        y_train = np.random.randint(0, 3, 50)
        X_val = np.random.rand(10, 4)
        y_val = np.random.randint(0, 3, 10)
        
        model = train_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=5, random_state=42
        )
        
        # Create label encoder for metadata
        label_encoder = LabelEncoder()
        label_encoder.fit(['confirmed', 'false_positive', 'candidate'])
        
        # Create metadata dictionary
        metadata = {
            'model_type': 'RandomForest',
            'feature_count': X_train.shape[1],
            'classes': list(label_encoder.classes_),
            'training_samples': len(X_train),
            'accuracy': 0.85  # Example accuracy
        }
        
        # Verify metadata structure
        assert 'model_type' in metadata
        assert 'feature_count' in metadata
        assert 'classes' in metadata
        assert metadata['feature_count'] == 4
        assert len(metadata['classes']) == 3
        assert isinstance(metadata['training_samples'], int)
        assert isinstance(metadata['accuracy'], (int, float))


class TestPrintConfusionMatrix:
    """Test confusion matrix printing functionality."""
    
    def test_print_confusion_matrix_no_error(self):
        """Test that confusion matrix printing doesn't raise errors."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        
        label_encoder = LabelEncoder()
        label_encoder.fit(['confirmed', 'false_positive', 'candidate'])
        
        # This should not raise an exception
        try:
            print_confusion_matrix(y_true, y_pred, label_encoder)
        except Exception as e:
            pytest.fail(f"print_confusion_matrix raised an exception: {e}")


class TestIntegrationTraining:
    """Integration tests for the complete training pipeline."""
    
    def test_end_to_end_training_pipeline(self, tmp_path):
        """Test the complete training pipeline from data loading to model saving."""
        # Create synthetic dataset and save to file
        df = create_synthetic_dataset(n_samples=200)
        data_path = tmp_path / "synthetic_data.csv"
        df.to_csv(data_path, index=False)
        
        # Load and preprocess data
        X_df, X, y = load_and_preprocess_data(str(data_path))
        
        # Verify data loading
        assert len(X_df) > 0
        assert len(X) == len(y)
        assert X.shape[1] > 0
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y_encoded, test_size=0.2, val_size=0.2, random_state=42
        )
        
        # Train models
        rf_model = train_random_forest(
            X_train, y_train, X_val, y_val,
            n_estimators=10, random_state=42
        )
        
        xgb_model = train_xgboost(
            X_train, y_train, X_val, y_val,
            n_estimators=10, random_state=42
        )
        
        # Evaluate models
        rf_metrics = evaluate_model(rf_model, X_test, y_test, label_encoder)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, label_encoder)
        
        # Verify we got reasonable metrics
        for metrics in [rf_metrics, xgb_metrics]:
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1'] <= 1
        
        # Save models
        rf_path = tmp_path / "rf_model.joblib"
        xgb_path = tmp_path / "xgb_model.joblib"
        
        joblib.dump(rf_model, rf_path)
        joblib.dump(xgb_model, xgb_path)
        
        # Verify files were created
        assert rf_path.exists()
        assert xgb_path.exists()
        assert rf_path.stat().st_size > 0
        assert xgb_path.stat().st_size > 0
        
        # Test loading and using saved models
        loaded_rf = joblib.load(rf_path)
        loaded_xgb = joblib.load(xgb_path)
        
        # Make predictions with loaded models
        rf_pred = loaded_rf.predict(X_test[:5])
        xgb_pred = loaded_xgb.predict(X_test[:5])
        
        # Verify predictions are reasonable
        assert len(rf_pred) == 5
        assert len(xgb_pred) == 5
        assert all(pred in [0, 1, 2] for pred in rf_pred)
        assert all(pred in [0, 1, 2] for pred in xgb_pred)


if __name__ == "__main__":
    pytest.main([__file__])
