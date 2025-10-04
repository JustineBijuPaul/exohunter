"""
Tests for data preprocessing functions.

Tests preprocessing functionality including data transformation, scaling,
and shape validation across different modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.train_baseline import load_and_preprocess_data, standardize_feature_names
from web.streamlit.app import preprocess_data


class TestLoadAndPreprocessData:
    """Test the load_and_preprocess_data function from train_baseline.py."""
    
    def test_load_synthetic_data_shapes(self, tmp_path):
        """Test that synthetic data generation produces expected shapes."""
        # Test with non-existent file path to trigger synthetic data generation
        non_existent_path = str(tmp_path / "non_existent.csv")
        
        X_df, X, y = load_and_preprocess_data(non_existent_path)
        
        # Verify return types
        assert isinstance(X_df, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Verify shapes match
        assert len(X_df) == len(X) == len(y)
        assert X_df.shape[1] == X.shape[1]
        
        # Verify we have some data
        assert len(X_df) > 0
        assert X_df.shape[1] > 0
    
    def test_load_real_data_shapes(self, tmp_path, mock_kepler_data):
        """Test loading and preprocessing real CSV data."""
        # Create a test CSV file
        test_csv = tmp_path / "test_data.csv"
        mock_kepler_data.to_csv(test_csv, index=False)
        
        X_df, X, y = load_and_preprocess_data(str(test_csv))
        
        # Verify return types
        assert isinstance(X_df, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Verify shapes match
        assert len(X_df) == len(X) == len(y)
        assert X_df.shape[1] == X.shape[1]
        
        # Verify we have the expected number of features
        expected_features = [
            'orbital_period', 'planet_radius', 'transit_duration',
            'stellar_teff', 'stellar_radius', 'stellar_logg'
        ]
        available_features = [col for col in expected_features if col in X_df.columns]
        assert len(available_features) == X_df.shape[1]
    
    def test_feature_standardization(self, mock_kepler_data):
        """Test that features are properly standardized."""
        # Create test data with various column names that should be standardized
        test_data = pd.DataFrame({
            'koi_period': [1.0, 2.0, 3.0],
            'koi_prad': [0.5, 1.0, 1.5],
            'koi_duration': [2.0, 4.0, 6.0],
            'koi_stemp': [5000, 6000, 7000],
            'koi_srad': [0.8, 1.0, 1.2],
            'koi_slogg': [4.0, 4.5, 5.0],
            'disposition': ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE']
        })
        
        standardized = standardize_feature_names(test_data)
        
        # Check that columns were renamed to standard names
        expected_columns = [
            'orbital_period', 'planet_radius', 'transit_duration',
            'stellar_teff', 'stellar_radius', 'stellar_logg', 'disposition'
        ]
        for col in expected_columns:
            assert col in standardized.columns
    
    def test_missing_value_handling(self, tmp_path):
        """Test that missing values are handled properly."""
        # Create test data with missing values
        test_data = pd.DataFrame({
            'orbital_period': [1.0, np.nan, 3.0, 4.0],
            'planet_radius': [0.5, 1.0, np.nan, 1.5],
            'transit_duration': [2.0, 4.0, 6.0, np.nan],
            'stellar_teff': [5000, 6000, 7000, 8000],
            'stellar_radius': [0.8, 1.0, 1.2, 1.4],
            'stellar_logg': [4.0, 4.5, 5.0, 5.5],
            'disposition': ['confirmed', 'false_positive', 'candidate', 'confirmed']
        })
        
        test_csv = tmp_path / "test_missing.csv"
        test_data.to_csv(test_csv, index=False)
        
        X_df, X, y = load_and_preprocess_data(str(test_csv))
        
        # Verify no NaN values in the result
        assert not np.isnan(X).any()
        assert not pd.isna(X_df).any().any()
    
    def test_outlier_removal(self, tmp_path):
        """Test that extreme outliers are removed."""
        # Create test data with outliers
        normal_values = np.random.normal(100, 10, 97)  # 97 normal values
        outliers = np.array([500, 600, 700])  # 3 extreme outliers
        test_values = np.concatenate([normal_values, outliers])
        
        test_data = pd.DataFrame({
            'orbital_period': test_values,
            'planet_radius': np.random.normal(1.0, 0.1, 100),
            'transit_duration': np.random.normal(4.0, 0.5, 100),
            'stellar_teff': np.random.normal(6000, 500, 100),
            'stellar_radius': np.random.normal(1.0, 0.1, 100),
            'stellar_logg': np.random.normal(4.5, 0.2, 100),
            'disposition': ['confirmed'] * 50 + ['false_positive'] * 30 + ['candidate'] * 20
        })
        
        test_csv = tmp_path / "test_outliers.csv"
        test_data.to_csv(test_csv, index=False)
        
        X_df, X, y = load_and_preprocess_data(str(test_csv))
        
        # Should have fewer rows after outlier removal
        assert len(X_df) < len(test_data)
        
        # Check that extreme values are removed
        period_values = X_df['orbital_period'].values
        assert period_values.max() < 400  # Should be less than the extreme outliers
    
    def test_invalid_dispositions_filtered(self, tmp_path):
        """Test that invalid disposition values are filtered out."""
        test_data = pd.DataFrame({
            'orbital_period': [1.0, 2.0, 3.0, 4.0, 5.0],
            'planet_radius': [0.5, 1.0, 1.5, 2.0, 2.5],
            'transit_duration': [2.0, 4.0, 6.0, 8.0, 10.0],
            'stellar_teff': [5000, 6000, 7000, 8000, 9000],
            'stellar_radius': [0.8, 1.0, 1.2, 1.4, 1.6],
            'stellar_logg': [4.0, 4.5, 5.0, 5.5, 6.0],
            'disposition': ['confirmed', 'false_positive', 'candidate', 'unknown', 'invalid']
        })
        
        test_csv = tmp_path / "test_invalid_dispositions.csv"
        test_data.to_csv(test_csv, index=False)
        
        X_df, X, y = load_and_preprocess_data(str(test_csv))
        
        # Should only have valid dispositions
        valid_dispositions = ['confirmed', 'candidate', 'false_positive']
        assert all(disp in valid_dispositions for disp in y)
        assert len(y) == 3  # Only 3 valid dispositions from the test data


class TestStreamlitPreprocessData:
    """Test the preprocess_data function from streamlit app."""
    
    def test_preprocess_with_labels(self):
        """Test preprocessing data that includes labels."""
        test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.0, 1.5],
            'feature3': [2.0, 4.0, 6.0],
            'label': ['confirmed', 'false_positive', 'candidate']
        })
        
        X, y = preprocess_data(test_data)
        
        # Verify shapes
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 3
        assert X.shape[1] == 3  # 3 feature columns
        
        # Verify labels are separated
        assert 'label' not in X.columns
        assert list(y) == ['confirmed', 'false_positive', 'candidate']
    
    def test_preprocess_without_labels(self):
        """Test preprocessing data without labels (prediction mode)."""
        test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.0, 1.5],
            'feature3': [2.0, 4.0, 6.0]
        })
        
        X, y = preprocess_data(test_data)
        
        # Verify shapes
        assert isinstance(X, pd.DataFrame)
        assert y is None
        assert len(X) == 3
        assert X.shape[1] == 3
        
        # Verify all columns preserved
        assert list(X.columns) == ['feature1', 'feature2', 'feature3']
    
    def test_preprocess_empty_dataframe(self):
        """Test preprocessing an empty DataFrame."""
        test_data = pd.DataFrame()
        
        X, y = preprocess_data(test_data)
        
        assert isinstance(X, pd.DataFrame)
        assert y is None
        assert len(X) == 0
    
    def test_preprocess_single_row(self):
        """Test preprocessing a single row of data."""
        test_data = pd.DataFrame({
            'feature1': [1.0],
            'feature2': [0.5],
            'feature3': [2.0],
            'label': ['confirmed']
        })
        
        X, y = preprocess_data(test_data)
        
        assert len(X) == 1
        assert len(y) == 1
        assert X.shape[1] == 3
        assert y.iloc[0] == 'confirmed'


class TestStandardizeFeatureNames:
    """Test the standardize_feature_names function."""
    
    def test_koi_to_standard_mapping(self):
        """Test KOI column name mappings."""
        test_df = pd.DataFrame({
            'koi_period': [1.0],
            'koi_prad': [0.5],
            'koi_duration': [2.0],
            'koi_stemp': [5000],
            'koi_srad': [0.8],
            'koi_slogg': [4.0]
        })
        
        result = standardize_feature_names(test_df)
        
        expected_columns = [
            'orbital_period', 'planet_radius', 'transit_duration',
            'stellar_teff', 'stellar_radius', 'stellar_logg'
        ]
        
        for col in expected_columns:
            assert col in result.columns
    
    def test_alternative_name_mappings(self):
        """Test alternative column name mappings."""
        test_df = pd.DataFrame({
            'period': [1.0],
            'radius': [0.5],
            'duration': [2.0],
            'stellar_rad': [0.8]
        })
        
        result = standardize_feature_names(test_df)
        
        assert 'orbital_period' in result.columns
        assert 'planet_radius' in result.columns
        assert 'transit_duration' in result.columns
        assert 'stellar_radius' in result.columns
    
    def test_no_changes_for_standard_names(self):
        """Test that already standard names are unchanged."""
        test_df = pd.DataFrame({
            'orbital_period': [1.0],
            'planet_radius': [0.5],
            'transit_duration': [2.0],
            'stellar_teff': [5000],
            'stellar_radius': [0.8],
            'stellar_logg': [4.0]
        })
        
        result = standardize_feature_names(test_df)
        
        # All columns should remain the same
        assert list(result.columns) == list(test_df.columns)
    
    def test_original_dataframe_unchanged(self):
        """Test that the original DataFrame is not modified."""
        test_df = pd.DataFrame({
            'koi_period': [1.0],
            'koi_prad': [0.5]
        })
        
        original_columns = list(test_df.columns)
        _ = standardize_feature_names(test_df)
        
        # Original DataFrame should be unchanged
        assert list(test_df.columns) == original_columns


class TestDataShapeValidation:
    """Test data shape validation across preprocessing functions."""
    
    def test_consistent_shapes_across_functions(self, mock_kepler_data):
        """Test that different preprocessing functions maintain shape consistency."""
        # Test with streamlit preprocessing
        X_streamlit, y_streamlit = preprocess_data(mock_kepler_data)
        
        # Verify basic shape properties
        if y_streamlit is not None:
            assert len(X_streamlit) == len(y_streamlit)
        
        assert X_streamlit.shape[0] > 0
        assert X_streamlit.shape[1] > 0
    
    def test_shape_preservation_after_standardization(self):
        """Test that standardization preserves DataFrame shape."""
        test_df = pd.DataFrame({
            'koi_period': [1.0, 2.0, 3.0],
            'koi_prad': [0.5, 1.0, 1.5],
            'other_col': ['a', 'b', 'c']
        })
        
        original_shape = test_df.shape
        standardized = standardize_feature_names(test_df)
        
        # Shape should be preserved
        assert standardized.shape == original_shape
    
    def test_feature_count_validation(self, tmp_path):
        """Test that the number of features is as expected."""
        # Create minimal valid dataset
        test_data = pd.DataFrame({
            'orbital_period': [1.0, 2.0],
            'planet_radius': [0.5, 1.0],
            'transit_duration': [2.0, 4.0],
            'stellar_teff': [5000, 6000],
            'stellar_radius': [0.8, 1.0],
            'stellar_logg': [4.0, 4.5],
            'disposition': ['confirmed', 'false_positive']
        })
        
        test_csv = tmp_path / "test_features.csv"
        test_data.to_csv(test_csv, index=False)
        
        X_df, X, y = load_and_preprocess_data(str(test_csv))
        
        # Should have 6 features (the main feature columns)
        assert X_df.shape[1] == 6
        assert X.shape[1] == 6


if __name__ == "__main__":
    pytest.main([__file__])
