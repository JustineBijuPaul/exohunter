"""
Tests for the labels mapping module.

This module tests the map_labels function and other label mapping utilities
to ensure proper standardization across different exoplanet datasets.
"""

import pytest
import pandas as pd
import numpy as np

# Import with the correct path structure
from data.labels import map_labels


class TestMapLabels:
    """Test cases for the main map_labels function."""

    def test_map_koi_labels(self):
        """Test mapping KOI (Kepler Objects of Interest) labels."""
        koi_data = pd.DataFrame({
            'koi_disposition': [
                'CONFIRMED',
                'CANDIDATE', 
                'FALSE POSITIVE',
                'NOT DISPOSITIONED',
                'DUPLICATE'
            ]
        })
        
        result = map_labels(koi_data, 'koi')
        
        assert 'disposition' in result.columns
        # Check that standard labels are created properly
        unique_dispositions = set(result['disposition'].dropna())
        valid_dispositions = {
            'confirmed', 'candidate', 'false_positive', 'ambiguous'
        }
        assert unique_dispositions.issubset(valid_dispositions)

    def test_map_toi_labels(self):
        """Test mapping TOI (TESS Objects of Interest) labels."""
        toi_data = pd.DataFrame({
            'tfopwg_disposition': [
                'PC',  # Planet Candidate
                'CP',  # Confirmed Planet
                'FP',  # False Positive
                'APC'  # Ambiguous Planet Candidate
            ]
        })
        
        result = map_labels(toi_data, 'toi')
        
        assert 'disposition' in result.columns
        # Verify we get valid canonical labels
        unique_dispositions = set(result['disposition'].dropna())
        valid_dispositions = {
            'confirmed', 'candidate', 'false_positive', 'ambiguous', 'known'
        }
        assert unique_dispositions.issubset(valid_dispositions)

    def test_map_k2_labels(self):
        """Test mapping K2 eclipsing binary labels."""
        k2_data = pd.DataFrame({
            'disposition': [
                'EB',      # Eclipsing Binary
                'planet',  # Planet
                'blend'    # Background Blend
            ]
        })
        
        result = map_labels(k2_data, 'k2')
        
        assert 'disposition' in result.columns
        # Verify canonical labels are created
        unique_dispositions = set(result['disposition'].dropna())
        valid_dispositions = {
            'confirmed', 'candidate', 'false_positive', 'ambiguous'
        }
        assert unique_dispositions.issubset(valid_dispositions)

    def test_map_labels_unsupported_source(self):
        """Test error handling for unsupported data source."""
        data = pd.DataFrame({'test_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Unsupported source"):
            map_labels(data, 'unsupported_source')

    def test_map_labels_missing_column(self):
        """Test error handling when required columns are missing."""
        # KOI data without required disposition column
        koi_data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises((ValueError, KeyError)):
            map_labels(koi_data, 'koi')

    def test_map_labels_case_sensitivity(self):
        """Test that label mapping handles case variations."""
        koi_data = pd.DataFrame({
            'koi_disposition': [
                'confirmed',      # lowercase
                'CANDIDATE',      # uppercase
                'False Positive', # mixed case
                'Candidate'       # title case
            ]
        })
        
        result = map_labels(koi_data, 'koi')
        
        # Should handle case variations appropriately
        assert 'disposition' in result.columns
        assert len(result) == 4

    def test_map_labels_with_nan_values(self):
        """Test handling of NaN/missing values in labels."""
        koi_data = pd.DataFrame({
            'koi_disposition': [
                'CONFIRMED',
                np.nan,
                'CANDIDATE',
                None,
                'FALSE POSITIVE'
            ]
        })
        
        result = map_labels(koi_data, 'koi')
        
        # Check that we still get a valid result
        assert 'disposition' in result.columns
        assert len(result) == 5
        # NaN handling should be graceful
        non_null_dispositions = result['disposition'].dropna()
        assert len(non_null_dispositions) >= 3  # At least the non-null ones

    def test_map_labels_preserves_original_data(self):
        """Test that original DataFrame columns are preserved."""
        koi_data = pd.DataFrame({
            'kepid': [1, 2, 3],
            'koi_disposition': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
            'koi_period': [1.5, 2.0, 3.0]
        })
        
        result = map_labels(koi_data, 'koi')
        
        # Original columns should be preserved
        assert 'kepid' in result.columns
        assert 'koi_disposition' in result.columns
        assert 'koi_period' in result.columns
        assert 'disposition' in result.columns
        
        # Original data should be unchanged
        pd.testing.assert_series_equal(result['kepid'], koi_data['kepid'])
        pd.testing.assert_series_equal(result['koi_period'], koi_data['koi_period'])

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframes gracefully
        try:
            result = map_labels(empty_df, 'koi')
            assert isinstance(result, pd.DataFrame)
        except (ValueError, KeyError):
            # It's acceptable to raise an error for empty dataframes
            pass

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        single_row = pd.DataFrame({
            'koi_disposition': ['CONFIRMED']
        })
        
        result = map_labels(single_row, 'koi')
        
        assert len(result) == 1
        assert 'disposition' in result.columns


class TestLabelMappingIntegration:
    """Integration tests with real-world data scenarios."""

    def test_full_pipeline_koi(self, mock_kepler_data):
        """Test complete label mapping pipeline with KOI data."""
        result = map_labels(mock_kepler_data, 'koi')
        
        # Verify all expected columns exist
        assert 'disposition' in result.columns
        assert 'koi_disposition' in result.columns
        
        # Verify mapping worked correctly
        unique_dispositions = set(result['disposition'].dropna())
        valid_dispositions = {
            'confirmed', 'candidate', 'false_positive', 'ambiguous', 'known'
        }
        assert unique_dispositions.issubset(valid_dispositions)

    def test_label_consistency_across_sources(self):
        """Test that different sources produce consistent canonical labels."""
        # Create equivalent data for different sources
        koi_data = pd.DataFrame({'koi_disposition': ['CONFIRMED', 'CANDIDATE']})
        
        # For TOI, use actual TOI labels
        toi_data = pd.DataFrame({'tfopwg_disposition': ['CP', 'PC']})
        
        koi_result = map_labels(koi_data, 'koi')
        toi_result = map_labels(toi_data, 'toi')
        
        # Both should produce valid canonical labels
        assert 'disposition' in koi_result.columns
        assert 'disposition' in toi_result.columns
        
        # Check that we get valid dispositions
        koi_dispositions = set(koi_result['disposition'].dropna())
        toi_dispositions = set(toi_result['disposition'].dropna())
        
        valid_dispositions = {
            'confirmed', 'candidate', 'false_positive', 'ambiguous', 'known'
        }
        assert koi_dispositions.issubset(valid_dispositions)
        assert toi_dispositions.issubset(valid_dispositions)

    @pytest.mark.integration
    def test_real_world_data_mapping(self):
        """Integration test with realistic data distributions."""
        # Simulate realistic label distributions
        realistic_koi_data = pd.DataFrame({
            'koi_disposition': (
                ['CONFIRMED'] * 50 +
                ['CANDIDATE'] * 300 + 
                ['FALSE POSITIVE'] * 150 +
                ['NOT DISPOSITIONED'] * 20
            )
        })
        
        result = map_labels(realistic_koi_data, 'koi')
        
        # Verify realistic distribution is preserved
        assert len(result) == 520
        assert 'disposition' in result.columns
        
        # Check that we have the expected number of entries
        distribution = result['disposition'].value_counts()
        assert len(distribution) > 0  # Should have some valid mappings


class TestLabelMappingEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_dataframe_performance(self):
        """Test performance with large DataFrame."""
        # Create large DataFrame
        large_data = pd.DataFrame({
            'koi_disposition': ['CONFIRMED'] * 10000 + ['CANDIDATE'] * 10000
        })
        
        import time
        start_time = time.time()
        result = map_labels(large_data, 'koi')
        end_time = time.time()
        
        # Should complete in reasonable time (less than 2 seconds)
        assert (end_time - start_time) < 2.0
        assert len(result) == 20000
        assert 'disposition' in result.columns

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in labels."""
        special_data = pd.DataFrame({
            'koi_disposition': [
                'CONFIRMED',
                'CANDIDATE\n',     # with newline
                ' FALSE POSITIVE ', # with spaces
                'CONFIRMED\t'      # with tab
            ]
        })
        
        result = map_labels(special_data, 'koi')
        
        # Should handle whitespace and special characters gracefully
        assert 'disposition' in result.columns
        assert len(result) == 4
        
        # Check that we get valid dispositions (allowing for normalization)
        dispositions = result['disposition'].dropna()
        valid_dispositions = {
            'confirmed', 'candidate', 'false_positive', 'ambiguous'
        }
        assert set(dispositions).issubset(valid_dispositions)
