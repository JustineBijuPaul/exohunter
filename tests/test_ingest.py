"""
Tests for the data ingestion module.

This module tests the download_dataset function with mocking to avoid actual
network requests during testing.
"""

import pytest
import os
import pandas as pd
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from urllib.error import HTTPError, URLError

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.ingest import download_dataset, load_dataset


class TestDownloadDataset:
    """Test cases for the download_dataset function."""
    
    @patch('data.ingest.urlretrieve')
    def test_download_success(self, mock_urlretrieve, tmp_path):
        """Test successful dataset download."""
        dest_path = tmp_path / "test_dataset.csv"
        mock_urlretrieve.return_value = (str(dest_path), None)
        
        # Create a dummy file to simulate download
        dest_path.write_text("test,data\n1,2\n")
        
        result = download_dataset("http://example.com/dataset.csv", dest_path)
        
        assert result is True
        mock_urlretrieve.assert_called_once_with("http://example.com/dataset.csv", dest_path)
        assert dest_path.exists()
    
    @patch('data.ingest.urlretrieve')
    def test_download_http_error(self, mock_urlretrieve, tmp_path):
        """Test handling of HTTP errors during download."""
        dest_path = tmp_path / "test_dataset.csv"
        mock_urlretrieve.side_effect = HTTPError(
            url="http://example.com/dataset.csv",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None
        )
        
        result = download_dataset("http://example.com/dataset.csv", dest_path)
        
        assert result is False
        assert not dest_path.exists()


class TestLoadDataset:
    """Test cases for the load_dataset function."""
    
    def test_load_valid_csv_file(self, tmp_path, mock_kepler_data):
        """Test loading a valid CSV file."""
        test_file = tmp_path / "test_data.csv"
        mock_kepler_data.to_csv(test_file, index=False)
        
        result = load_dataset(test_file)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) > 0
        assert len(result.columns) > 0
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a file that doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            load_dataset(nonexistent_file)


if __name__ == "__main__":
    pytest.main([__file__])
