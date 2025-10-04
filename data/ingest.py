"""
Data ingestion module for exoplanet datasets.

This module provides functions to download and load Kepler KOI, K2, and TESS TOI
CSV datasets from URLs or local paths with robust error handling and data cleaning.
"""

import gzip
import os
import time
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import pandas as pd


def download_dataset(
    url: str, 
    dest_path: Union[str, Path], 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> bool:
    """
    Download a dataset from a URL with retry logic and error handling.
    
    This function downloads datasets (typically CSV or gzipped CSV files) from
    astronomical databases like NASA Exoplanet Archive. It includes retry logic
    to handle temporary network failures and proper error handling for various
    failure modes.
    
    Args:
        url: The URL to download the dataset from
        dest_path: Local path where the file should be saved
        max_retries: Maximum number of download attempts (default: 3)
        retry_delay: Delay in seconds between retry attempts (default: 1.0)
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        ValueError: If URL or dest_path is empty/invalid
        OSError: If unable to create destination directory
        
    Examples:
        >>> success = download_dataset(
        ...     "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=...",
        ...     "data/kepler_koi.csv"
        ... )
        >>> if success:
        ...     print("Download completed successfully")
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")
    
    dest_path = Path(dest_path)
    if not dest_path.name:
        raise ValueError("Destination path must include filename")
    
    # Create destination directory if it doesn't exist
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create destination directory: {e}")
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {url} to {dest_path} (attempt {attempt + 1}/{max_retries})")
            urlretrieve(url, dest_path)
            
            # Verify the file was created and has content
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"Successfully downloaded {dest_path.stat().st_size} bytes")
                return True
            else:
                print("Downloaded file is empty or doesn't exist")
                
        except (HTTPError, URLError) as e:
            print(f"Network error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to download after {max_retries} attempts")
                
        except OSError as e:
            print(f"File system error: {e}")
            return False
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
    
    return False


def load_dataset(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load and clean a dataset from a local file path.
    
    This function loads CSV datasets (including gzipped files) and performs
    basic cleaning operations such as standardizing column names, handling
    missing values, and ensuring consistent data types for common exoplanet
    dataset formats (Kepler KOI, K2, TESS TOI).
    
    Args:
        path: Path to the dataset file (CSV or CSV.gz)
        
    Returns:
        pandas.DataFrame or None: Cleaned DataFrame if successful, None if failed
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the file format is not supported
        
    Examples:
        >>> df = load_dataset("data/kepler_koi.csv")
        >>> if df is not None:
        ...     print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        >>> # Loading gzipped file
        >>> df = load_dataset("data/tess_toi.csv.gz")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    if not path.suffix.lower() in ['.csv', '.gz']:
        if path.suffixes[-2:] != ['.csv', '.gz']:
            raise ValueError(f"Unsupported file format: {path.suffix}. Only CSV and CSV.gz files are supported.")
    
    try:
        # Determine if file is gzipped
        is_gzipped = path.suffix.lower() == '.gz' or path.suffixes[-2:] == ['.csv', '.gz']
        
        if is_gzipped:
            print(f"Loading gzipped dataset from {path}")
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, comment='#', low_memory=False)
        else:
            print(f"Loading dataset from {path}")
            df = pd.read_csv(path, comment='#', low_memory=False)
        
        print(f"Initial dataset shape: {df.shape}")
        
        # Clean and standardize the dataset
        df = _clean_dataset(df)
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df
        
    except pd.errors.EmptyDataError:
        print(f"Error: Dataset file is empty: {path}")
        return None
        
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return None
        
    except UnicodeDecodeError as e:
        print(f"Error decoding file (encoding issue): {e}")
        return None
        
    except OSError as e:
        print(f"Error reading file: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return None


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize a dataset DataFrame.
    
    This internal function performs common cleaning operations:
    - Standardizes column names (lowercase, underscores)
    - Removes completely empty rows and columns
    - Converts numeric columns to appropriate types
    - Handles common exoplanet dataset column naming conventions
    
    Args:
        df: Raw DataFrame to clean
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    # Store original column count for reporting
    original_cols = len(df.columns)
    original_rows = len(df)
    
    # Standardize column names
    df.columns = df.columns.str.strip()  # Remove whitespace
    df.columns = df.columns.str.lower()  # Convert to lowercase
    df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
    df.columns = df.columns.str.replace('-', '_')  # Replace hyphens with underscores
    df.columns = df.columns.str.replace('.', '_')  # Replace dots with underscores
    df.columns = df.columns.str.replace('(', '')  # Remove parentheses
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.replace('[', '')  # Remove brackets
    df.columns = df.columns.str.replace(']', '')
    
    # Remove duplicate column names by adding suffix
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup 
                                                        for i in range(sum(cols == dup))]
    df.columns = cols
    
    # Drop completely empty rows and columns
    df = df.dropna(how='all')  # Remove rows where all values are NaN
    df = df.loc[:, ~df.isnull().all()]  # Remove columns where all values are NaN
    
    # Convert numeric columns to appropriate types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric, keeping as object if conversion fails
            df[col] = pd.to_numeric(df[col], errors='ignore')
    
    # Common column name standardizations for exoplanet datasets
    column_mappings = {
        'kepoi_name': 'koi_name',
        'kep_id': 'kepid',
        'ra_deg': 'ra',
        'dec_deg': 'dec',
        'toi_id': 'toi',
        'tic_id': 'tic',
        'planet_radius': 'radius',
        'planet_period': 'period',
        'transit_epoch': 'epoch',
        'transit_duration': 'duration'
    }
    
    df = df.rename(columns=column_mappings)
    
    cleaned_cols = len(df.columns)
    cleaned_rows = len(df)
    
    print(f"Dataset cleaning summary:")
    print(f"  Columns: {original_cols} → {cleaned_cols}")
    print(f"  Rows: {original_rows} → {cleaned_rows}")
    
    return df
