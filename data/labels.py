"""
Label mapping utilities for exoplanet datasets.

This module provides functions to standardize label/disposition columns across
different exoplanet survey datasets (Kepler KOI, K2, TESS TOI) into a unified
classification system.
"""

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np


def map_labels(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Convert dataset-specific label columns to unified disposition column.
    
    This function takes a DataFrame from a specific exoplanet survey and maps
    the source-specific labels/dispositions to a canonical set of values:
    - 'confirmed': Confirmed exoplanet
    - 'candidate': Planet candidate requiring further validation
    - 'false_positive': False positive detection
    - 'ambiguous': Unclear or conflicting classification
    - 'known': Previously known exoplanet (for some datasets)
    
    Args:
        df: DataFrame containing the dataset
        source: Source dataset identifier ('koi', 'toi', 'k2')
        
    Returns:
        pandas.DataFrame: DataFrame with added 'disposition' column
        
    Raises:
        ValueError: If source is not supported or required columns are missing
        
    Examples:
        >>> import pandas as pd
        >>> koi_df = pd.DataFrame({
        ...     'koi_disposition': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        ... })
        >>> result = map_labels(koi_df, 'koi')
        >>> result['disposition'].tolist()
        ['confirmed', 'candidate', 'false_positive']
        
        >>> toi_df = pd.DataFrame({
        ...     'tfopwg_disposition': ['PC', 'FP', 'CP']
        ... })
        >>> result = map_labels(toi_df, 'toi')
        >>> result['disposition'].tolist()
        ['confirmed', 'false_positive', 'candidate']
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        return df.copy()
    
    source = source.lower().strip()
    supported_sources = ['koi', 'toi', 'k2']
    
    if source not in supported_sources:
        raise ValueError(f"Unsupported source '{source}'. Supported sources: {supported_sources}")
    
    # Create a copy to avoid modifying original DataFrame
    result_df = df.copy()
    
    # Apply source-specific mapping
    if source == 'koi':
        result_df = _map_koi_labels(result_df)
    elif source == 'toi':
        result_df = _map_toi_labels(result_df)
    elif source == 'k2':
        result_df = _map_k2_labels(result_df)
    
    return result_df


def _map_koi_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Kepler Objects of Interest (KOI) labels to canonical dispositions.
    
    KOI datasets typically use 'koi_disposition' column with values like:
    - CONFIRMED, CANDIDATE, FALSE POSITIVE, NOT DISPOSITIONED
    """
    # Common column name variations for KOI disposition
    disposition_cols = ['koi_disposition', 'disposition', 'koi_disp']
    source_col = None
    
    for col in disposition_cols:
        if col in df.columns:
            source_col = col
            break
    
    if source_col is None:
        raise ValueError(f"No KOI disposition column found. Expected one of: {disposition_cols}")
    
    # KOI mapping dictionary
    koi_mapping = {
        'CONFIRMED': 'confirmed',
        'CANDIDATE': 'candidate',
        'FALSE POSITIVE': 'false_positive',
        'NOT DISPOSITIONED': 'ambiguous',
        # Alternative spellings/formats
        'confirmed': 'confirmed',
        'candidate': 'candidate',
        'false positive': 'false_positive',
        'false_positive': 'false_positive',
        'not dispositioned': 'ambiguous',
        'ambiguous': 'ambiguous',
        # Handle NaN/null values
        np.nan: 'ambiguous',
        None: 'ambiguous'
    }
    
    # Apply mapping
    df['disposition'] = df[source_col].astype(str).str.upper().map(
        {k.upper() if k is not None and not pd.isna(k) else k: v 
         for k, v in koi_mapping.items()}
    )
    
    # Handle any unmapped values
    unmapped_mask = df['disposition'].isna()
    if unmapped_mask.any():
        unmapped_values = df.loc[unmapped_mask, source_col].unique()
        print(f"Warning: Unmapped KOI disposition values found: {unmapped_values}")
        df.loc[unmapped_mask, 'disposition'] = 'ambiguous'
    
    return df


def _map_toi_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map TESS Objects of Interest (TOI) labels to canonical dispositions.
    
    TOI datasets typically use 'tfopwg_disposition' or 'disposition' column with values like:
    - PC (Planet Candidate), CP (Confirmed Planet), FP (False Positive), 
    - APC (Ambiguous Planet Candidate), KP (Known Planet)
    """
    # Common column name variations for TOI disposition
    disposition_cols = ['tfopwg_disposition', 'disposition', 'toi_disposition', 'toi_disp']
    source_col = None
    
    for col in disposition_cols:
        if col in df.columns:
            source_col = col
            break
    
    if source_col is None:
        raise ValueError(f"No TOI disposition column found. Expected one of: {disposition_cols}")
    
    # TOI mapping dictionary
    toi_mapping = {
        'PC': 'candidate',           # Planet Candidate
        'CP': 'confirmed',           # Confirmed Planet
        'FP': 'false_positive',      # False Positive
        'APC': 'ambiguous',          # Ambiguous Planet Candidate
        'KP': 'known',               # Known Planet
        # Full names
        'Planet Candidate': 'candidate',
        'Confirmed Planet': 'confirmed',
        'False Positive': 'false_positive',
        'Ambiguous Planet Candidate': 'ambiguous',
        'Known Planet': 'known',
        # Lowercase versions
        'pc': 'candidate',
        'cp': 'confirmed',
        'fp': 'false_positive',
        'apc': 'ambiguous',
        'kp': 'known',
        # Handle NaN/null values
        np.nan: 'ambiguous',
        None: 'ambiguous'
    }
    
    # Apply mapping
    df['disposition'] = df[source_col].astype(str).map(toi_mapping)
    
    # Handle any unmapped values
    unmapped_mask = df['disposition'].isna()
    if unmapped_mask.any():
        unmapped_values = df.loc[unmapped_mask, source_col].unique()
        print(f"Warning: Unmapped TOI disposition values found: {unmapped_values}")
        df.loc[unmapped_mask, 'disposition'] = 'ambiguous'
    
    return df


def _map_k2_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map K2 mission labels to canonical dispositions.
    
    K2 datasets may use various column names and values similar to KOI but
    sometimes with different conventions or additional categories.
    """
    # Common column name variations for K2 disposition
    disposition_cols = ['k2_disposition', 'disposition', 'epic_disposition', 'k2_disp']
    source_col = None
    
    for col in disposition_cols:
        if col in df.columns:
            source_col = col
            break
    
    if source_col is None:
        raise ValueError(f"No K2 disposition column found. Expected one of: {disposition_cols}")
    
    # K2 mapping dictionary (similar to KOI but may have additional categories)
    k2_mapping = {
        'CONFIRMED': 'confirmed',
        'CANDIDATE': 'candidate', 
        'FALSE POSITIVE': 'false_positive',
        'NOT DISPOSITIONED': 'ambiguous',
        'KNOWN PLANET': 'known',
        # Alternative spellings/formats
        'confirmed': 'confirmed',
        'candidate': 'candidate',
        'false positive': 'false_positive',
        'false_positive': 'false_positive',
        'not dispositioned': 'ambiguous',
        'known planet': 'known',
        'known_planet': 'known',
        'ambiguous': 'ambiguous',
        # Handle NaN/null values
        np.nan: 'ambiguous',
        None: 'ambiguous'
    }
    
    # Apply mapping
    df['disposition'] = df[source_col].astype(str).str.upper().map(
        {k.upper() if k is not None and not pd.isna(k) else k: v 
         for k, v in k2_mapping.items()}
    )
    
    # Handle any unmapped values
    unmapped_mask = df['disposition'].isna()
    if unmapped_mask.any():
        unmapped_values = df.loc[unmapped_mask, source_col].unique()
        print(f"Warning: Unmapped K2 disposition values found: {unmapped_values}")
        df.loc[unmapped_mask, 'disposition'] = 'ambiguous'
    
    return df


def get_disposition_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get a summary count of disposition values in a DataFrame.
    
    Args:
        df: DataFrame with 'disposition' column
        
    Returns:
        dict: Dictionary with disposition values as keys and counts as values
        
    Examples:
        >>> df = pd.DataFrame({'disposition': ['confirmed', 'candidate', 'confirmed']})
        >>> get_disposition_summary(df)
        {'confirmed': 2, 'candidate': 1}
    """
    if 'disposition' not in df.columns:
        raise ValueError("DataFrame must have 'disposition' column")
    
    return df['disposition'].value_counts().to_dict()
