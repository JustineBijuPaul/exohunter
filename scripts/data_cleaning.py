"""
Data Quality Analysis and Cleaning for Exoplanet Datasets

This script performs comprehensive data analysis and cleaning on exoplanet datasets
to improve model performance and accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality(df, dataset_name):
    """Analyze data quality issues in a dataset."""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    print(f"\nMISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing %', ascending=False)
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df) > 0:
        print(missing_df.head(15))
    else:
        print("No missing values found!")
    
    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"\nDUPLICATE ROWS: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Data types
    print(f"\nDATA TYPES:")
    print(df.dtypes.value_counts())
    
    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNUMERICAL COLUMNS STATISTICS:")
        print(df[numeric_cols].describe())
        
        # Check for infinite values
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"\nINFINITE VALUES:")
            for col, count in inf_counts.items():
                print(f"  {col}: {count}")
        
        # Check for outliers using IQR method
        print(f"\nOUTLIER ANALYSIS (using IQR method):")
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        if outlier_counts:
            for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {col}: {count} outliers ({count/len(df)*100:.1f}%)")
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nCATEGORICAL COLUMNS:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"    Values: {df[col].value_counts().to_dict()}")
    
    return missing_df, duplicates, outlier_counts if 'outlier_counts' in locals() else {}

def clean_exoplanets_combined(df):
    """Clean the combined exoplanets dataset."""
    print(f"\nCleaning exoplanets_combined.csv...")
    df_clean = df.copy()
    
    # Remove rows where disposition is missing (key target variable)
    if 'disposition' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['disposition'])
        print(f"Removed rows with missing disposition")
    
    # Handle missing values in key features
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # For orbital_period, planet_radius, stellar parameters - these are critical
    critical_cols = ['orbital_period', 'planet_radius', 'stellar_teff', 'stellar_radius']
    available_critical = [col for col in critical_cols if col in df_clean.columns]
    
    if available_critical:
        # Remove rows where ALL critical features are missing
        df_clean = df_clean.dropna(subset=available_critical, how='all')
        print(f"Removed rows missing all critical features")
    
    # Fill missing values with median for numerical columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dups = initial_len - len(df_clean)
    if removed_dups > 0:
        print(f"Removed {removed_dups} duplicate rows")
    
    # Handle outliers for key numerical features
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.01)  # Use 1st and 99th percentile instead of IQR
            Q3 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q3)
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def clean_kepler_koi(df):
    """Clean the Kepler KOI dataset."""
    print(f"\nCleaning kepler_koi.csv...")
    df_clean = df.copy()
    
    # Remove rows where koi_disposition is missing
    if 'koi_disposition' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['koi_disposition'])
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dups = initial_len - len(df_clean)
    if removed_dups > 0:
        print(f"Removed {removed_dups} duplicate rows")
    
    # Handle outliers
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.01)
            Q3 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q3)
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def clean_tess_toi(df):
    """Clean the TESS TOI dataset."""
    print(f"\nCleaning tess_toi.csv...")
    df_clean = df.copy()
    
    # Remove rows where tfopwg_disp is missing
    if 'tfopwg_disp' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['tfopwg_disp'])
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dups = initial_len - len(df_clean)
    if removed_dups > 0:
        print(f"Removed {removed_dups} duplicate rows")
    
    # Handle outliers
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.01)
            Q3 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q3)
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def clean_k2_candidates(df):
    """Clean the K2 candidates dataset."""
    print(f"\nCleaning k2_candidates.csv...")
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dups = initial_len - len(df_clean)
    if removed_dups > 0:
        print(f"Removed {removed_dups} duplicate rows")
    
    # Handle outliers
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.01)
            Q3 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q3)
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def feature_engineering(df, dataset_name):
    """Perform feature engineering on the dataset."""
    print(f"\nPerforming feature engineering on {dataset_name}...")
    df_engineered = df.copy()
    
    # Create new features based on existing ones
    if 'orbital_period' in df_engineered.columns or 'koi_period' in df_engineered.columns:
        period_col = 'orbital_period' if 'orbital_period' in df_engineered.columns else 'koi_period'
        if period_col in df_engineered.columns:
            # Log of orbital period (often has exponential distribution)
            df_engineered[f'{period_col}_log'] = np.log1p(df_engineered[period_col])
            
    if 'planet_radius' in df_engineered.columns or 'koi_prad' in df_engineered.columns:
        radius_col = 'planet_radius' if 'planet_radius' in df_engineered.columns else 'koi_prad'
        if radius_col in df_engineered.columns:
            # Earth radius categories
            df_engineered[f'{radius_col}_category'] = pd.cut(
                df_engineered[radius_col], 
                bins=[0, 1.25, 2.0, 4.0, float('inf')], 
                labels=['Sub-Earth', 'Earth-size', 'Sub-Neptune', 'Jupiter-size']
            )
    
    # Stellar temperature categories
    if 'stellar_teff' in df_engineered.columns or 'koi_steff' in df_engineered.columns:
        teff_col = 'stellar_teff' if 'stellar_teff' in df_engineered.columns else 'koi_steff'
        if teff_col in df_engineered.columns:
            df_engineered[f'{teff_col}_category'] = pd.cut(
                df_engineered[teff_col],
                bins=[0, 3700, 5200, 6000, 7500, float('inf')],
                labels=['M-dwarf', 'K-dwarf', 'G-dwarf', 'F-dwarf', 'Hot-star']
            )
    
    # Calculate density if mass and radius are available
    mass_cols = ['koi_smass', 'st_mass']
    radius_cols = ['stellar_radius', 'koi_srad', 'st_rad']
    
    available_mass = [col for col in mass_cols if col in df_engineered.columns]
    available_radius = [col for col in radius_cols if col in df_engineered.columns]
    
    if available_mass and available_radius:
        mass_col = available_mass[0]
        radius_col = available_radius[0]
        # Stellar density (mass/radius^3)
        df_engineered['stellar_density'] = df_engineered[mass_col] / (df_engineered[radius_col] ** 3)
    
    print(f"Feature engineering completed. New shape: {df_engineered.shape}")
    return df_engineered

def main():
    """Main function to analyze and clean all datasets."""
    data_dir = Path("data")
    
    # Load datasets
    datasets = {
        'exoplanets_combined': 'exoplanets_combined.csv',
        'kepler_koi': 'kepler_koi.csv',
        'tess_toi': 'tess_toi.csv',
        'k2_candidates': 'k2_candidates.csv'
    }
    
    cleaned_datasets = {}
    
    for name, filename in datasets.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\nLoading {filename}...")
            df = pd.read_csv(filepath)
            
            # Analyze data quality
            missing_df, duplicates, outliers = analyze_data_quality(df, name)
            
            # Clean the dataset
            if name == 'exoplanets_combined':
                df_clean = clean_exoplanets_combined(df)
            elif name == 'kepler_koi':
                df_clean = clean_kepler_koi(df)
            elif name == 'tess_toi':
                df_clean = clean_tess_toi(df)
            elif name == 'k2_candidates':
                df_clean = clean_k2_candidates(df)
            
            # Feature engineering
            df_final = feature_engineering(df_clean, name)
            
            # Save cleaned dataset
            output_path = data_dir / f"{name}_cleaned.csv"
            df_final.to_csv(output_path, index=False)
            print(f"Saved cleaned dataset to {output_path}")
            
            cleaned_datasets[name] = df_final
        else:
            print(f"Warning: {filepath} not found!")
    
    # Create a unified cleaned dataset for training
    if 'exoplanets_combined' in cleaned_datasets:
        main_dataset = cleaned_datasets['exoplanets_combined']
        
        # Save the main cleaned dataset for model training
        main_output_path = data_dir / "cleaned_training_data.csv"
        main_dataset.to_csv(main_output_path, index=False)
        print(f"\nSaved main training dataset to {main_output_path}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL CLEANING SUMMARY")
        print(f"{'='*60}")
        print(f"Main training dataset shape: {main_dataset.shape}")
        print(f"Features available: {list(main_dataset.columns)}")
        
        # Class distribution
        if 'disposition' in main_dataset.columns:
            print(f"\nClass distribution:")
            print(main_dataset['disposition'].value_counts())

if __name__ == "__main__":
    main()