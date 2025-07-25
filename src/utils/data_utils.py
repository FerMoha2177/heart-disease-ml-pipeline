from config.logging import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

import pandas as pd
import scipy.stats as stats
import numpy as np


# Missing Data Pattern Analysis

def drop_id(df):
    """Drop the id column"""
    df = df.copy()
    for col in df.columns:
        if col == 'id' or col == '_id':
            df = df.drop(columns=[col])
    return df

def get_missing_data_summary(df):
    """
    Getting where the missing values are coming from seeing if it leans towards a specific column

    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Missing data summary
    """
    missing_df = df.copy()
    missing_summary = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / df.shape[0]
        
        if missing_count > 0:
            missing_summary.append({
            'Column': col,
            'Missing Count': f"{missing_count:,}",
            'Missing %': f"{missing_pct:.1%}",
            'Severity': 'High' if missing_pct > 0.3 else 'Medium' if missing_pct > 0.1 else 'Low'
        })
    
    missing_df = pd.DataFrame(missing_summary)
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    return missing_df

def test_missing_bias(df, col, target='num'):
    """
    Using the Chi-Square test is a statistical test that can be used to determine 
    if there is a significant association between two categorical variables.

    p-value < 0.05: INTERPRETATION: "Missing pattern is NOT RANDOM - it's associated with outcome"
    IMPLICATION: MNAR suspected
    REASONING: If data were MAR, missingness would be independent of outcome

    p-value >= 0.05: INTERPRETATION: "Missing pattern is RANDOM"
    IMPLICATION: MAR or MCAR
    REASONING: Missingness is independent of outcome

    Usage:
    test_missing_bias(df, 'age', 'target')

    or 

    for col in df.columns:
        test_missing_bias(df, col, target)
    
    Args:
        df (pd.DataFrame): Input DataFrame
        col (str): Column to test missingness for
        target (str): Target column for comparison
    
    Returns:
        bool: True if MNAR suspected, False otherwise
    """
    #logger.info(f"Missing pattern correlation with target column {col}")
    df_copy = df.copy()
    df_copy[f'{col}_missing'] = df_copy[col].isnull()
    
    # Chi-square test: missing pattern vs target
    contingency = pd.crosstab(df_copy[f'{col}_missing'], df_copy[target])
    chi2_statistic, p_value, _, _ = stats.chi2_contingency(contingency)
    
    #logger.info(f"Chi-square test for missing pattern correlation with target column {col}:")
    #logger.info(f"  Chi-square statistic: {chi2_statistic:.4f}, p-value: {p_value:.4f}")
    #logger.info(f"  {'MNAR suspected' if p_value < 0.05 else 'Possibly MAR'}")
    
    return p_value < 0.05

def analyze_missing_patterns(df, high_missing_cols=['ca', 'thal', 'slope']):
    """
    Check if missing values occur together (suggests MNAR)
    
    Usage:
    analyze_missing_patterns(df, high_missing_cols=['ca', 'thal', 'slope'])
    
    Args:
        df (pd.DataFrame): Input DataFrame
        high_missing_cols (list): Columns with high missing values
    
    Returns:
        pd.DataFrame: Correlation matrix of missing patterns
    """
    df_copy = df.copy()
    
    # Create missing indicators
    for col in high_missing_cols:
        df_copy[f'{col}_missing'] = df_copy[col].isnull()
    
    # Check correlation between missing patterns
    missing_corr = df_copy[[f'{col}_missing' for col in high_missing_cols]].corr()
    
    #logger.info(f"Missing pattern correlations:")
    #logger.info(f"\n{missing_corr}")
    
    # High correlation suggests systematic missingness (MNAR)
    return missing_corr


def missing_by_dataset(df, col):
    """
    Check if different studies have different missing patterns
    
    Usage:
    missing_by_dataset(df, 'age')
    
    Args:
        df (pd.DataFrame): Input DataFrame
        col (str): Column to check missingness for
    
    Returns:
        bool: True if different studies have different missing patterns, False otherwise
    """
    df_copy = df.copy()
    
    missing_by_study = df_copy.groupby('dataset')[col].apply(lambda x: x.isnull().mean())
    
    #logger.info(f"\n{col} missing rates by dataset:")
    # for dataset, missing_rate in missing_by_study.items():
    #     logger.info(f"  {dataset}: {missing_rate:.1%} missing")
    
    max_diff = missing_by_study.max() - missing_by_study.min()
    return max_diff > 0.2  # 20% threshold


def missing_by_outcome(df, col, target='num'):
    """
    Check if missingness varies by outcome
    
    Usage:
    missing_by_outcome(df, 'age', 'target')
    
    Args:
        df (pd.DataFrame): Input DataFrame
        col (str): Column to check missingness for
        target (str): Target column for comparison
    
    Returns:
        bool: True if missingness varies by outcome, False otherwise
    """
    df_copy = df.copy()
    missing_by_target = df_copy.groupby(target)[col].apply(lambda x: x.isnull().mean())
    
    # logger.info(f"\n{col} missing rates by target value:")
    # for target_val, missing_rate in missing_by_target.items():
    #     logger.info(f"  Target {target_val}: {missing_rate:.1%} missing")
    
    # Large differences suggest MNAR
    max_diff = missing_by_target.max() - missing_by_target.min()
    #logger.info(f"  Max difference: {max_diff:.1%}")
    #logger.info(f"  {'MNAR likely' if max_diff > 0.1 else 'Possibly MAR'}")
    
    return max_diff > 0.1


def find_medically_impossible_values(df):
    """
    Find and return rows with medically impossible values
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary with column names and impossible value counts
    """
    
    # Define valid ranges
    valid_ranges = {
        'age': (1, 120),
        'chol': (100, 600), 
        'trestbps': (50, 300),
        'thalach': (60, 220)
    }
    
    impossible_values = {}
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            # Find values outside valid range
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            # Exclude NaN values from the check
            invalid_mask = invalid_mask & df[col].notna()
            
            impossible_count = invalid_mask.sum()
            if impossible_count > 0:
                impossible_values[col] = {
                    'count': impossible_count,
                    'invalid_rows': df[invalid_mask].index.tolist()
                }
                
                logger.warning(f"{col}: {impossible_count} impossible values found")
                logger.warning(f"Range should be {min_val}-{max_val}")
                logger.warning(f"Found values: {df.loc[invalid_mask, col].tolist()}")
    
    return impossible_values

def handle_impossible_zeros(df):
    """Replace medically impossible zeros with NaN
    
    - Cholesterol can't be 0 Patient would be dead
    - Blood pressure can't be 0  Patient would be dead
    - Heart rate can't be 0 Patient would be dead
    - Age can't be 0 Patient wouldnt be born

    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with medically impossible zeros replaced with NaN
    """
    df_clean = df.copy()
    
    # Only replace zeros that are medically impossible
    impossible_zero_cols = ['chol', 'trestbps', 'thalch', 'age']
    
    for col in impossible_zero_cols:
        if col in df_clean.columns:
            # Count zeros before replacement
            zero_count = (df_clean[col] == 0.0).sum()
            
            # Replace zeros with NaN
            df_clean.loc[df_clean[col] == 0.0, col] = np.nan
            
            logger.info(f"{col}: Replaced {zero_count} impossible zeros with NaN")
    
    return df_clean

def impute_missing_values(df):
    """Impute all missing values including the new NaNs from zeros"""
    df_imputed = df.copy()
    
    # Numerical columns using the median to impute because it is robust to outliers
    numerical_impute_cols = ['chol', 'trestbps', 'thalch', 'oldpeak']
    
    for col in numerical_impute_cols:
        if col in df_imputed.columns:
            median_value = df_imputed[col].median()
            missing_count = df_imputed[col].isnull().sum()
            
            df_imputed[col].fillna(median_value, inplace=True)
            logger.info(f"{col}: Imputed {missing_count} missing values with median {median_value:.1f}")
    
    # Categorical columns using the mode to impute because it is the most common value
    categorical_impute_cols = ['fbs', 'exang', 'restecg']
    
    for col in categorical_impute_cols:
        if col in df_imputed.columns:
            mode_value = df_imputed[col].mode()[0]
            missing_count = df_imputed[col].isnull().sum()
            
            df_imputed[col].fillna(mode_value, inplace=True)
            logger.info(f"{col}: Imputed {missing_count} missing values with mode '{mode_value}'")
    
    return df_imputed
    