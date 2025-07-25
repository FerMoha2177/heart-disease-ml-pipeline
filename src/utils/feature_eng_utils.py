from config.logging import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np


# Feature Engineering

def min_max_scale(df):
    """Min-max scale all features"""
    scaler = MinMaxScaler()
    df_scaled = df.copy()

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = numerical_cols.drop('target') # dont need to scale target

    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    return df_scaled

def get_binary_features(df):
    """Get binary features even if its not a boolean"""
    binary_cols = []  # Start with empty list
    
    # Get boolean columns
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()  # Convert to list
    binary_cols.extend(bool_cols)  # need to extend rather than append or sizing issues will occur
    logger.info(f"Boolean Columns: {bool_cols}")
    
    # Get categorical columns that have exactly 2 unique values
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() == 2:
            binary_cols.append(col)  # Now append individual column name
    
    logger.info(f"Final Binary Columns: {binary_cols}")
    return binary_cols

def get_categorical_features(df, exclude_binary=True):
    """Get categorical features (excluding binary if specified)"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if exclude_binary:
        # Remove binary categorical features
        binary_categorical = []
        for col in categorical_cols:
            if df[col].nunique() == 2:
                binary_categorical.append(col)
        
        # Keep only multi-class categorical features
        categorical_cols = [col for col in categorical_cols if col not in binary_categorical]
    
    logger.info(f"Multi-class Categorical Columns: {categorical_cols}")
    return categorical_cols

def simple_label_encoding(df, binary_features):
    """Simple label encoding for binary features"""
    df = df.copy()
    
    for col in binary_features:
        le = LabelEncoder()
        # Handle missing values if any
        if df[col].isnull().any():
            logger.warning(f"Column {col} has missing values, filling with mode")
            df[col] = df[col].fillna(df[col].mode()[0])
        
        df[col] = le.fit_transform(df[col])
        logger.info(f"Encoded {col}: {df[col].unique()}")
    
    return df

def one_hot_encoding(df, categorical_features):
    """One-hot encoding for multi-class categorical features"""
    df = df.copy()
    
    for col in categorical_features:
        # Create dummy variables with explicit dtype=int
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', dtype=int)
        
        # Drop original column and add dummy columns
        df = df.drop(col, axis=1)
        df = pd.concat([df, dummies], axis=1)
        
        logger.info(f"One-hot encoded {col}: created {len(dummies.columns)} new columns")
    
    return df

def k_highest_features(df,target_col='target', k=10):
    """Using K best method to select the k highest features
    Mutual Information (MI) measures how much information one variable gives you about
    another variable. In this case:

    Higher score = Feature provides MORE information about heart disease
    Lower score = Feature provides LESS information about heart disease
    Score of 0 = Feature provides NO information (completely independent)
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Target column name    
        k (int): Number of features to select
    
    Returns:
        pd.DataFrame: DataFrame with selected features
    """
    df_copy = df.copy()
    X = df_copy.drop(target_col, axis=1)
    y = df_copy[target_col]

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit_transform(X, y)

    feature_scores = selector.scores_
    logger.info(f"Type: {type(feature_scores)}")
    logger.info(f"Shape: {feature_scores.shape}")
    logger.info(f"Sample scores: {feature_scores[:5]}")

    # Make it readable
    scores_df = pd.DataFrame({
    'feature': X.columns,
    'score': feature_scores
    }).sort_values('score', ascending=False)

    logger.info("\nFEATURE SELECTION MASK (which features were selected):")
    feature_mask = selector.get_support()
    logger.info(f"Type: {type(feature_mask)}")
    logger.info(f"Shape: {feature_mask.shape}")

    # Get selected feature names
    logger.info("\nSELECTED FEATURE NAMES:")
    selected_features = X.columns[feature_mask].tolist()
    logger.info(f"Selected features: {selected_features}")

    rejected_features = X.columns[~feature_mask].tolist() # short for not feature mask
    logger.info(f"Rejected features: {rejected_features}")

    return selected_features, rejected_features, scores_df
    
    

def random_forest_feature_selection(df, target_col='target', k=10):
    """Using Random Forest feature selection to select the k highest features"""
    df_copy = df.copy()
    X = df_copy.drop(target_col, axis=1)
    y = df_copy[target_col]
    
    # Initialize and train the Random Forest classifier
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X, y) # quick training for important scores
    
    # Get feature importances
    importances = rf_selector.feature_importances_
    
    # Create a DataFrame to store feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top k features
    selected_features = feature_importances.head(k)['feature'].tolist()
    rejected_features = feature_importances.tail(len(feature_importances) - k)['feature'].tolist()
    
    logger.info("\nSELECTED FEATURE NAMES:")
    logger.info(f"Selected features: {selected_features}")
    logger.info(f"Rejected features: {rejected_features}")
    return selected_features, rejected_features, feature_importances
    