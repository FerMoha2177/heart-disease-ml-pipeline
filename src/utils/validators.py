"""
Pydantic validators for heart disease prediction API
Defines input/output schemas based on Heart Disease UCI dataset
"""

from typing import Dict, List, Optional, Any
import pandas as pd

def validate_patient_data(data_dict):
    """Validate patient data meets expected ranges"""
    df = pd.DataFrame([data_dict])
    check_missing_values(df, required_columns)
    validate_data_types(df, expected_types)
    check_age_range(age)
    validate_cholesterol_levels(chol)
    validate_heart_rate(thalach)
    
def check_missing_values(df, required_columns):
    """Check for missing values in critical columns"""
    missing_values = df[required_columns].isnull().sum()
    if missing_values.any():
        raise ValueError(f"Missing values found in columns: {missing_values}")
    
def validate_data_types(df, expected_types):
    """Ensure columns have correct data types"""
    type_mismatch = (df.dtypes != expected_types).any()
    if type_mismatch:
        raise ValueError(f"Type mismatch found in columns: {type_mismatch}")
    
def check_age_range(age):
    """Validate age is within reasonable bounds (1-120)"""
    if age < 1 or age > 120:
        raise ValueError("Age must be between 1 and 120")
    
def validate_cholesterol_levels(chol):
    """Check cholesterol values are realistic (100-600)"""
    if chol < 100 or chol > 600:
        raise ValueError("Cholesterol must be between 100 and 600")
    
def validate_heart_rate(thalach):
    """Ensure heart rate is within normal ranges (60-220)"""
    if thalach < 60 or thalach > 220:
        raise ValueError("Heart rate must be between 60 and 220")