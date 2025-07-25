"""
Pydantic validators for heart disease prediction API
Defines input/output schemas based on Heart Disease UCI dataset
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import json
import joblib
import logging
from config.logging import setup_logging
logger = logging.getLogger(__name__)

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

def test_pipeline_artifacts():
    """Test that saved artifacts can be loaded and used"""
    try:
        # Load artifacts
        test_scaler = joblib.load('../models/preprocessing_scaler.pkl')
        test_encoders = joblib.load('../models/categorical_encoders.pkl')
        
        with open('../models/feature_columns.json', 'r') as f:
            test_feature_info = json.load(f)
        
        with open('../models/preprocessing_metadata.json', 'r') as f:
            test_metadata = json.load(f)
        
        logger.info("All artifacts loaded successfully")
        logger.info(f"   Scaler: {type(test_scaler).__name__}")
        logger.info(f"   Feature count: {len(test_feature_info['feature_columns'])}")
        logger.info(f"   Pipeline version: {test_metadata['pipeline_version']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False

def generate_sample_requests():
    """Generate sample requests for API testing"""
    try:
        sample_requests = {
            'numeric_format_example': {
                "age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250,
                "fbs": 0, "restecg": 0, "thalach": 140, "exang": 1,
                "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 0
            },
            'string_format_example': {
                "age": 55, "sex": "male", "cp": "asymptomatic", "trestbps": 130, "chol": 250,
                "fbs": "false", "restecg": "normal", "thalach": 140, "exang": "yes",
                "oldpeak": 1.5, "slope": "flat", "ca": 1, "thal": "normal"
            },
            'test_cases': [
                {
                    'name': 'high_risk_patient',
                    'data': {
                        "age": 67, "sex": "male", "cp": "asymptomatic", "trestbps": 160, "chol": 286,
                        "fbs": "true", "restecg": "LV hypertrophy", "thalach": 108, "exang": "yes",
                        "oldpeak": 1.5, "slope": "flat", "ca": 3, "thal": "normal"
                    }
                },
                {
                    'name': 'low_risk_patient', 
                    'data': {
                        "age": 29, "sex": "female", "cp": "non_anginal_pain", "trestbps": 130, "chol": 204,
                        "fbs": "false", "restecg": "normal", "thalach": 202, "exang": "no",
                        "oldpeak": 0.0, "slope": "upsloping", "ca": 0, "thal": "normal"
                    }
                }
            ]
        }

        with open('../models/sample_requests.json', 'w') as f:
            json.dump(sample_requests, f, indent=2)
        logger.info("Saved sample API requests")
        
    except Exception as e:
        logger.error(f"Failed to generate sample requests: {str(e)}")
        raise