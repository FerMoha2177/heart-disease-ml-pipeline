"""
Model service for loading and serving ML predictions
Handles the trained heart disease prediction model
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path

from config.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

MODEL_DIR = "../../models"
class MLModelService:
    """
    Service for loading and serving ML model predictions
    """
    
    def __init__(self):
        self.model = None
        self.model_path = os.getenv("MODEL_PATH", "models/heart_disease_model.pkl")
        self.version = os.getenv("MODEL_VERSION", "1.0.0")
        self.model_type = "classification"
        self.loaded_at = None
        
        # Expected feature order (from your Gold layer)
        self.expected_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    async def load_model(self) -> bool:
        """
        Load the trained ML model from disk
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                
                # Try alternative paths
                alternative_paths = [
                    "models/heart_disease_classifier.joblib",
                    "../models/heart_disease_model.pkl",
                    "../models/heart_disease_classifier.joblib"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        self.model_path = alt_path
                        logger.info(f"Found model at alternative path: {alt_path}")
                        break
                else:
                    logger.error("No model file found in any expected location")
                    return False
            
            # Load the model
            self.model = joblib.load(self.model_path)
            self.loaded_at = datetime.now()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Expected features: {len(self.expected_features)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    async def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using the loaded ML model

        **Input**: Patient data including age, sex, chest pain type, etc.
        **Output**: Prediction (0/1), probability, confidence level, and risk factors
        
        **Example Request:**
        ```json
        {
            "age": 55,
            "sex": 1,
            "cp": 3,
            "trestbps": 130,
            "chol": 250,
            "fbs": 0,
            "restecg": 0,
            "thalach": 140,
            "exang": 1,
            "oldpeak": 1.5,
            "slope": 1,
            "ca": 1,
            "thal": 0
        }
        ```
        """
        try:
            if not self.model:
                raise ValueError("Model is not loaded")
            
            logger.debug(f"Making prediction for patient data: {patient_data}")
            
            # Convert patient data to DataFrame with correct feature order
            df = pd.DataFrame([patient_data])
            
            # Ensure all expected features are present and in correct order
            for feature in self.expected_features:
                if feature not in df.columns:
                    logger.warning(f"Missing feature '{feature}', setting to 0")
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.expected_features]
            
            logger.debug(f"Input shape for model: {df.shape}")
            logger.debug(f"Feature values: {df.iloc[0].to_dict()}")
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            prediction_proba = self.model.predict_proba(df)[0]
            
            # Get probability of heart disease (class 1)
            heart_disease_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            
            # Determine confidence level
            max_prob = max(prediction_proba)
            if max_prob > 0.8:
                confidence = "high"
            elif max_prob > 0.6:
                confidence = "medium"  
            else:
                confidence = "low"
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(patient_data, prediction)
            
            result = {
                "prediction": int(prediction),
                "probability": float(heart_disease_probability),
                "confidence": confidence,
                "risk_factors": risk_factors
            }
            
            logger.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _identify_risk_factors(self, patient_data: Dict[str, Any], prediction: int) -> List[str]:
        """
        Identify risk factors based on patient data and prediction
        """
        risk_factors = []
        
        try:
            # Only identify risk factors if heart disease is predicted
            if prediction == 1:
                # Age risk
                if patient_data.get('age', 0) > 55:
                    risk_factors.append("advanced_age")
                
                # Cholesterol risk
                if patient_data.get('chol', 0) > 240:
                    risk_factors.append("high_cholesterol")
                
                # Blood pressure risk
                if patient_data.get('trestbps', 0) > 140:
                    risk_factors.append("high_blood_pressure")
                
                # Gender risk (males have higher risk)
                if patient_data.get('sex', 0) == 1:
                    risk_factors.append("male_gender")
                
                # Exercise induced angina
                if patient_data.get('exang', 0) == 1:
                    risk_factors.append("exercise_induced_angina")
                
                # Chest pain type (asymptomatic is highest risk)
                if patient_data.get('cp', 0) == 3:
                    risk_factors.append("asymptomatic_chest_pain")
                
                # Fasting blood sugar
                if patient_data.get('fbs', 0) == 1:
                    risk_factors.append("high_fasting_blood_sugar")
                
                # Low maximum heart rate achieved
                if patient_data.get('thalach', 200) < 120:
                    risk_factors.append("low_max_heart_rate")
                
        except Exception as e:
            logger.warning(f"Error identifying risk factors: {e}")
            risk_factors = ["analysis_error"]
        
        return risk_factors
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the loaded model
        """
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "model_type": type(self.model).__name__ if self.model else None,
            "version": self.version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "expected_features": self.expected_features,
            "feature_count": len(self.expected_features)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check for the model service
        """
        return {
            "service": "MLModelService",
            "status": "healthy" if self.model else "unhealthy",
            "model_loaded": self.model is not None,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat()
        }