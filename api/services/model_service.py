"""
Model service for loading and serving ML predictions
Handles the trained heart disease prediction model
"""

import os
import joblib
from typing import Dict, Any
import logging
from datetime import datetime

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

        self.preprocessing_service = None
        
        # Expected feature order (from  Gold layer)
        self.expected_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
    
    async def load_model(self) -> bool:
        """
        Load the trained ML model from disk
        """
        try:
            # Import here to avoid circular imports
            from api.dependencies import get_preprocessing_service_optional

            # Get preprocessing service (without raising exceptions)
            self.preprocessing_service = get_preprocessing_service_optional()
            if not self.preprocessing_service or not self.preprocessing_service.is_loaded:
                logger.error("Preprocessing service not available or not loaded")
                return False
            
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
            
            if not self.preprocessing_service or not self.preprocessing_service.is_loaded:
                raise ValueError("Preprocessing service is not ready")
            
            # Apply complete preprocessing pipeline
            processed_data = self.preprocessing_service.preprocess_for_prediction(patient_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Determine confidence
            max_prob = max(probability)
            confidence = "high" if max_prob > 0.8 else "medium" if max_prob > 0.6 else "low"
            
            # Extract risk factors (basic implementation)
            risk_factors = self._identify_risk_factors(patient_data, prediction)
            
            return {
                "prediction": int(prediction),
                "probability": float(probability[1]),  # Probability of heart disease
                "confidence": confidence,
                "risk_factors": risk_factors,
                "model_version": self.version
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _identify_risk_factors(self, patient_data: Dict[str, Any], prediction: int) -> list:
        """Basic risk factor identification"""
        risk_factors = []
        
        if prediction == 1:  # Heart disease predicted
            # Check for common risk factors
            if patient_data.get('age', 0) > 55:
                risk_factors.append("advanced_age")
            
            if patient_data.get('chol', 0) > 240:
                risk_factors.append("high_cholesterol")
            
            if patient_data.get('trestbps', 0) > 140:
                risk_factors.append("high_blood_pressure")
            
            # Handle both string and numeric formats
            sex = patient_data.get('sex', 0)
            if sex in ['male', 'Male', 'M', 1]:
                risk_factors.append("male_gender")
            
            exang = patient_data.get('exang', 0)
            if exang in ['yes', 'Yes', 'true', 'True', 1]:
                risk_factors.append("exercise_induced_angina")
            
            cp = patient_data.get('cp', 0)
            if cp in ['asymptomatic', 3]:
                risk_factors.append("asymptomatic_chest_pain")
        
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
            "feature_count": len(self.expected_features),
            "preprocessing_status": self.preprocessing_service.get_preprocessing_info() if self.preprocessing_service else None
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