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
import database_service

logger = logging.getLogger(__name__)


class MLModelService:
    """
    Service for loading and serving ML model predictions
    """
    
    def __init__(self):
        self.model = None
        self.model_path = os.getenv("MODEL_PATH", "models/heart_disease_model.pkl")
        self.version = os.getenv("MODEL_VERSION", "1.0.0")
        self.model_type = "classification"
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.loaded_at = None
    
    async def load_model(self) -> bool:
        """
        Load the trained ML model from disk
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}")
                # Create a dummy model for development
                self._create_dummy_model()
                return True
            
            self.model = joblib.load(self.model_path)
            self.loaded_at = datetime.utcnow()
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    
    async def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using the loaded ML model
        """
        try:
            if not self.model:
                raise ValueError("Model is not loaded")
            
            # Convert patient data to a DataFrame
            df = pd.DataFrame([patient_data]) #TODO: need to read from Gold Layer
            
            # Ensure all required features are present
            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Make prediction
            prediction = self.model.predict(df[self.feature_names])
            
            # Return prediction result
            return {
                "prediction": prediction[0],
                "probability": self.model.predict_proba(df[self.feature_names])[0][1],
                "confidence": "high" if prediction[0] == 1 else "low"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise