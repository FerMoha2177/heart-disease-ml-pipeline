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

logger = logging.getLogger(__name__)


class ModelService:
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