# api/services/preprocessing_service.py
"""
Preprocessing service that loads and applies the complete preprocessing pipeline
Uses artifacts saved during model training to ensure consistency
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class PreprocessingService:
    """
    Loads saved preprocessing artifacts and applies the complete transformation pipeline
    Raw Input → Bronze → Silver → Gold → Model-Ready Data
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        
        # Loaded artifacts
        self.scaler = None
        self.categorical_encoders = {}
        self.feature_info = None
        self.pipeline_metadata = None
        
        # Processing state
        self.is_loaded = False
        self.load_error = None
        
        # Load all artifacts at initialization
        try:
            self._load_all_artifacts()
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load preprocessing artifacts: {e}")
            self.is_loaded = False
    
    def _load_all_artifacts(self):
        """Load all saved preprocessing artifacts"""
        try:
            logger.info("Loading preprocessing artifacts...")
            
            # 1. Load fitted scaler
            scaler_path = self.model_dir / "preprocessing_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded fitted MinMaxScaler")
            else:
                logger.warning("No saved scaler found")
            
            # 2. Load categorical encoders/mappings
            encoders_path = self.model_dir / "categorical_encoders.pkl"
            if encoders_path.exists():
                self.categorical_encoders = joblib.load(encoders_path)
                logger.info("Loaded categorical encoders")
            else:
                logger.warning("No categorical encoders found")
                self._create_fallback_encoders()
            
            # 3. Load feature information
            feature_path = self.model_dir / "feature_columns.json"
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    self.feature_info = json.load(f)
                logger.info(f"Loaded feature info: {len(self.feature_info['feature_columns'])} features")
            else:
                logger.error("No feature information found")
                raise FileNotFoundError("feature_columns.json is required")
            
            # 4. Load pipeline metadata
            metadata_path = self.model_dir / "preprocessing_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.pipeline_metadata = json.load(f)
                logger.info(f"Loaded pipeline metadata v{self.pipeline_metadata.get('pipeline_version', 'unknown')}")
            
            self.is_loaded = True
            logger.info("All preprocessing artifacts loaded successfully!")
            
        except Exception as e:
            self.load_error = str(e)
            logger.error(f"Failed to load preprocessing artifacts: {e}")
            self.is_loaded = False
    
    def _create_fallback_encoders(self):
        """Create fallback encoders if saved ones not available"""
        logger.warning("Creating fallback categorical encoders...")
        self.categorical_encoders = {
            'sex': {'male': 1, 'female': 0, 'Male': 1, 'Female': 0, 'M': 1, 'F': 0},
            'cp': {'typical_angina': 0, 'atypical_angina': 1, 'non_anginal_pain': 2, 'asymptomatic': 3},
            'fbs': {'false': 0, 'true': 1, 'no': 0, 'yes': 1},
            'restecg': {'normal': 0, 'ST-T abnormality': 1, 'LV hypertrophy': 2},
            'exang': {'no': 0, 'yes': 1, 'false': 0, 'true': 1},
            'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
            'thal': {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2, 'unknown': 3}
        }
    
    def preprocess_for_prediction(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        MAIN METHOD: Apply complete preprocessing pipeline to raw input
        
        Args:
            patient_data: Raw patient data from API request
            
        Returns:
            Preprocessed numpy array ready for model prediction
            
        Raises:
            ValueError: If preprocessing fails or artifacts not loaded
        """
        if not self.is_loaded:
            raise ValueError(f"Preprocessing service not properly loaded: {self.load_error}")
        
        try:
            logger.info("Starting preprocessing pipeline...")
            
            # Step 1: Convert to DataFrame
            df = pd.DataFrame([patient_data])
            logger.debug(f"Input data shape: {df.shape}")
            
            # Step 2: Apply categorical encoding (Bronze → Silver)
            df_encoded = self._apply_categorical_encoding(df)
            
            # Step 3: Validate and clip values
            df_validated = self._validate_and_clip(df_encoded)
            
            # Step 4: Ensure feature order matches expected input
            df_ordered = self._ensure_feature_order(df_validated)
            
            # Step 5: Apply scaling (Silver → Gold)
            df_scaled = self._apply_scaling(df_ordered)
            
            # Step 6: Select final features for model
            processed_array = self._select_final_features(df_scaled)
            
            logger.info(f"Preprocessing complete. Output shape: {processed_array.shape}")
            return processed_array
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ValueError(f"Preprocessing error: {str(e)}")
    
    def _apply_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding to string features"""
        df_encoded = df.copy()
        
        # Convert categorical strings to numeric values
        for feature in df_encoded.columns:
            if df_encoded[feature].dtype == 'object':
                if feature in self.categorical_encoders:
                    encoder_map = self.categorical_encoders[feature]
                    try:
                        value = df_encoded[feature].iloc[0]
                        if value in encoder_map:
                            df_encoded[feature] = encoder_map[value]
                        else:
                            logger.warning(f"Unknown categorical value for {feature}: {value}")
                            df_encoded[feature] = 0
                    except (ValueError, TypeError):
                        logger.warning(f"Error encoding {feature}, using 0")
                        df_encoded[feature] = 0
        
        return df_encoded
    
    def _validate_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clip values to reasonable ranges"""
        df_validated = df.copy()
        
        # Get validation ranges from metadata
        if self.pipeline_metadata and 'validation_ranges' in self.pipeline_metadata:
            ranges = self.pipeline_metadata['validation_ranges']
            
            for feature, range_info in ranges.items():
                if feature in df_validated.columns:
                    min_val, max_val = range_info['min'], range_info['max']
                    original_val = df_validated[feature].iloc[0]
                    clipped_val = np.clip(original_val, min_val, max_val)
                    
                    if original_val != clipped_val:
                        logger.warning(f"Clipped {feature}: {original_val} → {clipped_val}")
                    
                    df_validated[feature] = clipped_val
        
        return df_validated
    
    def _ensure_feature_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure features are in the correct order and all expected features are present"""
        # Get expected raw features from pipeline metadata (not feature_info)
        if self.pipeline_metadata and 'api_input_format' in self.pipeline_metadata:
            expected_features = self.pipeline_metadata['api_input_format']['expected_raw_features']
        else:
            # Fallback to the standard feature order
            expected_features = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            logger.warning("Using fallback expected features")
        
        # Reindex to match expected feature order, filling missing with 0
        df_ordered = df.reindex(columns=expected_features, fill_value=0)
        
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features filled with 0: {missing_features}")
        
        return df_ordered
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MinMax scaling using fitted scaler"""
        if not self.scaler:
            logger.warning("No scaler available, skipping scaling")
            return df
            
        try:
            df_scaled = df.copy()
            scaled_data = self.scaler.transform(df_scaled)
            df_scaled = pd.DataFrame(scaled_data, columns=df_scaled.columns, index=df_scaled.index)
            logger.debug("Applied MinMax scaling")
            return df_scaled
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return df
    
    def _select_final_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select only the features used by the final model"""
        if self.feature_info and 'selected_features' in self.feature_info:
            selected_features = self.feature_info['selected_features']
            
            # Filter DataFrame to only include selected features
            df_final = df.reindex(columns=selected_features, fill_value=0)
            logger.debug(f"Selected {len(selected_features)} features for model")
            return df_final.values
        else:
            logger.warning("No feature selection info, using all features")
            return df.values
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about loaded preprocessing artifacts"""
        return {
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "artifacts_status": {
                "scaler": self.scaler is not None,
                "categorical_encoders": len(self.categorical_encoders) > 0,
                "feature_info": self.feature_info is not None,
                "pipeline_metadata": self.pipeline_metadata is not None
            },
            "pipeline_info": {
                "version": self.pipeline_metadata.get("pipeline_version", "unknown") if self.pipeline_metadata else "unknown",
                "expected_features": self._get_expected_raw_features(),
                "final_features": self.feature_info.get("selected_features", []) if self.feature_info else [],
                "transformation_steps": self.pipeline_metadata.get("complete_transformation_steps", []) if self.pipeline_metadata else []
            }
        }
    
    def _get_expected_raw_features(self) -> List[str]:
        """Get expected raw features from the correct location"""
        if self.pipeline_metadata and 'api_input_format' in self.pipeline_metadata:
            return self.pipeline_metadata['api_input_format']['expected_raw_features']
        else:
            return [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
    
    def validate_input(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data before preprocessing"""
        validation_result = {"valid": True, "errors": [], "warnings": []}
        
        expected_features = self._get_expected_raw_features()
        
        # Check for missing required features
        missing_features = [f for f in expected_features if f not in patient_data]
        if missing_features:
            validation_result["warnings"].append(f"Missing features will be filled with 0: {missing_features}")
        
        # Check for unexpected features
        unexpected_features = [f for f in patient_data.keys() if f not in expected_features]
        if unexpected_features:
            validation_result["warnings"].append(f"Unexpected features will be ignored: {unexpected_features}")
        
        return validation_result