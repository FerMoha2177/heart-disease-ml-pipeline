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
            
            # Step 3: Validate and clip values (Silver quality checks)
            df_validated = self._validate_and_clip(df_encoded)
            
            # Step 4: Ensure correct feature order (Silver → Gold preparation)
            df_ordered = self._ensure_feature_order(df_validated)
            
            # Step 5: Apply scaling (Gold layer transformation)
            scaled_array = self._apply_scaling(df_ordered)
            
            # Step 6: Apply feature selection (Gold layer final form)
            final_array = self._apply_feature_selection(scaled_array, df_ordered.columns.tolist())
            
            logger.info(f"Preprocessing complete: {final_array.shape}")
            return final_array
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Preprocessing error: {e}")
    
    def _apply_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding using saved mappings"""
        df_encoded = df.copy()
        
        categorical_features = self.feature_info.get('categorical_features', [])
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                value = df_encoded[feature].iloc[0]
                
                # Try to encode the value
                if feature in self.categorical_encoders:
                    mapping = self.categorical_encoders[feature]
                    if value in mapping:
                        df_encoded[feature] = mapping[value]
                        logger.debug(f"Encoded {feature}: '{value}' → {mapping[value]}")
                    else:
                        # Try as numeric
                        try:
                            df_encoded[feature] = int(float(value))
                            logger.debug(f"Used numeric value for {feature}: {value}")
                        except (ValueError, TypeError):
                            logger.warning(f"Unknown value '{value}' for {feature}, using 0")
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
        expected_features = self.feature_info['expected_raw_features']
        
        # Reindex to match expected feature order, filling missing with 0
        df_ordered = df.reindex(columns=expected_features, fill_value=0)
        
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features filled with 0: {missing_features}")
        
        return df_ordered
    
    def _apply_scaling(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the saved MinMaxScaler"""
        if self.scaler is not None:
            # Apply scaling to the entire feature set
            scaled_array = self.scaler.transform(df)
            logger.debug("Applied saved MinMaxScaler")
            return scaled_array
        else:
            logger.warning("No saved scaler available, using raw values")
            return df.values
    
    def _apply_feature_selection(self, scaled_array: np.ndarray, column_names: List[str]) -> np.ndarray:
        """Apply feature selection to match training data"""
        if 'selected_features' not in self.feature_info:
            # No feature selection was applied
            return scaled_array
        
        selected_features = self.feature_info['selected_features']
        
        # Find indices of selected features
        try:
            selected_indices = [column_names.index(feature) for feature in selected_features 
                              if feature in column_names]
            
            if len(selected_indices) != len(selected_features):
                logger.warning(f"Some selected features not found in input")
            
            # Select only the chosen features
            final_array = scaled_array[:, selected_indices]
            logger.debug(f"Applied feature selection: {scaled_array.shape[1]} → {final_array.shape[1]} features")
            return final_array
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return scaled_array
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Return information about the loaded preprocessing pipeline"""
        return {
            "status": "loaded" if self.is_loaded else "error",
            "error": self.load_error if not self.is_loaded else None,
            "artifacts_loaded": {
                "scaler": self.scaler is not None,
                "categorical_encoders": len(self.categorical_encoders) > 0,
                "feature_info": self.feature_info is not None,
                "pipeline_metadata": self.pipeline_metadata is not None
            },
            "pipeline_info": {
                "version": self.pipeline_metadata.get("pipeline_version", "unknown") if self.pipeline_metadata else "unknown",
                "expected_features": self.feature_info.get("expected_raw_features", []) if self.feature_info else [],
                "final_features": self.feature_info.get("selected_features", []) if self.feature_info else [],
                "transformation_steps": self.pipeline_metadata.get("complete_transformation_steps", []) if self.pipeline_metadata else []
            }
        }
    
    def validate_input(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data before preprocessing"""
        validation_result = {"valid": True, "errors": [], "warnings": []}
        
        if not self.feature_info:
            validation_result["valid"] = False
            validation_result["errors"].append("Feature information not loaded")
            return validation_result
        
        expected_features = self.feature_info.get("expected_raw_features", [])
        
        # Check for missing required features
        missing_features = [f for f in expected_features if f not in patient_data]
        if missing_features:
            validation_result["warnings"].append(f"Missing features will be filled with 0: {missing_features}")
        
        # Check for unexpected features
        unexpected_features = [f for f in patient_data.keys() if f not in expected_features]
        if unexpected_features:
            validation_result["warnings"].append(f"Unexpected features will be ignored: {unexpected_features}")
        
        return validation_result