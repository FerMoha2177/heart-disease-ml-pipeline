# api/services/preprocessing_service.py
"""
Preprocessing service that loads and applies the complete preprocessing pipeline
Uses artifacts saved during model training to ensure consistency



"""
# api/services/preprocessing_service.py
"""
Simplified preprocessing service using existing notebook functions
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler

# Import your existing functions
from src.utils.feature_eng_utils import (
    simple_label_encoding,
    one_hot_encoding,
    get_binary_features,
    get_categorical_features
)
from src.utils.data_utils import drop_id

logger = logging.getLogger(__name__)

class PreprocessingService:
    """
    Simple preprocessing using the same functions from notebooks 02 & 03
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.is_loaded = True
        self.load_error = None
        
        # Load the final feature list (what the model expects)
        try:
            import json
            with open(self.model_dir / "feature_columns.json", 'r') as f:
                feature_info = json.load(f)
                self.selected_features = feature_info['selected_features']
                logger.info(f"Loaded {len(self.selected_features)} selected features")
        except Exception as e:
            logger.error(f"Could not load feature info: {e}")
            self.is_loaded = False
    
    def preprocess_for_prediction(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        Apply the SAME transformations from notebooks 02 & 03 to a single row
        """
        try:
            # Step 1: Convert to DataFrame (same as notebooks)
            df = pd.DataFrame([patient_data])
            logger.info(f"Input data shape: {df.shape}")
            
            # Step 2: Apply the SAME transformations from notebook 02 (Bronze → Silver)
            df = self._apply_bronze_to_silver_transforms(df)
            
            # Step 3: Apply the SAME transformations from notebook 03 (Silver → Gold)  
            df = self._apply_silver_to_gold_transforms(df)
            
            # Step 4: Select final features (same as model training)
            df_final = df.reindex(columns=self.selected_features, fill_value=0)
            
            logger.info(f"Final shape: {df_final.shape}")
            return df_final.values
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Preprocessing error: {e}")
    
    def _apply_bronze_to_silver_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Bronze → Silver transforms (from notebook 02)
        """
        # Handle missing values (same logic as notebook 02)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Fill numeric with median
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
            else:
                # Fill categorical with mode or default
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].empty else 0)
        
        # Handle impossible zeros (same as notebook 02)
        impossible_zero_columns = ['trestbps', 'chol', 'thalach']
        for col in impossible_zero_columns:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 120)
        
        logger.debug("Applied Bronze → Silver transforms")
        return df
    
    def _apply_silver_to_gold_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Silver → Gold transforms (from notebook 03) - MODIFIED for API
        """

        logger.info(f"Before transforms - columns: {list(df.columns)}")
        logger.info(f"Before transforms - dtypes: {df.dtypes.to_dict()}")
        logger.info(f"Before transforms - sample values: {df.iloc[0].to_dict()}")
        
        # Step 1: Binary features encoding (same as notebook 03)
        binary_cols = get_binary_features(df)
        if binary_cols:
            df = simple_label_encoding(df, binary_cols)
        
        # Step 2: Multi-class categorical encoding (same as notebook 03)
        categorical_cols = get_categorical_features(df, exclude_binary=True)
        if categorical_cols:
            df = one_hot_encoding(df, categorical_cols)
        
        # Step 3: Min-max scaling (MODIFIED - no target column in API requests)
        df = self._min_max_scale_api(df)
        
        # Step 4: Drop any ID columns if present
        df = drop_id(df)
        
        logger.debug("Applied Silver → Gold transforms")
        logger.info(f"After transforms - columns: {list(df.columns)}")
        logger.info(f"After transforms - dtypes: {df.dtypes.to_dict()}")
        logger.info(f"After transforms - sample values: {df.iloc[0].to_dict()}")
        return df
    
    def _min_max_scale_api(self, df):
        """Use the SAME scaler that was used during training"""
        try:
            # Load the pre-fitted scaler from training
            scaler = joblib.load(self.model_dir / "preprocessing_scaler.pkl")
            
            df_scaled = df.copy()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
            if 'target' in numerical_cols:
                numerical_cols = numerical_cols.drop('target')

            if len(numerical_cols) > 0:
                # Use transform (not fit_transform) with pre-fitted scaler
                df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])
                logger.debug(f"Scaled {len(numerical_cols)} numerical columns using pre-fitted scaler")
            
            return df_scaled
            
        except Exception as e:
            logger.error(f"Could not load pre-fitted scaler: {e}")
            # Fallback - return original data
            return df
        
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get info about preprocessing service"""
        return {
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "selected_features_count": len(self.selected_features) if hasattr(self, 'selected_features') else 0,
            "method": "Using same functions from notebooks 02 & 03"
        }
    
    def validate_input(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation"""
        return {"valid": True, "errors": [], "warnings": []}
# import pandas as pd
# import numpy as np
# import joblib
# import json
# import logging
# from pathlib import Path
# from typing import Dict, Any, Optional, List

# logger = logging.getLogger(__name__)

# class PreprocessingService:
#     """
#     Loads saved preprocessing artifacts and applies the complete transformation pipeline

#     Raw Input → Basic Encoding (sex,fbs,exang) → One-Hot Encoding (cp,restecg,slope) → 
#     Validation → Feature Ordering → Scaling → Feature Selection → Model-Ready Data
#     """
    
#     def __init__(self, model_dir: str = "models"):
#         self.model_dir = Path(model_dir)
        
#         # Loaded artifacts
#         self.scaler = None
#         self.categorical_encoders = {}
#         self.feature_info = None
#         self.pipeline_metadata = None
        
#         # Processing state
#         self.is_loaded = False
#         self.load_error = None
        
#         # Load all artifacts at initialization
#         try:
#             self._load_all_artifacts()
#         except Exception as e:
#             self.load_error = str(e)
#             logger.error(f"Failed to load preprocessing artifacts: {e}")
#             self.is_loaded = False
    
#     def _load_all_artifacts(self):
#         """Load all saved preprocessing artifacts"""
#         try:
#             logger.info("Loading preprocessing artifacts...")
            
#             # 1. Load fitted scaler
#             scaler_path = self.model_dir / "preprocessing_scaler.pkl"
#             if scaler_path.exists():
#                 self.scaler = joblib.load(scaler_path)
#                 logger.info("Loaded fitted MinMaxScaler")
#             else:
#                 logger.warning("No saved scaler found")
            
#             # 2. Load categorical encoders/mappings
#             encoders_path = self.model_dir / "categorical_encoders.pkl"
#             if encoders_path.exists():
#                 self.categorical_encoders = joblib.load(encoders_path)
#                 logger.info("Loaded categorical encoders")
#             else:
#                 logger.warning("No categorical encoders found")
#                 self._create_fallback_encoders()
            
#             # 3. Load feature information
#             feature_path = self.model_dir / "feature_columns.json"
#             if feature_path.exists():
#                 with open(feature_path, 'r') as f:
#                     self.feature_info = json.load(f)
#                 logger.info(f"Loaded feature info: {len(self.feature_info['feature_columns'])} features")
#             else:
#                 logger.error("No feature information found")
#                 raise FileNotFoundError("feature_columns.json is required")
            
#             # 4. Load pipeline metadata
#             metadata_path = self.model_dir / "preprocessing_metadata.json"
#             if metadata_path.exists():
#                 with open(metadata_path, 'r') as f:
#                     self.pipeline_metadata = json.load(f)
#                 logger.info(f"Loaded pipeline metadata v{self.pipeline_metadata.get('pipeline_version', 'unknown')}")
            
#             self.is_loaded = True
#             logger.info("All preprocessing artifacts loaded successfully!")
            
#         except Exception as e:
#             self.load_error = str(e)
#             logger.error(f"Failed to load preprocessing artifacts: {e}")
#             self.is_loaded = False
    
#     def _create_fallback_encoders(self):
#         """Create fallback encoders if saved ones not available"""
#         logger.warning("Creating fallback categorical encoders...")
#         self.categorical_encoders = {
#             'sex': {'male': 1, 'female': 0, 'Male': 1, 'Female': 0, 'M': 1, 'F': 0},
#             'cp': {'typical_angina': 0, 'atypical_angina': 1, 'non_anginal_pain': 2, 'asymptomatic': 3},
#             'fbs': {'false': 0, 'true': 1, 'no': 0, 'yes': 1},
#             'restecg': {'normal': 0, 'ST-T abnormality': 1, 'LV hypertrophy': 2},
#             'exang': {'no': 0, 'yes': 1, 'false': 0, 'true': 1},
#             'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
#             'thal': {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2, 'unknown': 3}
#         }
    
#     def preprocess_for_prediction(self, patient_data: Dict[str, Any]) -> np.ndarray:
#         """
#         MAIN METHOD: Apply complete preprocessing pipeline to raw input
        
#         Args:
#             patient_data: Raw patient data from API request
            
#         Returns:
#             Preprocessed numpy array ready for model prediction
            
#         Raises:
#             ValueError: If preprocessing fails or artifacts not loaded
#         """
#         if not self.is_loaded:
#             raise ValueError(f"Preprocessing service not properly loaded: {self.load_error}")
        
#         try:
#             logger.info("Starting preprocessing pipeline...")
            
#             # Step 1: Convert to DataFrame
#             df = pd.DataFrame([patient_data])
#             logger.debug(f"Input data shape: {df.shape}")
            
#             # Step 2: Apply basic categorical encoding (sex, fbs, exang → 0/1)
#             df_encoded = self._apply_basic_categorical_encoding(df)
            
#             # Step 3: Apply one-hot encoding (cp, restecg, slope → multiple columns)
#             df_onehot = self._apply_one_hot_encoding(df_encoded)
            
#             # Step 4: Validate and clip values
#             df_validated = self._validate_and_clip(df_onehot)
            
#             # Step 5: Ensure feature order matches expected input
#             df_ordered = self._ensure_feature_order(df_validated)
            
#             # Step 6: Apply scaling (Silver → Gold)
#             df_scaled = self._apply_scaling(df_ordered)
            
#             # Step 7: Select final features for model
#             processed_array = self._select_final_features(df_scaled)
            
#             logger.info(f"Preprocessing complete. Output shape: {processed_array.shape}")
#             return processed_array
            
#         except Exception as e:
#             logger.error(f"Preprocessing failed: {str(e)}")
#             raise ValueError(f"Preprocessing error: {str(e)}")
    
#     def _apply_basic_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Apply basic categorical encoding for binary features (sex, fbs, exang)"""
#         df_encoded = df.copy()
        
#         # Binary features that get simple 0/1 encoding
#         binary_features = ['sex', 'fbs', 'exang']
        
#         for feature in binary_features:
#             if feature in df_encoded.columns:
#                 value = df_encoded[feature].iloc[0]
                
#                 if feature in self.categorical_encoders:
#                     encoder_map = self.categorical_encoders[feature]
#                     if value in encoder_map:
#                         df_encoded[feature] = encoder_map[value]
#                         logger.debug(f"Encoded {feature}: '{value}' → {encoder_map[value]}")
#                     else:
#                         # Try to use as numeric
#                         try:
#                             df_encoded[feature] = int(float(value))
#                             logger.debug(f"Used numeric value for {feature}: {value}")
#                         except (ValueError, TypeError):
#                             logger.warning(f"Unknown value '{value}' for {feature}, using 0")
#                             df_encoded[feature] = 0
        
#         return df_encoded
    
#     def _apply_one_hot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Apply one-hot encoding for multi-class categorical features (cp, restecg, slope)"""
#         df_onehot = df.copy()
        
#         # Multi-class features that need one-hot encoding
#         multiclass_features = ['cp', 'restecg', 'slope']
        
#         for feature in multiclass_features:
#             if feature in df_onehot.columns:
#                 value = df_onehot[feature].iloc[0]
                
#                 # Convert string categories to numeric first if needed
#                 if isinstance(value, str) and feature in self.categorical_encoders:
#                     encoder_map = self.categorical_encoders[feature]
#                     if value in encoder_map:
#                         value = encoder_map[value]
#                     else:
#                         logger.warning(f"Unknown string value '{value}' for {feature}, using 0")
#                         value = 0
                
#                 # Ensure value is numeric
#                 try:
#                     numeric_value = int(float(value))
#                 except (ValueError, TypeError):
#                     logger.warning(f"Could not convert {feature} value '{value}' to numeric, using 0")
#                     numeric_value = 0
                
#                 # Create one-hot encoded columns based on the feature type
#                 if feature == 'cp':
#                     # cp: 0=typical_angina, 1=atypical_angina, 2=non_anginal, 3=asymptomatic
#                     df_onehot[f'{feature}_asymptomatic'] = 1 if numeric_value == 3 else 0
#                     df_onehot[f'{feature}_atypical angina'] = 1 if numeric_value == 1 else 0
#                     df_onehot[f'{feature}_non-anginal'] = 1 if numeric_value == 2 else 0
#                     # Note: typical_angina is the reference category (all others = 0)
                    
#                 elif feature == 'restecg':
#                     # restecg: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy
#                     df_onehot[f'{feature}_normal'] = 1 if numeric_value == 0 else 0
#                     df_onehot[f'{feature}_st-t abnormality'] = 1 if numeric_value == 1 else 0
#                     # Note: LV hypertrophy is the reference category
                    
#                 elif feature == 'slope':
#                     # slope: 0=upsloping, 1=flat, 2=downsloping
#                     df_onehot[f'{feature}_flat'] = 1 if numeric_value == 1 else 0
#                     df_onehot[f'{feature}_not_tested'] = 0  # This seems to be an artifact from training
#                     df_onehot[f'{feature}_upsloping'] = 1 if numeric_value == 0 else 0
#                     # Note: downsloping is the reference category
                
#                 # Remove original column
#                 df_onehot = df_onehot.drop(feature, axis=1)
#                 logger.debug(f"One-hot encoded {feature} (value={numeric_value})")
        
#         return df_onehot
    
#     def _validate_and_clip(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Validate and clip values to reasonable ranges"""
#         df_validated = df.copy()
        
#         # Get validation ranges from metadata
#         if self.pipeline_metadata and 'validation_ranges' in self.pipeline_metadata:
#             ranges = self.pipeline_metadata['validation_ranges']
            
#             for feature, range_info in ranges.items():
#                 if feature in df_validated.columns:
#                     min_val, max_val = range_info['min'], range_info['max']
#                     original_val = df_validated[feature].iloc[0]
#                     clipped_val = np.clip(original_val, min_val, max_val)
                    
#                     if original_val != clipped_val:
#                         logger.warning(f"Clipped {feature}: {original_val} → {clipped_val}")
                    
#                     df_validated[feature] = clipped_val
        
#         return df_validated
    
#     def _ensure_feature_order(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Ensure all expected features are present and in correct order"""
#         # Get expected features after one-hot encoding from metadata
#         if self.pipeline_metadata and 'transformation_details' in self.pipeline_metadata:
#             scaling_features = self.pipeline_metadata['transformation_details']['scaling']['numerical_features']
#         else:
#             # Fallback: expected features after one-hot encoding
#             scaling_features = [
#                 'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak',
#                 'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal',
#                 'restecg_normal', 'restecg_st-t abnormality',
#                 'slope_flat', 'slope_not_tested', 'slope_upsloping'
#             ]
#             logger.warning("Using fallback feature list for scaling")
        
#         # Reindex to match expected feature order, filling missing with 0
#         df_ordered = df.reindex(columns=scaling_features, fill_value=0)
        
#         missing_features = [f for f in scaling_features if f not in df.columns]
#         if missing_features:
#             logger.debug(f"Added missing features with 0: {missing_features}")
        
#         return df_ordered
    
#     def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Apply MinMax scaling using fitted scaler"""
#         if not self.scaler:
#             logger.warning("No scaler available, skipping scaling")
#             return df
            
#         try:
#             df_scaled = df.copy()
#             scaled_data = self.scaler.transform(df_scaled)
#             df_scaled = pd.DataFrame(scaled_data, columns=df_scaled.columns, index=df_scaled.index)
#             logger.debug("Applied MinMax scaling successfully")
#             return df_scaled
#         except Exception as e:
#             logger.error(f"Scaling failed: {e}")
#             return df
    
#     def _select_final_features(self, df: pd.DataFrame) -> np.ndarray:
#         """Select only the features used by the final model"""
#         if self.feature_info and 'selected_features' in self.feature_info:
#             selected_features = self.feature_info['selected_features']
            
#             # Filter DataFrame to only include selected features
#             df_final = df.reindex(columns=selected_features, fill_value=0)
#             logger.debug(f"Selected {len(selected_features)} features for model")
#             return df_final.values
#         else:
#             logger.warning("No feature selection info, using all features")
#             return df.values
    
#     def get_preprocessing_info(self) -> Dict[str, Any]:
#         """Get information about loaded preprocessing artifacts"""
#         return {
#             "is_loaded": self.is_loaded,
#             "load_error": self.load_error,
#             "artifacts_status": {
#                 "scaler": self.scaler is not None,
#                 "categorical_encoders": len(self.categorical_encoders) > 0,
#                 "feature_info": self.feature_info is not None,
#                 "pipeline_metadata": self.pipeline_metadata is not None
#             },
#             "pipeline_info": {
#                 "version": self.pipeline_metadata.get("pipeline_version", "unknown") if self.pipeline_metadata else "unknown",
#                 "expected_features": self._get_expected_raw_features(),
#                 "final_features": self.feature_info.get("selected_features", []) if self.feature_info else [],
#                 "transformation_steps": self.pipeline_metadata.get("complete_transformation_steps", []) if self.pipeline_metadata else []
#             }
#         }
    
#     def _get_expected_raw_features(self) -> List[str]:
#         """Get expected raw features from the correct location"""
#         if self.pipeline_metadata and 'api_input_format' in self.pipeline_metadata:
#             return self.pipeline_metadata['api_input_format']['expected_raw_features']
#         else:
#             return [
#                 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
#                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
#             ]
    
#     def validate_input(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Validate input data before preprocessing"""
#         validation_result = {"valid": True, "errors": [], "warnings": []}
        
#         expected_features = self._get_expected_raw_features()
        
#         # Check for missing required features
#         missing_features = [f for f in expected_features if f not in patient_data]
#         if missing_features:
#             validation_result["warnings"].append(f"Missing features will be filled with 0: {missing_features}")
        
#         # Check for unexpected features
#         unexpected_features = [f for f in patient_data.keys() if f not in expected_features]
#         if unexpected_features:
#             validation_result["warnings"].append(f"Unexpected features will be ignored: {unexpected_features}")
        
#         return validation_result