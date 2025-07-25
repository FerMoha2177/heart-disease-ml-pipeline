"""
Pydantic models for heart disease prediction API
Defines input/output schemas based on Heart Disease UCI dataset
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from datetime import datetime


class PatientData(BaseModel):
    """
    Input model for patient data based on Heart Disease UCI dataset
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 55,
                "sex": 1,
                "cp": 0,
                "trestbps": 140,
                "chol": 200,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        }
    )
    
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0=normal, 1=fixed defect, 2=reversible defect, 3=unknown)")

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError('Age must be between 1 and 120')
        return v

    @field_validator('chol')
    @classmethod
    def validate_cholesterol(cls, v):
        if v < 100 or v > 600:
            raise ValueError('Cholesterol must be between 100 and 600 mg/dl')
        return v

    @field_validator('thalach')
    @classmethod
    def validate_heart_rate(cls, v):
        if v < 60 or v > 220:
            raise ValueError('Maximum heart rate must be between 60 and 220')
        return v


class PredictionResponse(BaseModel):
    """
    Output model for prediction results
    """
    model_config = ConfigDict(
        protected_namespaces=(),  # Allow model_version field
        json_schema_extra={
            "example": {
                "prediction": 1,
                "probability": 0.75,
                "confidence": "high",
                "risk_factors": ["high_cholesterol", "exercise_angina"],
                "timestamp": "2025-07-22T14:30:00.000000",
                "model_version": "1.0.0"
            }
        }
    )
    
    prediction: int = Field(..., description="Predicted class (0 = no heart disease, 1 = heart disease)")
    probability: float = Field(..., description="Probability of heart disease")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    risk_factors: Optional[list] = Field(None, description="Identified risk factors")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(default="1.0.0", description="Model version used")


class PredictionRequest(BaseModel):
    """
    Request wrapper for batch predictions (future enhancement)
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_data": {
                    "age": 55,
                    "sex": 1,
                    "cp": 0,
                    "trestbps": 140,
                    "chol": 200,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 150,
                    "exang": 0,
                    "oldpeak": 1.5,
                    "slope": 1,
                    "ca": 0,
                    "thal": 2
                },
                "include_risk_factors": True
            }
        }
    )
    
    patient_data: PatientData
    include_risk_factors: bool = Field(default=True, description="Include risk factor analysis")