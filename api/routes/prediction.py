"""
Prediction routes for heart disease prediction API
Fixed to use real model predictions
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging
from config.logging import setup_logging

from api.models.prediction import PredictionResponse, PatientData
from api.dependencies import get_model_service
from api.services.model_service import MLModelService

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(patient_data: PatientData, model_service: MLModelService = Depends(get_model_service)):
    """
    Predict heart disease risk for a patient using trained ML model
    
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
        logger.info(f"Received prediction request for patient: age={patient_data.age}")
        
        # Make prediction using real model
        prediction_result = await model_service.predict(patient_data.dict())
        
        # Log successful prediction
        logger.info(f"Prediction successful: {prediction_result['prediction']}")
        
        # Return structured response
        return PredictionResponse(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            risk_factors=prediction_result.get("risk_factors", []),
            timestamp=datetime.utcnow(),
            model_version="1.0.0"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 503 from dependencies)
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your prediction request"
        )


# dummy endpoint for testing/debugging
@router.post("/dummy-predict")
async def dummy_predict_heart_disease(patient_data: PatientData):
    """
    Simple dummy prediction endpoint for testing
    NOTE: This uses fake logic, not the real ML model
    """
    try:
        logger.info(f"Got DUMMY prediction request for patient age: {patient_data.age}")
        
        # DUMMY LOGIC - NOT REAL PREDICTION
        risk_score = (patient_data.age * 0.01 + 
                     patient_data.chol * 0.001 + 
                     patient_data.trestbps * 0.002)
        
        prediction = 1 if risk_score > 2.0 else 0
        probability = min(risk_score / 4.0, 1.0)
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence="medium",
            model_version="dummy-1.0.0",
            timestamp=datetime.utcnow(),
            risk_factors=["dummy_age", "dummy_cholesterol", "dummy_blood_pressure"]
        )
        
    except Exception as e:
        logger.error(f"Dummy prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Dummy prediction failed")