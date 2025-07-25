"""
Pydantic routes for heart disease prediction API
Defines endpoints for prediction from the models
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from api.services.model_service import MLModelService
from api.main import get_model_service
from api.models.prediction import PredictionRequest, PredictionResponse, PatientData

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(
    patient_data: PatientData,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Predict heart disease risk for a patient
    
    **Input**: Patient data including age, sex, chest pain type, etc.
    **Output**: Prediction (0/1), probability, confidence level, and risk factors
    
    **Example Request:**
    ```json
    {
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
    ```
    """
    try:
        # Log incoming request
        logger.info(f"Received prediction request for patient: age={patient_data.age}, sex={patient_data.sex}")
        
        # Check if model is loaded
        if not model_service or not model_service.is_loaded():
            logger.error("Model service not available or model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model is not available. Please try again later."
            )
        
        # Convert patient data to prediction format
        prediction_result = await model_service.predict(patient_data)
        
        # Log successful prediction
        logger.info(f"Prediction successful: {prediction_result['prediction']} (probability: {prediction_result['probability']:.3f})")
        
        # Return structured response
        return PredictionResponse(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            confidence=prediction_result["confidence"],
            risk_factors=prediction_result.get("risk_factors", []),
            timestamp=datetime.utcnow(),
            model_version=model_service.version
        )
        
    except ValueError as ve:
        # Handle validation errors
        logger.warning(f"Validation error in prediction: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data: {str(ve)}"
        )
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your prediction request"
        )


# dummy endpoint for testing
@router.post("/dummy-predict")
async def dummy_predict_heart_disease(patient_data: PatientData):
    """
    Simple prediction endpoint
    
    Input: Patient data (age, sex, chol, etc.)
    Output: Prediction result
    """
    try:
        logger.info(f"Got prediction request for patient age: {patient_data.age}")
        
        risk_score = (patient_data.age * 0.01 + 
                     patient_data.chol * 0.001 + 
                     patient_data.trestbps * 0.002)
        
        prediction = 1 if risk_score > 2.0 else 0
        probability = min(risk_score / 4.0, 1.0)
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence="medium",
            model_version="1.0.0",
            timestamp=datetime.now(),
            risk_factors=["age", "cholesterol", "blood pressure"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


