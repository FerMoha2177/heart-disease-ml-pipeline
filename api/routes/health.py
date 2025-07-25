"""
Health check endpoints for monitoring API and database status
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from api.models.health import HealthResponse, DatabaseHealthResponse
from api.services.database_service import DatabaseService
from api.main import get_database_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint
    Returns API status and timestamp
    """
    return HealthResponse(
        status="healthy",
        message="Heart Disease Prediction API is running",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )


@router.get("/db-health", response_model=DatabaseHealthResponse)
async def database_health_check(
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Database health check endpoint
    Tests MongoDB connection and collection access
    """
    try:
        # Test database connection
        db_status = await db_service.health_check()
        
        if db_status["connected"]:
            return DatabaseHealthResponse(
                status="healthy",
                message="Database connection successful",
                timestamp=datetime.now(datetime.timezone.utc),
                database_name=db_status["database"],
                collections_count=db_status["collections_count"],
                connection_status="connected"
            )
        else:
            return DatabaseHealthResponse(
                status="unhealthy",
                message="Database connection failed",
                timestamp=datetime.now(datetime.timezone.utc),
                database_name="unknown",
                collections_count=0,
                connection_status="disconnected"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database health check failed: {str(e)}"
        )


@router.get("/model-health")
async def model_health_check():
    """
    Model health check endpoint
    Verifies that the ML model is loaded and ready
    """
    try:
        from api.main import get_model_service
        model_service = get_model_service()
        
        if model_service and model_service.is_loaded():
            return {
                "status": "healthy",
                "message": "ML model is loaded and ready",
                "timestamp": datetime.now(datetime.timezone.utc),
                "model_version": getattr(model_service, 'version', '1.0.0'),
                "model_type": getattr(model_service, 'model_type', 'classification')
            }
        else:
            return {
                "status": "unhealthy",
                "message": "ML model is not loaded",
                "timestamp": datetime.now(datetime.timezone.utc)
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model health check failed: {str(e)}"
        )