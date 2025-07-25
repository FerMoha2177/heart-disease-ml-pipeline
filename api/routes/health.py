"""
Health check endpoints for monitoring API and database status
"""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import logging
from config.logging import setup_logging
setup_logging()

from api.dependencies import get_model_service, get_database_service
from api.services.model_service import MLModelService
from api.services.database_service import DatabaseService



# Remove circular import - we'll get services via dependency injection
logger = logging.getLogger(__name__)

router = APIRouter()
@router.get("/health")
async def health_check():
    """
    Basic health check endpoint - REQUIRED BY ASSESSMENT
    Returns 200 OK if API is running
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Heart Disease Prediction API is running"
    }

@router.get("/health/detailed")
async def detailed_health_check(
    model_service: MLModelService = Depends(get_model_service),
    database_service: DatabaseService = Depends(get_database_service)
):
    """
    Detailed health check including model and database status
    """
    try:
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check model service
        if model_service:
            model_health = model_service.health_check()
            health_info["services"]["model"] = model_health
        else:
            health_info["services"]["model"] = {
                "status": "unavailable",
                "error": "Model service not initialized"
            }
        
        # Check database service  
        if database_service:
            db_health = await database_service.health_check()
            health_info["services"]["database"] = db_health
        else:
            health_info["services"]["database"] = {
                "status": "unavailable", 
                "error": "Database service not initialized"
            }
        
        # Determine overall status
        model_ok = health_info["services"]["model"].get("status") == "healthy"
        db_ok = health_info["services"]["database"].get("connected", False)
        
        if not model_ok or not db_ok:
            health_info["status"] = "degraded"
            health_info["issues"] = []
            
            if not model_ok:
                health_info["issues"].append("ML model not loaded")
            if not db_ok:
                health_info["issues"].append("Database connection issues")
        
        return health_info
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/model-info")
async def get_model_info(model_service: MLModelService = Depends(get_model_service)):
    """
    Get detailed information about the loaded model
    """
    try:
        if not model_service:
            raise HTTPException(status_code=503, detail="Model service not available")
        
        return model_service.get_model_info()
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve model info: {str(e)}")