"""
Dependency injection for FastAPI services
Provides centralized access to application services
"""

from fastapi import HTTPException
from typing import Optional
import logging

from api.services.model_service import MLModelService
from api.services.database_service import DatabaseService
from api.services.preprocessing_service import PreprocessingService

logger = logging.getLogger(__name__)

# Global service instances (set by main.py during startup)
_model_service: Optional[MLModelService] = None
_database_service: Optional[DatabaseService] = None
_preprocessing_service: Optional[PreprocessingService] = None

def set_model_service(service: MLModelService):
    """Set the global model service instance"""
    global _model_service
    _model_service = service
    logger.info("Model service registered in dependencies")

def set_database_service(service: DatabaseService):
    """Set the global database service instance"""
    global _database_service
    _database_service = service
    logger.info("Database service registered in dependencies")

def set_preprocessing_service(service: PreprocessingService):
    """Set the global preprocessing service instance"""
    global _preprocessing_service
    _preprocessing_service = service
    logger.info("Preprocessing service registered in dependencies")

def get_model_service() -> MLModelService:
    """
    Dependency to get the model service
    Raises HTTP 503 if service is not available
    """
    if _model_service is None:
        logger.error("Model service not available")
        raise HTTPException(
            status_code=503,
            detail="Model service is not available. Please try again later."
        )
    
    if not _model_service.model:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded. Please try again later."
        )
    
    return _model_service

def get_database_service() -> DatabaseService:
    """
    Dependency to get the database service
    Raises HTTP 503 if service is not available
    """
    if _database_service is None:
        logger.error("Database service not available")
        raise HTTPException(
            status_code=503,
            detail="Database service is not available. Please try again later."
        )
    
    return _database_service

def get_preprocessing_service() -> PreprocessingService:
    """
    Dependency to get the preprocessing service
    Raises HTTP 503 if service is not available
    """
    if _preprocessing_service is None:
        logger.error("Preprocessing service not available")
        raise HTTPException(
            status_code=503,
            detail="Preprocessing service is not available. Please try again later."
        )
    
    if not _preprocessing_service.is_loaded:
        logger.error("Preprocessing artifacts not loaded")
        raise HTTPException(
            status_code=503,
            detail="Preprocessing pipeline is not loaded. Please try again later."
        )
    
    return _preprocessing_service

def get_model_service_optional() -> Optional[MLModelService]:
    """
    Get model service without raising exceptions
    Returns None if not available
    """
    return _model_service

def get_database_service_optional() -> Optional[DatabaseService]:
    """
    Get database service without raising exceptions
    Returns None if not available
    """
    return _database_service

def get_preprocessing_service_optional() -> Optional[PreprocessingService]:
    """
    Get preprocessing service without raising exceptions
    Returns None if not available
    """
    return _preprocessing_service