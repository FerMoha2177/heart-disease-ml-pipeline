"""
Pydantic models for health check responses
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class HealthResponse(BaseModel):
    """
    Basic health check response model
    """
    status: str
    message: str
    timestamp: datetime
    version: str

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Heart Disease Prediction API is running",
                "timestamp": "2025-07-22T14:30:00.000000",
                "version": "1.0.0"
            }
        }


class DatabaseHealthResponse(BaseModel):
    """
    Database health check response model
    """
    status: str
    message: str
    timestamp: datetime
    database_name: str
    collections_count: int
    connection_status: str

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Database connection successful",
                "timestamp": "2025-07-22T14:30:00.000000",
                "database_name": "healthcare",
                "collections_count": 3,
                "connection_status": "connected"
            }
        }


class ModelHealthResponse(BaseModel):
    """
    Model health check response model
    """
    status: str
    message: str
    timestamp: datetime
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    is_loaded: bool

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "ML model is loaded and ready",
                "timestamp": "2025-07-22T14:30:00.000000",
                "model_version": "1.0.0",
                "model_type": "classification",
                "is_loaded": True
            }
        }