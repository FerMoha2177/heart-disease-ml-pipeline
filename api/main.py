"""
FastAPI application for Heart Disease Prediction
Main entry point for the REST API
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging

# Now we can import from config
from config.logging import setup_logging

# Import route modules
from api.routes import health, prediction
from api.services.model_service import MLModelService
from api.services.database_service import DatabaseService
from api.services.preprocessing_service import PreprocessingService
from api import dependencies

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global services
model_service = None
database_service = None
preprocessing_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_service, database_service, preprocessing_service
    
    try:
        # Startup
        logger.info("Starting Heart Disease Prediction API...")
        
        # Initialize database service
        try:
            database_service = DatabaseService()
            db_connected = await database_service.connect()
            if db_connected:
                logger.info("Database service initialized")
                dependencies.set_database_service(database_service)
            else:
                logger.warning("Database connection failed, but continuing...")
        except Exception as e:
            logger.warning(f"Database service failed: {e}, but continuing...")
            database_service = None
        
        # Initialize preprocessing service FIRST (before model service)
        try:
            preprocessing_service = PreprocessingService()
            if preprocessing_service.is_loaded:
                logger.info("Preprocessing service initialized")
                dependencies.set_preprocessing_service(preprocessing_service)
            else:
                logger.error("Preprocessing service failed to load")
        except Exception as e:
            logger.error(f"Preprocessing service error: {e}")
            preprocessing_service = None
        
        # Initialize model service (depends on preprocessing service)
        try:
            model_service = MLModelService()
            model_loaded = await model_service.load_model()
            if model_loaded:
                logger.info("Model service initialized")
                dependencies.set_model_service(model_service)
            else:
                logger.error("Model service failed to load")
        except Exception as e:
            logger.error(f"Model service error: {e}")
            model_service = None
        
        logger.info("API startup complete!")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        yield  # Still start the app even if there are issues
    finally:
        # Shutdown
        logger.info("Shutting down API...")
        if database_service:
            try:
                await database_service.disconnect()
            except:
                pass
        logger.info("API shutdown complete!")


# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered API for predicting heart disease risk",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )