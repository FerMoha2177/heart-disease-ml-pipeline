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
from api import dependencies

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global services
model_service = None
database_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_service, database_service
    
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
        
        # Initialize model service
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


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML API for predicting heart disease using medallion architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(prediction.router, prefix="/api/v1", tags=["Predictions"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Remove the old dependency functions - now using dependencies.py


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True
    )