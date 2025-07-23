"""
FastAPI application for Heart Disease Prediction
Main entry point for the REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Import route modules
from api.routes import health, prediction
from api.services.model_service import ModelService
from api.services.database_service import DatabaseService

# Load environment variables
load_dotenv()

# Global services
model_service = None
database_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_service, database_service
    
    # Startup
    print("🚀 Starting Heart Disease Prediction API...")
    
    # Initialize services
    database_service = DatabaseService()
    await database_service.connect()
    
    model_service = ModelService()
    await model_service.load_model()
    
    print("✅ API startup complete!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down API...")
    if database_service:
        await database_service.disconnect()
    print("✅ API shutdown complete!")


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


# Make services available to routes
def get_model_service() -> ModelService:
    return model_service


def get_database_service() -> DatabaseService:
    return database_service


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