# api/dependencies.py
from api.services.model_service import MLModelService
from api.services.database_service import DatabaseService

model_service: MLModelService = None
database_service: DatabaseService = None

def get_model_service() -> MLModelService:
    return model_service

def get_database_service() -> DatabaseService:
    return database_service

def set_model_service(service: MLModelService):
    global model_service
    model_service = service

def set_database_service(service: DatabaseService):
    global database_service
    database_service = service