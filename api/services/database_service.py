"""
Database service for MongoDB operations
Handles connections and CRUD operations for medallion architecture
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    MongoDB database service for medallion architecture
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        self.database_name = os.getenv("MONGODB_DATABASE_NAME", "healthcare")
        
        # Collection names for medallion architecture
        self.collections = {
            "bronze": "heart_disease_bronze",
            "silver": "heart_disease_silver", 
            "gold": "heart_disease_gold"
        }
    
    async def connect(self) -> bool:
        """
        Establish connection to MongoDB
        """
        try:
            if not self.connection_string:
                raise ValueError("MongoDB connection string not found in environment variables")
            
            self.client = AsyncIOMotorClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000
            )
            
            # Test the connection
            await self.client.admin.command('ping')
            
            self.database = self.client[self.database_name]
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            
            # Ensure collections exist
            await self._ensure_collections()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    async def disconnect(self):
        """
        Close MongoDB connection
        """
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health and return status
        """
        try:
            if not self.client:
                return {"connected": False, "error": "No client connection"}
            
            # Ping the database
            await self.client.admin.command('ping')
            
            # Count collections
            collections = await self.database.list_collection_names()
            
            return {
                "connected": True,
                "database": self.database_name,
                "collections_count": len(collections),
                "collections": collections
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    async def _ensure_collections(self):
        """
        Ensure all required collections exist
        """
        try:
            existing_collections = await self.database.list_collection_names()
            
            for layer, collection_name in self.collections.items():
                if collection_name not in existing_collections:
                    await self.database.create_collection(collection_name)
                    logger.info(f"Created collection: {collection_name}")
                    
        except Exception as e:
            logger.error(f"Error ensuring collections: {str(e)}")
    
    # Bronze Layer Operations
    async def insert_bronze_data(self, data: List[Dict]) -> bool:
        """
        Insert raw data into Bronze layer
        """
        try:
            collection = self.database[self.collections["bronze"]]
            result = await collection.insert_many(data)
            logger.info(f"Inserted {len(result.inserted_ids)} records into Bronze layer")
            return True
        except Exception as e:
            logger.error(f"Error inserting Bronze data: {str(e)}")
            return False
    
    async def get_bronze_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve data from Bronze layer
        """
        try:
            collection = self.database[self.collections["bronze"]]
            cursor = collection.find({})
            
            if limit:
                cursor = cursor.limit(limit)
                
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Error retrieving Bronze data: {str(e)}")
            return []
    
    # Silver Layer Operations
    async def insert_silver_data(self, data: List[Dict]) -> bool:
        """
        Insert processed data into Silver layer
        """
        try:
            collection = self.database[self.collections["silver"]]
            
            # Clear existing data first (for reprocessing)
            await collection.delete_many({})
            
            result = await collection.insert_many(data)
            logger.info(f"Inserted {len(result.inserted_ids)} records into Silver layer")
            return True
        except Exception as e:
            logger.error(f"Error inserting Silver data: {str(e)}")
            return False
    
    async def get_silver_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve data from Silver layer
        """
        try:
            collection = self.database[self.collections["silver"]]
            cursor = collection.find({})
            
            if limit:
                cursor = cursor.limit(limit)
                
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Error retrieving Silver data: {str(e)}")
            return []
    
    # Gold Layer Operations
    async def insert_gold_data(self, data: List[Dict]) -> bool:
        """
        Insert refined data into Gold layer
        """
        try:
            collection = self.database[self.collections["gold"]]
            
            # Clear existing data first (for reprocessing)
            await collection.delete_many({})
            
            result = await collection.insert_many(data)
            logger.info(f"Inserted {len(result.inserted_ids)} records into Gold layer")
            return True
        except Exception as e:
            logger.error(f"Error inserting Gold data: {str(e)}")
            return False
    
    async def get_gold_data(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve data from Gold layer
        """
        try:
            collection = self.database[self.collections["gold"]]
            cursor = collection.find({})
            
            if limit:
                cursor = cursor.limit(limit)
                
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Error retrieving Gold data: {str(e)}")
            return []
    
    # General collection operations
    async def get_collection_count(self, layer: str) -> int:
        """
        Get document count for a specific layer
        """
        try:
            collection_name = self.collections.get(layer)
            if not collection_name:
                return 0
                
            collection = self.database[collection_name]
            return await collection.count_documents({})
        except Exception as e:
            logger.error(f"Error counting documents in {layer}: {str(e)}")
            return 0
    
    async def get_sample_records(self, layer: str, count: int = 2) -> List[Dict]:
        """
        Get sample records from a specific layer
        """
        try:
            collection_name = self.collections.get(layer)
            if not collection_name:
                return []
                
            collection = self.database[collection_name]
            cursor = collection.find({}).limit(count)
            return await cursor.to_list(length=count)
        except Exception as e:
            logger.error(f"Error getting sample records from {layer}: {str(e)}")
            return []