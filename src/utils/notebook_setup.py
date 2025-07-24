async def setup_notebook_environment():
    """Standard setup for all notebooks"""
    from api.services.database_service import DatabaseService
    from config.logging import setup_logging
    import logging
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    dbs = DatabaseService()
    await dbs.connect()
    
    health = await dbs.health_check()
    logger.info(f"Database connected: {health['connected']}")
    logger.info(f"Database collections: {health['collections']}")
    logger.info(f"Database collections count: {health['collections_count']}")

    return dbs, logger
    