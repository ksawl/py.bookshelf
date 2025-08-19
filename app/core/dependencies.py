# app/core/dependencies.py
"""
Centralized dependency injection for FastAPI.
Provides clean separation of concerns and easier testing.
"""

from typing import Annotated, Optional
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.database_service import DatabaseService
from app.services.pinecone_service import PineconeService
from app.services.bookshelf_service import BookshelfService
from app.services.background_processor import BackgroundProcessor
from app.core.logging import get_logger

logger = get_logger(__name__)


_database_service: Optional[DatabaseService] = None

def get_database_service(
    settings: Annotated[Settings, Depends(get_settings)]
) -> DatabaseService:
    """Get database service dependency."""
    global _database_service
    if _database_service is None:
        logger.info("Creating database service", database_url=settings.DATABASE_URL)
        _database_service = DatabaseService(settings=settings)
    return _database_service


def get_pinecone_service(
    settings: Annotated[Settings, Depends(get_settings)]
) -> PineconeService:
    """Get Pinecone service dependency."""
    logger.debug("Creating Pinecone service")
    return PineconeService(settings=settings)


def get_bookshelf_service(
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[DatabaseService, Depends(get_database_service)]
) -> BookshelfService:
    """Get bookshelf service dependency."""
    logger.debug("Creating bookshelf service")
    return BookshelfService(settings=settings, database=database)


def get_background_processor(
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[DatabaseService, Depends(get_database_service)]
) -> BackgroundProcessor:
    """Get background processor dependency."""
    logger.debug("Creating background processor")
    return BackgroundProcessor(settings=settings, database=database)
