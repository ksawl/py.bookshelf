# app/core/database_engine.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from functools import lru_cache


@lru_cache()
def get_database_engine(database_url: str):
    """Create and cache database engine"""
    # Convert sqlite:/// to sqlite+aiosqlite:///
    if database_url.startswith("sqlite:///"):
        database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    
    # SQLite specific configuration
    connect_args = {}
    if "sqlite" in database_url:
        connect_args = {
            "check_same_thread": False,
        }
    
    engine = create_async_engine(
        database_url,
        connect_args=connect_args,
        poolclass=StaticPool if "sqlite" in database_url else None,
        echo=False,  # Set to True for SQL debugging
    )
    
    return engine


@lru_cache()
def get_session_maker(database_url: str):
    """Create and cache session maker"""
    engine = get_database_engine(database_url)
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_async_session(database_url: str) -> AsyncSession:
    """Get async database session"""
    session_maker = get_session_maker(database_url)
    async with session_maker() as session:
        yield session
