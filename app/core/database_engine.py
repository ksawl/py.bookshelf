# app/core/database_engine.py

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from functools import lru_cache


@lru_cache()
def get_database_engine(database_url: str):
    """Create and cache async database engine"""
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
def get_sync_database_engine(database_url: str):
    """Create and cache synchronous database engine with connection pool"""
    # Keep original sqlite:/// format for sync engine
    sync_url = database_url
    if sync_url.startswith("sqlite+aiosqlite:///"):
        sync_url = sync_url.replace("sqlite+aiosqlite:///", "sqlite:///")

    # SQLite specific configuration
    connect_args = {}
    engine_kwargs = {
        "connect_args": connect_args,
        "echo": False,  # Set to True for SQL debugging
    }
    
    if "sqlite" in sync_url:
        connect_args.update({
            "check_same_thread": False,
        })
        engine_kwargs["poolclass"] = StaticPool
    else:
        # For other databases, use connection pool
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": 10,  # Max pool size
            "max_overflow": 20,  # Additional connections beyond pool_size
            "pool_pre_ping": True,  # Validate connections before use
        })

    engine = create_engine(sync_url, **engine_kwargs)
    return engine


@lru_cache()
def get_sync_session_maker(database_url: str):
    """Create and cache synchronous session maker"""
    engine = get_sync_database_engine(database_url)
    return sessionmaker(
        engine,
        class_=Session,
        expire_on_commit=False,
    )
