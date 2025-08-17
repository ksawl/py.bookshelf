# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Set


class Settings(BaseSettings):
    # файловые/общие
    ALLOWED_EXTENSIONS: Set[str] = {"docx", "odt", "pdf", "txt"}

    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: Optional[str] = "us-east-1"
    PINECONE_INDEX: Optional[str] = None
    PINECONE_SERVERLESS_CLOUD: Optional[str] = "aws"
    PINECONE_SERVERLESS_REGION: Optional[str] = "us-east-1"

    # параметры обработки
    CHUNK_TOKENS: int = 500
    OVERLAP_PCT: float = 0.2
    BATCH_SIZE: int = 100
    ENCODING_NAME: Optional[str] = None

    # модели / ключи
    EMBED_MODEL_NAME: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    # ollama / llm
    OLLAMA_API_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    MAX_CONTEXT_CHARS: int = 4000
    TOP_K: int = 5

    # pydantic-settings uses model_config / SettingsConfigDict
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """
    Кешированная фабрика. FastAPI будет вызывать get_settings() через Depends,
    а lru_cache гарантирует единоразовое создание Settings на процесс.
    """
    return Settings()
