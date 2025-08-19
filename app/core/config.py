# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional, Set
from app.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./bookshelf.db"

    # File/general settings
    ALLOWED_EXTENSIONS: Set[str] = {"docx", "odt", "pdf", "txt"}

    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: Optional[str] = "us-east-1"
    PINECONE_INDEX: Optional[str] = None
    PINECONE_SERVERLESS_CLOUD: Optional[str] = "aws"
    PINECONE_SERVERLESS_REGION: Optional[str] = "us-east-1"

    # Processing parameters
    CHUNK_TOKENS: int = 500
    OVERLAP_PCT: float = 0.2
    BATCH_SIZE: int = 100
    ENCODING_NAME: Optional[str] = None

    # Models / keys
    EMBED_MODEL_NAME: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: Optional[str] = None

    # Ollama / LLM
    OLLAMA_API_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    MAX_CONTEXT_CHARS: int = 4000
    TOP_K: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("PINECONE_API_KEY")
    @classmethod
    def validate_pinecone_key(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            raise ConfigurationError(
                "PINECONE_API_KEY is required for Pinecone operations"
            )
        return v

    @field_validator("OLLAMA_API_BASE_URL")
    @classmethod
    def validate_ollama_url(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            raise ConfigurationError(
                "OLLAMA_API_BASE_URL is required for LLM operations"
            )
        return v

    @field_validator("LLM_MODEL")
    @classmethod
    def validate_llm_model(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            raise ConfigurationError("LLM_MODEL is required for question answering")
        return v

    @field_validator("EMBED_MODEL_NAME")
    @classmethod
    def validate_embed_model(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            raise ConfigurationError(
                "EMBED_MODEL_NAME is required for embedding generation"
            )
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v:
            raise ConfigurationError("DATABASE_URL is required")
        return v

    def validate_runtime_dependencies(self) -> None:
        """Validate that all required external services are configured."""
        errors = []

        if not self.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")

        if not self.OLLAMA_API_BASE_URL:
            errors.append("OLLAMA_API_BASE_URL is not set")

        if not self.LLM_MODEL:
            errors.append("LLM_MODEL is not set")

        if not self.EMBED_MODEL_NAME:
            errors.append("EMBED_MODEL_NAME is not set")

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {', '.join(errors)}"
            )


@lru_cache()
def get_settings() -> Settings:
    """
    Cached factory. FastAPI will call get_settings() through Depends,
    and lru_cache guarantees single Settings creation per process.
    """
    return Settings()
