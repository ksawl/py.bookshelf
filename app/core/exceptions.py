# app/core/exceptions.py
"""
Custom exceptions for the bookshelf application.
Provides structured error handling with clear error types and messages.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException


class BookshelfError(Exception):
    """Base exception for all bookshelf operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BookshelfError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(BookshelfError):
    """Raised when application configuration is invalid."""

    pass


class PineconeServiceError(BookshelfError):
    """Raised when Pinecone operations fail."""

    pass


class DocumentProcessingError(BookshelfError):
    """Raised when document processing fails."""

    pass


class EmbeddingGenerationError(BookshelfError):
    """Raised when embedding generation fails."""

    pass


class LLMServiceError(BookshelfError):
    """Raised when LLM service operations fail."""

    pass


class DatabaseError(BookshelfError):
    """Raised when database operations fail."""

    pass


class FileProcessingError(BookshelfError):
    """Raised when file processing operations fail."""

    pass


# HTTP Exception factories for FastAPI
def create_http_exception(
    status_code: int, message: str, details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create a structured HTTP exception."""
    detail = {"message": message}
    if details:
        detail["details"] = details
    return HTTPException(status_code=status_code, detail=detail)


def validation_http_error(
    message: str, details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create a 400 validation error."""
    return create_http_exception(400, message, details)


def not_found_http_error(resource: str, identifier: str) -> HTTPException:
    """Create a 404 not found error."""
    return create_http_exception(
        404, f"{resource} not found", {"resource": resource, "identifier": identifier}
    )


def internal_server_http_error(
    message: str, details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create a 500 internal server error."""
    return create_http_exception(500, f"Internal server error: {message}", details)
