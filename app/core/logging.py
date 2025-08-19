# app/core/logging.py
import logging
import structlog
from typing import Any, Dict
import sys


def setup_logging(log_level: str = "INFO", enable_json: bool = False) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_json: Whether to output JSON formatted logs
    """
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add logger to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(func_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Create a structured log context for function calls."""
    return {
        "function": func_name,
        "params": {k: v for k, v in kwargs.items() if not k.startswith('_')}
    }
