"""Structured logging configuration using structlog."""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(environment: str = "development", log_level: str = "INFO") -> None:
    """
    Configure structured logging with JSON output.

    Args:
        environment: 'development' or 'production'
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure standard logging
    log_level_int = getattr(logging, log_level.upper())

    # JSON formatter for stdout
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s",
        timestamp=True,
        rename_fields={"timestamp": "timestamp", "level": "level"},
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(log_level_int)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)
    root_logger.addHandler(console_handler)

    # Remove any existing handlers
    for handler in root_logger.handlers[1:]:
        root_logger.removeHandler(handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Add custom context if needed
            _add_environment_context,
            # Convert to JSON for production
            structlog.processors.JSONRenderer() if environment == "production" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_environment_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add environment-specific context to log events."""
    event_dict["timestamp"] = datetime.utcnow().isoformat()
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


# Convenience logger for module-level use
log = get_logger(__name__)
