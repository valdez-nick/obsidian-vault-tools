"""Logging utilities for CLI commands."""

import structlog
from typing import Optional


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup structured logging with consistent configuration."""
    level = "DEBUG" if verbose else "INFO"
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    import logging
    logging.basicConfig(
        level=getattr(logging, level),
        filename=log_file,
        format='%(message)s'
    )
    
    return structlog.get_logger(__name__)


def get_logger(name: str):
    """Get a logger instance."""
    return structlog.get_logger(name)