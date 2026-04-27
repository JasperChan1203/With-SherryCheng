"""
Logging utilities for RLQAS Phase 1.

Provides consistent logging configuration across all modules.
"""

import logging
import sys
import os
from typing import Optional

# Default log level from environment variable
LOG_LEVEL = os.environ.get("RLQAS_LOG_LEVEL", "INFO").upper()

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure root logger
def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    stream=sys.stdout,
):
    """Configure root logger with consistent settings.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Format string for log messages
        date_format: Date format string
        stream: Output stream (default: stdout)
    """
    if level is None:
        level = LOG_LEVEL
    if format_string is None:
        format_string = LOG_FORMAT
    if date_format is None:
        date_format = DATE_FORMAT

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(format_string, date_format)
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Also capture warnings from warnings module
    logging.captureWarnings(True)

    # Log configuration
    root_logger.info(f"Logging configured with level {level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger instance
    """
    # Ensure logging is configured (idempotent)
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)


# Default configuration on import
# Commented out to avoid configuring automatically; let modules call get_logger
# setup_logging()