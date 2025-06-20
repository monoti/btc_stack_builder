"""
Logger module for BTC Stack-Builder Bot.

This module provides a comprehensive logging configuration for the application,
including structured JSON logging, different log levels, and proper error handling.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

# Third-party imports
import structlog
from pythonjsonlogger import jsonlogger

# Default log level
DEFAULT_LOG_LEVEL = "INFO"

# Log directory
LOG_DIR = os.environ.get("BTC_STACK_BUILDER_LOG_DIR", "logs")

# Log file names
MAIN_LOG_FILE = "btc_stack_builder.log"
ERROR_LOG_FILE = "error.log"
TRADE_LOG_FILE = "trades.log"

# Max log file size (10MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files
BACKUP_COUNT = 5


def ensure_log_directory() -> str:
    """
    Ensure the log directory exists.

    Returns:
        Path to the log directory
    """
    # Fetch LOG_DIR from environment variable directly to respond to monkeypatching in tests
    log_dir_path_str = os.environ.get("BTC_STACK_BUILDER_LOG_DIR", "logs")
    log_dir = Path(log_dir_path_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def get_log_level() -> int:
    """
    Get the log level from environment variable or use default.

    Returns:
        Logging level as an integer
    """
    log_level_name = os.environ.get("BTC_STACK_BUILDER_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    return getattr(logging, log_level_name, logging.INFO)


def configure_json_formatter() -> jsonlogger.JsonFormatter:
    """
    Configure JSON formatter for logs.

    Returns:
        Configured JSON formatter
    """
    # Define log format with all relevant fields
    log_format = {
        "timestamp": "%(asctime)s",
        "level": "%(levelname)s",
        "name": "%(name)s",
        "module": "%(module)s",
        "function": "%(funcName)s",
        "line": "%(lineno)d",
        "process": "%(process)d",
        "thread": "%(thread)d",
        "message": "%(message)s",
    }

    # Create JSON formatter
    formatter = jsonlogger.JsonFormatter(
        json_ensure_ascii=False, json_default=str, fmt=json.dumps(log_format)
    )

    return formatter


def configure_console_handler() -> logging.Handler:
    """
    Configure console handler for logs.

    Returns:
        Configured console handler
    """
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(configure_json_formatter())

    return console_handler


def configure_file_handler(log_file: str, level: int = logging.DEBUG) -> logging.Handler:
    """
    Configure file handler for logs with rotation.

    Args:
        log_file: Log file name
        level: Log level for this handler

    Returns:
        Configured file handler
    """
    # Ensure log directory exists
    log_dir = ensure_log_directory()
    log_path = os.path.join(log_dir, log_file)

    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(configure_json_formatter())

    return file_handler


def add_process_info(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Add process information to log record.

    Args:
        logger: Logger instance (unused, part of structlog processor signature)
        method_name: Name of the logging method (e.g., "info", "error")
        event_dict: The event dictionary being processed.

    Returns:
        Modified event dictionary with process information.
    """
    event_dict["pid"] = os.getpid()
    event_dict["process_name"] = sys.argv[0]
    return event_dict


def add_exception_info(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Add exception information to log record if an exception is being handled.
    This processor is designed to be used *before* structlog.processors.format_exc_info.

    Args:
        logger: Logger instance (unused)
        method_name: Name of the logging method
        event_dict: The event dictionary.

    Returns:
        Modified event dictionary.
    """
    # This custom processor is mostly redundant if using `structlog.processors.format_exc_info`
    # and passing `exc_info=True` to the logger call, as `format_exc_info` handles it.
    # However, if we want specific fields for type/message before the full traceback:
    if event_dict.get("exc_info"):  # Check if exc_info was passed to the log call
        exc_info_tuple = sys.exc_info()
        if exc_info_tuple != (None, None, None):
            exception_type, exception_value, _ = exc_info_tuple
            event_dict["exception_type"] = exception_type.__name__ if exception_type else None
            event_dict["exception_message"] = str(exception_value) if exception_value else None
            # `structlog.processors.format_exc_info` will add the full traceback to 'exception'
    return event_dict


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Add ISO-format timestamp to log record.

    Args:
        logger: Logger instance (unused)
        method_name: Name of the logging method
        event_dict: The event dictionary.

    Returns:
        Modified event dictionary with timestamp.
    """
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def setup_logger(name: str = "btc_stack_builder") -> logging.Logger:
    """
    Set up and configure the application logger.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Get log level
    log_level = get_log_level()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_timestamp,
            add_process_info,
            add_exception_info,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Add console handler
    logger.addHandler(configure_console_handler())

    # Add main file handler
    logger.addHandler(configure_file_handler(MAIN_LOG_FILE))

    # Add error file handler (ERROR and above)
    error_handler = configure_file_handler(ERROR_LOG_FILE, logging.ERROR)
    logger.addHandler(error_handler)

    # Add trade file handler (for trade-specific logs)
    # Use the canonical trade logger name, consistent with get_trade_logger()
    trade_logger_instance = logging.getLogger("btc_stack_builder.trades")
    trade_logger_instance.setLevel(log_level)
    trade_logger_instance.propagate = False
    # Clear existing handlers for the trade logger to avoid duplication if setup_logger is called multiple times
    if trade_logger_instance.handlers:
        trade_logger_instance.handlers.clear()
    trade_logger_instance.addHandler(configure_file_handler(TRADE_LOG_FILE))

    return logger


def get_trade_logger() -> logging.Logger:
    """
    Get the trade-specific logger.

    Returns:
        Trade logger instance
    """
    return logging.getLogger("btc_stack_builder.trades")


def log_execution_time(logger: logging.Logger | None = None) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (default: None, uses root logger)

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or logging.getLogger()

            # Log function start
            log.debug(f"Starting {func.__name__}")

            # Record start time
            start_time = datetime.utcnow()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Log function completion
                log.debug(
                    f"Completed {func.__name__}", execution_time=execution_time, status="success"
                )

                return result
            except Exception as e:
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Log function error
                log.error(
                    f"Error in {func.__name__}: {str(e)}",
                    execution_time=execution_time,
                    status="error",
                    exc_info=True,
                )

                # Re-raise the exception
                raise

        return wrapper

    return decorator


def log_strategy_execution(strategy_name: str) -> Callable:
    """
    Decorator to log strategy execution details.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(f"btc_stack_builder.strategies.{strategy_name}")

            # Log strategy execution start
            logger.info(
                f"Executing strategy: {strategy_name}", strategy=strategy_name, action="start"
            )

            # Record start time
            start_time = datetime.utcnow()

            try:
                # Execute strategy
                result = func(*args, **kwargs)

                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Log strategy execution completion
                logger.info(
                    f"Strategy execution completed: {strategy_name}",
                    strategy=strategy_name,
                    action="complete",
                    execution_time=execution_time,
                    status="success",
                )

                return result
            except Exception as e:
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Log strategy execution error
                logger.error(
                    f"Strategy execution failed: {strategy_name} - {str(e)}",
                    strategy=strategy_name,
                    action="error",
                    execution_time=execution_time,
                    status="error",
                    exc_info=True,
                )

                # Re-raise the exception
                raise

        return wrapper

    return decorator


# Initialize default logger
default_app_logger = setup_logger()
