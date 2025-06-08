import json
import logging

import shutil
from io import StringIO
from pathlib import Path

import pytest

# Module to be tested
from btc_stack_builder.core.logger import (
    MAIN_LOG_FILE,
    TRADE_LOG_FILE,
    ensure_log_directory,
    setup_logger,
)

# The global instance `default_app_logger` from logger.py is not directly used in tests,
# as tests will call setup_logger() for fresh instances.

TEST_LOG_DIR = "test_logs_temp"


@pytest.fixture(autouse=True)
def setup_teardown_test_log_dir(monkeypatch):
    """Ensure a clean test log directory for each test and set env var."""
    # Set environment variable for log directory
    monkeypatch.setenv("BTC_STACK_BUILDER_LOG_DIR", TEST_LOG_DIR)

    # Ensure log directory exists at the start of the test session
    log_dir_path = Path(TEST_LOG_DIR)
    if not log_dir_path.exists():
        log_dir_path.mkdir(parents=True, exist_ok=True)

    yield  # Test runs here

    # Teardown: Remove the test log directory after all tests in the module are done
    # This is tricky with autouse; a session-scoped fixture might be better for overall cleanup.
    # For now, let individual tests clean up specific files if needed, or do it manually after a run.
    # If many files are created, a more robust cleanup is needed.
    # For simplicity, let's try to remove it if it's empty or contains expected log files.
    if log_dir_path.exists():
        shutil.rmtree(log_dir_path, ignore_errors=True)


@pytest.fixture
def clean_handlers():
    """Fixture to remove handlers from any loggers created by the module."""
    yield
    # Cleanup: remove handlers from loggers that might have been configured
    loggers_to_clear = [
        logging.getLogger("test_logger"),
        logging.getLogger("btc_stack_builder"),
        logging.getLogger("btc_stack_builder.trades"),
    ]
    for log_instance in loggers_to_clear:
        if hasattr(log_instance, "handlers"):
            for handler in list(log_instance.handlers):  # Iterate over a copy
                log_instance.removeHandler(handler)
                handler.close()


class TestLoggerSetup:

    def test_logger_creation_with_default_name_and_level(self, monkeypatch, clean_handlers):
        monkeypatch.setenv("BTC_STACK_BUILDER_LOG_LEVEL", "INFO")
        # Reload the logger module to apply monkeypatched env vars if logger is initialized at import time
        # However, setup_logger is a function, so we call it directly.

        logger = setup_logger()  # Uses default name "btc_stack_builder"

        assert logger.name == "btc_stack_builder"
        assert logger.getEffectiveLevel() == logging.INFO
        # Expect 3 handlers: console, main file, error file for the main logger
        assert len(logger.handlers) == 3

        # Clean up specifically for this logger instance
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def test_logger_creation_with_custom_name_and_level(self, monkeypatch, clean_handlers):
        monkeypatch.setenv("BTC_STACK_BUILDER_LOG_LEVEL", "DEBUG")
        logger_name = "my_custom_test_logger"

        logger = setup_logger(name=logger_name)

        assert logger.name == logger_name
        assert logger.getEffectiveLevel() == logging.DEBUG
        assert len(logger.handlers) == 3  # Console, main file, error file

        # Clean up specifically for this logger instance
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def test_ensure_log_directory_creation(self, monkeypatch):
        # This test relies on the setup_teardown_test_log_dir fixture to set the base dir
        # and clean it up.
        log_dir_path = Path(TEST_LOG_DIR)

        # Remove it if it exists from a previous test to ensure ensure_log_directory creates it
        if log_dir_path.exists():
            shutil.rmtree(log_dir_path)

        # ensure_log_directory now reads from os.environ directly,
        # which is set by the setup_teardown_test_log_dir fixture's monkeypatch.setenv.
        # No need to setattr on the module if it's internally using os.environ.get.
        returned_log_dir = ensure_log_directory()

        assert log_dir_path.exists()
        assert log_dir_path.is_dir()
        assert str(log_dir_path) == returned_log_dir

        # Clean up this specific directory immediately for this test
        if log_dir_path.exists():
            shutil.rmtree(log_dir_path)


class TestLogOutput:

    @pytest.fixture
    def capturing_logger(self, monkeypatch):
        monkeypatch.setenv("BTC_STACK_BUILDER_LOG_LEVEL", "DEBUG")
        log_stream = StringIO()

        # Ensure structlog is configured by running the application's setup
        # Use a throwaway name for this initial configuration run if necessary,
        # to avoid interfering with the logger name we actually want to capture.
        setup_logger(name="temp_config_logger_for_capturing_fixture")

        # Get the logger instance using structlog's own API.
        # This ensures it's a structlog BoundLogger.
        # Need to import structlog for this.
        import structlog

        structlog_logger_to_capture = structlog.get_logger("capture_logger")

        # Its output goes to the stdlib logger with the same name.
        # We configure that stdlib logger to write to our stream.
        stdlib_logger_for_capture = logging.getLogger("capture_logger")

        # Clear any pre-existing handlers for this specific logger
        # (e.g. if setup_logger was called with "capture_logger" before elsewhere)
        stdlib_logger_for_capture.handlers.clear()
        stdlib_logger_for_capture.propagate = False  # Avoid duplicate logs from root/parent

        stdlib_logger_for_capture.setLevel(logging.DEBUG)  # Ensure all messages pass to handler

        test_handler = logging.StreamHandler(log_stream)
        # No formatter on this handler, as structlog's JSONRenderer
        # should have already formatted the log event into a JSON string.
        stdlib_logger_for_capture.addHandler(test_handler)

        # Ensure structlog's output is not accidentally re-wrapped by a default config
        # by explicitly configuring it if reset_defaults is too broad.
        # The setup_logger call above should handle this.

        return structlog_logger_to_capture, log_stream  # Return the structlog logger

    def test_log_message_format_and_content(self, capturing_logger):
        logger, log_stream = capturing_logger

        test_message = "This is a test info message with some details."
        extra_data = {"key1": "value1", "custom_field": 123}
        logger.info(test_message, **extra_data)

        log_output = log_stream.getvalue()
        assert log_output  # Check if anything was logged

        try:
            log_json = json.loads(log_output)
        except json.JSONDecodeError:
            pytest.fail(f"Log output is not valid JSON: {log_output}")

        assert (
            log_json["event"] == test_message
        )  # structlog typically uses 'event' for the main message
        assert log_json["level"] == "info"  # structlog uses lowercase
        assert log_json["logger"] == "capture_logger"  # from structlog.stdlib.add_logger_name
        assert "timestamp" in log_json
        assert "pid" in log_json
        # Check for the extra data
        assert log_json["key1"] == "value1"
        assert log_json["custom_field"] == 123
        # Standard fields added by structlog or our processors
        assert "process_name" in log_json
        # Note: 'module', 'funcName', 'lineno' are added by stdlib logging,
        # structlog might not add them by default unless using specific processors like
        # structlog.stdlib.ProcessorFormatter.wrap_for_formatter(logging.LogRecord)
        # The current structlog config uses JSONRenderer directly.

    def test_different_log_levels(self, capturing_logger):
        logger, log_stream = capturing_logger  # Logger is at DEBUG level from fixture

        logger.debug("A debug message.")
        logger.info("An info message.")
        logger.warning("A warning message.")
        logger.error("An error message.")
        logger.critical("A critical message.")

        log_output = log_stream.getvalue().strip()
        log_lines = log_output.split("\n")

        assert len(log_lines) == 5  # All levels should be logged

        log_levels_captured = [json.loads(line)["level"] for line in log_lines]
        assert log_levels_captured == ["debug", "info", "warning", "error", "critical"]

    def test_log_level_filtering(self, monkeypatch):  # Removed clean_handlers
        monkeypatch.setenv("BTC_STACK_BUILDER_LOG_LEVEL", "WARNING")
        log_stream = StringIO()

        # Ensure structlog is configured by a main setup call
        setup_logger(name="temp_config_for_filter_test")

        # Get the logger using structlog's API
        import structlog  # Ensure structlog is imported if not already at top level

        logger = structlog.get_logger("filter_logger")

        # Configure the underlying stdlib logger for capturing
        stdlib_logger = logging.getLogger("filter_logger")
        stdlib_logger.handlers.clear()
        stdlib_logger.propagate = False
        stdlib_logger.setLevel(logging.WARNING)  # Set level based on monkeypatch

        test_handler = logging.StreamHandler(log_stream)
        stdlib_logger.addHandler(test_handler)

        logger.debug("This debug should not appear.")
        logger.info("This info should not appear.")
        logger.warning("This warning should appear.")
        logger.error("This error should appear.")

        log_output = log_stream.getvalue().strip()
        log_lines = log_output.split("\n")

        assert len(log_lines) == 2
        log_levels_captured = [json.loads(line)["level"] for line in log_lines]
        assert log_levels_captured == ["warning", "error"]

    def test_exception_logging(self, capturing_logger):
        logger, log_stream = capturing_logger

        try:
            raise ValueError("This is a test exception.")
        except ValueError:
            logger.error("Caught an exception", exc_info=True)  # exc_info=True is key for stdlib
            # For structlog, add_exception_info processor should handle it if an exception is active

        log_output = log_stream.getvalue()
        assert log_output
        log_json = json.loads(log_output)

        assert log_json["level"] == "error"
        assert "exception_type" in log_json
        assert log_json["exception_type"] == "ValueError"
        assert (
            "This is a test exception." in log_json["exception_message"]
        )  # if add_exception_info works
        # structlog's format_exc_info processor adds 'exception' field with traceback string
        assert "exception" in log_json
        assert "Traceback (most recent call last):" in log_json["exception"]
        assert "ValueError: This is a test exception." in log_json["exception"]


class TestTradeLogger:

    def test_trade_logger_writes_to_separate_file(self, monkeypatch, clean_handlers):
        monkeypatch.setenv("BTC_STACK_BUILDER_LOG_LEVEL", "INFO")

        # Ensure clean state for log files
        log_dir_path = Path(TEST_LOG_DIR)
        main_log_file = log_dir_path / MAIN_LOG_FILE
        trade_log_file = log_dir_path / TRADE_LOG_FILE
        if main_log_file.exists():
            main_log_file.unlink()
        if trade_log_file.exists():
            trade_log_file.unlink()

        # Setup main logger (this also configures structlog globally and sets up handlers for "main_for_trade_test")
        # It also sets up handlers for "btc_stack_builder.trades" via direct getLogger and addHandler.
        setup_logger(
            name="main_for_trade_test"
        )  # Call for its side-effects (handler setup, structlog.configure)

        # Get the logger instance using structlog's own API.
        import structlog  # ensure imported

        main_app_logger = structlog.get_logger(
            "main_for_trade_test"
        )  # Explicitly get structlog logger
        trade_logger = structlog.get_logger("btc_stack_builder.trades")

        trade_message = "This is a trade log message."
        main_logger_message = "This is a main logger message."

        # Log messages
        trade_logger.info(trade_message, source="trade_test", logger_type="trade")
        main_app_logger.info(main_logger_message, source="main_test", logger_type="main")

        # Close handlers to ensure logs are flushed
        # Be careful here: setup_logger creates handlers. If we close them all,
        # we might be closing handlers for other loggers too if not managed carefully.
        # The clean_handlers fixture should take care of this after the test.
        # For file reading, it's better to close them explicitly.

        # It's tricky to manage handlers when the function under test (setup_logger) also manages them globally.
        # A better approach for file testing might be to mock the file handlers' write methods.
        # For now, close handlers associated with these specific loggers.

        # Close handlers for "main_for_trade_test"
        for handler in list(logging.getLogger("main_for_trade_test").handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
        # Close handlers for "btc_stack_builder.trades"
        for handler in list(logging.getLogger("btc_stack_builder.trades").handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()

        assert trade_log_file.exists(), f"Trade log file {trade_log_file} was not created."

        trade_log_content = trade_log_file.read_text()
        # Parse the outer JSON for trade log
        outer_trade_log_json = json.loads(trade_log_content)
        assert "message" in outer_trade_log_json
        inner_trade_log_json_str = outer_trade_log_json["message"]
        inner_trade_log_json = json.loads(inner_trade_log_json_str)

        assert inner_trade_log_json["event"] == trade_message
        assert inner_trade_log_json.get("logger_type") == "trade"
        assert inner_trade_log_json.get("source") == "trade_test"
        assert main_logger_message not in trade_log_content  # Check raw content for absence

        assert main_log_file.exists(), f"Main log file {main_log_file} was not created."
        main_log_content = main_log_file.read_text()
        # Parse the outer JSON for main log
        outer_main_log_json = json.loads(main_log_content)
        assert "message" in outer_main_log_json
        inner_main_log_json_str = outer_main_log_json["message"]
        inner_main_log_json = json.loads(inner_main_log_json_str)

        assert inner_main_log_json["event"] == main_logger_message
        assert inner_main_log_json.get("logger_type") == "main"
        assert inner_main_log_json.get("source") == "main_test"
        assert trade_message not in main_log_content  # Check raw content for absence

        # Clean up created log files
        if main_log_file.exists():
            main_log_file.unlink(missing_ok=True)
        if trade_log_file.exists():
            trade_log_file.unlink(missing_ok=True)


# TODO: Add tests for decorators if time permits
# class TestLogDecorators:
# @log_execution_time(logger=test_logger_instance)
# def my_func(): ...
