"""
BTC Stack-Builder Bot - Autonomous Bitcoin Accumulation System

This package provides a complete system for automated Bitcoin accumulation through
multiple strategies: basis harvest, funding capture, and options premium collection.

The bot's prime directive is to maximize the quantity of Bitcoin (measured in satoshis)
under its management over the long term, achieving a yield superior to a simple
buy-and-hold (HODL) strategy while operating under a strict, low-risk framework.
"""

__version__ = "0.1.0"
__author__ = "BTC Stack Builder Team"
__license__ = "MIT"


from btc_stack_builder.core.logger import default_app_logger as logger

# Setup package-level logger
logger.info(f"BTC Stack-Builder Bot v{__version__} initialized")
