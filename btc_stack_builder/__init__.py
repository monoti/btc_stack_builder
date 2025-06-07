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

# Import core components
from btc_stack_builder.core.logger import logger
from btc_stack_builder.core.constants import (
    CORE_HODL_ALLOCATION,
    BASIS_HARVEST_ALLOCATION,
    FUNDING_CAPTURE_ALLOCATION,
    OPTION_PREMIUM_ALLOCATION,
    MARGIN_RATIO_WARNING_THRESHOLD,
    MARGIN_RATIO_CRITICAL_THRESHOLD,
)

# Import configuration
from btc_stack_builder.config import config

# Import gateway base classes
from btc_stack_builder.gateways import (
    ExchangeGateway,
    GatewayError,
    ConnectionError,
    AuthenticationError,
    OrderError,
    RateLimitError,
)

# Import risk management
from btc_stack_builder.risk import MarginGuard

# Setup package-level logger
logger.info(f"BTC Stack-Builder Bot v{__version__} initialized")
