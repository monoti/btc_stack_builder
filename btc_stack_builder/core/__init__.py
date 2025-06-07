"""
Core module for BTC Stack-Builder Bot.

This module exports the main models, constants, and utilities used throughout the project.
"""

__version__ = "0.1.0"

# Re-export common models and utilities
from .constants import (
    # Strategy constants
    BASIS_ENTRY_THRESHOLD,
    BASIS_HARVEST_ALLOCATION,
    BASIS_MAX_LEVERAGE,
    # Portfolio constants
    CORE_HODL_ALLOCATION,
    FUNDING_CAPTURE_ALLOCATION,
    FUNDING_ENTRY_THRESHOLD,
    FUNDING_MAX_LEVERAGE,
    FUNDING_PROFIT_TARGET,
    # Risk constants
    GLOBAL_STOP_LOSS_THRESHOLD,
    MARGIN_RATIO_CRITICAL_THRESHOLD,
    MARGIN_RATIO_WARNING_THRESHOLD,
    OPTION_DELTA_TARGET,
    OPTION_PREMIUM_ALLOCATION,
)

# Initialize logger
from .logger import setup_logger
from .models import (
    # Exchange models
    Exchange,
    ExchangeCredentials,
    MarginLevel,
    # Option models
    Option,
    OptionStatus,
    OptionType,
    OrderSide,
    OrderStatus,
    OrderType,
    # Portfolio models
    Portfolio,
    PortfolioAllocation,
    # Position models
    Position,
    PositionSide,
    PositionStatus,
    # Risk models
    RiskParameters,
    # Strategy models
    Strategy,
    StrategyConfig,
    StrategyState,
    SubPortfolio,
    # Trade models
    Trade,
    TradeStatus,
)
from .utils import (
    btc_to_satoshi,
    calculate_annualized_basis,
    calculate_funding_rate,
    calculate_margin_ratio,
    calculate_option_delta,
    calculate_position_pnl,
    datetime_to_timestamp,
    format_btc_amount,
    satoshi_to_btc,
    timestamp_to_datetime,
)

logger = setup_logger()

# Export all for easier imports
__all__ = [
    # Version
    "__version__",
    # Models
    "Exchange",
    "ExchangeCredentials",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "Portfolio",
    "SubPortfolio",
    "PortfolioAllocation",
    "Strategy",
    "StrategyState",
    "StrategyConfig",
    "Position",
    "PositionSide",
    "PositionStatus",
    "RiskParameters",
    "MarginLevel",
    "Trade",
    "TradeStatus",
    "Option",
    "OptionType",
    "OptionStatus",
    # Constants
    "CORE_HODL_ALLOCATION",
    "BASIS_HARVEST_ALLOCATION",
    "FUNDING_CAPTURE_ALLOCATION",
    "OPTION_PREMIUM_ALLOCATION",
    "GLOBAL_STOP_LOSS_THRESHOLD",
    "MARGIN_RATIO_WARNING_THRESHOLD",
    "MARGIN_RATIO_CRITICAL_THRESHOLD",
    "BASIS_ENTRY_THRESHOLD",
    "BASIS_MAX_LEVERAGE",
    "FUNDING_ENTRY_THRESHOLD",
    "FUNDING_MAX_LEVERAGE",
    "FUNDING_PROFIT_TARGET",
    "OPTION_DELTA_TARGET",
    # Utilities
    "calculate_annualized_basis",
    "calculate_funding_rate",
    "calculate_margin_ratio",
    "calculate_option_delta",
    "calculate_position_pnl",
    "format_btc_amount",
    "satoshi_to_btc",
    "btc_to_satoshi",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    # Logger
    "logger",
]
