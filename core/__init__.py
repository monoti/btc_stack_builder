"""
Core module for BTC Stack-Builder Bot.

This module exports the main models, constants, and utilities used throughout the project.
"""

__version__ = "0.1.0"

# Re-export common models and utilities
from .models import (
    # Exchange models
    Exchange,
    ExchangeCredentials,
    OrderType,
    OrderSide,
    OrderStatus,
    
    # Portfolio models
    Portfolio,
    SubPortfolio,
    PortfolioAllocation,
    
    # Strategy models
    Strategy,
    StrategyState,
    StrategyConfig,
    
    # Position models
    Position,
    PositionSide,
    PositionStatus,
    
    # Risk models
    RiskParameters,
    MarginLevel,
    
    # Trade models
    Trade,
    TradeStatus,
    
    # Option models
    Option,
    OptionType,
    OptionStatus,
)

from .constants import (
    # Portfolio constants
    CORE_HODL_ALLOCATION,
    BASIS_HARVEST_ALLOCATION,
    FUNDING_CAPTURE_ALLOCATION,
    OPTION_PREMIUM_ALLOCATION,
    
    # Risk constants
    GLOBAL_STOP_LOSS_THRESHOLD,
    MARGIN_RATIO_WARNING_THRESHOLD,
    MARGIN_RATIO_CRITICAL_THRESHOLD,
    
    # Strategy constants
    BASIS_ENTRY_THRESHOLD,
    BASIS_MAX_LEVERAGE,
    FUNDING_ENTRY_THRESHOLD,
    FUNDING_MAX_LEVERAGE,
    FUNDING_PROFIT_TARGET,
    OPTION_DELTA_TARGET,
)

from .utils import (
    calculate_annualized_basis,
    calculate_funding_rate,
    calculate_margin_ratio,
    calculate_option_delta,
    calculate_position_pnl,
    format_btc_amount,
    satoshi_to_btc,
    btc_to_satoshi,
    timestamp_to_datetime,
    datetime_to_timestamp,
)

# Initialize logger
from .logger import setup_logger
logger = setup_logger()

# Export all for easier imports
__all__ = [
    # Version
    "__version__",
    
    # Models
    "Exchange", "ExchangeCredentials", "OrderType", "OrderSide", "OrderStatus",
    "Portfolio", "SubPortfolio", "PortfolioAllocation",
    "Strategy", "StrategyState", "StrategyConfig",
    "Position", "PositionSide", "PositionStatus",
    "RiskParameters", "MarginLevel",
    "Trade", "TradeStatus",
    "Option", "OptionType", "OptionStatus",
    
    # Constants
    "CORE_HODL_ALLOCATION", "BASIS_HARVEST_ALLOCATION", 
    "FUNDING_CAPTURE_ALLOCATION", "OPTION_PREMIUM_ALLOCATION",
    "GLOBAL_STOP_LOSS_THRESHOLD", "MARGIN_RATIO_WARNING_THRESHOLD", 
    "MARGIN_RATIO_CRITICAL_THRESHOLD",
    "BASIS_ENTRY_THRESHOLD", "BASIS_MAX_LEVERAGE",
    "FUNDING_ENTRY_THRESHOLD", "FUNDING_MAX_LEVERAGE", "FUNDING_PROFIT_TARGET",
    "OPTION_DELTA_TARGET",
    
    # Utilities
    "calculate_annualized_basis", "calculate_funding_rate", "calculate_margin_ratio",
    "calculate_option_delta", "calculate_position_pnl",
    "format_btc_amount", "satoshi_to_btc", "btc_to_satoshi",
    "timestamp_to_datetime", "datetime_to_timestamp",
    
    # Logger
    "logger",
]
