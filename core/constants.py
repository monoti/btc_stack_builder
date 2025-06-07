"""
Constants module for BTC Stack-Builder Bot.

This module defines all the configuration constants used throughout the application,
including portfolio allocations, risk thresholds, and strategy parameters.
"""
from decimal import Decimal
from typing import Final


# ======== Portfolio Allocation Constants ========
# Core HODL (60% of total assets)
CORE_HODL_ALLOCATION: Final[Decimal] = Decimal("0.60")
# Basis Harvest (25% of total assets)
BASIS_HARVEST_ALLOCATION: Final[Decimal] = Decimal("0.25")
# Funding Capture (10% of total assets)
FUNDING_CAPTURE_ALLOCATION: Final[Decimal] = Decimal("0.10")
# Option Premium (5% of total assets)
OPTION_PREMIUM_ALLOCATION: Final[Decimal] = Decimal("0.05")

# Validate total allocation equals 100%
assert sum([
    CORE_HODL_ALLOCATION,
    BASIS_HARVEST_ALLOCATION,
    FUNDING_CAPTURE_ALLOCATION,
    OPTION_PREMIUM_ALLOCATION
]) == Decimal("1.00"), "Portfolio allocations must sum to 1.00"


# ======== Risk Management Constants ========
# Global stop-loss threshold (-70% price movement from entry)
GLOBAL_STOP_LOSS_THRESHOLD: Final[Decimal] = Decimal("-0.70")

# Margin ratio thresholds
MARGIN_RATIO_WARNING_THRESHOLD: Final[Decimal] = Decimal("4.50")  # 450%
MARGIN_RATIO_CRITICAL_THRESHOLD: Final[Decimal] = Decimal("4.00")  # 400%

# Margin check interval (in seconds)
MARGIN_CHECK_INTERVAL: Final[int] = 300  # 5 minutes


# ======== Basis Harvest Strategy Constants ========
# Minimum annualized basis percentage to enter a position
BASIS_ENTRY_THRESHOLD: Final[Decimal] = Decimal("0.05")  # 5%

# Maximum leverage for basis harvest strategy
BASIS_MAX_LEVERAGE: Final[Decimal] = Decimal("1.5")

# Days before expiry to start rolling position
BASIS_ROLL_START_DAYS: Final[int] = 21
BASIS_ROLL_END_DAYS: Final[int] = 14


# ======== Funding Capture Strategy Constants ========
# Funding rate threshold to enter a position (per 8-hour period)
FUNDING_ENTRY_THRESHOLD: Final[Decimal] = Decimal("-0.0001")  # -0.01%

# Maximum leverage for funding capture strategy
FUNDING_MAX_LEVERAGE: Final[Decimal] = Decimal("2.0")

# Profit target for exiting funding capture positions
FUNDING_PROFIT_TARGET: Final[Decimal] = Decimal("0.12")  # 12%


# ======== Option Premium Strategy Constants ========
# Target delta for put options (approx. 20% probability of being ITM)
OPTION_DELTA_TARGET: Final[Decimal] = Decimal("0.20")

# Option expiry range (in days)
OPTION_MIN_EXPIRY_DAYS: Final[int] = 60
OPTION_MAX_EXPIRY_DAYS: Final[int] = 90


# ======== Exchange Constants ========
# Exchange identifiers
EXCHANGE_BINANCE: Final[str] = "binance"
EXCHANGE_DERIBIT: Final[str] = "deribit"

# Instrument types
INSTRUMENT_SPOT: Final[str] = "spot"
INSTRUMENT_FUTURES_QUARTERLY: Final[str] = "futures_quarterly"
INSTRUMENT_FUTURES_PERPETUAL: Final[str] = "futures_perpetual"
INSTRUMENT_OPTIONS: Final[str] = "options"

# Specific instrument symbols
SYMBOL_BTC_USD_SPOT: Final[str] = "BTC/USD"
SYMBOL_BTC_USD_PERP: Final[str] = "BTCUSD_PERP"
# Quarterly futures symbol pattern (to be formatted with expiry date)
SYMBOL_BTC_USD_QUARTERLY_PATTERN: Final[str] = "BTCUSD_{expiry}"


# ======== Monitoring Constants ========
# Prometheus metrics update interval (in seconds)
METRICS_UPDATE_INTERVAL: Final[int] = 60  # 1 minute

# Alert levels
ALERT_INFO: Final[str] = "INFO"
ALERT_WARNING: Final[str] = "WARNING"
ALERT_CRITICAL: Final[str] = "CRITICAL"


# ======== Database Constants ========
# Database tables
TABLE_POSITIONS: Final[str] = "positions"
TABLE_TRADES: Final[str] = "trades"
TABLE_PORTFOLIO: Final[str] = "portfolio"
TABLE_OPTIONS: Final[str] = "options"

# Database connection retry settings
DB_MAX_RETRIES: Final[int] = 5
DB_RETRY_DELAY: Final[int] = 5  # seconds


# ======== Scheduler Constants ========
# Strategy execution schedules
BASIS_HARVEST_SCHEDULE: Final[str] = "0 0 * * *"  # Daily at midnight
FUNDING_CAPTURE_SCHEDULE: Final[str] = "*/10 * * * *"  # Every 10 minutes
OPTION_PREMIUM_SCHEDULE: Final[str] = "0 0 1 * *"  # Monthly on the 1st

# Profit harvesting schedules
BASIS_PROFIT_HARVEST_SCHEDULE: Final[str] = "0 0 * * 1"  # Weekly on Monday
FUNDING_PROFIT_HARVEST_SCHEDULE: Final[str] = "0 0 1 * *"  # Monthly on the 1st
OPTION_PROFIT_HARVEST_SCHEDULE: Final[str] = "0 0 1 * *"  # Monthly on the 1st


# ======== Target Performance Constants ========
# Target annual yield (in BTC terms)
TARGET_ANNUAL_YIELD_MIN: Final[Decimal] = Decimal("0.06")  # 6%
TARGET_ANNUAL_YIELD_MAX: Final[Decimal] = Decimal("0.10")  # 10%
