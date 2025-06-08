"""
Core data models for BTC Stack-Builder Bot.

This module defines all the Pydantic models used throughout the application,
including exchange models, portfolio models, strategy models, position models,
risk models, trade models, and option models.
"""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

# ======== Exchange Models ========


class OrderType(str, Enum):
    """Order types supported by exchanges."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(str, Enum):
    """Order sides (buy/sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Status of an order."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(BaseModel):
    """Model representing an exchange order."""

    id: UUID = Field(default_factory=uuid4)
    exchange_id: str = Field(description="Exchange-specific order ID")
    exchange: str = Field(description="Exchange name (e.g., 'binance', 'deribit')")
    symbol: str = Field(description="Trading pair or contract symbol")
    order_type: OrderType
    side: OrderSide
    price: Decimal | None = Field(None, description="Limit price (None for market orders)")
    amount: Decimal = Field(description="Order quantity in base currency or contracts")
    filled_amount: Decimal = Field(default=Decimal("0"), description="Filled quantity")
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    strategy_id: UUID | None = Field(None, description="Associated strategy ID")
    position_id: UUID | None = Field(None, description="Associated position ID")

    @model_validator(mode="after")
    def validate_price_for_limit_orders(self) -> "Order":
        """Ensure limit orders have a price."""
        limit_types = [
            OrderType.LIMIT,
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_LIMIT,
        ]
        if self.order_type in limit_types and self.price is None:
            raise ValueError(f"Price must be specified for {self.order_type} orders")
        return self


class ExchangeType(str, Enum):
    """Types of exchanges."""

    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"


class ExchangeCredentials(BaseModel):
    """API credentials for exchange access."""

    api_key: str
    api_secret: str
    passphrase: str | None = None  # Some exchanges require a passphrase
    is_testnet: bool = False

    class Config:
        frozen = True  # Immutable after creation


class Exchange(BaseModel):
    """Exchange configuration and metadata."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Exchange name (e.g., 'binance', 'deribit')")
    type: ExchangeType
    credentials: ExchangeCredentials
    base_url: str | None = None
    enabled: bool = True
    features: dict[str, bool] = Field(
        default_factory=lambda: {
            "margin_trading": False,
            "futures_trading": False,
            "options_trading": False,
            "cross_margin": False,
            "isolated_margin": False,
        }
    )
    rate_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "requests_per_second": 10,
            "orders_per_second": 5,
        }
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ======== Portfolio Models ========


class SubPortfolioType(str, Enum):
    """Types of sub-portfolios."""

    CORE_HODL = "core_hodl"
    BASIS_HARVEST = "basis_harvest"
    FUNDING_CAPTURE = "funding_capture"
    OPTION_PREMIUM = "option_premium"


class PortfolioAllocation(BaseModel):
    """Allocation percentages for sub-portfolios."""

    core_hodl: Decimal = Field(ge=0, le=1)
    basis_harvest: Decimal = Field(ge=0, le=1)
    funding_capture: Decimal = Field(ge=0, le=1)
    option_premium: Decimal = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_total_allocation(self) -> "PortfolioAllocation":
        """Ensure allocations sum to 1.0 (100%)."""
        total = self.core_hodl + self.basis_harvest + self.funding_capture + self.option_premium
        if total != Decimal("1.0"):
            raise ValueError(f"Portfolio allocations must sum to 1.0, got {total}")
        return self


class SubPortfolio(BaseModel):
    """Model representing a sub-portfolio."""

    id: UUID = Field(default_factory=uuid4)
    type: SubPortfolioType
    allocation_percentage: Decimal = Field(ge=0, le=1)
    current_balance_btc: Decimal = Field(ge=0)
    current_balance_usd: Decimal = Field(ge=0)
    target_balance_btc: Decimal = Field(ge=0)
    target_balance_usd: Decimal = Field(ge=0)
    rebalance_threshold: Decimal = Field(default=Decimal("0.05"))  # 5% deviation triggers rebalance
    last_rebalanced: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def balance_deviation(self) -> Decimal:
        """Calculate deviation from target balance."""
        if self.target_balance_btc == 0:
            return Decimal("0")
        return (self.current_balance_btc - self.target_balance_btc) / self.target_balance_btc

    @property
    def needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing."""
        return abs(self.balance_deviation) > self.rebalance_threshold


class Portfolio(BaseModel):
    """Model representing the complete portfolio."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    total_balance_btc: Decimal = Field(ge=0)
    total_balance_usd: Decimal = Field(ge=0)
    btc_price_usd: Decimal = Field(ge=0)
    allocation: PortfolioAllocation
    sub_portfolios: dict[SubPortfolioType, SubPortfolio]
    cold_wallet_address: str = Field(description="BTC address for core HODL storage")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_sub_portfolios(self) -> "Portfolio":
        """Ensure all required sub-portfolios exist."""
        required_types = set(SubPortfolioType)
        actual_types = set(self.sub_portfolios.keys())

        if required_types != actual_types:
            missing = required_types - actual_types
            extra = actual_types - required_types
            error_msg = []
            if missing:
                missing_str = f"Missing sub-portfolios: {', '.join(t.value for t in missing)}"
                error_msg.append(missing_str)
            if extra:
                extra_str = f"Extra sub-portfolios: {', '.join(t.value for t in extra)}"
                error_msg.append(extra_str)
            raise ValueError(". ".join(error_msg))

        return self


# ======== Strategy Models ========


class StrategyType(str, Enum):
    """Types of trading strategies."""

    BASIS_HARVEST = "basis_harvest"
    FUNDING_CAPTURE = "funding_capture"
    OPTION_PREMIUM = "option_premium"


class StrategyStatus(str, Enum):
    """Status of a strategy."""

    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class StrategyConfig(BaseModel):
    """Base configuration for all strategies."""

    enabled: bool = True
    max_positions: int = Field(default=1, ge=1)
    max_capital_allocation: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)

    # Strategy-specific parameters will be added by subclasses


class BasisHarvestConfig(StrategyConfig):
    """Configuration for basis harvest strategy."""

    entry_threshold: Decimal = Field(
        default=Decimal("0.05"), description="Minimum annualized basis %"
    )
    max_leverage: Decimal = Field(default=Decimal("1.5"), ge=1, le=10)
    roll_start_days: int = Field(default=21, description="Days before expiry to start rolling")
    roll_end_days: int = Field(default=14, description="Days before expiry to finish rolling")


class FundingCaptureConfig(StrategyConfig):
    """Configuration for funding capture strategy."""

    entry_threshold: Decimal = Field(
        default=Decimal("-0.0001"), description="Funding rate threshold"
    )
    max_leverage: Decimal = Field(default=Decimal("2.0"), ge=1, le=10)
    profit_target: Decimal = Field(default=Decimal("0.12"), description="Profit target for exit")


class OptionPremiumConfig(StrategyConfig):
    """Configuration for option premium strategy."""

    delta_target: Decimal = Field(
        default=Decimal("0.20"), description="Target delta for put options"
    )
    min_expiry_days: int = Field(default=60)
    max_expiry_days: int = Field(default=90)


class StrategyState(BaseModel):
    """Current state of a strategy."""

    last_run: datetime | None = None
    next_run: datetime | None = None
    active_positions: int = 0
    total_profit_btc: Decimal = Field(default=Decimal("0"))
    current_allocation_btc: Decimal = Field(default=Decimal("0"))
    errors: list[str] = Field(default_factory=list)
    custom_state: dict = Field(
        default_factory=dict, description="Strategy-specific state variables"
    )


class Strategy(BaseModel):
    """Model representing a trading strategy."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    type: StrategyType
    status: StrategyStatus = Field(default=StrategyStatus.ACTIVE)
    sub_portfolio_type: SubPortfolioType
    exchange: str
    config: BasisHarvestConfig | FundingCaptureConfig | OptionPremiumConfig
    state: StrategyState = Field(default_factory=StrategyState)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_config_type(self) -> "Strategy":
        """Ensure config type matches strategy type."""
        expected_config_type = {
            StrategyType.BASIS_HARVEST: BasisHarvestConfig,
            StrategyType.FUNDING_CAPTURE: FundingCaptureConfig,
            StrategyType.OPTION_PREMIUM: OptionPremiumConfig,
        }.get(self.type)

        if not isinstance(self.config, expected_config_type):
            err_msg = (
                f"Strategy type {self.type} requires config of type "
                f"{expected_config_type.__name__}"
            )
            raise ValueError(err_msg)

        return self


# ======== Position Models ========


class PositionSide(str, Enum):
    """Position sides."""

    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Status of a position."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class Position(BaseModel):
    """Model representing a trading position."""

    id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID
    exchange: str
    symbol: str
    side: PositionSide
    entry_price: Decimal
    current_price: Decimal
    size: Decimal = Field(description="Position size in BTC or contracts")
    leverage: Decimal = Field(default=Decimal("1.0"))
    liquidation_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    unrealized_pnl: Decimal = Field(default=Decimal("0"))
    realized_pnl: Decimal = Field(default=Decimal("0"))
    fees_paid: Decimal = Field(default=Decimal("0"))
    funding_paid: Decimal = Field(default=Decimal("0"))
    funding_received: Decimal = Field(default=Decimal("0"))
    status: PositionStatus = Field(default=PositionStatus.OPEN)
    open_orders: list[UUID] = Field(
        default_factory=list, description="IDs of associated open orders"
    )
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    last_update_time: datetime = Field(default_factory=datetime.utcnow)
    close_time: datetime | None = None
    metadata: dict = Field(default_factory=dict, description="Position-specific metadata")

    @property
    def duration(self) -> float | None:
        """Calculate position duration in days."""
        if self.status == PositionStatus.CLOSED and self.close_time:
            return (self.close_time - self.entry_time).total_seconds() / 86400
        elif self.status == PositionStatus.OPEN:
            return (datetime.utcnow() - self.entry_time).total_seconds() / 86400
        return None

    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate PnL as percentage of position value."""
        if self.entry_price == 0:
            return Decimal("0")

        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.current_price) / self.entry_price


# ======== Risk Models ========


class MarginLevel(str, Enum):
    """Margin level status."""

    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    LIQUIDATION = "liquidation"


class RiskParameters(BaseModel):
    """Risk management parameters."""

    global_stop_loss_threshold: Decimal = Field(default=Decimal("-0.70"))
    margin_ratio_warning_threshold: Decimal = Field(default=Decimal("4.50"))
    margin_ratio_critical_threshold: Decimal = Field(default=Decimal("4.00"))
    max_position_size_btc: Decimal = Field(default=Decimal("10.0"))
    max_position_size_percentage: Decimal = Field(default=Decimal("0.25"))
    max_leverage: Decimal = Field(default=Decimal("3.0"))
    max_drawdown: Decimal = Field(default=Decimal("0.20"))
    correlation_threshold: Decimal = Field(default=Decimal("0.7"))


class MarginStatus(BaseModel):
    """Current margin status for an account."""

    exchange: str
    account_type: Literal["spot", "futures", "options"]
    wallet_balance: Decimal
    unrealized_pnl: Decimal
    maintenance_margin: Decimal
    initial_margin: Decimal
    margin_ratio: Decimal
    margin_level: MarginLevel
    margin_call_price: Decimal | None = None
    liquidation_price: Decimal | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def available_balance(self) -> Decimal:
        """Calculate available balance."""
        return self.wallet_balance + self.unrealized_pnl - self.initial_margin


# ======== Trade Models ========


class TradeType(str, Enum):
    """Types of trades."""

    SPOT = "spot"
    FUTURES = "futures"
    OPTION = "option"
    FUNDING = "funding"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


class TradeStatus(str, Enum):
    """Status of a trade."""

    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELED = "canceled"


class Trade(BaseModel):
    """Model representing a trade or transaction."""

    id: UUID = Field(default_factory=uuid4)
    exchange: str
    trade_type: TradeType
    symbol: str
    side: OrderSide
    price: Decimal
    amount: Decimal
    cost: Decimal = Field(description="Total cost in quote currency")
    fee: Decimal = Field(default=Decimal("0"))
    fee_currency: str
    order_id: UUID | None = None
    position_id: UUID | None = None
    strategy_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: TradeStatus = Field(default=TradeStatus.PENDING)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def calculate_cost(self) -> "Trade":
        """Calculate cost if not provided."""
        if self.cost == 0 and self.price > 0 and self.amount > 0:
            self.cost = self.price * self.amount
        return self


# ======== Option Models ========


class OptionType(str, Enum):
    """Option types."""

    CALL = "call"
    PUT = "put"


class OptionStatus(str, Enum):
    """Status of an option position."""

    OPEN = "open"
    EXERCISED = "exercised"
    EXPIRED_OTM = "expired_otm"  # Expired out-of-the-money
    EXPIRED_ITM = "expired_itm"  # Expired in-the-money
    CLOSED = "closed"  # Manually closed before expiry


class Option(BaseModel):
    """Model representing an option position."""

    id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID
    exchange: str
    underlying: str = Field(description="Underlying asset symbol")
    strike_price: Decimal
    expiry_date: datetime
    option_type: OptionType
    side: Literal["buy", "sell"] = Field(description="Whether option was bought or sold")
    size: Decimal = Field(description="Number of contracts")
    premium: Decimal = Field(description="Option premium per contract")
    total_premium: Decimal = Field(description="Total premium received/paid")
    collateral: Decimal = Field(
        default=Decimal("0"), description="Collateral locked for selling options"
    )
    status: OptionStatus = Field(default=OptionStatus.OPEN)
    delta: Decimal = Field(default=Decimal("0"))
    gamma: Decimal = Field(default=Decimal("0"))
    theta: Decimal = Field(default=Decimal("0"))
    vega: Decimal = Field(default=Decimal("0"))
    implied_volatility: Decimal = Field(default=Decimal("0"))
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    exit_time: datetime | None = None
    pnl: Decimal = Field(default=Decimal("0"))

    @property
    def days_to_expiry(self) -> float:
        """Calculate days remaining until expiry."""
        # Use the same timezone as expiry_date, or assume UTC if naive
        tz = self.expiry_date.tzinfo or UTC
        now = datetime.now(tz)
        if now > self.expiry_date:
            return 0.0
        return (self.expiry_date - now).total_seconds() / 86400

    @property
    def is_itm(self) -> bool:
        """Check if option is currently in-the-money."""
        # This is a simplified check - in reality would need current price of underlying
        # For now, we'll rely on the delta as a proxy
        if self.option_type == OptionType.CALL:
            return self.delta > Decimal("0.5")
        else:  # PUT
            return self.delta < Decimal("-0.5")
