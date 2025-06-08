"""
Trading strategies module for BTC Stack-Builder Bot.

This module provides implementations of the three core trading strategies:
1. Basis Harvest - Capturing the contango premium in quarterly futures contracts (25%)
2. Funding Capture - Collecting negative funding payments from perpetual futures (10%)
3. Option Premium - Selling out-of-the-money put options for premium collection (5%)

Each strategy is designed to maximize BTC accumulation while adhering to strict
risk parameters and operating within its allocated capital portion.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from btc_stack_builder.core.constants import (
    BASIS_ENTRY_THRESHOLD,
    BASIS_MAX_LEVERAGE,
    BASIS_ROLL_END_DAYS,
    BASIS_ROLL_START_DAYS,
    FUNDING_ENTRY_THRESHOLD,
    FUNDING_MAX_LEVERAGE,
    FUNDING_PROFIT_TARGET,
    OPTION_DELTA_TARGET,
    OPTION_MAX_EXPIRY_DAYS,
    OPTION_MIN_EXPIRY_DAYS,
)
from btc_stack_builder.core.logger import log_strategy_execution, logger
from btc_stack_builder.core.models import (
    BasisHarvestConfig,
    FundingCaptureConfig,
    Option,
    OptionPremiumConfig,
    OptionStatus,
    OptionType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    Strategy,
    StrategyConfig,
    StrategyType,
    SubPortfolio,
    SubPortfolioType,
)
from btc_stack_builder.core.utils import (
    calculate_annualized_basis,
    calculate_funding_rate,
    calculate_option_delta,
    calculate_position_pnl,
    get_next_quarterly_expiry,
    parse_quarterly_futures_symbol,
)
from btc_stack_builder.gateways import ExchangeGateway


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(
        self, gateway: ExchangeGateway, sub_portfolio: SubPortfolio, config: StrategyConfig
    ):
        """
        Initialize the strategy.

        Args:
            gateway: Exchange gateway for executing trades
            sub_portfolio: Sub-portfolio allocated to this strategy
            config: Strategy-specific configuration
        """
        self.gateway = gateway
        self.sub_portfolio = sub_portfolio
        self.config = config
        self.positions: list[Position] = []
        self.orders: list[Order] = []
        self.initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the strategy, load current positions and state.

        This method should be called before using any other methods.
        """
        pass

    @abstractmethod
    async def execute(self) -> dict[str, Any]:
        """
        Execute the strategy logic.

        Returns:
            Dictionary with execution results
        """
        pass

    @abstractmethod
    async def harvest_profits(self) -> Decimal:
        """
        Harvest profits from the strategy and transfer to Core HODL.

        Returns:
            Amount of BTC harvested
        """
        pass

    @abstractmethod
    async def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            True if all positions were closed successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the strategy.

        Returns:
            Dictionary with strategy status information
        """
        pass


class BasisHarvestStrategy(BaseStrategy):
    """
    Basis Harvest Strategy implementation.

    This strategy captures the contango premium in quarterly futures contracts
    by going long on the futures when the annualized basis exceeds the threshold.
    """

    def __init__(
        self,
        gateway: ExchangeGateway,
        sub_portfolio: SubPortfolio,
        config: BasisHarvestConfig | dict[str, Any],
    ):
        """
        Initialize the Basis Harvest strategy.

        Args:
            gateway: Exchange gateway for executing trades
            sub_portfolio: Sub-portfolio allocated to this strategy
            config: Strategy-specific configuration
        """
        if not isinstance(config, BasisHarvestConfig):
            config = BasisHarvestConfig(**config)

        super().__init__(gateway, sub_portfolio, config)
        self.config: BasisHarvestConfig = config
        self.current_quarterly_symbol: str | None = None
        self.next_quarterly_symbol: str | None = None

    async def initialize(self) -> None:
        """
        Initialize the strategy, identify current and next quarterly futures.
        """
        logger.info("Initializing Basis Harvest strategy")

        # Find current and next quarterly futures symbols
        # This is a placeholder - actual implementation would scan available futures
        self.current_quarterly_symbol = "BTCUSD_250628"  # Example: June 2025 quarterly
        self.next_quarterly_symbol = "BTCUSD_250926"  # Example: September 2025 quarterly

        # Load existing positions
        positions = await self.gateway.get_positions()
        self.positions = [
            pos
            for pos in positions
            if pos.symbol in (self.current_quarterly_symbol, self.next_quarterly_symbol)
        ]

        # Set margin type to CROSSED for capital efficiency
        if self.current_quarterly_symbol:
            await self.gateway.set_margin_type(self.current_quarterly_symbol, "CROSSED")

        self.initialized = True
        logger.info(
            "Basis Harvest strategy initialized",
            current_quarterly=self.current_quarterly_symbol,
            next_quarterly=self.next_quarterly_symbol,
            positions=len(self.positions),
        )

    @log_strategy_execution("basis_harvest")
    async def execute(self) -> dict[str, Any]:
        """
        Execute the Basis Harvest strategy logic.

        1. Calculate the annualized basis for the current quarterly futures
        2. If basis > threshold, enter or maintain position
        3. If close to expiry, roll to next quarterly futures

        Returns:
            Dictionary with execution results
        """
        if not self.initialized:
            await self.initialize()

        results = {
            "action": "none",
            "basis_percentage": None,
            "position_opened": False,
            "position_closed": False,
            "roll_executed": False,
            "error": None,
        }

        try:
            # Check if we need to roll positions
            if self.current_quarterly_symbol:
                expiry_date = parse_quarterly_futures_symbol(self.current_quarterly_symbol)
                if expiry_date:
                    days_to_expiry = (expiry_date - datetime.now()).days

                    if (
                        days_to_expiry <= BASIS_ROLL_START_DAYS
                        and days_to_expiry > BASIS_ROLL_END_DAYS
                    ):
                        # Time to start rolling to next quarterly
                        await self._roll_position()
                        results["action"] = "roll"
                        results["roll_executed"] = True
                        return results

            # Calculate current basis
            if not self.current_quarterly_symbol:
                results["error"] = "No quarterly futures symbol available"
                return results

            basis = await self.gateway.get_futures_basis("BTC/USD", self.current_quarterly_symbol)
            results["basis_percentage"] = float(basis) * 100  # Convert to percentage

            # Check if basis exceeds threshold
            if basis >= self.config.entry_threshold:
                # Basis is favorable, enter or maintain position
                if not self.positions:
                    # No position yet, enter new position
                    position_size = self._calculate_position_size()

                    # Set leverage
                    await self.gateway.set_leverage(
                        self.current_quarterly_symbol, int(self.config.max_leverage)
                    )

                    # Create order
                    order = await self.gateway.create_order(
                        symbol=self.current_quarterly_symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        amount=position_size,
                    )

                    self.orders.append(order)
                    results["action"] = "enter"
                    results["position_opened"] = True

                    logger.info(
                        "Opened Basis Harvest position",
                        symbol=self.current_quarterly_symbol,
                        size=str(position_size),
                        basis_percentage=f"{float(basis) * 100:.2f}%",
                    )
                else:
                    # Already have position, maintain it
                    results["action"] = "hold"

                    logger.info(
                        "Maintaining Basis Harvest position",
                        symbol=self.current_quarterly_symbol,
                        basis_percentage=f"{float(basis) * 100:.2f}%",
                    )
            else:
                # Basis below threshold, consider exiting
                if self.positions:
                    # Close position if basis is significantly below threshold
                    if basis < (self.config.entry_threshold * Decimal("0.5")):
                        await self.close_all_positions()
                        results["action"] = "exit"
                        results["position_closed"] = True

                        logger.info(
                            "Closed Basis Harvest position due to low basis",
                            basis_percentage=f"{float(basis) * 100:.2f}%",
                            threshold=f"{float(self.config.entry_threshold) * 100:.2f}%",
                        )
                    else:
                        # Basis below threshold but not significantly, hold position
                        results["action"] = "hold"
                else:
                    # No position and basis below threshold, do nothing
                    results["action"] = "wait"

            return results

        except Exception as e:
            logger.error("Error executing Basis Harvest strategy", exc_info=True)
            results["error"] = str(e)
            return results

    async def harvest_profits(self) -> Decimal:
        """
        Harvest profits from the strategy and transfer to Core HODL.

        This method calculates realized profits and initiates a withdrawal
        to the Core HODL cold-wallet address.

        Returns:
            Amount of BTC harvested
        """
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Calculate realized PnL from closed positions
        # 2. Initiate withdrawal to cold wallet if above threshold
        harvested_amount = Decimal("0")

        logger.info("Harvesting Basis profits", amount=str(harvested_amount))

        return harvested_amount

    async def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            True if all positions were closed successfully, False otherwise
        """
        if not self.positions:
            return True

        success = True
        for position in self.positions:
            try:
                order = await self.gateway.create_order(
                    symbol=position.symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    amount=position.size,
                )

                self.orders.append(order)
                logger.info(
                    f"Closed position {position.id}",
                    symbol=position.symbol,
                    size=str(position.size),
                )
            except Exception as e:
                logger.error(
                    f"Error closing position {position.id}",
                    symbol=position.symbol,
                    error=str(e),
                    exc_info=True,
                )
                success = False

        # Refresh positions list
        if success:
            self.positions = []

        return success

    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the strategy.

        Returns:
            Dictionary with strategy status information
        """
        return {
            "type": "basis_harvest",
            "initialized": self.initialized,
            "current_quarterly": self.current_quarterly_symbol,
            "next_quarterly": self.next_quarterly_symbol,
            "positions": len(self.positions),
            "orders": len(self.orders),
            "config": {
                "entry_threshold": float(self.config.entry_threshold),
                "max_leverage": float(self.config.max_leverage),
                "roll_start_days": self.config.roll_start_days,
                "roll_end_days": self.config.roll_end_days,
            },
        }

    async def _roll_position(self) -> bool:
        """
        Roll position from current quarterly to next quarterly futures.

        Returns:
            True if roll was successful, False otherwise
        """
        if not self.current_quarterly_symbol or not self.next_quarterly_symbol:
            logger.error("Cannot roll position: quarterly symbols not defined")
            return False

        # Check if we have an open position to roll
        current_positions = [p for p in self.positions if p.symbol == self.current_quarterly_symbol]
        if not current_positions:
            logger.info("No positions to roll")
            return True  # Nothing to do

        try:
            # Close current position
            for position in current_positions:
                await self.gateway.create_order(
                    symbol=position.symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    amount=position.size,
                )

            # Calculate new position size for next quarterly
            position_size = self._calculate_position_size()

            # Set leverage for next quarterly
            await self.gateway.set_leverage(
                self.next_quarterly_symbol, int(self.config.max_leverage)
            )

            # Set margin type for next quarterly
            await self.gateway.set_margin_type(self.next_quarterly_symbol, "CROSSED")

            # Open position in next quarterly
            await self.gateway.create_order(
                symbol=self.next_quarterly_symbol,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                amount=position_size,
            )

            # Update quarterly symbols
            self.current_quarterly_symbol = self.next_quarterly_symbol

            # Find new next quarterly (placeholder logic)
            current_expiry = parse_quarterly_futures_symbol(self.current_quarterly_symbol)
            if current_expiry:
                next_expiry = get_next_quarterly_expiry(current_expiry)
                # Format next quarterly symbol (example format)
                year_str = str(next_expiry.year)[2:]
                month_str = str(next_expiry.month).zfill(2)
                day_str = str(next_expiry.day).zfill(2)
                self.next_quarterly_symbol = f"BTCUSD_{year_str}{month_str}{day_str}"

            logger.info(
                "Successfully rolled position",
                from_symbol=self.current_quarterly_symbol,
                to_symbol=self.next_quarterly_symbol,
                size=str(position_size),
            )

            return True

        except Exception as e:
            logger.error(
                "Error rolling position",
                from_symbol=self.current_quarterly_symbol,
                to_symbol=self.next_quarterly_symbol,
                error=str(e),
                exc_info=True,
            )
            return False

    def _calculate_position_size(self) -> Decimal:
        """
        Calculate appropriate position size based on available capital and leverage.

        Returns:
            Position size in BTC
        """
        # Placeholder implementation
        # In a real implementation, this would consider:
        # 1. Available capital in the sub-portfolio
        # 2. Current BTC price
        # 3. Leverage setting
        # 4. Risk parameters

        # Simple example: Use 90% of available balance with configured leverage
        available_btc = self.sub_portfolio.current_balance_btc * Decimal("0.9")
        position_size = available_btc * self.config.max_leverage

        # Round to 3 decimal places (0.001 BTC precision)
        return position_size.quantize(Decimal("0.001"))


class FundingCaptureStrategy(BaseStrategy):
    """
    Funding Capture Strategy implementation.

    This strategy opportunistically captures negative funding rates in perpetual
    futures by going long when funding is sufficiently negative.
    """

    def __init__(
        self,
        gateway: ExchangeGateway,
        sub_portfolio: SubPortfolio,
        config: FundingCaptureConfig | dict[str, Any],
    ):
        """
        Initialize the Funding Capture strategy.

        Args:
            gateway: Exchange gateway for executing trades
            sub_portfolio: Sub-portfolio allocated to this strategy
            config: Strategy-specific configuration
        """
        if not isinstance(config, FundingCaptureConfig):
            config = FundingCaptureConfig(**config)

        super().__init__(gateway, sub_portfolio, config)
        self.config: FundingCaptureConfig = config
        self.perpetual_symbol = "BTCUSD_PERP"
        self.funding_history = []

    async def initialize(self) -> None:
        """
        Initialize the strategy, load current positions and funding history.
        """
        logger.info("Initializing Funding Capture strategy")

        # Load existing positions
        positions = await self.gateway.get_positions()
        self.positions = [pos for pos in positions if pos.symbol == self.perpetual_symbol]

        # Set margin type to CROSSED for capital efficiency
        await self.gateway.set_margin_type(self.perpetual_symbol, "CROSSED")

        # Get recent funding history
        try:
            self.funding_history = await self.gateway.get_funding_history(
                self.perpetual_symbol, limit=20
            )
        except Exception as e:
            logger.warning("Could not retrieve funding history", error=str(e))

        self.initialized = True
        logger.info(
            "Funding Capture strategy initialized",
            symbol=self.perpetual_symbol,
            positions=len(self.positions),
        )

    @log_strategy_execution("funding_capture")
    async def execute(self) -> dict[str, Any]:
        """
        Execute the Funding Capture strategy logic.

        1. Check current funding rate
        2. If funding rate <= threshold, enter position
        3. Exit when funding becomes non-negative or profit target is reached

        Returns:
            Dictionary with execution results
        """
        if not self.initialized:
            await self.initialize()

        results = {
            "action": "none",
            "funding_rate": None,
            "position_opened": False,
            "position_closed": False,
            "reason": None,
            "error": None,
        }

        try:
            # Get current funding rate
            funding_info = await self.gateway.get_funding_rate(self.perpetual_symbol)
            funding_rate = funding_info["funding_rate"]
            results["funding_rate"] = float(funding_rate)

            # Check if we have an open position
            has_position = len(self.positions) > 0

            if has_position:
                # We have a position, check exit conditions

                # Exit condition 1: Funding rate becomes non-negative
                if funding_rate >= Decimal("0"):
                    await self.close_all_positions()
                    results["action"] = "exit"
                    results["position_closed"] = True
                    results["reason"] = "funding_positive"

                    logger.info(
                        "Closed Funding Capture position due to non-negative funding",
                        funding_rate=f"{float(funding_rate) * 100:.4f}%",
                    )
                    return results

                # Exit condition 2: Profit target reached
                if self.positions:
                    position = self.positions[0]
                    pnl = position.pnl_percentage

                    if pnl >= self.config.profit_target:
                        await self.close_all_positions()
                        results["action"] = "exit"
                        results["position_closed"] = True
                        results["reason"] = "profit_target"

                        logger.info(
                            "Closed Funding Capture position due to profit target reached",
                            pnl=f"{float(pnl) * 100:.2f}%",
                            target=f"{float(self.config.profit_target) * 100:.2f}%",
                        )
                        return results

                # Still holding position
                results["action"] = "hold"
                logger.info(
                    "Maintaining Funding Capture position",
                    funding_rate=f"{float(funding_rate) * 100:.4f}%",
                )

            else:
                # No position, check entry condition
                if funding_rate <= self.config.entry_threshold:
                    # Funding rate is below threshold, enter position
                    position_size = self._calculate_position_size()

                    # Set leverage
                    await self.gateway.set_leverage(
                        self.perpetual_symbol, int(self.config.max_leverage)
                    )

                    # Create order
                    order = await self.gateway.create_order(
                        symbol=self.perpetual_symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        amount=position_size,
                    )

                    self.orders.append(order)
                    results["action"] = "enter"
                    results["position_opened"] = True

                    logger.info(
                        "Opened Funding Capture position",
                        symbol=self.perpetual_symbol,
                        size=str(position_size),
                        funding_rate=f"{float(funding_rate) * 100:.4f}%",
                    )
                else:
                    # Funding rate not negative enough, wait
                    results["action"] = "wait"
                    logger.info(
                        "Waiting for more negative funding rate",
                        current_rate=f"{float(funding_rate) * 100:.4f}%",
                        threshold=f"{float(self.config.entry_threshold) * 100:.4f}%",
                    )

            return results

        except Exception as e:
            logger.error("Error executing Funding Capture strategy", exc_info=True)
            results["error"] = str(e)
            return results

    async def harvest_profits(self) -> Decimal:
        """
        Harvest profits from the strategy and transfer to Core HODL.

        This method calculates realized profits from funding payments and
        closed positions, then initiates a withdrawal to the Core HODL cold-wallet.

        Returns:
            Amount of BTC harvested
        """
        # Placeholder implementation
        harvested_amount = Decimal("0")

        logger.info("Harvesting Funding Capture profits", amount=str(harvested_amount))

        return harvested_amount

    async def close_all_positions(self) -> bool:
        """
        Close all open positions.

        Returns:
            True if all positions were closed successfully, False otherwise
        """
        if not self.positions:
            return True

        success = True
        for position in self.positions:
            try:
                order = await self.gateway.create_order(
                    symbol=position.symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                    amount=position.size,
                )

                self.orders.append(order)
                logger.info(
                    f"Closed position {position.id}",
                    symbol=position.symbol,
                    size=str(position.size),
                )
            except Exception as e:
                logger.error(
                    f"Error closing position {position.id}",
                    symbol=position.symbol,
                    error=str(e),
                    exc_info=True,
                )
                success = False

        # Refresh positions list
        if success:
            self.positions = []

        return success

    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the strategy.

        Returns:
            Dictionary with strategy status information
        """
        return {
            "type": "funding_capture",
            "initialized": self.initialized,
            "perpetual_symbol": self.perpetual_symbol,
            "positions": len(self.positions),
            "orders": len(self.orders),
            "funding_history_entries": len(self.funding_history),
            "config": {
                "entry_threshold": float(self.config.entry_threshold),
                "max_leverage": float(self.config.max_leverage),
                "profit_target": float(self.config.profit_target),
            },
        }

    def _calculate_position_size(self) -> Decimal:
        """
        Calculate appropriate position size based on available capital and leverage.

        Returns:
            Position size in BTC
        """
        # Placeholder implementation
        available_btc = self.sub_portfolio.current_balance_btc * Decimal("0.9")
        position_size = available_btc * self.config.max_leverage

        # Round to 3 decimal places (0.001 BTC precision)
        return position_size.quantize(Decimal("0.001"))


class OptionPremiumStrategy(BaseStrategy):
    """
    Option Premium Strategy implementation.

    This strategy sells cash-secured put options with a delta of approximately 0.20,
    representing an approximate 20% probability of being in-the-money at expiry.
    """

    def __init__(
        self,
        gateway: ExchangeGateway,
        sub_portfolio: SubPortfolio,
        config: OptionPremiumConfig | dict[str, Any],
    ):
        """
        Initialize the Option Premium strategy.

        Args:
            gateway: Exchange gateway for executing trades
            sub_portfolio: Sub-portfolio allocated to this strategy
            config: Strategy-specific configuration
        """
        if not isinstance(config, OptionPremiumConfig):
            config = OptionPremiumConfig(**config)

        super().__init__(gateway, sub_portfolio, config)
        self.config: OptionPremiumConfig = config
        self.options: list[Option] = []

    async def initialize(self) -> None:
        """
        Initialize the strategy, load current option positions.
        """
        logger.info("Initializing Option Premium strategy")

        # Placeholder - in a real implementation, this would:
        # 1. Fetch current option positions from Deribit
        # 2. Load option chain data
        # 3. Initialize any necessary state

        self.initialized = True
        logger.info("Option Premium strategy initialized", options=len(self.options))

    @log_strategy_execution("option_premium")
    async def execute(self) -> dict[str, Any]:
        """
        Execute the Option Premium strategy logic.

        1. Check for expired options and handle assignments/expirations
        2. If capital available, find new put options to sell
        3. Select strikes with delta closest to target (0.20)

        Returns:
            Dictionary with execution results
        """
        if not self.initialized:
            await self.initialize()

        results = {
            "action": "none",
            "option_sold": False,
            "premium_collected": None,
            "strike_price": None,
            "expiry_days": None,
            "delta": None,
            "error": None,
        }

        try:
            # Handle expired options (placeholder)
            await self._handle_expired_options()

            # Check if we have capital available to sell new puts
            available_capital = self._calculate_available_capital()
            if available_capital <= Decimal("0"):
                results["action"] = "wait"
                logger.info("No capital available for new option positions")
                return results

            # Find suitable options to sell (placeholder)
            option_chain = await self._get_option_chain()

            # Filter for puts with appropriate expiry
            now = datetime.now()
            valid_options = [
                opt
                for opt in option_chain
                if (
                    opt["option_type"] == "put"
                    and self.config.min_expiry_days
                    <= (opt["expiry_date"] - now).days
                    <= self.config.max_expiry_days
                )
            ]

            if not valid_options:
                results["action"] = "wait"
                logger.info("No suitable options found")
                return results

            # Find option with delta closest to target
            target_delta = self.config.delta_target
            best_option = min(valid_options, key=lambda opt: abs(opt["delta"] - target_delta))

            # Calculate position size based on available capital
            max_contracts = (available_capital / best_option["strike_price"]).quantize(
                Decimal("0.01")
            )

            # Sell the put option
            order = await self.gateway.create_order(
                symbol=best_option["symbol"],
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL,
                amount=max_contracts,
                price=best_option["bid"],  # Sell at the bid price
            )

            self.orders.append(order)

            # Create option record
            option = Option(
                strategy_id=None,  # Will be filled by database
                exchange="deribit",
                underlying="BTC",
                strike_price=best_option["strike_price"],
                expiry_date=best_option["expiry_date"],
                option_type=OptionType.PUT,
                side="sell",
                size=max_contracts,
                premium=best_option["bid"],
                total_premium=best_option["bid"] * max_contracts,
                collateral=best_option["strike_price"] * max_contracts,
                delta=best_option["delta"],
            )

            self.options.append(option)

            results["action"] = "sell_put"
            results["option_sold"] = True
            results["premium_collected"] = float(option.total_premium)
            results["strike_price"] = float(option.strike_price)
            results["expiry_days"] = option.days_to_expiry
            results["delta"] = float(option.delta)

            logger.info(
                "Sold put option",
                strike=str(option.strike_price),
                expiry=option.expiry_date.isoformat(),
                contracts=str(option.size),
                premium=str(option.total_premium),
                delta=str(option.delta),
            )

            return results

        except Exception as e:
            logger.error("Error executing Option Premium strategy", exc_info=True)
            results["error"] = str(e)
            return results

    async def harvest_profits(self) -> Decimal:
        """
        Harvest profits from the strategy and transfer to Core HODL.

        This method calculates realized profits from expired options and
        initiates a withdrawal to the Core HODL cold-wallet.

        Returns:
            Amount of BTC harvested
        """
        # Placeholder implementation
        harvested_amount = Decimal("0")

        logger.info("Harvesting Option Premium profits", amount=str(harvested_amount))

        return harvested_amount

    async def close_all_positions(self) -> bool:
        """
        Close all open option positions by buying them back.

        Returns:
            True if all positions were closed successfully, False otherwise
        """
        if not self.options:
            return True

        success = True
        for option in self.options:
            if option.status != OptionStatus.OPEN:
                continue

            try:
                # Buy back the option to close position
                order = await self.gateway.create_order(
                    symbol=f"BTC-{option.strike_price}-{option.expiry_date.strftime('%d%b%y')}-P",  # Example format
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,  # Buy to close the short position
                    amount=option.size,
                )

                self.orders.append(order)
                option.status = OptionStatus.CLOSED
                option.exit_time = datetime.now(UTC)

                logger.info(
                    f"Closed option position {option.id}",
                    strike=str(option.strike_price),
                    expiry=option.expiry_date.isoformat(),
                )
            except Exception as e:
                logger.error(
                    f"Error closing option position {option.id}",
                    strike=str(option.strike_price),
                    expiry=option.expiry_date.isoformat(),
                    error=str(e),
                    exc_info=True,
                )
                success = False

        return success

    def get_status(self) -> dict[str, Any]:
        """
        Get the current status of the strategy.

        Returns:
            Dictionary with strategy status information
        """
        # Count options by status
        status_counts = {}
        for status in OptionStatus:
            status_counts[status.value] = len([o for o in self.options if o.status == status])

        return {
            "type": "option_premium",
            "initialized": self.initialized,
            "open_options": status_counts.get(OptionStatus.OPEN.value, 0),
            "expired_otm_options": status_counts.get(OptionStatus.EXPIRED_OTM.value, 0),
            "exercised_options": status_counts.get(OptionStatus.EXERCISED.value, 0),
            "closed_options": status_counts.get(OptionStatus.CLOSED.value, 0),
            "total_premium_collected": sum(o.total_premium for o in self.options),
            "config": {
                "delta_target": float(self.config.delta_target),
                "min_expiry_days": self.config.min_expiry_days,
                "max_expiry_days": self.config.max_expiry_days,
            },
        }

    async def _handle_expired_options(self) -> None:
        """Handle expired options, processing assignments or expirations."""
        now = datetime.now(UTC)

        for option in self.options:
            if option.status != OptionStatus.OPEN:
                continue

            if option.expiry_date <= now:
                # Option has expired, check if it was ITM or OTM
                spot_price = await self._get_spot_price()

                if option.option_type == OptionType.PUT and spot_price < option.strike_price:
                    # Put is ITM, handle assignment
                    option.status = OptionStatus.EXERCISED
                    logger.info(
                        "Put option assigned",
                        strike=str(option.strike_price),
                        spot_price=str(spot_price),
                        contracts=str(option.size),
                    )
                else:
                    # Option expired OTM
                    option.status = OptionStatus.EXPIRED_OTM
                    logger.info(
                        "Option expired worthless",
                        strike=str(option.strike_price),
                        spot_price=str(spot_price),
                        premium=str(option.total_premium),
                    )

    async def _get_option_chain(self) -> list[dict[str, Any]]:
        """
        Get the current option chain from the exchange.

        Returns:
            List of available options with their details
        """
        # Placeholder implementation
        # In a real implementation, this would fetch the option chain from Deribit
        spot_price = await self._get_spot_price()

        # Generate some sample options
        option_chain = []
        now = datetime.now(UTC)

        # Add options with different strikes and expiries
        for days in [60, 75, 90]:
            expiry = now + timedelta(days=days)
            for strike_pct in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                strike = (spot_price * Decimal(str(strike_pct))).quantize(Decimal("100"))

                # Calculate approximate delta based on strike distance
                # This is a very simplified approximation
                delta_approx = Decimal("0.5") - ((spot_price - strike) / spot_price)

                # Add put option
                option_chain.append(
                    {
                        "symbol": f"BTC-{strike}-{expiry.strftime('%d%b%y')}-P",
                        "option_type": "put",
                        "strike_price": strike,
                        "expiry_date": expiry,
                        "bid": (
                            spot_price * Decimal("0.01") * (Decimal("1") - strike_pct)
                        ).quantize(Decimal("0.0001")),
                        "ask": (
                            spot_price * Decimal("0.012") * (Decimal("1") - strike_pct)
                        ).quantize(Decimal("0.0001")),
                        "delta": delta_approx.quantize(Decimal("0.01")),
                        "implied_vol": Decimal("0.7"),
                    }
                )

        return option_chain

    async def _get_spot_price(self) -> Decimal:
        """
        Get the current BTC spot price.

        Returns:
            Current BTC price in USD
        """
        # Placeholder implementation
        ticker = await self.gateway.get_ticker("BTC/USD")
        return ticker["last"]

    def _calculate_available_capital(self) -> Decimal:
        """
        Calculate available capital for new option positions.

        Returns:
            Available capital in BTC
        """
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Check current sub-portfolio balance
        # 2. Subtract collateral locked in existing positions
        # 3. Apply any safety margin

        # Simple example: 90% of current balance minus locked collateral
        total_collateral = sum(o.collateral for o in self.options if o.status == OptionStatus.OPEN)
        available = (self.sub_portfolio.current_balance_btc * Decimal("0.9")) - total_collateral

        return max(Decimal("0"), available)


# Export strategy classes
__all__ = [
    "BaseStrategy",
    "BasisHarvestStrategy",
    "FundingCaptureStrategy",
    "OptionPremiumStrategy",
]
