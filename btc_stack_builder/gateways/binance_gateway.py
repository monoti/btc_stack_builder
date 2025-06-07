"""
Binance Gateway module for BTC Stack-Builder Bot.

This module provides a gateway implementation for interacting with Binance exchange,
including spot and futures markets. It handles authentication, rate limiting,
error handling, and conversion between exchange-specific and internal data models.
"""

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import ccxt
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from btc_stack_builder.core.logger import logger
from btc_stack_builder.core.models import (
    ExchangeCredentials,
    MarginLevel,
    MarginStatus,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
)
from btc_stack_builder.core.utils import (
    calculate_annualized_basis,
    calculate_days_to_expiry,
    calculate_margin_ratio,
    parse_quarterly_futures_symbol,
    timestamp_to_datetime,
)
from btc_stack_builder.gateways import (
    AuthenticationError,
    ExchangeGateway,
    GatewayConnectionError,
    GatewayError,
    InsufficientFundsError,
    OrderError,
    RateLimitError,
)


class BinanceGateway(ExchangeGateway):
    """Gateway implementation for Binance exchange."""

    # Constants for Binance-specific configuration
    SPOT_MARKET_TYPE = "spot"
    FUTURES_MARKET_TYPE = "delivery"  # COIN-M futures
    PERPETUAL_MARKET_TYPE = "future"  # USDT-M futures with perpetuals

    # Maximum number of retry attempts for API calls
    MAX_RETRIES = 3

    # Rate limiting parameters
    RATE_LIMIT_WEIGHT_THRESHOLD = 0.8  # 80% of rate limit

    def __init__(self, credentials: ExchangeCredentials, is_testnet: bool = False):
        """
        Initialize the Binance gateway.

        Args:
            credentials: API credentials for Binance
            is_testnet: Whether to connect to the testnet environment
        """
        super().__init__(credentials, is_testnet)

        # Initialize CCXT clients for different market types
        self.spot_client = None
        self.futures_client = None
        self.perpetual_client = None

        # Track rate limits
        self.rate_limit_last_check = time.time()
        self.rate_limit_counter = 0
        self.rate_limit_max = 1200  # Default, will be updated from exchange info

        # Cache for exchange information
        self.symbols_info = {}
        self.last_ticker_cache = {}
        self.last_funding_rate_cache = {}

    async def initialize(self) -> None:
        """
        Initialize the gateway connections to Binance.

        This method creates the CCXT clients for spot and futures markets
        and fetches initial exchange information.

        Raises:
            ConnectionError: If connection to Binance fails
            AuthenticationError: If authentication fails
        """
        try:
            # Common options for all clients
            options = {
                "apiKey": self.credentials.api_key.get_secret_value(),
                "secret": self.credentials.api_secret.get_secret_value(),
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",  # Default to spot market
                    "adjustForTimeDifference": True,
                    "recvWindow": 5000,  # 5 seconds
                },
            }

            # Add testnet configuration if needed
            if self.is_testnet:
                options["options"]["test"] = True

            # Initialize spot client
            self.spot_client = ccxt.binance(options)

            # Initialize futures client (COIN-M)
            futures_options = options.copy()
            futures_options["options"] = options["options"].copy()
            futures_options["options"]["defaultType"] = "delivery"
            self.futures_client = ccxt.binance(futures_options)

            # Initialize perpetual futures client (USDT-M)
            perp_options = options.copy()
            perp_options["options"] = options["options"].copy()
            perp_options["options"]["defaultType"] = "future"
            self.perpetual_client = ccxt.binance(perp_options)

            # Load markets for all clients
            await asyncio.gather(
                self._async_execute(self.spot_client.load_markets),
                self._async_execute(self.futures_client.load_markets),
                self._async_execute(self.perpetual_client.load_markets),
            )

            # Cache symbol information
            self._cache_symbols_info()

            # Fetch exchange information including rate limits
            exchange_info = await self._async_execute(self.spot_client.public_get_exchangeinfo)
            self._process_exchange_info(exchange_info)

            # Test authentication by fetching account information
            await self.get_balance()

            logger.info(
                "Binance gateway initialized successfully",
                testnet=self.is_testnet,
                spot_markets=len(self.spot_client.markets),
                futures_markets=len(self.futures_client.markets),
                perpetual_markets=len(self.perpetual_client.markets),
            )

            self.initialized = True

        except ccxt.AuthenticationError as e:
            logger.error("Binance authentication failed", exc_info=True)
            raise AuthenticationError(f"Binance authentication failed: {str(e)}") from e
        except ccxt.NetworkError as e:
            logger.error("Binance connection failed", exc_info=True)
            raise GatewayConnectionError(f"Binance connection failed: {str(e)}") from e
        except Exception as e:
            logger.error("Binance gateway initialization failed", exc_info=True)
            raise GatewayError(f"Binance gateway initialization failed: {str(e)}") from e

    def _cache_symbols_info(self) -> None:
        """Cache symbol information from loaded markets."""
        # Cache spot symbols
        for symbol, market in self.spot_client.markets.items():
            self.symbols_info[symbol] = {
                "type": self.SPOT_MARKET_TYPE,
                "base": market["base"],
                "quote": market["quote"],
                "precision": market["precision"],
                "limits": market["limits"],
                "info": market["info"],
            }

        # Cache futures symbols
        for symbol, market in self.futures_client.markets.items():
            self.symbols_info[symbol] = {
                "type": self.FUTURES_MARKET_TYPE,
                "base": market["base"],
                "quote": market["quote"],
                "precision": market["precision"],
                "limits": market["limits"],
                "info": market["info"],
                "expiry": market.get("expiry"),
                "contract_size": market.get("contractSize", 1),
            }

        # Cache perpetual symbols
        for symbol, market in self.perpetual_client.markets.items():
            self.symbols_info[symbol] = {
                "type": self.PERPETUAL_MARKET_TYPE,
                "base": market["base"],
                "quote": market["quote"],
                "precision": market["precision"],
                "limits": market["limits"],
                "info": market["info"],
                "contract_size": market.get("contractSize", 1),
            }

    def _process_exchange_info(self, exchange_info: dict[str, Any]) -> None:
        """
        Process exchange information to extract rate limits and other metadata.

        Args:
            exchange_info: Exchange information from Binance API
        """
        # Extract rate limits
        if "rateLimits" in exchange_info:
            for limit in exchange_info["rateLimits"]:
                if limit["rateLimitType"] == "REQUEST_WEIGHT" and limit["interval"] == "MINUTE":
                    self.rate_limit_max = int(limit["limit"])
                    break

        # Store exchange info for later use
        self.exchange_info = exchange_info

    def _get_client_for_symbol(self, symbol: str) -> ccxt.Exchange:
        """
        Get the appropriate CCXT client for a given symbol.

        Args:
            symbol: Trading pair or contract symbol

        Returns:
            CCXT client for the symbol's market type

        Raises:
            ValueError: If symbol is not recognized
        """
        if not self.initialized:
            raise GatewayError("Binance gateway not initialized")

        if symbol not in self.symbols_info:
            raise ValueError(f"Unknown symbol: {symbol}")

        symbol_type = self.symbols_info[symbol]["type"]

        if symbol_type == self.SPOT_MARKET_TYPE:
            return self.spot_client
        elif symbol_type == self.FUTURES_MARKET_TYPE:
            return self.futures_client
        elif symbol_type == self.PERPETUAL_MARKET_TYPE:
            return self.perpetual_client
        else:
            raise ValueError(f"Unsupported market type for symbol {symbol}: {symbol_type}")

    async def _handle_ccxt_error(self, e: Exception) -> None:
        """
        Handle CCXT errors and convert them to gateway-specific exceptions.

        Args:
            e: CCXT exception

        Raises:
            Appropriate gateway-specific exception
        """
        if isinstance(e, ccxt.AuthenticationError):
            raise AuthenticationError(f"Binance authentication error: {str(e)}") from e
        elif isinstance(e, ccxt.InsufficientFunds):
            raise InsufficientFundsError(f"Insufficient funds: {str(e)}")
        elif isinstance(e, ccxt.RateLimitExceeded):
            raise RateLimitError(f"Rate limit exceeded: {str(e)}")
        elif isinstance(e, ccxt.NetworkError):
            raise GatewayConnectionError(f"Network error: {str(e)}") from e
        elif isinstance(e, ccxt.ExchangeError):
            if "Order does not exist" in str(e):
                raise OrderError(f"Order not found: {str(e)}") from e
            else:
                raise OrderError(f"Exchange error: {str(e)}") from e
        else:
            raise GatewayError(f"Unexpected error: {str(e)}") from e

    async def _async_execute(self, func, *args, **kwargs):
        """
        Execute a synchronous CCXT method in an asynchronous context.

        Args:
            func: Synchronous CCXT method to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method call
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _execute_with_retry(self, method: str, client: ccxt.Exchange, *args, **kwargs) -> Any:
        """
        Execute a CCXT method with retry logic for transient errors.

        Args:
            method: Name of the CCXT method to call
            client: CCXT client to use
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method call

        Raises:
            GatewayError: If all retry attempts fail
        """
        try:
            # Get the method from the client
            func = getattr(client, method)

            # Execute the method (synchronously in a thread pool)
            result = await self._async_execute(func, *args, **kwargs)

            # Update rate limit counter
            self._update_rate_limit_counter(method)

            return result
        except Exception as e:
            logger.warning(
                f"Error executing {method}",
                method=method,
                args=args,
                kwargs=kwargs,
                error=str(e),
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    def _update_rate_limit_counter(self, method: str) -> None:
        """
        Update the rate limit counter based on the method called.

        Args:
            method: Name of the CCXT method called
        """
        # Simple rate limit tracking - in a production system, this would be more sophisticated
        current_time = time.time()
        if current_time - self.rate_limit_last_check > 60:  # Reset counter every minute
            self.rate_limit_counter = 0
            self.rate_limit_last_check = current_time

        # Increment counter based on method weight (simplified)
        if method.startswith("fetch"):
            self.rate_limit_counter += 1
        else:
            self.rate_limit_counter += 5  # Higher weight for mutations

        # Check if we're approaching the rate limit
        if self.rate_limit_counter > self.rate_limit_max * self.RATE_LIMIT_WEIGHT_THRESHOLD:
            logger.warning(
                "Approaching Binance rate limit",
                current=self.rate_limit_counter,
                max=self.rate_limit_max,
                threshold_percent=self.RATE_LIMIT_WEIGHT_THRESHOLD * 100,
            )

    def _convert_order_type(self, order_type: OrderType) -> str:
        """
        Convert internal OrderType to Binance order type.

        Args:
            order_type: Internal order type

        Returns:
            Binance order type string
        """
        order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_MARKET: "stop_market",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TAKE_PROFIT_MARKET: "take_profit_market",
            OrderType.TAKE_PROFIT_LIMIT: "take_profit_limit",
        }
        return order_type_map.get(order_type, "market")

    def _convert_order_side(self, side: OrderSide) -> str:
        """
        Convert internal OrderSide to Binance order side.

        Args:
            side: Internal order side

        Returns:
            Binance order side string
        """
        return side.value.lower()

    def _convert_position_side(self, side: str) -> PositionSide:
        """
        Convert Binance position side to internal PositionSide.

        Args:
            side: Binance position side

        Returns:
            Internal position side enum
        """
        side_lower = side.lower()
        if side_lower in ("long", "buy"):
            return PositionSide.LONG
        elif side_lower in ("short", "sell"):
            return PositionSide.SHORT
        else:
            raise ValueError(f"Unknown position side: {side}")

    def _convert_order_status(self, status: str) -> OrderStatus:
        """
        Convert Binance order status to internal OrderStatus.

        Args:
            status: Binance order status

        Returns:
            Internal order status enum
        """
        status_map = {
            "new": OrderStatus.OPEN,
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "cancelled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "pending": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
        }
        return status_map.get(status.lower(), OrderStatus.PENDING)

    def _convert_ccxt_order_to_internal(self, ccxt_order: dict[str, Any]) -> Order:
        """
        Convert CCXT order object to internal Order model.

        Args:
            ccxt_order: CCXT order object

        Returns:
            Internal Order model
        """
        # Extract order type
        order_type_str = ccxt_order.get("type", "market").lower()
        if "stop" in order_type_str and "limit" in order_type_str:
            order_type = OrderType.STOP_LIMIT
        elif "stop" in order_type_str:
            order_type = OrderType.STOP_MARKET
        elif "take_profit" in order_type_str and "limit" in order_type_str:
            order_type = OrderType.TAKE_PROFIT_LIMIT
        elif "take_profit" in order_type_str:
            order_type = OrderType.TAKE_PROFIT_MARKET
        elif "limit" in order_type_str:
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.MARKET

        # Extract order side
        side_str = ccxt_order.get("side", "").lower()
        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

        # Extract timestamps
        created_at = timestamp_to_datetime(ccxt_order.get("timestamp", time.time() * 1000) / 1000)
        updated_at = timestamp_to_datetime(
            ccxt_order.get("lastUpdateTimestamp", time.time() * 1000) / 1000
        )

        # Create internal order object
        return Order(
            exchange_id=str(ccxt_order.get("id", "")),
            exchange="binance",
            symbol=ccxt_order.get("symbol", ""),
            order_type=order_type,
            side=side,
            price=Decimal(str(ccxt_order.get("price", 0))),
            amount=Decimal(str(ccxt_order.get("amount", 0))),
            filled_amount=Decimal(str(ccxt_order.get("filled", 0))),
            status=self._convert_order_status(ccxt_order.get("status", "")),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _convert_ccxt_position_to_internal(self, ccxt_position: dict[str, Any]) -> Position:
        """
        Convert CCXT position object to internal Position model.

        Args:
            ccxt_position: CCXT position object

        Returns:
            Internal Position model
        """
        # Determine position side
        contracts = float(ccxt_position.get("contracts", 0))
        side = PositionSide.LONG if contracts > 0 else PositionSide.SHORT

        # Extract position size (absolute value)
        size = Decimal(str(abs(contracts)))

        # Extract prices
        entry_price = Decimal(str(ccxt_position.get("entryPrice", 0)))
        current_price = Decimal(str(ccxt_position.get("markPrice", entry_price)))
        liquidation_price = Decimal(str(ccxt_position.get("liquidationPrice", 0)))

        # Extract PnL information
        unrealized_pnl = Decimal(str(ccxt_position.get("unrealizedPnl", 0)))

        # Determine position status
        status = PositionStatus.OPEN

        # Extract timestamps
        timestamp = ccxt_position.get("timestamp", time.time() * 1000) / 1000
        entry_time = timestamp_to_datetime(timestamp)

        # Extract leverage
        leverage = Decimal(str(ccxt_position.get("leverage", 1)))

        # Create internal position object
        return Position(
            strategy_id=None,  # Will be filled by the strategy
            exchange="binance",
            symbol=ccxt_position.get("symbol", ""),
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            size=size,
            leverage=leverage,
            liquidation_price=liquidation_price,
            unrealized_pnl=unrealized_pnl,
            status=status,
            entry_time=entry_time,
        )

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get current ticker information for a symbol.

        Args:
            symbol: Trading pair or contract symbol

        Returns:
            Dictionary with ticker information
        """
        client = self._get_client_for_symbol(symbol)

        try:
            ticker = await self._execute_with_retry("fetch_ticker", client, symbol)

            # Cache the ticker for basis calculations
            self.last_ticker_cache[symbol] = ticker

            return {
                "symbol": symbol,
                "last": Decimal(str(ticker["last"])) if ticker["last"] else None,
                "bid": Decimal(str(ticker["bid"])) if ticker["bid"] else None,
                "ask": Decimal(str(ticker["ask"])) if ticker["ask"] else None,
                "high": Decimal(str(ticker["high"])) if ticker["high"] else None,
                "low": Decimal(str(ticker["low"])) if ticker["low"] else None,
                "volume": Decimal(str(ticker["volume"])) if ticker["volume"] else None,
                "timestamp": timestamp_to_datetime(ticker["timestamp"] / 1000),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)
            return {}  # Explicitly return an empty dictionary in case of an error
    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """
        Get current orderbook for a symbol.

        Args:
            symbol: Trading pair or contract symbol
            limit: Number of price levels to retrieve

        Returns:
            Dictionary with orderbook information
        """
        client = self._get_client_for_symbol(symbol)

        try:
            orderbook = await self._execute_with_retry("fetch_order_book", client, symbol, limit)

            # Convert to Decimal for precision
            bids = [
                [Decimal(str(price)), Decimal(str(amount))] for price, amount in orderbook["bids"]
            ]
            asks = [
                [Decimal(str(price)), Decimal(str(amount))] for price, amount in orderbook["asks"]
            ]

            return {
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "timestamp": timestamp_to_datetime(orderbook["timestamp"] / 1000),
            }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)
            return {}  # Explicitly return an empty dictionary in case of an error
    async def get_balance(self) -> dict[str, dict[str, Decimal]]:
        """
        Get account balances for all assets.

        Returns:
            Dictionary mapping asset names to balance information
        """
        balances = {}

        try:
            # Fetch spot balances
            spot_balance = await self._execute_with_retry("fetch_balance", self.spot_client)
            for currency, data in spot_balance.items():
                if currency not in ("info", "timestamp", "datetime", "free", "used", "total"):
                    balances[currency] = {
                        "free": Decimal(str(data.get("free", 0))),
                        "used": Decimal(str(data.get("used", 0))),
                        "total": Decimal(str(data.get("total", 0))),
                        "account_type": "spot",
                    }

            # Fetch futures balances
            futures_balance = await self._execute_with_retry("fetch_balance", self.futures_client)
            for currency, data in futures_balance.items():
                if currency not in ("info", "timestamp", "datetime", "free", "used", "total"):
                    balances[f"{currency}_futures"] = {
                        "free": Decimal(str(data.get("free", 0))),
                        "used": Decimal(str(data.get("used", 0))),
                        "total": Decimal(str(data.get("total", 0))),
                        "account_type": "futures",
                    }

            # Fetch perpetual futures balances
            perp_balance = await self._execute_with_retry("fetch_balance", self.perpetual_client)
            for currency, data in perp_balance.items():
                if currency not in ("info", "timestamp", "datetime", "free", "used", "total"):
                    balances[f"{currency}_perpetual"] = {
                        "free": Decimal(str(data.get("free", 0))),
                        "used": Decimal(str(data.get("used", 0))),
                        "total": Decimal(str(data.get("total", 0))),
                        "account_type": "perpetual",
                    }

            return balances
        except Exception as e:
            logger.error("Error fetching balances", exc_info=True)
            await self._handle_ccxt_error(e)
        return balances  # Explicitly return the balances dictionary, even if empty
    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        positions = []

        try:
            # Fetch futures positions
            futures_positions = await self._execute_with_retry(
                "fetch_positions", self.futures_client
            )
            for pos in futures_positions:
                if float(pos.get("contracts", 0)) != 0:  # Only include open positions
                    positions.append(self._convert_ccxt_position_to_internal(pos))

            # Fetch perpetual positions
            perp_positions = await self._execute_with_retry(
                "fetch_positions", self.perpetual_client
            )
            for pos in perp_positions:
                if float(pos.get("contracts", 0)) != 0:  # Only include open positions
                    positions.append(self._convert_ccxt_position_to_internal(pos))

            return positions
        except Exception as e:
            logger.error("Error fetching positions", exc_info=True)
            await self._handle_ccxt_error(e)
        return positions  # Explicitly return the positions list, even if empty
    async def get_margin_status(self) -> MarginStatus:
        """
        Get current margin status for futures account.

        Returns:
            MarginStatus object with current margin information
        """
        try:
            # Get account information for futures
            account_info = await self._execute_with_retry(
                "fapiPrivate_get_account", self.futures_client
            )

            # Extract margin information
            wallet_balance = Decimal(str(account_info.get("totalWalletBalance", 0)))
            unrealized_pnl = Decimal(str(account_info.get("totalUnrealizedProfit", 0)))
            maintenance_margin = Decimal(str(account_info.get("totalMaintMargin", 0)))
            initial_margin = Decimal(str(account_info.get("totalInitialMargin", 0)))

            # Calculate margin ratio
            margin_ratio = calculate_margin_ratio(wallet_balance, maintenance_margin)

            # Determine margin level
            margin_level = MarginLevel.SAFE
            if margin_ratio < Decimal("1.0"):
                margin_level = MarginLevel.LIQUIDATION
            elif margin_ratio < Decimal("4.0"):
                margin_level = MarginLevel.CRITICAL
            elif margin_ratio < Decimal("4.5"):
                margin_level = MarginLevel.WARNING

            return MarginStatus(
                exchange="binance",
                account_type="futures",
                wallet_balance=wallet_balance,
                unrealized_pnl=unrealized_pnl,
                maintenance_margin=maintenance_margin,
                initial_margin=initial_margin,
                margin_ratio=margin_ratio,
                margin_level=margin_level,
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            logger.error("Error fetching margin status", exc_info=True)
            await self._handle_ccxt_error(e)

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: Decimal,
        price: Decimal | None = None,
        params: dict[str, Any] | None = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading pair or contract symbol
            order_type: Type of order
            side: Order side (buy/sell)
            amount: Order quantity
            price: Order price (required for limit orders)
            params: Additional parameters specific to Binance

        Returns:
            Order object with order information

        Raises:
            OrderError: If order creation fails
        """
        client = self._get_client_for_symbol(symbol)
        params = params or {}

        # Convert to CCXT format
        ccxt_order_type = self._convert_order_type(order_type)
        ccxt_side = self._convert_order_side(side)

        try:
            # Create the order
            ccxt_order = await self._execute_with_retry(
                "create_order",
                client,
                symbol,
                ccxt_order_type,
                ccxt_side,
                float(amount),
                float(price) if price else None,
                params,
            )

            # Convert to internal format
            order = self._convert_ccxt_order_to_internal(ccxt_order)

            logger.info(
                f"Created order: {order.exchange_id}",
                order_id=order.exchange_id,
                symbol=symbol,
                type=ccxt_order_type,
                side=ccxt_side,
                amount=str(amount),
                price=str(price) if price else None,
            )

            return order
        except Exception as e:
            logger.error(
                f"Error creating order for {symbol}",
                symbol=symbol,
                type=ccxt_order_type,
                side=ccxt_side,
                amount=str(amount),
                price=str(price) if price else None,
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Exchange-specific order ID
            symbol: Trading pair or contract symbol

        Returns:
            True if order was cancelled successfully, False otherwise

        Raises:
            OrderError: If order cancellation fails
        """
        client = self._get_client_for_symbol(symbol)

        try:
            # Cancel the order
            result = await self._execute_with_retry("cancel_order", client, order_id, symbol)

            logger.info(
                f"Cancelled order: {order_id}", order_id=order_id, symbol=symbol, result=result
            )

            return True
        except Exception as e:
            if "Order does not exist" in str(e):
                logger.warning(
                    f"Order not found when cancelling: {order_id}", order_id=order_id, symbol=symbol
                )
                return False

            logger.error(
                f"Error cancelling order: {order_id}",
                order_id=order_id,
                symbol=symbol,
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    async def get_order(self, order_id: str, symbol: str) -> Order:
        """
        Get information about an order.

        Args:
            order_id: Exchange-specific order ID
            symbol: Trading pair or contract symbol

        Returns:
            Order object with order information

        Raises:
            OrderError: If order retrieval fails
        """
        client = self._get_client_for_symbol(symbol)

        try:
            # Fetch the order
            ccxt_order = await self._execute_with_retry("fetch_order", client, order_id, symbol)

            # Convert to internal format
            return self._convert_ccxt_order_to_internal(ccxt_order)
        except Exception as e:
            logger.error(
                f"Error fetching order: {order_id}", order_id=order_id, symbol=symbol, exc_info=True
            )
            await self._handle_ccxt_error(e)

    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """
        Get current funding rate for a perpetual contract.

        Args:
            symbol: Contract symbol

        Returns:
            Dictionary with funding rate information
        """
        try:
            # Ensure we're using the perpetual client
            if symbol not in self.perpetual_client.markets:
                raise ValueError(f"Symbol {symbol} is not a perpetual contract")

            # Fetch funding rate
            funding_info = await self._execute_with_retry(
                "fapiPublic_get_premiumindex",
                self.perpetual_client,
                {"symbol": self.perpetual_client.market_id(symbol)},
            )

            # Cache for later use
            self.last_funding_rate_cache[symbol] = funding_info

            # Extract and return relevant information
            result = {
                "symbol": symbol,
                "funding_rate": Decimal(str(funding_info.get("lastFundingRate", 0))),
                "funding_time": timestamp_to_datetime(
                    funding_info.get("nextFundingTime", 0) / 1000
                ),
                "mark_price": Decimal(str(funding_info.get("markPrice", 0))),
                "index_price": Decimal(str(funding_info.get("indexPrice", 0))),
            }

            return result
        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)
            return {
                "symbol": symbol,
                "funding_rate": None,
                "funding_time": None,
                "mark_price": None,
                "index_price": None,
            }

    async def get_futures_basis(self, spot_symbol: str, futures_symbol: str) -> Decimal:
        """
        Calculate the basis between spot and futures prices.

        Args:
            spot_symbol: Spot market symbol
            futures_symbol: Futures contract symbol

        Returns:
            Basis as a decimal percentage
        """
        try:
            # Fetch spot and futures prices
            spot_ticker = await self.get_ticker(spot_symbol)
            futures_ticker = await self.get_ticker(futures_symbol)

            if not spot_ticker["last"] or not futures_ticker["last"]:
                raise ValueError("Unable to get prices for basis calculation")

            spot_price = spot_ticker["last"]
            futures_price = futures_ticker["last"]

            # Get days to expiry
            expiry_date = parse_quarterly_futures_symbol(futures_symbol)
            if not expiry_date:
                raise ValueError(
                    f"Unable to parse expiry date from futures symbol: {futures_symbol}"
                )

            days_to_expiry = calculate_days_to_expiry(expiry_date)
            if days_to_expiry <= 0:
                return Decimal("0")

            # Calculate annualized basis
            basis = calculate_annualized_basis(futures_price, spot_price, days_to_expiry)

            return basis
        except Exception as e:
            logger.error(
                "Error calculating futures basis",
                spot_symbol=spot_symbol,
                futures_symbol=futures_symbol,
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    async def transfer_margin(
        self, amount: Decimal, asset: str, from_type: str, to_type: str
    ) -> bool:
        """
        Transfer assets between different account types (spot, futures, etc.).

        Args:
            amount: Amount to transfer
            asset: Asset to transfer (e.g., 'BTC')
            from_type: Source account type ('spot', 'futures', 'delivery')
            to_type: Destination account type ('spot', 'futures', 'delivery')

        Returns:
            True if transfer was successful, False otherwise
        """
        try:
            # Map account types to Binance transfer types
            type_map = {
                "spot": "SPOT",
                "futures": "FUTURES",
                "delivery": "COIN_FUTURE",
            }

            from_type_binance = type_map.get(from_type.lower())
            to_type_binance = type_map.get(to_type.lower())

            if not from_type_binance or not to_type_binance:
                raise ValueError(f"Invalid account types: {from_type} -> {to_type}")

            # Execute transfer
            result = await self._execute_with_retry(
                "sapi_post_asset_transfer",
                self.spot_client,
                {
                    "type": (
                        "MAIN_UMFUTURE"
                        if (from_type_binance == "SPOT" and to_type_binance == "FUTURES")
                        else "UMFUTURE_MAIN"
                    ),
                    "asset": asset,
                    "amount": float(amount),
                },
            )

            logger.info(
                f"Transferred {amount} {asset} from {from_type} to {to_type}",
                amount=str(amount),
                asset=asset,
                from_type=from_type,
                to_type=to_type,
                result=result,
            )

            return True
        except Exception as e:
            logger.error(
                "Error transferring margin",
                amount=str(amount),
                asset=asset,
                from_type=from_type,
                to_type=to_type,
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a futures symbol.

        Args:
            symbol: Futures contract symbol
            leverage: Leverage value (1-125)

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_client_for_symbol(symbol)

            # Set leverage
            result = await self._execute_with_retry("set_leverage", client, leverage, symbol)

            logger.info(
                f"Set leverage for {symbol} to {leverage}x",
                symbol=symbol,
                leverage=leverage,
                result=result,
            )

            return True
        except Exception as e:
            logger.error(
                f"Error setting leverage for {symbol}",
                symbol=symbol,
                leverage=leverage,
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type for a futures symbol.

        Args:
            symbol: Futures contract symbol
            margin_type: Margin type ('ISOLATED' or 'CROSSED')

        Returns:
            True if successful, False otherwise
        """
        try:
            client = self._get_client_for_symbol(symbol)

            # Normalize margin type
            margin_type = margin_type.upper()
            if margin_type not in ("ISOLATED", "CROSSED"):
                raise ValueError(f"Invalid margin type: {margin_type}")

            # Set margin type
            result = await self._execute_with_retry("set_margin_mode", client, margin_type, symbol)

            logger.info(
                f"Set margin type for {symbol} to {margin_type}",
                symbol=symbol,
                margin_type=margin_type,
                result=result,
            )

            return True
        except Exception as e:
            # Ignore error if margin type is already set
            if "No need to change margin type" in str(e):
                logger.info(
                    f"Margin type for {symbol} already set to {margin_type}",
                    symbol=symbol,
                    margin_type=margin_type,
                )
                return True

            logger.error(
                f"Error setting margin type for {symbol}",
                symbol=symbol,
                margin_type=margin_type,
                exc_info=True,
            )
            await self._handle_ccxt_error(e)

    async def get_funding_history(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get funding payment history for a perpetual contract.

        Args:
            symbol: Contract symbol
            limit: Maximum number of records to retrieve

        Returns:
            List of funding payment records
        """
        try:
            # Ensure we're using the perpetual client
            if symbol not in self.perpetual_client.markets:
                raise ValueError(f"Symbol {symbol} is not a perpetual contract")

            # Fetch funding history
            history = await self._execute_with_retry(
                "fapiPrivate_get_income",
                self.perpetual_client,
                {
                    "symbol": self.perpetual_client.market_id(symbol),
                    "incomeType": "FUNDING_FEE",
                    "limit": limit,
                },
            )

            # Process and return results
            result = []
            for item in history:
                result.append(
                    {
                        "symbol": symbol,
                        "amount": Decimal(str(item.get("income", 0))),
                        "timestamp": timestamp_to_datetime(item.get("time", 0) / 1000),
                        "asset": item.get("asset", ""),
                        "info": item,
                    }
                )

            return result
        except Exception as e:
            logger.error(f"Error fetching funding history for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all open orders, optionally filtered by symbol.

        Args:
            symbol: Trading pair or contract symbol (optional)

        Returns:
            List of open orders
        """
        orders = []

        try:
            # If symbol is provided, use the appropriate client
            if symbol:
                client = self._get_client_for_symbol(symbol)
                ccxt_orders = await self._execute_with_retry("fetch_open_orders", client, symbol)
                for order in ccxt_orders:
                    orders.append(self._convert_ccxt_order_to_internal(order))
            else:
                # Fetch from all markets
                spot_orders = await self._execute_with_retry("fetch_open_orders", self.spot_client)
                futures_orders = await self._execute_with_retry(
                    "fetch_open_orders", self.futures_client
                )
                perp_orders = await self._execute_with_retry(
                    "fetch_open_orders", self.perpetual_client
                )

                # Convert to internal format
                for order in spot_orders + futures_orders + perp_orders:
                    orders.append(self._convert_ccxt_order_to_internal(order))

            return orders
        except Exception as e:
            logger.error("Error fetching open orders", symbol=symbol, exc_info=True)
            await self._handle_ccxt_error(e)

    async def close(self) -> None:
        """
        Close the gateway connection and release resources.
        """
        try:
            # Close all clients
            clients = [self.spot_client, self.futures_client, self.perpetual_client]

            for client in clients:
                if client:
                    # Standard ccxt doesn't have async close method
                    pass

            logger.info("Binance gateway closed")
        except Exception:
            logger.error("Error closing Binance gateway", exc_info=True)
            # Don't re-raise, just log the error
