"""
Exchange gateway module for BTC Stack-Builder Bot.

This module provides gateway classes for interacting with cryptocurrency exchanges.
Each gateway abstracts the API calls, authentication, rate limiting, and error handling
for a specific exchange.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from btc_stack_builder.core.models import (
    ExchangeCredentials,
    MarginStatus,
    Order,
    OrderSide,
    # OrderStatus, # Unused in this file
    OrderType,
    Position,
    # PositionSide, # Unused in this file
    # PositionStatus, # Unused in this file
)

from .binance_gateway import BinanceGateway
from .deribit_gateway import DeribitGateway


class GatewayError(Exception):
    """Base exception for all gateway-related errors."""

    pass


class GatewayConnectionError(GatewayError):
    """Exception raised when connection to exchange fails."""

    pass


class AuthenticationError(GatewayError):
    """Exception raised when authentication with exchange fails."""

    pass


class OrderError(GatewayError):
    """Exception raised when order placement or cancellation fails."""

    pass


class RateLimitError(GatewayError):
    """Exception raised when exchange rate limit is exceeded."""

    pass


class InsufficientFundsError(GatewayError):
    """Exception raised when account has insufficient funds."""

    pass


class ExchangeGateway(ABC):
    """Base abstract class for all exchange gateways."""

    def __init__(self, credentials: ExchangeCredentials, is_testnet: bool = False):
        """
        Initialize the exchange gateway.

        Args:
            credentials: API credentials for the exchange
            is_testnet: Whether to connect to the testnet/sandbox environment
        """
        self.credentials = credentials
        self.is_testnet = is_testnet
        self.exchange_info: dict[str, Any] = {}
        self.initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the gateway connection and fetch exchange information.

        This method should be called before using any other methods.
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get current ticker information for a symbol.

        Args:
            symbol: Trading pair or contract symbol

        Returns:
            Dictionary with ticker information
        """
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """
        Get current orderbook for a symbol.

        Args:
            symbol: Trading pair or contract symbol
            limit: Number of price levels to retrieve

        Returns:
            Dictionary with orderbook information
        """
        pass

    @abstractmethod
    async def get_balance(self) -> dict[str, dict[str, Decimal]]:
        """
        Get account balances.

        Returns:
            Dictionary mapping asset names to balance information
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get open positions.

        Returns:
            List of open positions
        """
        pass

    @abstractmethod
    async def get_margin_status(self) -> MarginStatus:
        """
        Get current margin status.

        Returns:
            MarginStatus object with current margin information
        """
        pass

    @abstractmethod
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
            params: Additional parameters specific to the exchange

        Returns:
            Order object with order information

        Raises:
            OrderError: If order creation fails
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """
        Get current funding rate for a perpetual contract.

        Args:
            symbol: Contract symbol

        Returns:
            Dictionary with funding rate information
        """
        pass

    @abstractmethod
    async def get_futures_basis(self, spot_symbol: str, futures_symbol: str) -> Decimal:
        """
        Calculate the basis between spot and futures prices.

        Args:
            spot_symbol: Spot market symbol
            futures_symbol: Futures contract symbol

        Returns:
            Basis as a decimal percentage
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the gateway connection and release resources.
        """
        pass


# Import specific gateway implementations
# Export gateway classes
__all__ = [
    "ExchangeGateway",
    "BinanceGateway",
    "DeribitGateway",
    "GatewayError",
    "GatewayConnectionError",
    "AuthenticationError",
    "OrderError",
    "RateLimitError",
    "InsufficientFundsError",
]
