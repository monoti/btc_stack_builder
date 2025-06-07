"""
Deribit Gateway module for BTC Stack-Builder Bot.

This module provides a gateway implementation for interacting with Deribit exchange,
focusing on options trading. It handles authentication, rate limiting,
error handling, and conversion between exchange-specific and internal data models.
"""
import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import ccxt.pro as ccxtpro
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from btc_stack_builder.core.logger import logger
from btc_stack_builder.core.models import (
    Exchange, ExchangeCredentials, Order, OrderSide, OrderStatus, OrderType,
    Position, PositionSide, PositionStatus, MarginStatus, MarginLevel,
    Option, OptionType, OptionStatus
)
from btc_stack_builder.core.utils import (
    calculate_option_delta,
    calculate_option_greeks,
    timestamp_to_datetime,
    datetime_to_timestamp
)
from btc_stack_builder.gateways import (
    ExchangeGateway,
    GatewayError,
    ConnectionError,
    AuthenticationError,
    OrderError,
    RateLimitError,
    InsufficientFundsError
)


class DeribitGateway(ExchangeGateway):
    """Gateway implementation for Deribit exchange, specializing in options trading."""
    
    # Constants for Deribit-specific configuration
    INSTRUMENT_TYPE_OPTION = 'option'
    INSTRUMENT_TYPE_FUTURE = 'future'
    INSTRUMENT_TYPE_SPOT = 'spot'
    
    # Maximum number of retry attempts for API calls
    MAX_RETRIES = 3
    
    # Rate limiting parameters
    RATE_LIMIT_WEIGHT_THRESHOLD = 0.8  # 80% of rate limit
    
    def __init__(self, credentials: ExchangeCredentials, is_testnet: bool = False):
        """
        Initialize the Deribit gateway.
        
        Args:
            credentials: API credentials for Deribit
            is_testnet: Whether to connect to the testnet environment
        """
        super().__init__(credentials, is_testnet)
        
        # Initialize CCXT client
        self.client = None
        
        # Track rate limits
        self.rate_limit_last_check = time.time()
        self.rate_limit_counter = 0
        self.rate_limit_max = 300  # Default, will be updated from exchange info
        
        # Cache for exchange information
        self.instruments_info = {}
        self.option_chains = {}
        self.last_ticker_cache = {}
    
    async def initialize(self) -> None:
        """
        Initialize the gateway connection to Deribit.
        
        This method creates the CCXT client and fetches initial exchange information.
        
        Raises:
            ConnectionError: If connection to Deribit fails
            AuthenticationError: If authentication fails
        """
        try:
            # Set up client options
            options = {
                'apiKey': self.credentials.api_key.get_secret_value(),
                'secret': self.credentials.api_secret.get_secret_value(),
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,  # 5 seconds
                }
            }
            
            # Add testnet configuration if needed
            if self.is_testnet:
                options['urls'] = {
                    'api': 'https://test.deribit.com',
                }
            
            # Initialize client
            self.client = ccxtpro.deribit(options)
            
            # Load markets
            await self.client.load_markets()
            
            # Cache instrument information
            self._cache_instruments_info()
            
            # Test authentication by fetching account information
            await self.get_balance()
            
            logger.info(
                "Deribit gateway initialized successfully",
                testnet=self.is_testnet,
                options_instruments=len([i for i in self.client.markets.values() if i.get('type') == 'option']),
                futures_instruments=len([i for i in self.client.markets.values() if i.get('type') == 'future']),
                spot_instruments=len([i for i in self.client.markets.values() if i.get('type') == 'spot'])
            )
            
            self.initialized = True
            
        except ccxtpro.AuthenticationError as e:
            logger.error("Deribit authentication failed", exc_info=True)
            raise AuthenticationError(f"Deribit authentication failed: {str(e)}")
        except ccxtpro.NetworkError as e:
            logger.error("Deribit connection failed", exc_info=True)
            raise ConnectionError(f"Deribit connection failed: {str(e)}")
        except Exception as e:
            logger.error("Deribit gateway initialization failed", exc_info=True)
            raise GatewayError(f"Deribit gateway initialization failed: {str(e)}")
    
    def _cache_instruments_info(self) -> None:
        """Cache instrument information from loaded markets."""
        for symbol, market in self.client.markets.items():
            instrument_type = market.get('type', '')
            
            self.instruments_info[symbol] = {
                'type': instrument_type,
                'base': market.get('base', ''),
                'quote': market.get('quote', ''),
                'precision': market.get('precision', {}),
                'limits': market.get('limits', {}),
                'info': market.get('info', {}),
            }
            
            # Add option-specific information
            if instrument_type == 'option':
                option_info = market.get('info', {})
                self.instruments_info[symbol].update({
                    'strike': Decimal(str(option_info.get('strike', 0))),
                    'option_type': option_info.get('option_type', '').lower(),  # 'call' or 'put'
                    'expiry': market.get('expiry'),
                    'settlement_currency': option_info.get('settlement_currency', ''),
                })
    
    async def _handle_ccxt_error(self, e: Exception) -> None:
        """
        Handle CCXT errors and convert them to gateway-specific exceptions.
        
        Args:
            e: CCXT exception
            
        Raises:
            Appropriate gateway-specific exception
        """
        if isinstance(e, ccxtpro.AuthenticationError):
            raise AuthenticationError(f"Deribit authentication error: {str(e)}")
        elif isinstance(e, ccxtpro.InsufficientFunds):
            raise InsufficientFundsError(f"Insufficient funds: {str(e)}")
        elif isinstance(e, ccxtpro.RateLimitExceeded):
            raise RateLimitError(f"Rate limit exceeded: {str(e)}")
        elif isinstance(e, ccxtpro.NetworkError):
            raise ConnectionError(f"Network error: {str(e)}")
        elif isinstance(e, ccxtpro.ExchangeError):
            if "Order not found" in str(e):
                raise OrderError(f"Order not found: {str(e)}")
            else:
                raise OrderError(f"Exchange error: {str(e)}")
        else:
            raise GatewayError(f"Unexpected error: {str(e)}")
    
    @retry(
        retry=retry_if_exception_type((ccxtpro.NetworkError, ccxtpro.ExchangeNotAvailable)),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def _execute_with_retry(self, method: str, *args, **kwargs) -> Any:
        """
        Execute a CCXT method with retry logic for transient errors.
        
        Args:
            method: Name of the CCXT method to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Result of the method call
            
        Raises:
            GatewayError: If all retry attempts fail
        """
        if not self.client:
            raise GatewayError("Deribit gateway not initialized")
            
        try:
            # Get the method from the client
            func = getattr(self.client, method)
            
            # Execute the method
            result = await func(*args, **kwargs)
            
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
                exc_info=True
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
        if method.startswith('fetch'):
            self.rate_limit_counter += 1
        else:
            self.rate_limit_counter += 5  # Higher weight for mutations
        
        # Check if we're approaching the rate limit
        if self.rate_limit_counter > self.rate_limit_max * self.RATE_LIMIT_WEIGHT_THRESHOLD:
            logger.warning(
                "Approaching Deribit rate limit",
                current=self.rate_limit_counter,
                max=self.rate_limit_max,
                threshold_percent=self.RATE_LIMIT_WEIGHT_THRESHOLD * 100
            )
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """
        Convert internal OrderType to Deribit order type.
        
        Args:
            order_type: Internal order type
            
        Returns:
            Deribit order type string
        """
        order_type_map = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_MARKET: 'stop_market',
            OrderType.STOP_LIMIT: 'stop_limit',
            OrderType.TAKE_PROFIT_MARKET: 'take_profit',
            OrderType.TAKE_PROFIT_LIMIT: 'take_profit',
        }
        return order_type_map.get(order_type, 'market')
    
    def _convert_order_side(self, side: OrderSide) -> str:
        """
        Convert internal OrderSide to Deribit order side.
        
        Args:
            side: Internal order side
            
        Returns:
            Deribit order side string
        """
        return side.value.lower()
    
    def _convert_position_side(self, side: str) -> PositionSide:
        """
        Convert Deribit position side to internal PositionSide.
        
        Args:
            side: Deribit position side
            
        Returns:
            Internal position side enum
        """
        side_lower = side.lower()
        if side_lower in ('long', 'buy'):
            return PositionSide.LONG
        elif side_lower in ('short', 'sell'):
            return PositionSide.SHORT
        else:
            raise ValueError(f"Unknown position side: {side}")
    
    def _convert_order_status(self, status: str) -> OrderStatus:
        """
        Convert Deribit order status to internal OrderStatus.
        
        Args:
            status: Deribit order status
            
        Returns:
            Internal order status enum
        """
        status_map = {
            'open': OrderStatus.OPEN,
            'filled': OrderStatus.FILLED,
            'rejected': OrderStatus.REJECTED,
            'cancelled': OrderStatus.CANCELED,
            'untriggered': OrderStatus.PENDING,
            'triggered': OrderStatus.OPEN,
            'archive': OrderStatus.FILLED,
        }
        return status_map.get(status.lower(), OrderStatus.PENDING)
    
    def _convert_ccxt_order_to_internal(self, ccxt_order: Dict[str, Any]) -> Order:
        """
        Convert CCXT order object to internal Order model.
        
        Args:
            ccxt_order: CCXT order object
            
        Returns:
            Internal Order model
        """
        # Extract order type
        order_type_str = ccxt_order.get('type', 'market').lower()
        if 'stop' in order_type_str and 'limit' in order_type_str:
            order_type = OrderType.STOP_LIMIT
        elif 'stop' in order_type_str:
            order_type = OrderType.STOP_MARKET
        elif 'take_profit' in order_type_str and 'limit' in order_type_str:
            order_type = OrderType.TAKE_PROFIT_LIMIT
        elif 'take_profit' in order_type_str:
            order_type = OrderType.TAKE_PROFIT_MARKET
        elif 'limit' in order_type_str:
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.MARKET
        
        # Extract order side
        side_str = ccxt_order.get('side', '').lower()
        side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
        
        # Extract timestamps
        created_at = timestamp_to_datetime(ccxt_order.get('timestamp', time.time() * 1000) / 1000)
        updated_at = timestamp_to_datetime(ccxt_order.get('lastUpdateTimestamp', time.time() * 1000) / 1000)
        
        # Create internal order object
        return Order(
            exchange_id=str(ccxt_order.get('id', '')),
            exchange='deribit',
            symbol=ccxt_order.get('symbol', ''),
            order_type=order_type,
            side=side,
            price=Decimal(str(ccxt_order.get('price', 0))),
            amount=Decimal(str(ccxt_order.get('amount', 0))),
            filled_amount=Decimal(str(ccxt_order.get('filled', 0))),
            status=self._convert_order_status(ccxt_order.get('status', '')),
            created_at=created_at,
            updated_at=updated_at,
        )
    
    def _convert_option_type(self, option_type: str) -> OptionType:
        """
        Convert Deribit option type to internal OptionType.
        
        Args:
            option_type: Deribit option type ('call' or 'put')
            
        Returns:
            Internal option type enum
        """
        option_type_lower = option_type.lower()
        if option_type_lower == 'call':
            return OptionType.CALL
        elif option_type_lower == 'put':
            return OptionType.PUT
        else:
            raise ValueError(f"Unknown option type: {option_type}")
    
    def _parse_option_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Parse a Deribit option symbol to extract its components.
        
        Example: 'BTC-24JUN22-30000-C' -> 
            {'underlying': 'BTC', 'expiry': '24JUN22', 'strike': 30000, 'type': 'call'}
        
        Args:
            symbol: Deribit option symbol
            
        Returns:
            Dictionary with option components
        """
        # Check if we already have this in the instruments info
        if symbol in self.instruments_info and self.instruments_info[symbol]['type'] == 'option':
            info = self.instruments_info[symbol]
            return {
                'underlying': info['base'],
                'expiry': info.get('expiry'),
                'strike': info.get('strike'),
                'type': info.get('option_type')
            }
        
        # Otherwise, parse the symbol
        try:
            parts = symbol.split('-')
            if len(parts) != 4:
                raise ValueError(f"Invalid option symbol format: {symbol}")
            
            underlying = parts[0]
            expiry = parts[1]
            strike = Decimal(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            
            return {
                'underlying': underlying,
                'expiry': expiry,
                'strike': strike,
                'type': option_type
            }
        except Exception as e:
            logger.error(f"Error parsing option symbol: {symbol}", exc_info=True)
            raise ValueError(f"Error parsing option symbol: {symbol}")
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair or contract symbol
            
        Returns:
            Dictionary with ticker information
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            ticker = await self._execute_with_retry('fetch_ticker', symbol)
            
            # Cache the ticker for later use
            self.last_ticker_cache[symbol] = ticker
            
            return {
                'symbol': symbol,
                'last': Decimal(str(ticker['last'])) if ticker['last'] else None,
                'bid': Decimal(str(ticker['bid'])) if ticker['bid'] else None,
                'ask': Decimal(str(ticker['ask'])) if ticker['ask'] else None,
                'high': Decimal(str(ticker['high'])) if ticker['high'] else None,
                'low': Decimal(str(ticker['low'])) if ticker['low'] else None,
                'volume': Decimal(str(ticker['volume'])) if ticker['volume'] else None,
                'timestamp': timestamp_to_datetime(ticker['timestamp'] / 1000),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get current orderbook for a symbol.
        
        Args:
            symbol: Trading pair or contract symbol
            limit: Number of price levels to retrieve
            
        Returns:
            Dictionary with orderbook information
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            orderbook = await self._execute_with_retry('fetch_order_book', symbol, limit)
            
            # Convert to Decimal for precision
            bids = [[Decimal(str(price)), Decimal(str(amount))] for price, amount in orderbook['bids']]
            asks = [[Decimal(str(price)), Decimal(str(amount))] for price, amount in orderbook['asks']]
            
            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': timestamp_to_datetime(orderbook['timestamp'] / 1000),
            }
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def get_balance(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get account balances for all assets.
        
        Returns:
            Dictionary mapping asset names to balance information
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            balance = await self._execute_with_retry('fetch_balance')
            
            result = {}
            for currency, data in balance.items():
                if currency not in ('info', 'timestamp', 'datetime', 'free', 'used', 'total'):
                    result[currency] = {
                        'free': Decimal(str(data.get('free', 0))),
                        'used': Decimal(str(data.get('used', 0))),
                        'total': Decimal(str(data.get('total', 0))),
                    }
            
            return result
        except Exception as e:
            logger.error("Error fetching balances", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Deribit doesn't have a direct fetch_positions method in CCXT
            # We need to use a custom endpoint
            positions = await self._execute_with_retry('privateGetGetPositions', {'currency': 'BTC'})
            
            result = []
            for pos_data in positions.get('result', []):
                # Skip positions with zero size
                if float(pos_data.get('size', 0)) == 0:
                    continue
                
                # Create position object
                position = Position(
                    strategy_id=None,  # Will be filled by the strategy
                    exchange='deribit',
                    symbol=pos_data.get('instrument_name', ''),
                    side=PositionSide.LONG if pos_data.get('direction', '') == 'buy' else PositionSide.SHORT,
                    entry_price=Decimal(str(pos_data.get('average_price', 0))),
                    current_price=Decimal(str(pos_data.get('mark_price', 0))),
                    size=Decimal(str(abs(float(pos_data.get('size', 0))))),
                    leverage=Decimal(str(pos_data.get('leverage', 1))),
                    liquidation_price=Decimal(str(pos_data.get('estimated_liquidation_price', 0))),
                    unrealized_pnl=Decimal(str(pos_data.get('floating_profit_loss', 0))),
                    status=PositionStatus.OPEN,
                    entry_time=timestamp_to_datetime(pos_data.get('creation_timestamp', 0) / 1000),
                )
                
                result.append(position)
            
            return result
        except Exception as e:
            logger.error("Error fetching positions", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def get_margin_status(self) -> MarginStatus:
        """
        Get current margin status.
        
        Returns:
            MarginStatus object with current margin information
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get account summary
            account_summary = await self._execute_with_retry(
                'privateGetGetAccountSummary', 
                {'currency': 'BTC'}
            )
            
            summary = account_summary.get('result', {})
            
            # Extract margin information
            wallet_balance = Decimal(str(summary.get('equity', 0)))
            unrealized_pnl = Decimal(str(summary.get('delta_total', 0)))
            maintenance_margin = Decimal(str(summary.get('maintenance_margin', 0)))
            initial_margin = Decimal(str(summary.get('initial_margin', 0)))
            
            # Calculate margin ratio
            margin_ratio = Decimal("999.99")  # Default high value
            if maintenance_margin > Decimal("0"):
                margin_ratio = wallet_balance / maintenance_margin
            
            # Determine margin level
            margin_level = MarginLevel.SAFE
            if margin_ratio < Decimal("1.0"):
                margin_level = MarginLevel.LIQUIDATION
            elif margin_ratio < Decimal("4.0"):
                margin_level = MarginLevel.CRITICAL
            elif margin_ratio < Decimal("4.5"):
                margin_level = MarginLevel.WARNING
            
            return MarginStatus(
                exchange='deribit',
                account_type='options',
                wallet_balance=wallet_balance,
                unrealized_pnl=unrealized_pnl,
                maintenance_margin=maintenance_margin,
                initial_margin=initial_margin,
                margin_ratio=margin_ratio,
                margin_level=margin_level,
                timestamp=datetime.now(timezone.utc),
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
        price: Optional[Decimal] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair or contract symbol
            order_type: Type of order
            side: Order side (buy/sell)
            amount: Order quantity
            price: Order price (required for limit orders)
            params: Additional parameters specific to Deribit
            
        Returns:
            Order object with order information
            
        Raises:
            OrderError: If order creation fails
        """
        if not self.initialized:
            await self.initialize()
        
        params = params or {}
        
        # Convert to CCXT format
        ccxt_order_type = self._convert_order_type(order_type)
        ccxt_side = self._convert_order_side(side)
        
        try:
            # Create the order
            ccxt_order = await self._execute_with_retry(
                'create_order',
                symbol,
                ccxt_order_type,
                ccxt_side,
                float(amount),
                float(price) if price else None,
                params
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
                price=str(price) if price else None
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
                exc_info=True
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
        if not self.initialized:
            await self.initialize()
        
        try:
            # Cancel the order
            result = await self._execute_with_retry('cancel_order', order_id, symbol)
            
            logger.info(
                f"Cancelled order: {order_id}",
                order_id=order_id,
                symbol=symbol,
                result=result
            )
            
            return True
        except Exception as e:
            if "Order not found" in str(e):
                logger.warning(
                    f"Order not found when cancelling: {order_id}",
                    order_id=order_id,
                    symbol=symbol
                )
                return False
            
            logger.error(
                f"Error cancelling order: {order_id}",
                order_id=order_id,
                symbol=symbol,
                exc_info=True
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
        if not self.initialized:
            await self.initialize()
        
        try:
            # Fetch the order
            ccxt_order = await self._execute_with_retry('fetch_order', order_id, symbol)
            
            # Convert to internal format
            return self._convert_ccxt_order_to_internal(ccxt_order)
        except Exception as e:
            logger.error(
                f"Error fetching order: {order_id}",
                order_id=order_id,
                symbol=symbol,
                exc_info=True
            )
            await self._handle_ccxt_error(e)
    
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a perpetual contract.
        
        Args:
            symbol: Contract symbol
            
        Returns:
            Dictionary with funding rate information
        """
        if not self.initialized:
            await self.initialize()
        
        # Deribit doesn't have traditional funding rates like Binance
        # For perpetual contracts, it uses a different mechanism
        # This is a placeholder implementation
        return {
            'symbol': symbol,
            'funding_rate': Decimal("0"),
            'funding_time': datetime.now(timezone.utc) + timedelta(hours=8),
            'mark_price': Decimal("0"),
            'index_price': Decimal("0"),
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
        if not self.initialized:
            await self.initialize()
        
        # Placeholder implementation - Deribit doesn't have the same futures structure as Binance
        return Decimal("0")
    
    async def get_option_chain(
        self, 
        underlying: str = 'BTC', 
        expiry_range_days: Tuple[int, int] = (0, 180)
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the complete option chain for an underlying asset.
        
        Args:
            underlying: Underlying asset symbol (e.g., 'BTC')
            expiry_range_days: Tuple of (min_days, max_days) to filter by expiry
            
        Returns:
            Dictionary mapping expiry dates to lists of option instruments
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Calculate date range
            now = datetime.now(timezone.utc)
            min_date = now + timedelta(days=expiry_range_days[0])
            max_date = now + timedelta(days=expiry_range_days[1])
            
            # Get all instruments
            instruments = await self._execute_with_retry(
                'publicGetGetInstruments',
                {
                    'currency': underlying,
                    'kind': 'option',
                    'expired': False
                }
            )
            
            result = {}
            for instrument in instruments.get('result', []):
                # Parse expiry date
                expiry_timestamp = instrument.get('expiration_timestamp', 0) / 1000
                expiry_date = timestamp_to_datetime(expiry_timestamp)
                
                # Skip if outside date range
                if expiry_date < min_date or expiry_date > max_date:
                    continue
                
                # Format expiry key
                expiry_key = expiry_date.strftime('%Y-%m-%d')
                
                # Initialize list for this expiry if not exists
                if expiry_key not in result:
                    result[expiry_key] = []
                
                # Add instrument data
                option_data = {
                    'symbol': instrument.get('instrument_name', ''),
                    'underlying': instrument.get('base_currency', ''),
                    'expiry_date': expiry_date,
                    'strike': Decimal(str(instrument.get('strike', 0))),
                    'option_type': instrument.get('option_type', '').lower(),
                    'settlement_currency': instrument.get('settlement_currency', ''),
                    'contract_size': Decimal(str(instrument.get('contract_size', 1))),
                    'is_active': instrument.get('is_active', False),
                    'creation_date': timestamp_to_datetime(instrument.get('creation_timestamp', 0) / 1000),
                }
                
                result[expiry_key].append(option_data)
            
            # Cache the option chain
            self.option_chains[underlying] = result
            
            return result
        except Exception as e:
            logger.error(f"Error fetching option chain for {underlying}", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def get_option_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed market data for a specific option.
        
        Args:
            symbol: Option symbol (e.g., 'BTC-24JUN22-30000-C')
            
        Returns:
            Dictionary with option market data including Greeks
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get ticker data
            ticker = await self.get_ticker(symbol)
            
            # Get option details
            option_details = self._parse_option_symbol(symbol)
            
            # Get underlying price
            underlying_ticker = await self.get_ticker(f"{option_details['underlying']}-PERPETUAL")
            underlying_price = underlying_ticker['last']
            
            # Calculate time to expiry in years
            now = datetime.now(timezone.utc)
            expiry_date = datetime.strptime(option_details['expiry'], '%d%b%y').replace(tzinfo=timezone.utc)
            time_to_expiry_years = (expiry_date - now).total_seconds() / (365.25 * 24 * 60 * 60)
            
            # Get option Greeks from Deribit
            # In a real implementation, we would fetch this from the API
            # For now, we'll calculate them using our utility functions
            strike_price = option_details['strike']
            option_type = option_details['type']
            
            # Placeholder values for volatility and risk-free rate
            volatility = Decimal("0.7")  # 70% implied volatility
            risk_free_rate = Decimal("0.03")  # 3% risk-free rate
            
            # Calculate Greeks
            greeks = calculate_option_greeks(
                underlying_price,
                strike_price,
                Decimal(str(time_to_expiry_years)),
                risk_free_rate,
                volatility,
                option_type
            )
            
            # Combine all data
            return {
                'symbol': symbol,
                'underlying': option_details['underlying'],
                'underlying_price': underlying_price,
                'strike': strike_price,
                'expiry_date': expiry_date,
                'time_to_expiry_years': Decimal(str(time_to_expiry_years)),
                'option_type': option_type,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['volume'],
                'open_interest': Decimal("0"),  # Placeholder
                'implied_volatility': volatility,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho'],
                'timestamp': ticker['timestamp'],
            }
        except Exception as e:
            logger.error(f"Error fetching option market data for {symbol}", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def get_option_settlement_history(
        self, 
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get option settlement history.
        
        Args:
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            limit: Maximum number of records to retrieve
            
        Returns:
            List of settlement records
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Set default timestamps if not provided
            if end_timestamp is None:
                end_timestamp = int(time.time() * 1000)
            if start_timestamp is None:
                start_timestamp = end_timestamp - (30 * 24 * 60 * 60 * 1000)  # 30 days
            
            # Get settlement history
            settlements = await self._execute_with_retry(
                'privateGetGetSettlementHistoryByInstrument',
                {
                    'currency': 'BTC',
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                    'count': limit
                }
            )
            
            result = []
            for settlement in settlements.get('result', []):
                # Process each settlement record
                record = {
                    'symbol': settlement.get('instrument_name', ''),
                    'settlement_price': Decimal(str(settlement.get('settlement_price', 0))),
                    'timestamp': timestamp_to_datetime(settlement.get('timestamp', 0) / 1000),
                    'profit_loss': Decimal(str(settlement.get('profit_loss', 0))),
                    'index_price': Decimal(str(settlement.get('index_price', 0))),
                    'mark_price': Decimal(str(settlement.get('mark_price', 0))),
                    'type': settlement.get('type', ''),
                }
                
                result.append(record)
            
            return result
        except Exception as e:
            logger.error("Error fetching option settlement history", exc_info=True)
            await self._handle_ccxt_error(e)
    
    async def close(self) -> None:
        """
        Close the gateway connection and release resources.
        """
        if self.client:
            try:
                await self.client.close()
                logger.info("Deribit gateway closed")
            except Exception as e:
                logger.error("Error closing Deribit gateway", exc_info=True)
                # Don't re-raise, just log the error
