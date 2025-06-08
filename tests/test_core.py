"""
Unit tests for the BTC Stack-Builder Bot core module.

This module contains tests for the core components, including:
- Constants validation
- Utility functions
- Data models and validation
"""

import uuid
from datetime import UTC, datetime, timezone
from decimal import Decimal

import pytest

from btc_stack_builder.core.constants import (
    BASIS_ENTRY_THRESHOLD,
    BASIS_HARVEST_ALLOCATION,
    CORE_HODL_ALLOCATION,
    FUNDING_CAPTURE_ALLOCATION,
    FUNDING_ENTRY_THRESHOLD,
    GLOBAL_STOP_LOSS_THRESHOLD,
    MARGIN_RATIO_CRITICAL_THRESHOLD,
    MARGIN_RATIO_WARNING_THRESHOLD,
    OPTION_DELTA_TARGET,
    OPTION_PREMIUM_ALLOCATION,
)
from btc_stack_builder.core.models import (
    ExchangeCredentials,
    Option,
    OptionStatus,
    OptionType,
    Order,
    OrderSide,
    OrderType,
    PortfolioAllocation,
    Position,
    PositionSide,
    PositionStatus,
)
from btc_stack_builder.core.utils import (
    btc_to_satoshi,
    calculate_annualized_basis,
    calculate_funding_rate,
    calculate_margin_ratio,
    calculate_position_pnl,
    datetime_to_timestamp,
    format_btc_amount,
    satoshi_to_btc,
    timestamp_to_datetime,
)


class TestConstants:
    """Tests for core constants."""

    def test_portfolio_allocation_sums_to_one(self):
        """Test that portfolio allocations sum to 1.0 (100%)."""
        total_allocation = (
            CORE_HODL_ALLOCATION
            + BASIS_HARVEST_ALLOCATION
            + FUNDING_CAPTURE_ALLOCATION
            + OPTION_PREMIUM_ALLOCATION
        )
        assert total_allocation == Decimal("1.0"), "Portfolio allocations must sum to 1.0"

    def test_margin_thresholds(self):
        """Test that margin thresholds are properly ordered."""
        assert (
            MARGIN_RATIO_WARNING_THRESHOLD > MARGIN_RATIO_CRITICAL_THRESHOLD
        ), "Warning threshold must be higher than critical threshold"

    def test_basis_entry_threshold_positive(self):
        """Test that basis entry threshold is positive."""
        assert BASIS_ENTRY_THRESHOLD > Decimal("0"), "Basis entry threshold must be positive"

    def test_funding_entry_threshold_negative(self):
        """Test that funding entry threshold is negative."""
        assert FUNDING_ENTRY_THRESHOLD < Decimal("0"), "Funding entry threshold must be negative"

    def test_option_delta_target_range(self):
        """Test that option delta target is within valid range."""
        assert (
            Decimal("0") < OPTION_DELTA_TARGET < Decimal("1")
        ), "Option delta target must be between 0 and 1"

    def test_global_stop_loss_threshold(self):
        """Test that global stop loss threshold is negative."""
        assert GLOBAL_STOP_LOSS_THRESHOLD < Decimal(
            "0"
        ), "Global stop loss threshold must be negative"


class TestUtils:
    """Tests for utility functions."""

    def test_calculate_annualized_basis(self):
        """Test annualized basis calculation."""
        # Test case: 5% basis over 90 days
        futures_price = Decimal("52500")
        spot_price = Decimal("50000")
        days_to_expiry = 90

        expected_basis = (Decimal("52500") / Decimal("50000") - 1) * (
            Decimal("365") / Decimal("90")
        )
        calculated_basis = calculate_annualized_basis(futures_price, spot_price, days_to_expiry)

        assert abs(calculated_basis - expected_basis) < Decimal(
            "0.001"
        ), f"Expected {expected_basis}, got {calculated_basis}"

    def test_calculate_funding_rate(self):
        """Test funding rate calculation."""
        mark_price = Decimal("51000")
        index_price = Decimal("50000")

        # Expected funding rate: (51000 - 50000) / 50000 = 0.02 (2%)
        expected_rate = Decimal("0.02")
        calculated_rate = calculate_funding_rate(mark_price, index_price)

        assert abs(calculated_rate - expected_rate) < Decimal(
            "0.001"
        ), f"Expected {expected_rate}, got {calculated_rate}"

    def test_calculate_margin_ratio(self):
        """Test margin ratio calculation."""
        wallet_balance = Decimal("10000")
        maintenance_margin = Decimal("2000")

        # Expected ratio: 10000 / 2000 = 5.0 (500%)
        expected_ratio = Decimal("5.0")
        calculated_ratio = calculate_margin_ratio(wallet_balance, maintenance_margin)

        assert (
            calculated_ratio == expected_ratio
        ), f"Expected {expected_ratio}, got {calculated_ratio}"

    def test_calculate_position_pnl(self):
        """Test position PnL calculation."""
        entry_price = Decimal("50000")
        current_price = Decimal("55000")
        position_size = Decimal("1.0")
        position_side = "long"

        # Expected PnL: (55000 - 50000) * 1.0 = 5000
        result = calculate_position_pnl(entry_price, current_price, position_size, position_side)

        assert result["absolute_pnl"] == Decimal(
            "5000"
        ), f"Expected 5000, got {result['absolute_pnl']}"
        assert result["percentage_pnl"] == Decimal(
            "0.1"
        ), f"Expected 0.1 (10%), got {result['percentage_pnl']}"

    def test_satoshi_to_btc_conversion(self):
        """Test satoshi to BTC conversion."""
        satoshis = 123456789
        expected_btc = Decimal("1.23456789")

        assert (
            satoshi_to_btc(satoshis) == expected_btc
        ), f"Expected {expected_btc}, got {satoshi_to_btc(satoshis)}"

    def test_btc_to_satoshi_conversion(self):
        """Test BTC to satoshi conversion."""
        btc = Decimal("1.23456789")
        expected_satoshis = 123456789

        assert (
            btc_to_satoshi(btc) == expected_satoshis
        ), f"Expected {expected_satoshis}, got {btc_to_satoshi(btc)}"

    def test_timestamp_to_datetime(self):
        """Test timestamp to datetime conversion."""
        timestamp = 1654012800  # 2022-05-31 12:00:00 UTC
        print(f"Input timestamp: {timestamp}")
        dt = timestamp_to_datetime(timestamp)
        print(f"Converted datetime: {repr(dt)}, hour: {dt.hour}, tzinfo: {repr(dt.tzinfo)}")
        # Assertions as before:
        assert dt.year == 2022
        assert dt.month == 5
        assert dt.day == 31
        assert dt.hour == 12, f"Hour was {dt.hour}, expected 12. Full datetime: {repr(dt)}"
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo == UTC, f"tzinfo was {repr(dt.tzinfo)}, expected {repr(UTC)}"

    def test_datetime_to_timestamp(self):
        """Test datetime to timestamp conversion."""
        dt_input = datetime(2022, 5, 31, 12, 0, 0, tzinfo=UTC)
        expected_timestamp = 1654012800
        print(f"Input datetime: {repr(dt_input)}")
        calculated_timestamp = datetime_to_timestamp(dt_input)
        print(f"Calculated timestamp: {calculated_timestamp}, Expected: {expected_timestamp}")
        assert (
            calculated_timestamp == expected_timestamp
        ), f"Expected {expected_timestamp}, got {calculated_timestamp}. Input datetime was {repr(dt_input)}"

    def test_format_btc_amount(self):
        """Test BTC amount formatting."""
        amount = Decimal("1.23456789")

        # Test with default precision (8) and symbol
        formatted = format_btc_amount(amount)
        assert formatted == "1.23456789 BTC", f"Expected '1.23456789 BTC', got '{formatted}'"

        # Test with custom precision (4) and no symbol
        formatted = format_btc_amount(amount, precision=4, include_symbol=False)
        assert formatted == "1.2346", f"Expected '1.2346', got '{formatted}'"


class TestModels:
    """Tests for data models."""

    def test_exchange_credentials_model(self):
        """Test ExchangeCredentials model validation."""
        # Valid credentials
        creds = ExchangeCredentials(
            api_key="test_key",
            api_secret="test_secret",  # noqa: S105,S106
            is_testnet=True,
        )
        assert creds.api_key == "test_key"
        assert creds.api_secret == "test_secret"
        assert creds.is_testnet is True

    def test_portfolio_allocation_validation(self):
        """Test PortfolioAllocation model validation."""
        # Valid allocation
        allocation = PortfolioAllocation(
            core_hodl=Decimal("0.6"),
            basis_harvest=Decimal("0.25"),
            funding_capture=Decimal("0.1"),
            option_premium=Decimal("0.05"),
        )
        assert allocation.core_hodl == Decimal("0.6")
        assert allocation.basis_harvest == Decimal("0.25")
        assert allocation.funding_capture == Decimal("0.1")
        assert allocation.option_premium == Decimal("0.05")

        # Invalid allocation (sum > 1.0)
        with pytest.raises(ValueError):
            PortfolioAllocation(
                core_hodl=Decimal("0.7"),
                basis_harvest=Decimal("0.25"),
                funding_capture=Decimal("0.1"),
                option_premium=Decimal("0.05"),
            )

    def test_order_model(self):
        """Test Order model validation."""
        # Valid market order
        market_order = Order(
            exchange_id="123456",
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Decimal("0.1"),
        )
        assert market_order.exchange_id == "123456"
        assert market_order.order_type == OrderType.MARKET
        assert market_order.side == OrderSide.BUY
        assert market_order.amount == Decimal("0.1")
        assert market_order.price is None  # Price not required for market orders

        # Valid limit order
        limit_order = Order(
            exchange_id="123457",
            exchange="binance",
            symbol="BTC/USDT",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
        )
        assert limit_order.price == Decimal("50000")

        # Invalid limit order (missing price)
        with pytest.raises(ValueError):
            Order(
                exchange_id="123458",
                exchange="binance",
                symbol="BTC/USDT",
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL,
                amount=Decimal("0.1"),
            )

    def test_position_model(self):
        """Test Position model validation."""
        # Valid position
        position = Position(
            strategy_id=str(uuid.uuid4()),
            exchange="binance",
            symbol="BTCUSD_PERP",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            size=Decimal("1.0"),
            leverage=Decimal("2.0"),
        )
        assert position.side == PositionSide.LONG
        assert position.entry_price == Decimal("50000")
        assert position.current_price == Decimal("51000")
        assert position.size == Decimal("1.0")
        assert position.leverage == Decimal("2.0")
        assert position.status == PositionStatus.OPEN

        # Test PnL calculation
        assert position.pnl_percentage == Decimal("0.02")  # (51000 - 50000) / 50000 = 0.02 (2%)

    def test_option_model(self):
        """Test Option model validation."""
        # Valid option
        expiry_date = datetime(2023, 6, 30, 16, 0, 0, tzinfo=UTC)
        option = Option(
            strategy_id=str(uuid.uuid4()),
            exchange="deribit",
            underlying="BTC",
            strike_price=Decimal("50000"),
            expiry_date=expiry_date,
            option_type=OptionType.PUT,
            side="sell",
            size=Decimal("1.0"),
            premium=Decimal("2500"),
            total_premium=Decimal("2500"),
            collateral=Decimal("50000"),
        )
        assert option.option_type == OptionType.PUT
        assert option.side == "sell"
        assert option.strike_price == Decimal("50000")
        assert option.premium == Decimal("2500")
        assert option.status == OptionStatus.OPEN

        # Test days to expiry calculation
        days_to_expiry = option.days_to_expiry
        assert isinstance(days_to_expiry, float)
        assert days_to_expiry >= 0


if __name__ == "__main__":
    pytest.main()
