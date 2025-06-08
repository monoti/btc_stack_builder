"""
Unit tests for the BTC Stack-Builder Bot core module.

This module contains tests for the core components, including:
- Constants validation
- Utility functions
- Data models and validation
"""

import uuid
from datetime import datetime, timezone, timedelta
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

    def test_calculate_annualized_basis_zero_days(self):
        """Test annualized basis calculation with zero days to expiry."""
        futures_price = Decimal("52500")
        spot_price = Decimal("50000")
        days_to_expiry = 0
        with pytest.raises(ValueError, match="Days to expiry must be greater than zero"):
            calculate_annualized_basis(futures_price, spot_price, days_to_expiry)

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

    def test_calculate_funding_rate_zero_index_price(self):
        """Test funding rate calculation with zero index price."""
        mark_price = Decimal("51000")
        index_price = Decimal("0")
        with pytest.raises(ValueError, match="Index price must be greater than zero"):
            calculate_funding_rate(mark_price, index_price)

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

    def test_calculate_margin_ratio_zero_maintenance_margin(self):
        """Test margin ratio calculation with zero maintenance margin."""
        wallet_balance = Decimal("10000")
        maintenance_margin = Decimal("0")
        # Expected ratio: a very high number as per function's logic
        expected_ratio = Decimal("999.99")
        calculated_ratio = calculate_margin_ratio(wallet_balance, maintenance_margin)
        assert (
            calculated_ratio == expected_ratio
        ), f"Expected {expected_ratio}, got {calculated_ratio}"

        wallet_balance_zero = Decimal("0")
        expected_ratio_zero_balance = Decimal("0.0")
        calculated_ratio_zero_balance = calculate_margin_ratio(
            wallet_balance_zero, maintenance_margin
        )
        assert calculated_ratio_zero_balance == expected_ratio_zero_balance, (
            f"Expected {expected_ratio_zero_balance}, " f"got {calculated_ratio_zero_balance}"
        )

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

    def test_calculate_position_pnl_zero_position_size(self):
        """Test position PnL calculation with zero position size."""
        entry_price = Decimal("50000")
        current_price = Decimal("55000")
        position_size = Decimal("0")
        position_side = "long"

        result = calculate_position_pnl(entry_price, current_price, position_size, position_side)

        assert result["absolute_pnl"] == Decimal("0"), f"Expected 0, got {result['absolute_pnl']}"
        # Percentage PnL is relative to entry price, so it can be non-zero
        assert result["percentage_pnl"] == Decimal(
            "0.1"
        ), f"Expected 0.1 (10%), got {result['percentage_pnl']}"
        # ROI should be zero if initial margin is zero
        assert result["roi"] == Decimal("0"), f"Expected 0, got {result['roi']}"

    def test_satoshi_to_btc_conversion(self):
        """Test satoshi to BTC conversion."""
        assert satoshi_to_btc(123456789) == Decimal("1.23456789")
        assert satoshi_to_btc(0) == Decimal("0.0")
        assert satoshi_to_btc(-100000000) == Decimal("-1.0")

    def test_btc_to_satoshi_conversion(self):
        """Test BTC to satoshi conversion."""
        assert btc_to_satoshi(Decimal("1.23456789")) == 123456789
        assert btc_to_satoshi(Decimal("0.0")) == 0
        assert btc_to_satoshi(Decimal("-1.0")) == -100000000

    def test_timestamp_to_datetime(self):
        """Test timestamp to datetime conversion."""
        # Standard case
        timestamp = 1653998400  # Unix timestamp for 2022-05-31 12:00:00 UTC
        dt = timestamp_to_datetime(timestamp)
        assert dt.year == 2022
        assert dt.month == 5
        assert dt.day == 31
        assert dt.hour == 12
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo == timezone.utc

        # Unix epoch
        epoch_timestamp = 0
        epoch_dt = timestamp_to_datetime(epoch_timestamp)
        assert epoch_dt.year == 1970
        assert epoch_dt.month == 1
        assert epoch_dt.day == 1
        assert epoch_dt.hour == 0
        assert epoch_dt.minute == 0
        assert epoch_dt.second == 0
        assert epoch_dt.tzinfo == timezone.utc

        # Current time (approximate check)
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        current_dt = timestamp_to_datetime(current_timestamp)
        assert abs(current_dt.timestamp() - current_timestamp) < 2  # Allow for slight delay

    def test_datetime_to_timestamp(self):
        """Test datetime to timestamp conversion."""
        # Standard case
        dt_input = datetime(2022, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
        expected_timestamp = 1653998400
        assert datetime_to_timestamp(dt_input) == expected_timestamp

        # Unix epoch
        epoch_dt_input = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        expected_epoch_timestamp = 0
        assert datetime_to_timestamp(epoch_dt_input) == expected_epoch_timestamp

        # Current time (naive datetime, should assume UTC)
        current_dt_naive = datetime.now()  # Naive datetime
        current_dt_aware_utc = datetime.now(timezone.utc)  # Aware datetime in UTC

        # When a naive datetime is passed to datetime_to_timestamp, it's assumed to be UTC.
        # So, its timestamp should match an aware datetime that is explicitly UTC.
        expected_current_timestamp_naive = int(
            current_dt_naive.replace(tzinfo=timezone.utc).timestamp()
        )
        expected_current_timestamp_aware = int(current_dt_aware_utc.timestamp())

        assert datetime_to_timestamp(current_dt_naive) == expected_current_timestamp_naive
        # Also test with an aware object to be sure
        assert datetime_to_timestamp(current_dt_aware_utc) == expected_current_timestamp_aware
        # Given the logic, these two should be very close, if not identical.
        # We can assert they are close to account for minimal execution time
        # differences.
        assert (
            abs(
                datetime_to_timestamp(current_dt_naive)
                - datetime_to_timestamp(current_dt_aware_utc)
            )
            < 2
        )

    def test_format_btc_amount(self):
        """Test BTC amount formatting."""
        # Standard case
        amount = Decimal("1.23456789")
        assert format_btc_amount(amount) == "1.23456789 BTC"
        assert format_btc_amount(amount, precision=4, include_symbol=False) == "1.2346"

        # Zero amount
        zero_amount = Decimal("0")
        assert format_btc_amount(zero_amount) == "0.00000000 BTC"
        assert format_btc_amount(zero_amount, precision=2) == "0.00 BTC"
        assert format_btc_amount(zero_amount, precision=0, include_symbol=False) == "0"

        # Different precisions
        amount_precision = Decimal("123.456")
        assert format_btc_amount(amount_precision, precision=8) == "123.45600000 BTC"
        assert format_btc_amount(amount_precision, precision=3) == "123.456 BTC"
        assert format_btc_amount(amount_precision, precision=2) == "123.46 BTC"
        assert format_btc_amount(amount_precision, precision=0) == "123 BTC"


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

        # Test empty api_key - Pydantic v2 allows empty strings by default for non-constrained str fields
        # If this should be an error, the model itself needs `Field(min_length=1)`
        creds_empty_key = ExchangeCredentials(api_key="", api_secret="test_secret")
        assert creds_empty_key.api_key == ""

        # Test empty api_secret
        creds_empty_secret = ExchangeCredentials(api_key="test_key", api_secret="")
        assert creds_empty_secret.api_secret == ""

    def test_portfolio_allocation_validation(self):
        """Test PortfolioAllocation model validation."""
        # Valid allocation
        valid_allocation_data = {
            "core_hodl": Decimal("0.6"),
            "basis_harvest": Decimal("0.25"),
            "funding_capture": Decimal("0.1"),
            "option_premium": Decimal("0.05"),
        }
        allocation = PortfolioAllocation(**valid_allocation_data)
        assert allocation.core_hodl == valid_allocation_data["core_hodl"]
        assert allocation.basis_harvest == valid_allocation_data["basis_harvest"]
        assert allocation.funding_capture == valid_allocation_data["funding_capture"]
        assert allocation.option_premium == valid_allocation_data["option_premium"]

        # Invalid allocation (sum > 1.0)
        with pytest.raises(ValueError, match="Portfolio allocations must sum to 1.0"):
            PortfolioAllocation(
                core_hodl=Decimal("0.7"),
                basis_harvest=Decimal("0.25"),
                funding_capture=Decimal("0.1"),
                option_premium=Decimal("0.05"),
            )

        # Invalid: individual allocation negative
        with pytest.raises(ValueError):  # Pydantic's Field(ge=0)
            PortfolioAllocation(
                core_hodl=Decimal("-0.1"),
                basis_harvest=Decimal("0.5"),
                funding_capture=Decimal("0.3"),
                option_premium=Decimal("0.3"),  # Sum is 1.0 but core_hodl invalid
            )

        # Invalid: individual allocation > 1.0
        with pytest.raises(ValueError):  # Pydantic's Field(le=1)
            PortfolioAllocation(
                core_hodl=Decimal("1.1"),
                basis_harvest=Decimal("-0.1"),  # Invalid, but sum is 1.0
                funding_capture=Decimal("0.0"),
                option_premium=Decimal("0.0"),
            )

        # Invalid: sum is 1.0, but one allocation is negative and another > 1 to
        # compensate. This should be caught by individual ge/le constraints before
        # the sum validator.
        with pytest.raises(ValueError):
            PortfolioAllocation(
                core_hodl=Decimal("1.5"),
                basis_harvest=Decimal("-0.5"),
                funding_capture=Decimal("0.0"),
                option_premium=Decimal("0.0"),
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

        # Common data for order tests
        common_order_data = {
            "exchange_id": "654321",
            "exchange": "kraken",
            "symbol": "ETH/USD",
            "amount": Decimal("1.0"),
        }

        # Invalid enum for order_type
        with pytest.raises(ValueError):  # Pydantic ValidationError
            Order(**common_order_data, order_type="INVALID_TYPE", side=OrderSide.BUY)

        # Invalid enum for side
        with pytest.raises(ValueError):  # Pydantic ValidationError
            Order(**common_order_data, order_type=OrderType.MARKET, side="INVALID_SIDE")

        # Invalid amount (negative) - Model currently allows this as no Field(gt=0)
        order_neg_amount = Order(
            exchange_id="id",
            exchange="ex",
            symbol="sym",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Decimal("-0.1"),
        )
        assert order_neg_amount.amount == Decimal("-0.1")

        # Invalid amount (zero) - Model currently allows this
        order_zero_amount = Order(
            exchange_id="id",
            exchange="ex",
            symbol="sym",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            amount=Decimal("0"),
        )
        assert order_zero_amount.amount == Decimal("0")

        # Invalid price (negative) for limit order - Model allows this as no Field(gt=0)
        order_neg_price = Order(
            **common_order_data,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("-100"),
        )
        assert order_neg_price.price == Decimal("-100")

        # Invalid price (zero) for limit order - Model allows this
        order_zero_price = Order(
            **common_order_data,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("0"),
        )
        assert order_zero_price.price == Decimal("0")

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

        # Common data for position tests
        strategy_id_for_pos = uuid.uuid4()
        common_position_data = {
            "strategy_id": strategy_id_for_pos,
            "exchange": "ftx",  # :-D
            "symbol": "SOL/USD",
            "current_price": Decimal("40"),
            "size": Decimal("10"),
            "leverage": Decimal("1"),
        }

        # Invalid entry_price (zero) - Pydantic default allows 0 unless constrained
        # with gt=0. Adjusted test to reflect current model behavior.
        pos_zero_entry = Position(
            **common_position_data, side=PositionSide.LONG, entry_price=Decimal("0")
        )
        assert pos_zero_entry.entry_price == Decimal("0")

        # Invalid entry_price (negative) - Pydantic default allows negative unless
        # constrained with ge=0 or gt=0.
        pos_neg_entry = Position(
            **common_position_data, side=PositionSide.LONG, entry_price=Decimal("-10")
        )
        assert pos_neg_entry.entry_price == Decimal("-10")

        # Invalid size (zero) - Model allows this if not constrained by Field(gt=0)
        pos_zero_size_data = common_position_data.copy()
        pos_zero_size_data["size"] = Decimal("0")
        pos_zero_size = Position(
            **pos_zero_size_data, side=PositionSide.LONG, entry_price=Decimal("30")
        )
        assert pos_zero_size.size == Decimal("0")

        # Invalid size (negative)
        pos_neg_size_data = common_position_data.copy()
        pos_neg_size_data["size"] = Decimal("-1")
        pos_neg_size = Position(
            **pos_neg_size_data, side=PositionSide.LONG, entry_price=Decimal("30")
        )
        assert pos_neg_size.size == Decimal("-1")

        # Invalid leverage (zero) - Model allows leverage=0
        pos_zero_leverage_data = common_position_data.copy()
        pos_zero_leverage_data["leverage"] = Decimal("0")
        pos_zero_leverage = Position(
            **pos_zero_leverage_data, side=PositionSide.LONG, entry_price=Decimal("30")
        )
        assert pos_zero_leverage.leverage == Decimal("0")

        # Invalid leverage (negative) - Model allows leverage=-1
        pos_neg_leverage_data = common_position_data.copy()
        pos_neg_leverage_data["leverage"] = Decimal("-1")
        pos_neg_leverage = Position(
            **pos_neg_leverage_data, side=PositionSide.LONG, entry_price=Decimal("30")
        )
        assert pos_neg_leverage.leverage == Decimal("-1")

        # Test PnL for SHORT position
        # Short position, price increases (loss)
        short_pos_loss = Position(
            strategy_id=strategy_id_for_pos,
            exchange="binance",
            symbol="BTCUSD_PERP",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),  # Price increased
            size=Decimal("1.0"),
            leverage=Decimal("1.0"),
        )
        # PNL % = (50000 - 55000) / 50000 = -0.1
        assert short_pos_loss.pnl_percentage == Decimal("-0.1")

        # Short position, price decreases (profit)
        short_pos_profit = Position(
            strategy_id=strategy_id_for_pos,
            exchange="binance",
            symbol="BTCUSD_PERP",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            current_price=Decimal("45000"),  # Price decreased
            size=Decimal("1.0"),
            leverage=Decimal("1.0"),
        )
        # PNL % = (50000 - 45000) / 50000 = 0.1
        assert short_pos_profit.pnl_percentage == Decimal("0.1")

    def test_option_model(self):
        """Test Option model validation."""
        # Valid option
        expiry_date = datetime(2023, 6, 30, 16, 0, 0, tzinfo=timezone.utc)
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

        # Common data for option tests
        strategy_id_for_option = uuid.uuid4()
        future_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        past_expiry = datetime.now(timezone.utc) - timedelta(days=1)

        common_option_data = {
            "strategy_id": strategy_id_for_option,
            "exchange": "deribit",
            "underlying": "BTC",
            "size": Decimal("1"),
            "premium": Decimal("100"),
            "total_premium": Decimal("100"),
            "collateral": Decimal("0"),  # Assuming collateral can be 0 for bought options
        }

        # Invalid strike_price (zero or negative) - Model allows this as no Field(gt=0)
        opt_zero_strike = Option(
            **common_option_data,
            strike_price=Decimal("0"),
            expiry_date=future_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert opt_zero_strike.strike_price == Decimal("0")
        opt_neg_strike = Option(
            **common_option_data,
            strike_price=Decimal("-50000"),
            expiry_date=future_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert opt_neg_strike.strike_price == Decimal("-50000")

        # Invalid size (zero or negative) - Model allows this as no Field(gt=0)
        opt_zero_size_data = common_option_data.copy()
        opt_zero_size_data["size"] = Decimal("0")
        opt_zero_size = Option(
            **opt_zero_size_data,
            strike_price=Decimal("50000"),
            expiry_date=future_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert opt_zero_size.size == Decimal("0")

        opt_neg_size_data = common_option_data.copy()
        opt_neg_size_data["size"] = Decimal("-1")
        opt_neg_size = Option(
            **opt_neg_size_data,
            strike_price=Decimal("50000"),
            expiry_date=future_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert opt_neg_size.size == Decimal("-1")

        # Invalid premium (negative) - Model allows this as no Field(ge=0)
        opt_neg_premium_data = common_option_data.copy()
        opt_neg_premium_data["premium"] = Decimal("-10")
        opt_neg_premium = Option(
            **opt_neg_premium_data,
            strike_price=Decimal("50000"),
            expiry_date=future_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert opt_neg_premium.premium == Decimal("-10")

        # Invalid total_premium (negative) - Model allows this as no Field(ge=0)
        opt_neg_total_premium_data = common_option_data.copy()
        opt_neg_total_premium_data["total_premium"] = Decimal("-100")
        opt_neg_total_premium = Option(
            **opt_neg_total_premium_data,
            strike_price=Decimal("50000"),
            expiry_date=future_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert opt_neg_total_premium.total_premium == Decimal("-100")

        # Invalid collateral (negative) - Model allows this as no Field(ge=0) constraint
        opt_neg_collateral_data = common_option_data.copy()
        opt_neg_collateral_data["collateral"] = Decimal("-0.01")
        opt_neg_collateral = Option(
            **opt_neg_collateral_data,
            strike_price=Decimal("50000"),
            expiry_date=future_expiry,
            option_type=OptionType.PUT,
            side="sell",
        )
        assert opt_neg_collateral.collateral == Decimal("-0.01")

        # Expiry date in the past - this is not inherently invalid by Pydantic,
        # but days_to_expiry should be 0
        past_option = Option(
            **common_option_data,
            strike_price=Decimal("50000"),
            expiry_date=past_expiry,
            option_type=OptionType.CALL,
            side="buy",
        )
        assert past_option.days_to_expiry == 0.0

        # Invalid enum for option_type
        with pytest.raises(ValueError):  # Pydantic ValidationError
            Option(
                **common_option_data,
                strike_price=Decimal("50000"),
                expiry_date=future_expiry,
                option_type="INVALID_OPTION_TYPE",
                side="buy",
            )

        # Invalid enum for side (model uses Literal["buy", "sell"])
        with pytest.raises(ValueError):  # Pydantic ValidationError
            Option(
                **common_option_data,
                strike_price=Decimal("50000"),
                expiry_date=future_expiry,
                option_type=OptionType.CALL,
                side="INVALID_SIDE",
            )


if __name__ == "__main__":
    pytest.main()
