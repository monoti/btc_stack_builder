"""
Utility functions for BTC Stack-Builder Bot.

This module provides various calculation and conversion functions used throughout the application,
including basis calculation, funding rate calculation, margin ratio calculation, option pricing,
position PnL calculation, and various Bitcoin/timestamp conversion utilities.
"""
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, Optional, Tuple, Union, List, Any

import numpy as np
from scipy.stats import norm

# Set decimal precision for financial calculations
getcontext().prec = 18
getcontext().rounding = ROUND_HALF_UP

# Constants for BTC conversion
SATOSHI_PER_BTC = Decimal("100000000")  # 1 BTC = 100,000,000 satoshis


def calculate_annualized_basis(
    futures_price: Decimal, 
    spot_price: Decimal, 
    days_to_expiry: int
) -> Decimal:
    """
    Calculate the annualized basis percentage for a futures contract.
    
    Basis % = ((Futures_Price / Spot_Price) - 1) * (365 / Days_to_Expiry) * 100
    
    Args:
        futures_price: Current price of the futures contract
        spot_price: Current spot price of the underlying asset
        days_to_expiry: Number of days until the futures contract expires
        
    Returns:
        Annualized basis percentage as a decimal (e.g., 0.05 for 5%)
        
    Raises:
        ValueError: If spot_price is zero or days_to_expiry is zero
        ZeroDivisionError: If spot_price or days_to_expiry is zero
    """
    if spot_price <= 0:
        raise ValueError("Spot price must be greater than zero")
    if days_to_expiry <= 0:
        raise ValueError("Days to expiry must be greater than zero")
    
    # Calculate raw basis
    basis = (futures_price / spot_price) - Decimal("1.0")
    
    # Annualize the basis
    annualized_basis = basis * (Decimal("365") / Decimal(str(days_to_expiry)))
    
    return annualized_basis


def calculate_funding_rate(
    mark_price: Decimal, 
    index_price: Decimal, 
    interest_rate: Decimal = Decimal("0.0"),
    premium_index_weight: Decimal = Decimal("1.0")
) -> Decimal:
    """
    Calculate the funding rate for perpetual futures contracts.
    
    The funding rate is typically calculated as:
    Funding Rate = Premium Index + clamp(Interest Rate - Premium Index, -0.05%, 0.05%)
    
    Where Premium Index = (Mark Price - Index Price) / Index Price
    
    Args:
        mark_price: Current mark price of the perpetual contract
        index_price: Current index price of the underlying asset
        interest_rate: Current interest rate (default: 0.0)
        premium_index_weight: Weight of the premium index in the calculation (default: 1.0)
        
    Returns:
        Funding rate as a decimal (e.g., 0.0001 for 0.01%)
        
    Raises:
        ValueError: If index_price is zero
    """
    if index_price <= 0:
        raise ValueError("Index price must be greater than zero")
    
    # Calculate premium index
    premium_index = (mark_price - index_price) / index_price
    
    # Apply weight to premium index
    weighted_premium = premium_index * premium_index_weight
    
    # Calculate interest rate component (clamped between -0.05% and 0.05%)
    interest_component = interest_rate - premium_index
    clamped_interest = max(min(interest_component, Decimal("0.0005")), Decimal("-0.0005"))
    
    # Calculate funding rate
    funding_rate = weighted_premium + clamped_interest
    
    return funding_rate


def calculate_margin_ratio(
    wallet_balance: Decimal, 
    maintenance_margin: Decimal
) -> Decimal:
    """
    Calculate the margin ratio for risk management.
    
    Margin Ratio = (Wallet Balance / Maintenance Margin) * 100
    
    Args:
        wallet_balance: Current wallet balance
        maintenance_margin: Current maintenance margin requirement
        
    Returns:
        Margin ratio as a decimal (e.g., 4.5 for 450%)
        
    Raises:
        ValueError: If maintenance_margin is zero
    """
    if maintenance_margin <= 0:
        if wallet_balance > 0:
            # If there's no maintenance margin but we have balance, return a very high ratio
            return Decimal("999.99")
        return Decimal("0.0")
    
    # Calculate margin ratio
    margin_ratio = (wallet_balance / maintenance_margin)
    
    return margin_ratio


def calculate_option_delta(
    spot_price: Decimal,
    strike_price: Decimal,
    time_to_expiry_years: Decimal,
    risk_free_rate: Decimal,
    volatility: Decimal,
    option_type: str
) -> Decimal:
    """
    Calculate the delta of an option using the Black-Scholes model.
    
    Delta represents the rate of change of the option price with respect to changes
    in the underlying asset's price.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry_years: Time to expiry in years (e.g., 0.25 for 3 months)
        risk_free_rate: Risk-free interest rate as a decimal (e.g., 0.02 for 2%)
        volatility: Implied volatility as a decimal (e.g., 0.5 for 50%)
        option_type: Type of option ('call' or 'put')
        
    Returns:
        Delta of the option as a decimal
        
    Raises:
        ValueError: If option_type is not 'call' or 'put'
    """
    # Convert Decimal to float for numpy/scipy calculations
    s = float(spot_price)
    k = float(strike_price)
    t = float(time_to_expiry_years)
    r = float(risk_free_rate)
    v = float(volatility)
    
    # Handle edge case for very short expiry
    if t <= 0:
        if option_type.lower() == 'call':
            return Decimal("1.0") if s > k else Decimal("0.0")
        elif option_type.lower() == 'put':
            return Decimal("-1.0") if s < k else Decimal("0.0")
        else:
            raise ValueError("Option type must be 'call' or 'put'")
    
    # Calculate d1 from Black-Scholes
    d1 = (math.log(s / k) + (r + (v ** 2) / 2) * t) / (v * math.sqrt(t))
    
    # Calculate delta based on option type
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    elif option_type.lower() == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    # Convert back to Decimal
    return Decimal(str(delta))


def calculate_option_greeks(
    spot_price: Decimal,
    strike_price: Decimal,
    time_to_expiry_years: Decimal,
    risk_free_rate: Decimal,
    volatility: Decimal,
    option_type: str
) -> Dict[str, Decimal]:
    """
    Calculate all option Greeks using the Black-Scholes model.
    
    Args:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry_years: Time to expiry in years (e.g., 0.25 for 3 months)
        risk_free_rate: Risk-free interest rate as a decimal (e.g., 0.02 for 2%)
        volatility: Implied volatility as a decimal (e.g., 0.5 for 50%)
        option_type: Type of option ('call' or 'put')
        
    Returns:
        Dictionary containing all option Greeks (delta, gamma, theta, vega, rho)
        
    Raises:
        ValueError: If option_type is not 'call' or 'put'
    """
    # Convert Decimal to float for numpy/scipy calculations
    s = float(spot_price)
    k = float(strike_price)
    t = float(time_to_expiry_years)
    r = float(risk_free_rate)
    v = float(volatility)
    
    # Handle edge case for very short expiry
    if t <= 0:
        if option_type.lower() == 'call':
            delta = 1.0 if s > k else 0.0
        elif option_type.lower() == 'put':
            delta = -1.0 if s < k else 0.0
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        return {
            "delta": Decimal(str(delta)),
            "gamma": Decimal("0"),
            "theta": Decimal("0"),
            "vega": Decimal("0"),
            "rho": Decimal("0")
        }
    
    # Calculate d1 and d2 from Black-Scholes
    d1 = (math.log(s / k) + (r + (v ** 2) / 2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    
    # Calculate option price
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        price = s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        delta = norm.cdf(d1) - 1
        price = k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    # Calculate gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (s * v * math.sqrt(t))
    
    # Calculate theta
    if option_type.lower() == 'call':
        theta = -s * norm.pdf(d1) * v / (2 * math.sqrt(t)) - r * k * math.exp(-r * t) * norm.cdf(d2)
    else:  # put
        theta = -s * norm.pdf(d1) * v / (2 * math.sqrt(t)) + r * k * math.exp(-r * t) * norm.cdf(-d2)
    
    # Calculate vega (same for calls and puts)
    vega = s * math.sqrt(t) * norm.pdf(d1) / 100  # Divided by 100 for 1% change in volatility
    
    # Calculate rho
    if option_type.lower() == 'call':
        rho = k * t * math.exp(-r * t) * norm.cdf(d2) / 100  # Divided by 100 for 1% change in rate
    else:  # put
        rho = -k * t * math.exp(-r * t) * norm.cdf(-d2) / 100
    
    # Convert back to Decimal
    return {
        "delta": Decimal(str(delta)),
        "gamma": Decimal(str(gamma)),
        "theta": Decimal(str(theta)),
        "vega": Decimal(str(vega)),
        "rho": Decimal(str(rho))
    }


def calculate_position_pnl(
    entry_price: Decimal,
    current_price: Decimal,
    position_size: Decimal,
    position_side: str,
    leverage: Decimal = Decimal("1.0"),
    fees: Decimal = Decimal("0.0"),
    funding: Decimal = Decimal("0.0")
) -> Dict[str, Decimal]:
    """
    Calculate the profit and loss of a position.
    
    Args:
        entry_price: Entry price of the position
        current_price: Current price of the asset
        position_size: Size of the position in base currency or contracts
        position_side: Side of the position ('long' or 'short')
        leverage: Leverage used for the position (default: 1.0)
        fees: Total fees paid for the position (default: 0.0)
        funding: Total funding payments (positive or negative) (default: 0.0)
        
    Returns:
        Dictionary containing PnL information (absolute_pnl, percentage_pnl, roi)
        
    Raises:
        ValueError: If position_side is not 'long' or 'short'
    """
    if position_side.lower() not in ['long', 'short']:
        raise ValueError("Position side must be 'long' or 'short'")
    
    # Calculate price difference
    if position_side.lower() == 'long':
        price_diff = current_price - entry_price
    else:  # short
        price_diff = entry_price - current_price
    
    # Calculate absolute PnL
    absolute_pnl = price_diff * position_size
    
    # Adjust for fees and funding
    adjusted_pnl = absolute_pnl - fees + funding
    
    # Calculate percentage PnL (relative to position value)
    if entry_price <= 0:
        percentage_pnl = Decimal("0.0")
    else:
        percentage_pnl = price_diff / entry_price
    
    # Calculate ROI (return on investment, considering leverage)
    position_value = entry_price * position_size
    initial_margin = position_value / leverage
    
    if initial_margin <= 0:
        roi = Decimal("0.0")
    else:
        roi = adjusted_pnl / initial_margin
    
    return {
        "absolute_pnl": adjusted_pnl,
        "percentage_pnl": percentage_pnl,
        "roi": roi
    }


def calculate_drawdown(
    peak_value: Decimal,
    current_value: Decimal
) -> Dict[str, Decimal]:
    """
    Calculate the drawdown from a peak value.
    
    Args:
        peak_value: Peak value of the portfolio or position
        current_value: Current value of the portfolio or position
        
    Returns:
        Dictionary containing drawdown information (absolute_drawdown, percentage_drawdown)
    """
    if peak_value <= 0:
        return {
            "absolute_drawdown": Decimal("0.0"),
            "percentage_drawdown": Decimal("0.0")
        }
    
    # Calculate absolute drawdown
    absolute_drawdown = peak_value - current_value
    
    # Calculate percentage drawdown
    percentage_drawdown = absolute_drawdown / peak_value
    
    return {
        "absolute_drawdown": absolute_drawdown,
        "percentage_drawdown": percentage_drawdown
    }


def calculate_annualized_return(
    initial_value: Decimal,
    final_value: Decimal,
    days_held: int
) -> Decimal:
    """
    Calculate the annualized return of an investment.
    
    Args:
        initial_value: Initial investment value
        final_value: Final investment value
        days_held: Number of days the investment was held
        
    Returns:
        Annualized return as a decimal (e.g., 0.1 for 10%)
        
    Raises:
        ValueError: If initial_value is zero or days_held is zero
    """
    if initial_value <= 0:
        raise ValueError("Initial value must be greater than zero")
    if days_held <= 0:
        raise ValueError("Days held must be greater than zero")
    
    # Calculate total return
    total_return = final_value / initial_value
    
    # Convert to annualized return
    years_held = Decimal(str(days_held)) / Decimal("365")
    annualized_return = total_return ** (Decimal("1") / years_held) - Decimal("1")
    
    return annualized_return


def calculate_rolling_cost(
    current_contract_price: Decimal,
    next_contract_price: Decimal,
    position_size: Decimal,
    fees_percentage: Decimal
) -> Dict[str, Decimal]:
    """
    Calculate the cost of rolling a futures contract position.
    
    Args:
        current_contract_price: Price of the current contract
        next_contract_price: Price of the next contract
        position_size: Size of the position in contracts
        fees_percentage: Trading fees as a decimal percentage
        
    Returns:
        Dictionary containing rolling cost information
    """
    # Calculate price difference between contracts
    price_diff = next_contract_price - current_contract_price
    
    # Calculate basis cost/profit
    basis_cost = price_diff * position_size
    
    # Calculate trading fees
    close_fee = current_contract_price * position_size * fees_percentage
    open_fee = next_contract_price * position_size * fees_percentage
    total_fees = close_fee + open_fee
    
    # Calculate total rolling cost
    total_cost = basis_cost + total_fees
    
    return {
        "basis_cost": basis_cost,
        "fees_cost": total_fees,
        "total_cost": total_cost
    }


def format_btc_amount(
    amount: Decimal,
    precision: int = 8,
    include_symbol: bool = True
) -> str:
    """
    Format a BTC amount with proper precision.
    
    Args:
        amount: BTC amount to format
        precision: Decimal precision to display (default: 8)
        include_symbol: Whether to include the BTC symbol (default: True)
        
    Returns:
        Formatted BTC amount string
    """
    # Round to specified precision
    rounded_amount = amount.quantize(Decimal('0.' + '0' * precision))
    
    # Format with comma separators for thousands
    formatted_amount = f"{rounded_amount:,}"
    
    # Add BTC symbol if requested
    if include_symbol:
        return f"{formatted_amount} BTC"
    
    return formatted_amount


def format_percentage(
    percentage: Decimal,
    precision: int = 2,
    include_symbol: bool = True
) -> str:
    """
    Format a percentage value with proper precision.
    
    Args:
        percentage: Percentage value as a decimal (e.g., 0.0567 for 5.67%)
        precision: Decimal precision to display (default: 2)
        include_symbol: Whether to include the % symbol (default: True)
        
    Returns:
        Formatted percentage string
    """
    # Convert to percentage (multiply by 100)
    percentage_value = percentage * Decimal("100")
    
    # Round to specified precision
    rounded_percentage = percentage_value.quantize(Decimal('0.' + '0' * precision))
    
    # Format with comma separators for thousands
    formatted_percentage = f"{rounded_percentage:,}"
    
    # Add % symbol if requested
    if include_symbol:
        return f"{formatted_percentage}%"
    
    return formatted_percentage


def satoshi_to_btc(satoshi_amount: int) -> Decimal:
    """
    Convert satoshis to BTC.
    
    Args:
        satoshi_amount: Amount in satoshis
        
    Returns:
        Equivalent amount in BTC as a Decimal
    """
    return Decimal(str(satoshi_amount)) / SATOSHI_PER_BTC


def btc_to_satoshi(btc_amount: Decimal) -> int:
    """
    Convert BTC to satoshis.
    
    Args:
        btc_amount: Amount in BTC
        
    Returns:
        Equivalent amount in satoshis as an integer
    """
    satoshi_amount = btc_amount * SATOSHI_PER_BTC
    return int(satoshi_amount)


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert a Unix timestamp to a datetime object.
    
    Args:
        timestamp: Unix timestamp (seconds since epoch)
        
    Returns:
        Equivalent datetime object (UTC)
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert a datetime object to a Unix timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        Equivalent Unix timestamp (seconds since epoch)
    """
    if dt.tzinfo is None:
        # Assume UTC if no timezone is specified
        dt = dt.replace(tzinfo=timezone.utc)
    
    return int(dt.timestamp())


def days_between_dates(start_date: datetime, end_date: datetime) -> int:
    """
    Calculate the number of days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of days between the dates
    """
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
    delta = end_date - start_date
    return delta.days


def calculate_days_to_expiry(expiry_date: datetime) -> int:
    """
    Calculate the number of days until a futures contract expires.
    
    Args:
        expiry_date: Expiry date of the contract
        
    Returns:
        Number of days until expiry
    """
    now = datetime.now(timezone.utc)
    return max(0, days_between_dates(now, expiry_date))


def parse_quarterly_futures_symbol(symbol: str) -> Optional[datetime]:
    """
    Parse a quarterly futures symbol to extract the expiry date.
    
    Example: 'BTCUSD_251226' -> datetime(2025, 12, 26)
    
    Args:
        symbol: Futures contract symbol
        
    Returns:
        Expiry date as datetime object, or None if parsing fails
    """
    try:
        # Extract date part (last 6 digits)
        date_part = symbol.split('_')[-1]
        if len(date_part) != 6:
            return None
        
        # Parse YY, MM, DD
        yy = int(date_part[0:2])
        mm = int(date_part[2:4])
        dd = int(date_part[4:6])
        
        # Adjust year (assuming 20xx for simplicity)
        year = 2000 + yy
        
        # Create datetime object
        expiry_date = datetime(year, mm, dd, 16, 0, 0, tzinfo=timezone.utc)  # 16:00 UTC is common
        
        return expiry_date
    except (ValueError, IndexError):
        return None


def get_next_quarterly_expiry(current_date: Optional[datetime] = None) -> datetime:
    """
    Get the next quarterly expiry date.
    
    Quarterly expiries are typically the last Friday of March, June, September, and December.
    
    Args:
        current_date: Current date (default: today)
        
    Returns:
        Next quarterly expiry date
    """
    if current_date is None:
        current_date = datetime.now(timezone.utc)
    
    year = current_date.year
    month = current_date.month
    
    # Determine the next quarter month
    if month < 3:
        quarter_month = 3
    elif month < 6:
        quarter_month = 6
    elif month < 9:
        quarter_month = 9
    elif month < 12:
        quarter_month = 12
    else:
        # If December, move to March of next year
        quarter_month = 3
        year += 1
    
    # Find the last Friday of the quarter month
    # Start with the last day of the month
    if quarter_month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = quarter_month + 1
        next_year = year
    
    last_day = datetime(next_year, next_month, 1, tzinfo=timezone.utc) - timezone.timedelta(days=1)
    
    # Find the last Friday
    weekday = last_day.weekday()
    if weekday < 4:  # If last day is earlier than Friday (Monday=0, Friday=4)
        days_to_subtract = weekday + 3
    else:  # If last day is Friday or later
        days_to_subtract = weekday - 4
    
    last_friday = last_day - timezone.timedelta(days=days_to_subtract)
    
    # Set time to 16:00 UTC (common futures expiry time)
    expiry_date = datetime(last_friday.year, last_friday.month, last_friday.day, 
                          16, 0, 0, tzinfo=timezone.utc)
    
    return expiry_date
