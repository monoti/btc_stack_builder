"""
Risk management module for BTC Stack-Builder Bot.

This module provides risk management components, including the MarginGuard
which continuously monitors margin ratios and takes action to prevent liquidations.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from btc_stack_builder.core.constants import (
    MARGIN_CHECK_INTERVAL,
    MARGIN_RATIO_CRITICAL_THRESHOLD,
    MARGIN_RATIO_WARNING_THRESHOLD,
)
from btc_stack_builder.core.logger import logger
from btc_stack_builder.core.models import (
    MarginLevel,
    MarginStatus,
    Position,
    PositionSide,
    PositionStatus,
)
from btc_stack_builder.gateways import ExchangeGateway


class AlertLevel(str, Enum):
    """Alert levels for margin monitoring."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class MarginGuard:
    """
    MarginGuard monitors exchange margin ratios and takes action to prevent liquidations.

    This is the most critical risk management service. It runs independently and continuously,
    checking the overall account margin ratio at regular intervals. When the margin ratio
    falls below warning or critical thresholds, it takes appropriate action:

    1. WARNING: Send alert
    2. CRITICAL: Send alert and transfer BTC from spot to futures wallet
    3. EMERGENCY: Begin safe unwinding of positions
    """

    def __init__(
        self,
        gateways: dict[str, ExchangeGateway],
        warning_threshold: float = float(MARGIN_RATIO_WARNING_THRESHOLD),
        critical_threshold: float = float(MARGIN_RATIO_CRITICAL_THRESHOLD),
        check_interval: int = MARGIN_CHECK_INTERVAL,
    ):
        """
        Initialize the MarginGuard.

        Args:
            gateways: Dictionary mapping exchange names to gateway instances
            warning_threshold: Margin ratio warning threshold (default: 450%)
            critical_threshold: Margin ratio critical threshold (default: 400%)
            check_interval: Interval between margin checks in seconds (default: 300)
        """
        self.gateways = gateways
        self.warning_threshold = Decimal(str(warning_threshold))
        self.critical_threshold = Decimal(str(critical_threshold))
        self.check_interval = check_interval
        self.running = False
        self.task = None
        self.last_alert_level: dict[str, AlertLevel] = {}
        self.last_margin_status: dict[str, MarginStatus] = {}

    async def start(self) -> None:
        """Start the margin monitoring loop."""
        if self.running:
            logger.warning("MarginGuard is already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "MarginGuard started",
            warning_threshold=f"{float(self.warning_threshold):.2f}",
            critical_threshold=f"{float(self.critical_threshold):.2f}",
            check_interval=self.check_interval,
        )

    async def stop(self) -> None:
        """Stop the margin monitoring loop."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

        logger.info("MarginGuard stopped")

    async def check_margin(self) -> dict[str, MarginStatus]:
        """
        Check margin status for all exchanges.

        Returns:
            Dictionary mapping exchange names to margin status
        """
        results = {}

        for exchange_name, gateway in self.gateways.items():
            try:
                margin_status = await gateway.get_margin_status()
                results[exchange_name] = margin_status

                # Check margin level and send alerts if necessary
                await self._process_margin_status(exchange_name, margin_status)

                # Store last margin status
                self.last_margin_status[exchange_name] = margin_status

            except Exception as e:
                logger.error(
                    f"Error checking margin for {exchange_name}",
                    exchange=exchange_name,
                    error=str(e),
                    exc_info=True,
                )

        return results

    async def _monitor_loop(self) -> None:
        """Main monitoring loop that runs continuously."""
        while self.running:
            try:
                await self.check_margin()
            except Exception:
                logger.error("Error in margin monitoring loop", exc_info=True)

            # Wait for next check interval
            await asyncio.sleep(self.check_interval)

    async def _process_margin_status(self, exchange: str, status: MarginStatus) -> None:
        """
        Process margin status and take appropriate action.

        Args:
            exchange: Exchange name
            status: Current margin status
        """
        margin_ratio = status.margin_ratio
        alert_level = self._get_alert_level(margin_ratio)

        # Check if alert level has changed
        last_level = self.last_alert_level.get(exchange)
        if last_level != alert_level:
            self._send_alert(exchange, status, alert_level)
            self.last_alert_level[exchange] = alert_level

        # Take action based on alert level
        if alert_level == AlertLevel.WARNING:
            # Just send alert, no action needed
            pass

        elif alert_level == AlertLevel.CRITICAL:
            # Send alert and top up margin if possible
            await self._top_up_margin(exchange, status)

        elif alert_level == AlertLevel.EMERGENCY:
            # Critical situation, begin unwinding positions
            await self._unwind_positions(exchange, status)

    def _get_alert_level(self, margin_ratio: Decimal) -> AlertLevel:
        """
        Determine alert level based on margin ratio.

        Args:
            margin_ratio: Current margin ratio

        Returns:
            Alert level
        """
        if margin_ratio < Decimal("1.1"):  # Near liquidation
            return AlertLevel.EMERGENCY
        elif margin_ratio < self.critical_threshold:
            return AlertLevel.CRITICAL
        elif margin_ratio < self.warning_threshold:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    def _send_alert(self, exchange: str, status: MarginStatus, level: AlertLevel) -> None:
        """
        Send alert for margin status.

        Args:
            exchange: Exchange name
            status: Current margin status
            level: Alert level
        """
        if level == AlertLevel.INFO:
            logger.info(
                f"Margin status for {exchange}: {float(status.margin_ratio):.2f}x",
                exchange=exchange,
                margin_ratio=float(status.margin_ratio),
                wallet_balance=float(status.wallet_balance),
                maintenance_margin=float(status.maintenance_margin),
            )
        elif level == AlertLevel.WARNING:
            logger.warning(
                f"WARNING: Low margin ratio for {exchange}: {float(status.margin_ratio):.2f}x",
                exchange=exchange,
                margin_ratio=float(status.margin_ratio),
                wallet_balance=float(status.wallet_balance),
                maintenance_margin=float(status.maintenance_margin),
                threshold=float(self.warning_threshold),
            )
        elif level == AlertLevel.CRITICAL:
            logger.error(
                f"CRITICAL: Very low margin ratio for {exchange}: {float(status.margin_ratio):.2f}x",
                exchange=exchange,
                margin_ratio=float(status.margin_ratio),
                wallet_balance=float(status.wallet_balance),
                maintenance_margin=float(status.maintenance_margin),
                threshold=float(self.critical_threshold),
            )
        elif level == AlertLevel.EMERGENCY:
            logger.critical(
                f"EMERGENCY: Extreme low margin ratio for {exchange}: {float(status.margin_ratio):.2f}x",
                exchange=exchange,
                margin_ratio=float(status.margin_ratio),
                wallet_balance=float(status.wallet_balance),
                maintenance_margin=float(status.maintenance_margin),
            )

    async def _top_up_margin(self, exchange: str, status: MarginStatus) -> bool:
        """
        Top up margin by transferring BTC from spot to futures wallet.

        Args:
            exchange: Exchange name
            status: Current margin status

        Returns:
            True if top-up was successful, False otherwise
        """
        try:
            gateway = self.gateways.get(exchange)
            if not gateway:
                logger.error(f"Gateway not found for {exchange}")
                return False

            # Calculate required top-up amount to reach target margin ratio
            target_ratio = self.warning_threshold  # Target the warning threshold
            current_maintenance_margin = status.maintenance_margin
            current_wallet_balance = status.wallet_balance

            # Calculate how much additional balance we need
            required_balance = current_maintenance_margin * target_ratio
            shortfall = required_balance - current_wallet_balance

            if shortfall <= Decimal("0"):
                logger.info(f"No margin top-up needed for {exchange}")
                return True

            # Add 10% buffer
            transfer_amount = shortfall * Decimal("1.1")

            # Round to 6 decimal places
            transfer_amount = transfer_amount.quantize(Decimal("0.000001"))

            logger.info(
                f"Transferring {transfer_amount} BTC from spot to futures for {exchange}",
                exchange=exchange,
                amount=str(transfer_amount),
                current_ratio=float(status.margin_ratio),
                target_ratio=float(target_ratio),
            )

            # Execute the transfer
            success = await gateway.transfer_margin(
                amount=transfer_amount, asset="BTC", from_type="spot", to_type="futures"
            )

            if success:
                logger.info(
                    f"Successfully topped up margin for {exchange}",
                    exchange=exchange,
                    amount=str(transfer_amount),
                )
            else:
                logger.error(
                    f"Failed to top up margin for {exchange}",
                    exchange=exchange,
                    amount=str(transfer_amount),
                )

            return success

        except Exception as e:
            logger.error(
                f"Error topping up margin for {exchange}",
                exchange=exchange,
                error=str(e),
                exc_info=True,
            )
            return False

    async def _unwind_positions(self, exchange: str, status: MarginStatus) -> bool:
        """
        Safely unwind positions to reduce margin usage.

        This is a last resort action when margin ratio is critically low
        and top-up is not possible.

        Args:
            exchange: Exchange name
            status: Current margin status

        Returns:
            True if unwinding was successful, False otherwise
        """
        try:
            gateway = self.gateways.get(exchange)
            if not gateway:
                logger.error(f"Gateway not found for {exchange}")
                return False

            # Get all open positions
            positions = await gateway.get_positions()
            if not positions:
                logger.warning(f"No positions to unwind for {exchange}")
                return True

            # Sort positions by risk (highest leverage first)
            positions.sort(key=lambda p: p.leverage, reverse=True)

            logger.warning(
                f"EMERGENCY: Unwinding positions for {exchange} due to critical margin ratio",
                exchange=exchange,
                margin_ratio=float(status.margin_ratio),
                positions=len(positions),
            )

            # Close positions one by one, starting with highest leverage
            success = True
            for position in positions:
                try:
                    # Close position with market order
                    await gateway.create_order(
                        symbol=position.symbol,
                        order_type="market",
                        side="sell" if position.side == "long" else "buy",
                        amount=position.size,
                    )

                    logger.info(
                        f"Closed position {position.symbol} as part of emergency unwind",
                        exchange=exchange,
                        symbol=position.symbol,
                        size=str(position.size),
                        side=position.side,
                    )

                    # Check if margin ratio has improved enough
                    new_status = await gateway.get_margin_status()
                    if new_status.margin_ratio >= self.warning_threshold:
                        logger.info(
                            "Margin ratio restored to safe level after unwinding some positions",
                            exchange=exchange,
                            margin_ratio=float(new_status.margin_ratio),
                        )
                        break

                except Exception as e:
                    logger.error(
                        f"Error closing position {position.symbol}",
                        exchange=exchange,
                        symbol=position.symbol,
                        error=str(e),
                        exc_info=True,
                    )
                    success = False

            return success

        except Exception as e:
            logger.error(
                f"Error unwinding positions for {exchange}",
                exchange=exchange,
                error=str(e),
                exc_info=True,
            )
            return False

    def get_status(self) -> dict[str, Any]:
        """
        Get current status of the MarginGuard.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self.running,
            "warning_threshold": float(self.warning_threshold),
            "critical_threshold": float(self.critical_threshold),
            "check_interval": self.check_interval,
            "last_alert_levels": {k: v.value for k, v in self.last_alert_level.items()},
            "last_margin_ratios": {
                k: float(v.margin_ratio) for k, v in self.last_margin_status.items()
            },
        }


# Export components
__all__ = [
    "MarginGuard",
    "AlertLevel",
]
