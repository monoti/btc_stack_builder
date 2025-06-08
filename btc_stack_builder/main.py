#!/usr/bin/env python3
"""
BTC Stack-Builder Bot - Main Application Entry Point

This module serves as the entry point for the BTC Stack-Builder Bot. It initializes
all components, sets up the event loop, and orchestrates the overall application flow.
"""
import argparse
import asyncio
import os
import signal
import sys

import uvloop
from prometheus_client import start_http_server

from btc_stack_builder import __version__
from btc_stack_builder.config import Environment, config
from btc_stack_builder.core.logger import logger, setup_logger
from btc_stack_builder.core.models import (
    ExchangeCredentials,
    Portfolio,
    PortfolioAllocation,
    Strategy,
    StrategyType,
    SubPortfolio,
    SubPortfolioType,
)
from btc_stack_builder.db.session import init_db
from btc_stack_builder.gateways import ExchangeGateway
from btc_stack_builder.gateways.binance_gateway import BinanceGateway
from btc_stack_builder.gateways.deribit_gateway import DeribitGateway
from btc_stack_builder.monitoring.metrics import register_metrics, setup_metrics
from btc_stack_builder.risk.margin_guard import MarginGuard
from btc_stack_builder.scheduler.task_manager import TaskManager
from btc_stack_builder.strategies.basis_harvest import BasisHarvestStrategy
from btc_stack_builder.strategies.funding_capture import FundingCaptureStrategy
from btc_stack_builder.strategies.put_wheel import OptionPremiumStrategy

# Global variables for clean shutdown
running = True
gateways: dict[str, ExchangeGateway] = {}
strategies: dict[StrategyType, Strategy] = {}  # Check if StrategyType is used
task_manager: TaskManager | None = None
margin_guard: MarginGuard | None = None
# Module-level 'global' statement removed as it's not standard; globals are implicit.
# The below comments were too long.
# For clarity, the global variables are those defined above at module scope.


async def initialize_gateways() -> dict[str, ExchangeGateway]:
    """
    Initialize exchange gateways based on configuration.

    Returns:
        Dictionary mapping exchange names to gateway instances
    """
    gateway_map = {}

    # Initialize Binance gateway if enabled
    if config.binance.enabled:
        logger.info("Initializing Binance gateway")
        binance_creds = ExchangeCredentials(
            api_key=config.binance.credentials.api_key.get_secret_value(),  # type: ignore
            api_secret=config.binance.credentials.api_secret.get_secret_value(),  # type: ignore
            is_testnet=config.binance.use_testnet,
        )
        binance_gateway = BinanceGateway(binance_creds, config.binance.use_testnet)  # type: ignore
        await binance_gateway.initialize()
        gateway_map["binance"] = binance_gateway

    # Initialize Deribit gateway if enabled
    if config.deribit.enabled:
        logger.info("Initializing Deribit gateway")
        deribit_credentials = ExchangeCredentials(
            api_key=config.deribit.credentials.api_key.get_secret_value(),  # type: ignore
            api_secret=config.deribit.credentials.api_secret.get_secret_value(),  # type: ignore
            is_testnet=config.deribit.use_testnet,
        )
        deribit_gateway = DeribitGateway(
            deribit_credentials, config.deribit.use_testnet  # type: ignore
        )
        await deribit_gateway.initialize()
        gateway_map["deribit"] = deribit_gateway

    return gateway_map


async def initialize_portfolio() -> Portfolio:
    """
    Initialize the portfolio based on configuration.

    Returns:
        Portfolio instance
    """
    logger.info("Initializing portfolio")

    # Create portfolio allocation
    allocation = PortfolioAllocation(
        core_hodl=config.portfolio.core_hodl_allocation,
        basis_harvest=config.portfolio.basis_harvest_allocation,
        funding_capture=config.portfolio.funding_capture_allocation,
        option_premium=config.portfolio.option_premium_allocation,
    )

    # Create sub-portfolios
    sub_portfolios = {}

    # Get initial balances from exchanges
    btc_price_usd = Decimal("0")
    total_balance_btc = Decimal("0")

    # Try to get BTC price from Binance
    if "binance" in gateways:
        try:
            ticker = await gateways["binance"].get_ticker("BTC/USDT")
            btc_price_usd = ticker["last"]

            # Get BTC balance
            balances = await gateways["binance"].get_balance()
            if "BTC" in balances:
                total_balance_btc += balances["BTC"]["total"]
        except Exception:
            logger.error("Error getting BTC price or balance", exc_info=True)

    # Calculate total balance in USD
    total_balance_usd = total_balance_btc * btc_price_usd

    # Create sub-portfolios
    for sub_type in SubPortfolioType:
        allocation_percentage = getattr(allocation, sub_type.value)
        target_balance_btc = total_balance_btc * allocation_percentage
        target_balance_usd = total_balance_usd * allocation_percentage

        sub_portfolios[sub_type] = SubPortfolio(
            type=sub_type,
            allocation_percentage=allocation_percentage,
            current_balance_btc=Decimal("0"),  # Will be updated later
            current_balance_usd=Decimal("0"),  # Will be updated later
            target_balance_btc=target_balance_btc,
            target_balance_usd=target_balance_usd,
        )

    # Create portfolio
    portfolio = Portfolio(
        name="BTC Stack-Builder Portfolio",
        total_balance_btc=total_balance_btc,
        total_balance_usd=total_balance_usd,
        btc_price_usd=btc_price_usd,
        allocation=allocation,
        sub_portfolios=sub_portfolios,
        cold_wallet_address=config.portfolio.cold_wallet_address,
    )

    return portfolio


async def initialize_strategies(portfolio: Portfolio) -> dict[StrategyType, Strategy]:
    """
    Initialize trading strategies based on configuration.

    Args:
        portfolio: Portfolio instance

    Returns:
        Dictionary mapping strategy types to strategy instances
    """
    logger.info("Initializing strategies")
    strategy_map = {}

    # Initialize Basis Harvest strategy if enabled
    if config.basis_harvest.enabled and "binance" in gateways:
        logger.info("Initializing Basis Harvest strategy")
        basis_strategy = BasisHarvestStrategy(
            gateways["binance"],
            portfolio.sub_portfolios[SubPortfolioType.BASIS_HARVEST],
            config.basis_harvest,
        )
        await basis_strategy.initialize()
        strategy_map[StrategyType.BASIS_HARVEST] = basis_strategy

    # Initialize Funding Capture strategy if enabled
    if config.funding_capture.enabled and "binance" in gateways:
        logger.info("Initializing Funding Capture strategy")
        funding_strategy = FundingCaptureStrategy(
            gateways["binance"],
            portfolio.sub_portfolios[SubPortfolioType.FUNDING_CAPTURE],
            config.funding_capture,
        )
        await funding_strategy.initialize()
        strategy_map[StrategyType.FUNDING_CAPTURE] = funding_strategy

    # Initialize Option Premium strategy if enabled
    if config.option_premium.enabled and "deribit" in gateways:
        logger.info("Initializing Option Premium strategy")
        option_strategy = OptionPremiumStrategy(
            gateways["deribit"],
            portfolio.sub_portfolios[SubPortfolioType.OPTION_PREMIUM],
            config.option_premium,
        )
        await option_strategy.initialize()
        strategy_map[StrategyType.OPTION_PREMIUM] = option_strategy

    return strategy_map


async def initialize_risk_manager() -> MarginGuard:
    """
    Initialize the risk manager.

    Returns:
        MarginGuard instance
    """
    logger.info("Initializing Risk Manager")

    # Create margin guard
    margin_guard = MarginGuard(
        gateways=gateways,
        warning_threshold=config.risk.margin_ratio_warning_threshold,
        critical_threshold=config.risk.margin_ratio_critical_threshold,
        check_interval=300,  # 5 minutes
    )

    # Start margin guard
    await margin_guard.start()

    return margin_guard


async def initialize_task_manager(
    portfolio: Portfolio, strategies: dict[StrategyType, Strategy]
) -> TaskManager:
    """
    Initialize the task scheduler.

    Args:
        portfolio: Portfolio instance
        strategies: Dictionary of strategy instances

    Returns:
        TaskManager instance
    """
    logger.info("Initializing Task Manager")

    # Create task manager
    task_manager = TaskManager(portfolio=portfolio, strategies=strategies, gateways=gateways)

    # Register strategy tasks
    if StrategyType.BASIS_HARVEST in strategies:
        task_manager.register_strategy_task(
            StrategyType.BASIS_HARVEST, "0 0 * * *"  # Daily at midnight
        )
        task_manager.register_profit_harvest_task(
            StrategyType.BASIS_HARVEST, "0 0 * * 1"  # Weekly on Monday
        )

    if StrategyType.FUNDING_CAPTURE in strategies:
        task_manager.register_strategy_task(
            StrategyType.FUNDING_CAPTURE, "*/10 * * * *"  # Every 10 minutes
        )
        task_manager.register_profit_harvest_task(
            StrategyType.FUNDING_CAPTURE, "0 0 1 * *"  # Monthly on the 1st
        )

    if StrategyType.OPTION_PREMIUM in strategies:
        task_manager.register_strategy_task(
            StrategyType.OPTION_PREMIUM, "0 0 1 * *"  # Monthly on the 1st
        )
        task_manager.register_profit_harvest_task(
            StrategyType.OPTION_PREMIUM, "0 0 1 * *"  # Monthly on the 1st
        )

    # Start task manager
    await task_manager.start()

    return task_manager


async def shutdown() -> None:
    """
    Perform a clean shutdown of all components.
    """
    global running  # 'running' is assigned to, so 'global' is appropriate here.
    # For module-level 'gateways', 'strategies', 'task_manager', 'margin_guard',
    # 'global' is not needed in this function as they are only being read.

    logger.info("Shutting down BTC Stack-Builder Bot")
    running = False

    # Stop task manager
    if task_manager:
        logger.info("Stopping Task Manager")
        await task_manager.stop()

    # Stop margin guard
    if margin_guard:
        logger.info("Stopping Risk Manager")
        await margin_guard.stop()

    # Close gateways
    for name, gateway in gateways.items():
        logger.info(f"Closing {name} gateway")
        await gateway.close()

    logger.info("Shutdown complete")


def handle_signal(sig, frame) -> None:
    """
    Handle termination signals.
    """
    logger.info(f"Received signal {sig}, initiating shutdown")
    asyncio.create_task(shutdown())


async def main_async(args: argparse.Namespace) -> int:
    """
    Asynchronous main function.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    # This function (main_async) assigns to module-level globals,
    # so 'global' keyword is needed here for them.
    global gateways, strategies, task_manager, margin_guard

    try:
        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handle_signal)

        # Initialize database
        logger.info("Initializing database")
        await init_db()

        # Initialize exchange gateways
        gateways = await initialize_gateways()
        if not gateways:
            logger.error("No exchange gateways were initialized")
            return 1

        # Initialize portfolio
        portfolio = await initialize_portfolio()

        # Initialize strategies
        strategies = await initialize_strategies(portfolio)

        # Initialize risk manager
        margin_guard = await initialize_risk_manager()

        # Initialize task manager
        task_manager = await initialize_task_manager(portfolio, strategies)

        # Start metrics server if not in dry run mode
        if not config.dry_run and not args.no_metrics:
            logger.info(f"Starting metrics server on port {args.metrics_port}")
            setup_metrics(gateways, strategies, portfolio)
            register_metrics()
            start_http_server(args.metrics_port)

        logger.info(
            f"BTC Stack-Builder Bot v{__version__} started in " f"{config.environment.value} mode"
        )

        # Keep the main task running until shutdown
        while running:
            await asyncio.sleep(1)

        return 0

    except Exception:
        logger.critical("Unhandled exception in main loop", exc_info=True)
        await shutdown()
        return 1


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="BTC Stack-Builder Bot")

    parser.add_argument(
        "--version",
        action="version",
        version=f"BTC Stack-Builder Bot v{__version__}",
    )

    parser.add_argument("--config-dir", type=str, help="Path to configuration directory")

    parser.add_argument(
        "--environment",
        type=str,
        choices=[e.value for e in Environment],
        help="Environment to run in (development, testnet, production)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trades)",
    )

    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics server")

    parser.add_argument("--metrics-port", type=int, default=8000, help="Port for metrics server")

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code
    """
    # Parse command-line arguments
    args = parse_args()

    # Set environment variables from arguments
    if args.config_dir:
        os.environ["BTC_STACK_BUILDER_CONFIG_DIR"] = args.config_dir

    if args.environment:
        os.environ["BTC_STACK_BUILDER_ENVIRONMENT"] = args.environment

    if args.log_level:
        os.environ["BTC_STACK_BUILDER_LOG_LEVEL"] = args.log_level

    if args.dry_run:
        os.environ["BTC_STACK_BUILDER_DRY_RUN"] = "true"

    # Set up logging
    logger = setup_logger()

    # Log startup information
    logger.info(
        f"Starting BTC Stack-Builder Bot v{__version__}",
        environment=config.environment.value,
        dry_run=config.dry_run,
    )

    # Use uvloop for better performance
    uvloop.install()

    # Run the async main function
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    from decimal import Decimal  # Required for portfolio initialization

    sys.exit(main())
