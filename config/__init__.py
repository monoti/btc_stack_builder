"""
Configuration module for BTC Stack-Builder Bot.

This module provides utilities for loading, validating, and accessing
configuration settings from YAML files and environment variables.
"""
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union, Type, TypeVar, cast

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from btc_stack_builder.core.logger import logger


# Type variable for configuration models
T = TypeVar('T', bound=BaseSettings)


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTNET = "testnet"
    PRODUCTION = "production"


class ExchangeCredentials(BaseModel):
    """API credentials for exchange access."""
    api_key: SecretStr
    api_secret: SecretStr
    passphrase: Optional[SecretStr] = None
    is_testnet: bool = False


class BinanceConfig(BaseModel):
    """Binance-specific configuration."""
    enabled: bool = True
    credentials: ExchangeCredentials
    base_url: Optional[str] = None
    use_testnet: bool = False
    futures_base_url: Optional[str] = None
    rate_limit_requests: int = 1200  # requests per minute
    rate_limit_orders: int = 10  # orders per second


class DeribitConfig(BaseModel):
    """Deribit-specific configuration."""
    enabled: bool = True
    credentials: ExchangeCredentials
    base_url: Optional[str] = None
    use_testnet: bool = False
    rate_limit_requests: int = 300  # requests per minute
    rate_limit_orders: int = 5  # orders per second


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    username: str
    password: SecretStr
    database: str = "btc_stack_builder"
    pool_size: int = 5
    max_overflow: int = 10
    ssl_mode: Optional[str] = None
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        password = self.password.get_secret_value()
        return f"postgresql://{self.username}:{password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[SecretStr] = None
    ssl: bool = False
    
    @property
    def connection_string(self) -> str:
        """Generate Redis connection string."""
        if self.password:
            password = self.password.get_secret_value()
            auth = f":{password}@"
        else:
            auth = ""
        
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class TelegramAlertsConfig(BaseModel):
    """Telegram alerts configuration."""
    enabled: bool = False
    bot_token: Optional[SecretStr] = None
    chat_id: Optional[str] = None


class RiskConfig(BaseModel):
    """Risk management configuration."""
    global_stop_loss_threshold: float = -0.70  # 70% price movement
    margin_ratio_warning_threshold: float = 4.50  # 450%
    margin_ratio_critical_threshold: float = 4.00  # 400%
    max_position_size_btc: float = 10.0
    max_position_size_percentage: float = 0.25  # 25% of portfolio
    max_leverage: float = 3.0
    max_drawdown: float = 0.20  # 20% drawdown


class BasisHarvestConfig(BaseModel):
    """Basis harvest strategy configuration."""
    enabled: bool = True
    entry_threshold: float = 0.05  # 5% annualized basis
    max_leverage: float = 1.5
    roll_start_days: int = 21
    roll_end_days: int = 14


class FundingCaptureConfig(BaseModel):
    """Funding capture strategy configuration."""
    enabled: bool = True
    entry_threshold: float = -0.0001  # -0.01% funding rate
    max_leverage: float = 2.0
    profit_target: float = 0.12  # 12% profit target


class OptionPremiumConfig(BaseModel):
    """Option premium strategy configuration."""
    enabled: bool = True
    delta_target: float = 0.20
    min_expiry_days: int = 60
    max_expiry_days: int = 90


class PortfolioConfig(BaseModel):
    """Portfolio configuration."""
    core_hodl_allocation: float = 0.60  # 60%
    basis_harvest_allocation: float = 0.25  # 25%
    funding_capture_allocation: float = 0.10  # 10%
    option_premium_allocation: float = 0.05  # 5%
    cold_wallet_address: str


class AppConfig(BaseSettings):
    """Main application configuration."""
    model_config = SettingsConfigDict(
        env_prefix="BTC_STACK_BUILDER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Application settings
    app_name: str = "BTC Stack-Builder Bot"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    
    # Portfolio configuration
    portfolio: PortfolioConfig
    
    # Exchange configurations
    binance: BinanceConfig
    deribit: DeribitConfig
    
    # Strategy configurations
    basis_harvest: BasisHarvestConfig = Field(default_factory=BasisHarvestConfig)
    funding_capture: FundingCaptureConfig = Field(default_factory=FundingCaptureConfig)
    option_premium: OptionPremiumConfig = Field(default_factory=OptionPremiumConfig)
    
    # Risk configuration
    risk: RiskConfig = Field(default_factory=RiskConfig)
    
    # Infrastructure configurations
    database: DatabaseConfig
    redis: RedisConfig = Field(default_factory=RedisConfig)
    
    # Monitoring and alerting
    telegram_alerts: TelegramAlertsConfig = Field(default_factory=TelegramAlertsConfig)
    
    # Additional settings
    dry_run: bool = False  # If True, don't execute actual trades


def get_config_dir() -> Path:
    """
    Get the configuration directory path.
    
    Returns:
        Path to the configuration directory
    """
    # Check for custom config directory in environment variable
    env_config_dir = os.environ.get("BTC_STACK_BUILDER_CONFIG_DIR")
    if env_config_dir:
        config_dir = Path(env_config_dir)
    else:
        # Default to config directory in project root
        config_dir = Path(__file__).parent.parent.parent / "config"
    
    # Ensure directory exists
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
    
    return config_dir


def get_environment() -> Environment:
    """
    Get the current environment from environment variable or default to development.
    
    Returns:
        Current environment enum
    """
    env_name = os.environ.get("BTC_STACK_BUILDER_ENVIRONMENT", "development").lower()
    
    try:
        return Environment(env_name)
    except ValueError:
        logger.warning(f"Invalid environment '{env_name}', using DEVELOPMENT")
        return Environment.DEVELOPMENT


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {file_path}", exc_info=True)
        raise e


def load_config(
    config_class: Type[T],
    config_name: str,
    environment: Optional[Environment] = None
) -> T:
    """
    Load and validate configuration for a specific component.
    
    Args:
        config_class: Pydantic model class for configuration validation
        config_name: Name of the configuration file (without extension)
        environment: Environment to load configuration for (default: current environment)
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValidationError: If the configuration is invalid
    """
    if environment is None:
        environment = get_environment()
    
    config_dir = get_config_dir()
    
    # Base configuration file (required)
    base_config_path = config_dir / f"{config_name}.yaml"
    
    # Environment-specific configuration file (optional)
    env_config_path = config_dir / f"{config_name}.{environment.value}.yaml"
    
    # Load base configuration
    try:
        config_data = load_yaml_config(base_config_path)
    except FileNotFoundError:
        if config_name == "app":
            logger.error(f"Base configuration file not found: {base_config_path}")
            sys.exit(1)
        else:
            # For component configs, use empty dict if file not found
            config_data = {}
    
    # Load and merge environment-specific configuration if it exists
    if env_config_path.exists():
        env_config_data = load_yaml_config(env_config_path)
        config_data = deep_merge(config_data, env_config_data)
    
    # Create and validate configuration object
    try:
        return config_class(**config_data)
    except ValidationError as e:
        logger.error(f"Invalid configuration in {config_name}: {e}")
        raise


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with values from override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override or add value
            result[key] = value
    
    return result


def load_app_config() -> AppConfig:
    """
    Load the main application configuration.
    
    Returns:
        Validated AppConfig object
    """
    return load_config(AppConfig, "app")


# Create and export a global configuration instance
config = load_app_config()
