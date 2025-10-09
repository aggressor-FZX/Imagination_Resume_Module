"""
Configuration management using pydantic_settings
Based on Context7 research findings for latest pydantic_settings API
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings using pydantic_settings for environment variable management.

    Based on Context7 research:
    - Automatic loading from .env files and environment variables
    - Full Pydantic validation and type conversion
    - Case-insensitive environment variable matching
    - Nested configuration support
    """

    # Required API Keys
    openai_api_key: str
    anthropic_api_key: str

    # Optional Context7 API Key for documentation
    context7_api_key: Optional[str] = None

    # Application Settings
    confidence_threshold: float = 0.7
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    max_request_size: int = 10 * 1024 * 1024  # 10MB

    # Environment
    environment: str = "development"
    debug: bool = False

    # Pricing Configuration (customizable)
    openai_price_input_per_1k: float = 0.0005
    openai_price_output_per_1k: float = 0.0015
    anthropic_price_input_per_1k: float = 0.003
    anthropic_price_output_per_1k: float = 0.015

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # CORS Settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra environment variables
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()