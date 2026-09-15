"""Centralized application configuration using Pydantic Settings.

All values are loaded from environment variables (with .env file support).
Safe defaults allow the app to run locally without GPU, paid APIs, or external services.
"""

from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "ConsciousAI Journal"
    app_env: str = "development"
    app_version: str = "2.0.0a1"
    debug: bool = True

    # --- Database ---
    database_url: str = "sqlite:///./data/consciousai.db"

    # --- Logging ---
    log_level: str = "INFO"

    # --- AI Provider ---
    ai_provider: str = "mock"

    # --- AI Model Names ---
    llm_model_name: str = "google/flan-t5-large"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    emotion_model_name: str = "facebook/bart-large-mnli"
    value_model_name: str = "facebook/bart-large-mnli"

    # --- HuggingFace ---
    hf_token: str = ""

    # --- CORS ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- Authentication & JWT ---
    jwt_secret_key: str = "consciousai-default-insecure-dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 1440  # 24 hours

    # --- Database Pooling & Performance ---
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # --- Server Execution ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    graceful_timeout: int = 30
    trusted_proxies: str = "127.0.0.1"

    # --- Redis & Distributed Caching ---
    redis_url: str | None = None

    # --- Rate Limiting ---
    rate_limit_enabled: bool = True
    rate_limit_login_per_minute: int = 10
    rate_limit_register_per_minute: int = 5

    # --- Security & Hardening ---
    security_headers_enabled: bool = True
    max_request_body_bytes: int = 1_048_576  # 1 MB
    enable_api_docs: bool = True

    @model_validator(mode="after")
    def validate_production_safeguards(self) -> Settings:
        """Validate that production and staging configurations satisfy strict security safeguards."""
        env_lower = self.app_env.lower()
        if env_lower in ("production", "staging"):
            # 1. Debug must be disabled in production and staging
            if self.debug:
                msg = f"In {env_lower} (APP_ENV={self.app_env}), DEBUG must be set to False."
                raise ValueError(msg)

            # 2. JWT Algorithm must be HS256
            if self.jwt_algorithm.upper() != "HS256":
                msg = (
                    f"In {env_lower} (APP_ENV={self.app_env}), JWT_ALGORITHM must be 'HS256', "
                    f"got '{self.jwt_algorithm}'."
                )
                raise ValueError(msg)

            # 3. JWT Secret Key strength and placeholder checks
            placeholder_keys = {
                "consciousai-default-insecure-dev-secret-change-in-production",
                "dev-secret-key-change-in-production-min-32-chars",
                "changethis",
                "secret",
                "placeholder",
            }
            if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
                msg = (
                    f"In {env_lower} (APP_ENV={self.app_env}), JWT_SECRET_KEY must be set to a secure "
                    "random secret of at least 32 characters and cannot use the default development key."
                )
                raise ValueError(msg)

            lower_secret = self.jwt_secret_key.lower()
            if (
                self.jwt_secret_key in placeholder_keys
                or "change-in-production" in lower_secret
                or "insecure" in lower_secret
            ):
                msg = (
                    f"In {env_lower} (APP_ENV={self.app_env}), JWT_SECRET_KEY cannot use known "
                    "placeholder or default development keys."
                )
                raise ValueError(msg)

            # 4. Wildcard CORS origins forbidden in production and staging
            if "*" in self.cors_origins_list:
                msg = f"In {env_lower} (APP_ENV={self.app_env}), CORS_ORIGINS cannot contain wildcard '*'."
                raise ValueError(msg)

        return self

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        allowed = {"development", "testing", "staging", "production"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"Invalid app_env '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return lower

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"Invalid log level '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return upper

    @field_validator("ai_provider")
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        allowed = {"mock", "huggingface"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"Invalid AI provider '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return lower

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    @property
    def is_testing(self) -> bool:
        return self.app_env.lower() == "testing"

    @property
    def is_staging(self) -> bool:
        return self.app_env.lower() == "staging"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def is_production_like(self) -> bool:
        return self.app_env.lower() in ("staging", "production")


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Each call creates a new instance so tests can override environment
    variables and get fresh settings. For production use, cache via
    FastAPI's dependency system.
    """
    return Settings()
