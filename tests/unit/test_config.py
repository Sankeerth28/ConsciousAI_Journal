"""Unit tests for application configuration."""

from __future__ import annotations

import pytest

from app.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Tests for default configuration values."""

    def test_default_app_name(self):
        settings = Settings()
        assert settings.app_name == "ConsciousAI Journal"

    def test_default_app_env(self):
        settings = Settings()
        assert settings.app_env == "development"

    def test_default_debug(self):
        settings = Settings()
        assert settings.debug is True

    def test_default_ai_provider(self):
        settings = Settings()
        assert settings.ai_provider == "mock"

    def test_default_log_level(self):
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_database_url(self):
        settings = Settings()
        assert "sqlite" in settings.database_url

    def test_is_development(self):
        settings = Settings()
        assert settings.is_development is True
        assert settings.is_production is False


class TestSettingsValidation:
    """Tests for configuration validation."""

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_valid_log_levels_accepted(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            settings = Settings(log_level=level)
            assert settings.log_level == level

    def test_log_level_case_insensitive(self):
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_invalid_ai_provider_raises(self):
        with pytest.raises(ValueError):
            Settings(ai_provider="openai")

    def test_valid_ai_providers_accepted(self):
        for provider in ("mock", "huggingface"):
            settings = Settings(ai_provider=provider)
            assert settings.ai_provider == provider

    def test_ai_provider_case_insensitive(self):
        settings = Settings(ai_provider="MOCK")
        assert settings.ai_provider == "mock"


class TestCorsOrigins:
    """Tests for CORS origins parsing."""

    def test_single_origin(self):
        settings = Settings(cors_origins="http://localhost:3000")
        assert settings.cors_origins_list == ["http://localhost:3000"]

    def test_multiple_origins(self):
        settings = Settings(cors_origins="http://localhost:3000,http://localhost:5173")
        assert settings.cors_origins_list == ["http://localhost:3000", "http://localhost:5173"]

    def test_origins_with_whitespace(self):
        settings = Settings(cors_origins=" http://a.com , http://b.com ")
        assert settings.cors_origins_list == ["http://a.com", "http://b.com"]

    def test_empty_origins(self):
        settings = Settings(cors_origins="")
        assert settings.cors_origins_list == []


class TestEnvironmentOverride:
    """Tests for environment variable overrides."""

    def test_env_override_app_name(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_NAME", "Test App")
        settings = Settings()
        assert settings.app_name == "Test App"

    def test_env_override_debug(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEBUG", "false")
        settings = Settings()
        assert settings.debug is False

    def test_env_override_ai_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AI_PROVIDER", "huggingface")
        settings = Settings()
        assert settings.ai_provider == "huggingface"


class TestGetSettings:
    """Tests for the get_settings factory function."""

    def test_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_returns_fresh_instance_each_call(self):
        s1 = get_settings()
        s2 = get_settings()
        # Different instances (not cached by default)
        assert s1 is not s2


class TestProductionSafeguards:
    """Tests for production-mode security safeguards in Settings."""

    def test_production_rejects_debug_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )
        with pytest.raises(ValueError, match="DEBUG must be set to False"):
            Settings()

    def test_production_rejects_wildcard_cors(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )
        monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com,*")
        with pytest.raises(ValueError, match="cannot contain wildcard"):
            Settings()

    def test_production_rejects_invalid_jwt_algorithm(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )
        monkeypatch.setenv("JWT_ALGORITHM", "RS256")
        with pytest.raises(ValueError, match="JWT_ALGORITHM must be 'HS256'"):
            Settings()

    def test_production_rejects_placeholder_secret(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production-min-32-chars")
        with pytest.raises(ValueError, match="cannot use known placeholder"):
            Settings()

    def test_production_accepts_valid_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )
        monkeypatch.setenv(
            "CORS_ORIGINS", "https://consciousai.app,https://journal.consciousai.app"
        )
        settings = Settings()
        assert settings.is_production is True
        assert settings.debug is False
        assert settings.jwt_algorithm == "HS256"

    def test_staging_rejects_debug_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "staging_super_secret_key_that_is_at_least_32_chars_long!"
        )
        with pytest.raises(ValueError, match="DEBUG must be set to False"):
            Settings()

    def test_staging_rejects_weak_secret(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "short")
        with pytest.raises(ValueError, match="at least 32 characters"):
            Settings()

    def test_staging_accepts_valid_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "staging_super_secret_key_that_is_at_least_32_chars_long!"
        )
        settings = Settings()
        assert settings.is_staging is True
        assert settings.is_production_like is True
        assert settings.debug is False

    def test_invalid_app_env_rejected(self):
        with pytest.raises(ValueError, match="Invalid app_env"):
            Settings(app_env="invalid_environment")

    def test_database_pool_and_server_settings_defaults(self):
        settings = Settings()
        assert settings.db_pool_size == 10
        assert settings.db_max_overflow == 20
        assert settings.db_pool_timeout == 30
        assert settings.db_pool_recycle == 1800
        assert settings.workers == 1
        assert settings.graceful_timeout == 30
        assert settings.trusted_proxies == "127.0.0.1"
