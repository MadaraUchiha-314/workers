"""Application settings with environment-based configuration.

This module provides settings management for the application with support for
different environments (local, e2e, prod). Settings can be overridden via
environment variables or secrets.json file.
"""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get the project root directory (where secrets.json is located)."""
    return Path(__file__).parent.parent.parent


def _load_secrets() -> dict[str, str]:
    """Load secrets from secrets.json file.

    Returns:
        Dictionary containing secrets, or empty dict if file not found.
    """
    secrets_path = _get_project_root() / "secrets.json"
    if secrets_path.exists():
        with secrets_path.open() as f:
            result: dict[str, str] = json.load(f)
            return result
    logger.warning("secrets.json not found at %s", secrets_path)
    return {}


class BaseSettings(BaseModel):
    """Base settings for all environments."""

    # Environment name
    environment: str = "local"

    # LLM settings
    llm_api_key: str = Field(default="ollama")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str = Field(default="gpt-oss:20b")

    # Logging settings
    log_level: str = Field(default="INFO")


class LocalSettings(BaseSettings):
    """Settings for local development environment.

    Uses Ollama as the default LLM backend.
    """

    environment: str = "local"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-5.2"


class E2ESettings(BaseSettings):
    """Settings for end-to-end testing environment.

    Uses Ollama for testing by default.
    """

    environment: str = "e2e"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"


class ProdSettings(BaseSettings):
    """Settings for production environment.

    Uses OpenAI API as the default LLM backend.
    Requires secrets.json or environment variables for API key.
    """

    environment: str = "prod"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"


# Mapping of environment names to settings classes
ENVIRONMENT_SETTINGS: dict[str, type[BaseSettings]] = {
    "local": LocalSettings,
    "e2e": E2ESettings,
    "prod": ProdSettings,
}


@lru_cache
def get_settings() -> BaseSettings:
    """Get settings for the current environment.

    The environment is determined by the APP_ENV environment variable.
    Settings are loaded in the following order (later overrides earlier):
    1. Default values from the settings class
    2. API key from secrets.json (openai-api-key -> llm_api_key)

    Returns:
        Settings instance for the current environment.
    """
    # Determine environment
    env_name = os.environ.get("APP_ENV", "local").lower()

    # Get the appropriate settings class
    settings_class = ENVIRONMENT_SETTINGS.get(env_name, LocalSettings)
    if env_name not in ENVIRONMENT_SETTINGS:
        logger.warning(
            "Unknown environment '%s', falling back to local settings", env_name
        )

    # Start with default settings
    settings_dict: dict[str, str] = {}

    # Load API key from secrets.json
    secrets = _load_secrets()
    if secrets.get("openai-api-key"):
        settings_dict["llm_api_key"] = secrets["openai-api-key"]

    # Create and return settings
    return settings_class(**settings_dict)
