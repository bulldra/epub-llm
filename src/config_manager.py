"""
Configuration manager for EPUB application settings.
"""

import logging
import os
from typing import Any

import yaml


class AppConfig:
    """Configuration manager for application settings."""

    def __init__(self, config_path: str | None = None):
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "../config/app_config.yaml"
            )

        self.config_path = config_path
        self._config: dict[str, Any] = {}
        self._load_config()
        self._apply_env_overrides()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            self.logger.debug("Loaded app configuration from %s", self.config_path)
        except (OSError, yaml.YAMLError) as e:
            self.logger.warning(
                "Failed to load app config from %s: %s. Using defaults.",
                self.config_path,
                e,
            )
            self._config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default application configuration."""
        return {
            "server": {"host": "0.0.0.0", "port": 8000},
            "directories": {
                "epub_dir": "epub",
                "cache_dir": "cache",
                "log_dir": "log",
                "config_dir": "config",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "file_encoding": "utf-8",
                "files": {"app_log": "epub-app.log", "server_log": "server.log"},
            },
            "cache": {
                "text_cache": True,
                "cover_cache": True,
            },
        }

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = self.get("env_overrides", {})

        for env_var, config_path in env_mappings.items():
            env_str_value = os.getenv(env_var)
            if env_str_value is not None:
                # 型変換
                env_value: str | bool | int | float
                if env_str_value.lower() in ("true", "false"):
                    env_value = env_str_value.lower() == "true"
                elif env_str_value.isdigit():
                    env_value = int(env_str_value)
                elif env_str_value.replace(".", "").isdigit():
                    env_value = float(env_str_value)
                else:
                    env_value = env_str_value

                self.set(config_path, env_value)
                self.logger.debug(
                    "Applied env override: %s = %s", config_path, env_value
                )

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path."""
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value by dot-separated key path."""
        keys = key_path.split(".")
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self._apply_env_overrides()
        self.logger.debug("App configuration reloaded")
