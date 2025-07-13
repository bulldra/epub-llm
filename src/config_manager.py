"""
Configuration manager for Smart RAG system and application settings.
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
            self.logger.info("Loaded app configuration from %s", self.config_path)
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
            "llm": {
                "dev_mode": False,
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "embedding_model_name": "mlx-community/bge-small-en-v1.5-mlx",
                "generation": {
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                },
            },
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
                "files": {"app_log": "epub-llm.log", "server_log": "server.log"},
            },
            "cache": {
                "embeddings_cache": True,
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
                self.logger.info(
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
        self.logger.info("App configuration reloaded")


class SmartRAGConfig:
    """Configuration manager for Smart RAG parameters."""

    def __init__(self, config_path: str | None = None):
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            # デフォルトの設定ファイルパス
            config_path = os.path.join(
                os.path.dirname(__file__), "../config/smart_rag_config.yaml"
            )

        self.config_path = config_path
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            self.logger.info("Loaded configuration from %s", self.config_path)
        except (OSError, yaml.YAMLError) as e:
            self.logger.warning(
                "Failed to load config from %s: %s. Using defaults.",
                self.config_path,
                e,
            )
            self._config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "hybrid_search": {
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "default_top_k": 10,
                "candidate_multiplier": 2,
            },
            "bm25": {"k1": 1.2, "b": 0.75, "epsilon": 0.25},
            "chunking": {
                "chunk_size": 4000,
                "overlap": 500,
                "sentence_boundary_search": 200,
            },
            "reranking": {
                "diversity_weight": 0.2,
                "quality_weight": 0.1,
                "overlap_weight": 0.3,
                "similarity_threshold": 0.7,
            },
            "query_expansion": {
                "max_synonym_queries": 3,
                "max_search_queries": 5,
                "llm_expansion_max_length": 100,
            },
            "context_compression": {
                "max_context_length": 8000,
                "max_sentences_per_result": 3,
                "results_per_book": 3,
            },
            "book_weighting": {
                "title_match_bonus": 0.3,
                "author_match_bonus": 0.2,
                "recent_book_bonus": 0.1,
                "historical_book_bonus": 0.05,
                "max_weight": 2.0,
            },
            "adaptive_strategy": {
                "factual": {
                    "semantic_weight": 0.5,
                    "keyword_weight": 0.5,
                    "top_k": 5,
                    "diversity_weight": 0.1,
                },
                "procedural": {
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3,
                    "top_k": 15,
                    "diversity_weight": 0.3,
                    "max_context_length": 10000,
                },
                "explanatory": {
                    "semantic_weight": 0.8,
                    "keyword_weight": 0.2,
                    "top_k": 10,
                    "diversity_weight": 0.2,
                },
                "comparison": {
                    "semantic_weight": 0.7,
                    "keyword_weight": 0.3,
                    "top_k": 15,
                    "diversity_weight": 0.4,
                },
            },
            "specificity_adjustments": {
                "high": {"top_k_reduction": 2, "keyword_weight_boost": 0.1},
                "low": {"top_k_increase": 2, "diversity_weight_boost": 0.1},
            },
            "performance": {
                "cache_embeddings": True,
                "cache_bm25_index": True,
                "parallel_book_processing": True,
            },
            "logging": {
                "level": "INFO",
                "detailed_search_logs": True,
                "performance_logs": True,
            },
            "ui": {
                "show_relevance_scores": True,
                "show_book_weights": True,
                "use_fire_emoji": True,
                "max_title_display_length": 50,
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path."""
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default

            # Try to get from default config
            default_config = self._get_default_config()
            value = default_config
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return None

    def get_hybrid_search_params(self) -> dict[str, Any]:
        """Get hybrid search parameters."""
        return {
            "semantic_weight": self.get("hybrid_search.semantic_weight", 0.7),
            "keyword_weight": self.get("hybrid_search.keyword_weight", 0.3),
            "top_k": self.get("hybrid_search.default_top_k", 10),
            "candidate_multiplier": self.get("hybrid_search.candidate_multiplier", 2),
        }

    def get_bm25_params(self) -> dict[str, float]:
        """Get BM25 parameters."""
        return {
            "k1": self.get("bm25.k1", 1.2),
            "b": self.get("bm25.b", 0.75),
            "epsilon": self.get("bm25.epsilon", 0.25),
        }

    def get_chunking_params(self) -> dict[str, int]:
        """Get text chunking parameters."""
        return {
            "chunk_size": self.get("chunking.chunk_size", 4000),
            "overlap": self.get("chunking.overlap", 500),
            "sentence_boundary_search": self.get(
                "chunking.sentence_boundary_search", 200
            ),
        }

    def get_reranking_params(self) -> dict[str, float]:
        """Get re-ranking parameters."""
        return {
            "diversity_weight": self.get("reranking.diversity_weight", 0.2),
            "quality_weight": self.get("reranking.quality_weight", 0.1),
            "overlap_weight": self.get("reranking.overlap_weight", 0.3),
            "similarity_threshold": self.get("reranking.similarity_threshold", 0.7),
        }

    def get_query_expansion_params(self) -> dict[str, int]:
        """Get query expansion parameters."""
        return {
            "max_synonym_queries": self.get("query_expansion.max_synonym_queries", 3),
            "max_search_queries": self.get("query_expansion.max_search_queries", 5),
            "llm_expansion_max_length": self.get(
                "query_expansion.llm_expansion_max_length", 100
            ),
        }

    def get_context_compression_params(self) -> dict[str, int]:
        """Get context compression parameters."""
        return {
            "max_context_length": self.get(
                "context_compression.max_context_length", 8000
            ),
            "max_sentences_per_result": self.get(
                "context_compression.max_sentences_per_result", 3
            ),
            "results_per_book": self.get("context_compression.results_per_book", 3),
        }

    def get_book_weighting_params(self) -> dict[str, float]:
        """Get book weighting parameters."""
        return {
            "title_match_bonus": self.get("book_weighting.title_match_bonus", 0.3),
            "author_match_bonus": self.get("book_weighting.author_match_bonus", 0.2),
            "recent_book_bonus": self.get("book_weighting.recent_book_bonus", 0.1),
            "historical_book_bonus": self.get(
                "book_weighting.historical_book_bonus", 0.05
            ),
            "max_weight": self.get("book_weighting.max_weight", 2.0),
        }

    def get_adaptive_strategy_params(self, query_type: str) -> dict[str, Any]:
        """Get adaptive strategy parameters for specific query type."""
        base_params = self.get_hybrid_search_params()

        type_params = self.get(f"adaptive_strategy.{query_type}", {})
        if type_params:
            base_params.update(type_params)

        return base_params

    def get_specificity_adjustments(self, specificity: str) -> dict[str, Any]:
        """Get specificity-based adjustments."""
        result = self.get(f"specificity_adjustments.{specificity}", {})
        return dict(result) if result else {}

    def get_ui_params(self) -> dict[str, Any]:
        """Get UI parameters."""
        return {
            "show_relevance_scores": self.get("ui.show_relevance_scores", True),
            "show_book_weights": self.get("ui.show_book_weights", True),
            "use_fire_emoji": self.get("ui.use_fire_emoji", True),
            "max_title_display_length": self.get("ui.max_title_display_length", 50),
        }

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self.logger.info("Configuration reloaded")

    def save_config(self, config_path: str | None = None) -> None:
        """Save current configuration to file."""
        save_path = config_path or self.config_path

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info("Configuration saved to %s", save_path)
        except (OSError, yaml.YAMLError) as e:
            self.logger.error("Failed to save config to %s: %s", save_path, e)

    def update_config(self, key_path: str, value: Any) -> None:
        """Update configuration value."""
        keys = key_path.split(".")
        config = self._config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value
        self.logger.info("Updated config: %s = %s", key_path, value)


# Global config instance
_config_instance: SmartRAGConfig | None = None


def get_config() -> SmartRAGConfig:
    """Get global configuration instance."""
    if _config_instance is None:
        globals()["_config_instance"] = SmartRAGConfig()
    # 型安全のため再取得
    instance = _config_instance
    if instance is None:
        # ここには到達しないはずだが型保証
        instance = SmartRAGConfig()
        globals()["_config_instance"] = instance
    return instance


def reload_config() -> None:
    """Reload global configuration."""
    if _config_instance is not None:
        _config_instance.reload_config()
    else:
        globals()["_config_instance"] = SmartRAGConfig()
