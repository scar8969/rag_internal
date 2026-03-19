import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self._config: dict[str, Any] = {}
        self._config_path = config_path
        if config_path and Path(config_path).exists():
            self._load_from_file(config_path)
        else:
            self._load_defaults()

    def _load_defaults(self) -> None:
        self._config = {
            "app": {
                "name": "RAG Internal Tool",
                "log_level": "INFO",
            },
            "database": {
                "url": os.getenv("DATABASE_URL", "sqlite:///./data/app.db"),
            },
            "embedding": {
                "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
                "model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
                "dimensions": 1536,
                "batch_size": 100,
            },
            "vector_store": {
                "type": "chroma",
                "persist_directory": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
                "collection_name": "documents",
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "max_tokens": 2000,
            },
            "generation": {
                "provider": os.getenv("LLM_PROVIDER", "openai"),
                "model": os.getenv("LLM_MODEL", "gpt-4"),
                "temperature": 0.7,
                "max_tokens": 2000,
                "system_prompt": "You are a helpful assistant. Use the provided context to answer the user's question. If the answer cannot be determined from the context, say so clearly.",
            },
            "auth": {
                "jwt_secret": os.getenv("JWT_SECRET", "change-me-in-production"),
                "jwt_algorithm": "HS256",
                "jwt_expiration_hours": 24,
                "jwt_refresh_days": 7,
                "bcrypt_rounds": 12,
            },
            "rate_limit": {
                "per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            },
        }

    def _load_from_file(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            file_config = yaml.safe_load(f)
        self._load_defaults()
        self._merge_config(file_config)

    def _merge_config(self, new_config: dict[str, Any]) -> None:
        for key, value in new_config.items():
            if key in self._config and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def embedding_provider(self) -> str:
        return self.get("embedding.provider", "openai")

    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model", "text-embedding-ada-002")

    @property
    def embedding_dimensions(self) -> int:
        return self.get("embedding.dimensions", 1536)

    @property
    def vector_store_type(self) -> str:
        return self.get("vector_store.type", "chroma")

    @property
    def vector_store_dir(self) -> str:
        return self.get("vector_store.persist_directory", "./data/chroma")

    @property
    def collection_name(self) -> str:
        return self.get("vector_store.collection_name", "documents")

    @property
    def retrieval_top_k(self) -> int:
        return self.get("retrieval.top_k", 5)

    @property
    def retrieval_similarity_threshold(self) -> float:
        return self.get("retrieval.similarity_threshold", 0.7)

    @property
    def retrieval_max_tokens(self) -> int:
        return self.get("retrieval.max_tokens", 2000)

    @property
    def generation_provider(self) -> str:
        return self.get("generation.provider", "openai")

    @property
    def generation_model(self) -> str:
        return self.get("generation.model", "gpt-4")

    @property
    def generation_temperature(self) -> float:
        return self.get("generation.temperature", 0.7)

    @property
    def generation_max_tokens(self) -> int:
        return self.get("generation.max_tokens", 2000)

    @property
    def system_prompt(self) -> str:
        return self.get("generation.system_prompt", "")

    @property
    def jwt_secret(self) -> str:
        return self.get("auth.jwt_secret", "change-me")

    @property
    def jwt_algorithm(self) -> str:
        return self.get("auth.jwt_algorithm", "HS256")

    @property
    def jwt_expiration_hours(self) -> int:
        return self.get("auth.jwt_expiration_hours", 24)

    @property
    def jwt_refresh_days(self) -> int:
        return self.get("auth.jwt_refresh_days", 7)

    @property
    def bcrypt_rounds(self) -> int:
        return self.get("auth.bcrypt_rounds", 12)

    @property
    def rate_limit_per_minute(self) -> int:
        return self.get("rate_limit.per_minute", 60)

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

    @property
    def openrouter_api_key(self) -> Optional[str]:
        return os.getenv("OPENROUTER_API_KEY")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def ollama_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
