import os
import tempfile
from pathlib import Path

import pytest
import yaml


class TestConfig:
    def test_load_defaults(self):
        from rag.config import Config

        config = Config()
        assert config.get("app.name") == "RAG System"
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.vector_store_dir == "./data/chroma"
        assert config.retrieval_top_k == 5
        assert config.generation_provider == "openai"
        assert config.generation_model == "gpt-4"

    def test_load_from_yaml(self):
        from rag.config import Config

        config_data = {
            "app": {"name": "Test RAG"},
            "embedding": {"provider": "cohere", "model": "embed-multilingual-v3.0"},
            "generation": {"model": "gpt-3.5-turbo"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = Config(config_path)
            assert config.get("app.name") == "Test RAG"
            assert config.embedding_provider == "cohere"
            assert config.embedding_model == "embed-multilingual-v3.0"
            assert config.generation_model == "gpt-3.5-turbo"
        finally:
            Path(config_path).unlink()

    def test_get_property(self):
        from rag.config import Config

        config = Config()
        assert config.retrieval_top_k == 5
        assert config.retrieval_similarity_threshold == 0.7
        assert config.generation_temperature == 0.7

    def test_set_property(self):
        from rag.config import Config

        config = Config()
        config.set("app.name", "New Name")
        assert config.get("app.name") == "New Name"

    def test_api_key_fallback(self):
        from rag.config import Config

        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENROUTER_API_KEY"] = "test-router-key"

        config = Config()
        assert config.get_api_key("openai") == "test-router-key"

        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        assert config.get_api_key("openai") == "test-openai-key"

    def test_anthropic_fallback(self):
        from rag.config import Config

        os.environ["ANTHROPIC_API_KEY"] = ""
        os.environ["OPENROUTER_API_KEY"] = "test-router-key"

        config = Config()
        assert config.get_api_key("anthropic") == "test-router-key"

    def test_ollama_config(self):
        from rag.config import Config

        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

        config = Config()
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.get_api_key("ollama") == "http://localhost:11434"
