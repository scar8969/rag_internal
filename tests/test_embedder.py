from unittest.mock import MagicMock, patch

import pytest


class TestEmbedder:
    def test_embedder_provider_registry(self):
        from rag.embedder import Embedder

        assert "openai" in Embedder.PROVIDERS
        assert "cohere" in Embedder.PROVIDERS
        assert "huggingface" in Embedder.PROVIDERS
        assert "ollama" in Embedder.PROVIDERS

    def test_unknown_provider_raises(self):
        from rag.embedder import Embedder

        with pytest.raises(ValueError, match="Unknown provider"):
            Embedder(provider="nonexistent")

    @patch("rag.embedder.OpenAIEmbedder")
    def test_openai_embedder_init(self, mock_openai):
        from rag.embedder import Embedder

        mock_instance = MagicMock()
        mock_instance.dimensions = 1536
        mock_openai.return_value = mock_instance

        embedder = Embedder(provider="openai", api_key="test-key", model="text-embedding-ada-002")
        assert embedder.dimensions == 1536

    @patch("rag.embedder.HuggingFaceEmbedder")
    def test_huggingface_embedder_init(self, mock_hf):
        from rag.embedder import Embedder

        mock_instance = MagicMock()
        mock_instance.dimensions = 384
        mock_hf.return_value = mock_instance

        embedder = Embedder(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2")
        assert embedder.dimensions == 384


class TestOpenAIEmbedder:
    @patch("rag.embedder.OpenAI")
    def test_embed_returns_list(self, mock_openai):
        from rag.embedder import OpenAIEmbedder

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 1536

    @patch("rag.embedder.OpenAI")
    def test_embed_batch(self, mock_openai):
        from rag.embedder import OpenAIEmbedder

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [[0.1] * 1536, [0.2] * 1536]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key")
        results = embedder.embed_batch(["text1", "text2"])

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_dimensions_property(self):
        from rag.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="test-key", model="text-embedding-ada-002")
        assert embedder.dimensions == 1536

        embedder3 = OpenAIEmbedder(api_key="test-key", model="text-embedding-3-small")
        assert embedder3.dimensions == 1536

        embedder_large = OpenAIEmbedder(api_key="test-key", model="text-embedding-3-large")
        assert embedder_large.dimensions == 3072


class TestOllamaEmbedder:
    @patch("rag.embedder.httpx.Client")
    def test_ollama_embed(self, mock_httpx):
        from rag.embedder import OllamaEmbedder

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 4096}
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        embedder = OllamaEmbedder()
        result = embedder.embed("test")

        assert len(result) == 4096
        mock_client.post.assert_called_once()

    def test_ollama_dimensions(self):
        from rag.embedder import OllamaEmbedder

        embedder = OllamaEmbedder()
        assert embedder.dimensions == 4096
