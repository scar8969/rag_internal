import sys
from unittest.mock import MagicMock, patch

import pytest

mock_openai = MagicMock()
mock_openai.OpenAI = MagicMock()

mock_chromadb = MagicMock()
mock_chroma_client = MagicMock()
mock_collection = MagicMock()
mock_chromadb.PersistentClient = MagicMock(return_value=mock_chroma_client)
mock_chromadb.config = MagicMock()
mock_chromadb.config.Settings = MagicMock()

sys.modules["openai"] = mock_openai
sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.config"] = mock_chromadb.config

httpx_mock = MagicMock()
sys.modules["httpx"] = httpx_mock


@pytest.fixture(autouse=True)
def mock_chroma_collection():
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col
    mock_chromadb.PersistentClient.return_value = mock_client
    
    yield mock_col


@pytest.fixture
def mock_rag_components():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 1536
    mock_embedder.dimensions = 1536
    
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Test response"
    
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    mock_store.list_sources.return_value = []
    mock_store.get_by_source.return_value = []
    mock_store.search.return_value = []
    mock_store.delete_all.return_value = None
    
    return {
        "embedder": mock_embedder,
        "llm": mock_llm,
        "store": mock_store,
    }
