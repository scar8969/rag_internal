import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestVectorStore:
    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_init_creates_collection(self, mock_client):
        from rag.vector_store import VectorStore

        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(persist_directory="./test_data", collection_name="test")
        assert mock_client.called

    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_add_chunks(self, mock_client):
        from rag.vector_store import DocumentChunk, VectorStore

        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(persist_directory="./test_data", collection_name="test")

        chunks = [
            DocumentChunk(
                id="chunk1",
                text="Test text 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"source": "test.txt"},
            ),
            DocumentChunk(
                id="chunk2",
                text="Test text 2",
                embedding=[0.4, 0.5, 0.6],
                metadata={"source": "test.txt"},
            ),
        ]

        store.add(chunks)
        mock_collection.upsert.assert_called_once()

    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_search(self, mock_client):
        from rag.vector_store import VectorStore

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["Text 1", "Text 2"]],
            "metadatas": [{"source": "test.txt"}, {"source": "test.txt"}],
            "distances": [[0.1, 0.2]],
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(persist_directory="./test_data", collection_name="test")
        results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=2)

        assert len(results) == 2
        assert results[0].text == "Text 1"

    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_delete_by_source(self, mock_client):
        from rag.vector_store import VectorStore

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["chunk1"],
            "documents": ["Text"],
            "metadatas": [{"source": "test.txt"}],
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(persist_directory="./test_data", collection_name="test")
        store.delete_by_source("test.txt")

        mock_collection.delete.assert_called_once_with(ids=["chunk1"])

    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_count(self, mock_client):
        from rag.vector_store import VectorStore

        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(persist_directory="./test_data", collection_name="test")
        assert store.count() == 10

    @patch("rag.vector_store.chromadb.PersistentClient")
    def test_list_sources(self, mock_client):
        from rag.vector_store import VectorStore

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["c1", "c2"],
            "metadatas": [
                {"source": "file1.txt"},
                {"source": "file2.txt"},
            ],
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = VectorStore(persist_directory="./test_data", collection_name="test")
        sources = store.list_sources()

        assert "file1.txt" in sources
        assert "file2.txt" in sources


class TestGenerateIds:
    def test_generate_chunk_id(self):
        from rag.vector_store import generate_chunk_id

        chunk_id = generate_chunk_id("doc1", 0)
        assert chunk_id.startswith("doc1_0_")
        assert len(chunk_id) > 10

    def test_generate_document_id(self):
        from rag.vector_store import generate_document_id

        doc_id = generate_document_id("https://example.com/test")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
