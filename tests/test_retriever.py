from unittest.mock import MagicMock, patch

import pytest


class TestRetriever:
    @patch("rag.retriever.Embedder")
    @patch("rag.retriever.VectorStore")
    def test_retrieve_returns_chunks(self, mock_store, mock_embedder):
        from rag.retriever import RetrievedChunk, Retriever

        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed.return_value = [0.1, 0.2, 0.3]
        mock_embedder.return_value = mock_embedder_instance

        mock_store_instance = MagicMock()
        mock_store_instance.search.return_value = [
            MagicMock(
                text="Test chunk 1",
                metadata={"source": "test.txt", "distance": 0.1},
            ),
            MagicMock(
                text="Test chunk 2",
                metadata={"source": "test.txt", "distance": 0.2},
            ),
        ]
        mock_store.return_value = mock_store_instance

        retriever = Retriever(
            embedder=mock_embedder_instance,
            vector_store=mock_store_instance,
            top_k=5,
            similarity_threshold=0.7,
        )

        results = retriever.retrieve("test query")

        assert len(results) == 2
        assert results[0].text == "Test chunk 1"
        assert results[0].source == "test.txt"

    @patch("rag.retriever.Embedder")
    @patch("rag.retriever.VectorStore")
    def test_retrieve_with_threshold(self, mock_store, mock_embedder):
        from rag.retriever import Retriever

        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed.return_value = [0.1]
        mock_embedder.return_value = mock_embedder_instance

        mock_store_instance = MagicMock()
        mock_store_instance.search.return_value = [
            MagicMock(text="Low similarity", metadata={"source": "a.txt", "distance": 0.8}),
        ]
        mock_store.return_value = mock_store_instance

        retriever = Retriever(
            embedder=mock_embedder_instance,
            vector_store=mock_store_instance,
            top_k=5,
            similarity_threshold=0.7,
        )

        results = retriever.retrieve("test")

        assert len(results) == 0

    @patch("rag.retriever.Embedder")
    @patch("rag.retriever.VectorStore")
    def test_build_context(self, mock_store, mock_embedder):
        from rag.retriever import RetrievedChunk, Retriever

        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance

        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance

        retriever = Retriever(
            embedder=mock_embedder_instance,
            vector_store=mock_store_instance,
        )

        chunks = [
            RetrievedChunk(
                text="Content 1",
                source="file1.txt",
                distance=0.1,
                metadata={"source": "file1.txt"},
            ),
            RetrievedChunk(
                text="Content 2",
                source="file2.txt",
                distance=0.2,
                metadata={"source": "file2.txt"},
            ),
        ]

        context = retriever.build_context(chunks)

        assert "file1.txt" in context
        assert "file2.txt" in context
        assert "Content 1" in context

    @patch("rag.retriever.Embedder")
    @patch("rag.retriever.VectorStore")
    def test_build_sources(self, mock_store, mock_embedder):
        from rag.retriever import RetrievedChunk, Retriever

        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance

        mock_store_instance = MagicMock()
        mock_store.return_value = mock_store_instance

        retriever = Retriever(
            embedder=mock_embedder_instance,
            vector_store=mock_store_instance,
        )

        chunks = [
            RetrievedChunk(
                text="Content 1",
                source="file1.txt",
                distance=0.1,
                metadata={"source": "file1.txt"},
            ),
            RetrievedChunk(
                text="More from file1",
                source="file1.txt",
                distance=0.2,
                metadata={"source": "file1.txt"},
            ),
            RetrievedChunk(
                text="Content 2",
                source="file2.txt",
                distance=0.3,
                metadata={"source": "file2.txt"},
            ),
        ]

        sources = retriever.build_sources(chunks)

        assert len(sources) == 2
        sources_list = [s["source"] for s in sources]
        assert "file1.txt" in sources_list
        assert "file2.txt" in sources_list


class TestRetrievedChunk:
    def test_dataclass_fields(self):
        from rag.retriever import RetrievedChunk

        chunk = RetrievedChunk(
            text="test",
            source="test.txt",
            distance=0.1,
            metadata={"key": "value"},
        )

        assert chunk.text == "test"
        assert chunk.source == "test.txt"
        assert chunk.distance == 0.1
        assert chunk.metadata["key"] == "value"
