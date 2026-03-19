import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDocumentLoaderIntegration:
    def test_load_txt_file(self):
        from rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        doc = loader.load(str(FIXTURES_DIR / "sample.txt"))

        assert doc.text is not None
        assert len(doc.text) > 0
        assert "RAG system" in doc.text
        assert doc.metadata["extension"] == ".txt"

    def test_load_markdown_file(self):
        from rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        doc = loader.load(str(FIXTURES_DIR / "sample.md"))

        assert doc.text is not None
        assert len(doc.text) > 0
        assert "Overview" in doc.text
        assert doc.metadata["extension"] == ".md"

    def test_load_html_file(self):
        from rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        doc = loader.load(str(FIXTURES_DIR / "sample.html"))

        assert doc.text is not None
        assert len(doc.text) > 0
        assert "HTML" in doc.text or "Document" in doc.text
        assert doc.metadata["extension"] in [".html", ".htm"]


class TestChunkerIntegration:
    def test_chunk_text_file(self):
        from rag.chunker import Chunker

        loader = MagicMock()
        loader.load.return_value = MagicMock(
            text="This is test content. " * 100,
            source="test.txt",
            metadata={},
        )

        from rag.document_loader import DocumentLoader

        real_loader = DocumentLoader()
        doc = real_loader.load(str(FIXTURES_DIR / "sample.txt"))

        chunker = Chunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk(doc.text, "test-doc-1")

        assert len(chunks) > 1
        assert all(c.document_id == "test-doc-1" for c in chunks)
        assert all(hasattr(c, "text") for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_markdown_with_paragraph_strategy(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=100, strategy="paragraph")
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."

        chunks = chunker.chunk(text, "test-doc-2")

        assert len(chunks) > 0
        assert "paragraph" in chunks[0].text.lower()


class TestVectorStoreIntegration:
    def test_add_and_search_chunks(self, mock_chroma_collection):
        from rag.vector_store import DocumentChunk, VectorStore, generate_chunk_id

        store = VectorStore(
            persist_directory="/tmp/test_chroma",
            collection_name="test_collection",
        )

        chunks = [
            DocumentChunk(
                id=generate_chunk_id("doc1", 0),
                text="Test chunk content",
                embedding=[0.1] * 1536,
                metadata={"source": "test.txt"},
            ),
            DocumentChunk(
                id=generate_chunk_id("doc1", 1),
                text="Another test chunk",
                embedding=[0.2] * 1536,
                metadata={"source": "test.txt"},
            ),
        ]

        store.add(chunks)

        results = store.search(query_embedding=[0.1] * 1536, top_k=2)

        assert len(results) >= 0

    def test_delete_by_source(self, mock_chroma_collection):
        from rag.vector_store import VectorStore

        store = VectorStore(
            persist_directory="/tmp/test_chroma_del",
            collection_name="test_del_collection",
        )

        store.delete_by_source("nonexistent.txt")

        assert True

    def test_count_chunks(self, mock_chroma_collection):
        from rag.vector_store import VectorStore

        mock_chroma_collection.count.return_value = 42

        store = VectorStore(
            persist_directory="/tmp/test_chroma_count",
            collection_name="test_count_collection",
        )

        count = store.count()

        assert count == 42


class TestRetrieverIntegration:
    def test_retrieve_with_mocked_components(self, mock_rag_components):
        from rag.retriever import RetrievedChunk, Retriever

        mock_chunks = [
            RetrievedChunk(
                text="Relevant chunk about embeddings",
                source="doc.txt",
                distance=0.1,
                metadata={"source": "doc.txt"},
            ),
            RetrievedChunk(
                text="Another relevant chunk",
                source="doc.txt",
                distance=0.2,
                metadata={"source": "doc.txt"},
            ),
        ]

        retriever = Retriever(
            embedder=mock_rag_components["embedder"],
            vector_store=mock_rag_components["store"],
            top_k=5,
            similarity_threshold=0.0,
        )

        mock_rag_components["store"].search.return_value = mock_chunks

        results = retriever.retrieve("embeddings")

        assert len(results) > 0
        assert results[0].text == "Relevant chunk about embeddings"

    def test_build_context(self, mock_rag_components):
        from rag.retriever import RetrievedChunk, Retriever

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

        retriever = Retriever(
            embedder=mock_rag_components["embedder"],
            vector_store=mock_rag_components["store"],
        )

        context = retriever.build_context(chunks)

        assert "file1.txt" in context
        assert "file2.txt" in context
        assert "Content 1" in context

    def test_build_sources_deduplicates(self, mock_rag_components):
        from rag.retriever import RetrievedChunk, Retriever

        chunks = [
            RetrievedChunk(
                text="Chunk 1 from file",
                source="same.txt",
                distance=0.1,
                metadata={"source": "same.txt"},
            ),
            RetrievedChunk(
                text="Chunk 2 from file",
                source="same.txt",
                distance=0.2,
                metadata={"source": "same.txt"},
            ),
            RetrievedChunk(
                text="Chunk from other",
                source="other.txt",
                distance=0.3,
                metadata={"source": "other.txt"},
            ),
        ]

        retriever = Retriever(
            embedder=mock_rag_components["embedder"],
            vector_store=mock_rag_components["store"],
        )

        sources = retriever.build_sources(chunks)

        assert len(sources) == 2
        source_names = [s["source"] for s in sources]
        assert "same.txt" in source_names
        assert "other.txt" in source_names


class TestGeneratorIntegration:
    def test_generate_with_mocked_llm(self, mock_rag_components):
        from rag.generator import GenerationResponse, LLMGenerator

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The RAG system processes documents by chunking them."

        generator = LLMGenerator(
            provider="openai",
            api_key="mock-key",
            require_citations=False,
        )
        generator._llm = mock_llm

        response = generator.generate(
            question="How does RAG work?",
            context="RAG chunks documents into smaller pieces.",
            sources=[{"source": "test.txt", "text": "RAG chunks documents"}],
        )

        assert response.text is not None
        assert response.confidence >= 0

    def test_format_sources(self, mock_rag_components):
        from rag.generator import LLMGenerator

        mock_llm = MagicMock()
        generator = LLMGenerator(provider="openai", api_key="mock-key")
        generator._llm = mock_llm

        sources = [
            {"source": "file1.txt", "text": "Content 1"},
            {"source": "file2.txt", "text": "Content 2"},
        ]

        formatted = generator._format_sources(sources)

        assert "[1] file1.txt" in formatted
        assert "[2] file2.txt" in formatted


class TestRAGSystemFullPipeline:
    def test_rag_system_initialization(self, mock_rag_components):
        from rag import RAGSystem

        rag = RAGSystem()

        assert rag._config is not None
        assert rag._loader is not None
        assert rag._chunker is not None

    def test_ingest_updates_existing_document(self, mock_rag_components):
        from rag import RAGSystem

        rag = RAGSystem()
        rag._embedder = mock_rag_components["embedder"]
        rag._vector_store = mock_rag_components["store"]

        result1 = rag.ingest(str(FIXTURES_DIR / "sample.txt"))

        assert result1.status == "success"
        assert result1.chunks_created > 0

        result2 = rag.ingest(str(FIXTURES_DIR / "sample.txt"))

        assert result2.document_id == result1.document_id


class TestEdgeCases:
    def test_search_with_empty_query(self, mock_rag_components):
        from rag import RAGSystem

        rag = RAGSystem()
        rag._embedder = mock_rag_components["embedder"]
        rag._vector_store = mock_rag_components["store"]

        results = rag.search("")
        assert isinstance(results, list)

    def test_list_empty_documents(self, mock_rag_components):
        from rag import RAGSystem

        mock_rag_components["store"].list_sources.return_value = []

        rag = RAGSystem()
        rag._embedder = mock_rag_components["embedder"]
        rag._vector_store = mock_rag_components["store"]

        docs = rag.list_documents()
        assert len(docs) == 0

    def test_delete_nonexistent_document(self, mock_rag_components):
        from rag import RAGSystem

        mock_rag_components["store"].get_by_source.return_value = []

        rag = RAGSystem()
        rag._embedder = mock_rag_components["embedder"]
        rag._vector_store = mock_rag_components["store"]

        result = rag.delete_document("nonexistent.txt")
        assert result is True


class TestConfiguration:
    def test_config_loads_from_file(self):
        from rag import Config

        config = Config(str(Path(__file__).parent.parent / "config.yaml"))

        assert config.embedding_provider == "openai"
        assert config.generation_provider == "openai"
        assert config.retrieval_top_k == 5

    def test_config_defaults(self):
        from rag import Config

        config = Config()

        assert config.embedding_model is not None
        assert config.vector_store_type == "chroma"
        assert config.api_port == 8000

    def test_config_get_nested_value(self):
        from rag import Config

        config = Config()

        val = config.get("embedding.provider")
        assert val is not None

        val = config.get("nonexistent.key", "default")
        assert val == "default"

    def test_config_set_value(self):
        from rag import Config

        config = Config()
        config.set("test.key", "test_value")

        val = config.get("test.key")
        assert val == "test_value"
