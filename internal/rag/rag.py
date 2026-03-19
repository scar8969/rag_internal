from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from .chunker import Chunker
from .config import Config
from .document_loader import Document, DocumentLoader
from .embedder import Embedder
from .generator import GenerationResponse, LLMGenerator
from .retriever import RetrievedChunk, Retriever
from .vector_store import DocumentChunk, VectorStore, generate_chunk_id, generate_document_id


@dataclass
class IngestResult:
    document_id: str
    chunks_created: int
    status: str


class RAGSystem:
    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        self._loader = DocumentLoader()
        self._chunker = Chunker()

        self._embedder = Embedder(
            provider=self._config.embedding_provider,
            api_key=self._config.openai_api_key,
            model=self._config.embedding_model,
            dimensions=self._config.embedding_dimensions,
            base_url=self._config.ollama_base_url,
        )

        self._vector_store = VectorStore(
            persist_directory=self._config.vector_store_dir,
            collection_name=self._config.collection_name,
        )

        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._vector_store,
            top_k=self._config.retrieval_top_k,
            similarity_threshold=self._config.retrieval_similarity_threshold,
            max_tokens=self._config.retrieval_max_tokens,
        )

        self._generator = LLMGenerator(
            provider=self._config.generation_provider,
            api_key=self._config.openai_api_key,
            model=self._config.generation_model,
            temperature=self._config.generation_temperature,
            max_tokens=self._config.generation_max_tokens,
            system_prompt=self._config.system_prompt,
        )

    def ingest(self, source: str, user_id: str, metadata: Optional[dict] = None) -> IngestResult:
        doc = self._loader.load(source)
        document_id = generate_document_id(doc.source)

        existing_chunks = self._vector_store.get_by_source(doc.source, user_id)
        if existing_chunks:
            self._vector_store.delete_by_source(doc.source, user_id)

        chunks = self._chunker.chunk(doc.text, doc.source, {**doc.metadata, **(metadata or {})})

        chunk_objects = []
        for chunk in chunks:
            embedding = self._embedder.embed(chunk.text)
            chunk_obj = DocumentChunk(
                id=generate_chunk_id(document_id, chunk.index),
                text=chunk.text,
                embedding=embedding,
                metadata={
                    **chunk.metadata,
                    "document_id": document_id,
                    "source": doc.source,
                    "user_id": user_id,
                },
            )
            chunk_objects.append(chunk_obj)

        self._vector_store.add(chunk_objects, user_id)

        return IngestResult(
            document_id=document_id,
            chunks_created=len(chunks),
            status="success",
        )

    def ingest_directory(self, directory: str, user_id: str) -> list[IngestResult]:
        docs = self._loader.load_directory(directory)
        results = []
        for doc in docs:
            result = self.ingest(doc.source, user_id, doc.metadata)
            results.append(result)
        return results

    def ingest_url(self, url: str, user_id: str, metadata: Optional[dict] = None) -> IngestResult:
        return self.ingest(url, user_id, metadata)

    def search(self, query: str, user_id: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        return self._retriever.retrieve(query, user_id, top_k)

    def query(
        self,
        question: str,
        user_id: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
    ) -> GenerationResponse:
        chunks = self._retriever.retrieve(question, user_id, top_k)

        context = self._retriever.build_context(chunks)
        sources = self._retriever.build_sources(chunks) if include_sources else []

        return self._generator.generate(question, context, sources)

    def stream_query(
        self,
        question: str,
        user_id: str,
        top_k: Optional[int] = None,
    ) -> Generator[tuple[str, GenerationResponse], None, None]:
        chunks = self._retriever.retrieve(question, user_id, top_k)

        context = self._retriever.build_context(chunks)
        sources = self._retriever.build_sources(chunks)

        for token, response in self._generator.stream(question, context, sources):
            yield token, response

    def delete_document(self, source: str, user_id: str) -> bool:
        self._vector_store.delete_by_source(source, user_id)
        return True

    def list_documents(self, user_id: str) -> list[str]:
        return self._vector_store.list_sources(user_id)

    def count_documents(self, user_id: str) -> int:
        return self._vector_store.count(user_id)

    def clear(self, user_id: str) -> None:
        self._vector_store.delete_all(user_id)
