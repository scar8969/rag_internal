from dataclasses import dataclass
from typing import Optional

from .embedder import Embedder
from .vector_store import DocumentChunk, VectorStore
from .chunker import TokenCounter


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float
    metadata: dict


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        max_tokens: int = 2000,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._max_tokens = max_tokens
        self._token_counter = TokenCounter()

    def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed(query)
        top_k = top_k or self._top_k

        raw_chunks = self._vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=top_k * 2,
        )

        chunks = []
        for chunk in raw_chunks:
            distance = chunk.metadata.get("distance", 1.0)
            if self._similarity_threshold:
                if distance > (1 - self._similarity_threshold):
                    continue

            chunks.append(
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.metadata.get("source", ""),
                    distance=distance,
                    metadata=chunk.metadata,
                )
            )

            if len(chunks) >= top_k:
                break

        return self._limit_by_tokens(chunks)

    def _limit_by_tokens(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if self._max_tokens <= 0:
            return chunks

        total_tokens = 0
        limited_chunks = []

        for chunk in chunks:
            chunk_tokens = self._token_counter.count(chunk.text)
            if total_tokens + chunk_tokens > self._max_tokens:
                remaining_tokens = self._max_tokens - total_tokens
                if remaining_tokens > 50:
                    truncated_text = self._token_counter.truncate(chunk.text, remaining_tokens)
                    limited_chunks.append(
                        RetrievedChunk(
                            text=truncated_text,
                            source=chunk.source,
                            distance=chunk.distance,
                            metadata=chunk.metadata,
                        )
                    )
                break

            total_tokens += chunk_tokens
            limited_chunks.append(chunk)

        return limited_chunks

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
        context_parts = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")
            context_parts.append(f"Source: {source}\n{chunk.text}\n")

        return "\n---\n".join(context_parts)

    def build_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        sources = []
        seen = set()

        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")
            if source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                })

        return sources
