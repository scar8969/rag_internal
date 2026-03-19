from dataclasses import dataclass, field
from typing import Optional
import uuid
import hashlib


@dataclass
class DocumentChunk:
    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_{chunk_index}"


def generate_document_id(source: str) -> str:
    hash_input = f"{source}_{uuid.uuid4()}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name.replace(" ", "_").lower()
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, chunks: list[DocumentChunk], user_id: str) -> None:
        collection = self._get_collection()
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[DocumentChunk]:
        collection = self._get_collection()
        
        filter_where = {"user_id": user_id}
        if where:
            filter_where.update(where)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_where,
            include=["documents", "metadatas", "distances"],
        )
        
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunks.append(DocumentChunk(
                    id=doc_id,
                    text=results["documents"][0][i],
                    embedding=[],
                    metadata={
                        **results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    },
                ))
        
        return chunks

    def get_by_source(self, source: str, user_id: str) -> list[DocumentChunk]:
        collection = self._get_collection()
        
        results = collection.get(
            where={"source": source, "user_id": user_id},
            include=["documents", "metadatas", "embeddings"],
        )
        
        chunks = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                chunks.append(DocumentChunk(
                    id=doc_id,
                    text=results["documents"][i],
                    embedding=results["embeddings"][i] if results.get("embeddings") else [],
                    metadata=results["metadatas"][i],
                ))
        
        return chunks

    def delete_by_source(self, source: str, user_id: str) -> None:
        collection = self._get_collection()
        collection.delete(where={"source": source, "user_id": user_id})

    def list_sources(self, user_id: str) -> list[str]:
        collection = self._get_collection()
        
        results = collection.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        
        sources = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])
        
        return list(sources)

    def count(self, user_id: Optional[str] = None) -> int:
        collection = self._get_collection()
        
        if user_id:
            results = collection.get(where={"user_id": user_id}, include=[])
            return len(results["ids"])
        
        return collection.count()

    def delete_all(self, user_id: str) -> None:
        collection = self._get_collection()
        collection.delete(where={"user_id": user_id})
