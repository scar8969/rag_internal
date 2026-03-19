import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextChunk:
    text: str
    index: int
    metadata: dict


@dataclass
class ChunkResult:
    chunks: list[TextChunk]
    total_chars: int
    total_tokens: int


class TokenCounter:
    def count(self, text: str) -> int:
        return len(text) // 4

    def truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        return text[:max_chars]


class Chunker:
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        strategy: str = "fixed",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.token_counter = TokenCounter()

    def chunk(
        self,
        text: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> ChunkResult:
        metadata = metadata or {}
        
        if self.strategy == "fixed":
            return self._fixed_chunk(text, source, metadata)
        elif self.strategy == "paragraph":
            return self._paragraph_chunk(text, source, metadata)
        else:
            return self._fixed_chunk(text, source, metadata)

    def _fixed_chunk(
        self,
        text: str,
        source: str,
        metadata: dict,
    ) -> ChunkResult:
        text = self._clean_text(text)
        tokens_per_chunk = self.chunk_size
        overlap_tokens = self.overlap

        chars_per_chunk = tokens_per_chunk * 4
        overlap_chars = overlap_tokens * 4

        chunks = []
        start = 0
        total_chars = len(text)
        index = 0

        while start < len(text):
            end = start + chars_per_chunk
            
            if end < len(text):
                chunk_text = text[start:end]
            else:
                chunk_text = text[start:]

            if chunk_text.strip():
                chunks.append(TextChunk(
                    text=chunk_text.strip(),
                    index=index,
                    metadata={
                        **metadata,
                        "source": source,
                        "chunk_index": index,
                        "char_start": start,
                        "char_end": end,
                    },
                ))
                index += 1

            start = end - overlap_chars
            if start <= 0:
                start = end

        return ChunkResult(
            chunks=chunks,
            total_chars=total_chars,
            total_tokens=sum(self.token_counter.count(c.text) for c in chunks),
        )

    def _paragraph_chunk(
        self,
        text: str,
        source: str,
        metadata: dict,
    ) -> ChunkResult:
        text = self._clean_text(text)
        paragraphs = re.split(r"\n\s*\n", text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = self.token_counter.count(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=index,
                    metadata={
                        **metadata,
                        "source": source,
                        "chunk_index": index,
                    },
                ))
                index += 1
                current_chunk = []
                current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                index=index,
                metadata={
                    **metadata,
                    "source": source,
                    "chunk_index": index,
                },
            ))

        return ChunkResult(
            chunks=chunks,
            total_chars=len(text),
            total_tokens=sum(self.token_counter.count(c.text) for c in chunks),
        )

    def _clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
