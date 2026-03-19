import pytest


class TestChunker:
    def test_fixed_chunking_basic(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=10, overlap=2, strategy="fixed")
        text = "one two three four five six seven eight nine ten eleven twelve"

        chunks = chunker.chunk(text, "doc1")

        assert len(chunks) > 0
        assert all(chunk.document_id == "doc1" for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)

    def test_fixed_chunking_preserves_total_chunks(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=5, overlap=1, strategy="fixed")
        text = "a b c d e f g h i j k l"

        chunks = chunker.chunk(text, "doc1")

        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_paragraph_chunking(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=100, strategy="paragraph")
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."

        chunks = chunker.chunk(text, "doc2")

        assert len(chunks) > 0
        assert "First paragraph" in chunks[0].text or len(chunks) > 1

    def test_semantic_chunking(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=20, strategy="semantic")
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunker.chunk(text, "doc3")

        assert len(chunks) > 0

    def test_clean_text(self):
        from rag.chunker import Chunker

        chunker = Chunker()
        dirty_text = "Hello   \n\n\n  World  \n\n"
        clean = chunker._clean_text(dirty_text)
        assert "   " not in clean
        assert clean == "Hello\n\nWorld"

    def test_chunk_with_metadata(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=10)
        text = "word1 word2 word3 word4 word5"
        metadata = {"source": "test.txt", "author": "tester"}

        chunks = chunker.chunk(text, "doc4", metadata)

        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "tester"

    def test_empty_text(self):
        from rag.chunker import Chunker

        chunker = Chunker()
        chunks = chunker.chunk("", "doc5")
        assert len(chunks) == 0

    def test_single_word(self):
        from rag.chunker import Chunker

        chunker = Chunker(chunk_size=10)
        chunks = chunker.chunk("hello", "doc6")
        assert len(chunks) == 1
        assert chunks[0].text == "hello"


class TestTokenCounter:
    def test_token_counter_fallback(self):
        from rag.chunker import TokenCounter

        counter = TokenCounter("nonexistent-encoding")
        text = "hello world"
        count = counter.count(text)
        assert count == 2

    def test_truncate(self):
        from rag.chunker import TokenCounter

        counter = TokenCounter("nonexistent-encoding")
        text = "one two three four five"
        truncated = counter.truncate(text, 3)
        assert len(truncated.split()) <= 3
