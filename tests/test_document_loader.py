import tempfile
from pathlib import Path

import pytest


class TestDocumentLoader:
    def test_load_text_file(self):
        from rag.document_loader import DocumentLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is a test document.\nWith multiple lines.")
            temp_path = f.name

        try:
            loader = DocumentLoader()
            doc = loader.load(temp_path)
            assert doc.text == "This is a test document.\nWith multiple lines."
            assert doc.source == str(Path(temp_path).absolute())
            assert "filename" in doc.metadata
            assert doc.metadata["extension"] == ".txt"
        finally:
            Path(temp_path).unlink()

    def test_load_markdown_file(self):
        from rag.document_loader import DocumentLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Heading\n\nThis is **markdown** content.")
            temp_path = f.name

        try:
            loader = DocumentLoader()
            doc = loader.load(temp_path)
            assert "Heading" in doc.text
            assert doc.metadata["extension"] == ".md"
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        from rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_load_unsupported_extension(self):
        from rag.document_loader import DocumentLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            loader = DocumentLoader()
            with pytest.raises(ValueError, match="Unsupported file type"):
                loader.load(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_is_url(self):
        from rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        assert loader._is_url("https://example.com") is True
        assert loader._is_url("http://test.org/page") is True
        assert loader._is_url("/local/path/file.txt") is False
        assert loader._is_url("not-a-url") is False

    def test_clean_text(self):
        from rag.document_loader import DocumentLoader

        loader = DocumentLoader()
        dirty_text = "Hello\u200B  \n\n\n  World  \n\n"
        clean_text = loader.clean_text(dirty_text)
        assert clean_text == "Hello World"

    def test_load_directory(self):
        from rag.document_loader import DocumentLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").write_text("Content 1")
            Path(tmpdir, "file2.md").write_text("# Markdown")
            Path(tmpdir, "file3.xyz").write_text("Ignored")

            loader = DocumentLoader()
            docs = loader.load_directory(tmpdir)

            assert len(docs) == 2
            sources = [d.source for d in docs]
            assert any("file1.txt" in s for s in sources)
            assert any("file2.md" in s for s in sources)
            assert not any("file3.xyz" in s for s in sources)
