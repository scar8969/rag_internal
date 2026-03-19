from unittest.mock import MagicMock, patch

import pytest


class TestGenerator:
    @patch("rag.generator.OpenAILLM")
    def test_generate_with_citations(self, mock_llm):
        from rag.generator import Generator

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Test answer with citation [1]"
        mock_llm.return_value = mock_llm_instance

        generator = Generator(
            provider="openai",
            api_key="test-key",
            require_citations=True,
        )

        response = generator.generate(
            question="What is test?",
            context="",
            sources=[{"source": "test.txt", "text": "Test content"}],
        )

        assert response.text == "Test answer with citation [1]"
        assert response.sources is not None

    @patch("rag.generator.OpenAILLM")
    def test_generate_without_citations(self, mock_llm):
        from rag.generator import Generator

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Test answer"
        mock_llm.return_value = mock_llm_instance

        generator = Generator(
            provider="openai",
            api_key="test-key",
            require_citations=False,
        )

        response = generator.generate(
            question="What is test?",
            context="Context text",
            sources=[],
        )

        assert "Test answer" in response.text

    def test_unknown_provider_raises(self):
        from rag.generator import Generator

        with pytest.raises(ValueError, match="Unknown provider"):
            Generator(provider="nonexistent", api_key="test")

    @patch("rag.generator.OpenAILLM")
    def test_confidence_estimation(self, mock_llm):
        from rag.generator import Generator

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "The answer is 42."
        mock_llm.return_value = mock_llm_instance

        generator = Generator(
            provider="openai",
            api_key="test-key",
            allow_unknown=True,
        )

        response = generator.generate(
            question="What is answer?",
            context="",
            sources=[{"source": "test.txt", "text": "content"}],
        )

        assert response.confidence == 0.85

    @patch("rag.generator.OpenAILLM")
    def test_confidence_unknown(self, mock_llm):
        from rag.generator import Generator

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "I don't know the answer."
        mock_llm.return_value = mock_llm_instance

        generator = Generator(
            provider="openai",
            api_key="test-key",
            allow_unknown=True,
        )

        response = generator.generate(
            question="What is answer?",
            context="",
            sources=[{"source": "test.txt", "text": "content"}],
        )

        assert response.confidence == 0.3

    @patch("rag.generator.OpenAILLM")
    def test_format_sources(self, mock_llm):
        from rag.generator import Generator

        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        generator = Generator(provider="openai", api_key="test-key")

        sources = [
            {"source": "file1.txt", "text": "Content 1"},
            {"source": "file2.txt", "text": "Content 2"},
        ]

        formatted = generator._format_sources(sources)

        assert "[1] file1.txt" in formatted
        assert "[2] file2.txt" in formatted


class TestGenerationResponse:
    def test_dataclass_fields(self):
        from rag.generator import GenerationResponse

        response = GenerationResponse(
            text="Answer",
            sources=[{"source": "test.txt"}],
            confidence=0.9,
        )

        assert response.text == "Answer"
        assert len(response.sources) == 1
        assert response.confidence == 0.9


class TestLLMProviders:
    @patch("rag.generator.OpenAI")
    def test_openai_llm_generate(self, mock_openai):
        from rag.generator import OpenAILLM

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated text"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = OpenAILLM(api_key="test-key")
        result = llm.generate("Test prompt")

        assert result == "Generated text"

    @patch("rag.generator.httpx.Client")
    def test_ollama_llm_generate(self, mock_httpx):
        from rag.generator import OllamaLLM

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Ollama response"}
        mock_client.post.return_value = mock_response
        mock_httpx.return_value = mock_client

        llm = OllamaLLM()
        result = llm.generate("Test prompt")

        assert result == "Ollama response"
