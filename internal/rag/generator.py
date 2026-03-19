from dataclasses import dataclass
from typing import Optional, Generator
import os


@dataclass
class GenerationResponse:
    text: str
    sources: list[dict]
    confidence: float


class LLMGenerator:
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: str = "",
        require_citations: bool = True,
        allow_unknown: bool = True,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.require_citations = require_citations
        self.allow_unknown = allow_unknown

    def generate(
        self,
        question: str,
        context: str,
        sources: list[dict],
    ) -> GenerationResponse:
        if self.provider == "openai":
            return self._generate_openai(question, context, sources)
        elif self.provider == "openai-compatible":
            return self._generate_openai_compatible(question, context, sources)
        elif self.provider == "anthropic":
            return self._generate_anthropic(question, context, sources)
        elif self.provider == "ollama":
            return self._generate_ollama(question, context, sources)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(self, question: str, context: str) -> str:
        if not context.strip():
            if self.allow_unknown:
                return f"Question: {question}\n\nAnswer: The provided documents do not contain information to answer this question."
            else:
                return f"Question: {question}\n\nAnswer: I don't have enough context to answer this question."

        if self.require_citations:
            prompt = f"""You are a helpful assistant. Use the provided context to answer the user's question.
If the answer cannot be determined from the context, say so clearly.

Context:
{context}

Question: {question}

Answer (cite your sources at the end using the format [/filename]):"""
        else:
            prompt = f"""You are a helpful assistant. Use the provided context to answer the user's question.
If the answer cannot be determined from the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

        return prompt

    def _generate_openai(self, question: str, context: str, sources: list[dict]) -> GenerationResponse:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            prompt = self._build_prompt(question, context)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            text = response.choices[0].message.content or ""
            
            return GenerationResponse(
                text=text,
                sources=sources,
                confidence=0.85,
            )
        except ImportError:
            pass
        
        import httpx
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        prompt = self._build_prompt(question, context)
        
        client = httpx.Client()
        response = client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        
        return GenerationResponse(
            text=text,
            sources=sources,
            confidence=0.85,
        )

    def _generate_openai_compatible(self, question: str, context: str, sources: list[dict]) -> GenerationResponse:
        import httpx
        
        api_key = self.api_key
        base_url = self.api_key
        
        if not api_key:
            raise ValueError("API key/base URL not set")
        
        prompt = self._build_prompt(question, context)
        
        client = httpx.Client()
        response = client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        
        return GenerationResponse(
            text=text,
            sources=sources,
            confidence=0.85,
        )

    def _generate_anthropic(self, question: str, context: str, sources: list[dict]) -> GenerationResponse:
        import httpx
        
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        prompt = self._build_prompt(question, context)
        
        client = httpx.Client()
        response = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "system": self.system_prompt,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        
        data = response.json()
        text = data["content"][0]["text"]
        
        return GenerationResponse(
            text=text,
            sources=sources,
            confidence=0.85,
        )

    def _generate_ollama(self, question: str, context: str, sources: list[dict]) -> GenerationResponse:
        import httpx
        
        base_url = self.api_key or "http://localhost:11434"
        model = self.model or "llama2"
        
        prompt = self._build_prompt(question, context)
        
        client = httpx.Client()
        response = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        
        data = response.json()
        text = data["response"]
        
        return GenerationResponse(
            text=text,
            sources=sources,
            confidence=0.85,
        )

    def stream(
        self,
        question: str,
        context: str,
        sources: list[dict],
    ) -> Generator[tuple[str, GenerationResponse], None, None]:
        if self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                
                prompt = self._build_prompt(question, context)
                
                stream = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                
                full_text = ""
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    full_text += token
                    yield token, GenerationResponse(
                        text=full_text,
                        sources=sources,
                        confidence=0.85,
                    )
                return
            except ImportError:
                pass
        
        response = self.generate(question, context, sources)
        yield response.text, response
