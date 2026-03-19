from typing import Optional
import os


class Embedder:
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        dimensions: int = 1536,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url
        self.token = token

    def embed(self, text: str) -> list[float]:
        if self.provider == "openai":
            return self._embed_openai(text)
        elif self.provider == "openai-compatible":
            return self._embed_openai_compatible(text)
        elif self.provider == "huggingface":
            return self._embed_huggingface(text)
        elif self.provider == "ollama":
            return self._embed_ollama(text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _embed_openai(self, text: str) -> list[float]:
        try:
            import openai
        except ImportError:
            import httpx
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            
            client = httpx.Client()
            response = client.post(
                f"{base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": text[:8191],
                    "model": self.model,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        
        client = openai.OpenAI(api_key=self.api_key)
        response = client.embeddings.create(
            input=text[:8191],
            model=self.model,
        )
        return response.data[0].embedding

    def _embed_openai_compatible(self, text: str) -> list[float]:
        import httpx
        
        api_key = self.api_key
        base_url = self.base_url
        
        if not api_key:
            raise ValueError("API key not set")
        
        client = httpx.Client()
        response = client.post(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": text[:8191],
                "model": self.model,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _embed_huggingface(self, text: str) -> list[float]:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError("Install transformers for HuggingFace embeddings")
        
        if not hasattr(self, "_hf_model"):
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model, token=self.token)
            self._hf_model = AutoModel.from_pretrained(self.model, token=self.token)
            self._hf_model.eval()
        
        inputs = self._hf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self._hf_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        
        return embedding.tolist()

    def _embed_ollama(self, text: str) -> list[float]:
        import httpx
        
        base_url = self.base_url or "http://localhost:11434"
        model = self.model or "llama2"
        
        client = httpx.Client()
        response = client.post(
            f"{base_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    input=[t[:8191] for t in texts],
                    model=self.model,
                )
                return [item.embedding for item in response.data]
            except ImportError:
                pass
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            import httpx
            client = httpx.Client()
            response = client.post(
                f"{base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": [t[:8191] for t in texts],
                    "model": self.model,
                },
                timeout=120,
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        
        return [self.embed(text) for text in texts]
