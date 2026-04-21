"""Ollama embeddings wrapper with vault-title caching."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import requests

from config import CACHE_DIR, MODEL_EMBED, OLLAMA_URL
from src.utils.cache import JSONCache


class EmbeddingClient:
    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        model: str = MODEL_EMBED,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._cache = JSONCache(CACHE_DIR / "embeddings")

    def _cache_key(self, text: str) -> str:
        return f"{self.model}::{text}"

    def embed(self, text: str) -> Optional[List[float]]:
        cached = self._cache.get(self._cache_key(text))
        if cached is not None:
            return cached

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vec = response.json().get("embedding")
        except requests.RequestException as e:
            print(f"  [embeddings] request failed: {e}")
            return None

        if not vec:
            return None

        self._cache.set(self._cache_key(text), vec)
        return vec

    def embed_many(self, texts: Sequence[str]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for t in texts:
            vec = self.embed(t)
            if vec is not None:
                out[t] = vec
        return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_k_similar(
    query_vec: Sequence[float],
    candidates: Dict[str, Sequence[float]],
    k: int = 15,
) -> List[tuple[str, float]]:
    scored = [(label, cosine(query_vec, vec)) for label, vec in candidates.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
