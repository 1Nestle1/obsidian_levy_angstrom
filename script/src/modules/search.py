"""SearXNG search wrapper with retries + domain presets."""
from __future__ import annotations

from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    DEFAULT_MAX_RESULTS,
    HTTP_BACKOFF,
    HTTP_RETRIES,
    HTTP_TIMEOUT,
    SEARXNG_URL,
)

DOMAIN_PRESETS = {
    "Math":     {"categories": "science", "engines": "wikipedia,arxiv,duckduckgo"},
    "Biology":  {"categories": "science", "engines": "wikipedia,pubmed,duckduckgo"},
    "Code":     {"categories": "it",      "engines": "stackoverflow,github,duckduckgo"},
    "Physics":  {"categories": "science", "engines": "wikipedia,arxiv,duckduckgo"},
}


def _build_session() -> requests.Session:
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class SearchClient:
    def __init__(self, base_url: str = SEARXNG_URL, timeout: int = HTTP_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = _build_session()

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        domain: Optional[str] = None,
    ) -> List[dict]:
        params = {"q": query, "format": "json"}
        if domain and domain in DOMAIN_PRESETS:
            params.update(DOMAIN_PRESETS[domain])

        try:
            response = self.session.get(
                f"{self.base_url}/search",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"[search] request failed: {e}")
            return []
        except ValueError:
            print("[search] non-JSON response from SearXNG")
            return []

        results = []
        for r in data.get("results", [])[:max_results]:
            url = r.get("url")
            if not url:
                continue
            results.append({
                "title":   r.get("title") or url,
                "url":     url,
                "snippet": r.get("content", ""),
                "engine":  r.get("engine", ""),
            })
        return results
