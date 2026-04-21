"""Fetch + clean source pages with caching, retries, and quality scoring."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests
import trafilatura
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    CACHE_DIR,
    HTTP_BACKOFF,
    HTTP_RETRIES,
    HTTP_TIMEOUT,
    MAX_EXTRACT_WORKERS,
    MIN_SOURCE_CHARS,
    SOURCE_CACHE_TTL,
    TRUSTED_SOURCES,
)
from src.utils.cache import JSONCache

_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)


def _build_session() -> requests.Session:
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (ObsidianResearchBot)"})
    return session


class MultiSourceExtractor:
    def __init__(
        self,
        max_workers: int = MAX_EXTRACT_WORKERS,
        timeout: int = HTTP_TIMEOUT,
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = _build_session()
        self.cache = JSONCache(CACHE_DIR / "sources", ttl_seconds=SOURCE_CACHE_TTL)

    def extract_single(self, url: str) -> Optional[Dict]:
        cached = self.cache.get(url)
        if cached is not None:
            print(f"  [cache] {url[:70]}")
            return cached

        try:
            print(f"  [fetch] {url[:70]}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as e:
            print(f"  [fetch] failed: {str(e)[:80]}")
            return None

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            output_format="markdown",
        )

        if not text or len(text) < MIN_SOURCE_CHARS:
            print(f"  [extract] too short ({len(text) if text else 0} chars)")
            return None

        title = _extract_title(html) or url.split("//")[-1].split("/")[0]
        quality = _score_quality(url, text)

        record = {
            "url": url,
            "title": title,
            "content": text,
            "quality_score": quality,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

        self.cache.set(url, record)
        return record

    def extract_multiple(self, urls: List[str], top_n: int = 4) -> List[Dict]:
        print(f"\n[extract] {len(urls)} urls, top_n={top_n}")

        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for r in pool.map(self.extract_single, urls):
                if r is not None:
                    results.append(r)

        if not results:
            print("[extract] no sources passed quality bar")
            return []

        results.sort(key=lambda x: x["quality_score"], reverse=True)
        picks = results[:top_n]
        for r in picks:
            print(f"  [keep] {r['quality_score']}  {r['title'][:60]}")
        return picks


def _extract_title(html: str) -> Optional[str]:
    m = _TITLE_RE.search(html)
    return m.group(1).strip() if m else None


def _score_quality(url: str, text: str) -> int:
    score = 50
    if any(trusted in url for trusted in TRUSTED_SOURCES):
        score += 30
    if len(text) > 2000:
        score += 10
    if len(text) > 6000:
        score += 5
    if "```" in text:
        score += 5
    return score
