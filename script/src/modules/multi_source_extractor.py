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

# Fenced code blocks from trafilatura's markdown output.
# Captures optional language hint after the opening fence.
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
# Markdown heading to use as context label for a code block.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

_MAX_CODE_BLOCKS_PER_SOURCE = 3
_MAX_CODE_BLOCK_CHARS = 1500
_MIN_CODE_BLOCK_CHARS = 20


def _harvest_code_blocks(markdown_text: str) -> List[Dict]:
    """Pull fenced code blocks out of trafilatura markdown output.

    Returns up to N blocks, each with language, code, and the nearest
    preceding heading as context. No LLM involved — pure regex."""
    if "```" not in markdown_text:
        return []

    headings: List[tuple[int, str]] = [
        (m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(markdown_text)
    ]

    blocks: List[Dict] = []
    for m in _FENCE_RE.finditer(markdown_text):
        code = m.group(2).rstrip()
        if len(code) < _MIN_CODE_BLOCK_CHARS:
            continue
        if len(code) > _MAX_CODE_BLOCK_CHARS:
            code = code[:_MAX_CODE_BLOCK_CHARS].rstrip() + "\n# ... (truncated)"
        lang = (m.group(1) or "").strip() or "text"
        # Find most recent heading before this block.
        pos = m.start()
        context = ""
        for h_pos, h_text in headings:
            if h_pos < pos:
                context = h_text
            else:
                break
        blocks.append({"language": lang, "code": code, "context": context})
        if len(blocks) >= _MAX_CODE_BLOCKS_PER_SOURCE:
            break
    return blocks


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
        code_blocks = _harvest_code_blocks(text)
        if code_blocks:
            print(f"  [code]  harvested {len(code_blocks)} block(s)")

        record = {
            "url": url,
            "title": title,
            "content": text,
            "quality_score": quality,
            "word_count": len(text.split()),
            "char_count": len(text),
            "code_blocks": code_blocks,
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
