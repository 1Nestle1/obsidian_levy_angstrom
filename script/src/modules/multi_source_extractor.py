"""Fetch + clean source pages with caching, retries, and quality scoring."""
from __future__ import annotations

import html
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

# --- Code-block harvesting from raw HTML (preserves language hints) ---
# Pages encode language as <code class="language-jsx"> or class="hljs-python" or
# class="lang-go" — trafilatura's markdown output drops these, so we parse HTML.
_HTML_CODE_RE = re.compile(
    r"<pre[^>]*>\s*<code([^>]*)>(.+?)</code>\s*</pre>",
    re.DOTALL | re.IGNORECASE,
)
_HTML_HEADING_RE = re.compile(
    r"<h([1-6])[^>]*>(.+?)</h\1>", re.DOTALL | re.IGNORECASE,
)
_HTML_CLASS_RE = re.compile(r'class\s*=\s*"([^"]+)"', re.IGNORECASE)
_HTML_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_LANG_CLASS_RE = re.compile(
    r"\b(?:language|lang|hljs|highlight|brush)[-:]([a-z0-9+#]+)", re.IGNORECASE,
)

_MAX_CODE_BLOCKS_PER_SOURCE = 3
_MAX_CODE_BLOCK_CHARS = 1500
_MIN_CODE_BLOCK_CHARS = 30

# Map common short hints to canonical Obsidian-friendly fence labels.
_LANG_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "rb": "ruby", "sh": "bash", "shell": "bash", "zsh": "bash",
    "yml": "yaml", "md": "markdown", "tex": "latex",
    "c++": "cpp", "cs": "csharp", "objc": "objectivec",
    "html5": "html", "xml": "xml",
}


def _normalize_lang(name: str) -> str:
    name = (name or "").strip().lower()
    return _LANG_ALIASES.get(name, name)


def _detect_lang_from_class(class_attr: str) -> Optional[str]:
    if not class_attr:
        return None
    for token in class_attr.split():
        m = _LANG_CLASS_RE.match(token)
        if m:
            return _normalize_lang(m.group(1))
    return None


def _detect_lang_from_content(code: str) -> str:
    """Heuristic content-based language detection. Order matters — most
    specific rules first."""
    s = code.strip()
    head = s[:600]

    # JSX / React (must beat plain JS)
    if re.search(r"\bimport\s+React\b|<[A-Z][A-Za-z0-9]*\s|<\/[A-Z]", head):
        return "jsx"
    # TypeScript
    if re.search(r"\binterface\s+\w+|:\s*\w+(\[\])?\s*[=;)]|<\w+>\s*\(", head):
        return "typescript"
    # Plain JavaScript
    if re.search(r"=>|\bconst\s+\w+\s*=|\blet\s+\w+\s*=|\brequire\(", head):
        return "javascript"
    # Python
    if re.search(r"^\s*def\s+\w+\(|^\s*from\s+\w+\s+import|\bprint\(", head, re.MULTILINE):
        return "python"
    # Go
    if re.search(r"^\s*package\s+\w+|^\s*func\s+\w+\(", head, re.MULTILINE):
        return "go"
    # Rust
    if re.search(r"^\s*fn\s+\w+\(|let\s+mut\s+\w+", head, re.MULTILINE):
        return "rust"
    # Java
    if re.search(r"public\s+class\s+\w+|System\.out\.println", head):
        return "java"
    # C / C++
    if re.search(r"#include\s*<|int\s+main\s*\(", head):
        return "cpp"
    # SQL
    if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)\b", head, re.IGNORECASE):
        return "sql"
    # HTML
    if re.search(r"<!DOCTYPE|<html|<body", head, re.IGNORECASE):
        return "html"
    # CSS
    if re.search(r"^\s*[.#]?[a-zA-Z][\w-]*\s*\{[^}]*:\s*[^;]+;", head, re.MULTILINE):
        return "css"
    # Bash / shell
    if re.search(r"^\s*(\$|#!\s*/|sudo |npm |pip |git )", head, re.MULTILINE):
        return "bash"
    # JSON
    if re.match(r"\s*[{\[]", s) and re.search(r'"\w+"\s*:', head):
        return "json"
    # LaTeX / math
    if re.search(r"\\frac|\\begin\{|\$\$", head):
        return "latex"
    return "text"


def _strip_html_to_text(s: str) -> str:
    return html.unescape(_HTML_TAG_STRIP_RE.sub("", s)).strip()


def _harvest_code_blocks(html_text: str) -> List[Dict]:
    """Parse raw HTML for <pre><code>...</code></pre> blocks. Extracts language
    from class attribute when present; falls back to content heuristics.
    Tags each block with the nearest preceding <h1>-<h6> as context."""
    if "<code" not in html_text.lower():
        return []

    headings: List[tuple[int, str]] = [
        (m.start(), _strip_html_to_text(m.group(2)))
        for m in _HTML_HEADING_RE.finditer(html_text)
    ]

    blocks: List[Dict] = []
    for m in _HTML_CODE_RE.finditer(html_text):
        attrs, raw = m.group(1), m.group(2)
        code = html.unescape(_HTML_TAG_STRIP_RE.sub("", raw)).rstrip()
        if len(code) < _MIN_CODE_BLOCK_CHARS:
            continue
        if len(code) > _MAX_CODE_BLOCK_CHARS:
            code = code[:_MAX_CODE_BLOCK_CHARS].rstrip() + "\n# ... (truncated)"

        cls_match = _HTML_CLASS_RE.search(attrs or "")
        lang = _detect_lang_from_class(cls_match.group(1)) if cls_match else None
        if not lang or lang == "text":
            lang = _detect_lang_from_content(code)

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
        # Harvest code from raw HTML (preserves language attribute) and fall
        # back to detection from content for unlabelled blocks.
        code_blocks = _harvest_code_blocks(html)
        if code_blocks:
            langs = ", ".join(b["language"] for b in code_blocks)
            print(f"  [code]  harvested {len(code_blocks)} block(s) [{langs}]")

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
