"""Two-pass LLM note generation: extract-per-source then synthesize note pack."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    CACHE_DIR,
    EXTRACT_OPTIONS,
    LLM_CACHE_TTL,
    MAX_SOURCE_CHARS_PER_PASS1,
    MODEL_EXTRACT,
    MODEL_SYNTHESIZE,
    SYNTHESIZE_OPTIONS,
)
from src.modules.ollama_client import OllamaClient, OllamaError
from src.utils.cache import JSONCache


_PASS1_PROMPT = """You are an extraction tool. Read ONE source and output JSON only.
Do not invent facts. If a field is unknown, use an empty list or string.

SOURCE_URL: {url}
SOURCE_TITLE: {title}
SOURCE_TEXT:
<<<
{text}
>>>

Output JSON with exactly these keys:
{{
  "url": "{url}",
  "one_line_summary": "string, <=200 chars",
  "key_claims":    [{{"claim": "string", "quote": "short verbatim quote"}}],
  "definitions":   [{{"term": "string", "definition": "1-2 sentences"}}],
  "concepts":      ["Concept Name", "..."],
  "open_questions":["string", "..."]
}}

Rules:
- Output ONLY JSON, no prose, no markdown fences.
- Keep it compact: at most 4 key_claims, 3 definitions, 5 concepts, 3 open_questions.
- `concepts` are generic/reusable names (e.g. "Backpropagation"), not full sentences.
- Every claim must be supported by SOURCE_TEXT.
- Do NOT output code examples — those are harvested separately."""


# ----------------------------- TECHNICAL TEMPLATE -----------------------------
# Used for code, math, ML, engineering topics. One big DOC with numbered sections.
# Examples section is LEFT EMPTY — we inject harvested code blocks deterministically
# after the LLM returns.
_PASS2_TECHNICAL = """You are writing a comprehensive Obsidian note on a technical topic.
Output ONE markdown document. No prose before or after. Stop at `=== END ===`.

TOPIC: {topic}
DOMAIN: {domain}
SUBTHEME: {subtheme}
ACCESS_DATE: {access_date}

EXISTING NOTES YOU CAN LINK (use `[[title]]` syntax when relevant):
{existing_links_block}

SOURCE SUMMARIES (JSON — only factual ground truth; do not invent facts):
{summaries_json}

RULES:
- Output ONE file, starting with the frontmatter below and ending with `=== END ===`.
- Write like a textbook chapter: clear, detailed, comprehensive. Aim for 700-1200 words.
- Use the exact section numbering and headers shown below.
- In section 4 "Examples", write ONLY the single line `<!-- EXAMPLES_PLACEHOLDER -->` and nothing else. Code examples are inserted automatically later.
- In section 7 "Sources", write ONLY the single line `<!-- SOURCES_PLACEHOLDER -->`. Sources are inserted automatically later.
- Never output literal placeholders like `<title>`, `<URL>`, `<Name>`, or text in (parentheses-as-instructions).
- When you reference a reusable concept, use `[[Concept Name]]` wikilinks; do not prefix with "Concept - ".
- In "Key Claims" under bullets, end each bullet with `(Source: https://...)` where the URL matches a SOURCE SUMMARIES url.

BEGIN OUTPUT:

=== FILE: 01_Topics/{topic_filename}.md ===
---
topic: {topic}
parent: "[[{parent_link}]]"
domain: {domain}
subtheme: {subtheme}
created: {access_date}
tags: [topic, {family_tag}]
---
# {topic}: A Comprehensive Guide

## 1. Overview

(write 2-3 paragraph overview: what the topic is, why it matters, where it fits)

## 2. Key Concepts

(bullet list: each bullet is `- **Term**: 1-2 sentence definition`. 4-8 bullets.)

## 3. How It Works

(explain the mechanics in 2-4 paragraphs. Prose, not bullets.)

## 4. Examples

<!-- EXAMPLES_PLACEHOLDER -->

## 5. Best Practices

(bullet list of actionable guidelines, 4-7 bullets)

## 6. Common Pitfalls

(bullet list of mistakes to avoid, 3-6 bullets)

## 7. Sources

<!-- SOURCES_PLACEHOLDER -->

=== END ===
"""


# ----------------------------- ESSAY TEMPLATE -----------------------------
# Used for non-technical topics (history, philosophy, general knowledge).
_PASS2_ESSAY = """You are writing a personal Obsidian note on a general-knowledge topic.
Output ONE markdown document. Stop at `=== END ===`.

TOPIC: {topic}
DOMAIN: {domain}
SUBTHEME: {subtheme}
ACCESS_DATE: {access_date}

EXISTING NOTES YOU CAN LINK (use `[[title]]` syntax when relevant):
{existing_links_block}

SOURCE SUMMARIES (JSON — only factual ground truth):
{summaries_json}

RULES:
- Output ONE file only. 500-900 words.
- In "Sources", write ONLY `<!-- SOURCES_PLACEHOLDER -->`. It is filled in automatically.
- End every factual claim in "Key Points" with `(Source: https://...)` matching a URL from SOURCE SUMMARIES.
- No literal `<title>`, `<URL>`, or parenthesized instructions.

BEGIN OUTPUT:

=== FILE: 01_Topics/{topic_filename}.md ===
---
topic: {topic}
parent: "[[{parent_link}]]"
domain: {domain}
subtheme: {subtheme}
created: {access_date}
tags: [topic, {family_tag}]
---
# {topic}

## Overview

(2-3 paragraph summary)

## Background

(context and history; 2-3 paragraphs)

## Key Points

(bullet list of main claims, each ending with source citation)

## Open Questions

(bullet list of things left unresolved or debated)

## Sources

<!-- SOURCES_PLACEHOLDER -->

=== END ===
"""


class LLMNoteGenerator:
    def __init__(
        self,
        ollama: Optional[OllamaClient] = None,
        extract_model: str = MODEL_EXTRACT,
        synthesize_model: str = MODEL_SYNTHESIZE,
        synth_client: Optional[object] = None,
    ):
        """Primary `ollama` client drives Pass 1 (extract).
        `synth_client` drives Pass 2 (synthesize); defaults to the same client
        for single-provider mode. Hybrid mode passes a different client here."""
        self.ollama = ollama or OllamaClient()
        self.synth_client = synth_client or self.ollama
        self.extract_model = extract_model
        self.synthesize_model = synthesize_model
        self.cache = JSONCache(CACHE_DIR / "llm_pass1", ttl_seconds=LLM_CACHE_TTL)

    # ---------- Pass 1 ----------
    def extract_source(self, source: Dict) -> Optional[Dict]:
        url = source["url"]
        cache_key = f"{self.extract_model}::{url}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        text = source["content"][:MAX_SOURCE_CHARS_PER_PASS1]
        prompt = _PASS1_PROMPT.format(
            url=url,
            title=source.get("title", url),
            text=text,
        )

        try:
            data = self.ollama.generate_json(
                self.extract_model, prompt, options=EXTRACT_OPTIONS
            )
        except OllamaError as e:
            print(f"  [pass1] LLM error: {e}")
            return None

        if not isinstance(data, dict):
            print(f"  [pass1] invalid JSON from model for {url}")
            return None

        data.setdefault("url", url)
        self.cache.set(cache_key, data)
        return data

    def extract_all(self, sources: List[Dict]) -> List[Dict]:
        summaries = []
        for s in sources:
            print(f"  [pass1] {s['title'][:60]}")
            summary = self.extract_source(s)
            if summary:
                summaries.append(summary)
        return summaries

    # ---------- Pass 2 ----------
    def synthesize_pack(
        self,
        topic: str,
        topic_filename: str,
        domain: str,
        subtheme: str,
        summaries: List[Dict],
        existing_links: List[str],
        new_concepts: List[str],
        template_family: str = "essay",
        parent_topic: Optional[str] = None,
    ) -> Optional[str]:
        access_date = datetime.now(timezone.utc).date().isoformat()

        template = _PASS2_TECHNICAL if template_family == "technical" else _PASS2_ESSAY
        family_tag = template_family
        parent_link = parent_topic or f"MOC - {domain}"

        prompt = template.format(
            topic=topic,
            topic_filename=topic_filename,
            domain=domain,
            subtheme=subtheme,
            access_date=access_date,
            family_tag=family_tag,
            parent_link=parent_link,
            existing_links_block=_bullet(existing_links) or "(none)",
            summaries_json=json.dumps(summaries, indent=2, ensure_ascii=False),
        )

        try:
            return self.synth_client.generate(
                self.synthesize_model,
                prompt,
                options=SYNTHESIZE_OPTIONS,
                stop=["=== END ==="],
            )
        except Exception as e:  # noqa: BLE001 — ollama + cloud clients raise different errors
            print(f"[pass2] LLM error: {e}")
            return None


def _bullet(items: List[str]) -> str:
    return "\n".join(f"- {x}" for x in items)
