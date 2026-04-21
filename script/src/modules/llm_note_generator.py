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
  "code_examples": [{{"language": "python", "code": "string", "purpose": "string"}}],
  "open_questions":["string", "..."],
  "credibility_notes": "string"
}}

Rules:
- Output ONLY JSON, no prose, no markdown fences.
- `concepts` are generic/reusable names (e.g. "Backpropagation"), not full sentences.
- Every claim must be supported by SOURCE_TEXT."""


_PASS2_PROMPT = """You are writing a personal Obsidian research pack.

TOPIC: {topic}
TOPIC_FILENAME: {topic_filename}
DOMAIN: {domain}
SUBTHEME: {subtheme}
ACCESS_DATE: {access_date}

DOMAIN HUB LINK (use verbatim): [[MOC - {domain}]]

EXISTING NODES TO LINK (use each at least once as [[title]]):
{existing_links_block}

CONCEPT NAMES you MUST produce as their own 03_Concepts/ file blocks:
{new_concepts_block}

SOURCE SUMMARIES (JSON — ONLY factual ground truth; every citation must be one of these URLs):
{summaries_json}

OUTPUT RULES (READ CAREFULLY):
1. Produce the files listed below, ONCE each, in this order. Do not repeat any file.
2. Under each empty `##` heading, write real content drawn from SOURCE SUMMARIES.
   - If there is genuinely nothing to say for a heading, delete that heading entirely. Do NOT leave it blank and do NOT write placeholder text.
3. NEVER output text inside (parentheses-as-instructions) or <angle brackets>. Substitute real values.
4. NEVER output the literal strings `Concept - X`, `Concept - ...`, `<title>`, `<URL>`, or `<Name>`.
5. Wikilinks have the form [[Concept - RealName]] where RealName is a concept from CONCEPT NAMES above.
6. Every `(Source: https://...)` citation URL must match a URL in SOURCE SUMMARIES.
7. Each `## Key Claims` bullet must end with `(Source: https://...)`.
8. End the entire response with the literal line `=== END ===` and output nothing after it.

FILES TO PRODUCE (exact order, exact headers):

=== FILE: 02_Research/Research - {topic_filename}.md ===
---
topic: {topic}
domain: {domain}
subtheme: {subtheme}
created: {access_date}
tags: [research]
---
# Research - {topic}

## Abstract


## Key Claims


## Open Questions


## Takeaways / Next Actions


## Linked Concepts


=== FILE: 04_MOCs/MOC - {topic_filename}.md ===
---
topic: {topic}
tags: [moc]
---
# MOC - {topic}

- [[Research - {topic_filename}]]
- [[Sources - {topic_filename}]]
- Domain: [[MOC - {domain}]]
- Concepts:

=== FILE: 05_Sources/Sources - {topic_filename}.md ===
---
topic: {topic}
tags: [sources]
---
# Sources - {topic}


{concept_file_skeletons}

=== END ===
"""


def _concept_skeleton(concept_name: str) -> str:
    """Concept file skeleton — empty sections, no parenthetical hints."""
    return f"""=== FILE: 03_Concepts/{concept_name}.md ===
---
concept: {concept_name}
tags: [concept]
---
# {concept_name}

## Definition


## Why It Matters


## Example


## Sources

"""


class LLMNoteGenerator:
    def __init__(
        self,
        ollama: Optional[OllamaClient] = None,
        extract_model: str = MODEL_EXTRACT,
        synthesize_model: str = MODEL_SYNTHESIZE,
    ):
        self.ollama = ollama or OllamaClient()
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
    ) -> Optional[str]:
        access_date = datetime.now(timezone.utc).date().isoformat()

        # Ensure every concept is prefixed "Concept - " for consistency.
        normalized_concepts: List[str] = []
        for c in new_concepts or []:
            c = c.strip()
            if not c:
                continue
            if not c.lower().startswith("concept - "):
                c = f"Concept - {c}"
            normalized_concepts.append(c)
        if not normalized_concepts:
            # Ensure at least one concept so the pack is meaningful.
            normalized_concepts = [f"Concept - {topic.title()}"]

        concept_skeletons = "\n".join(_concept_skeleton(c) for c in normalized_concepts)

        prompt = _PASS2_PROMPT.format(
            topic=topic,
            topic_filename=topic_filename,
            domain=domain,
            subtheme=subtheme,
            access_date=access_date,
            existing_links_block=_bullet(existing_links) or "(none)",
            new_concepts_block=_bullet(normalized_concepts),
            summaries_json=json.dumps(summaries, indent=2, ensure_ascii=False),
            concept_file_skeletons=concept_skeletons,
        )

        try:
            return self.ollama.generate(
                self.synthesize_model,
                prompt,
                options=SYNTHESIZE_OPTIONS,
                stop=["=== END ==="],
            )
        except OllamaError as e:
            print(f"[pass2] LLM error: {e}")
            return None


def _bullet(items: List[str]) -> str:
    return "\n".join(f"- {x}" for x in items)
