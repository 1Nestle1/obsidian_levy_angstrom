"""Vault-aware theme/subtheme detection with embedding candidate ranking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import EXTRACT_OPTIONS, MODEL_EXTRACT
from src.modules.embeddings import EmbeddingClient, top_k_similar
from src.modules.ollama_client import OllamaClient, OllamaError
from src.modules.vault_index import VaultIndex


@dataclass
class ThemeResult:
    domain: str
    domain_is_new: bool
    subtheme: str
    linked_existing_nodes: List[str]
    proposed_new_concepts: List[str]
    confidence: float
    reasoning: str
    template_family: str = "essay"  # "technical" or "essay"
    parent_topic: Optional[str] = None  # broader topic, e.g. "React" for "React Hooks"

    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "domain_is_new": self.domain_is_new,
            "subtheme": self.subtheme,
            "linked_existing_nodes": self.linked_existing_nodes,
            "proposed_new_concepts": self.proposed_new_concepts,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "template_family": self.template_family,
            "parent_topic": self.parent_topic,
        }


_TECHNICAL_DOMAINS = {
    "code", "programming", "software", "frontend", "backend", "web",
    "javascript", "typescript", "python", "rust", "go", "java",
    "react", "vue", "angular", "node", "devops", "cloud", "database",
    "math", "mathematics", "algebra", "calculus", "statistics",
    "machine learning", "ml", "ai", "deep learning", "algorithms",
    "physics", "engineering", "cryptography", "security",
}


def _infer_template_family(domain: str, query: str) -> str:
    """Heuristic: technical (code/math/sci) gets the big-DOC template,
    everything else gets the lighter essay template."""
    blob = f"{domain} {query}".lower()
    for marker in _TECHNICAL_DOMAINS:
        if marker in blob:
            return "technical"
    # Also treat any query with code-like tokens as technical.
    if any(tok in query for tok in ("()", "{}", "::", "->", "=>")):
        return "technical"
    return "essay"


_PROMPT_TEMPLATE = """You classify a research query for an Obsidian vault.

QUERY: {query}

EXISTING DOMAINS (MOCs already in vault):
{domains_block}

EXISTING TOPICS (you may suggest one of these as parent_topic):
{topics_block}

CANDIDATE EXISTING NODES (top matches by similarity — you may link these):
{candidates_block}

Output JSON only, no prose, no markdown fences:
{{
  "domain": "one of the existing MOC names above OR a short new domain name",
  "domain_is_new": true | false,
  "subtheme": "2-4 word phrase, e.g. 'Linear Algebra'",
  "parent_topic": "name of a BROADER topic that subsumes the query (e.g. 'React' for query 'React Hooks'), or null if the query is itself a top-level topic",
  "linked_existing_nodes": ["exact title from CANDIDATE list, verbatim, or empty list"],
  "proposed_new_concepts": ["short concept name", "..."],
  "confidence": 0.0,
  "reasoning": "one short sentence"
}}

HARD RULES:
- linked_existing_nodes entries MUST appear verbatim in the CANDIDATE list above. Never invent.
- If domain matches no existing MOC, set domain_is_new=true with a short new name (1-2 words).
- parent_topic is a BROADER concept than the query. If query is "React Hooks", parent_topic is "React". If query is itself top-level (e.g. "React"), parent_topic is null.
- proposed_new_concepts are short atomic names (e.g. "useEffect", "useState") — leaves of the topic tree, NOT broader concepts.
- confidence is a float between 0 and 1."""


class ThemeDetector:
    def __init__(
        self,
        vault_index: VaultIndex,
        ollama: Optional[OllamaClient] = None,
        embeddings: Optional[EmbeddingClient] = None,
        model: str = MODEL_EXTRACT,
    ):
        self.vault = vault_index
        self.ollama = ollama or OllamaClient()
        self.embeddings = embeddings or EmbeddingClient()
        self.model = model

    def _candidate_labels(self) -> List[str]:
        labels: List[str] = []
        labels.extend(n.filename_stem for n in self.vault.concepts)
        labels.extend(n.filename_stem for n in self.vault.research)
        labels.extend(n.filename_stem for n in self.vault.topics)
        return labels

    def _rank_candidates(self, query: str, top_k: int = 15) -> List[str]:
        labels = self._candidate_labels()
        if not labels:
            return []

        query_vec = self.embeddings.embed(query)
        if query_vec is None:
            # Fallback: return first top_k unsorted (better than nothing).
            return labels[:top_k]

        label_vecs = self.embeddings.embed_many(labels)
        if not label_vecs:
            return labels[:top_k]

        ranked = top_k_similar(query_vec, label_vecs, k=top_k)
        return [label for label, _score in ranked]

    def detect(self, query: str) -> ThemeResult:
        domains = self.vault.domain_titles()
        candidates = self._rank_candidates(query)
        topic_titles = [n.filename_stem for n in self.vault.topics]

        prompt = _PROMPT_TEMPLATE.format(
            query=query,
            domains_block=_bullet(domains) or "(none yet — suggest a new domain)",
            topics_block=_bullet(topic_titles) or "(none yet)",
            candidates_block=_bullet(candidates) or "(vault is empty)",
        )

        try:
            data = self.ollama.generate_json(
                self.model, prompt, options=EXTRACT_OPTIONS
            )
        except OllamaError as e:
            print(f"[theme] LLM failure, falling back: {e}")
            data = None

        if not isinstance(data, dict):
            return _fallback_result(query, candidates)

        return _validate_result(data, candidates, domains, query=query)


def _bullet(items: List[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


def _fallback_result(query: str, candidates: List[str]) -> ThemeResult:
    return ThemeResult(
        domain="General",
        domain_is_new=True,
        subtheme=query[:40],
        linked_existing_nodes=[],
        proposed_new_concepts=[],
        confidence=0.0,
        reasoning="LLM unavailable or returned invalid JSON; using fallback.",
        template_family=_infer_template_family("General", query),
    )


def _validate_result(
    data: Dict,
    candidates: List[str],
    existing_domains: List[str],
    query: str = "",
) -> ThemeResult:
    candidate_set = set(candidates)
    linked = [x for x in data.get("linked_existing_nodes", []) if x in candidate_set]

    domain = str(data.get("domain") or "General").strip() or "General"
    domain_is_new = bool(data.get("domain_is_new", domain not in existing_domains))

    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    parent = data.get("parent_topic")
    parent_str = str(parent).strip() if parent else None
    if parent_str and parent_str.lower() in {"null", "none", ""}:
        parent_str = None

    return ThemeResult(
        domain=domain,
        domain_is_new=domain_is_new,
        subtheme=str(data.get("subtheme") or "").strip(),
        linked_existing_nodes=linked,
        proposed_new_concepts=[
            str(c).strip() for c in data.get("proposed_new_concepts", []) if c
        ],
        confidence=max(0.0, min(1.0, conf)),
        reasoning=str(data.get("reasoning") or "").strip(),
        template_family=_infer_template_family(
            domain, f"{query} {data.get('subtheme') or ''}"
        ),
        parent_topic=parent_str,
    )
