"""Parse '=== FILE: path ===' blocks from LLM synthesis output and write safely."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from src.utils.filenames import sanitize_filename
from src.utils.safe_paths import atomic_write_text, resolve_under, UnsafePathError

_FILE_HEADER = re.compile(r"^===\s*FILE:\s*(?P<path>.+?)\s*===\s*$", re.MULTILINE)
_END_MARKER = re.compile(r"^===\s*END\s*===\s*$", re.MULTILINE)
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")

# Literal template placeholders that small models sometimes echo verbatim.
_PLACEHOLDER_LITERALS = [
    "[[Concept - X]]",
    "[[Concept - ...]]",
    "[<title>](<URL>)",
    "<title>",
    "<URL>",
    "<Name>",
]


def _strip_placeholder_lines(content: str) -> str:
    """Drop lines that are pure template placeholder leakage."""
    keep = []
    for line in content.splitlines():
        if any(lit in line for lit in _PLACEHOLDER_LITERALS):
            continue
        keep.append(line)
    return "\n".join(keep).rstrip() + "\n"


@dataclass
class PackFile:
    rel_path: str
    content: str


@dataclass
class ParsedPack:
    files: List[PackFile]

    def by_kind(self, folder_prefix: str) -> List[PackFile]:
        return [f for f in self.files if f.rel_path.startswith(folder_prefix)]

    def concept_filenames(self) -> Set[str]:
        out: Set[str] = set()
        for f in self.files:
            if f.rel_path.startswith("03_Concepts/"):
                stem = Path(f.rel_path).stem
                out.add(stem)
        return out


def parse(raw: str) -> ParsedPack:
    """Split raw LLM output into file blocks. Ignores anything before first header
    or after === END ===."""
    if not raw:
        return ParsedPack(files=[])

    end = _END_MARKER.search(raw)
    body = raw[: end.start()] if end else raw

    matches = list(_FILE_HEADER.finditer(body))
    # Collect every occurrence, then dedupe by rel_path keeping the longest body
    # (small models sometimes emit the same block 2-3 times with varying completeness).
    by_path: Dict[str, PackFile] = {}
    order: List[str] = []
    for i, m in enumerate(matches):
        rel_path = m.group("path").strip()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[content_start:content_end].strip() + "\n"
        content = _strip_placeholder_lines(content)

        prev = by_path.get(rel_path)
        if prev is None:
            by_path[rel_path] = PackFile(rel_path=rel_path, content=content)
            order.append(rel_path)
        elif len(content) > len(prev.content):
            by_path[rel_path] = PackFile(rel_path=rel_path, content=content)

    files = [by_path[p] for p in order]
    return ParsedPack(files=files)


def sanitize_rel_path(rel_path: str) -> str:
    """Sanitize filename portion while keeping the folder intact."""
    p = Path(rel_path)
    if not p.parts:
        return rel_path

    folder = Path(*p.parts[:-1]) if len(p.parts) > 1 else Path("")
    stem = p.stem
    suffix = p.suffix or ".md"

    safe_stem = sanitize_filename(stem)
    safe_name = f"{safe_stem}{suffix}"
    return str(folder / safe_name) if str(folder) else safe_name


@dataclass
class WriteReport:
    written: List[Path]
    skipped: List[tuple[str, str]]    # (rel_path, reason)


def write_pack(pack: ParsedPack, vault_root: Path) -> WriteReport:
    written: List[Path] = []
    skipped: List[tuple[str, str]] = []

    for f in pack.files:
        safe_rel = sanitize_rel_path(f.rel_path)
        try:
            target = resolve_under(vault_root, safe_rel)
        except UnsafePathError as e:
            skipped.append((f.rel_path, f"unsafe path: {e}"))
            continue

        try:
            atomic_write_text(target, f.content)
            written.append(target)
        except OSError as e:
            skipped.append((f.rel_path, f"write failed: {e}"))

    return WriteReport(written=written, skipped=skipped)


@dataclass
class ValidationReport:
    missing_concepts: List[str]          # concept wikilinks with no matching file block
    unknown_links: List[str]             # wikilinks that neither exist in vault nor in pack
    orphan_concept_files: List[str]      # concept files not referenced anywhere


def validate_pack(
    pack: ParsedPack,
    required_concepts: List[str],
    existing_vault_titles: Set[str],
) -> ValidationReport:
    concept_files = pack.concept_filenames()
    text_blob = "\n\n".join(f.content for f in pack.files)

    referenced: Set[str] = set()
    for m in _WIKILINK.finditer(text_blob):
        referenced.add(m.group(1).strip())

    def _normalize_required(name: str) -> str:
        name = name.strip()
        if not name.lower().startswith("concept - "):
            name = f"Concept - {name}"
        return name

    missing_concepts: List[str] = []
    for required in required_concepts:
        normalized = _normalize_required(required)
        if normalized not in concept_files and normalized not in existing_vault_titles:
            missing_concepts.append(normalized)

    unknown_links: List[str] = []
    for link in referenced:
        if link in concept_files:
            continue
        if link in existing_vault_titles:
            continue
        if link.startswith(("Research - ", "MOC - ", "Sources - ", "Concept - ")):
            if link.startswith("Concept - ") and link not in concept_files:
                unknown_links.append(link)
            continue

    orphan_concepts = [
        stem for stem in concept_files if stem not in referenced
    ]

    return ValidationReport(
        missing_concepts=missing_concepts,
        unknown_links=unknown_links,
        orphan_concept_files=orphan_concepts,
    )
