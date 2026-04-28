"""Scan the Obsidian vault and build a live index of MOCs, Concepts, Research notes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import FOLDERS, VAULT_PATH

_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


@dataclass
class VaultNote:
    path: Path
    kind: str               # "moc" | "concept" | "research" | "topic" | "source"
    title: str              # H1 if present, else filename stem
    filename_stem: str
    links: List[str] = field(default_factory=list)


@dataclass
class VaultIndex:
    vault_root: Path
    mocs: List[VaultNote] = field(default_factory=list)
    concepts: List[VaultNote] = field(default_factory=list)
    research: List[VaultNote] = field(default_factory=list)
    topics: List[VaultNote] = field(default_factory=list)

    @property
    def all_notes(self) -> List[VaultNote]:
        return self.mocs + self.concepts + self.research + self.topics

    def find_concept(self, title: str) -> Optional[VaultNote]:
        needle = title.strip().lower()
        for note in self.concepts:
            if note.title.lower() == needle or note.filename_stem.lower() == needle:
                return note
        return None

    def domain_titles(self) -> List[str]:
        return [n.filename_stem for n in self.mocs if n.filename_stem.lower().startswith("moc - ")]


def _scan_folder(folder: Path, kind: str) -> List[VaultNote]:
    notes: List[VaultNote] = []
    if not folder.exists():
        return notes

    for path in folder.glob("*.md"):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        h1 = _H1.search(content)
        title = h1.group(1).strip() if h1 else path.stem
        links = [m.group(1).strip() for m in _WIKILINK.finditer(content)]

        notes.append(VaultNote(
            path=path, kind=kind, title=title,
            filename_stem=path.stem, links=links,
        ))
    return notes


def build_index(vault_path: str | Path = VAULT_PATH) -> VaultIndex:
    root = Path(vault_path)
    return VaultIndex(
        vault_root=root,
        mocs=_scan_folder(root / FOLDERS["mocs"], "moc"),
        concepts=_scan_folder(root / FOLDERS["concepts"], "concept"),
        research=_scan_folder(root / FOLDERS["research"], "research"),
        topics=_scan_folder(root / FOLDERS["topics"], "topic"),
    )


def ensure_vault_structure(vault_path: str | Path = VAULT_PATH) -> None:
    """Create the standard folder skeleton if missing."""
    root = Path(vault_path)
    root.mkdir(parents=True, exist_ok=True)
    for sub in FOLDERS.values():
        (root / sub).mkdir(parents=True, exist_ok=True)
