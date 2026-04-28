"""Idempotent writers for parent topics, concept stubs, and Children-list links.

Every helper here is safe to call repeatedly: existing files are inspected and
only appended to, never overwritten. This protects user edits across re-runs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.filenames import sanitize_filename


_CHILDREN_HEADER = "## Children"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------
# Concept leaf stubs (small atomic notes in 03_Concepts/)
# --------------------------------------------------------------------------

def write_concept_stub(
    vault_root: Path,
    name: str,
    definition: str,
    parent_topic: str,
    domain: str,
    source_url: str = "",
    source_title: str = "",
) -> Optional[Path]:
    """Create `03_Concepts/{Name}.md` if missing. If it exists, do nothing.
    Returns the file path on creation, None when skipped."""
    fname = sanitize_filename(name)
    if not fname:
        return None
    target = vault_root / "03_Concepts" / f"{fname}.md"
    if target.exists():
        return None

    body_lines = [
        "---",
        f"concept: {name}",
        f'parent: "[[{parent_topic}]]"',
        f"domain: {domain}",
        f"created: {_today()}",
        "tags: [concept, leaf]",
        "---",
        f"# {name}",
        "",
    ]
    if definition:
        body_lines += [f"> {definition}", ""]

    body_lines += [
        f"Part of [[{parent_topic}]].",
        "",
        "## Notes",
        "",
        "*(Stub — run `smart_research.py \"" + name + "\"` to expand.)*",
        "",
    ]

    if source_url:
        link_text = source_title or source_url
        body_lines += ["## Mentioned in", "", f"- [{link_text}]({source_url})", ""]

    _write(target, "\n".join(body_lines))
    return target


# --------------------------------------------------------------------------
# Parent topic stubs (slim hub in 01_Topics/)
# --------------------------------------------------------------------------

def write_parent_topic_stub(
    vault_root: Path,
    name: str,
    domain: str,
    grandparent: Optional[str] = None,
    blurb: str = "",
) -> Optional[Path]:
    """Create `01_Topics/{Name}.md` as a slim navigation hub if missing.
    Grandparent defaults to the domain MOC. Returns the path on creation."""
    fname = sanitize_filename(name)
    if not fname:
        return None
    target = vault_root / "01_Topics" / f"{fname}.md"
    if target.exists():
        return None

    parent_link = grandparent or f"MOC - {domain}"
    body_lines = [
        "---",
        f"topic: {name}",
        f'parent: "[[{parent_link}]]"',
        f"domain: {domain}",
        f"created: {_today()}",
        "tags: [topic, hub]",
        "---",
        f"# {name}",
        "",
    ]
    if blurb:
        body_lines += [blurb, ""]
    body_lines += [
        f"Part of [[{parent_link}]].",
        "",
        _CHILDREN_HEADER,
        "",
    ]
    _write(target, "\n".join(body_lines))
    return target


# --------------------------------------------------------------------------
# Idempotent child-link append: adds `- [[Child]] — subtheme` to ## Children
# --------------------------------------------------------------------------

def append_child_link(
    parent_path: Path,
    child_name: str,
    subtheme: str = "",
) -> bool:
    """Append `- [[Child]]` under `## Children` in parent_path. Idempotent —
    skips if the link is already present anywhere in the file. Returns True
    if the file changed."""
    if not parent_path.exists():
        return False

    text = _read(parent_path)
    link_marker = f"[[{child_name}]]"
    if link_marker in text:
        return False

    new_line = f"- [[{child_name}]]"
    if subtheme:
        new_line += f" — {subtheme}"

    if _CHILDREN_HEADER in text:
        # Insert right after the header (preserve any existing children below).
        idx = text.find(_CHILDREN_HEADER)
        # Find end of the header line + the blank line after it (if any).
        line_end = text.find("\n", idx)
        # Walk past consecutive list bullets to keep ordering stable: we append
        # at the end of the children block, before the next non-bullet line.
        cursor = line_end + 1
        while cursor < len(text):
            line_break = text.find("\n", cursor)
            if line_break == -1:
                line_break = len(text)
            line = text[cursor:line_break]
            if line.startswith("- ") or line.strip() == "":
                cursor = line_break + 1
                continue
            break
        # cursor now points at either the next section or EOF.
        new_text = text[:cursor].rstrip() + "\n" + new_line + "\n\n" + text[cursor:].lstrip()
    else:
        # No Children section — append one at the end of the file.
        new_text = text.rstrip() + f"\n\n{_CHILDREN_HEADER}\n\n{new_line}\n"

    _write(parent_path, new_text)
    return True


# --------------------------------------------------------------------------
# Domain MOC: list ROOT topics only (those whose parent is the MOC itself)
# --------------------------------------------------------------------------

def link_root_topic_in_moc(
    vault_root: Path, domain: str, root_topic: str, subtheme: str = "",
) -> Optional[Path]:
    """Add a root topic to its domain MOC. Creates the MOC if missing.
    Idempotent on the link."""
    moc_path = vault_root / "04_MOCs" / f"MOC - {sanitize_filename(domain)}.md"
    if moc_path.exists():
        text = _read(moc_path)
        if f"[[{root_topic}]]" in text:
            return moc_path
        new_line = f"- [[{root_topic}]]"
        if subtheme:
            new_line += f" — {subtheme}"
        _write(moc_path, text.rstrip() + "\n" + new_line + "\n")
    else:
        new_line = f"- [[{root_topic}]]"
        if subtheme:
            new_line += f" — {subtheme}"
        _write(moc_path, (
            f"---\ndomain: {domain}\ntags: [moc]\ncreated: {_today()}\n---\n"
            f"# MOC - {domain}\n\nTopics in this domain:\n\n{new_line}\n"
        ))
    return moc_path


# --------------------------------------------------------------------------
# Concept stub batch helper (caps at 5, ranks by mention frequency)
# --------------------------------------------------------------------------

def rank_concepts_by_mentions(
    summaries: List[Dict], proposed: List[str], max_n: int = 5,
) -> List[str]:
    """Order concepts by how many summaries mention them. Caps at max_n.
    Strips the 'Concept - ' prefix the LLM sometimes adds."""
    cleaned = []
    seen = set()
    for c in proposed or []:
        name = (c or "").strip()
        if name.lower().startswith("concept - "):
            name = name[len("Concept - "):].strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)

    # Score by count of summaries that mention each concept (case-insensitive).
    def mentions(name: str) -> int:
        needle = name.lower()
        count = 0
        for s in summaries:
            blob = (s.get("one_line_summary", "") + " "
                    + " ".join(c.get("claim", "") for c in s.get("key_claims", []))
                    + " " + " ".join(s.get("concepts", []))).lower()
            if needle in blob:
                count += 1
        return count

    cleaned.sort(key=mentions, reverse=True)
    return cleaned[:max_n]


def find_definition(summaries: List[Dict], concept: str) -> str:
    """Pull a 1-2 sentence definition for `concept` from Pass 1 summaries."""
    needle = concept.strip().lower()
    for s in summaries:
        for d in s.get("definitions", []) or []:
            term = (d.get("term") or "").strip().lower()
            if term == needle or term.endswith(needle) or needle in term:
                defn = (d.get("definition") or "").strip()
                if defn:
                    return defn
    return ""
