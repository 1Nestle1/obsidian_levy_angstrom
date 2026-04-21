"""Filesystem-safe filename sanitization (Windows-aware)."""
import re
import unicodedata

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RUN = re.compile(r"\s+")


def sanitize_filename(name: str, max_length: int = 120) -> str:
    """Return a filesystem-safe filename stem (no extension).

    - NFKD normalize + strip non-ASCII marks
    - Replace illegal characters with '-'
    - Collapse whitespace
    - Trim trailing dots/spaces (Windows quirk)
    - Reject reserved names
    - Cap length
    """
    if not name or not name.strip():
        return "untitled"

    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")

    cleaned = _ILLEGAL_CHARS.sub("-", normalized)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip(" .")

    if not cleaned:
        return "untitled"

    if cleaned.upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .-")

    return cleaned or "untitled"


def concept_filename(concept_title: str) -> str:
    """Build 'Concept - X.md' style filename from a title like 'Concept - X' or 'X'."""
    stem = concept_title.strip()
    if not stem.lower().startswith("concept - "):
        stem = f"Concept - {stem}"
    return f"{sanitize_filename(stem)}.md"
