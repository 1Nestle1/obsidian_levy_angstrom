"""Path-traversal guard for files written from LLM output."""
from pathlib import Path


class UnsafePathError(ValueError):
    pass


def resolve_under(root: Path, rel_path: str) -> Path:
    """Resolve rel_path under root, rejecting traversal outside root.

    Raises UnsafePathError if the final path escapes root.
    """
    root = Path(root).resolve()
    candidate = (root / rel_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise UnsafePathError(
            f"Path '{rel_path}' resolves outside vault root {root}"
        ) from e

    if any(part in ("..", "") for part in Path(rel_path).parts):
        raise UnsafePathError(f"Path '{rel_path}' contains traversal segments")

    return candidate


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically: temp file + os.replace."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)
