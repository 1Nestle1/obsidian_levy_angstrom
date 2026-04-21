"""Content-addressed JSON cache for URL fetches and LLM outputs."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class JSONCache:
    def __init__(self, cache_dir: Path, ttl_seconds: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        if self.ttl is not None:
            age = time.time() - payload.get("_stored_at", 0)
            if age > self.ttl:
                return None

        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        payload = {"_key": key, "_stored_at": time.time(), "value": value}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
