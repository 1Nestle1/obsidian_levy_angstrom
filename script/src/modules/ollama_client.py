"""Thin Ollama /api/generate wrapper with retries + model fallback."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from config import OLLAMA_URL


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def pick_available(self, preferred: List[str]) -> Optional[str]:
        available = set(self.list_models())
        for name in preferred:
            if name in available:
                return name
        return None

    def generate(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        stop: Optional[List[str]] = None,
        retries: int = 2,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if stop:
            payload["options"]["stop"] = stop
        if json_mode:
            payload["format"] = "json"

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except requests.RequestException as e:
                last_err = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))

        raise OllamaError(f"Ollama generate failed after {retries + 1} tries: {last_err}")

    def generate_json(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        retries: int = 1,
    ) -> Optional[Any]:
        """Generate with format=json and parse; one repair retry on parse failure."""
        raw = self.generate(model, prompt, options=options, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        if retries <= 0:
            return None

        repair_prompt = (
            "The following text should be valid JSON but isn't. "
            "Return ONLY valid JSON, nothing else:\n\n" + raw
        )
        try:
            fixed = self.generate(model, repair_prompt, options=options, json_mode=True)
            return json.loads(fixed)
        except (json.JSONDecodeError, OllamaError):
            return None
