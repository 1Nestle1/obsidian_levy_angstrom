"""OpenAI-compatible chat/completions client (Groq, OpenRouter, Gemini, Cerebras, ...).

Duck-types the same interface as OllamaClient: list_models / generate / generate_json.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import HTTP_BACKOFF, HTTP_RETRIES


class LLMError(RuntimeError):
    pass


# Ollama-style option keys → OpenAI-compat chat params.
_OPT_RENAME = {
    "num_predict": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "stop": "stop",
}
_OPT_DROP = {"num_ctx", "repeat_penalty"}


def _translate_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not options:
        return {}
    out: Dict[str, Any] = {}
    for k, v in options.items():
        if k in _OPT_DROP:
            continue
        out[_OPT_RENAME.get(k, k)] = v
    return out


def _build_session() -> requests.Session:
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 180,
        provider_label: str = "openai_compat",
    ):
        if not base_url:
            raise LLMError("OpenAICompatClient requires base_url")
        if not api_key:
            raise LLMError("OpenAICompatClient requires api_key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.provider_label = provider_label
        self.session = _build_session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def list_models(self) -> List[str]:
        try:
            r = self.session.get(f"{self.base_url}/models", timeout=15)
            r.raise_for_status()
            data = r.json().get("data") or r.json().get("models") or []
            out: List[str] = []
            for item in data:
                if isinstance(item, dict):
                    name = item.get("id") or item.get("name")
                    if name:
                        out.append(name)
                elif isinstance(item, str):
                    out.append(item)
            return out
        except (requests.RequestException, ValueError):
            return []

    def pick_available(self, preferred: List[str]) -> Optional[str]:
        available = set(self.list_models())
        if not available:
            return preferred[0] if preferred else None
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
        params = _translate_options(options)
        if stop:
            params["stop"] = stop

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **params,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError(f"empty choices in response: {data}")
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
                return content.strip()
            except (requests.RequestException, ValueError, LLMError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))

        raise LLMError(
            f"{self.provider_label} generate failed after {retries + 1} tries: {last_err}"
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None,
        retries: int = 1,
    ) -> Optional[Any]:
        raw = self.generate(model, prompt, options=options, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        if retries <= 0:
            return None

        repair = (
            "The following should be valid JSON but isn't. "
            "Return ONLY valid JSON, nothing else:\n\n" + raw
        )
        try:
            fixed = self.generate(model, repair, options=options, json_mode=True)
            return json.loads(fixed)
        except (json.JSONDecodeError, LLMError):
            return None
