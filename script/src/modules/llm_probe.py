"""Probe LLM providers: reachability, auth, available models, latency.

Runs all configured providers in parallel and renders a picker table.
Also caches the last chosen provider so re-runs don't need to re-probe.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

import config as _cfg
from src.modules.llm_factory import list_providers, resolve_provider
from src.modules.openai_compat_client import LLMError


_CACHE_FILE = _cfg.CACHE_DIR / "last_provider.json"


@dataclass
class ProbeResult:
    name: str
    reachable: bool = False
    auth_ok: bool = False
    available_models: List[str] = field(default_factory=list)
    latency_ms: int = 0
    extract_model: str = ""
    synthesize_model: str = ""
    extract_available: bool = False
    synthesize_available: bool = False
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.reachable and self.auth_ok

    @property
    def status(self) -> str:
        if self.error and not self.reachable:
            return "✗ unreachable"
        if not self.auth_ok:
            return "✗ no key"
        if not self.available_models:
            return "? reachable, no models"
        if not self.extract_available or not self.synthesize_available:
            return "! model missing"
        return "✓ ready"


def probe_provider(name: str, timeout: float = 5.0) -> ProbeResult:
    result = ProbeResult(name=name)
    try:
        resolved = resolve_provider(name)
    except LLMError as e:
        result.error = str(e)
        return result

    result.extract_model = resolved.extract_model
    result.synthesize_model = resolved.synthesize_model

    profile = _cfg.PROVIDER_PROFILES[name]
    needs_key = bool(profile.get("api_key_env"))

    if needs_key and not resolved.api_key:
        result.error = f"missing env var {profile['api_key_env']}"
        # Still try to note the base_url exists; just can't auth.
        result.auth_ok = False
        return result

    # Ollama has a different endpoint; everything else is OpenAI-compat /models.
    t0 = time.perf_counter()
    try:
        if name == "ollama":
            r = requests.get(f"{resolved.base_url.rstrip('/')}/api/tags", timeout=timeout)
            r.raise_for_status()
            result.available_models = [m["name"] for m in r.json().get("models", [])]
            result.auth_ok = True  # no auth needed
        else:
            if not resolved.base_url:
                result.error = "no base_url configured"
                return result
            r = requests.get(
                f"{resolved.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {resolved.api_key}"},
                timeout=timeout,
            )
            if r.status_code == 401:
                result.reachable = True
                result.error = "auth rejected (401)"
                return result
            r.raise_for_status()
            payload = r.json()
            items = payload.get("data") or payload.get("models") or []
            models: List[str] = []
            for it in items:
                if isinstance(it, dict):
                    mid = it.get("id") or it.get("name")
                    if mid:
                        models.append(mid)
                elif isinstance(it, str):
                    models.append(it)
            result.available_models = models
            result.auth_ok = True
        result.reachable = True
    except requests.RequestException as e:
        result.error = f"{type(e).__name__}: {e}"
        return result
    finally:
        result.latency_ms = int((time.perf_counter() - t0) * 1000)

    available_set = set(result.available_models)
    result.extract_available = (
        result.extract_model in available_set if available_set else True
    )
    result.synthesize_available = (
        result.synthesize_model in available_set if available_set else True
    )
    return result


def probe_all(timeout: float = 5.0) -> List[ProbeResult]:
    names = list_providers()
    results: List[Optional[ProbeResult]] = [None] * len(names)
    with ThreadPoolExecutor(max_workers=min(6, len(names))) as ex:
        futs = {ex.submit(probe_provider, n, timeout): i for i, n in enumerate(names)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[i] = ProbeResult(name=names[i], error=f"probe crashed: {e}")
    return [r for r in results if r is not None]


def render_table(results: List[ProbeResult]) -> str:
    header = f"{'#':<3}{'provider':<12}{'status':<22}{'models':<8}{'extract':<28}{'synthesize':<28}{'latency':<10}"
    lines = [header, "-" * len(header)]
    for i, r in enumerate(results, 1):
        models_n = str(len(r.available_models)) if r.reachable else "-"
        ex_ok = "" if r.extract_available else "✗"
        sy_ok = "" if r.synthesize_available else "✗"
        ext = f"{r.extract_model[:25]}{ex_ok}" if r.extract_model else "-"
        syn = f"{r.synthesize_model[:25]}{sy_ok}" if r.synthesize_model else "-"
        lat = f"{r.latency_ms}ms" if r.latency_ms else "-"
        lines.append(
            f"{i:<3}{r.name:<12}{r.status:<22}{models_n:<8}{ext:<28}{syn:<28}{lat:<10}"
        )
        if r.error and not r.reachable:
            lines.append(f"     └─ {r.error}")
    return "\n".join(lines)


def choose_llm(timeout: float = 5.0) -> Optional[str]:
    """Probe all providers and let user pick. Returns provider name or None."""
    print("\nProbing providers...")
    results = probe_all(timeout=timeout)
    print(render_table(results))

    ready = [r for r in results if r.ready]
    if not ready:
        print("\n[abort] no providers ready. Fix API keys or start Ollama, then retry.")
        return None

    while True:
        raw = input(f"\nPick provider [1-{len(results)}] (enter to cancel): ").strip().lower()
        if not raw:
            return None
        if raw.isdigit():
            i = int(raw)
            if 1 <= i <= len(results):
                chosen = results[i - 1]
                if not chosen.ready:
                    print(f"  {chosen.name} is not ready ({chosen.status}). Pick another.")
                    continue
                save_last_provider(chosen.name)
                return chosen.name
        # name match
        for r in results:
            if r.name == raw and r.ready:
                save_last_provider(r.name)
                return r.name
        print("  invalid choice")


def save_last_provider(name: str) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({"provider": name}), encoding="utf-8")
    except OSError:
        pass


def load_last_provider() -> Optional[str]:
    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        name = data.get("provider")
        if isinstance(name, str) and name in list_providers():
            return name
    except (OSError, ValueError):
        pass
    return None
