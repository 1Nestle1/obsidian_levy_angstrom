"""Factory: returns an LLM client and resolves model tiers.

Every client shares the same duck-typed interface used by the pipeline:
    list_models()  -> List[str]
    generate(model, prompt, options=None, json_mode=False, stop=None, retries=2) -> str
    generate_json(model, prompt, options=None, retries=1) -> Optional[Any]
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol

import config as _cfg
from src.modules.ollama_client import OllamaClient
from src.modules.openai_compat_client import LLMError, OpenAICompatClient


class LLMClient(Protocol):
    def list_models(self) -> list[str]: ...
    def generate(self, model, prompt, options=None, json_mode=False, stop=None, retries=2) -> str: ...
    def generate_json(self, model, prompt, options=None, retries=1): ...


@dataclass
class ResolvedProvider:
    name: str
    base_url: str
    api_key: str
    extract_model: str
    synthesize_model: str


def resolve_provider(provider: Optional[str] = None) -> ResolvedProvider:
    """Resolve the active provider, honoring --provider override, then env, then default."""
    name = (provider or _cfg.LLM_PROVIDER).lower()
    profile = _cfg.PROVIDER_PROFILES.get(name)
    if profile is None:
        raise LLMError(
            f"Unknown provider '{name}'. "
            f"Valid: {', '.join(_cfg.PROVIDER_PROFILES.keys())}"
        )

    api_key = ""
    if profile["api_key_env"]:
        api_key = os.environ.get(profile["api_key_env"], "")

    # Env-var model overrides take precedence over profile defaults.
    extract = os.environ.get("MODEL_EXTRACT") or profile["extract"]
    synthesize = os.environ.get("MODEL_SYNTHESIZE") or profile["synthesize"]

    return ResolvedProvider(
        name=name,
        base_url=profile["base_url"],
        api_key=api_key,
        extract_model=extract,
        synthesize_model=synthesize,
    )


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Instantiate the configured LLM client. Raises LLMError if cloud creds missing."""
    p = resolve_provider(provider)

    if p.name == "ollama":
        return OllamaClient(base_url=p.base_url)

    if not p.base_url:
        raise LLMError(f"Provider '{p.name}' needs a base_url (check config).")
    if not p.api_key:
        profile = _cfg.PROVIDER_PROFILES[p.name]
        raise LLMError(
            f"Provider '{p.name}' needs API key in env var "
            f"{profile['api_key_env']!r}."
        )

    return OpenAICompatClient(
        base_url=p.base_url,
        api_key=p.api_key,
        provider_label=p.name,
    )


def describe_provider(provider: Optional[str] = None) -> str:
    try:
        p = resolve_provider(provider)
    except LLMError as e:
        return f"(unresolved: {e})"
    loc = p.base_url or "(local)"
    return f"provider={p.name} base_url={loc} extract={p.extract_model} synth={p.synthesize_model}"


def list_providers() -> list[str]:
    return list(_cfg.PROVIDER_PROFILES.keys())


def interactive_pick(available: Optional[list[str]] = None) -> str:
    """Prompt the user to pick a provider. Returns the chosen provider name."""
    providers = available or list_providers()
    print("\nPick LLM provider:")
    for i, name in enumerate(providers, 1):
        hint = _provider_hint(name)
        print(f"  {i}. {name:10s}  {hint}")
    while True:
        raw = input(f"Choice [1-{len(providers)}] (or name): ").strip().lower()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(providers):
                return providers[idx - 1]
        if raw in providers:
            return raw
        print("  invalid choice")


def _provider_hint(name: str) -> str:
    profile = _cfg.PROVIDER_PROFILES.get(name, {})
    env = profile.get("api_key_env")
    if not env:
        return "local, no API key"
    set_marker = "✓" if os.environ.get(env) else "✗"
    return f"needs ${env} [{set_marker}]"
