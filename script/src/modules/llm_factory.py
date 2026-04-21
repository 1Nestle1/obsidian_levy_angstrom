"""Factory: returns an LLM client based on LLM_PROVIDER env/config.

All returned clients share the same duck-typed interface used by the pipeline:
    list_models()  -> List[str]
    generate(model, prompt, options=None, json_mode=False, stop=None, retries=2) -> str
    generate_json(model, prompt, options=None, retries=1) -> Optional[Any]
"""
from __future__ import annotations

from typing import Protocol

from config import LLM_API_KEY, LLM_BASE_URL, LLM_PROVIDER
from src.modules.ollama_client import OllamaClient
from src.modules.openai_compat_client import LLMError, OpenAICompatClient


class LLMClient(Protocol):
    def list_models(self) -> list[str]: ...
    def generate(self, model, prompt, options=None, json_mode=False, stop=None, retries=2) -> str: ...
    def generate_json(self, model, prompt, options=None, retries=1): ...


def get_llm_client() -> LLMClient:
    """Instantiate the configured LLM client. Raises if cloud creds are missing."""
    provider = LLM_PROVIDER

    if provider == "ollama":
        return OllamaClient()

    if not LLM_BASE_URL:
        raise LLMError(f"LLM_PROVIDER={provider} requires a base_url (check config).")
    if not LLM_API_KEY:
        raise LLMError(
            f"LLM_PROVIDER={provider} requires an API key in the expected env var. "
            "See PROVIDER_PROFILES in config.py for which variable to set."
        )

    return OpenAICompatClient(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        provider_label=provider,
    )


def describe_provider() -> str:
    return f"provider={LLM_PROVIDER} base_url={LLM_BASE_URL or '(local)'}"
