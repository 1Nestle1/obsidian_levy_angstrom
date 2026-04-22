"""Configuration for the research pipeline."""
import os
from pathlib import Path

# === PATHS ===
# Point this at your Obsidian vault root. Override via env var OBSIDIAN_VAULT.
VAULT_PATH = os.environ.get(
    "OBSIDIAN_VAULT",
    r"C:\Users\Artyom\Documents\VS\LEVY\levy_angstrom",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = _SCRIPT_DIR / "data" / "cache"
RUNS_DIR = _SCRIPT_DIR / "data" / "runs"

# === VAULT STRUCTURE ===
FOLDERS = {
    "topics":   "01_Topics",
    "research": "02_Research",
    "concepts": "03_Concepts",
    "mocs":     "04_MOCs",
    "sources":  "05_Sources",
    "code":     "06_Code",
    "templates": "_templates",
    "assets":   "_assets",
}

# === SERVICES ===
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# === LLM PROVIDER ===
# Choose how the pipeline talks to an LLM. Override via env var LLM_PROVIDER.
#   ollama     - local Ollama at OLLAMA_URL (default)
#   groq       - https://api.groq.com (needs GROQ_API_KEY)
#   openrouter - https://openrouter.ai (needs OPENROUTER_API_KEY)
#   gemini     - Google Gemini via OpenAI-compat endpoint (needs GEMINI_API_KEY)
#   cerebras   - https://api.cerebras.ai (needs CEREBRAS_API_KEY)
#   github     - GitHub Models (needs GITHUB_TOKEN)
#   custom     - any OpenAI-compat endpoint; set OPENAI_COMPAT_BASE_URL + OPENAI_COMPAT_API_KEY
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

# Per-provider defaults. Models chosen to favor the provider's free tier.
PROVIDER_PROFILES = {
    "ollama": {
        "base_url": OLLAMA_URL,
        "api_key_env": None,
        "extract":    "gemma3:1b",
        "synthesize": "gemma3:1b",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "extract":    "llama-3.1-8b-instant",
        "synthesize": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "extract":    "meta-llama/llama-3.3-70b-instruct:free",
        "synthesize": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "extract":    "gemini-2.0-flash",
        "synthesize": "gemini-2.5-flash",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "extract":    "llama-3.1-8b",
        "synthesize": "llama-3.3-70b",
    },
    "github": {
        "base_url": "https://models.inference.ai.azure.com",
        "api_key_env": "GITHUB_TOKEN",
        "extract":    "gpt-4o-mini",
        "synthesize": "gpt-4o-mini",
    },
    "custom": {
        "base_url": os.environ.get("OPENAI_COMPAT_BASE_URL", ""),
        "api_key_env": "OPENAI_COMPAT_API_KEY",
        "extract":    os.environ.get("MODEL_EXTRACT", ""),
        "synthesize": os.environ.get("MODEL_SYNTHESIZE", ""),
    },
}

_active = PROVIDER_PROFILES.get(LLM_PROVIDER, PROVIDER_PROFILES["ollama"])

# Resolved model tiers (env var overrides profile default).
MODEL_EXTRACT = os.environ.get("MODEL_EXTRACT") or _active["extract"]
MODEL_SYNTHESIZE = os.environ.get("MODEL_SYNTHESIZE") or _active["synthesize"]
# Embeddings stay local by default even in cloud mode — cheap, fast, no quota.
MODEL_EMBED = os.environ.get("MODEL_EMBED", "nomic-embed-text")
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama").lower()

# Resolved cloud base URL + API key (empty for ollama).
LLM_BASE_URL = _active["base_url"]
LLM_API_KEY = os.environ.get(_active["api_key_env"], "") if _active["api_key_env"] else ""

# Model settings per pass.
# Pass 1 is intentionally tight: cloud providers bill per token and code
# examples are now harvested via regex (no LLM), so Pass 1 only needs a
# short JSON summary per source.
EXTRACT_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_ctx": 4096,
    "num_predict": 500,
    "repeat_penalty": 1.15,
}

SYNTHESIZE_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "num_ctx": 8192,
    "num_predict": 4096,
    "repeat_penalty": 1.1,
}

# === FETCH / RETRY ===
HTTP_TIMEOUT = 15
HTTP_RETRIES = 3
HTTP_BACKOFF = 1.5
MAX_EXTRACT_WORKERS = 5

# Minimum length for a source to be considered usable.
MIN_SOURCE_CHARS = 400
# Source text truncation before Pass 1 (chars, not tokens — approximate).
# Kept tight to stay well under 8B-model free-tier budgets.
MAX_SOURCE_CHARS_PER_PASS1 = 3500

# === SEARCH ===
DEFAULT_MAX_RESULTS = 8
DEFAULT_TOP_SOURCES = 4

# Trusted source hints (used only for soft quality bonus, not gating).
TRUSTED_SOURCES = [
    "github.com", "stackoverflow.com", "docs.python.org", "react.dev",
    "developer.mozilla.org", "realpython.com", "wikipedia.org", "arxiv.org",
    "nature.com", "ncbi.nlm.nih.gov", "nih.gov",
]

# === CACHE ===
SOURCE_CACHE_TTL = 7 * 24 * 3600        # 7 days
LLM_CACHE_TTL = 3 * 24 * 3600           # 3 days
VAULT_INDEX_TTL = 10 * 60               # 10 minutes
