"""Configuration: env vars, pricing table, model registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AzureConfig:
    di_endpoint: str
    di_key: str
    aoai_endpoint: str
    aoai_key: str
    aoai_api_version: str
    dep_gpt5: str
    dep_gpt51: str
    dep_gpt5_mini: str
    dep_gpt54_mini: str
    dep_gpt4o: str
    dep_gpt4o_mini: str
    dep_judge: str


def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val or ""


def load_config() -> AzureConfig:
    return AzureConfig(
        di_endpoint=_env("AZURE_DI_ENDPOINT"),
        di_key=_env("AZURE_DI_KEY"),
        aoai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        aoai_key=_env("AZURE_OPENAI_KEY"),
        aoai_api_version=_env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        dep_gpt5=_env("AOAI_DEPLOYMENT_GPT5", "gpt-5"),
        dep_gpt51=_env("AOAI_DEPLOYMENT_GPT51", "gpt-5.1"),
        dep_gpt5_mini=_env("AOAI_DEPLOYMENT_GPT5_MINI", "gpt-5-mini"),
        dep_gpt54_mini=_env("AOAI_DEPLOYMENT_GPT54_MINI", "gpt-5.4-mini"),
        dep_gpt4o=_env("AOAI_DEPLOYMENT_GPT4O", "gpt-4o"),
        dep_gpt4o_mini=_env("AOAI_DEPLOYMENT_GPT4O_MINI", "gpt-4o-mini"),
        dep_judge=_env("AOAI_DEPLOYMENT_JUDGE", "gpt-5"),
    )


# Pricing per 1M tokens (USD). Update as needed for your contract.
# Source: Azure OpenAI public pricing (approximate, May 2026).
LLM_PRICING = {
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-mini": {"input": 0.25, "output": 2.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# Document Intelligence per-page pricing (USD).
DI_PRICING = {
    "prebuilt-read": 0.0015,
    "prebuilt-layout": 0.010,
    "prebuilt-invoice": 0.050,
}


# --- AAD authentication helpers (used when *_KEY env vars are empty) ---

_COG_SCOPE = "https://cognitiveservices.azure.com/.default"


def get_async_aad_credential():
    """Return an async DefaultAzureCredential (lazy-imported)."""
    from azure.identity.aio import DefaultAzureCredential

    return DefaultAzureCredential()


def get_sync_aad_credential():
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def get_aoai_token_provider():
    """Sync bearer-token provider for AsyncAzureOpenAI (`azure_ad_token_provider`)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    return get_bearer_token_provider(DefaultAzureCredential(), _COG_SCOPE)


def llm_cost(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = LLM_PRICING.get(model_key)
    if not p:
        return 0.0
    return (prompt_tokens / 1_000_000) * p["input"] + (
        completion_tokens / 1_000_000
    ) * p["output"]


def di_cost(model_id: str, pages: int) -> float:
    return DI_PRICING.get(model_id, 0.010) * max(pages, 1)
