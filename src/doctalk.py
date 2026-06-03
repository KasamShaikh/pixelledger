"""DocTalk: strictly grounded chat over extracted text, per pipeline.

Each selected pipeline answers the same question independently, using only that
pipeline's extracted text/JSON. This lets users compare how accurately each
extraction captured the document by asking the same questions of each.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from openai import AsyncAzureOpenAI

from .config import AzureConfig, get_aoai_token_provider, llm_cost


DOCTALK_SYSTEM = """You are DocTalk, an assistant that answers questions about a \
single document using ONLY the extracted text provided below.

How to answer:
- Base every answer ONLY on the EXTRACTED DOCUMENT content. Never use outside \
knowledge.
- Interpret the user's question generously: correct obvious typos and understand \
synonyms or paraphrases. For example "booking amount", "booking charge" and \
"booking fee" refer to the same thing; "total amount" means the total, etc.
- When the document contains the information (even under a slightly different label \
or wording), give the answer and quote the value exactly as it appears.
- Only reply exactly "Not found in the document." when the requested information is \
genuinely absent from the extracted content.
- Keep answers concise.

EXTRACTED DOCUMENT (source: {label}):
\"\"\"
{extracted}
\"\"\"
"""


@dataclass
class DocSource:
    """One pipeline's extracted output to chat against."""

    label: str
    extracted_text: str
    structured_json: Optional[dict[str, Any]] = None


def _is_gpt5(model_key: str) -> bool:
    return model_key.startswith("gpt-5")


def _build_client(cfg: AzureConfig) -> AsyncAzureOpenAI:
    if cfg.aoai_key:
        return AsyncAzureOpenAI(
            azure_endpoint=cfg.aoai_endpoint,
            api_key=cfg.aoai_key,
            api_version=cfg.aoai_api_version,
        )
    return AsyncAzureOpenAI(
        azure_endpoint=cfg.aoai_endpoint,
        azure_ad_token_provider=get_aoai_token_provider(),
        api_version=cfg.aoai_api_version,
    )


def _context_block(source: DocSource) -> str:
    extracted = source.extracted_text or ""
    if source.structured_json:
        import json

        extracted += "\n\nSTRUCTURED FIELDS (JSON):\n" + json.dumps(
            source.structured_json, indent=2, ensure_ascii=False
        )
    return extracted.strip()


async def answer_for_pipeline(
    cfg: AzureConfig,
    source: DocSource,
    history: list[dict[str, str]],
    question: str,
    *,
    model_key: str,
    deployment: str,
) -> tuple[str, float]:
    """Answer a single question grounded in one pipeline's extracted text.

    `history` is a list of {"role": "user"|"assistant", "content": str} for this
    pipeline only. Returns (answer_text, cost_usd).
    """
    if not cfg.aoai_endpoint:
        return ("Azure OpenAI is not configured.", 0.0)

    system = DOCTALK_SYSTEM.format(label=source.label, extracted=_context_block(source))
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    kwargs: dict[str, Any] = {"model": deployment, "messages": messages}
    if _is_gpt5(model_key):
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["temperature"] = 0.0

    client = _build_client(cfg)
    try:
        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        cost = 0.0
        if resp.usage:
            cost = llm_cost(
                model_key, resp.usage.prompt_tokens, resp.usage.completion_tokens
            )
        return (text.strip(), cost)
    except Exception as exc:  # noqa: BLE001
        return (f"Error: {type(exc).__name__}: {exc}", 0.0)
    finally:
        await client.close()


async def answer_all(
    cfg: AzureConfig,
    sources: list[DocSource],
    histories: dict[str, list[dict[str, str]]],
    question: str,
    *,
    model_key: str,
    deployment: str,
) -> dict[str, dict[str, Any]]:
    """Ask the same question of every source concurrently.

    `histories` maps source.label -> that pipeline's prior turns.
    Returns {label: {"text": str, "cost": float}}.
    """
    tasks = [
        answer_for_pipeline(
            cfg,
            source,
            histories.get(source.label, []),
            question,
            model_key=model_key,
            deployment=deployment,
        )
        for source in sources
    ]
    results = await asyncio.gather(*tasks)
    return {
        source.label: {"text": text, "cost": cost}
        for source, (text, cost) in zip(sources, results)
    }
