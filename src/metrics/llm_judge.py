"""LLM-as-judge scoring of pipeline outputs."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncAzureOpenAI

from ..config import AzureConfig, get_aoai_token_provider


JUDGE_SYSTEM = """You are an expert evaluator of OCR/document-extraction outputs.
Score the extraction on a 1-5 integer scale for each of the following dimensions:
- accuracy: how faithfully it captures the source content
- completeness: how much of the relevant content is captured
- structure: how well key/value, tables, and layout are preserved
Return STRICT JSON of shape:
{"accuracy": int, "completeness": int, "structure": int, "rationale": "..."}
"""


async def judge(
    cfg: AzureConfig,
    extracted_text: str,
    reference_text: str | None,
    *,
    deployment: str | None = None,
) -> dict[str, Any]:
    if not cfg.aoai_endpoint:
        return {"error": "Azure OpenAI not configured"}

    if cfg.aoai_key:
        client = AsyncAzureOpenAI(
            azure_endpoint=cfg.aoai_endpoint,
            api_key=cfg.aoai_key,
            api_version=cfg.aoai_api_version,
        )
    else:
        client = AsyncAzureOpenAI(
            azure_endpoint=cfg.aoai_endpoint,
            azure_ad_token_provider=get_aoai_token_provider(),
            api_version=cfg.aoai_api_version,
        )
    user = (
        f"EXTRACTED OUTPUT:\n{extracted_text}\n\n"
        + (f"REFERENCE GROUND TRUTH:\n{reference_text}\n\n" if reference_text else "")
        + "Score it."
    )
    try:
        resp = await client.chat.completions.create(
            model=deployment or cfg.dep_judge,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        await client.close()
