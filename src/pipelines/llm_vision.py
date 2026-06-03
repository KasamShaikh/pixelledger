"""LLM vision pipeline: send page images directly to an Azure OpenAI chat model."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional

from openai import AsyncAzureOpenAI

from ..config import AzureConfig, get_aoai_token_provider, llm_cost
from .base import DocumentInput, Pipeline, PipelineResult


def _is_gpt5(model_key: str) -> bool:
    return model_key.startswith("gpt-5")


class LLMVisionPipeline(Pipeline):
    def __init__(
        self,
        cfg: AzureConfig,
        deployment: str,
        model_key: str,
        display_name: Optional[str] = None,
    ):
        self.cfg = cfg
        self.deployment = deployment
        self.model_key = model_key
        self.id = f"llm-vision-{model_key}"
        self.display_name = display_name or f"{model_key} (vision)"

    async def run(
        self,
        doc: DocumentInput,
        prompt: str = "",
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        json_schema: Optional[dict[str, Any]] = None,
        top_p: Optional[float] = 1.0,
        seed: Optional[int] = 42,
        **_: Any,
    ) -> PipelineResult:
        result = PipelineResult(
            pipeline_id=self.id,
            display_name=self.display_name,
            model_id=self.deployment,
            api_version=self.cfg.aoai_api_version,
            seed=seed,
            top_p=top_p,
            temperature=temperature if not _is_gpt5(self.model_key) else None,
            reasoning_effort=reasoning_effort if _is_gpt5(self.model_key) else None,
        )
        if not self.cfg.aoai_endpoint:
            result.error = "Azure OpenAI endpoint not configured"
            return result
        if not doc.images:
            result.error = "No page images available; preprocess produced 0 pages."
            return result

        if self.cfg.aoai_key:
            client = AsyncAzureOpenAI(
                azure_endpoint=self.cfg.aoai_endpoint,
                api_key=self.cfg.aoai_key,
                api_version=self.cfg.aoai_api_version,
            )
        else:
            client = AsyncAzureOpenAI(
                azure_endpoint=self.cfg.aoai_endpoint,
                azure_ad_token_provider=get_aoai_token_provider(),
                api_version=self.cfg.aoai_api_version,
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img_bytes in doc.images:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
            )

        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": content}],
        }
        if _is_gpt5(self.model_key):
            # Reasoning models: only reasoning_effort is meaningful.
            # top_p / seed are not supported on gpt-5 reasoning deployments.
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if seed is not None:
                kwargs["seed"] = seed

        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": json_schema,
                    "strict": False,
                },
            }

        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            result.raw_text = text

            if json_schema:
                try:
                    result.structured_json = json.loads(text)
                except json.JSONDecodeError:
                    result.structured_json = None
            else:
                # Best-effort: if model returned JSON anyway, parse it
                stripped = text.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    try:
                        result.structured_json = json.loads(stripped)
                    except json.JSONDecodeError:
                        pass

            usage = response.usage
            if usage:
                result.prompt_tokens = usage.prompt_tokens
                result.completion_tokens = usage.completion_tokens
                result.cost_usd = llm_cost(
                    self.model_key, usage.prompt_tokens, usage.completion_tokens
                )
            result.pages = len(doc.images)
            result.system_fingerprint = getattr(response, "system_fingerprint", None)
            result.raw_response = response.model_dump()
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.latency_ms = int((time.perf_counter() - start) * 1000)
            await client.close()
        return result
