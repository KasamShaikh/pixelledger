"""Hybrid pipeline: Document Intelligence -> LLM for structuring."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from openai import AsyncAzureOpenAI

from ..config import AzureConfig, get_aoai_token_provider, llm_cost
from ..postprocess import (
    build_numeric_legend,
    default_normalize_config,
    stitch_markdown_tables,
)
from .base import DocumentInput, Pipeline, PipelineResult
from .doc_intelligence import DEFAULT_DI_API_VERSION, DocIntelligencePipeline
from .llm_vision import _is_gpt5


class HybridDIPipeline(Pipeline):
    """DI extracts text/markdown -> LLM cleans up & structures."""

    def __init__(
        self,
        cfg: AzureConfig,
        deployment: str,
        model_key: str,
        di_model_id: str = "prebuilt-layout",
        display_name: Optional[str] = None,
        di_api_version: str = DEFAULT_DI_API_VERSION,
    ):
        self.cfg = cfg
        self.deployment = deployment
        self.model_key = model_key
        self.di_model_id = di_model_id
        self.di_api_version = di_api_version
        self.id = f"hybrid-di-{model_key}"
        self.display_name = display_name or f"DI + {model_key} (hybrid)"

    async def run(
        self,
        doc: DocumentInput,
        structuring_prompt: str = "",
        temperature: float = 0.0,
        reasoning_effort: str = "medium",
        json_schema: Optional[dict[str, Any]] = None,
        top_p: Optional[float] = 1.0,
        seed: Optional[int] = 42,
        stitch_tables: bool = True,
        normalize_numbers: bool = True,
        normalize_config: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> PipelineResult:
        result = PipelineResult(
            pipeline_id=self.id,
            display_name=self.display_name,
            model_id=f"{self.di_model_id} + {self.deployment}",
            api_version=self.cfg.aoai_api_version,
            di_api_version=self.di_api_version,
            di_model=self.di_model_id,
            seed=seed,
            top_p=top_p,
            temperature=temperature if not _is_gpt5(self.model_key) else None,
            reasoning_effort=reasoning_effort if _is_gpt5(self.model_key) else None,
        )

        # Step 1: DI
        di = DocIntelligencePipeline(
            self.cfg, self.di_model_id, api_version=self.di_api_version
        )
        di_result = await di.run(doc)
        if di_result.error:
            result.error = f"DI step failed: {di_result.error}"
            result.latency_ms = di_result.latency_ms
            return result

        # Always retain DI-derived metadata so confidence is available
        # even if the downstream LLM step fails.
        result.pages = di_result.pages
        result.confidence_avg = di_result.confidence_avg
        result.per_line_confidence = di_result.per_line_confidence
        result.cost_usd = di_result.cost_usd

        # Step 1b: post-process DI markdown for grounding (non-destructive).
        di_markdown = di_result.raw_text or ""
        legend = ""
        if stitch_tables and di_markdown:
            stitched, merges = stitch_markdown_tables(di_markdown)
            if merges:
                di_markdown = stitched
                result.postprocess_applied.append(f"stitched_tables={merges}")
        if normalize_numbers and di_markdown:
            legend = build_numeric_legend(
                di_markdown, normalize_config or default_normalize_config()
            )
            if legend:
                result.postprocess_applied.append("numeric_legend")

        # Step 2: LLM structuring on DI markdown
        if not self.cfg.aoai_endpoint:
            result.error = "Azure OpenAI endpoint not configured"
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

        user_message_parts = [
            structuring_prompt,
            "\n\n---DOCUMENT_INTELLIGENCE_MARKDOWN---\n",
            di_markdown,
            "\n---END---",
        ]
        if legend:
            user_message_parts.append("\n\n" + legend)
        user_message = "".join(user_message_parts)
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": [{"role": "user", "content": user_message}],
        }
        if _is_gpt5(self.model_key):
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
                result.cost_usd = di_result.cost_usd + llm_cost(
                    self.model_key, usage.prompt_tokens, usage.completion_tokens
                )
            else:
                result.cost_usd = di_result.cost_usd
            result.system_fingerprint = getattr(response, "system_fingerprint", None)
            result.raw_response = {
                "di": di_result.raw_response,
                "llm": response.model_dump(),
                "di_markdown_postprocessed": di_markdown,
                "numeric_legend": legend,
            }
        except Exception as exc:  # noqa: BLE001
            result.error = f"LLM step failed: {type(exc).__name__}: {exc}"
            result.raw_response = {
                "di": di_result.raw_response,
                "llm": None,
                "di_markdown_postprocessed": di_markdown,
                "numeric_legend": legend,
            }
        finally:
            llm_latency = int((time.perf_counter() - start) * 1000)
            result.latency_ms = di_result.latency_ms + llm_latency
            await client.close()
        return result
