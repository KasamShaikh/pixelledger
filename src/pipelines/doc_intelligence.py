"""Azure Document Intelligence pipeline (prebuilt-read / -layout / -invoice)."""

from __future__ import annotations

import time
from typing import Any

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

from ..config import AzureConfig, di_cost, get_async_aad_credential
from .base import DocumentInput, Pipeline, PipelineResult


# Pin to the GA API version for reproducibility (best-practice default).
DEFAULT_DI_API_VERSION = "2024-11-30"


class DocIntelligencePipeline(Pipeline):
    def __init__(
        self,
        cfg: AzureConfig,
        model_id: str = "prebuilt-layout",
        api_version: str = DEFAULT_DI_API_VERSION,
    ):
        self.cfg = cfg
        self.model_id = model_id
        self.api_version = api_version
        self.id = f"di-{model_id}"
        self.display_name = f"Document Intelligence ({model_id})"

    async def run(self, doc: DocumentInput, **kwargs: Any) -> PipelineResult:
        result = PipelineResult(
            pipeline_id=self.id,
            display_name=self.display_name,
            model_id=self.model_id,
            di_api_version=self.api_version,
            di_model=self.model_id,
        )
        if not self.cfg.di_endpoint:
            result.error = "Azure DI endpoint not configured"
            return result

        credential = (
            AzureKeyCredential(self.cfg.di_key)
            if self.cfg.di_key
            else get_async_aad_credential()
        )

        start = time.perf_counter()
        try:
            async with DocumentIntelligenceClient(
                endpoint=self.cfg.di_endpoint,
                credential=credential,
                api_version=self.api_version,
            ) as client:
                poller = await client.begin_analyze_document(
                    self.model_id,
                    AnalyzeDocumentRequest(bytes_source=doc.content),
                    output_content_format="markdown",
                )
                analyze = await poller.result()

            result.raw_text = getattr(analyze, "content", "") or ""
            result.pages = len(getattr(analyze, "pages", []) or [])

            # per-line confidence from pages
            confidences: list[float] = []
            for page in analyze.pages or []:
                for line in getattr(page, "lines", None) or []:
                    # lines don't always carry confidence; fall back to words
                    for word in getattr(line, "words", None) or []:
                        if word.confidence is not None:
                            confidences.append(float(word.confidence))
            if confidences:
                result.per_line_confidence = confidences
                result.confidence_avg = sum(confidences) / len(confidences)

            # structured fields (prebuilt-invoice etc.)
            docs = getattr(analyze, "documents", None) or []
            if docs:
                first = docs[0]
                fields = getattr(first, "fields", None) or {}
                structured: dict[str, Any] = {}
                for name, field in fields.items():
                    structured[name] = {
                        "value": getattr(field, "content", None)
                        or getattr(field, "value_string", None),
                        "confidence": getattr(field, "confidence", None),
                    }
                if structured:
                    result.structured_json = structured

            result.cost_usd = di_cost(self.model_id, result.pages)
            result.raw_response = (
                analyze.as_dict() if hasattr(analyze, "as_dict") else None
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.latency_ms = int((time.perf_counter() - start) * 1000)
            # Close AAD credential if we created one (AzureKeyCredential has no close)
            close = getattr(credential, "close", None)
            if close is not None and not isinstance(credential, AzureKeyCredential):
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass
        return result
