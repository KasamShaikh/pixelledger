"""Common pipeline contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentInput:
    """Document passed to a pipeline."""

    filename: str
    content: bytes  # original file bytes (pdf/png/jpg/tiff)
    mime_type: str
    images: list[bytes] = field(
        default_factory=list
    )  # per-page PNG bytes (after preprocess)


@dataclass
class PipelineResult:
    pipeline_id: str
    display_name: str
    raw_text: str = ""
    structured_json: Optional[dict[str, Any]] = None
    confidence_avg: Optional[float] = None
    per_line_confidence: list[float] = field(default_factory=list)
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model_id: str = ""
    pages: int = 0
    error: Optional[str] = None
    raw_response: Any = None
    # Determinism / audit trail
    system_fingerprint: Optional[str] = None
    seed: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    api_version: Optional[str] = None
    di_api_version: Optional[str] = None
    di_model: Optional[str] = None
    postprocess_applied: list[str] = field(default_factory=list)
    run_index: int = 1


class Pipeline(ABC):
    id: str
    display_name: str

    @abstractmethod
    async def run(self, doc: DocumentInput, **kwargs: Any) -> PipelineResult: ...
