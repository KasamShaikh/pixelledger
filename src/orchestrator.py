"""Run multiple pipelines concurrently, optionally with N repeated runs each."""

from __future__ import annotations

import asyncio
from typing import Any

from .pipelines.base import DocumentInput, Pipeline, PipelineResult


async def run_all(
    pipelines: list[Pipeline],
    doc: DocumentInput,
    per_pipeline_kwargs: dict[str, dict[str, Any]] | None = None,
    repeat_n: int = 1,
) -> list[PipelineResult]:
    """Run each pipeline `repeat_n` times concurrently.

    Returns a flat list of `PipelineResult`; each result has `run_index` set
    (1..repeat_n). Group by `pipeline_id` to compute determinism / variance.
    """
    per_pipeline_kwargs = per_pipeline_kwargs or {}
    repeat_n = max(1, int(repeat_n))

    async def _safe(p: Pipeline, run_index: int) -> PipelineResult:
        try:
            res = await p.run(doc, **per_pipeline_kwargs.get(p.id, {}))
            res.run_index = run_index
            return res
        except Exception as exc:  # noqa: BLE001
            return PipelineResult(
                pipeline_id=p.id,
                display_name=p.display_name,
                error=f"orchestrator: {type(exc).__name__}: {exc}",
                run_index=run_index,
            )

    tasks = [_safe(p, run_index=i + 1) for p in pipelines for i in range(repeat_n)]
    return await asyncio.gather(*tasks)
