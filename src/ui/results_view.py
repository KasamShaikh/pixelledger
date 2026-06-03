"""Results rendering: per-pipeline tabs + Compare + Determinism."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import AsyncAzureOpenAI

from ..config import AzureConfig, get_aoai_token_provider
from ..doctalk import DocSource, answer_all
from ..metrics.schema_metrics import field_scores, section_scores
from ..metrics.text_metrics import compute_cer_wer, html_diff
from ..pipelines.base import PipelineResult

# ---------- Helpers ----------


def _group_by_pipeline(
    results: list[PipelineResult],
) -> dict[str, list[PipelineResult]]:
    """Preserve pipeline order from first occurrence."""
    out: dict[str, list[PipelineResult]] = {}
    for r in results:
        out.setdefault(r.pipeline_id, []).append(r)
    for k in out:
        out[k].sort(key=lambda x: x.run_index)
    return out


def _kpi_row(r: PipelineResult) -> None:
    c1, c2 = st.columns(2)
    c1.metric(
        "Avg confidence",
        f"{r.confidence_avg:.2f}" if r.confidence_avg is not None else "—",
    )
    c2.metric("Tokens (in/out)", f"{r.prompt_tokens}/{r.completion_tokens}")


def _audit_table(r: PipelineResult) -> pd.DataFrame:
    rows = [
        ("Model", r.model_id),
        ("DI model", r.di_model or "—"),
        ("AOAI api-version", r.api_version or "—"),
        ("DI api-version", r.di_api_version or "—"),
        ("Temperature", r.temperature if r.temperature is not None else "—"),
        ("top_p", r.top_p if r.top_p is not None else "—"),
        ("Seed", r.seed if r.seed is not None else "—"),
        ("Reasoning effort", r.reasoning_effort or "—"),
        ("system_fingerprint", r.system_fingerprint or "—"),
        ("Post-processing", ", ".join(r.postprocess_applied) or "none"),
        ("Tokens in / out", f"{r.prompt_tokens} / {r.completion_tokens}"),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _confidence_unavailable_reason(r: PipelineResult) -> str:
    if r.error:
        return "Confidence is unavailable because this run ended with an error."
    if r.pipeline_id.startswith("llm-vision"):
        return "This pipeline does not emit per-word confidence scores."
    if r.di_model:
        return (
            "Document Intelligence did not return word-level confidence values for "
            "this document or model output."
        )
    return "Confidence scores are unavailable for this pipeline."


def _render_single_run(r: PipelineResult) -> None:
    if r.error:
        st.error(r.error)
        with st.expander("Audit trail"):
            st.dataframe(_audit_table(r), use_container_width=True, hide_index=True)
        return
    _kpi_row(r)
    sub = st.tabs(["Raw text", "Structured JSON", "Confidence", "Audit trail"])
    with sub[0]:
        st.markdown(r.raw_text or "_(empty)_")
        with st.expander("View as plain text"):
            st.text(r.raw_text)
    with sub[1]:
        if r.structured_json:
            st.json(r.structured_json)
        else:
            st.info("No structured JSON returned (enable JSON schema mode in sidebar).")
    with sub[2]:
        if r.per_line_confidence:
            df = pd.DataFrame({"confidence": r.per_line_confidence})
            df["idx"] = df.index
            fig = px.bar(
                df, x="idx", y="confidence", title="Per-word/line confidence (DI)"
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"confidence-{r.pipeline_id}-{r.run_index}",
            )
        else:
            st.info(_confidence_unavailable_reason(r))
    with sub[3]:
        st.dataframe(_audit_table(r), use_container_width=True, hide_index=True)


def _render_pipeline_tab(runs: list[PipelineResult]) -> None:
    if len(runs) == 1:
        _render_single_run(runs[0])
        return
    # Multiple runs: per-run sub-tabs + aggregate KPIs
    c1, c2 = st.columns(2)
    c1.metric("Runs", f"{len(runs)} ({sum(1 for r in runs if r.error)} errors)")
    c2.metric(
        "Best confidence",
        f"{max((r.confidence_avg or 0) for r in runs if not r.error):.2f}"
        if any(not r.error for r in runs)
        else "—",
    )
    run_tabs = st.tabs([f"Run {r.run_index}" for r in runs])
    for tab, r in zip(run_tabs, runs):
        with tab:
            _render_single_run(r)


# ---------- Executive summary ----------


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return (value - low) / (high - low)


def _fmt_cost(value: float) -> str:
    return f"${value:.4f}" if value < 1 else f"${value:.2f}"


def _fmt_seconds(ms: float) -> str:
    seconds = ms / 1000
    return f"{seconds:.1f}s" if seconds >= 1 else f"{ms:.0f}ms"


def _summary_rows(
    grouped: dict[str, list[PipelineResult]],
    gt_text: str | None,
    gt_json: dict | None,
    judge_scores: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, runs in grouped.items():
        r = _aggregate_for_pipeline(runs)
        successful = [x for x in runs if not x.error]
        latencies = [x.latency_ms for x in successful] or [r.latency_ms]
        costs = [x.cost_usd for x in successful] or [r.cost_usd]
        text_metrics = (
            compute_cer_wer(gt_text or "", r.raw_text)
            if gt_text and not r.error
            else {}
        )
        f1_scores = (
            field_scores(r.structured_json, gt_json)
            if gt_json and not r.error
            else None
        )
        judge = (judge_scores or {}).get(r.pipeline_id, {}) if judge_scores else {}
        judge_values = [
            _number(judge.get(k)) for k in ("accuracy", "completeness", "structure")
        ]
        numeric_judges = [v for v in judge_values if v is not None]
        rows.append(
            {
                "pipeline_id": pid,
                "pipeline": r.display_name,
                "runs": len(runs),
                "avg_latency_ms": sum(latencies) / len(latencies),
                "total_cost_usd": sum(costs),
                "cer": text_metrics.get("cer"),
                "wer": text_metrics.get("wer"),
                "field_f1": f1_scores["f1"] if f1_scores else None,
                "judge_accuracy": _number(judge.get("accuracy")),
                "judge_completeness": _number(judge.get("completeness")),
                "judge_structure": _number(judge.get("structure")),
                "judge_avg": sum(numeric_judges) / len(numeric_judges)
                if numeric_judges
                else None,
                "judge_rationale": str(judge.get("rationale") or ""),
                "error": r.error or "",
            }
        )
    return rows


def _rank_summary(
    rows: list[dict[str, Any]], gt_text: str | None
) -> dict[str, Any] | None:
    valid = [row for row in rows if not row["error"]]
    if not valid:
        return None

    rows_with_cer = [row for row in valid if row["cer"] is not None]
    rows_with_judge = [row for row in valid if row["judge_avg"] is not None]
    if rows_with_cer:
        best_accuracy = min(rows_with_cer, key=lambda row: row["cer"])
    elif rows_with_judge:
        best_accuracy = max(
            rows_with_judge,
            key=lambda row: row["judge_accuracy"] or row["judge_avg"] or 0,
        )
    else:
        best_accuracy = valid[0]

    if gt_text and rows_with_cer:
        recommended = best_accuracy
        reason = "lowest character error rate against the uploaded ground truth"
    elif rows_with_judge:

        def composite(row: dict[str, Any]) -> float:
            accuracy = row["judge_accuracy"] or row["judge_avg"] or 0
            completeness = row["judge_completeness"] or row["judge_avg"] or 0
            structure = row["judge_structure"] or row["judge_avg"] or 0
            return 0.5 * accuracy + 0.25 * completeness + 0.25 * structure

        recommended = max(rows_with_judge, key=composite)
        reason = (
            "strongest judge-weighted balance of accuracy, completeness, and structure"
        )
    else:
        recommended = best_accuracy
        reason = "best available accuracy signal because no ground truth or judge scores are available"

    return {
        "recommended": recommended,
        "best_accuracy": best_accuracy,
        "reason": reason,
        "valid": valid,
    }


def _summary_bullets(summary: dict[str, Any], gt_text: str | None) -> list[str]:
    rec = summary["recommended"]
    best_accuracy = summary["best_accuracy"]
    bullets = []
    if gt_text and rec["cer"] is not None:
        bullets.append(
            f"Choose {rec['pipeline']} when final text accuracy matters most; "
            f"it has the lowest CER at {rec['cer']:.3f}."
        )
    elif rec["judge_avg"] is not None:
        bullets.append(
            f"Choose {rec['pipeline']} for the best quality signal; "
            f"its judge average is {rec['judge_avg']:.1f}/5."
        )
    else:
        bullets.append(
            f"Choose {rec['pipeline']} as the safest successful option from this run."
        )
    if best_accuracy["pipeline"] != rec["pipeline"]:
        bullets.append(
            f"For accuracy-only decisions, compare closely with {best_accuracy['pipeline']} before choosing."
        )
    return bullets


def _model_family_from_label(name: str) -> str:
    n = name.lower()
    if "gpt-5" in n:
        return "gpt5"
    if "gpt-4" in n or "gpt4" in n:
        return "gpt4"
    return "other"


def _allow_cost_guidance(rows: list[dict[str, Any]]) -> bool:
    valid = [row for row in rows if not row.get("error")]
    families = {_model_family_from_label(row["pipeline"]) for row in valid}
    return len(valid) >= 2 and families and families <= {"gpt5"}


def _single_model_assessment(
    row: dict[str, Any], gt_text: str | None
) -> tuple[list[str], list[str]]:
    went_well: list[str] = []
    improve: list[str] = []

    if gt_text and row["cer"] is not None:
        went_well.append(
            f"OCR accuracy measured against ground truth with CER {row['cer']:.3f}."
        )
        if row["cer"] > 0.10:
            improve.append(
                "Character-level errors are still noticeable; tighten extraction prompt/post-processing."
            )
    elif row["judge_avg"] is not None:
        went_well.append(
            f"Judge quality signal is {row['judge_avg']:.1f}/5 across available dimensions."
        )
        if row["judge_accuracy"] is not None and row["judge_accuracy"] < 4:
            improve.append("Improve factual extraction accuracy for critical fields.")
        if row["judge_structure"] is not None and row["judge_structure"] < 4:
            improve.append(
                "Improve JSON structure consistency for downstream consumers."
            )
    else:
        went_well.append(
            "The run completed successfully and produced extractable content."
        )
        improve.append(
            "Add ground truth or judge scoring to get measurable quality signals."
        )

    if row["wer"] is not None and row["wer"] > 0.15:
        improve.append(
            "Word-level error rate suggests cleanup is needed on noisy sections."
        )
    if not improve:
        improve.append(
            "Run with additional benchmark samples to validate stability before production use."
        )

    return went_well, improve


def _narrative_payload(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    pipelines = []
    for row in rows:
        item = {
            "pipeline": row["pipeline"],
            "cer": row["cer"],
            "judge_accuracy": row["judge_accuracy"],
            "judge_completeness": row["judge_completeness"],
            "judge_structure": row["judge_structure"],
            "error": row["error"],
        }
        pipelines.append(item)
    return {
        "mode": mode,
        "allow_cost_guidance": _allow_cost_guidance(rows),
        "recommended": summary["recommended"]["pipeline"],
        "reason": summary["reason"],
        "pipelines": pipelines,
    }


async def _generate_business_narrative(
    cfg: AzureConfig,
    payload: dict[str, Any],
    *,
    deployment: str,
) -> str:
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
    try:
        mode = str(payload.get("mode") or "multi")
        if mode == "single":
            system_prompt = (
                "Explain OCR results for a business user in 3 to 4 short sentences. "
                "Do not recommend between models. Describe only: what went well and what can be improved. "
                "Use only the provided metrics and do not invent figures."
            )
        else:
            cost_rule = (
                "Cost guidance is allowed only when allow_cost_guidance is true in the JSON. "
                "If it is false, do not include any cost recommendation. "
            )
            system_prompt = (
                "Explain OCR pipeline comparison results for a business user. "
                "Write 3 to 4 short sentences. Recommend only successful pipelines. "
                "Use only the metrics in the provided JSON and do not invent figures. "
                "Focus on accuracy-related outcomes. " + cost_rule
            )
        resp = await client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            reasoning_effort="minimal",
        )
        return (resp.choices[0].message.content or "").strip()
    finally:
        await client.close()


def _cached_business_narrative(cfg: AzureConfig, payload: dict[str, Any]) -> str | None:
    if not cfg.aoai_endpoint:
        return None
    deployment = cfg.dep_gpt5_mini
    cache_key = hashlib.sha256(
        json.dumps(
            {"deployment": deployment, "payload": payload}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    cache = st.session_state.setdefault("_business_summary_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    try:
        text = asyncio.run(
            _generate_business_narrative(
                cfg,
                payload,
                deployment=deployment,
            )
        )
    except Exception as exc:  # noqa: BLE001
        cache[cache_key] = f"AI narrative unavailable: {type(exc).__name__}: {exc}"
        return cache[cache_key]
    cache[cache_key] = text
    return text


def _render_executive_summary(
    grouped: dict[str, list[PipelineResult]],
    ground_truth_text: str | None,
    ground_truth_json: dict | None,
    judge_scores: dict[str, dict[str, Any]] | None,
    *,
    cfg: AzureConfig | None = None,
    show_ai_summary: bool = True,
) -> None:
    rows = _summary_rows(grouped, ground_truth_text, ground_truth_json, judge_scores)
    summary = _rank_summary(rows, ground_truth_text)
    if not summary:
        return

    single_model_mode = len(summary["valid"]) == 1
    rec = summary["recommended"]
    best_accuracy = summary["best_accuracy"]

    if single_model_mode:
        row = summary["valid"][0]
        went_well, improve = _single_model_assessment(row, ground_truth_text)
        went_well_html = "".join(f"<li>{html.escape(item)}</li>" for item in went_well)
        improve_html = "".join(f"<li>{html.escape(item)}</li>" for item in improve)
        st.markdown(
            f"""
<div class="executive-summary-card">
  <div class="summary-kicker">Business decision aid</div>
  <div class="summary-title">📌 Run assessment</div>
  <div class="summary-verdict">{html.escape(row["pipeline"])} finished successfully. This assessment focuses on current strengths and improvement opportunities.</div>
  <div class="summary-sections">
    <div class="summary-section">
      <div class="summary-section-title">What went well</div>
      <ul>{went_well_html}</ul>
    </div>
    <div class="summary-section">
      <div class="summary-section-title">What can be improved</div>
      <ul>{improve_html}</ul>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        bullets = _summary_bullets(summary, ground_truth_text)
        accuracy_label = (
            f"Best accuracy: {best_accuracy['pipeline']} (CER {best_accuracy['cer']:.3f})"
            if best_accuracy["cer"] is not None
            else f"Best quality: {best_accuracy['pipeline']} ({(best_accuracy['judge_avg'] or 0):.1f}/5)"
        )
        bullet_html = "".join(f"<li>{html.escape(item)}</li>" for item in bullets)
        st.markdown(
            f"""
<div class="executive-summary-card">
  <div class="summary-kicker">Business decision aid</div>
  <div class="summary-title">🧭 Executive summary</div>
  <div class="summary-verdict"><strong>Recommended:</strong> {html.escape(rec["pipeline"])}. This is the best choice for this run because it has the {html.escape(summary["reason"])}.</div>
  <div class="summary-chips">
    <span class="summary-chip primary">Recommended: {html.escape(rec["pipeline"])}</span>
    <span class="summary-chip">{html.escape(accuracy_label)}</span>
  </div>
  <ul>{bullet_html}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ Disclaimer: This is an initial starting point to help you decide — not a "
        "final recommendation. Real model selection depends on many additional factors "
        "(document variety, volume, latency, cost ceilings, compliance, and ground-truth "
        "validation at scale). These results come purely from a personal demo desk and "
        "should be validated on your own data before any production decision."
    )

    if show_ai_summary and cfg is not None:
        payload = _narrative_payload(
            rows,
            summary,
            mode="single" if single_model_mode else "multi",
        )
        with st.spinner("Writing business summary with GPT-5 mini…"):
            narrative = _cached_business_narrative(cfg, payload)
        if narrative and not narrative.startswith("AI narrative unavailable:"):
            st.info(narrative)
        elif narrative:
            st.caption(narrative)

    if not single_model_mode and rec["judge_rationale"]:
        with st.expander("Why this recommendation?"):
            st.write(rec["judge_rationale"][:400])


# ---------- Determinism ----------


def _pairwise_cer_matrix(runs: list[PipelineResult]) -> pd.DataFrame:
    n = len(runs)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0.0
                continue
            a = runs[i].raw_text or ""
            b = runs[j].raw_text or ""
            mat[i][j] = compute_cer_wer(a, b).get("cer") or 0.0
    return pd.DataFrame(
        mat,
        index=[f"Run {r.run_index}" for r in runs],
        columns=[f"Run {r.run_index}" for r in runs],
    )


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    elif obj is None:
        return out
    else:
        out[prefix] = str(obj).strip().lower()
    return out


def _field_disagreement_count(runs: list[PipelineResult]) -> pd.DataFrame:
    flattened = [_flatten(r.structured_json or {}) for r in runs]
    all_keys = set().union(*flattened) if flattened else set()
    rows = []
    for k in sorted(all_keys):
        values = {f.get(k, "") for f in flattened}
        if len(values) > 1:
            rows.append({"field": k, "distinct_values": len(values)})
    return pd.DataFrame(rows)


def _render_determinism_tab(grouped: dict[str, list[PipelineResult]]) -> None:
    multi = {pid: runs for pid, runs in grouped.items() if len(runs) > 1}
    if not multi:
        st.info(
            "Determinism analysis activates when **Repeated runs ≥ 2** in the sidebar. "
            "Re-run with 2–5 runs to see drift across runs."
        )
        return
    for pid, runs in multi.items():
        st.markdown(f"### {runs[0].display_name}")
        # Audit table — fingerprints / seeds across runs
        audit = pd.DataFrame(
            {
                "run": [r.run_index for r in runs],
                "latency_ms": [r.latency_ms for r in runs],
                "system_fingerprint": [r.system_fingerprint or "—" for r in runs],
                "seed": [r.seed if r.seed is not None else "—" for r in runs],
                "top_p": [r.top_p if r.top_p is not None else "—" for r in runs],
                "tokens_out": [r.completion_tokens for r in runs],
                "raw_text_len": [len(r.raw_text or "") for r in runs],
            }
        )
        st.dataframe(audit, use_container_width=True, hide_index=True)

        # Pairwise CER heatmap
        mat = _pairwise_cer_matrix(runs)
        fig = px.imshow(
            mat,
            text_auto=".3f",
            color_continuous_scale="RdYlGn_r",
            zmin=0,
            zmax=max(0.2, mat.values.max()),
            title="Pairwise CER between runs (lower = more deterministic)",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"determinism-pairwise-cer-{pid}",
        )

        # Field-level disagreement (only if structured JSON exists)
        if any(r.structured_json for r in runs):
            disagree = _field_disagreement_count(runs)
            if disagree.empty:
                st.success("✅ All structured fields identical across runs.")
            else:
                st.warning(f"{len(disagree)} field(s) differ across runs.")
                st.dataframe(disagree, use_container_width=True, hide_index=True)
        st.divider()


# ---------- Top-level render ----------


def render_results(
    results: list[PipelineResult],
    ground_truth_text: str | None,
    ground_truth_json: dict | None,
    judge_scores: dict[str, dict[str, Any]] | None,
    *,
    cfg: AzureConfig | None = None,
    show_ai_summary: bool = True,
) -> None:
    grouped = _group_by_pipeline(results)
    _render_executive_summary(
        grouped,
        ground_truth_text,
        ground_truth_json,
        judge_scores,
        cfg=cfg,
        show_ai_summary=show_ai_summary,
    )
    pipeline_ids = list(grouped.keys())
    tab_labels = [grouped[pid][0].display_name for pid in pipeline_ids] + [
        "📊 Compare",
        "🔁 Determinism",
        "💬 DocTalk",
    ]
    tabs = st.tabs(tab_labels)
    for tab, pid in zip(tabs[: len(pipeline_ids)], pipeline_ids):
        with tab:
            _render_pipeline_tab(grouped[pid])

    with tabs[-3]:
        _render_compare(
            grouped,
            ground_truth_text,
            ground_truth_json,
            judge_scores,
        )
    with tabs[-2]:
        _render_determinism_tab(grouped)
    with tabs[-1]:
        _render_doctalk(grouped, cfg)


def _render_doctalk(
    grouped: dict[str, list[PipelineResult]],
    cfg: AzureConfig | None,
) -> None:
    st.subheader("💬 DocTalk")
    st.caption(
        "Chat with the extracted text. Each selected pipeline answers from its own "
        "extraction only, so you can compare accuracy. Answers are strictly grounded "
        "— if something isn't in the extracted text, the answer is "
        '"Not found in the document."'
    )

    if cfg is None or not cfg.aoai_endpoint:
        st.info("Azure OpenAI is not configured, so DocTalk is unavailable.")
        return

    # Build one source per pipeline from its representative run with text.
    sources: list[DocSource] = []
    for runs in grouped.values():
        rep = _aggregate_for_pipeline(runs)
        text = (rep.raw_text or "").strip()
        if not text and not rep.structured_json:
            continue
        sources.append(
            DocSource(
                label=rep.display_name,
                extracted_text=rep.raw_text or "",
                structured_json=rep.structured_json,
            )
        )

    if not sources:
        st.info("No extracted text is available yet. Run an extraction first.")
        return

    all_labels = [s.label for s in sources]
    selected_labels = st.multiselect(
        "Pipelines to ask",
        options=all_labels,
        default=all_labels,
        key="doctalk_selected",
        help="Each selected pipeline answers the same question from its own extraction.",
    )
    active_sources = [s for s in sources if s.label in selected_labels]

    history: list[dict[str, Any]] = st.session_state.get("doctalk_history", [])

    # Render prior turns.
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            answers = turn.get("answers", {})
            labels = list(answers.keys())
            if labels:
                cols = st.columns(len(labels))
                for col, lbl in zip(cols, labels):
                    with col:
                        st.markdown(f"**{lbl}**")
                        st.markdown(answers[lbl].get("text", ""))
                        cost = answers[lbl].get("cost", 0.0)
                        if cost:
                            st.caption(f"~${cost:.4f}")

    question = st.chat_input("Ask a question about the document…")
    if question:
        if not active_sources:
            st.warning("Select at least one pipeline to ask.")
            return

        # Per-pipeline prior turns for context.
        histories: dict[str, list[dict[str, str]]] = {
            s.label: [] for s in active_sources
        }
        for turn in history:
            for lbl, ans in turn.get("answers", {}).items():
                if lbl in histories:
                    histories[lbl].append({"role": "user", "content": turn["question"]})
                    histories[lbl].append(
                        {"role": "assistant", "content": ans.get("text", "")}
                    )

        with st.spinner("Asking each pipeline…"):
            answers = asyncio.run(
                answer_all(
                    cfg,
                    active_sources,
                    histories,
                    question,
                    model_key="gpt-4o-mini",
                    deployment=cfg.dep_gpt4o_mini,
                )
            )

        history.append({"question": question, "answers": answers})
        st.session_state["doctalk_history"] = history
        st.rerun()

    if history:
        if st.button("Clear chat", key="doctalk_clear"):
            st.session_state["doctalk_history"] = []
            st.rerun()


def _aggregate_for_pipeline(runs: list[PipelineResult]) -> PipelineResult:
    """Pick the first successful run as the representative; if all error, use first."""
    for r in runs:
        if not r.error:
            return r
    return runs[0]


def _render_compare(
    grouped: dict[str, list[PipelineResult]],
    gt_text: str | None,
    gt_json: dict | None,
    judge_scores: dict[str, dict[str, Any]] | None,
) -> None:
    rep = {pid: _aggregate_for_pipeline(runs) for pid, runs in grouped.items()}
    rows = []
    for pid, r in rep.items():
        runs = grouped[pid]
        text_metrics = (
            compute_cer_wer(gt_text or "", r.raw_text)
            if gt_text
            else {"cer": None, "wer": None}
        )
        f1_scores = field_scores(r.structured_json, gt_json) if gt_json else None
        judge = (judge_scores or {}).get(r.pipeline_id, {}) if judge_scores else {}
        rows.append(
            {
                "pipeline": r.display_name,
                "cer": text_metrics["cer"],
                "wer": text_metrics["wer"],
                "field_f1": f1_scores["f1"] if f1_scores else None,
                "judge_accuracy": judge.get("accuracy"),
                "judge_completeness": judge.get("completeness"),
                "judge_structure": judge.get("structure"),
                "error": r.error or "",
            }
        )

    df = pd.DataFrame(rows)
    st.subheader("Accuracy summary")
    st.dataframe(df, use_container_width=True, hide_index=True)

    valid = [r for r in rows if not r["error"]]
    if valid and any(r["cer"] is not None for r in valid):
        winner = min((r for r in valid if r["cer"] is not None), key=lambda r: r["cer"])
        st.success(
            f"🏆 Lowest CER: **{winner['pipeline']}** at CER={winner['cer']:.3f}"
        )

    if df["cer"].notna().any():
        fig = px.bar(
            df.dropna(subset=["cer"]),
            x="pipeline",
            y=["cer", "wer"],
            barmode="group",
            title="CER / WER (lower is better)",
        )
        st.plotly_chart(fig, use_container_width=True, key="compare-cer-wer")
    else:
        st.caption("Upload ground truth to see CER/WER.")

    if df["field_f1"].notna().any():
        fig = px.bar(
            df.dropna(subset=["field_f1"]),
            x="pipeline",
            y="field_f1",
            title="Field-level F1 (higher is better)",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True, key="compare-field-f1")

    # Section-level F1 (only meaningful if gt_json has top-level sections)
    if gt_json and isinstance(gt_json, dict):
        section_rows = []
        for pid, r in rep.items():
            sec = section_scores(r.structured_json, gt_json)
            for section, scores in sec.items():
                section_rows.append(
                    {
                        "pipeline": r.display_name,
                        "section": section,
                        "f1": scores["f1"],
                    }
                )
        if section_rows:
            sdf = pd.DataFrame(section_rows)
            fig = px.bar(
                sdf,
                x="section",
                y="f1",
                color="pipeline",
                barmode="group",
                title="Section-level F1 (higher is better)",
                range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True, key="compare-section-f1")

    if judge_scores:
        judge_rows = []
        for pid, r in rep.items():
            j = judge_scores.get(r.pipeline_id, {})
            for k in ("accuracy", "completeness", "structure"):
                if isinstance(j.get(k), (int, float)):
                    judge_rows.append(
                        {"pipeline": r.display_name, "dimension": k, "score": j[k]}
                    )
        if judge_rows:
            jdf = pd.DataFrame(judge_rows)
            fig = px.line_polar(
                jdf,
                r="score",
                theta="dimension",
                color="pipeline",
                line_close=True,
                range_r=[0, 5],
                title="LLM-as-judge",
            )
            st.plotly_chart(fig, use_container_width=True, key="compare-judge-radar")

    if gt_text:
        st.subheader("Side-by-side diff vs ground truth")
        for pid, r in rep.items():
            if r.error:
                continue
            with st.expander(r.display_name):
                st.components.v1.html(
                    html_diff(
                        gt_text,
                        r.raw_text,
                        ref_label="Ground truth",
                        hyp_label=r.display_name,
                    ),
                    height=400,
                    scrolling=True,
                )

    # Download bundle
    bundle = {
        "summary": rows,
        "ground_truth_text": gt_text,
        "ground_truth_json": gt_json,
        "judge_scores": judge_scores,
        "pipeline_outputs": [
            {
                "pipeline_id": r.pipeline_id,
                "display_name": r.display_name,
                "run_index": r.run_index,
                "raw_text": r.raw_text,
                "structured_json": r.structured_json,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
                "tokens_in": r.prompt_tokens,
                "tokens_out": r.completion_tokens,
                "system_fingerprint": r.system_fingerprint,
                "seed": r.seed,
                "top_p": r.top_p,
                "temperature": r.temperature,
                "reasoning_effort": r.reasoning_effort,
                "api_version": r.api_version,
                "di_api_version": r.di_api_version,
                "di_model": r.di_model,
                "postprocess_applied": r.postprocess_applied,
                "error": r.error,
            }
            for runs in grouped.values()
            for r in runs
        ],
    }
    st.download_button(
        "Download results (JSON)",
        data=json.dumps(bundle, indent=2),
        file_name="ocr_comparison_results.json",
        mime="application/json",
    )
