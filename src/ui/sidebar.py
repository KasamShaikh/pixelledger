"""Streamlit sidebar - all configuration knobs.

Layout (top → bottom):
1. Document preset
2. Model lineup
3. Document Intelligence
4. LLM settings + determinism (top_p / seed / repeated runs)
5. Post-processing (table stitching, numeric normalisation)
6. Output mode + optional JSON schema
7. Prompts (grounded defaults)
8. Preprocessing
9. Evaluation
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ..presets import list_presets, resolve_preset

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _available_prompt_files() -> list[str]:
    return sorted(p.name for p in PROMPTS_DIR.glob("*.md"))


def _apply_preset_to_state(preset: dict) -> None:
    """Push preset defaults into st.session_state so widgets adopt them."""
    d = preset.get("defaults", {})
    keys = {
        "di_model": d.get("di_model", "prebuilt-layout"),
        "output_mode": d.get("output_mode", "Strict JSON schema"),
        "stitch_tables": d.get("stitch_tables", True),
        "normalize_numbers": d.get("normalize_numbers", True),
        "top_p": float(d.get("top_p", 1.0)),
        "seed": int(d.get("seed", 42)),
        "temperature": float(d.get("temperature", 0.0)),
        "reasoning_effort": d.get("reasoning_effort", "medium"),
        "repeat_runs": int(d.get("repeat_runs", 1)),
        "extraction_prompt": preset.get("extraction_prompt") or "",
        "structuring_prompt": preset.get("structuring_prompt") or "",
        "json_schema_text": (
            json.dumps(preset["json_schema"], indent=2)
            if preset.get("json_schema")
            else ""
        ),
    }
    for k, v in keys.items():
        st.session_state[k] = v


def render_sidebar() -> dict:
    presets = list_presets()
    preset_map = {p["name"]: p for p in presets}

    with st.sidebar:
        st.header("Configuration")

        # ---------- 1. PRESET ----------
        st.subheader("1. Document preset")
        names = list(preset_map.keys()) or ["Free-form (no preset)"]
        default_name = next((n for n in names if "credit memo" in n.lower()), names[0])
        current = st.session_state.get("preset_name", default_name)
        if current not in names:
            current = default_name
        chosen = st.selectbox(
            "Scenario",
            names,
            index=names.index(current),
            key="preset_name",
        )
        if chosen and chosen != st.session_state.get("_applied_preset"):
            if chosen in preset_map:
                _apply_preset_to_state(resolve_preset(preset_map[chosen]))
            st.session_state["_applied_preset"] = chosen
        st.caption(preset_map.get(chosen, {}).get("description", ""))

        # ---------- 2. MODEL LINEUP ----------
        st.subheader("2. Model lineup")
        st.caption("Choose the document AI models to compare.")
        with st.expander("Available models", expanded=True):
            enable_hybrid_gpt54_mini = st.checkbox(
                "DI + GPT-5.4 Mini", value=True, key="pipe_h54m"
            )
            enable_hybrid_gpt51 = st.checkbox(
                "DI + GPT-5.1", value=True, key="pipe_h51"
            )
            enable_hybrid_gpt4o_mini = st.checkbox(
                "DI + GPT-4.0 Mini", value=False, key="pipe_h4om"
            )
            enable_gpt5_vision = st.checkbox("GPT-5 vision", value=False, key="pipe_v5")
            enable_di = st.checkbox("DI only", value=False, key="pipe_di")

        # ---------- 3. DOCUMENT INTELLIGENCE ----------
        st.subheader("3. Document Intelligence")
        di_models = ["prebuilt-layout", "prebuilt-read", "prebuilt-invoice"]
        if st.session_state.get("di_model") not in di_models:
            st.session_state["di_model"] = "prebuilt-layout"
        di_model = st.selectbox("DI model", di_models, key="di_model")

        # ---------- 4. LLM + DETERMINISM ----------
        st.subheader("4. LLM settings")
        st.session_state.setdefault("temperature", 0.0)
        temperature = st.slider(
            "Temperature (GPT-4o family)", 0.0, 1.0, step=0.1, key="temperature"
        )
        efforts = ["minimal", "low", "medium", "high"]
        if st.session_state.get("reasoning_effort") not in efforts:
            st.session_state["reasoning_effort"] = "medium"
        reasoning_effort = st.selectbox(
            "Reasoning effort (GPT-5 / 5.1)", efforts, key="reasoning_effort"
        )

        st.markdown("**Determinism** — best-practice defaults")
        st.session_state.setdefault("top_p", 1.0)
        top_p = st.slider(
            "top_p (GPT-4o family)",
            0.0,
            1.0,
            step=0.05,
            key="top_p",
            help="1.0 = no nucleus truncation; reduces sampling variance run-to-run.",
        )
        st.session_state.setdefault("seed", 42)
        seed = st.number_input(
            "Seed (GPT-4o family)",
            min_value=0,
            max_value=2_147_483_647,
            step=1,
            key="seed",
            help="Fixed seed → reproducible output for the same input.",
        )
        st.session_state.setdefault("repeat_runs", 1)
        repeat_runs = st.number_input(
            "Repeated runs per pipeline (variance check)",
            min_value=1,
            max_value=5,
            step=1,
            key="repeat_runs",
            help="Set ≥2 to see run-to-run drift in the Determinism tab.",
        )

        # ---------- 5. POST-PROCESSING ----------
        st.subheader("5. Post-processing")
        st.session_state.setdefault("stitch_tables", True)
        stitch_tables = st.checkbox(
            "Stitch tables across pages (DI markdown)",
            key="stitch_tables",
            help="Merges consecutive markdown tables with identical headers, so a "
            "table spanning multiple pages becomes one logical table.",
        )
        st.session_state.setdefault("normalize_numbers", True)
        normalize_numbers = st.checkbox(
            "Append numeric interpretation guide (currency / units)",
            key="normalize_numbers",
            help="Detects currency symbols (₹, $, €…) and unit tokens (Cr, Lakh, Mn, "
            "Bn, K) in the source. Appends a normalised reference table to the LLM "
            "prompt — does not modify the source.",
        )
        with st.expander("Customise normaliser"):
            st.session_state.setdefault(
                "currency_csv", "₹,$,€,£,¥,Rs.,Rs,INR,USD,EUR,GBP,JPY"
            )
            currency_csv = st.text_input(
                "Currency symbols (comma-separated)", key="currency_csv"
            )
            st.session_state.setdefault(
                "units_csv",
                "Cr=10000000\nCrore=10000000\nLakh=100000\nLac=100000\n"
                "Mn=1000000\nMillion=1000000\nBn=1000000000\nBillion=1000000000\n"
                "K=1000\nThousand=1000",
            )
            units_csv = st.text_area(
                "Unit aliases (one per line, name=multiplier)",
                height=160,
                key="units_csv",
            )

        # ---------- 6. OUTPUT MODE ----------
        st.subheader("6. Output mode")
        modes = ["Free-text / Markdown", "Strict JSON schema"]
        if st.session_state.get("output_mode") not in modes:
            st.session_state["output_mode"] = "Strict JSON schema"
        output_mode = st.radio("Output mode", modes, key="output_mode")
        json_schema_text = ""
        if output_mode == "Strict JSON schema":
            st.session_state.setdefault(
                "json_schema_text",
                '{\n  "type": "object",\n  "properties": {\n    "vendor": {"type": "string"},\n    "total": {"type": "string"}\n  }\n}',
            )
            json_schema_text = st.text_area(
                "JSON Schema", height=220, key="json_schema_text"
            )

        # ---------- 7. PROMPTS ----------
        st.subheader("7. Prompts")
        prompt_files = _available_prompt_files()
        default_ext = (
            "grounded_extraction.md"
            if "grounded_extraction.md" in prompt_files
            else (prompt_files[0] if prompt_files else "")
        )
        default_str = (
            "grounded_structuring.md"
            if "grounded_structuring.md" in prompt_files
            else (prompt_files[0] if prompt_files else "")
        )
        if prompt_files:
            ext_choice = st.selectbox(
                "Vision extraction prompt source",
                prompt_files,
                index=prompt_files.index(default_ext)
                if default_ext in prompt_files
                else 0,
                key="ext_prompt_file",
            )
            str_choice = st.selectbox(
                "DI→LLM structuring prompt source",
                prompt_files,
                index=prompt_files.index(default_str)
                if default_str in prompt_files
                else 0,
                key="str_prompt_file",
            )
        else:
            ext_choice = str_choice = ""

        if not st.session_state.get("extraction_prompt"):
            st.session_state["extraction_prompt"] = _load_prompt(ext_choice)
        extraction_prompt = st.text_area(
            "Extraction prompt (vision)", height=180, key="extraction_prompt"
        )
        if not st.session_state.get("structuring_prompt"):
            st.session_state["structuring_prompt"] = _load_prompt(str_choice)
        structuring_prompt = st.text_area(
            "Structuring prompt (hybrid DI → LLM)", height=180, key="structuring_prompt"
        )

        # ---------- 8. PREPROCESSING ----------
        st.subheader("8. Preprocessing")
        deskew = st.checkbox("Deskew", value=False)
        denoise = st.checkbox("Denoise", value=False)
        grayscale = st.checkbox("Grayscale", value=False)
        page_range_str = st.text_input("Page range (e.g. 1-3, empty = all)", value="")

        # ---------- 9. EVALUATION ----------
        st.subheader("9. Evaluation")
        run_judge = st.checkbox("Run LLM-as-judge", value=False)
        show_ai_summary = st.checkbox(
            "Show AI business summary",
            value=True,
            help="Uses GPT-5 mini to explain the comparison in business language.",
        )
        gt_file = st.file_uploader(
            "Ground-truth (text or JSON, optional)",
            type=["txt", "md", "json"],
            accept_multiple_files=False,
        )

    # Parse normalize config
    currency_symbols = [s.strip() for s in (currency_csv or "").split(",") if s.strip()]
    unit_multipliers: dict[str, int] = {}
    for line in (units_csv or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        name, _, val = line.partition("=")
        try:
            unit_multipliers[name.strip().lower()] = int(val.strip())
        except ValueError:
            continue
    normalize_config = {
        "currency_symbols": currency_symbols,
        "unit_multipliers": unit_multipliers,
        "max_tokens": 60,
    }

    return {
        "preset_name": chosen,
        "enable_di": enable_di,
        "enable_gpt5_vision": enable_gpt5_vision,
        "enable_hybrid_gpt54_mini": enable_hybrid_gpt54_mini,
        "enable_hybrid_gpt51": enable_hybrid_gpt51,
        "enable_hybrid_gpt4o_mini": enable_hybrid_gpt4o_mini,
        "di_model": di_model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "top_p": top_p,
        "seed": int(seed),
        "repeat_runs": int(repeat_runs),
        "stitch_tables": stitch_tables,
        "normalize_numbers": normalize_numbers,
        "normalize_config": normalize_config,
        "output_mode": output_mode,
        "json_schema_text": json_schema_text,
        "extraction_prompt": extraction_prompt,
        "structuring_prompt": structuring_prompt,
        "deskew": deskew,
        "denoise": denoise,
        "grayscale": grayscale,
        "page_range_str": page_range_str,
        "run_judge": run_judge,
        "show_ai_summary": show_ai_summary,
        "gt_file": gt_file,
    }
