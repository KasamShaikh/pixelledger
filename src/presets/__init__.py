"""Document presets: pluggable scenario bundles (prompt + schema + defaults).

Drop a new `.json` file into this folder and it will appear in the sidebar
dropdown automatically. Each preset is fully self-describing.

Preset schema:
{
  "name": "Credit Memo → Structured Note",
  "description": "Credit / sanction memos to structured note (banking).",
  "extraction_prompt_file": "grounded_extraction.md",     // optional
  "structuring_prompt_file": "grounded_structuring.md",   // optional
  "json_schema": { ... } | null,
  "defaults": {
    "di_model": "prebuilt-layout",
    "output_mode": "Strict JSON schema" | "Free-text / Markdown",
    "stitch_tables": true,
    "normalize_numbers": true,
    "top_p": 1.0,
    "seed": 42,
    "temperature": 0.0,
    "reasoning_effort": "medium",
    "repeat_runs": 1
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PRESETS_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PRESETS_DIR.parent / "prompts"


def list_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    for f in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_id"] = f.stem
            presets.append(data)
        except Exception:  # noqa: BLE001
            continue
    return presets


def load_prompt(filename: str | None) -> str:
    if not filename:
        return ""
    p = PROMPTS_DIR / filename
    return p.read_text(encoding="utf-8") if p.exists() else ""


def resolve_preset(preset: Dict[str, Any]) -> Dict[str, Any]:
    """Return a resolved preset with prompt text inlined."""
    out = dict(preset)
    out["extraction_prompt"] = load_prompt(preset.get("extraction_prompt_file"))
    out["structuring_prompt"] = load_prompt(preset.get("structuring_prompt_file"))
    out.setdefault("defaults", {})
    out.setdefault("json_schema", None)
    return out
