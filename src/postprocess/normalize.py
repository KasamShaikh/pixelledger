"""Detect numeric / currency tokens in source text and build a normalisation
legend appended to the LLM prompt so the model interprets units consistently.

We do NOT mutate the source markdown — banks reject any post-OCR rewrites of
numbers. Instead we append a side-table "INTERPRETATION GUIDE" listing each
unique numeric token and its normalised value. The LLM uses it for grounding.

Generic across currencies / locales — fully driven by a config dict.
"""

from __future__ import annotations

import re
from typing import Dict, List


def default_normalize_config() -> Dict[str, object]:
    return {
        # Currency symbols to detect (Unicode-safe)
        "currency_symbols": [
            "₹",
            "$",
            "€",
            "£",
            "¥",
            "Rs.",
            "Rs",
            "INR",
            "USD",
            "EUR",
            "GBP",
            "JPY",
        ],
        # Multiplier tokens (case-insensitive). Value is the numeric multiplier.
        "unit_multipliers": {
            "cr": 10_000_000,
            "crore": 10_000_000,
            "crores": 10_000_000,
            "lakh": 100_000,
            "lakhs": 100_000,
            "lac": 100_000,
            "lacs": 100_000,
            "mn": 1_000_000,
            "million": 1_000_000,
            "bn": 1_000_000_000,
            "billion": 1_000_000_000,
            "k": 1_000,
            "thousand": 1_000,
        },
        # Max tokens to include in legend (keeps prompt size bounded).
        "max_tokens": 60,
    }


_NUM = r"(?:\d{1,3}(?:[,\s]\d{2,3})+|\d+)(?:\.\d+)?"


def _build_pattern(cfg: Dict[str, object]) -> re.Pattern:
    syms = sorted(
        [re.escape(s) for s in cfg.get("currency_symbols", [])], key=len, reverse=True
    )
    units = sorted(
        [re.escape(u) for u in cfg.get("unit_multipliers", {}).keys()],
        key=len,
        reverse=True,
    )
    sym_alt = "|".join(syms) if syms else r"(?!x)x"  # never-match if empty
    unit_alt = "|".join(units) if units else r"(?!x)x"
    # Pattern A: <symbol> <number> [unit]
    # Pattern B: <number> <unit>   (e.g., "12.5 Cr")
    pat = rf"(?:(?:{sym_alt})\s*{_NUM}(?:\s*(?:{unit_alt}))?\b)|(?:{_NUM}\s*(?:{unit_alt})\b)"
    return re.compile(pat, re.IGNORECASE)


def _strip_separators(num_str: str) -> float:
    cleaned = num_str.replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def _normalize_token(token: str, cfg: Dict[str, object]) -> str:
    units = cfg["unit_multipliers"]  # type: ignore[index]
    syms = cfg["currency_symbols"]  # type: ignore[index]

    # Find unit (longest match)
    unit_mult = 1
    unit_used = ""
    for u, m in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(u)}\b", token, flags=re.IGNORECASE):
            unit_mult = m
            unit_used = u.title()
            break

    # Find currency symbol
    sym_used = ""
    for s in sorted(syms, key=len, reverse=True):
        if s in token:
            sym_used = s
            break

    # Find numeric part
    m = re.search(_NUM, token)
    if not m:
        return token + "  →  (unparsed)"
    raw = m.group(0)
    val = _strip_separators(raw)
    if val != val:  # NaN
        return token + "  →  (unparsed)"
    absolute = val * unit_mult

    parts = []
    if sym_used:
        parts.append(sym_used)
    parts.append(f"{absolute:,.2f}")
    parts.append("(absolute value)")
    if unit_used:
        parts.append(f"[unit: {unit_used} = ×{unit_mult:,}]")
    return f"{token.strip()}  →  {' '.join(parts)}"


def build_numeric_legend(text: str, cfg: Dict[str, object] | None = None) -> str:
    """Return a markdown legend string (empty if no tokens detected)."""
    if not text:
        return ""
    cfg = cfg or default_normalize_config()
    pat = _build_pattern(cfg)
    tokens: List[str] = []
    seen = set()
    for m in pat.finditer(text):
        tok = m.group(0).strip()
        key = re.sub(r"\s+", " ", tok.lower())
        if key in seen:
            continue
        seen.add(key)
        tokens.append(tok)
        if len(tokens) >= int(cfg.get("max_tokens", 60)):
            break
    if not tokens:
        return ""
    lines = [
        "### INTERPRETATION GUIDE (numeric tokens — for grounding only; do not rewrite values in output)"
    ]
    for tok in tokens:
        lines.append(f"- {_normalize_token(tok, cfg)}")
    return "\n".join(lines)
