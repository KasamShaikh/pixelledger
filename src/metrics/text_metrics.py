"""Text-level accuracy metrics: CER, WER, side-by-side diff."""

from __future__ import annotations

import difflib
import re
from typing import Optional

from jiwer import cer, wer


_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def compute_cer_wer(reference: str, hypothesis: str) -> dict[str, Optional[float]]:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return {"cer": None, "wer": None}
    try:
        return {"cer": float(cer(ref, hyp)), "wer": float(wer(ref, hyp))}
    except Exception:  # noqa: BLE001
        return {"cer": None, "wer": None}


def html_diff(
    reference: str,
    hypothesis: str,
    *,
    ref_label: str = "Reference",
    hyp_label: str = "Hypothesis",
) -> str:
    differ = difflib.HtmlDiff(wrapcolumn=80)
    return differ.make_table(
        (reference or "").splitlines(),
        (hypothesis or "").splitlines(),
        fromdesc=ref_label,
        todesc=hyp_label,
        context=True,
        numlines=2,
    )
