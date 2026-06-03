"""Merge consecutive markdown tables that share the same header row.

Document Intelligence often emits one markdown table per physical page when a
logical table spans pages. This breaks downstream LLM reasoning. We stitch
consecutive tables (optionally separated by page-break/page-number HTML
comments or blank lines) when their header rows are identical.

Generic; no bank- or domain-specific logic.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Strip HTML comments DI inserts at page boundaries.
_PAGE_MARKER = re.compile(
    r"<!--\s*(PageBreak|PageNumber=[^>]*?|PageHeader|PageFooter|PageFooter[^>]*?)\s*-->",
    re.IGNORECASE,
)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR_LINE = re.compile(r"^\s*\|\s*[:\-\s|]+\s*\|\s*$")


def _is_table_line(line: str) -> bool:
    return bool(_TABLE_LINE.match(line))


def _is_separator_line(line: str) -> bool:
    return bool(_SEPARATOR_LINE.match(line))


def _extract_tables(lines: List[str]) -> List[Tuple[int, int, List[str]]]:
    """Return list of (start_idx, end_idx_exclusive, table_lines)."""
    tables: List[Tuple[int, int, List[str]]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            start = i
            block: List[str] = []
            while i < n and _is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            # A valid markdown table needs header + separator + >=0 body rows.
            if len(block) >= 2 and _is_separator_line(block[1]):
                tables.append((start, i, block))
            continue
        i += 1
    return tables


def _normalize_header(header: str) -> str:
    cells = [c.strip().lower() for c in header.strip().strip("|").split("|")]
    return "||".join(cells)


def stitch_markdown_tables(markdown: str) -> Tuple[str, int]:
    """Return (stitched_markdown, num_merges_performed).

    Two tables are merged when, after stripping page markers and blank lines
    between them, their header rows (case-insensitive, whitespace-normalised)
    are identical.
    """
    if not markdown:
        return markdown, 0

    lines = markdown.splitlines()
    tables = _extract_tables(lines)
    if len(tables) < 2:
        return markdown, 0

    # Walk pairs; for each pair check the "between" region is only page markers / blanks
    keep_mask = [True] * len(lines)
    drop_separator_for: List[int] = []  # table indices whose header+separator we drop
    merges = 0

    prev_idx = 0
    while prev_idx < len(tables) - 1:
        curr_idx = prev_idx + 1
        _, prev_end, prev_block = tables[prev_idx]
        curr_start, _, curr_block = tables[curr_idx]

        between = lines[prev_end:curr_start]
        stripped = [
            ln for ln in between if ln.strip() and not _PAGE_MARKER.match(ln.strip())
        ]
        if not stripped and _normalize_header(prev_block[0]) == _normalize_header(
            curr_block[0]
        ):
            # Drop everything between (page markers + blanks) AND drop the curr table's header+separator
            for j in range(prev_end, curr_start):
                keep_mask[j] = False
            # Drop header (curr_start) and separator (curr_start+1)
            keep_mask[curr_start] = False
            keep_mask[curr_start + 1] = False
            merges += 1
            # Effectively the current table now becomes part of prev for subsequent comparisons;
            # update tables[curr_idx] start so further merges chain
            tables[curr_idx] = (tables[prev_idx][0], tables[curr_idx][1], prev_block)
        prev_idx = curr_idx

    if merges == 0:
        return markdown, 0

    out = [ln for ln, keep in zip(lines, keep_mask) if keep]
    return "\n".join(out), merges
