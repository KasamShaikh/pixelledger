# Plan — Run History (SQLite) + PII Redaction Toggle

Two independent features. They can ship in either order; Feature B touches the
run pipeline that Feature A persists, so if both ship, persist *after* redaction
so history reflects what was actually sent to the models.

---

## Feature A — Persist runs in SQLite for a run-history view

### Goal
Every analysis run is saved to a local SQLite database so users can browse past
runs, re-open results without re-spending tokens, and compare runs over time.

### Why SQLite (vs. the existing JSON pattern)
- Run payloads are large (raw text, structured JSON, per-line confidence) and
  grow unbounded — JSON-file rewrite-on-append (`_update_json`) does not scale.
- Run-history needs filtering/sorting (by user, date, filename, pipeline) which
  is trivial in SQL and painful in JSON.
- `sqlite3` is in the Python stdlib — no new dependency.

### Storage location & container caveat
- Default path: `DATA_DIR / "runs.db"` (i.e. `data/runs.db`), mirroring the
  existing `data/*.json` convention. Override via `RUNS_DB_PATH` env var.
- **Container caveat:** ACA filesystem is ephemeral, so `data/runs.db` is lost
  on revision roll (same limitation the JSON auth files already have). Document
  this. For durable history, a later iteration can point `RUNS_DB_PATH` at an
  Azure Files mount or swap the backend to Postgres — keep the data layer behind
  a thin module so that swap is localized.
- Add `data/*.db` to `.gitignore` (DB contains customer document text = PII).

### Schema
Two tables, parent/child:

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    created_utc   TEXT NOT NULL,            -- ISO-8601
    username      TEXT NOT NULL,
    filename      TEXT NOT NULL,
    mime_type     TEXT,
    pages         INTEGER,
    repeat_runs   INTEGER,
    pipeline_count INTEGER,
    redacted      INTEGER NOT NULL DEFAULT 0,  -- 1 if PII redaction applied (Feature B)
    gt_present    INTEGER NOT NULL DEFAULT 0,
    judge_present INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    options_json  TEXT                       -- sanitized opts snapshot (no secrets)
);

CREATE TABLE IF NOT EXISTS run_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    pipeline_id   TEXT NOT NULL,
    display_name  TEXT,
    run_index     INTEGER,
    result_json   TEXT NOT NULL              -- full PipelineResult serialized
);

CREATE INDEX IF NOT EXISTS idx_runs_user_time ON runs(username, created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_results_run ON run_results(run_id);
```

Notes:
- `result_json` stores the whole `PipelineResult` so the results view can be
  re-rendered byte-for-byte. `raw_response` (provider SDK object) is **dropped**
  before serialization — it is not JSON-safe and not needed for display.
- `options_json` is a scrubbed copy of `opts` (drop file handles, drop any keys;
  keep model/pipeline/preprocess choices) so a run can show "how it was run".

### New module: `src/storage/run_store.py`
Thin, dependency-free data layer. Functions:
- `_connect() -> sqlite3.Connection` — opens `RUNS_DB_PATH`, sets
  `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, runs `CREATE TABLE IF NOT EXISTS`.
- `save_run(run_meta: dict, results: list[PipelineResult]) -> None` — one
  transaction: insert into `runs`, bulk-insert `run_results`.
- `list_runs(username: str | None = None, limit: int = 100) -> list[dict]` —
  rows from `runs` for the history table (no heavy payloads).
- `load_run(run_id: str) -> tuple[dict, list[PipelineResult]]` — reconstruct
  meta + `PipelineResult` objects for re-rendering.
- `delete_run(run_id: str) -> None` — user deletes own run.
- Serialization helpers: `_result_to_json(r)` / `_result_from_json(d)` using
  `dataclasses.asdict` minus `raw_response`.

### Wiring into `app.py` (`_run_analysis`)
- After results + judge + gt are computed (right where
  `st.session_state["last_results"] = results` is set), call `save_run(...)`
  inside a `try/except` that logs but never blocks the UI on a DB error.
- Compute `total_cost_usd = sum(r.cost_usd for r in results)`.
- Pass the existing `run_id`, `current_user`, `uploaded.name`, `repeat_n`,
  `len(pipelines)`, and a scrubbed `opts` snapshot.

### UI — Run History view
- Add a top-nav entry "🕓 History" (next to existing nav). Gate to the logged-in
  user's own runs (`username == auth_username`); admins out of scope here.
- History table (via `list_runs`): columns Date, Filename, Pipelines, Repeat,
  Cost, Redacted, GT/Judge flags. Newest first.
- Row actions:
  - **Open** → `load_run(run_id)` → set `st.session_state["last_results"]`
    (+ gt/judge) → switch to the results workspace and `st.rerun()`. Reuses the
    *existing* `render_results(...)` — no duplicate rendering code.
  - **Delete** → `delete_run` with a confirm, then `st.rerun()`.
- DocTalk history is per-session and not persisted (out of scope).

### Tests (`tests/test_run_store.py`)
- Round-trip: `save_run` then `load_run` reproduces field values; `raw_response`
  is dropped; `run_index` preserved.
- `list_runs` filters by username and orders by `created_utc DESC`.
- `delete_run` cascades to `run_results`.
- Uses a temp DB path (`RUNS_DB_PATH` / monkeypatched) — no global state.

### Acceptance
- Running an analysis creates exactly one `runs` row + N `run_results` rows.
- Opening a history row re-renders identical results with zero new token spend.
- DB errors degrade gracefully (run still shown live; warning logged).
- All existing tests still pass.

---

## Feature B — PII redaction toggle for customer-provided documents

### Goal
An opt-in sidebar toggle that redacts personally identifiable information so that
sensitive customer documents can be demoed without sending raw PII to the models
(or, alternatively, redacts only what is shown/stored). Make the **stage**
explicit because it changes the privacy guarantee.

### Key design decision — redact BEFORE or AFTER the model call?
Two modes; pick the default, expose the other as a sub-option:

| Mode | What happens | Privacy | Accuracy impact |
|------|--------------|---------|-----------------|
| **Pre-OCR (input redaction)** | Mask PII on the page images before they reach DI / GPT vision | Strongest — PII never leaves the box to Azure | May hurt extraction (model sees boxes) — but that's the point for a privacy demo |
| **Post-extraction (output redaction)** | Run pipelines normally, redact PII in `raw_text` / `structured_json` before display + storage | Weaker — raw PII still sent to Azure | No accuracy impact; protects the saved history + screen |

**Recommended default: Post-extraction**, because the whole product compares OCR
accuracy and pre-OCR masking destroys the thing being measured. Offer Pre-OCR as
an explicit "maximum privacy (may reduce accuracy)" choice. Persist the chosen
mode in the run record (`redacted` flag + mode in `options_json`).

### Detection approach (no new heavy deps)
- Phase 1 — **regex/heuristic detectors** in a new `src/privacy/redact.py`:
  emails, phone numbers, credit-card-like (Luhn-checked) digit runs, government
  IDs (SSN/Aadhaar/PAN patterns), IBAN, dates of birth. Cheap, deterministic,
  offline, good enough for a demo. Replace each match with a typed token,
  e.g. `[REDACTED:EMAIL]`.
- Phase 2 (optional, later) — Azure AI Language **PII detection** for
  name/address/entity coverage. Keep the detector behind an interface so it can
  be swapped without touching call sites. Not in initial scope.

### New module: `src/privacy/redact.py`
- `PII_PATTERNS: dict[str, re.Pattern]` — labeled compiled regexes.
- `redact_text(text: str) -> tuple[str, list[dict]]` — returns redacted text plus
  a list of `{type, start, end}` spans (counts only; do **not** store the raw PII
  values).
- `redact_structured(obj) -> obj` — recurse dict/list, redact string leaves.
- `redact_image(png_bytes) -> bytes` (Pre-OCR mode only) — OCR the page (reuse
  DI words+bounding boxes already available), match PII spans to word boxes, draw
  filled rectangles over them with OpenCV (`cv2.rectangle`). Reuses existing
  `cv2`/`PIL` deps. If word boxes aren't available for a page, fall back to a
  warning and skip (don't silently pass PII through).

### Sidebar toggle (`src/ui/sidebar.py`)
- In the preprocessing/options expander, add:
  - `redact_pii = st.toggle("Redact PII", value=False, help=...)`
  - When on, a `st.radio` for mode: "After extraction (recommended)" /
    "Before OCR (max privacy)".
- Add both to the returned `opts` dict: `opts["redact_pii"]`,
  `opts["redact_mode"]`.

### Wiring into `app.py` (`_run_analysis`)
- **Post-extraction mode:** after `run_all(...)` returns, if
  `opts["redact_pii"]` and mode == post, map over `results` and replace
  `r.raw_text` / `r.structured_json` with redacted copies **before** judge,
  display, DocTalk, and `save_run`. Set `redacted=1` for the run record. Note:
  judge/GT comparison then runs on redacted text — surface a caption explaining
  this.
- **Pre-OCR mode:** redact `doc.images` (and skip re-deriving from `content`)
  *before* building pipelines. Carry a `redacted=True` flag into the run record.
- DocTalk: since it reads from the (now redacted) results, it automatically
  won't surface PII — verify with a test.

### UI surfacing
- When redaction is active, show a badge on the results header:
  "🛡️ PII redacted ({mode})".
- Add a short note to the existing disclaimer area that redaction is heuristic
  and not a compliance guarantee.

### Tests (`tests/test_redact.py`)
- `redact_text` masks email/phone/card (Luhn) and leaves normal text intact.
- Card detector rejects non-Luhn digit runs (no false positives on order IDs).
- `redact_structured` recurses nested dict/list and preserves shape/keys.
- Span list reports counts but never echoes the raw PII value.
- (Pre-OCR) `redact_image` returns valid PNG bytes of same dimensions.

### Acceptance
- Toggle OFF → behavior is byte-for-byte identical to today (regression-safe).
- Toggle ON (post) → no email/phone/card patterns appear in displayed text,
  DocTalk answers, or the saved SQLite run.
- Toggle ON (pre) → page images sent to pipelines have PII boxed out.
- Heuristic-only, offline, no new pip dependency for Phase 1.

---

## Suggested sequencing
1. Feature B module + tests (`src/privacy/redact.py`, post-extraction path only).
2. Feature A module + tests (`src/storage/run_store.py`).
3. Wire both into `_run_analysis` (redact → persist), add History + toggle UI.
4. Pre-OCR image redaction (optional follow-up).
5. `.gitignore` `data/*.db`; `.env.example` add `RUNS_DB_PATH`; README: History
   section + PII toggle note + ephemeral-storage caveat.
```
