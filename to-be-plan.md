# OCR Demo Implementation Plan

## Goal
Implement summary behavior, confidence handling, and UI refresh based on current feedback.

## Phase 1 - Summary behavior by model count
- [x] Add summary mode branching in `src/ui/results_view.py`.
- [x] Single model mode:
  - [x] Rename summary section to `Run assessment`.
  - [x] Remove recommendation language.
  - [x] Show only:
    - [x] What went well
    - [x] What can be improved
- [x] Multi-model mode:
  - [x] Keep recommendation with best model and reason.
  - [x] Keep comparative explanation for how the run performed.
- [x] Update AI narrative prompt:
  - [x] Single model: strengths + improvements only.
  - [x] Multi model: recommendation + why/how.

## Phase 2 - Cost recommendation guard (resolved behavior + regression safety)
- [x] Preserve existing behavior:
  - [x] No cost recommendation for mixed GPT-5.* and GPT-4.* selections.
  - [x] Cost suggestion allowed only when selection is GPT-5.* family comparisons.
- [x] Add guard logic in narrative payload/prompt to prevent regressions.

## Phase 3 - Confidence score fix
- [x] Fix hybrid confidence propagation in `src/pipelines/hybrid.py`.
- [x] Copy DI confidence and pages to hybrid result before LLM call.
- [x] Ensure confidence metadata survives LLM exception path.
- [x] Improve confidence tab messaging in `src/ui/results_view.py`:
  - [x] Distinguish unsupported pipeline vs provider-missing confidence vs failed pipeline.

## Phase 4 - UI look and feel refresh
- [x] Update `app.py` styling to Indigo-inspired travel-booking style:
  - [x] Top navigation-like header
  - [x] Softer blue-gray surfaces
  - [x] Cleaner cards/inputs/buttons/tabs
  - [x] Responsive behavior preserved

## Phase 5 - Validation
- [x] Run app and validate:
  - [x] Single model summary = no recommendation text.
  - [x] Multi-model summary = recommendation shown.
  - [x] Confidence chart appears when DI confidence exists.
  - [x] Confidence message explains reason when unavailable.
  - [x] Mixed GPT-5 + GPT-4 has no cost recommendation text.
- [x] Add lightweight regression tests for summary/cost guard helpers.
