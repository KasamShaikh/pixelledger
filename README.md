# OCR Accuracy Demo — Document Intelligence + GPT-5

A Streamlit demo that runs the **same document** through 5 OCR / extraction pipelines side-by-side, so customers can see for themselves why **Azure Document Intelligence + GPT-5** is the winning combination.

## Pipelines compared

| # | Pipeline | What it does |
|---|----------|--------------|
| 1 | **DI + GPT-5.4 mini** | DI Markdown → GPT-5.4 mini structuring |
| 2 | **DI + GPT-5.1** | DI Markdown → GPT-5.1 structuring |
| 3 | **DI + GPT-4.0 Mini** | DI Markdown → GPT-4o mini structuring |
| 4 | **GPT-5 vision** | Raw page image → GPT-5 multimodal |
| 5 | **DI only** | `prebuilt-layout`, `prebuilt-read`, or `prebuilt-invoice` |

## Features

- **Secure login** — username + passcode sign-in to enter the workspace
- **Request access** — visitors without a passcode can submit a short access request from the login screen (sent for review)
- Upload PDF / PNG / JPG / TIFF (multi-page supported)
- Side-by-side **raw text**, **structured JSON**, and **confidence heatmaps**
- Accuracy metrics: **CER / WER**, field-level **F1**, **LLM-as-judge** rubric
- **DocTalk** — chat with the extracted text; each selected pipeline answers the same question from its **own** extraction, side-by-side, so you can compare accuracy. Answers are grounded in the document (replies "Not found in the document." when the info is absent)
- Editable extraction & structuring prompts
- Strict JSON-schema output mode
- Preprocessing: deskew, denoise, grayscale, page range
- Optional ground-truth file (txt/md/json) unlocks accuracy metrics
- Downloadable JSON bundle of all results

## Setup

```powershell
cd ocr-accuracy-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env with your DI + Azure OpenAI keys / deployment names
streamlit run app.py
```

## Environment variables

See `.env.example`. You need:

- `AZURE_DI_ENDPOINT` / `AZURE_DI_KEY` — Document Intelligence resource
- `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_KEY` — Azure OpenAI resource
- `AOAI_DEPLOYMENT_GPT5`, `AOAI_DEPLOYMENT_GPT4O`, etc. — your deployment names

If GPT-5 is not yet provisioned in your region, point `AOAI_DEPLOYMENT_GPT5` to a compatible deployment and the demo still runs.

### Auth storage backend

By default, local auth/request/login JSON files are stored in `./data`.

- `AUTH_STORAGE_BACKEND=local` (default)
- `AUTH_STORAGE_BACKEND=blob` to store JSON in Azure Blob Storage

When using blob mode, set:

- `AUTH_BLOB_CONNECTION_STRING`
- `AUTH_BLOB_CONTAINER` (default: `appdata`)
- `AUTH_BLOB_PREFIX` (default: `auth`)

Blob mode persists:

- `users.json`
- `passcode_requests.json`
- `login_activity.json`

under the configured blob prefix.

## Deploy to Azure Container Apps (ACA)

This repo includes a starter deployment flow for:

- Resource Group: `rg-ocr-demo`
- Container App: `pixelledger`
- ACA Environment: `ks-pixelledger`

Files:

- `deploy/aca/deploy.ps1`
- `deploy/aca/set-model-env.ps1`
- `deploy/aca/README.md`

Quick start:

```powershell
./deploy/aca/deploy.ps1
./deploy/aca/set-model-env.ps1 -AzDiEndpoint <...> -AzDiKey <...> -AzOpenAiEndpoint <...> -AzOpenAiKey <...>
```

The deployment script enables blob-backed auth storage automatically for ACA.

## Project layout

```
ocr-accuracy-demo/
├── app.py                      # Streamlit entry
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py               # Env + pricing
│   ├── orchestrator.py         # Concurrent pipeline runner
│   ├── doctalk.py              # Grounded per-pipeline chat over extracted text
│   ├── preprocess.py           # PDF→PNG, deskew, denoise
│   ├── pipelines/
│   │   ├── base.py
│   │   ├── doc_intelligence.py
│   │   ├── llm_vision.py
│   │   └── hybrid.py
│   ├── metrics/
│   │   ├── text_metrics.py     # CER / WER / diff
│   │   ├── schema_metrics.py   # Field F1
│   │   └── llm_judge.py        # GPT-5 rubric scoring
│   ├── prompts/
│   │   ├── extraction.md
│   │   └── structuring.md
│   └── ui/
│       ├── sidebar.py
│       └── results_view.py
└── samples/                    # Place demo docs + optional GT here
```

## Sales talk-track

1. Sign in with your username and passcode (or request access from the login screen).
2. Upload a low-quality scanned invoice.
3. Enable all 5 pipelines, leave defaults.
4. Click **Run comparison**.
5. Open the **Compare** tab:
   - Show the lowest CER/WER and strongest field F1.
6. Switch to **Strict JSON schema** mode and re-run to show clean structured output.
7. Open the **DI** tab to show **confidence scores** (a unique DI advantage).
8. Toggle **LLM-as-judge** to add a qualitative rubric score.
9. Open the **DocTalk** tab and ask a question (e.g. "What is the booking fee?") — each pipeline answers from its own extraction, so weaker extractions visibly miss values others capture.

## Future work

- Persist runs in SQLite for run-history view
- PII redaction toggle for customer-provided documents
- Custom DI model support (trained extractors)
