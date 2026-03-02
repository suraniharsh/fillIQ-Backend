# fillIQ Backend — Business Card OCR & Contact Extractor

> Snap a business card → get a structured JSON contact record, instantly.

fillIQ uses two AI models running entirely on CPU/GPU to turn raw image pixels into clean, structured contact data ready to import into any CRM.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Request Flow](#request-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Contact Schema](#contact-schema)
- [Configuration](#configuration)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        fillIQ System                            │
│                                                                 │
│  ┌──────────────────────┐        ┌───────────────────────────┐  │
│  │   Next.js Frontend   │        │    FastAPI Backend         │  │
│  │   (webapp/)          │        │    (server.py)             │  │
│  │                      │        │                            │  │
│  │  ┌────────────────┐  │  HTTP  │  ┌─────────────────────┐  │  │
│  │  │  Upload UI     │◄─┼────────┼─►│  POST /ocr          │  │  │
│  │  │  (drag & drop) │  │        │  │  GET  /health       │  │  │
│  │  └────────────────┘  │        │  └────────┬────────────┘  │  │
│  │  ┌────────────────┐  │        │           │               │  │
│  │  │  Results Panel │  │        │  ┌────────▼────────────┐  │  │
│  │  │  (JSON viewer) │  │        │  │  LightOnOCR-2-1B    │  │  │
│  │  └────────────────┘  │        │  │  (OCR model, CPU)   │  │  │
│  └──────────────────────┘        │  └────────┬────────────┘  │  │
│                                  │           │  raw text      │  │
│                                  │  ┌────────▼────────────┐  │  │
│                                  │  │  Qwen2.5-0.5B       │  │  │
│                                  │  │  (JSON structuring) │  │  │
│                                  │  └────────┬────────────┘  │  │
│                                  │           │  contact JSON  │  │
│                                  │  ┌────────▼────────────┐  │  │
│                                  │  │  ocr_results/       │  │  │
│                                  │  │  (JSON file store)  │  │  │
│                                  │  └─────────────────────┘  │  │
│                                  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Request Flow

```
  Client (browser)
       │
       │  POST /ocr  (multipart image upload)
       ▼
  ┌────────────────────────────────────────────────┐
  │              FastAPI  /ocr  endpoint           │
  │                                                │
  │  1. Validate file size  (≤ 10 MB)             │
  │  2. Decode image bytes → PIL Image (RGB)       │
  └──────────────────────┬─────────────────────────┘
                         │
                         ▼  (serialized via asyncio.Lock)
  ┌────────────────────────────────────────────────┐
  │           LightOnOCR-2-1B  (CPU)               │
  │                                                │
  │  • apply_chat_template (vision + text prompt)  │
  │  • model.generate()  → token ids              │
  │  • decode → plain text                         │
  └──────────────────────┬─────────────────────────┘
                         │  raw OCR text
                         ▼
  ┌────────────────────────────────────────────────┐
  │       Qwen2.5-0.5B-Instruct  helper LLM        │
  │                                                │
  │  • System prompt with flat contact schema      │
  │  • User prompt: raw OCR text                   │
  │  • Output: JSON object (parsed & normalized)   │
  └──────────────────────┬─────────────────────────┘
                         │  structured contact dict
                         ▼
  ┌────────────────────────────────────────────────┐
  │               Response  +  Persist             │
  │                                                │
  │  • JSON written to ocr_results/<timestamp>.json│
  │  • HTTP 200 → { raw_text, structured_json,     │
  │                 json_helper_output }           │
  └────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer       | Technology                                      |
|-------------|-------------------------------------------------|
| Frontend    | Next.js 14, React 18, Tailwind CSS, TypeScript  |
| Backend     | Python 3.10-3.12, FastAPI, Uvicorn              |
| OCR model   | `lightonai/LightOnOCR-2-1B` (CPU inference)     |
| JSON helper | `Qwen/Qwen2.5-0.5B-Instruct`                    |
| ML runtime  | PyTorch, Hugging Face Transformers, LangChain   |
| Packaging   | `uv` (Python), `npm` (Node)                     |

---

## Project Structure

```
fillIQ-Backend/
├── server.py            # FastAPI application (OCR + JSON extraction)
├── main.py              # Entry-point placeholder
├── pyproject.toml       # Python project metadata & dependencies
├── uv.lock              # Locked dependency tree (uv)
├── ocr_results/         # Auto-created; stores per-request JSON files
└── webapp/              # Next.js frontend
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx
    │   └── globals.css
    ├── components/      # Shared React components
    ├── lib/             # Utility helpers
    ├── next.config.mjs
    ├── tailwind.config.js
    └── package.json
```

---

## Getting Started

### Prerequisites

- Python 3.10 – 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 18+ and npm (for the frontend)

### Backend

```bash
# 1. Install dependencies
uv sync

# 2. Start the API server (models download on first run — ~2 GB)
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`.

> **Note:** Model weights are fetched automatically from Hugging Face on first startup.  
> Set `JSON_MODEL_ID` environment variable to swap the helper LLM (default: `Qwen/Qwen2.5-0.5B-Instruct`).

### Frontend

```bash
cd webapp

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at `http://localhost:3000`.

---

## API Reference

### `GET /health`

Returns server health status.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /ocr`

Upload a business card image; receive raw OCR text and a structured contact JSON.

**Request** — `multipart/form-data`

| Field  | Type | Description                        |
|--------|------|------------------------------------|
| `file` | file | JPEG / PNG image (max **10 MB**)   |

**Response** — `application/json`

```json
{
  "raw_text": "John Doe\nSenior Engineer\njohn@example.com\n+1 555 0100",
  "structured_json": {
    "salutation": "",
    "first_name": "John",
    "last_name": "Doe",
    "gender": "",
    "email_id": "john@example.com",
    "mobile_no": "+1 555 0100",
    "no_of_employees": "",
    "country": "",
    "company_name": "",
    "title": "Senior Engineer",
    "industry": "",
    "city": "",
    "state": "",
    "status": ""
  },
  "json_helper_output": "{ ... raw LLM output ... }"
}
```

**Error codes**

| Code | Meaning                        |
|------|--------------------------------|
| 400  | Invalid or unreadable image    |
| 413  | Image exceeds 10 MB size limit |
| 503  | Models not yet loaded          |

---

## Contact Schema

Every response contains a `structured_json` object with the following flat fields:

```
┌──────────────────┬──────────────────────────────────┐
│ Field            │ Example value                    │
├──────────────────┼──────────────────────────────────┤
│ salutation       │ "Mr."                            │
│ first_name       │ "John"                           │
│ last_name        │ "Doe"                            │
│ gender           │ "Male"                           │
│ email_id         │ "john@example.com"               │
│ mobile_no        │ "+1 555 0100"                    │
│ no_of_employees  │ "500"                            │
│ country          │ "USA"                            │
│ company_name     │ "Acme Corp"                      │
│ title            │ "Senior Engineer"                │
│ industry         │ "Technology"                     │
│ city             │ "San Francisco"                  │
│ state            │ "CA"                             │
│ status           │ ""                               │
└──────────────────┴──────────────────────────────────┘
```

Missing fields are returned as empty strings `""`, never `null`.

---

## Configuration

| Environment Variable | Default                          | Description                         |
|----------------------|----------------------------------|-------------------------------------|
| `JSON_MODEL_ID`      | `Qwen/Qwen2.5-0.5B-Instruct`    | Hugging Face model ID for the JSON helper LLM |

CORS is pre-configured to accept requests from `localhost:3000` and `127.0.0.1:3000`.  
Edit the `allow_origins` list in `server.py` to add additional origins.
