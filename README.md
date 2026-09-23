# RAGs to Riches

An agentic retrieval-augmented generation assistant for synthetic HR
workflows. It includes a Python 3.12 API foundation, a fictional HR policy
corpus, synthetic employee data, and deterministic policy retrieval.

## Local setup

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
cp .env.example .env
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/health>. The expected response is:

```json
{"status":"ok","service":"RAGs to Riches","environment":"development"}
```

Run the test suite:

```bash
pytest
```

## Policy corpus and retrieval

The eight fictional policies in `policies/` use Markdown and plain text.
`policies/manifest.json` records their provenance and AI assistance. The JSON
records in `mock_data/` are explicitly synthetic and use reserved
`example.invalid` addresses.

Build the local SQLite FTS5 index reproducibly:

```bash
python -m rag build
```

Search chunks or produce a guarded, extractive answer with inline citations:

```bash
python -m rag search "fully remote tenure and location approval"
python -m rag answer "Can I use PTO during parental leave?"
```

The index uses heading-aware word chunks with fixed overlap, stable SHA-256
identifiers, BM25 ranking, and deterministic tie-breaking. Every result
contains `document_id`, `title`, `section`, `source`, and `snippet` metadata.
The answer path labels policy guidance, refuses unsupported questions without
inventing citations, and rejects common instruction-override or secret-seeking
queries. The generated `data/policy_index.sqlite3` is intentionally not
committed; rebuilding it is the source of truth.

This repository is a software demonstration. Its synthetic policies are not
legal, employment, medical, tax, or benefits advice.
