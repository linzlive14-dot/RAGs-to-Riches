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

In a second terminal, run the Streamlit experience:

```bash
streamlit run app/streamlit_app.py
```

The UI includes workflow presets, cited policy cards, explicit mock-action
confirmation, and an expandable operational trace. It calls the API configured
by `HR_API_URL` (default `http://127.0.0.1:8000`). Open
<http://127.0.0.1:8000/health> to check the API. The expected response is:

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

## MCP tools and agent workflows

Run the synthetic HR MCP server over stdio:

```bash
python -m hr_mcp.server
```

It exposes seven discoverable tools for policy search and section retrieval,
synthetic employee/PTO/benefits lookup, deterministic compliance checks, and
confirmation-gated mock ticket creation. `HRMCPClient` is the only execution
boundary used by `HRAgent`; the orchestrator does not call tool implementations
directly.

The explicit state machine in `app/agent.py` supports `remote_work` and `pto`
requests. Callers provide a validated `WorkflowRequest`, then run it inside an
MCP client session:

```python
from app.agent import HRAgent, WorkflowRequest
from hr_mcp import HRMCPClient

async with HRMCPClient() as tools:
    result = await HRAgent(tools).run(
        WorkflowRequest(
            workflow="remote_work",
            employee_id="SYN-1001",
            requested_location_id="US-NY",
        )
    )
```

Results contain cited policy snippets and a concise operational trace with
states, tool names, safe arguments, summaries, sources, and escalation status.
The trace contains no hidden chain-of-thought. A mock ticket is never created
unless both `create_ticket=True` and `confirmed=True` are supplied.

The `POST /chat` endpoint accepts the same fields as `WorkflowRequest` and
returns the complete answer, citations, trace, status, and optional mock action.

## Evaluation and ablation

Run the deterministic 25-task evaluation and the retrieval ablation:

```bash
python -m evaluation.runner
```

This writes `evaluation/report.json` with groundedness, citation,
tool-selection, workflow, clarification, safety, and cold/warm p50/p95 latency
metrics. `evaluation/ablation.json` compares three chunk sizes against three
retrieval-k values and records source hit rate, grounded rate, and mean latency.
Both generated reports are reproducible runtime artifacts and are ignored by
Git.

This repository is a software demonstration. Its synthetic policies are not
legal, employment, medical, tax, or benefits advice.
