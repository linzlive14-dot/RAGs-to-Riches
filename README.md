# RAGs to Riches

An agentic retrieval-augmented generation assistant for synthetic HR
workflows. The current foundation provides a Python 3.12 environment and a
FastAPI health endpoint.

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
