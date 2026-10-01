# AI tooling and provenance

## Tools used

- **Cursor (agent mode with Claude models)** was the primary development tool.
  It drafted the project plan from the rubric, then implemented the work in
  gated stages: foundation, corpus and RAG, MCP tools and agent, web and
  evaluation, and deployment and documentation. Each stage was reviewed and
  committed separately, which is visible in the Git history.
- **Cursor plan mode** produced the initial architecture and activity outline
  before any code was written.
- **Cursor subagents** handled parallel tasks such as model-slug verification and
  repository setup.
- **OpenRouter-hosted free models** run at runtime for intent classification,
  answer wording, and LLM-judge scoring during evaluation (see below).

## How the tools were used

| Area | AI contribution | Human role |
| --- | --- | --- |
| Planning | Turned the rubric into a staged plan and architecture | Chose scope, stack, and stage gates |
| Policy corpus | Drafted twelve fictional policies in Markdown and TXT | Directed topics and reviewed for coherence |
| Synthetic data | Generated employee, PTO, benefits, location, and ticket records | Checked that records exercise each workflow branch |
| Code | Wrote ingestion, index, MCP server/client, agent, API, UI, CI | Reviewed diffs, ran the app, reported failures |
| Tests and evaluation | Wrote unit, MCP, API, and gold-task tests | Defined expected outcomes and reviewed results |
| Documentation | Drafted README, design, and deployment docs | Asked for clarifications where instructions were unclear |

## What worked well

- **Staged, test-gated generation.** Asking for one plan stage at a time, with
  tests before moving on, kept each change reviewable and caught regressions
  early. The full suite runs in a few seconds, so it was cheap to rerun after
  every change.
- **Real MCP boundary from the start.** Asking explicitly for tools to be called
  through an MCP client, not as direct function calls, produced an architecture
  that matches the rubric without later refactoring.
- **Rapid synthetic content.** Fictional policies and records that line up with
  each other, for example tenure rules matching hire dates, took minutes rather
  than hours.
- **Explaining behavior.** Asking the agent why a request failed, such as why
  the UI fell back to deterministic text, gave precise, code-referenced answers.

## What did not work well

- **Free-model availability.** The first model chosen,
  `inclusionai/ling-3.0-flash-vl:free`, returned HTTP 404 from OpenRouter
  because the free variant was not available. The app showed only "LLM
  generation failed validation", which hid the real cause. Switching models in
  `.env` fixed it, but this showed that the fallback message is too generic.
- **Blank secrets treated as configured.** An empty `OPENROUTER_API_KEY=` line
  was read as a configured key, sent an empty bearer token, and failed the same
  opaque way. Settings now treat a blank key as unset.
- **Unknown inputs surfaced as tool outages.** Early versions raised an
  exception for an unknown location such as `US-WI`, which the agent reported as
  "tool unavailable". This was fixed by returning `found: false` findings the
  agent can explain.
- **Drift from the plan.** The plan called for Chroma with local embeddings.
  The first generated implementation used SQLite FTS5 keyword ranking only,
  which departed from the rubric's embedding requirement. Retrieval was later
  rebuilt as BM25 plus local `bge-small` embeddings in FAISS, merged with rank
  fusion. Natural-language gold questions then showed the difference
  (24/30 passed for BM25 only, 28/30 for hybrid).
- **Keyword grounding.** The keyword-overlap check refused answerable
  paraphrases ("When do I get paid?") and accepted an off-topic one. It was
  replaced by an embedding-similarity threshold set on a separate development
  probe.
- **Setup instructions.** Generated README steps assumed familiarity with
  virtual environments. For example, a new terminal reported
  `streamlit: command not found` until the README explained activating `.venv`
  in each terminal.
- **Evaluation scope.** The first evaluation runner never called the LLM, and
  its gold questions were keyword strings, so its perfect scores measured only
  the deterministic path on easy inputs. The gold set now uses natural
  questions with key-fact answers. The offline mode is labeled as such, and the
  LLM-graded mode scores model answers with a judge.

## Data provenance

`policies/manifest.json` identifies the policy corpus as fictional, AI-assisted
demonstration content. No external policy text was copied. The records under
`mock_data/` are generated examples, use reserved synthetic identifiers, and do
not represent real employees. Addresses use the reserved `example.invalid`
domain.

No prompt, source file, or runtime path requires real HR data. Contributors must
not add real employee records, credentials, proprietary policies, or provider
keys to the repository.

## Runtime model use

When `OPENROUTER_API_KEY` is set, the agent uses the model named by
`OPENROUTER_MODEL` through OpenRouter's OpenAI-compatible API for three jobs:

1. classifying the user's intent and extracting slots, which are re-validated
   against the message and the location register;
2. writing the final answer from the tool evidence, with citation IDs checked
   against retrieved sources;
3. judging groundedness and citation accuracy during evaluation.

Retrieval uses a local embedding model, `BAAI/bge-small-en-v1.5`, run on the
CPU through `fastembed`. It needs no key or network after the first download.
Retrieval, compliance decisions, and mock-action gating stay deterministic, so
the model cannot change an eligibility outcome. If the provider is unavailable,
the agent answers with deterministic cited wording and the trace records the
fallback. Keys are read from the environment and never appear in
`.env.example`, logs, traces, fixtures, or commits.

## Verification

AI-assisted changes are checked with:

- deterministic ingestion and retrieval tests;
- MCP discovery and real tool-call integration tests;
- API, workflow, clarification, failure, and action-confirmation tests;
- a 30-task gold evaluation run offline in CI and graded by an LLM judge on
  demand, plus retriever, chunking, and tool-availability ablations;
- CI on every pull request and push before deployment, including a start-up
  smoke test of the deployed process layout.

AI output was treated as a draft rather than authoritative HR guidance. The
authors remain responsible for the correctness, security, and integrity of the
submission.
