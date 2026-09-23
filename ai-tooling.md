# AI tooling and provenance

## Assistance used

Cursor agents assisted with project planning, implementation, test generation,
fictional policy drafting, synthetic record generation, documentation, and
review. The repository history and this file are the disclosure record for that
assistance.

AI output was treated as a draft rather than as authoritative HR guidance.
Human review should verify code, claims, and presentation before release or
demonstration.

## Data provenance

`policies/manifest.json` identifies the policy corpus as fictional, AI-assisted
demonstration content. No external policy text was copied. The records under
`mock_data/` are generated examples, use reserved synthetic identifiers, and do
not represent real employees. Addresses use the reserved `example.invalid`
domain.

No prompt, source file, or runtime path requires real HR data. Contributors must
not add real employee records, credentials, proprietary policies, or provider
keys to the repository.

## Verification

AI-assisted changes are checked with:

- deterministic ingestion and retrieval tests;
- MCP discovery and real tool-call integration tests;
- API, workflow, clarification, failure, and action-confirmation tests;
- a 25-task gold evaluation and retrieval ablation;
- CI on every pull request and push before deployment.

Generated evaluation reports are runtime artifacts rather than hand-edited
evidence. CI uploads them with the commit SHA.

## Runtime model use

The current application does not call a hosted or local LLM. The “agentic”
behavior is an explicit state machine using validated schemas, deterministic
retrieval, and real MCP calls. This choice keeps the demo reproducible and
provider-independent while making the lack of free-form language understanding
an explicit limitation.

If an OpenAI-compatible provider is added later, its base URL, model, and key
must be environment-configured; secrets must never enter `.env.example`, logs,
traces, fixtures, or commits. New model-generated claims must remain grounded
in returned policy evidence and must pass the same action-confirmation boundary.
