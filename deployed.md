# Deployment and operations

## Deployed application

| Item | URL |
| --- | --- |
| Application (Streamlit chat UI) | `https://<render-service-name>.onrender.com` |
| Public health endpoint | `https://<render-service-name>.onrender.com/_stcore/health` |
| API health endpoint (internal) | `http://127.0.0.1:8000/health` inside the service |

Replace the placeholders above with the verified Render URL after the first
successful deployment, and copy the same URL into the README.

## Render architecture

`render.yaml` defines one free-tier Python web service. Its build installs the
locked dependencies, installs the project, and creates the policy index. That
step downloads the embedding model (about 66 MB) into `data/models/` at build
time, so the running service never downloads it. `python -m app.deploy` then
supervises:

- FastAPI on private loopback port `8000`;
- Streamlit on Render's public `PORT`;
- one long-lived FastMCP subprocess over stdio, opened when FastAPI starts and
  restarted automatically if tool discovery fails;
- the FTS5 and FAISS index files and synthetic JSON data.

### Memory

Render's free tier allows 512 MB. Measured locally on macOS after one chat
request, before a browser session:

| Process | Hybrid retrieval | `HR_RETRIEVAL_MODE=bm25` |
| --- | --- | --- |
| Supervisor (`app.deploy`) | 24 MB | 20 MB |
| FastAPI | 58 MB | 55 MB |
| MCP server | 287 MB | 51 MB |
| Streamlit (idle) | 32 MB | 43 MB |
| Total | about 400 MB | about 170 MB |

The embedding model accounts for almost all of the difference, and each browser
session adds Streamlit memory. If Render reports out-of-memory restarts, set
`HR_RETRIEVAL_MODE=bm25` in the service environment. Search then uses keywords
only, with the quality cost shown in the ablation in `design-and-evaluation.md`.
`/health` reports the active `retrieval_mode`.

The agent calls OpenRouter for intent classification and answer wording when
`OPENROUTER_API_KEY` is set. Without the key, the service still runs and answers
with deterministic, cited fallback wording.

The public health endpoint is `/_stcore/health`. FastAPI's `/health` remains
available on the internal loopback interface. The public UI reaches it through
`HR_API_URL`, which the supervisor sets automatically.

## One-time provisioning

1. Push this repository to GitHub.
2. In Render, create a Blueprint from the repository and apply `render.yaml`.
3. When Render prompts for `OPENROUTER_API_KEY`, paste the key. It is declared
   with `sync: false`, so the value lives only in Render's secret store.
4. In the Render service settings, create a deploy hook.
5. In GitHub, create a protected `production` environment.
6. Add environment/repository secrets:
   - `OPENROUTER_API_KEY` (optional): enables the manual and weekly graded
     evaluation job.
   - `RENDER_DEPLOY_HOOK_URL`: the private Render deploy-hook URL.
   - `DEPLOYED_APP_URL`: the public service origin, without a trailing path.
7. Require the `test` job on `main` through branch protection.
8. Push a reviewed commit to `main`, then verify the CI artifact and deployed
   health check.
9. Replace the placeholder URLs above and in the README with the verified URL
   and commit them.

Do not expose the deploy-hook URL or the OpenRouter key in documentation, logs,
issue comments, or committed files.

## Deployment gate

`.github/workflows/ci.yml` runs on pull requests and pushes. The `test` job
installs the locked Python 3.12 environment, restores the cached embedding
model, builds the index, and runs:

1. all tests;
2. the offline gold evaluation (at least 90% of tasks must pass), the ablation,
   and HTTP latency, all needing no API key;
3. a start-up smoke test. It launches `python -m app.deploy` exactly as Render
   does, then requires Streamlit's `/_stcore/health`, an `ok` API `/health`
   with the MCP session connected and the index loaded, and a cited `/chat`
   answer.

Evaluation reports are uploaded as an artifact. Only a successful `test` job on
a push to `main` can enter the `deploy` job, which triggers Render and polls
`/_stcore/health` for up to ten minutes.

The LLM-graded evaluation is a separate `graded-evaluation` job. It runs on
manual dispatch and weekly, only when `OPENROUTER_API_KEY` is configured, and
does not gate deployment. A free-tier provider limit therefore cannot block a
release.

When the Render deployment secrets are absent, the deploy job reports a notice
and skips the deploy steps rather than deploying an untested build. Render
automatic deploys are disabled so a source push cannot race ahead of CI.

## Cold starts and persistence

Free Render services sleep after about 15 minutes without traffic. The first
request after a sleep can take 30–60 seconds while Render restarts the service,
FastAPI starts the MCP session, and Streamlit loads. The first search after
start-up also loads the embedding model (about 0.4 s locally). During the
demo, open the application a minute beforehand to wake it. Once warm, `/chat`
takes about 16 ms p50 locally without a model; OpenRouter adds several seconds
per answer. See the latency section of `design-and-evaluation.md`.

The policy index is built into each deployment and can also rebuild
deterministically at runtime.

The filesystem is ephemeral. Confirmed mock tickets may disappear on restart or
redeploy, which is appropriate for a demo. Do not treat the ticket file as a
database or connect this service to real HR systems.

## Operational checks

```bash
curl --fail https://<service>.onrender.com/_stcore/health
python -m evaluation.runner --mode offline --skip-ablation --http https://<api-origin>
```

The second command times `/health` and `/chat` against any reachable API
origin. On Render the API listens only on loopback, so run it inside the
service shell with `http://127.0.0.1:8000`.

Then load the UI and run both demo presets. If a deployment is unhealthy:

1. inspect the failed GitHub job and Render build/runtime logs;
2. verify Python 3.12, the build command, `OPENROUTER_API_KEY`, and both
   deployment secrets;
3. redeploy the last known-good commit from Render or revert the source commit;
4. rerun the CI workflow and confirm the public health endpoint.

If OpenRouter is unavailable or rate-limited, the application keeps answering
with deterministic cited guidance and the trace records the fallback.
