# Unreal Agent

An async-first agent harness from Unreal Labs.

- [packages/unreal-agent/](packages/unreal-agent/): the `unreal-agent` Python
  package. It contains the library and the `unreal-agent-runner` command.
- [benchmarks/](benchmarks/): benchmark runners.

## Glossary

- **Input**: an event with a caller-supplied globally unique ID that remains
  stable across redeliveries.
- **Inbox**: session-scoped, in-memory deduplication of external, control, and
  crash inputs.
- **Session**: append-only persisted history that can be forked.
- **LLM turn**: the coordinator-managed sequence around one logical LLM request.
- **Tool**: a capability described by a schema and bound to a translator.
- **Tool call**: a model-produced request to use a tool.
- **Tool translator**: validates a tool call and translates it into one or more
  operations. It runs synchronously on the coordinator's event loop and must not
  perform I/O or suspend the loop.
- **Tool call status**: the translation outcome: a validation error or references
  to submitted operations. Operation execution state is tracked separately;
  the translator formats these into a model-facing result.
- **Operation**: a serializable description of work produced by a tool translator
  for asynchronous execution. Implementations are encouraged to use the available
  [primitives](packages/unreal-agent/src/unreal_agent/primitives/).

## Components

| Component | Responsibility | Module |
| --- | --- | --- |
| Session inbox | Volatile, session-scoped input idempotency. | [`unreal_agent.inbox`](packages/unreal-agent/src/unreal_agent/inbox/) |
| Coordinator | Persist accepted inputs, run LLM turns, resolve tool translators through the registry, and dispatch committed operations. | [`unreal_agent.coordinator`](packages/unreal-agent/src/unreal_agent/coordinator/) |
| Session store | Persist canonical session history and operation state; support recovery and forks; atomically record tool-call status with operations. | [`unreal_agent.sessionstore`](packages/unreal-agent/src/unreal_agent/sessionstore/) |
| Context builder | Statefully assemble model input in memory. Return the model input together with a record of anything omitted, truncated, or compacted. Perform no I/O and accept no persistence dependencies. | [`unreal_agent.contextbuilder`](packages/unreal-agent/src/unreal_agent/contextbuilder/) |
| LLM Adapter | Send prepared model input to a provider and return a normalized completed response. Own authentication, cancellation, and provider errors. | [`unreal_agent.llm`](packages/unreal-agent/src/unreal_agent/llm/) |
| Tool registry | Own the fixed Bash, ViewImage, and skill-use definitions and their translators; expose the host-selected set. | [`unreal_agent.tool`](packages/unreal-agent/src/unreal_agent/tool/) |
| Tool translator | Validate a tool call and produce its status and operations. Format a recorded call status and prepared operation output into model results. Perform no I/O. | [`unreal_agent.tool`](packages/unreal-agent/src/unreal_agent/tool/) |
| Operation manager | Actor runtime for durable operations. The local implementation is swappable. | [`unreal_agent.operation`](packages/unreal-agent/src/unreal_agent/operation/) |

Other modules: [`unreal_agent.session`](packages/unreal-agent/src/unreal_agent/session.py)
(session items and inputs), [`unreal_agent.llm.responsesapi`](packages/unreal-agent/src/unreal_agent/llm/responsesapi/)
and [`unreal_agent.llm.clients`](packages/unreal-agent/src/unreal_agent/llm/clients/)
(provider clients), and [`unreal_agent.runner`](packages/unreal-agent/src/unreal_agent/runner/)
(the command).

## unreal-agent-runner

Run an AI agent from a prompt or JSON request. It writes session events to
stdout as JSONL and exits when the task finishes. It requires Python 3.14 and
[uv](https://docs.astral.sh/uv/).

### Install

Install the command from a checkout or from Git:

```sh
uv tool install ./packages/unreal-agent
uv tool install 'git+https://github.com/unreallabsai/unreal-agent#subdirectory=packages/unreal-agent'
```

Run it once without installing:

```sh
uvx --from 'git+https://github.com/unreallabsai/unreal-agent#subdirectory=packages/unreal-agent' \
  unreal-agent-runner -p 'Summarize this project.'
```

From a checkout, `uv run` uses the locked dependencies:

```sh
uv sync
uv run unreal-agent-runner -p 'Inspect this project and explain how to run its tests.'
```

`uv tool install` and `uvx` resolve dependencies from the package metadata and do
not read `uv.lock`. Release wheels are attached to each
[GitHub release](https://github.com/unreallabsai/unreal-agent/releases) and can be
installed with `uv tool install ./unreal_agent-<version>-py3-none-any.whl`.

### Usage

Set an OpenAI API key and run a prompt in the current directory:

```sh
export OPENAI_API_KEY="..."
unreal-agent-runner -p 'Inspect this project and explain how to run its tests.'
```

Choose a workspace and save the output:

```sh
unreal-agent-runner -workspace ./my-project -p 'Summarize this project.' > run.jsonl
```

Pass a JSON request as the one positional argument or through stdin:

```sh
unreal-agent-runner '{"prompt":"Summarize this project."}'
unreal-agent-runner < request.json
```

Place options before the positional request. `-p` and a positional request
cannot be combined. Run `unreal-agent-runner -h` for the options and the request
fields.

### Options

Options use Go flag syntax: one or two dashes, and `-name value` or
`-name=value`.

| Option | Default | Description |
| --- | --- | --- |
| `-p prompt` | | Send a request with the given prompt without reading stdin. |
| `-workspace dir` | `.` | Agent workspace and Bash working directory. |
| `-session-directory dir` | `$XDG_STATE_HOME/unreal-agent/sessions`, or `$HOME/.local/state/unreal-agent/sessions` | Directory containing session files. |
| `-log-directory dir` | unset | Also append each event to `<dir>/YYYYMMDD-HHMMSS.jsonl` (UTC). |
| `-tool-heartbeat-interval duration` | `10m0s` | Tool-wait heartbeat interval, as a Go duration such as `30s`. `0` disables it. |
| `-h`, `-help` | | Print usage and the request schema. |

### Request

A JSON object. Unknown fields are rejected.

| Field | Type | Description |
| --- | --- | --- |
| `messages` | array of `{role, content, message_id?}` | Non-empty list of user messages delivered in order. `role` must be `"user"` and defaults to it. `message_id` is a UUID and defaults to a generated one. Takes precedence over `prompt`. |
| `prompt` | string | One user message. Used when `messages` is absent. |
| `model` | string | Provider model ID. Defaults to `UNREAL_HARNESS_LLM_MODEL` or the provider default. |
| `max_attempts` | positive integer | LLM request attempts. Overrides `UNREAL_HARNESS_LLM_MAX_ATTEMPTS` (default 5). `1` disables retries. |
| `system_prompt` | string | Replaces the default system prompt. |
| `thinking_level` | `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` | Reasoning effort. Defaults to `"high"`. |
| `session_id` | non-empty string | Creates or resumes a persisted session. Defaults to a new UUID. |
| `disallowed_tools` | array of strings | Tool names excluded from model context and execution. |
| `extra_allowed_tools` | array of strings | Accepted and ignored. |
| `include_partial_messages` | boolean | Accepted and ignored. |

Supply `messages` or `prompt`.

### Providers and environment

| Variable | Description |
| --- | --- |
| `UNREAL_HARNESS_LLM_PROVIDER` | `openai` (default), `openai-codex`, `openrouter`, `fireworks`, or `ollama`. |
| `UNREAL_HARNESS_LLM_MODEL` | Model when the request has none. Only `openai` has a default (`gpt-6-astra`). |
| `UNREAL_HARNESS_LLM_BASE_URL` | Overrides the provider's base URL. |
| `UNREAL_HARNESS_LLM_API_KEY` | API key for any provider that takes one. Wins over the provider variable. |
| `UNREAL_HARNESS_LLM_MAX_ATTEMPTS` | LLM request attempts when the request has no `max_attempts`. |
| `XDG_STATE_HOME`, `HOME` | Locate the default session directory. |
| `SHELL` | Shell for the Bash tool. Defaults to `/bin/sh`. |

| Provider | Credentials | Base URL |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `openai-codex` | `OPENAI_CODEX_ACCESS_TOKEN` and `OPENAI_CODEX_ACCOUNT_ID`, or `OPENAI_CODEX_AUTH_FILE`, or `$CODEX_HOME/auth.json` (default `~/.codex/auth.json`) | `https://chatgpt.com/backend-api/codex` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `fireworks` | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` |
| `ollama` | none | `http://localhost:11434/v1` |

The runner loads `<workspace>/.env` (`NAME=value` lines) for the run. Variables
already set in the process environment win, except `SANDBOX_EGRESS_PROXY`,
which always applies and is copied to `HTTPS_PROXY`. Skills are discovered from
`<workspace>/.harness/skills`.

### Output and exit status

Each persisted session item is one JSON line on stdout with `Sequence`,
`RecordedAt`, `Kind`, and `Data`. On failure the runner writes `{"type":"error","message":"..."}` to stdout
and `unreal-agent-runner: <error>` to stderr.

| Exit status | Meaning |
| --- | --- |
| `0` | The session finished, or help was printed. |
| `1` | The run failed. |
| `130` | Interrupted by SIGINT. |

### Docker

The `unrea1labs/unreal-agent` image supports Linux on AMD64 and ARM64. Run it
with a project mounted as the workspace:

```sh
docker run --rm -i --user "$(id -u):$(id -g)" \
  -e OPENAI_API_KEY -v "$PWD:/workspace" \
  -v unreal-agent-state:/state \
  unrea1labs/unreal-agent:latest -p 'Summarize this project.'
```

Each release also publishes its Git tag (for example, `v0.1.0`) for version
pinning. Build the image locally with `docker build -t unreal-agent .`.

## Extending the harness

Harness components are composable, and alternative implementations of their interfaces are encouraged.

We intend to preserve these invariants:

- Session-store items are serializable, and the storage format is versioned.
- We'll do our best to maintain backwards compatibility for sessions.
  An unsupported session version will always cause an explicit error on resume.
- Operations are versioned and always serializable.

For example, a proxy operations manager can send serialized operations to a
local operations manager running in a process inside a remote sandbox, allowing
tools to execute there.
