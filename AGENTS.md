# Agent instructions

The harness is the `unreal-agent` Python package in `packages/unreal-agent`.
Its console script is `unreal-agent-runner`. The repository is a uv workspace
that requires Python 3.14.

## Commands

Run from the repository root.

```sh
uv sync                              # install the workspace
uv run pytest                        # tests
uv run ruff format .                 # format
uv run ruff check .                  # lint
uv run pyrefly check                 # type-check
make check test build                # CI also runs fuzz, Harbor, and Docker jobs
make fuzz                            # property tests with the heavier Hypothesis profile
make -C benchmarks/harbor test check build smoke BUNDLE=...   # Harbor adapter
docker build -t unreal-agent . && docker run --rm unreal-agent -h
```

Before committing, run `ruff format --check`, `ruff check`, and `pyrefly check`
on the paths you changed, plus the relevant tests.

## Layout

Modules live under `packages/unreal-agent/src/unreal_agent`:

| Module | Contents |
| --- | --- |
| `session` | Session items, inputs, and persisted kinds |
| `inbox` | Session-scoped input deduplication |
| `primitives` | File, process, channel, and compute primitives used by operations |
| `operation` | Operations and the local operation manager |
| `llm` | The adapter interface and normalized model types |
| `llm.responsesapi` | The OpenAI Responses API wire format, streaming, and retries |
| `llm.messagesapi` | The Anthropic Messages API wire format, stream assembly, and retries |
| `llm.clients.*` | Provider clients (anthropic, openai, openaicodex, openrouter, fireworks, ollama) |
| `contextbuilder` | Model input assembly, prompts, and skills |
| `tool`, `tool.bash`, `tool.viewimage` | Tool registry, static tools, and translators |
| `sessionstore`, `sessionstore.localfile` | Session store interface and the local file store |
| `settings` | Model settings: bundled defaults merged with user overrides |
| `coordinator` | The coordinator loop |
| `runner` | The `unreal-agent-runner` command |
| `gojson` | Go-compatible JSON encoding for persisted and wire formats |

Tests live in `packages/unreal-agent/tests/<package>/test_*.py`. Test fixtures
live in `testdata/` next to the tests that use them. Shared test helpers go in
`tests/<package>/_helpers.py` and are imported as `tests.<package>._helpers`.
`tests/conftest.py` registers the `fuzz` and `fuzz-ci` Hypothesis profiles,
selected with `HYPOTHESIS_PROFILE`.

## Conventions

The package began as a port of a Go implementation. Sessions, operations, and
model requests written by that implementation must keep working, so wire and
persisted formats stay byte-compatible with it.

- Preserve persisted formats, error semantics, and user-visible strings. Tests
  that pin a message keep the same text. Format changes need a version bump
  and an explicit error for unsupported versions.
- Serialized structs subclass `unreal_agent.gojson.Struct`. Keep the existing
  field names (Go-style `PascalCase`) and declaration order. Every field has a
  default. Use `Annotated[T, OmitZero]` for fields omitted when zero and
  `OmitEmpty` for fields omitted when empty. Use `gojson.Raw` for raw JSON
  values, `gojson.Timestamp` for timestamps, and `bytes` for binary data.
  Encode with `gojson.encode` and decode with `gojson.decode`; never
  round-trip raw JSON through `json.loads`/`json.dumps`.
- Polymorphic payloads selected by a sibling kind tag get an explicit codec
  that decodes the envelope with `Raw` and then decodes by kind.
- String enums are `StrEnum`. If an enum field can be persisted empty, give the
  enum a `NONE = ""` member so decoding accepts it. Internal, non-serialized
  state uses `@dataclass(slots=True)`.
- Errors that callers distinguish are exception classes checked with
  `isinstance`. Combined errors are an `ExceptionGroup` or a joined message,
  matching what callers inspect.
- Concurrency is asyncio. Cancellation is task cancellation, background work
  runs in tasks, and queues or async iterators carry messages between them.
  Blocking file and CPU work runs in `asyncio.to_thread`.
- Fake-time tests use `@pytest.mark.looptime`. Avoid threads inside them
  because `looptime` advances time whenever the loop is idle.
- Property tests use Hypothesis. Pin regression inputs with `@example`. Keep
  per-test settings derandomized for the default run; `make fuzz` scales them.
- Do not assert an iteration order that the code does not guarantee.
- Comments only for constraints, architecture decisions, or tradeoffs.

## Commits

Commit messages are one lowercase line prefixed with `feat:`, `fix:`, or
`chore:`. Stage only the paths you own.
