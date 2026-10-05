# Agent instructions

The Go implementation is being ported to Python under `packages/unreal-agent`.
Go code stays in place as the reference until cutover, then it is deleted.

## Commands

Run from the repository root.

```sh
uv sync                              # install the workspace
uv run pytest                        # tests
uv run ruff format .                 # format
uv run ruff check .                  # lint
uv run pyrefly check                 # type-check
go test ./harness/<pkg>/...          # run the Go reference tests (until cutover)
```

Before committing, run `ruff format --check`, `ruff check`, and `pyrefly check`
on the paths you changed, plus the relevant tests.

## Layout

The Python package mirrors the Go packages:

| Go | Python |
| --- | --- |
| `harness/session` | `unreal_agent.session` |
| `harness/inbox` | `unreal_agent.inbox` |
| `harness/primitives` | `unreal_agent.primitives` |
| `harness/operation` | `unreal_agent.operation` |
| `harness/llm` | `unreal_agent.llm` |
| `harness/llm/responsesapi` (+ `internal/openaiapi`) | `unreal_agent.llm.responsesapi` |
| `harness/llm/clients/*` | `unreal_agent.llm.clients.*` |
| `harness/contextbuilder` | `unreal_agent.contextbuilder` |
| `harness/tool`, `tool/bash`, `tool/viewimage` | `unreal_agent.tool`, `unreal_agent.tool.bash`, `unreal_agent.tool.viewimage` |
| `harness/sessionstore`, `sessionstore/localfile` | `unreal_agent.sessionstore`, `unreal_agent.sessionstore.localfile` |
| `harness/coordinator` | `unreal_agent.coordinator` |
| `cmd/internal/agentrunner`, `cmd/unreal-agent-runner` | `unreal_agent.runner` |

Tests live in `packages/unreal-agent/tests/<package>/test_*.py`. Test fixtures
are copied next to the tests that use them.

## Porting conventions

- Preserve behavior, persisted formats, error semantics, and user-visible
  strings. Tests that pin a message keep the same text.
- Serialized Go structs subclass `unreal_agent.gojson.Struct`. Keep Go field
  names and declaration order. Every field has a default. Mirror
  `json:",omitzero"` with `Annotated[T, OmitZero]` and `omitempty` with
  `OmitEmpty`. `jsontext.Value` is `gojson.Raw`, `time.Time` is
  `gojson.Timestamp`, `[]byte` is `bytes`. Encode with `gojson.encode` and
  decode with `gojson.decode`; never round-trip raw JSON through
  `json.loads`/`json.dumps`.
- Go `any` fields selected by a sibling kind tag get an explicit codec that
  decodes the envelope with `Raw` and then decodes by kind.
- String enums are `StrEnum`. If Go can persist an enum field's zero value,
  give the enum a `NONE = ""` member so decoding accepts it. Internal, non-serialized state uses
  `@dataclass(slots=True)`.
- Sentinel errors become exception classes; `errors.Is` becomes `isinstance`.
  `errors.Join` becomes an `ExceptionGroup` or a joined message, matching what
  callers inspect.
- `context.Context` maps to asyncio task cancellation. Goroutines become
  tasks, channels become `asyncio.Queue` or explicit async iterators, and
  `select` loops become one task waiting on an event or queue. Blocking file
  and CPU work runs in `asyncio.to_thread`.
- Tests built on `testing/synctest` use `@pytest.mark.looptime`. Avoid
  threads inside fake-time tests because `looptime` advances time whenever
  the loop is idle.
- Fuzz targets become Hypothesis tests. Go seed corpora become `@example`s.
- Go map iteration is random. Do not assert an order the Go code does not
  guarantee.
- Comments only for constraints, architecture decisions, or tradeoffs.

## Commits

Commit messages are one lowercase line prefixed with `feat:`, `fix:`, or
`chore:`. Stage only the paths you own.
