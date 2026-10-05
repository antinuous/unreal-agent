# Harbor evaluation adapter

Evaluates `unreal-agent-runner` as `unreal-agent` with Harbor 0.22.0. Requires uv
0.12 or later and Docker or Modal. The adapter is a member of the root uv workspace.

## Setup and build

From the repository root:

```sh
make -C benchmarks/harbor sync
make -C benchmarks/harbor build REVISION=HEAD ARCH=amd64
```

The build exports the committed revision with `git archive`, builds the
`unreal-agent` wheel, and installs it with its locked dependencies for
`manylinux_2_28` into a relocatable CPython 3.14 from python-build-standalone,
fetched with `uv python install`. It needs no Docker or emulation. `ARCH` is
`amd64` (default) or `arm64`; match the task containers.

The output directory `bin/harbor/<short-commit>/` holds:

- `unreal-agent-runner.tar.gz`, which contains `python/` and the
  `unreal-agent-runner` launcher.
- `manifest.json` with `revision`, `runner`, the archive `sha256`, `os`, `arch`,
  and `python_version`.

Existing bundles are not overwritten; set `BUNDLE` to a new directory to rebuild.
The agent verifies the checksum when it loads the bundle and again before upload.

The bundle runs in glibc 2.28+ Linux containers without a system Python or CA
certificates. It needs `/bin/sh`, `tar`, and `gzip` in the container. Musl
images such as Alpine are not supported. The launcher runs Python in isolated
mode (`-I`), so task `PYTHON*` variables are ignored, and it passes the caller's
environment to tool processes unchanged. TLS uses the bundled `certifi`
certificates unless `SSL_CERT_FILE` or `SSL_CERT_DIR` is set.

## Run

Use an absolute bundle path:

```sh
export OPENAI_API_KEY=...
uv run --project benchmarks/harbor --locked harbor run \
  -p /absolute/path/to/task \
  -a harness_harbor.agent:UnrealAgent \
  -m openai/gpt-6-astra \
  --ak bundle="$PWD/bin/harbor/<short-commit>" \
  --ak thinking_level=high
```

`thinking_level`: `low`, `medium`, `high` (default), `xhigh`, or `max`.
Also supports `anthropic/<model>` with `ANTHROPIC_API_KEY`,
`openrouter/<model>` with `OPENROUTER_API_KEY`, and
`fireworks_ai/<model>` with `FIREWORKS_AI_API_KEY`. Use Harbor's `--ae` option for
environment overrides.

For Terminal-Bench 4.0 on Modal, configure Modal credentials and run:

```sh
uv run --project benchmarks/harbor --locked --extra modal harbor run \
  -d terminal-bench/terminal-bench@4.0.0 \
  -e modal -a harness_harbor.agent:UnrealAgent \
  -m openai/gpt-6-astra \
  --ak bundle="$PWD/bin/harbor/<short-commit>" --ak thinking_level=max \
  -k 5 -n 40 --job-name tb4-unreal-agent
```

Inspect results with `harbor view jobs/<job-name>`.

## Output

- Bash and ViewImage tools.
- ViewImage returns images within 2000×2000 pixels and 5 MB minus 1 KB of base64
  content. Trajectories reference the returned images under `agent/images/` and
  include original format and coordinate-scaling metadata when applicable.
- Logs and sessions are under `agent/`; `agent/trajectory.json` contains ATIF
  observations and token totals.
- Trial metadata records `revision`, `bundle_sha256`, `arch`, `python_version`,
  and `thinking_level`.
- Observations are grouped by tool call. Their `extra` sequence, timestamp, and
  `available_before_turn` fields preserve asynchronous timing.

## Validation

```sh
make -C benchmarks/harbor sync check test
make -C benchmarks/harbor build ARCH=arm64 BUNDLE=../../bin/harbor/<short-commit>-arm64
make -C benchmarks/harbor smoke BUNDLE=../../bin/harbor/<short-commit>-arm64
```

`check` runs ruff and pyrefly; `test` runs pytest. The smoke test runs a Harbor
Docker trial against a local test model. Build the bundle for the Docker host
architecture (`docker info --format '{{.Architecture}}'`). The task image sets an
empty `LANG` and a hostile `PYTHONPATH`, and the test checks that tool processes
see neither `LC_CTYPE` nor `LD_LIBRARY_PATH`.

Reference: [Harbor agents](https://www.harborframework.com/docs/agents).
