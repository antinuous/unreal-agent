# Contributing

This repository contains selected components from a larger internal codebase.

We intend to share more components, features, and tools over time, as we evaluate their reliability, token usage, and impact on agent behavior.

We welcome GitHub issues with your questions, feature requests, and bug reports.

Unfortunately, we don't currently have the capacity to review and merge pull requests.

## Development

Install [uv](https://docs.astral.sh/uv/). It provides Python 3.14 when the
system has none. Run these from the repository root:

```sh
make sync     # uv sync --locked
make check    # ruff format --check, ruff check, pyrefly check
make test     # pytest
make build    # wheel and sdist in dist/
make fuzz     # property tests with the heavier Hypothesis profile
make format   # ruff format and ruff check --fix
```

`make fuzz` scales each property test's `max_examples` and draws random inputs.
Failing examples are stored in `.hypothesis/` and replay on the next run. Set
`HYPOTHESIS_PROFILE=fuzz-ci` for the shorter CI profile, and pass test paths with
`FUZZ_ARGS`.

See [AGENTS.md](AGENTS.md) for the Harbor and Docker commands, the package
layout, and coding conventions.
