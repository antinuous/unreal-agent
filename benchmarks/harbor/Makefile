.PHONY: sync check test build smoke

REVISION ?= HEAD
RUNNER ?= unreal-agent-runner
ARCH ?= amd64
BUNDLE ?= ../../bin/harbor/$(shell git rev-parse --short $(REVISION))

sync:
	uv sync --locked --all-packages --all-extras

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked pyrefly check

test:
	uv run --locked pytest

build:
	uv run --locked python build.py --revision "$(REVISION)" --runner "$(RUNNER)" --arch "$(ARCH)" --output "$(BUNDLE)"

smoke:
	uv run --locked python tests/smoke.py --bundle $(BUNDLE)
