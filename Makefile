.PHONY: sync test check format build fuzz

HYPOTHESIS_PROFILE ?= fuzz

sync:
	uv sync --locked

test:
	uv run pytest

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyrefly check

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	uv build --package unreal-agent --out-dir dist

fuzz:
	HYPOTHESIS_PROFILE=$(HYPOTHESIS_PROFILE) uv run pytest -m hypothesis --timeout=0 $(FUZZ_ARGS)
