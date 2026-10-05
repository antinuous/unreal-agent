.PHONY: build test check py-test py-check

build:
	go build -trimpath -o bin/unreal-agent-runner ./cmd/unreal-agent-runner
	go -C cmd/unreal-agent-tui build -trimpath -o ../../bin/unreal-agent-tui .

test:
	go test -race ./...
	go -C cmd/unreal-agent-tui test -race ./...

check:
	go vet ./...
	go -C cmd/unreal-agent-tui vet ./...
	@test -z "$$(gofmt -l cmd harness internal)" || { gofmt -l cmd harness internal; exit 1; }

py-test:
	uv run pytest

py-check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyrefly check
