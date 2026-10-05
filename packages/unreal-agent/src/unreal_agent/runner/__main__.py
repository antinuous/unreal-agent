"""unreal-agent-runner: execute one JSON request and write persisted session items as JSONL.

SIGINT cancels the run (exit 130). SIGTERM keeps the default action and
terminates the process immediately, as in the Go command.
"""

import asyncio
import os
import signal
import sys

from unreal_agent.runner.config import Config, ToolConfig, ToolFactory, Tools
from unreal_agent.runner.providers import default_providers
from unreal_agent.runner.request import Request, decode_request
from unreal_agent.runner.run import run_main
from unreal_agent.runner.streams import Reader
from unreal_agent.tool import new_registry

__all__ = ["main", "parse_request"]

NAME = "unreal-agent-runner"


def parse_request(input: Reader) -> tuple[Request, ToolFactory]:
    """Decode a request and configure only the static tools it does not disallow."""
    parsed = decode_request(input)

    async def new_tools(config: ToolConfig) -> Tools:
        return Tools(
            registry=new_registry(config.translators, *parsed.enabled_tools(*config.names))
        )

    return parsed, new_tools


def _getenv(name: str) -> str:
    return os.environ.get(name, "")


async def _main(args: list[str]) -> int:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    interrupted = False

    def interrupt() -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            task.cancel()

    loop.add_signal_handler(signal.SIGINT, interrupt)
    try:
        return await run_main(
            args,
            _getenv,
            sys.stdin.buffer,
            sys.stdout.buffer,
            sys.stderr.buffer,
            Config(name=NAME, parse_request=parse_request, providers=default_providers()),
        )
    finally:
        loop.remove_signal_handler(signal.SIGINT)


def main() -> None:
    sys.exit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
