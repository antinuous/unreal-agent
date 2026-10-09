"""unreal-agent-runner: execute one JSON request and write persisted session items as JSONL.

SIGINT and SIGTERM cancel the run, which exits 130 after cleanup. A write to a
closed pipe on stdout or stderr kills the process with SIGPIPE.
"""

import asyncio
import os
import signal
import sys
from typing import BinaryIO, Final, NoReturn

from unreal_agent.runner.config import Config, ToolConfig, ToolFactory, Tools
from unreal_agent.runner.providers import default_providers
from unreal_agent.runner.request import Request, decode_request
from unreal_agent.runner.run import run_main
from unreal_agent.runner.streams import Reader
from unreal_agent.tool import Registry

__all__ = ["main", "parse_request"]

NAME = "unreal-agent-runner"


def parse_request(input: Reader) -> tuple[Request, ToolFactory]:
    """Decode a request and configure only the static tools it does not disallow."""
    parsed = decode_request(input)

    async def new_tools(config: ToolConfig) -> Tools:
        return Tools(registry=Registry(config.translators, *parsed.enabled_tools(*config.names)))

    return parsed, new_tools


class _StandardStream:
    __slots__ = ("_stream",)

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def write(self, data: bytes, /) -> int:
        try:
            return self._stream.write(data)
        except BrokenPipeError:
            _die_of_sigpipe()

    def flush(self) -> None:
        try:
            self._stream.flush()
        except BrokenPipeError:
            _die_of_sigpipe()


def _die_of_sigpipe() -> NoReturn:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGPIPE)
    os._exit(128 + signal.SIGPIPE)


def _getenv(name: str) -> str:
    return os.environ.get(name, "")


_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


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

    for signum in _SIGNALS:
        loop.add_signal_handler(signum, interrupt)
    try:
        return await run_main(
            args,
            _getenv,
            sys.stdin.buffer,
            _StandardStream(sys.stdout.buffer),
            _StandardStream(sys.stderr.buffer),
            Config(name=NAME, parse_request=parse_request, providers=default_providers()),
        )
    finally:
        for signum in _SIGNALS:
            loop.remove_signal_handler(signum)


def main() -> None:
    code = asyncio.run(_main(sys.argv[1:]))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BrokenPipeError:
            _die_of_sigpipe()
    # Like Go's os.Exit: a stdin read abandoned by cancellation must not block
    # or crash interpreter shutdown.
    os._exit(code)


if __name__ == "__main__":
    main()
