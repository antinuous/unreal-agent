"""Cleanup and process-exit behavior that Go gets from defers and its runtime."""

import asyncio
import io
import os
import signal
import subprocess
import sys

import pytest
from unreal_agent.llm import Response
from unreal_agent.runner import Reader, Request, ToolConfig, ToolFactory, Tools, decode_request
from unreal_agent.runner.run import run_main
from unreal_agent.tool import Registry

from tests.runner._helpers import (
    SECRET_ENVIRONMENT,
    FakeClient,
    message_response,
    reader,
    runner_config,
)

_PROBE = "UNREAL_RUNNER_CLEANUP_PROBE"


@pytest.mark.parametrize(("fail", "want"), [(False, 0), (True, 130)], ids=["success", "failure"])
async def test_run_finishes_cleanup_when_cancelled_during_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    fail: bool,
    want: int,
) -> None:
    """Go runs every deferred cleanup even when the context is cancelled during one.

    The exit code comes from the run's result, so a successful run still exits 0.
    """

    def respond(_: object) -> Response:
        if fail:
            raise RuntimeError("model failed")
        return message_response("done")

    monkeypatch.delenv(_PROBE, raising=False)
    workspace = tmp_path_factory.mktemp("workspace")
    (workspace / ".env").write_text(f"{_PROBE}=inner\n")
    client = FakeClient(respond)
    closing, release = asyncio.Event(), asyncio.Event()
    tools_closed = False

    def parse_request(input: Reader) -> tuple[Request, ToolFactory]:
        parsed = decode_request(input)

        async def new_tools(config: ToolConfig) -> Tools:
            async def close() -> None:
                nonlocal tools_closed
                closing.set()
                await release.wait()
                tools_closed = True

            return Tools(
                registry=Registry(config.translators, *parsed.enabled_tools(*config.names)),
                close=close,
            )

        return parsed, new_tools

    config = runner_config(client)
    config.parse_request = parse_request
    async with asyncio.timeout(10):
        task = asyncio.create_task(
            run_main(
                [
                    "-workspace",
                    str(workspace),
                    "-session-directory",
                    str(tmp_path_factory.mktemp("sessions")),
                ],
                SECRET_ENVIRONMENT,
                reader('{"prompt":"hello"}'),
                io.BytesIO(),
                io.BytesIO(),
                config,
            )
        )
        await closing.wait()
        assert os.environ.get(_PROBE) == "inner"
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        assert await task == want
    assert tools_closed, "tool cleanup was interrupted"
    assert client.closed, "client was not closed"
    assert _PROBE not in os.environ, ".env changes were not restored"


def test_main_dies_of_sigpipe_when_stdout_is_closed() -> None:
    """Go's runtime raises SIGPIPE for a write to a closed pipe on fd 1 or 2."""
    read, write = os.pipe()
    os.close(read)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unreal_agent.runner", "-nope"],
            stdin=subprocess.DEVNULL,
            stdout=write,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    finally:
        os.close(write)
    assert result.returncode == -signal.SIGPIPE, result.stderr.decode()
    assert b"Exception ignored" not in result.stderr
    assert not result.stderr.endswith(b"broken pipe\n")
