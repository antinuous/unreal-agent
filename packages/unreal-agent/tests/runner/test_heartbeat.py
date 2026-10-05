"""Tool heartbeat flag tests."""

import asyncio
import io
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from unreal_agent import gojson
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD
from unreal_agent.inbox import ControlMode, Input, InputKind
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Response, Role, ToolCall, ToolResult
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.runner import Config, run_main
from unreal_agent.runner.run import LLM_API_KEY_ENVIRONMENT

from tests.runner._helpers import (
    FakeClient,
    call_run_main,
    decode_lines,
    environment,
    parse_test_request,
    reader,
    runner_config,
)

HEARTBEAT_PREFIX = "Heartbeat: waited 0.01 seconds for tool calls.\nRunning: "


def heartbeat_message_count(request: LLMRequest) -> int:
    return sum(
        1
        for item in request.Input
        if isinstance(item.Data, Message)
        and item.Data.Role == Role.USER
        and item.Data.Text.startswith(HEARTBEAT_PREFIX)
    )


@pytest.fixture
def release_pipe(tmp_path: Path) -> Iterator[int]:
    path = tmp_path / "release"
    os.mkfifo(path, 0o600)
    descriptor = os.open(path, os.O_RDWR)
    yield descriptor
    os.close(descriptor)


async def test_run_main_heartbeat_releases_waiting_bash_and_replays(
    tmp_path: Path, release_pipe: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workspace = str(tmp_path)
    sessions = str(tmp_path_factory.mktemp("sessions"))
    started = released = finished = False
    waiting_call = ToolCall(
        CallID="waiting-call",
        Name="Bash",
        Arguments='{"command":"read value < release; printf \'%s\' \\"$value\\""}',
    )

    def respond(request: LLMRequest) -> Response:
        nonlocal started, released, finished
        if not started:
            started = True
            return Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=waiting_call)])
        for item in request.Input:
            result = item.Data
            if not isinstance(result, ToolResult):
                continue
            value = result.Output[0].Value
            if result.CallID != "waiting-call" or value == TOOL_CALL_RUNNING_PAYLOAD:
                continue
            if value != "released":
                raise ValueError(f"unexpected Bash result: {value}")
            finished = True
            return Response()
        if heartbeat_message_count(request) == 0:
            raise ValueError("waiting turn has no heartbeat")
        if not released:
            os.write(release_pipe, b"released\n")
            released = True
        return Response()

    getenv = environment({LLM_API_KEY_ENVIRONMENT: "secret", "SHELL": "/bin/sh"})
    args = [
        "-workspace",
        workspace,
        "-session-directory",
        sessions,
        "-tool-heartbeat-interval",
        "10ms",
    ]
    async with asyncio.timeout(10):
        code, stdout, stderr = await call_run_main(
            args,
            getenv,
            reader('{"prompt":"run it","session_id":"heartbeat-session"}'),
            runner_config(FakeClient(respond)),
        )
        assert code == 0, stderr
        assert released, "heartbeat did not release Bash and deliver its result"
        assert finished, "heartbeat did not release Bash and deliver its result"
        heartbeats = 0
        for item in decode_lines(stdout):
            if not isinstance(item.Data, Input) or item.Data.Kind != InputKind.CONTROL:
                continue
            control = item.Data.decode_control_message()
            if control.Mode != ControlMode.HEARTBEAT:
                continue
            assert control.Reason.startswith(HEARTBEAT_PREFIX), control.Reason
            calls = gojson.decode(control.Reason.removeprefix(HEARTBEAT_PREFIX), list[ToolCall])
            assert calls == [waiting_call]
            heartbeats += 1
        assert heartbeats, "heartbeat was not recorded in the session log"

        def resumed(request: LLMRequest) -> Response:
            got = heartbeat_message_count(request)
            if got != heartbeats:
                raise ValueError(f"replayed heartbeat messages = {got}, want {heartbeats}")
            return Response()

        code, _, stderr = await call_run_main(
            args,
            getenv,
            reader('{"prompt":"summarize","session_id":"heartbeat-session"}'),
            runner_config(FakeClient(resumed)),
        )
        assert code == 0, stderr


@pytest.mark.parametrize("interval", ["-1s", "invalid", "999999999999999999999s"])
async def test_run_main_rejects_invalid_heartbeat_interval(interval: str) -> None:
    stderr = io.BytesIO()
    code = await run_main(
        ["-tool-heartbeat-interval", interval],
        lambda _: "",
        reader('{"prompt":"hello"}'),
        io.BytesIO(),
        stderr,
        Config(name="unreal-agent-runner", parse_request=parse_test_request),
    )
    assert code == 1, stderr.getvalue()
    assert b"heartbeat" in stderr.getvalue(), stderr.getvalue()
