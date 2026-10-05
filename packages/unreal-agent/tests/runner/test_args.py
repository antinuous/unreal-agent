"""Command-line argument tests."""

import io

import pytest
from unreal_agent import gojson
from unreal_agent.llm import Request as LLMRequest
from unreal_agent.llm import Response, StopReason
from unreal_agent.runner import Config, Request, run_main
from unreal_agent.runner.streams import Reader

from tests.runner._helpers import (
    SECRET_ENVIRONMENT,
    ErrorReader,
    FakeClient,
    call_run_main,
    must_not_read_environment,
    parse_test_request,
    reader,
    runner_config,
    user_messages,
)


@pytest.mark.parametrize(
    ("args", "stdin", "want"),
    [
        pytest.param([], '{"prompt":"from stdin"}', ["from stdin"], id="stdin"),
        pytest.param(['{"prompt":"from argument"}'], "", ["from argument"], id="positional"),
        pytest.param(
            ['{"messages":[{"content":"first"},{"content":"second"}]}'],
            "",
            ["first", "second"],
            id="positional messages",
        ),
        pytest.param(["-p", "plain prompt"], "", ["plain prompt"], id="prompt"),
        pytest.param([], '{"prompt":""}', [""], id="empty prompt on stdin"),
        pytest.param(['{"prompt":""}'], "", [""], id="empty positional prompt"),
        pytest.param(["-p", ""], "", [""], id="empty prompt flag"),
        pytest.param(
            ["-p", '  say "hello"\nC:\\work\t世界  '],
            "",
            ['  say "hello"\nC:\\work\t世界  '],
            id="quoted multiline prompt",
        ),
        pytest.param(["-p", '{"messages":[]}'], "", ['{"messages":[]}'], id="JSON text as prompt"),
    ],
)
async def test_run_main_request_sources(
    args: list[str], stdin: str, want: list[str], tmp_path_factory: pytest.TempPathFactory
) -> None:
    called = False

    def respond(request: LLMRequest) -> Response:
        nonlocal called
        called = True
        messages = user_messages(request)
        if messages != want:
            raise ValueError(f"user messages = {messages!r}, want {want!r}")
        return Response(ID="done", Stop=StopReason.COMPLETE)

    client = FakeClient(respond)
    input: Reader = reader(stdin) if stdin else ErrorReader()
    workspace = str(tmp_path_factory.mktemp("workspace"))
    sessions = str(tmp_path_factory.mktemp("sessions"))
    code, _, stderr = await call_run_main(
        ["-workspace", workspace, "-session-directory", sessions, *args],
        SECRET_ENVIRONMENT,
        input,
        runner_config(client),
    )
    assert (code, called, client.closed) == (0, True, True), stderr


@pytest.mark.parametrize(
    ("args", "want"),
    [
        pytest.param(
            ["-p", "hello", '{"prompt":"world"}'], "-p cannot be combined", id="conflicting inputs"
        ),
        pytest.param(
            ["-p=", '{"prompt":"world"}'], "-p cannot be combined", id="empty prompt conflicts"
        ),
        pytest.param(
            ['{"prompt":"hello"}', '{"prompt":"world"}'],
            "at most one positional JSON request",
            id="extra arguments",
        ),
        pytest.param(["plain text"], "invalid JSON", id="invalid JSON"),
        pytest.param(
            ['{"prompt":"hello","unknown":true}'], "unknown object member", id="unknown field"
        ),
        pytest.param([""], "empty input", id="empty argument"),
        pytest.param(['{"messages":[]}'], "messages must not be empty", id="invalid request"),
    ],
)
async def test_run_main_rejects_invalid_request_arguments(args: list[str], want: str) -> None:
    code, stdout, stderr = await call_run_main(
        args,
        must_not_read_environment,
        ErrorReader(),
        Config(name="test-runner", parse_request=parse_test_request),
    )
    assert code == 1, stderr
    assert want in stderr, stderr
    event = gojson.validate(stdout)
    assert event["type"] == "error", event
    assert want in event["message"], event


class ClosedPipeError(Exception):
    """Go's io.ErrClosedPipe."""

    def __init__(self) -> None:
        super().__init__("io: read/write on closed pipe")


class ClosedPipe:
    def write(self, data: bytes, /) -> int:
        raise ClosedPipeError

    def flush(self) -> None:
        pass


@pytest.mark.parametrize(
    ("args", "want"),
    [
        pytest.param(["-unknown"], "flag provided but not defined: -unknown", id="unknown flag"),
        pytest.param(["-h"], "write usage", id="help"),
    ],
)
async def test_run_main_usage_write_failure(args: list[str], want: str) -> None:
    stdout = io.BytesIO()
    code = await run_main(
        args,
        must_not_read_environment,
        ErrorReader(),
        stdout,
        ClosedPipe(),
        Config(name="test-runner", parse_request=parse_test_request),
    )
    assert code == 1, stdout.getvalue()
    event = gojson.validate(stdout.getvalue())
    assert event["type"] == "error"
    assert want in event["message"], event
    assert "write usage: " + str(ClosedPipeError()) in event["message"], event


async def test_run_main_help() -> None:
    def parse(_: Reader) -> tuple[Request, None]:
        raise AssertionError("help reached request parsing")

    code, stdout, stderr = await call_run_main(
        ["-h"],
        must_not_read_environment,
        ErrorReader(),
        Config(name="test-runner", parse_request=parse),
    )
    assert code == 0, (stdout, stderr)
    assert stdout == "", (stdout, stderr)
    for want in [
        "test-runner [options] < request.json",
        "test-runner [options] 'JSON request'",
        "test-runner [options] -p 'prompt'",
        "-p prompt",
        "-workspace",
        "-session-directory",
        "-log-directory",
        "-tool-heartbeat-interval",
        "$XDG_STATE_HOME/unreal-agent/sessions",
        "$HOME/.local/state/unreal-agent/sessions",
        "optional session JSONL log directory; unset writes only to stdout",
        "Request schema",
        "messages:",
        "role:",
        "content:",
        "message_id?:",
        "prompt:",
        "model:",
        "max_attempts:",
        "system_prompt:",
        "thinking_level:",
        "session_id:",
        "disallowed_tools:",
        "extra_allowed_tools:",
        "include_partial_messages:",
    ]:
        assert want in stderr, f"help missing {want!r}"
    assert "mcp_servers" not in stderr, "common request schema includes MCP servers"
