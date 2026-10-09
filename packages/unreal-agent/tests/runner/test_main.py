import contextlib
import io
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from unreal_agent.runner import Config, run_main
from unreal_agent.runner.__main__ import NAME, parse_request
from unreal_agent.runner.flags import REQUEST_HELP, usage
from unreal_agent.tool import BASH_NAME, SKILL_USE_NAME, VIEW_IMAGE_NAME

from tests.runner._helpers import (
    HTTPRequest,
    HTTPResponse,
    http_server,
    must_not_read_environment,
    reader,
)


async def test_parse_request_configures_static_tools() -> None:
    parsed, factory = parse_request(
        reader('{"prompt":"hello","disallowed_tools":["Bash","SkillUse"]}')
    )
    assert parsed.prompt == "hello"
    assert factory is not None
    from unreal_agent.runner import ToolConfig

    configured = await factory(ToolConfig(names=[BASH_NAME, VIEW_IMAGE_NAME, SKILL_USE_NAME]))
    names = [definition.tool.Name for definition in configured.registry.static_definitions()]
    assert names == ["ViewImage"]
    for name in ["Bash", "SkillUse", "McpSearch", "McpCall"]:
        assert configured.registry.resolve(name) is None, f"unavailable tool {name} resolves"
    assert not configured.remote_jobs, "slim runner configured remote jobs"


@pytest.mark.parametrize(
    "input",
    [
        '{"prompt":"hello","mcp_servers":{}}',
        '{"prompt":"hello","mcp_servers":{"test":"http://localhost"}}',
        '{"prompt":"hello","forward_auth":true}',
    ],
)
async def test_runner_rejects_unsupported_requests(input: str) -> None:
    stderr = io.BytesIO()
    code = await run_main(
        [],
        must_not_read_environment,
        reader(input),
        io.BytesIO(),
        stderr,
        Config(name=NAME, parse_request=parse_request),
    )
    assert code == 1, stderr.getvalue()
    assert b"unknown object member" in stderr.getvalue(), stderr.getvalue()


def _command(*args: str) -> list[str]:
    return [sys.executable, "-m", "unreal_agent.runner", *args]


def test_main_help_exits_zero() -> None:
    result = subprocess.run(_command("-h"), capture_output=True, check=False, timeout=60)
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr.startswith(usage(NAME).encode())
    assert result.stderr.endswith(REQUEST_HELP.encode())


def test_main_flag_error_exits_one() -> None:
    result = subprocess.run(_command("-nope"), capture_output=True, check=False, timeout=60)
    assert result.returncode == 1
    assert result.stdout == (b'{"type":"error","message":"flag provided but not defined: -nope"}\n')
    assert result.stderr.startswith(b"flag provided but not defined: -nope\nUsage:\n")
    assert result.stderr.endswith(b"unreal-agent-runner: flag provided but not defined: -nope\n")


@pytest.mark.parametrize(
    ("signum", "bash"),
    [
        pytest.param(signal.SIGINT, False, id="model/SIGINT"),
        pytest.param(signal.SIGTERM, False, id="model/SIGTERM"),
        pytest.param(signal.SIGINT, True, id="bash/SIGINT"),
        pytest.param(signal.SIGTERM, True, id="bash/SIGTERM"),
    ],
)
def test_runner_signals(signum: signal.Signals, bash: bool, tmp_path: Path) -> None:
    started: queue.Queue[int] = queue.Queue()
    release = threading.Event()
    command = ""
    if bash:
        ready_path, hold_path = tmp_path / "ready", tmp_path / "hold"
        for path in (ready_path, hold_path):
            os.mkfifo(path, 0o600)
        ready_fd = os.open(ready_path, os.O_RDWR)

        def read_ready() -> None:
            with os.fdopen(ready_fd, "rb") as ready:
                started.put(int(ready.readline()))

        threading.Thread(target=read_ready, daemon=True).start()
        command = "trap '' TERM; printf '%s\\n' \"$$\" > ready; read value < hold"

    def handle(_: HTTPRequest) -> HTTPResponse:
        if not bash:
            started.put(0)
            release.wait(30)
            return HTTPResponse(status=500)
        arguments = json.dumps({"command": command})
        response = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "id": "response-1",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "id": "function-1",
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "Bash",
                            "arguments": arguments,
                            "status": "completed",
                        }
                    ],
                },
            }
        )
        return HTTPResponse(
            headers={"Content-Type": "text/event-stream"}, body=f"data: {response}\n\n".encode()
        )

    with http_server(handle) as url:
        process = subprocess.Popen(
            _command(
                "-workspace",
                str(tmp_path),
                "-session-directory",
                str(tmp_path / "sessions"),
                "-p",
                "hello",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                "HOME": str(tmp_path),
                "XDG_CONFIG_HOME": str(tmp_path),
                "SHELL": "/bin/sh",
                "UNREAL_HARNESS_LLM_PROVIDER": "openai",
                "UNREAL_HARNESS_LLM_API_KEY": "test-key",
                "UNREAL_HARNESS_LLM_BASE_URL": url,
                "UNREAL_HARNESS_LLM_MAX_ATTEMPTS": "1",
            },
        )
        pid = 0
        try:
            try:
                pid = started.get(timeout=20)
            except queue.Empty:
                process.kill()
                output, _ = process.communicate()
                pytest.fail(f"runner did not become ready; output:\n{output.decode()}")
            if bash:
                assert pid > 1, f"invalid Bash process group: {pid}"
                os.killpg(pid, 0)
            process.send_signal(signum)
            output, _ = process.communicate(timeout=20)
        finally:
            release.set()
            if process.poll() is None:
                process.kill()
                process.wait()
            if pid > 1:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
    assert process.returncode == 130, output.decode()
    assert b'"type":"error"' not in output, output.decode()
    if bash:
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)
