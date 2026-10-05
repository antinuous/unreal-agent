"""Ports of cmd/unreal-agent-runner/tools_test.go, plus the console entry point."""

import io
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from unreal_agent.runner import Config, run_main
from unreal_agent.runner.__main__ import NAME, parse_request
from unreal_agent.runner.flags import REQUEST_HELP, usage
from unreal_agent.tool import static_names

from tests.runner._helpers import must_not_read_environment, reader


async def test_parse_request_configures_static_tools() -> None:
    parsed, factory = parse_request(
        reader('{"prompt":"hello","disallowed_tools":["Bash","SkillUse"]}')
    )
    assert parsed.prompt == "hello"
    assert factory is not None
    from unreal_agent.runner import ToolConfig

    configured = await factory(ToolConfig(names=static_names()))
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
        '{"prompt":"hello","unknown":true}',
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


def test_main_interrupt_exits_130(tmp_path: Path) -> None:
    """SIGINT while the model request is pending cancels the run without an error event."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    environment = {
        **os.environ,
        "UNREAL_HARNESS_LLM_BASE_URL": f"http://127.0.0.1:{port}",
        "OPENAI_API_KEY": "key",
    }
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
        stderr=subprocess.PIPE,
        env=environment,
    )
    try:
        listener.settimeout(30)
        connection, _ = listener.accept()
        time.sleep(0.2)
        process.send_signal(signal.SIGINT)
        stdout, _ = process.communicate(timeout=30)
        connection.close()
    finally:
        listener.close()
        if process.poll() is None:
            process.kill()
    assert process.returncode == 130
    assert b'"type":"error"' not in stdout
