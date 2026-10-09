import asyncio
import io
import json
import threading

import pytest
from unreal_agent.llm.clients import anthropic
from unreal_agent.runner import Config, default_providers, run_main

from tests.runner._helpers import (
    HTTPRequest,
    HTTPResponse,
    environment,
    http_server,
    parse_test_request,
    reader,
)

_PROVIDERS = [provider.name for provider in default_providers()]

_ANTHROPIC_STREAM = b"".join(
    b"data: " + event + b"\n\n"
    for event in (
        b'{"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant",'
        b'"content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
        b'{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        b'{"type":"content_block_delta","index":0,"delta":{"type":"text_delta",'
        b'"text":"anthropic works"}}',
        b'{"type":"content_block_stop","index":0}',
        b'{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}',
        b'{"type":"message_stop"}',
    )
)


def test_runner_anthropic_default_base_url() -> None:
    [selected] = [provider for provider in default_providers() if provider.name == "anthropic"]
    assert selected.base_url == anthropic.DEFAULT_BASE_URL
    assert selected.api_key_environment == "ANTHROPIC_API_KEY"


@pytest.mark.parametrize("override", [False, True])
async def test_runner_anthropic_uses_messages_api(
    override: bool, tmp_path_factory: pytest.TempPathFactory
) -> None:
    want_key = "override-key" if override else "anthropic-key"
    problems: list[str] = []
    requests = 0

    def handle(request: HTTPRequest) -> HTTPResponse:
        nonlocal requests
        requests += 1
        if (
            request.method != "POST"
            or request.path != "/v1/messages"
            or request.headers.get("x-api-key") != want_key
            or request.headers.get("anthropic-version") != "2023-06-01"
        ):
            problems.append("incorrect Messages API endpoint or authentication")
        body = json.loads(request.body)
        system = body.get("system") or []
        if (
            body.get("model") != "claude-test"
            or body.get("stream") is not True
            or len(system) != 1
            or not system[0]["text"].endswith("\n\nmy system prompt")
            or body.get("thinking", {}).get("type") != "adaptive"
            or body.get("output_config", {}).get("effort") != "high"
        ):
            problems.append(f"request = {body!r}")
        tools = [(tool.get("type"), tool.get("name")) for tool in body.get("tools") or []]
        if tools != [("custom", "Bash"), ("custom", "ViewImage")]:
            problems.append(f"tools = {tools!r}")
        problems.extend(
            f"unexpected message role {message['role']!r}"
            for message in body["messages"]
            if message["role"] != "user"
        )
        return HTTPResponse(headers={"Content-Type": "text/event-stream"}, body=_ANTHROPIC_STREAM)

    variables = {
        "UNREAL_HARNESS_LLM_PROVIDER": "anthropic",
        "UNREAL_HARNESS_LLM_MODEL": "claude-test",
        "ANTHROPIC_API_KEY": "anthropic-key",
    }
    if override:
        variables["UNREAL_HARNESS_LLM_API_KEY"] = want_key
    output, stderr = io.BytesIO(), io.BytesIO()
    with http_server(handle) as url:
        variables["UNREAL_HARNESS_LLM_BASE_URL"] = url + "/v1"
        code = await run_main(
            [
                "-workspace",
                str(tmp_path_factory.mktemp("workspace")),
                "-session-directory",
                str(tmp_path_factory.mktemp("sessions")),
            ],
            environment(variables),
            reader('{"prompt":"hello","system_prompt":"my system prompt"}'),
            output,
            stderr,
            Config(
                name="agent-runner",
                parse_request=parse_test_request,
                providers=default_providers(),
            ),
        )
    assert not problems
    assert code == 0, stderr.getvalue()
    assert requests == 1
    assert b"anthropic works" in output.getvalue(), stderr.getvalue()
    assert want_key.encode() not in output.getvalue(), "credential leaked into session output"


@pytest.mark.parametrize("provider", _PROVIDERS)
async def test_runner_provider_retries(
    provider: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    max_attempts = 2
    lock = threading.Lock()
    attempts = 0
    leaks: list[str] = []

    def handle(request: HTTPRequest) -> HTTPResponse:
        nonlocal attempts
        body = json.loads(request.body)
        if "max_attempts" in body:
            leaks.append("harness retry configuration leaked into provider request")
        with lock:
            attempts += 1
        return HTTPResponse(status=503)

    with http_server(handle) as url:
        async with asyncio.timeout(10):
            code = await run_main(
                [
                    "-workspace",
                    str(tmp_path_factory.mktemp("workspace")),
                    "-session-directory",
                    str(tmp_path_factory.mktemp("sessions")),
                ],
                environment(
                    {
                        "UNREAL_HARNESS_LLM_PROVIDER": provider,
                        "UNREAL_HARNESS_LLM_BASE_URL": url,
                        "OPENAI_CODEX_ACCESS_TOKEN": "subscription-token",
                        "OPENAI_CODEX_ACCOUNT_ID": "account-1",
                        "UNREAL_HARNESS_LLM_API_KEY": "test-key",
                    }
                ),
                reader(f'{{"prompt":"hello","model":"test","max_attempts":{max_attempts}}}'),
                io.BytesIO(),
                io.BytesIO(),
                Config(
                    name="unreal-agent-runner",
                    parse_request=parse_test_request,
                    providers=default_providers(),
                ),
            )
    assert not leaks
    assert (code, attempts) == (1, max_attempts)


async def test_runner_codex_uses_subscription_without_api_key(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    problems: list[str] = []

    def handle(request: HTTPRequest) -> HTTPResponse:
        if (
            request.headers.get("authorization") != "Bearer subscription-token"
            or request.headers.get("chatgpt-account-id") != "account"
        ):
            problems.append("wrong authentication")
        body = json.loads(request.body)
        if body.get("stream") is not True:
            problems.append(f"body = {body!r}")
        return HTTPResponse(
            headers={"Content-Type": "text/event-stream"},
            body=(
                b'data: {"type":"response.completed","response":{"id":"r","status":"completed",'
                b'"output":[{"id":"msg-1","type":"message","role":"assistant","status":"completed",'
                b'"content":[{"type":"output_text","text":"subscription works"}]}]}}\n\n'
            ),
        )

    output, stderr = io.BytesIO(), io.BytesIO()
    with http_server(handle) as url:
        code = await run_main(
            [
                "-workspace",
                str(tmp_path_factory.mktemp("workspace")),
                "-session-directory",
                str(tmp_path_factory.mktemp("sessions")),
            ],
            environment(
                {
                    "UNREAL_HARNESS_LLM_PROVIDER": "openai-codex",
                    "UNREAL_HARNESS_LLM_BASE_URL": url,
                    "OPENAI_CODEX_ACCESS_TOKEN": "subscription-token",
                    "OPENAI_CODEX_ACCOUNT_ID": "account",
                }
            ),
            reader('{"prompt":"hello","model":"gpt-test","system_prompt":"my system prompt"}'),
            output,
            stderr,
            Config(
                name="unreal-agent-runner",
                parse_request=parse_test_request,
                providers=default_providers(),
            ),
        )
    assert not problems
    assert code == 0, stderr.getvalue()
    assert b"subscription works" in output.getvalue(), stderr.getvalue()
    assert b"subscription-token" not in output.getvalue(), "credential leaked into session output"
