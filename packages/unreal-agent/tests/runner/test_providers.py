import asyncio
import io
import json
import threading

import pytest
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
