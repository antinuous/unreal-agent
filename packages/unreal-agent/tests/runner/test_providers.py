import asyncio
import dataclasses
import io
import json
import threading

import pytest
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Request,
    RequestOptions,
    Role,
    TokenUsage,
)
from unreal_agent.llm.clients import anthropic
from unreal_agent.runner import (
    Config,
    Provider,
    RunnerError,
    default_providers,
    find_provider,
    run_main,
)

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
    selected, _ = find_provider(default_providers(), "anthropic")
    assert selected.base_url == anthropic.DEFAULT_BASE_URL
    assert selected.api_key_environment == "ANTHROPIC_API_KEY"


def test_find_uses_supplied_providers() -> None:
    want = dataclasses.replace(
        default_providers()[0],
        name="custom",
        base_url="https://custom.example/v1",
        default_model="custom-model",
        api_key_environment="CUSTOM_API_KEY",
    )
    got, new_client = find_provider(
        [Provider(name="unconfigured", new_client=None), want], "custom"
    )
    assert (got.name, got.base_url, got.default_model, got.api_key_environment) == (
        want.name,
        want.base_url,
        want.default_model,
        want.api_key_environment,
    )
    assert new_client is want.new_client


@pytest.mark.parametrize(
    ("available", "requested", "want"),
    [
        pytest.param(
            [Provider(name="second", new_client=None), Provider(name="first", new_client=None)],
            "unknown",
            'unsupported provider "unknown"; available providers: second, first',
            id="unknown provider",
        ),
        pytest.param(
            [],
            "unknown",
            'unsupported provider "unknown"; available providers: ',
            id="no providers",
        ),
        pytest.param(
            [Provider(name="unconfigured", new_client=None)],
            "unconfigured",
            'provider "unconfigured" has no client factory',
            id="missing factory",
        ),
    ],
)
def test_find_errors(available: list[Provider], requested: str, want: str) -> None:
    with pytest.raises(RunnerError) as raised:
        find_provider(available, requested)
    assert str(raised.value) == want


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
            body.get("model") != "claude-opus-5-5"
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
        if request.method != "POST" or request.path not in ("/responses", "/messages"):
            leaks.append(f"unexpected request {request.method} {request.path}")
            return HTTPResponse(status=404)
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


def test_runner_provider_default_models() -> None:
    want = {"openai": "gpt-6-astra", "openai-codex": "gpt-6-astra", "anthropic": "claude-opus-5-5"}
    for provider in default_providers():
        assert provider.default_model == want.get(provider.name, ""), provider.name


async def test_runner_codex_uses_only_responses_api(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    problems: list[str] = []
    requests = 0

    def handle(request: HTTPRequest) -> HTTPResponse:
        nonlocal requests
        if request.method != "POST" or request.path != "/responses":
            problems.append(f"request = {request.method} {request.path}")
        requests += 1
        if (
            request.headers.get("authorization") != "Bearer subscription-token"
            or request.headers.get("chatgpt-account-id") != "account"
        ):
            problems.append("wrong authentication")
        body = json.loads(request.body)
        if body.get("stream") is not True or body.get("model") != "gpt-6-astra":
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
            reader('{"prompt":"hello","system_prompt":"my system prompt"}'),
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
    assert requests == 1, stderr.getvalue()
    assert b"subscription works" in output.getvalue(), stderr.getvalue()
    assert b"subscription-token" not in output.getvalue(), "credential leaked into session output"


@pytest.mark.parametrize(
    ("value", "want"),
    [
        pytest.param("", "", id="unset"),
        pytest.param("5m", "5m", id="5m"),
        pytest.param("1h", "1h", id="1h"),
    ],
)
async def test_anthropic_cache_ttl_configuration(value: str, want: str) -> None:
    cache_controls: list[dict[str, str]] = []

    def handle(request: HTTPRequest) -> HTTPResponse:
        cache_controls.append(json.loads(request.body)["cache_control"])
        stream = "".join(
            f"data: {event}\n\n"
            for event in (
                '{"type":"message_start","message":{"id":"cache-ttl","type":"message",'
                '"role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}',
                '{"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":0}}',
                '{"type":"message_stop"}',
            )
        )
        return HTTPResponse(headers={"Content-Type": "text/event-stream"}, body=stream.encode())

    _, new_client = find_provider(default_providers(), "anthropic")
    with http_server(handle) as url:
        client = new_client(
            "test-key", url, 1, lambda name: value if name == "ANTHROPIC_CACHE_TTL" else ""
        )
        try:
            request = Request(
                Model=Model(ID="claude-test"),
                Input=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="Hello"))],
            )
            await client.respond(request, RequestOptions())
        finally:
            await client.aclose()
    [cache_control] = cache_controls
    assert cache_control["type"] == "ephemeral"
    assert cache_control.get("ttl", "") == want, cache_control


@pytest.mark.parametrize("value", ["default", "10m"])
def test_anthropic_rejects_invalid_cache_ttl_configuration(value: str) -> None:
    selected, new_client = find_provider(default_providers(), "anthropic")
    with pytest.raises(ValueError, match=f'^ANTHROPIC_CACHE_TTL must be 5m or 1h, got "{value}"$'):
        new_client(
            "test-key",
            selected.base_url,
            1,
            lambda name: value if name == "ANTHROPIC_CACHE_TTL" else "",
        )


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        pytest.param("", True, id="default_enabled"),
        pytest.param("0", False, id="disabled_zero"),
        pytest.param("false", False, id="disabled_false"),
        pytest.param("1", True, id="enabled_one"),
        pytest.param("true", True, id="enabled_true"),
    ],
)
async def test_anthropic_fallback_configuration_and_continuation(value: str, enabled: bool) -> None:
    problems: list[str] = []
    count = 0

    def handle(request: HTTPRequest) -> HTTPResponse:
        nonlocal count
        count += 1
        turn = count
        body = json.loads(request.body)
        if body["model"] != "primary-model":
            problems.append(f"primary model changed to {body['model']!r}")
        beta = request.headers.get("anthropic-beta", "")
        if not enabled:
            if "fallbacks" in body or beta:
                problems.append("fallback enabled despite explicit opt-out")
        else:
            if body.get("fallbacks") != "default" or beta != "server-side-fallback-2026-07-01":
                problems.append(f"fallbacks = {body.get('fallbacks')!r}, beta = {beta!r}")
            messages = body["messages"]
            if turn == 2 and (
                len(messages) != 3
                or messages[1]["role"] != "assistant"
                or len(messages[1]["content"]) != 2
                or messages[1]["content"][0].get("type") != "fallback"
            ):
                problems.append(f"continuation lost fallback boundary: {messages!r}")
        model = "primary-model"
        usage = '{"input_tokens":10,"output_tokens":4}'
        if enabled:
            model = "fallback-model"
            iterations = (
                '{"type":"fallback_message","model":"fallback-model",'
                '"input_tokens":10,"output_tokens":4}'
            )
            if turn == 1:
                iterations = (
                    '{"type":"message","model":"primary-model","input_tokens":8,"output_tokens":2},'
                    + iterations
                )
            usage = '{"input_tokens":10,"output_tokens":4,"iterations":[' + iterations + "]}"
        events = [
            '{"type":"message_start","message":{"id":"message","type":"message",'
            f'"role":"assistant","content":[],"model":{json.dumps(model)},"usage":{{}}}}}}'
        ]
        if enabled and turn == 1:
            events += [
                '{"type":"content_block_start","index":0,"content_block":{"type":"fallback",'
                '"from":{"model":"primary-model"},"to":{"model":"fallback-model"}}}',
                '{"type":"content_block_stop","index":0}',
            ]
        events += [
            '{"type":"content_block_start","index":1,"content_block":{"type":"text","text":"Answer"}}',
            '{"type":"content_block_stop","index":1}',
            '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":' + usage + "}",
            '{"type":"message_stop"}',
        ]
        stream = "".join(f"data: {event}\n\n" for event in events)
        return HTTPResponse(headers={"Content-Type": "text/event-stream"}, body=stream.encode())

    _, new_client = find_provider(default_providers(), "anthropic")
    with http_server(handle) as url:
        client = new_client(
            "test-key", url, 1, lambda name: value if name == "ANTHROPIC_FALLBACK" else ""
        )
        try:
            request = Request(
                Model=Model(ID="primary-model"),
                Input=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="Hello"))],
            )
            for turn in range(2):
                response = await client.respond(request, RequestOptions())
                if enabled:
                    assert b'"fallback_message"' in bytes(response.Usage.Raw)
                want = {"primary-model": TokenUsage(InputTokens=10, OutputTokens=4)}
                if enabled:
                    want = {"fallback-model": TokenUsage(InputTokens=10, OutputTokens=4)}
                    if turn == 0:
                        want["primary-model"] = TokenUsage(InputTokens=8, OutputTokens=2)
                assert response.Usage.ByModel == want
                request.Input += response.Output or []
                request.Input.append(
                    Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="Continue"))
                )
        finally:
            await client.aclose()
    assert not problems
    assert count == 2


def test_anthropic_rejects_invalid_fallback_configuration() -> None:
    selected, new_client = find_provider(default_providers(), "anthropic")
    with pytest.raises(ValueError, match=r"^parse ANTHROPIC_FALLBACK: "):
        new_client(
            "test-key",
            selected.base_url,
            1,
            lambda name: "tru" if name == "ANTHROPIC_FALLBACK" else "",
        )
