"""Adapter tests over an in-process HTTP transport.

Transports are `httpx.MockTransport`: a disconnect is the `RemoteProtocolError`
httpx raises when a peer closes early, and a header timeout is
`httpx.ReadTimeout`. Every test runs on looptime with the wall clock driven by
loop time, so backoff waits through the real timer primitive without sleeping.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any, cast

import httpx
import pytest
from unreal_agent.gojson import Raw, Timestamp, compact
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Request,
    RequestOptions,
    Response,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
    Usage,
)
from unreal_agent.llm.responsesapi import (
    MAX_SSE_FRAME_BYTES,
    MODEL_RESPONSE_IDLE_TIMEOUT,
    APIError,
    CacheKeyPlacement,
    Config,
    Exchange,
    ResponsesAdapter,
    TransportError,
    decode_response,
)
from unreal_agent.primitives import RemoteClient, SSEFrameDelimiter
from unreal_agent.primitives import timer as timer_module

from tests.llm.responsesapi._helpers import (
    COMPLETED_RESPONSE,
    FALLBACK_CALL,
    FALLBACK_MESSAGE,
    detailed_request,
    valid_request,
)

# Nanosecond loop time lets jittered backoff deadlines land exactly.
pytestmark = pytest.mark.looptime(resolution=1e-9)

BASE_NS = 1_785_758_400 * 10**9
SECOND = 10**9
MILLISECOND = 10**6

DISCONNECT = "peer closed connection without sending complete message body"


@pytest.fixture(autouse=True)
def fake_wall_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Timestamp]:
    """Drive the wall clock from looptime's fake loop time."""
    starts: dict[asyncio.AbstractEventLoop, float] = {}

    def now() -> Timestamp:
        loop = asyncio.get_running_loop()
        start = starts.setdefault(loop, loop.time())
        return Timestamp(BASE_NS + round((loop.time() - start) * 10**9))

    monkeypatch.setattr(timer_module, "wall_clock", now)
    return now


class Stream(httpx.AsyncByteStream):
    """A response body: chunks, then an optional error or a wait for cancellation."""

    def __init__(
        self,
        chunks: list[bytes] | None = None,
        *,
        error: BaseException | None = None,
        hang: asyncio.Event | None = None,
        gate: asyncio.Event | None = None,
        gate_after: int = 0,
    ) -> None:
        self.chunks = list(chunks or [])
        self.error = error
        self.hang = hang
        self.gate = gate
        self.gate_after = gate_after

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if self.gate is not None and index == self.gate_after:
                await self.gate.wait()
            await asyncio.sleep(0)
            yield chunk
        if self.hang is not None:
            self.hang.set()
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error


def reply(
    status: int = 200,
    body: bytes | list[bytes] = b"",
    *,
    content_type: str | None = "text/event-stream",
    headers: dict[str, str] | None = None,
    error: BaseException | None = None,
    hang: asyncio.Event | None = None,
) -> httpx.Response:
    all_headers = dict(headers or {})
    if content_type is not None:
        all_headers["Content-Type"] = content_type
    chunks = body if isinstance(body, list) else ([body] if body else [])
    return httpx.Response(
        status, headers=all_headers, stream=Stream(chunks, error=error, hang=hang)
    )


def stream_body(response: bytes) -> bytes:
    """One compacted response.completed event."""
    return b'data: {"type":"response.completed","response":' + compact(response) + b"}\n\n"


type Handler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


def remote_client(handler: Handler) -> RemoteClient:
    return RemoteClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def new_adapter(handler: Handler, **config: Any) -> ResponsesAdapter:
    return ResponsesAdapter(remote_client(handler), Config(**config))


def new_test_adapter(handler: Handler, endpoint: str) -> ResponsesAdapter:
    return new_adapter(
        handler,
        endpoint=endpoint,
        headers={"Authorization": ["Bearer test-key"], "Content-Type": ["application/json"]},
    )


def retry_error_event(kind: str, code: str, message: str) -> bytes:
    if kind == "response.failed":
        event = {
            "type": "response.failed",
            "response": {
                "id": "old",
                "status": "failed",
                "error": {"code": code, "message": message},
                "output": [],
            },
        }
    else:
        event = {"type": "error", "code": code, "message": message}
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"


async def respond_error(
    adapter: ResponsesAdapter, request: Request | None = None, options: RequestOptions | None = None
) -> tuple[Response | None, BaseException | None]:
    try:
        response = await adapter.respond(request or valid_request(), options or RequestOptions())
    except Exception as error:
        return None, error
    return response, None


def assert_response_error(
    kind: str, code: str, response: Response | None, error: BaseException | None
) -> None:
    if kind == "response.failed":
        assert error is None, error
        assert response is not None
        assert response.Failure is not None
        assert response.Failure.Code == code
        assert response.Failure.Message != ""
        return
    assert isinstance(error, APIError), error
    assert error.code == code
    assert error.message.strip() != ""


async def settle() -> None:
    for _ in range(200):
        await asyncio.sleep(0)


async def test_adapter_remote_requests_use_uuid_correlation_ids() -> None:
    async def unused(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    adapter = new_adapter(unused, endpoint="https://example.com/responses")
    first = adapter.remote_request(b"", "").correlation_id
    second = adapter.remote_request(b"", "").correlation_id
    uuid.UUID(first)
    uuid.UUID(second)
    assert first != second


async def test_adapter_responds() -> None:
    bodies: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["Accept"] == "text/event-stream"
        bodies.append(json.loads(request.content))
        return reply(
            body=stream_body(b"""{
            "id":"resp-1",
            "object":"response",
            "status":"completed",
            "output":[
                {"id":"message-1","type":"message","role":"assistant","status":"completed",
                    "phase":"final_answer","content":[{"type":"output_text","text":"hello",
                    "annotations":[],"logprobs":[]}]},
                {"id":"reasoning-1","type":"reasoning","status":"completed",
                    "summary":[{"type":"summary_text","text":"Used tools."}],
                    "encrypted_content":"opaque"},
                {"id":"function-1","type":"function_call","call_id":"call-2","name":"weather",
                    "arguments":"{\\"city\\":\\"Paris\\"}","status":"completed"}
            ],
            "usage":{"input_tokens":20,"input_tokens_details":{"cached_tokens":8,
                "cache_write_tokens":3},"output_tokens":10,
                "output_tokens_details":{"reasoning_tokens":4},"total_tokens":30}
        }""")
        )

    adapter = new_test_adapter(handler, "https://example.test/responses")
    got = await adapter.respond(detailed_request(), RequestOptions())
    assert bodies[0]["stream"] is True
    assert bodies[0] == json.loads("""{
        "max_output_tokens":128,
        "stream":true,
        "store":false,
        "include":["reasoning.encrypted_content"],
        "input":[
            {"content":"Be concise.","role":"system"},
            {"content":"weather?","role":"user"},
            {"content":"Use tools.","role":"system"},
            {"content":[{"annotations":[],"logprobs":[],"text":"Checking.","type":"output_text"}],
                "id":"previous-message","phase":"commentary","role":"assistant",
                "status":"completed","type":"message"},
            {"id":"previous-reasoning","type":"reasoning","status":"completed",
                "summary":[{"type":"summary_text","text":"Checked the request."}],
                "content":[{"type":"reasoning_text","text":"verbatim"}],
                "encrypted_content":"previous-opaque"},
            {"arguments":"{\\"city\\":\\"London\\"}","call_id":"call-1","name":"weather",
                "id":"previous-tool-call","type":"function_call"},
            {"call_id":"call-1","output":[{"type":"input_text","text":"sunny"}],
                "type":"function_call_output"}
        ],
        "model":"gpt-test",
        "tools":[{
            "description":"Get weather",
            "name":"weather",
            "parameters":{
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"],
                "additionalProperties":false
            },
            "strict":false,
            "type":"function"
        }]
    }""")
    assert got == Response(
        ID="resp-1",
        Stop=StopReason.COMPLETE,
        Output=[
            Item(
                ProviderID="message-1",
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="hello", Phase="final_answer"),
            ),
            Item(
                ProviderID="reasoning-1",
                Type=ItemType.PROVIDER,
                Data=ProviderItem(
                    Type="reasoning",
                    Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text="Used tools."),
                    Raw=Raw(
                        b'{"id":"reasoning-1","type":"reasoning","status":"completed",'
                        b'"summary":[{"type":"summary_text","text":"Used tools."}],'
                        b'"encrypted_content":"opaque"}'
                    ),
                ),
            ),
            Item(
                ProviderID="function-1",
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-2", Name="weather", Arguments='{"city":"Paris"}'),
            ),
        ],
        Usage=Usage(
            InputTokens=20,
            CachedInputTokens=8,
            CacheWriteInputTokens=3,
            OutputTokens=10,
            ReasoningTokens=4,
            ByModel={
                "": TokenUsage(
                    InputTokens=20,
                    CachedInputTokens=8,
                    CacheWriteInputTokens=3,
                    OutputTokens=10,
                    ReasoningTokens=4,
                )
            },
            Raw=Raw(
                b'{"input_tokens":20,"input_tokens_details":{"cached_tokens":8,'
                b'"cache_write_tokens":3},"output_tokens":10,"output_tokens_details":'
                b'{"reasoning_tokens":4},"total_tokens":30}'
            ),
        ),
    )


async def test_adapter_returns_provider_errors() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(
            400,
            b'{"error":{"code":"invalid_request","message":"model is required","param":"model",'
            b'"type":"invalid_request_error"}}',
            content_type="application/json",
        )

    _, error = await respond_error(
        new_test_adapter(handler, "https://example.test/responses"), Request()
    )
    assert isinstance(error, APIError)
    assert (error.status_code, error.code, error.param, error.type) == (
        400,
        "invalid_request",
        "model",
        "invalid_request_error",
    )
    assert str(error) == "create response: responses API error invalid_request: model is required"


async def test_respond_cancellation() -> None:
    started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = new_test_adapter(handler, "https://example.test/responses")
    task = asyncio.create_task(adapter.respond(valid_request(), RequestOptions()))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_adapter_traces_provider_exchange() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(
            body=stream_body(b'{"id":"resp-1","status":"completed","output":[],"usage":{}}')
        )

    traced: list[Exchange] = []
    adapter = new_adapter(handler, endpoint="https://example.test/responses", trace=traced.append)
    await adapter.respond(valid_request(), RequestOptions())
    assert len(traced) == 1
    assert traced[0].status_code == 200
    assert b'"model":"gpt-test"' in traced[0].request_body
    assert b'"id":"resp-1"' in traced[0].response_body


async def test_new_adapter_requires_dependencies() -> None:
    with pytest.raises(ValueError, match="remote client must be set"):
        ResponsesAdapter(None, Config(endpoint="https://example.com/responses"))

    async def unused(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    with pytest.raises(ValueError, match="endpoint must be set"):
        ResponsesAdapter(remote_client(unused), Config())


async def test_adapter_model_requests_use_the_model_idle_bound() -> None:
    async def unused(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    adapter = new_test_adapter(unused, "http://example.invalid/responses")
    request = adapter.remote_request(b"", "")
    assert request.response_idle_timeout == MODEL_RESPONSE_IDLE_TIMEOUT
    assert request.retry_policy.max_attempts == 1
    assert request.sse is not None
    assert request.sse.max_frame_size == MAX_SSE_FRAME_BYTES
    assert request.sse.frame_delimiter == SSEFrameDelimiter.STRIP
    assert request.headers["Accept"] == ["text/event-stream"]
    assert request.method == "POST"


CACHE_KEYS = {
    "empty": ("", ""),
    "session": ("session-1", "84097828fc31a8c8d29210df48901a85de7fd013f686b17be77d1be29cb7a98b"),
    "another-session": (
        "session-2",
        "5d9061408048c12d053925aed45333a142997f26a2cd1e0c4a87678c53a1e3ae",
    ),
    "long": ("s" * 300, "2955c7328c57ca39d0568bb930a5360e6b0e7f33931639c819d7cbfeaf0a88c7"),
    "long-distinct": (
        "s" * 300 + "t",
        "39a446f53f3a89913a4cc295c285287870053e373201678b6a6401e7bbb5dfef",
    ),
    "unicode": ("会話", "098eb2e3728cd354d91d5ff240891cbb20e225247cda47eeb8d49686fe7bd248"),
    "header-characters": (
        "key\r\nwith\tcontrols",
        "45ffa7b66165175c09a7c0c0396d13359f51bdd60a348c910da142212717fb04",
    ),
}


@pytest.mark.parametrize(
    "placement",
    [
        pytest.param(CacheKeyPlacement(), id="disabled"),
        pytest.param(CacheKeyPlacement(use_prompt_cache_key_field=True), id="body"),
        pytest.param(CacheKeyPlacement(header="x-cache-affinity"), id="header"),
        pytest.param(
            CacheKeyPlacement(header="x-cache-affinity", use_prompt_cache_key_field=True), id="both"
        ),
    ],
)
async def test_adapter_cache_key_placement(placement: CacheKeyPlacement) -> None:
    failures: list[str] = []
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        case_name = body["input"][0]["content"]
        _, want = CACHE_KEYS[case_name]
        want_body = want if placement.use_prompt_cache_key_field else ""
        want_header = want if placement.header else ""
        if want_body == "":
            if "prompt_cache_key" in body:
                failures.append(f"{case_name}: unexpected prompt_cache_key")
        elif body.get("prompt_cache_key") != want_body:
            failures.append(f"{case_name}: prompt_cache_key = {body.get('prompt_cache_key')}")
        if request.headers.get("x-cache-affinity", "") != want_header:
            failures.append(
                f"{case_name}: affinity header = {request.headers.get('x-cache-affinity')}"
            )
        if request.headers.get("X-Test") != "kept":
            failures.append(f"{case_name}: X-Test = {request.headers.get('X-Test')}")
        return reply(
            body=stream_body(b'{"id":"resp-1","status":"completed","output":[],"usage":{}}')
        )

    headers = {"X-Test": ["kept"]}
    adapter = new_adapter(
        handler, endpoint="https://example.test", headers=headers, cache_key_placement=placement
    )

    async def run(name: str, key: str) -> None:
        request = valid_request()
        request.Input = [Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=name))]
        for _ in range(2):
            await adapter.respond(request, RequestOptions(CacheKey=key))

    await asyncio.gather(*(run(name, key) for name, (key, _) in CACHE_KEYS.items()))
    assert failures == []
    assert requests == 2 * len(CACHE_KEYS)
    assert headers == {"X-Test": ["kept"]}, "configured headers mutated"


async def test_headerless_http_error_preserves_details() -> None:
    body = (
        b'{"error":{"code":"invalid_token","message":"Token expired; renew credentials.",'
        b'"type":"authentication_error","param":null}}'
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(401, body, content_type=None)

    traced: list[Exchange] = []
    adapter = new_adapter(handler, endpoint="https://example.test", trace=traced.append)
    _, error = await respond_error(adapter)
    assert isinstance(error, APIError)
    assert (error.status_code, error.code, error.message, error.type) == (
        401,
        "invalid_token",
        "Token expired; renew credentials.",
        "authentication_error",
    )
    assert traced[-1].status_code == 401
    assert traced[-1].response_body == body


async def test_responses_streaming_returns_terminal_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "text/event-stream"
        event = (
            b'event: response.completed\r\ndata: {"type":"response.completed",\r\ndata: "response":'
            + COMPLETED_RESPONSE
            + b"}\r\n\r\n"
        )
        chunks = [
            b": keepalive\r\n\r\nevent: response.output_text.delta\r\n"
            b'data: {"type":"response.output_text.delta","delta":"not authoritative"}\r\n\r\n',
            *(event[index : index + 13] for index in range(0, len(event), 13)),
        ]
        return reply(body=chunks, content_type="text/event-stream; charset=utf-8")

    traced: list[Exchange] = []
    adapter = new_adapter(handler, endpoint="https://example.test", trace=traced.append)
    got = await asyncio.wait_for(adapter.respond(valid_request(), RequestOptions()), 5)
    assert got == decode_response(COMPLETED_RESPONSE)
    assert traced[-1].response_body == COMPLETED_RESPONSE
    assert traced[-1].status_code == 200


@pytest.mark.parametrize(
    ("content_type", "body", "match", "status", "code"),
    [
        pytest.param("text/event-stream", b"", "without a terminal", 0, "", id="empty"),
        pytest.param(
            "text/event-stream", b"data: [DONE]\n\n", "without a terminal", 0, "", id="done only"
        ),
        pytest.param(
            "text/event-stream",
            b'data: {"type":"response.completed","response":' + COMPLETED_RESPONSE + b"}",
            "without a terminal",
            0,
            "",
            id="truncated frame",
        ),
        pytest.param(
            "text/event-stream", b"data: {bad}\n\n", "invalid Responses", 0, "", id="invalid JSON"
        ),
        pytest.param(
            "application/json",
            COMPLETED_RESPONSE,
            "without a terminal response",
            0,
            "",
            id="wrong content type",
        ),
        pytest.param(
            "text/event-stream",
            b'data: {"type":"response.completed","response":{"status":"in_progress"}}\n\n',
            "unsupported response status",
            0,
            "",
            id="nonterminal response status",
        ),
        pytest.param(
            "text/event-stream",
            b'data: {"type":"error","code":"usage_limit_reached","message":"limit reached"}\n\n',
            "limit reached",
            0,
            "usage_limit_reached",
            id="error event",
        ),
        pytest.param(
            "text/event-stream",
            b'data: {"type":"error","error":{"type":"insufficient_quota",'
            b'"code":"credit_balance_exhausted","message":"You have no credits remaining."}}\n\n',
            "no credits remaining",
            0,
            "credit_balance_exhausted",
            id="error event nested",
        ),
        pytest.param(
            "application/json",
            b'{"error":{"code":"invalid_token","message":"expired"}}',
            "expired",
            401,
            "invalid_token",
            id="unauthorized",
        ),
        pytest.param(
            "application/json",
            b'{"error":{"code":"usage_limit_reached","message":"quota"}}',
            "quota",
            429,
            "usage_limit_reached",
            id="quota",
        ),
    ],
)
async def test_responses_stream_failures(
    content_type: str, body: bytes, match: str, status: int, code: str
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(status or 200, body, content_type=content_type)

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=1)
    response, error = await respond_error(adapter)
    assert response is None
    assert error is not None
    assert match in str(error)
    if code:
        assert isinstance(error, APIError)
        assert error.code == code
        if status:
            assert error.status_code == status


@pytest.mark.parametrize("disconnect", [False, True], ids=["disconnect=false", "disconnect=true"])
async def test_responses_stream_retry_discards_partial_attempt(disconnect: bool) -> None:
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        assert request.method == "POST"
        assert (request.url.path, request.url.query) == ("/responses", b"tenant=1")
        if len(bodies) == 1:
            return reply(
                body=[
                    b'data: {"type":"response.created","response":{"id":"old"},'
                    b'"sequence_number":0}\n\n',
                    b'data: {"type":"response.output_item.done","output_index":0,"item":'
                    + FALLBACK_CALL
                    + b',"sequence_number":1}\n\n',
                ],
                error=httpx.RemoteProtocolError(DISCONNECT) if disconnect else None,
            )
        return reply(
            body=[
                b'data: {"type":"response.created","response":{"id":"new"},'
                b'"sequence_number":0}\n\n',
                b'data: {"type":"response.output_item.done","output_index":1,"item":'
                + FALLBACK_MESSAGE
                + b"}\n\n",
                b'data: {"type":"response.completed","response":{"id":"new","status":"completed",'
                b'"output":[]}}\n\n',
            ]
        )

    adapter = new_adapter(
        handler, endpoint="https://example.test/responses?tenant=1", max_attempts=2
    )
    got = await adapter.respond(valid_request(), RequestOptions())
    assert len(bodies) == 2
    assert got.ID == "new"
    assert len(got.Output or []) == 1
    assert cast(Message, (got.Output or [])[0].Data).Text == "fallback"
    assert bodies[0] == bodies[1], "retry changed request body"


async def test_responses_stream_cancellation() -> None:
    started = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=b'data: {"type":"response.created"}\n\n', hang=started)

    adapter = new_adapter(handler, endpoint="https://example.test")
    task = asyncio.create_task(adapter.respond(valid_request(), RequestOptions()))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)


async def test_responses_traces_assembled_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(
            body=[
                b'data: {"type":"response.output_item.done","output_index":0,"item":'
                + FALLBACK_MESSAGE
                + b"}\n\n",
                b'data: {"type":"response.completed","response":{"id":"r","status":"completed",'
                b'"output":[]}}\n\n',
            ]
        )

    traces: list[Exchange] = []
    adapter = new_adapter(handler, endpoint="https://example.test", trace=traces.append)
    response = await adapter.respond(valid_request(), RequestOptions())
    assert response.ID == "r"
    assert len(response.Output or []) == 1
    assert len(traces) == 1
    assert cast(Message, (response.Output or [])[0].Data).Text == "fallback"


@pytest.mark.parametrize(
    ("prefix", "trailer", "error_text"),
    [
        pytest.param(b"", b"", "", id="EOF"),
        pytest.param("﻿".encode(), b"", "", id="leading BOM"),
        pytest.param(b": heartbeat\n\n" + "﻿".encode(), b"", "", id="BOM after heartbeat"),
        pytest.param(b"", b": heartbeat\n\n", "", id="heartbeat"),
        pytest.param(b"", b"data: [DONE]\n\n: heartbeat\n\n", "", id="done"),
        pytest.param(
            b"",
            b'data: {"type":"response.output_text.delta","delta":"unfinished"}\n\n',
            "",
            id="nonterminal",
        ),
        pytest.param(
            b"data: {bad}\n\n",
            b"",
            "invalid Responses stream event JSON",
            id="malformed before terminal",
        ),
        pytest.param(b"", b"data: {bad}\n\n", "", id="malformed after terminal"),
        pytest.param(
            b"",
            b'data: {bad}\n\ndata: {"type":"response.output_text.delta","delta":"unfinished"}\n\n',
            "",
            id="malformed followed by valid trailer",
        ),
    ],
)
async def test_responses_captures_terminal_response(
    prefix: bytes, trailer: bytes, error_text: str
) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return reply(
            body=prefix
            + b'data: {"type":"response.completed","response":'
            + COMPLETED_RESPONSE
            + b"}\n\n"
            + trailer
        )

    response, error = await respond_error(new_test_adapter(handler, "https://example.test"))
    assert calls == 1
    if error_text:
        assert error is not None
        assert error_text in str(error)
        assert response is None
    else:
        assert error is None, error
        assert response is not None
        assert response.ID == "resp-1"


@pytest.mark.parametrize("max_attempts", [1, 2])
async def test_responses_preserves_terminal_after_disconnect(max_attempts: int) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return reply(
            body=b'data: {"type":"response.completed","response":' + COMPLETED_RESPONSE + b"}\n\n",
            error=httpx.RemoteProtocolError(DISCONNECT),
        )

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=max_attempts)
    response = await adapter.respond(valid_request(), RequestOptions())
    assert calls == 1
    assert response == decode_response(COMPLETED_RESPONSE)


@pytest.mark.parametrize(
    ("limit", "attempts"),
    [
        pytest.param(None, 5, id="default"),
        pytest.param(1, 1, id="disabled"),
        pytest.param(2, 2, id="custom"),
    ],
)
async def test_adapter_retry_configuration(limit: int | None, attempts: int) -> None:
    async def unused(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    adapter = new_adapter(unused, endpoint="https://example.com/responses", max_attempts=limit)
    assert adapter.max_attempts == attempts
    assert adapter.remote_request(b"", "").retry_policy.max_attempts == 1


async def test_adapter_rejects_invalid_max_attempts() -> None:
    async def unused(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    for limit in (-1, 0):
        with pytest.raises(ValueError, match="max attempts must be positive"):
            new_adapter(unused, endpoint="https://example.com/responses", max_attempts=limit)


@pytest.mark.parametrize("failure", ["status", "connection", "partial response"])
async def test_adapter_retries_transient_failures(failure: str) -> None:
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            match failure:
                case "status":
                    return reply(502, b"bad gateway", content_type=None)
                case "connection":
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response."
                    )
                case _:
                    return reply(
                        body=b'data: {"type":"response.created","response":'
                        b'{"id":"discard-this-partial-response",',
                        headers={"Content-Length": "4096"},
                        error=httpx.RemoteProtocolError(DISCONNECT),
                    )
        return reply(
            body=stream_body(b'{"id":"recovered","status":"completed","output":[],"usage":{}}')
        )

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=2)
    response = await adapter.respond(valid_request(), RequestOptions())
    assert response.ID == "recovered"
    assert len(bodies) == 2
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("limit", [1, 2])
async def test_adapter_bounds_retries(limit: int) -> None:
    attempts = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return reply(503, content_type=None)

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=limit)
    _, error = await respond_error(adapter)
    assert isinstance(error, APIError)
    assert error.status_code == 503
    assert attempts == limit


async def test_adapter_does_not_retry_missing_required_parameter() -> None:
    attempts = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return reply(
            400,
            b'{"error":{"code":"missing_required_parameter","param":"model",'
            b'"message":"Missing required parameter: \'model\'."}}',
            content_type=None,
        )

    _, error = await respond_error(new_test_adapter(handler, "https://example.test"))
    assert isinstance(error, APIError)
    assert error.code == "missing_required_parameter"
    assert attempts == 1


@dataclass
class TimedReply:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


class TimedTransport:
    """Replays replies and records the wall clock of each call."""

    def __init__(self, replies: list[TimedReply], clock: Callable[[], Timestamp]) -> None:
        self.replies = replies
        self.clock = clock
        self.calls: list[int] = []

    async def __call__(self, _: httpx.Request) -> httpx.Response:
        current = self.replies[min(len(self.calls), len(self.replies) - 1)]
        self.calls.append(self.clock().unix_ns)
        content_type = "text/event-stream" if current.status == 200 else "application/json"
        return reply(
            current.status, current.body, content_type=content_type, headers=current.headers
        )


def timed_adapter(transport: TimedTransport, limit: int) -> ResponsesAdapter:
    return new_adapter(transport, endpoint="http://example.invalid/responses", max_attempts=limit)


@pytest.mark.parametrize(
    ("limit", "success", "replies"),
    [
        pytest.param(
            2,
            False,
            [TimedReply(200, retry_error_event("error", "server_error", "retry"))],
            id="top-level exhausted",
        ),
        pytest.param(
            2,
            False,
            [TimedReply(200, retry_error_event("response.failed", "server_error", "retry"))],
            id="nested exhausted",
        ),
        pytest.param(
            1,
            False,
            [
                TimedReply(
                    200,
                    retry_error_event(
                        "response.failed", "rate_limit_exceeded", "try again in 120s"
                    ),
                )
            ],
            id="disabled",
        ),
        pytest.param(
            2,
            False,
            [
                TimedReply(503),
                TimedReply(200, retry_error_event("response.failed", "server_error", "retry")),
            ],
            id="HTTP then nested",
        ),
        pytest.param(
            2,
            False,
            [
                TimedReply(200, retry_error_event("response.failed", "server_error", "retry")),
                TimedReply(503),
            ],
            id="nested then HTTP",
        ),
        pytest.param(
            4,
            True,
            [
                TimedReply(503),
                TimedReply(200, retry_error_event("response.failed", "server_error", "retry")),
                TimedReply(
                    200, retry_error_event("error", "rate_limit_exceeded", "try again in 1ms")
                ),
                TimedReply(
                    200,
                    b'data: {"type":"response.completed","response":{"id":"ok",'
                    b'"status":"completed","output":[]}}\n\n',
                ),
            ],
            id="mixed recovery",
        ),
    ],
)
async def test_exchange_retries_share_attempt_limit(
    limit: int, success: bool, replies: list[TimedReply], fake_wall_clock: Callable[[], Timestamp]
) -> None:
    transport = TimedTransport(replies, fake_wall_clock)
    response, error = await respond_error(timed_adapter(transport, limit))
    assert len(transport.calls) == limit
    assert (error is None and response is not None and response.Failure is None) == success
    assert fake_wall_clock().unix_ns == transport.calls[-1], "waited after the final attempt"
    if success:
        assert response is not None
        assert response.ID == "ok"


@pytest.mark.parametrize(
    ("kind", "code", "message", "status", "header", "want"),
    [
        pytest.param(
            "error",
            "rate_limit_exceeded",
            "try again in 1.25s",
            200,
            "",
            1250 * MILLISECOND,
            id="top-level",
        ),
        pytest.param(
            "response.failed",
            "rate_limit_exceeded",
            "try again in 28ms",
            200,
            "",
            28 * MILLISECOND,
            id="nested",
        ),
        pytest.param(
            "http", "rate_limit_exceeded", "try again in 45s", 429, "", 30 * SECOND, id="HTTP body"
        ),
        pytest.param("http", "", "", 503, "120", 30 * SECOND, id="HTTP header"),
        pytest.param(
            "http",
            "server_is_overloaded",
            "overloaded",
            503,
            "90",
            30 * SECOND,
            id="overload header",
        ),
        pytest.param(
            "response.failed",
            "rate_limit_exceeded",
            "try again in 1s",
            200,
            "90",
            30 * SECOND,
            id="nested header",
        ),
        pytest.param("http", "", "", 503, "date", 30 * SECOND, id="HTTP date"),
    ],
)
async def test_exchange_retry_hints(
    kind: str,
    code: str,
    message: str,
    status: int,
    header: str,
    want: int,
    fake_wall_clock: Callable[[], Timestamp],
) -> None:
    if header == "date":
        now = datetime.fromtimestamp(fake_wall_clock().unix_ns // SECOND, UTC)
        header = format_datetime(now + timedelta(minutes=1), usegmt=True)
    body = retry_error_event(kind, code, message)
    if kind == "http":
        body = json.dumps({"error": {"code": code, "message": message}}).encode()
    transport = TimedTransport([TimedReply(status, body, {"Retry-After": header})], fake_wall_clock)
    response, error = await respond_error(timed_adapter(transport, 2))
    assert len(transport.calls) == 2
    assert transport.calls[1] - transport.calls[0] == want
    if code:
        assert_response_error(kind, code, response, error)


@pytest.mark.parametrize(
    ("kind", "code", "status", "base"),
    [
        pytest.param("http", "", 503, 2 * SECOND, id="HTTP"),
        pytest.param("error", "server_error", 200, 2 * SECOND, id="top-level"),
        pytest.param("response.failed", "server_error", 200, 2 * SECOND, id="nested"),
        pytest.param("http", "server_is_overloaded", 503, 10 * SECOND, id="HTTP overload"),
        pytest.param("response.failed", "slow_down", 200, 10 * SECOND, id="nested overload"),
    ],
)
async def test_exchange_backoff_uses_timer(
    kind: str, code: str, status: int, base: int, fake_wall_clock: Callable[[], Timestamp]
) -> None:
    body = retry_error_event(kind, code, "retry")
    if kind == "http":
        body = json.dumps({"error": {"code": code}}).encode()
    transport = TimedTransport([TimedReply(status, body)], fake_wall_clock)
    response, error = await respond_error(timed_adapter(transport, 3))
    assert len(transport.calls) == 3
    assert error is not None or (response is not None and response.Failure is not None)
    for index in range(1, len(transport.calls)):
        expected = base * (1 << (index - 1))
        delay = transport.calls[index] - transport.calls[index - 1]
        assert expected * 4 // 5 <= delay <= expected


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(TimedReply(503), id="status"),
        pytest.param(TimedReply(503, headers={"Retry-After": "3600"}), id="header"),
        pytest.param(
            TimedReply(
                200,
                retry_error_event("response.failed", "rate_limit_exceeded", "try again in 3600s"),
            ),
            id="message",
        ),
    ],
)
async def test_exchange_backoff_cancellation(
    first: TimedReply, fake_wall_clock: Callable[[], Timestamp]
) -> None:
    transport = TimedTransport([first], fake_wall_clock)
    adapter = timed_adapter(transport, 3)
    task = asyncio.create_task(adapter.respond(valid_request(), RequestOptions()))
    await settle()
    assert len(transport.calls) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(1):
            await adapter.respond(valid_request(), RequestOptions())
    assert len(transport.calls) == 2, "retried after cancellation or deadline"


@pytest.mark.parametrize("kind", ["error", "response.failed"])
@pytest.mark.parametrize("code", ["rate_limit_exceeded", "server_error"])
async def test_exchange_restarts_failed_generation(kind: str, code: str) -> None:
    sent: list[tuple[bytes, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert (request.url.path, request.url.query) == ("/responses", b"tenant=1")
        assert request.headers["Authorization"] == "Bearer token"
        assert request.headers["Accept"] == "text/event-stream"
        wire = json.loads(request.content)
        assert wire["store"] is False
        assert wire["stream"] is True
        sent.append((request.content, request.headers.get("session-id", "")))
        if len(sent) == 1:
            return reply(
                headers={"Retry-After": "1"},
                body=[
                    b'data: {"type":"response.created","response":{"id":"old"},'
                    b'"sequence_number":0}\n\n',
                    b'data: {"type":"response.output_item.done","output_index":1,"item":'
                    + FALLBACK_CALL
                    + b',"sequence_number":7}\n\n',
                    retry_error_event(kind, code, "temporary failure"),
                ],
            )
        return reply(
            body=[
                b'data: {"type":"response.created","response":{"id":"new"}}\n\n',
                b'data: {"type":"response.output_item.done","output_index":0,"item":'
                + FALLBACK_MESSAGE
                + b"}\n\n",
                b'data: {"type":"response.completed","response":{"id":"new","status":"completed",'
                b'"output":[]}}\n\n',
            ]
        )

    traces: list[Exchange] = []
    adapter = new_adapter(
        handler,
        endpoint="https://example.test/responses?tenant=1",
        max_attempts=2,
        headers={"Authorization": ["Bearer token"]},
        cache_key_placement=CacheKeyPlacement(header="session-id", use_prompt_cache_key_field=True),
        trace=traces.append,
    )
    response = await adapter.respond(valid_request(), RequestOptions(CacheKey="same-session"))
    assert response.ID == "new"
    assert response.Failure is None
    assert len(sent) == 2
    assert len(traces) == 1
    assert len(response.Output or []) == 1
    assert cast(Message, (response.Output or [])[0].Data).Text == "fallback"
    assert sent[0] == sent[1]
    assert sent[0][1] != ""


@pytest.mark.parametrize("kind", ["error", "response.failed", "http"])
@pytest.mark.parametrize("code", ["context_length_exceeded", "invalid_api_key"])
async def test_exchange_does_not_retry_permanent_errors(kind: str, code: str) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if kind == "http":
            return reply(
                429,
                json.dumps({"error": {"code": code, "message": "permanent error"}}).encode(),
                content_type=None,
            )
        return reply(body=retry_error_event(kind, code, "permanent error"))

    response, error = await respond_error(new_test_adapter(handler, "https://example.test"))
    assert calls == 1
    assert_response_error(kind, code, response, error)


@pytest.mark.parametrize("kind", ["error", "response.failed"])
@pytest.mark.parametrize("code", ["insufficient_quota", "rate_limit_exceeded"])
async def test_exchange_preserves_sse_error_after_disconnect(kind: str, code: str) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return reply(
            headers={"Content-Length": "100000"},
            body=[
                retry_error_event(kind, code, "Please try again in 10ms."),
                b'data: {"type":"response.output_text.delta","delta":"trailer"}\n\n',
            ],
            error=httpx.RemoteProtocolError(DISCONNECT),
        )

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=2)
    response, error = await respond_error(adapter)
    assert calls == (2 if code == "rate_limit_exceeded" else 1)
    assert_response_error(kind, code, response, error)


@pytest.mark.parametrize(
    ("reason", "stop"),
    [("max_output_tokens", StopReason.MAX_OUTPUT_TOKENS), ("content_filter", StopReason.REFUSED)],
)
async def test_exchange_incomplete_responses_are_not_retried(reason: str, stop: StopReason) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return reply(
            body=b'data: {"type":"response.incomplete","response":{"id":"r","status":"incomplete",'
            b'"output":[],"incomplete_details":{"reason":"%s"}}}\n\n' % reason.encode()
        )

    response = await new_test_adapter(handler, "https://example.test").respond(
        valid_request(), RequestOptions()
    )
    assert response.Stop == stop
    assert calls == 1


async def test_exchange_retries_header_timeout() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out")
        return reply(body=stream_body(b'{"id":"recovered","status":"completed","output":[]}'))

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=2)
    response = await adapter.respond(valid_request(), RequestOptions())
    assert response.ID == "recovered"
    assert calls == 2


async def test_exchange_bounds_connection_failures() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    adapter = new_adapter(handler, endpoint="https://example.test", max_attempts=2)
    _, error = await respond_error(adapter)
    assert isinstance(error, TransportError)
    assert "disconnected" in str(error)
    assert calls == 2
