"""Adapter tests over an in-process `httpx.MockTransport`, on looptime with the wall clock
driven by loop time so backoff waits through the real timer primitive without sleeping."""

import asyncio
import json
import uuid
from collections.abc import Callable

import httpx
import pytest
from unreal_agent.gojson import Timestamp
from unreal_agent.llm import (
    ItemType,
    Message,
    Request,
    RequestOptions,
    Response,
    StopReason,
    Tool,
    ToolType,
)
from unreal_agent.llm.messagesapi import (
    MAX_SSE_FRAME_BYTES,
    MODEL_RESPONSE_IDLE_TIMEOUT,
    APIError,
    CacheTTL,
    Config,
    Exchange,
    MessagesAdapter,
    StreamEndedError,
    TransportError,
    decode_response,
)
from unreal_agent.primitives import RemoteClient, SSEFrameDelimiter
from unreal_agent.primitives import timer as timer_module

from tests.llm.messagesapi._helpers import (
    Handler,
    live_stream,
    new_adapter,
    reply,
    response_body,
    valid_request,
)

pytestmark = pytest.mark.looptime(resolution=1e-9)

BASE_NS = 1_785_758_400 * 10**9


@pytest.fixture(autouse=True)
def fake_wall_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Timestamp]:
    starts: dict[asyncio.AbstractEventLoop, float] = {}

    def now() -> Timestamp:
        loop = asyncio.get_running_loop()
        start = starts.setdefault(loop, loop.time())
        return Timestamp(BASE_NS + round((loop.time() - start) * 10**9))

    monkeypatch.setattr(timer_module, "wall_clock", now)
    return now


async def respond(adapter: MessagesAdapter, request: Request | None = None) -> Response:
    return await adapter.respond(request or valid_request(), RequestOptions())


def _counting(handler: Handler) -> tuple[Handler, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    async def wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return await handler(request)

    return wrapped, requests


async def test_adapter_validates_configuration() -> None:
    with pytest.raises(ValueError, match=r"^remote client must be set$"):
        MessagesAdapter(None, Config(endpoint="http://example.invalid/messages"))
    remote = RemoteClient()
    try:
        with pytest.raises(ValueError, match=r"^messages API endpoint must be set$"):
            MessagesAdapter(remote, Config())
        for attempts in (0, -1):
            with pytest.raises(ValueError, match=r"^max attempts must be positive$"):
                MessagesAdapter(
                    remote,
                    Config(endpoint="http://example.invalid/messages", max_attempts=attempts),
                )
        for ttl, quoted in (("10m", '"10m"'), ("default", '"default"')):
            with pytest.raises(ValueError, match=f"^unsupported cache TTL {quoted}$"):
                MessagesAdapter(
                    remote, Config(endpoint="http://example.invalid/messages", cache_ttl=ttl)
                )
    finally:
        await remote.aclose()


async def test_adapter_applies_cache_ttl() -> None:
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(await request.aread())
        return reply(body=live_stream("text"))

    await respond(new_adapter(handler, cache_ttl=CacheTTL.ONE_HOUR))
    [body] = bodies
    assert json.loads(body)["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_adapter_remote_request() -> None:
    async def unused(_: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    adapter = new_adapter(unused, headers={"X-Api-Key": ["key"], "accept": ["application/json"]})
    first = adapter.remote_request(b"body")
    second = adapter.remote_request(b"body")
    uuid.UUID(first.correlation_id)
    assert first.correlation_id != second.correlation_id
    assert first.source == "llm.messagesapi"
    assert first.method == "POST"
    assert first.headers == {
        "X-Api-Key": ["key"],
        "Accept": ["text/event-stream"],
        "Anthropic-Beta": ["server-side-fallback-2026-07-01"],
    }
    disabled = new_adapter(unused, fallback=False).remote_request(b"body")
    assert disabled.headers == {"Accept": ["text/event-stream"]}
    assert first.response_idle_timeout == MODEL_RESPONSE_IDLE_TIMEOUT
    assert first.retry_policy.max_attempts == 1
    assert first.sse is not None
    assert first.sse.max_frame_size == MAX_SSE_FRAME_BYTES
    assert first.sse.frame_delimiter == SSEFrameDelimiter.STRIP


@pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
async def test_adapter_retries_http_failures_with_the_same_request(status: int) -> None:
    stream = live_stream("text")
    bodies: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return reply(
                status,
                b'{"type":"error","error":{"type":"api_error","message":"retry"}}',
                content_type="application/json",
                headers={"Retry-After-Ms": "1"},
            )
        return reply(body=stream)

    traces: list[Exchange] = []
    adapter = new_adapter(handler, max_attempts=2, trace=traces.append)
    response = await respond(adapter)
    assert len(bodies) == 2
    assert len(traces) == 1
    message = (response.Output or [])[0].Data
    assert isinstance(message, Message)
    assert message.Text == "hello"
    assert bodies[0] == bodies[1], "request changed on retry"


@pytest.mark.parametrize(
    ("status", "attempts", "disable"),
    [(401, 1, False), (400, 1, False), (503, 2, False), (503, 1, True)],
)
async def test_adapter_stops_at_attempt_limit(status: int, attempts: int, disable: bool) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        headers = {"Retry-After-Ms": "1", "Request-Id": "req-test"}
        if disable:
            headers["X-Should-Retry"] = "false"
        return reply(
            status,
            b'{"type":"error","error":{"type":"api_error","message":"failed"}}',
            content_type="application/json",
            headers=headers,
        )

    counted, requests = _counting(handler)
    adapter = new_adapter(counted, max_attempts=2)
    with pytest.raises(APIError) as raised:
        await respond(adapter)
    assert raised.value.status_code == status
    assert raised.value.request_id == "req-test"
    assert str(raised.value) == "messages API error api_error: failed"
    assert len(requests) == attempts


async def test_adapter_cancellation_stops_http_work() -> None:
    started, closed = asyncio.Event(), asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=b": ping\n\n", hang=started, closed=closed)

    adapter = new_adapter(handler)
    task = asyncio.create_task(respond(adapter))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), 5)


async def test_adapter_does_not_send_canceled_or_invalid_requests() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=live_stream("text"))

    counted, requests = _counting(handler)
    adapter = new_adapter(counted)
    task = asyncio.create_task(respond(adapter))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    invalid = valid_request()
    invalid.Tools = [Tool(Type=ToolType.HOSTED, Name="web_search")]
    with pytest.raises(ValueError, match=r"unsupported tool type"):
        await respond(adapter, invalid)
    assert not requests, "unexpected provider requests"


@pytest.mark.parametrize("content_type", ["application/json", "text/plain", None])
async def test_adapter_requires_event_stream(content_type: str | None) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=response_body("end_turn", "[]"), content_type=content_type)

    counted, requests = _counting(handler)
    adapter = new_adapter(counted, max_attempts=2)
    with pytest.raises(ValueError, match=r"expected text/event-stream") as raised:
        await respond(adapter)
    assert str(raised.value) == (
        f'expected text/event-stream response, got content type "{content_type or ""}"'
    )
    assert len(requests) == 1


async def test_adapter_rejects_invalid_content_type() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=live_stream("text"), content_type="text/event-stream; charset")

    adapter = new_adapter(handler)
    with pytest.raises(
        ValueError, match=r"^invalid response content type: mime: invalid media parameter$"
    ):
        await respond(adapter)


async def test_adapter_assembles_chunked_live_stream() -> None:
    stream = live_stream("thinking")
    stopped = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        chunks = [stream[index : index + 37] for index in range(0, len(stream), 37)]
        # Completion is message_stop, not the server closing its connection.
        return reply(
            body=chunks,
            content_type="text/event-stream; charset=utf-8",
            hang=asyncio.Event(),
            closed=stopped,
        )

    traces: list[Exchange] = []
    adapter = new_adapter(handler, trace=traces.append)
    response = await respond(adapter)
    await asyncio.wait_for(stopped.wait(), 5)
    [trace] = traces
    assert response.Usage.ReasoningTokens == 210
    assert len(response.Output or []) == 2
    assert trace.status_code == 200
    assert b"event:" not in trace.response_body
    assert decode_response(trace.response_body) == response


@pytest.mark.parametrize("finalized", [False, True])
async def test_adapter_stream_emits_only_finalized_blocks(finalized: bool) -> None:
    stream = live_stream("truncated-tool")
    if finalized:
        stream = stream.replace(
            b"event: message_delta",
            b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
            b"event: message_delta",
            1,
        )

    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=stream)

    traces: list[Exchange] = []
    adapter = new_adapter(handler, trace=traces.append)
    response = await respond(adapter)
    assert response.Stop == StopReason.MAX_OUTPUT_TOKENS
    assert len(response.Output or []) == (1 if finalized else 0)
    assert decode_response(traces[0].response_body) == response


_OVERLOADED = (
    b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",'
    b'"message":"Overloaded"}}\n\n'
)
_OVERLOADED_EMPTY = (
    b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":""}}\n\n'
)


def _tools_prefix() -> bytes:
    prefix, found, _ = live_stream("tools").partition(b"event: message_delta")
    assert found, "fixture missing message_delta"
    return prefix


@pytest.mark.parametrize(
    "suffix", [b"", _OVERLOADED, _OVERLOADED_EMPTY], ids=["EOF", "overloaded", "empty message"]
)
async def test_adapter_retries_interrupted_stream_without_leaking_partial_calls(
    suffix: bytes,
) -> None:
    prefix = _tools_prefix()
    success = live_stream("text")
    attempts = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        body = prefix + suffix if attempts == 1 else success
        return reply(body=body, headers={"Retry-After-Ms": "1"})

    traces: list[Exchange] = []
    adapter = new_adapter(handler, max_attempts=2, trace=traces.append)
    response = await respond(adapter)
    assert attempts == 2
    assert len(traces) == 1
    output = response.Output or []
    assert len(output) == 1
    assert output[0].Type == ItemType.MESSAGE


@pytest.mark.parametrize(
    "suffix",
    [
        b"",
        b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error",'
        b'"message":"failed"}}\n\n',
        b"event: message_delta\ndata: {\n\n",
    ],
    ids=["EOF", "provider error", "malformed"],
)
async def test_adapter_stream_failure_does_not_return_partial_output(suffix: bytes) -> None:
    prefix = _tools_prefix()

    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(body=prefix + suffix, headers={"Request-Id": "req-stream"})

    counted, requests = _counting(handler)
    adapter = new_adapter(counted, max_attempts=1)
    with pytest.raises(Exception) as raised:  # noqa: PT011
        await respond(adapter)
    assert len(requests) == 1
    error = raised.value
    match suffix:
        case b"":
            assert isinstance(error, StreamEndedError)
        case b"event: message_delta\ndata: {\n\n":
            assert isinstance(error, ValueError)
            assert str(error).startswith("decode Messages stream event: ")
        case _:
            assert isinstance(error, APIError)
            assert (error.status_code, error.type, error.request_id) == (
                200,
                "invalid_request_error",
                "req-stream",
            )


async def test_adapter_traces_provider_errors() -> None:
    body = b'{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}'

    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(400, body, content_type="application/json")

    traces: list[Exchange] = []
    adapter = new_adapter(handler, trace=traces.append)
    with pytest.raises(APIError):
        await respond(adapter)
    assert [(trace.status_code, trace.response_body) for trace in traces] == [(400, body)]


async def test_adapter_bounds_error_bodies() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return reply(500, [b"x" * (1 << 20), b"x"], content_type="text/plain")

    counted, requests = _counting(handler)
    adapter = new_adapter(counted)
    with pytest.raises(ValueError, match=r"^messages API response exceeds 1048576 bytes$"):
        await respond(adapter)
    assert len(requests) == 1


async def test_adapter_retries_transport_failures() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return reply(body=live_stream("text"))

    adapter = new_adapter(handler, max_attempts=2)
    response = await respond(adapter)
    assert attempts == 2
    assert len(response.Output or []) == 1

    attempts = 0
    adapter = new_adapter(handler, max_attempts=1)
    with pytest.raises(TransportError):
        await respond(adapter)


async def test_adapter_cancellation_prevents_retry() -> None:
    attempts = 0
    answered = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        answered.set()
        return reply(
            429,
            b'{"type":"error","error":{"type":"rate_limit_error","message":"wait"}}',
            content_type="application/json",
            headers={"Retry-After": "60"},
        )

    adapter = new_adapter(handler)
    task = asyncio.create_task(respond(adapter))
    await asyncio.wait_for(answered.wait(), 5)
    for _ in range(100):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert attempts == 1


async def test_adapter_request_body_reaches_provider() -> None:
    bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "text/event-stream"
        bodies.append(json.loads(request.content))
        return reply(body=live_stream("text"))

    adapter = new_adapter(handler)
    await respond(adapter)
    assert bodies[0]["model"] == "claude-test"
    assert bodies[0]["stream"] is True
