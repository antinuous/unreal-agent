import asyncio
import contextlib
import errno
import gzip
import http
import socket
import ssl
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import httpx
import pytest
from unreal_agent.primitives import (
    TERMINAL_EVENT_TYPES,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    RemoteClient,
    RemoteCompletedResult,
    RemoteOutputResult,
    RemoteRequest,
    RemoteResponseStartedResult,
    RemoteRetryPolicy,
    RemoteRetryScheduledResult,
    RemoteSSEOptions,
    RemoteStreamFailureResult,
    SSEFrameDelimiter,
    default_remote_request,
    new_remote_http_client,
)
from unreal_agent.primitives._errors import (
    ContextCanceledError,
    DeadlineExceededError,
    UnexpectedEOFError,
)
from unreal_agent.primitives.channel import Channel
from unreal_agent.primitives.primitive_event import EventSink, send_event
from unreal_agent.primitives.remote import (
    is_retryable_remote_error,
    is_retryable_remote_status,
    prepare_remote_request,
    run_remote_request,
    stream_remote_response,
    wait_for_remote_retry,
)

IO_READ_CHUNK_SIZE = 32 * 1024

# --- Local HTTP/1.1 test server -------------------------------------------------


@dataclass
class ServerRequest:
    method: str
    path: str
    headers: Message
    body: bytes
    handler: BaseHTTPRequestHandler

    def wait_for_disconnect(self) -> None:
        try:
            while self.handler.rfile.read(1):
                pass
        except OSError:
            pass


class ResponseWriter:
    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._handler = handler
        self.headers: list[tuple[str, str]] = []
        self.status: int | None = None
        self._chunked = False
        self._declared: int | None = None
        self._written = 0

    def set_header(self, name: str, value: str) -> None:
        self.headers = [(key, item) for key, item in self.headers if key.lower() != name.lower()]
        self.headers.append((name, value))

    def add_header(self, name: str, value: str) -> None:
        self.headers.append((name, value))

    def write_header(self, status: int) -> None:
        if self.status is not None:
            return
        self.status = status
        lines = [f"HTTP/1.1 {status} {_phrase(status)}"]
        for name, value in self.headers:
            if name.lower() == "content-length":
                self._declared = int(value)
            lines.append(f"{name}: {value}")
        if self._declared is None and status not in (204, 304):
            self._chunked = True
            lines.append("Transfer-Encoding: chunked")
        self._handler.wfile.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        self._handler.wfile.flush()

    def write(self, data: bytes) -> None:
        self.write_header(200)
        if not data:
            return
        if self._chunked:
            self._handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        else:
            self._handler.wfile.write(data)
        self._written += len(data)
        self._handler.wfile.flush()

    def finish(self) -> None:
        if self.status is None:
            if not any(name.lower() == "content-length" for name, _ in self.headers):
                self.headers.append(("Content-Length", "0"))
            self.write_header(200)
        if self._chunked:
            self._handler.wfile.write(b"0\r\n\r\n")
            self._handler.wfile.flush()
        elif self._declared is not None and self._written < self._declared:
            self._handler.close_connection = True


def _phrase(status: int) -> str:
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return "Status"


type Handler = Callable[[ServerRequest, ResponseWriter], None]


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler: Handler) -> None:
        self.test_handler = handler
        self.connections = 0
        self.errors: list[BaseException] = []
        super().__init__(("127.0.0.1", 0), _RequestHandler)

    def process_request(self, request: Any, client_address: Any) -> None:
        self.connections += 1
        super().process_request(request, client_address)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _dispatch(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        request = ServerRequest(self.command, self.path, self.headers, body, self)
        writer = ResponseWriter(self)
        try:
            cast(_Server, self.server).test_handler(request, writer)
            writer.finish()
        except BrokenPipeError, ConnectionResetError:
            self.close_connection = True
        except BaseException as error:
            cast(_Server, self.server).errors.append(error)
            self.close_connection = True

    do_GET = do_POST = do_PUT = do_HEAD = do_DELETE = _dispatch


@contextmanager
def serve(handler: Handler) -> Iterator[_Server]:
    server = _Server(handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.daemon = True
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
    if server.errors:
        raise server.errors[0]


def redirect(writer: ResponseWriter, location: str, status: int) -> None:
    writer.set_header("Location", location)
    writer.set_header("Content-Type", "text/html; charset=utf-8")
    writer.write_header(status)
    writer.write(f'<a href="{location}">{_phrase(status)}</a>.\n'.encode())


# --- Mock transport bodies ------------------------------------------------------


class Body(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        *,
        error: BaseException | None = None,
        delay: float = 0.0,
        close_delay: float = 0.0,
        close_error: BaseException | None = None,
        on_read: Callable[[], None] | None = None,
    ) -> None:
        self.chunks = list(chunks or [])
        self.error = error
        self.delay = delay
        self.close_delay = close_delay
        self.close_error = close_error
        self.on_read = on_read
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.on_read is not None:
            self.on_read()
        for chunk in self.chunks:
            # Like a network read, each chunk suspends so an expired idle timer can fire.
            await asyncio.sleep(self.delay)
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        if self.close_error is not None:
            raise self.close_error


def mock_http_client(
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def body_client(body: Body, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    return mock_http_client(lambda _: httpx.Response(200, headers=headers or {}, stream=body))


# --- Event helpers --------------------------------------------------------------


async def collect(events: Channel[PrimitiveEvent]) -> list[PrimitiveEvent]:
    collected: list[PrimitiveEvent] = []
    while True:
        event = await asyncio.wait_for(events.get(), 10)
        collected.append(event)
        if event.type in TERMINAL_EVENT_TYPES:
            return collected


async def send(client: RemoteClient, request: RemoteRequest) -> list[PrimitiveEvent]:
    events: Channel[PrimitiveEvent] = Channel()
    client.send_request(request, events)
    return await collect(events)


async def send_test(request: RemoteRequest) -> list[PrimitiveEvent]:
    client = RemoteClient()
    try:
        return await send(client, request)
    finally:
        await client.aclose()


def of_type(events: list[PrimitiveEvent], event_type: PrimitiveEventType) -> list[PrimitiveEvent]:
    return [event for event in events if event.type == event_type]


def single(events: list[PrimitiveEvent], event_type: PrimitiveEventType) -> PrimitiveEvent:
    assert len(events) == 1, events
    assert events[0].type == event_type, events
    return events[0]


def attempt_output(events: list[PrimitiveEvent], attempt: int) -> bytes:
    output = b""
    for event in of_type(events, PrimitiveEventType.REMOTE_OUTPUT):
        result: RemoteOutputResult = event.result
        if result.attempt != attempt:
            continue
        assert result.offset == len(output)
        output += result.data
    return output


def assert_types(events: list[PrimitiveEvent], want: list[PrimitiveEventType]) -> None:
    got: list[PrimitiveEventType] = []
    for event in events:
        if (
            event.type == PrimitiveEventType.REMOTE_OUTPUT
            and got
            and got[-1] == PrimitiveEventType.REMOTE_OUTPUT
        ):
            continue
        got.append(event.type)
    assert got == want, events


def assert_identity(events: list[PrimitiveEvent], request: RemoteRequest) -> None:
    for event in events:
        assert (event.source, event.correlation_id) == (request.source, request.correlation_id)


async def settle() -> None:
    for _ in range(100):
        await asyncio.sleep(0)


async def with_cancel_requested[T](function: Callable[[], Awaitable[T]]) -> T:
    """Run `function` in a task whose cancellation was requested before it starts work."""
    result: list[T] = []

    async def body() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        result.append(await function())

    task = asyncio.create_task(body())
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return result[0]


# --- Tests ----------------------------------------------------------------------


def test_default_remote_request() -> None:
    request = default_remote_request("source", "correlation", "https://example.com")
    assert (request.source, request.correlation_id) == ("source", "correlation")
    assert (request.method, request.url) == ("GET", "https://example.com")
    assert request.headers == {}
    assert request.response_idle_timeout == 30.0
    assert request.sse is None
    assert request.retry_policy == RemoteRetryPolicy(
        max_attempts=5,
        initial_backoff=2.0,
        max_backoff=30.0,
        retryable_status_codes=[408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529],
    )


async def test_new_remote_http_client_owns_transport() -> None:
    first = new_remote_http_client()
    second = new_remote_http_client()
    try:
        transport = first._transport  # pyright: ignore[reportPrivateUsage]
        assert isinstance(transport, httpx.AsyncHTTPTransport)
        assert transport is not second._transport  # pyright: ignore[reportPrivateUsage]
    finally:
        await first.aclose()
        await second.aclose()


async def test_remote_client_wraps_configured_http_client() -> None:
    class CloseTracking(httpx.AsyncBaseTransport):
        close_calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise AssertionError("unexpected request")

        async def aclose(self) -> None:
            self.close_calls += 1

    transport = CloseTracking()
    http_client = httpx.AsyncClient(transport=transport)
    client = RemoteClient(http_client)
    assert client.http_client is http_client
    await client.aclose()
    assert transport.close_calls == 1


async def test_remote_client_reuses_connections_across_requests() -> None:
    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        writer.write_header(204)

    with serve(handler) as server:
        client = RemoteClient()
        try:
            request = default_remote_request("operation-1", "remote-1", server.url)
            request.retry_policy.max_attempts = 1
            for _ in range(2):
                events = await send(client, request)
                assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED, events
        finally:
            await client.aclose()
        assert server.connections == 1


async def test_send_remote_request_streams_response() -> None:
    body = b"response body " * (IO_READ_CHUNK_SIZE // 4)
    errors: list[str] = []

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        if request.method != "POST":
            errors.append(f"method = {request.method}")
        if request.headers.get("Content-Length") != str(len(b"request body")):
            errors.append(f"content length = {request.headers.get('Content-Length')}")
        if request.headers.get_all("X-Request") != ["one", "two"]:
            errors.append(f"X-Request = {request.headers.get_all('X-Request')}")
        if request.body != b"request body":
            errors.append(f"request body = {request.body!r}")
        writer.add_header("X-Response", "one")
        writer.add_header("X-Response", "two")
        writer.write_header(201)
        writer.write(body)

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.method = "POST"
        request.headers["X-Request"] = ["one", "two"]
        request.body = b"request body"
        events = await send_test(request)
    assert errors == []
    assert len(events) >= 3
    assert_identity(events, request)
    assert events[0].type == PrimitiveEventType.REMOTE_RESPONSE_STARTED
    started: RemoteResponseStartedResult = events[0].result
    assert (started.attempt, started.status_code) == (1, 201)
    assert started.headers.get_list("X-Response") == ["one", "two"]

    output = b""
    for event in events[1:-1]:
        assert event.type == PrimitiveEventType.REMOTE_OUTPUT
        result: RemoteOutputResult = event.result
        assert (result.attempt, result.offset) == (1, len(output))
        assert len(result.data) <= IO_READ_CHUNK_SIZE
        output += result.data
    assert output == body
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED
    assert events[-1].result == RemoteCompletedResult(1)


async def test_send_remote_request_frames_sse_responses() -> None:
    body = b"data: one\r\n\r\ndata: two\n\n"

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        writer.set_header("Content-Type", "text/event-stream; charset=utf-8")
        writer.write(body)

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.sse = RemoteSSEOptions(max_frame_size=len(body))
        events = await send_test(request)
    outputs = of_type(events, PrimitiveEventType.REMOTE_OUTPUT)
    want = [b"data: one", b"data: two"]
    assert len(outputs) == len(want), outputs
    offset = 0
    for event, data in zip(outputs, want, strict=True):
        assert event.result == RemoteOutputResult(1, offset, data)
        offset += len(data)
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED


async def test_send_remote_request_keeps_raw_output_for_non_sse_responses() -> None:
    body = b'{"ok":true}'

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        writer.set_header("Content-Type", "application/json")
        writer.write(body)

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.sse = RemoteSSEOptions(max_frame_size=1)
        events = await send_test(request)
    assert attempt_output(events, 1) == body


@pytest.mark.parametrize(
    "body",
    [b"data: partial", b"data: complete\n\npartial"],
    ids=["incomplete event", "complete event and tail"],
)
async def test_send_remote_request_preserves_framed_output_before_stream_failure(
    body: bytes,
) -> None:
    stream = Body([body], error=UnexpectedEOFError())
    client = RemoteClient(body_client(stream, {"Content-Type": "text/event-stream"}))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.sse = RemoteSSEOptions(
        max_frame_size=len(body), frame_delimiter=SSEFrameDelimiter.PASS_THROUGH
    )
    request.retry_policy = RemoteRetryPolicy(max_attempts=1)
    events = await send(client, request)
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_STREAM_FAILED,
            PrimitiveEventType.FAILED,
        ],
    )
    assert attempt_output(events, 1) == body
    assert stream.closed


async def test_send_remote_request_replays_body_across_temporary_redirect() -> None:
    requests: list[str] = []
    errors: list[str] = []

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        requests.append(request.path)
        if request.method != "POST":
            errors.append(f"method = {request.method}")
        if request.body != b"replayed body":
            errors.append(f"body = {request.body!r}")
        if request.path == "/start":
            writer.set_header("Location", "/target")
            writer.write_header(307)
        elif request.path == "/target":
            writer.write_header(201)
            writer.write(b"redirect completed")
        else:
            errors.append(f"unexpected path {request.path}")
            writer.write_header(404)

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url + "/start")
        request.method = "POST"
        request.body = b"replayed body"
        request.retry_policy.max_attempts = 1
        events = await send_test(request)
    assert errors == []
    assert len(requests) == 2
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    assert events[0].result.status_code == 201
    assert attempt_output(events, 1) == b"redirect completed"


@pytest.mark.parametrize("status_code", [503, 520, 521, 522, 523, 524, 529])
async def test_send_remote_request_retries_transient_response(status_code: int) -> None:
    attempts: list[int] = []
    errors: list[str] = []

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        if request.body != b"replayed":
            errors.append(f"request body = {request.body!r}")
        attempts.append(1)
        attempt = len(attempts)
        writer.set_header("X-Attempt", str(attempt))
        if attempt == 1:
            writer.write_header(status_code)
            writer.write(b"try later")
            return
        writer.write_header(200)
        writer.write(b"completed")

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.method = "POST"
        request.body = b"replayed"
        request.retry_policy.max_attempts = 2
        request.retry_policy.initial_backoff = 0
        request.retry_policy.max_backoff = 0
        events = await send_test(request)
    assert errors == []
    assert len(attempts) == 2
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_RETRY_SCHEDULED,
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    started = of_type(events, PrimitiveEventType.REMOTE_RESPONSE_STARTED)
    first: RemoteResponseStartedResult = started[0].result
    assert (first.attempt, first.status_code, first.headers.get("X-Attempt")) == (
        1,
        status_code,
        "1",
    )
    assert attempt_output(events, 1) == b"try later"
    retry: RemoteRetryScheduledResult = of_type(events, PrimitiveEventType.REMOTE_RETRY_SCHEDULED)[
        0
    ].result
    assert retry.attempt == 1
    assert retry.delay == 0
    assert str(status_code) in retry.reason
    second: RemoteResponseStartedResult = started[1].result
    assert (second.attempt, second.status_code, second.headers.get("X-Attempt")) == (2, 200, "2")
    assert attempt_output(events, 2) == b"completed"
    assert events[-1].result == RemoteCompletedResult(2)


def test_remote_retry_policy_accepts_additional_status_codes() -> None:
    policy = default_remote_request("operation-1", "remote-1", "https://example.com").retry_policy
    assert not is_retryable_remote_status(policy, 409)
    policy.retryable_status_codes.append(409)
    assert is_retryable_remote_status(policy, 409)


async def test_send_remote_request_uses_additional_retryable_status_code() -> None:
    attempts: list[int] = []

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        attempts.append(1)
        writer.write_header(409 if len(attempts) == 1 else 204)

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.retry_policy.max_attempts = 2
        request.retry_policy.initial_backoff = 0
        request.retry_policy.max_backoff = 0
        request.retry_policy.retryable_status_codes.append(409)
        events = await send_test(request)
    assert len(attempts) == 2
    assert [event.type for event in events] == [
        PrimitiveEventType.REMOTE_RESPONSE_STARTED,
        PrimitiveEventType.REMOTE_RETRY_SCHEDULED,
        PrimitiveEventType.REMOTE_RESPONSE_STARTED,
        PrimitiveEventType.REMOTE_COMPLETED,
    ]


async def test_send_remote_request_delivers_final_retryable_response() -> None:
    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        writer.write_header(503)
        writer.write(b"unavailable")

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.retry_policy = RemoteRetryPolicy(max_attempts=1)
        events = await send_test(request)
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    started: RemoteResponseStartedResult = events[0].result
    assert (started.attempt, started.status_code) == (1, 503)
    assert attempt_output(events, 1) == b"unavailable"
    assert events[-1].result == RemoteCompletedResult(1)


@pytest.mark.looptime
async def test_send_remote_request_retries_response_idle_timeout() -> None:
    attempts: list[int] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        attempts.append(1)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    client = RemoteClient(mock_http_client(handler))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.response_idle_timeout = 1.0
    request.retry_policy = RemoteRetryPolicy(max_attempts=2)
    events = await send(client, request)
    assert [event.type for event in events] == [
        PrimitiveEventType.REMOTE_RETRY_SCHEDULED,
        PrimitiveEventType.FAILED,
    ]
    assert len(attempts) == 2
    retry: RemoteRetryScheduledResult = events[0].result
    assert retry.attempt == 1
    assert retry.delay == 0
    assert "deadline exceeded" in retry.reason
    failure: PrimitiveFailureResult = events[1].result
    assert "attempt 2" in failure.error
    assert "deadline exceeded" in failure.error


@pytest.mark.looptime
async def test_send_remote_request_response_idle_timeout_resets_on_body_progress() -> None:
    body = Body([b"one", b"two", b"three"], delay=0.75)
    client = RemoteClient(body_client(body))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.response_idle_timeout = 1.0
    request.retry_policy.max_attempts = 1
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    events = await send(client, request)
    assert loop.time() - started_at > request.response_idle_timeout
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    assert attempt_output(events, 1) == b"onetwothree"


@pytest.mark.looptime
async def test_send_remote_request_response_idle_timeout_excludes_consumer_backpressure() -> None:
    body = Body([b"", b"one", b"two"])
    client = RemoteClient(body_client(body))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.response_idle_timeout = 1.0
    request.retry_policy.max_attempts = 1
    events: Channel[PrimitiveEvent] = Channel()
    client.send_request(request, events)

    started = await events.get()
    assert started.type == PrimitiveEventType.REMOTE_RESPONSE_STARTED
    await settle()
    await asyncio.sleep(2 * request.response_idle_timeout)
    collected = [started, *await collect(events)]
    assert_types(
        collected,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    assert attempt_output(collected, 1) == b"onetwo"


@pytest.mark.looptime
async def test_send_remote_request_ignores_response_close_error_after_eof() -> None:
    body = Body([b"complete"], close_delay=2.0, close_error=RuntimeError("close failed"))
    client = RemoteClient(body_client(body))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.response_idle_timeout = 1.0
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    assert body.closed


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        pytest.param(DeadlineExceededError(), True, id="idle timeout"),
        pytest.param(TimeoutError(), True, id="asyncio deadline"),
        pytest.param(UnexpectedEOFError(), True, id="closed response"),
        pytest.param(
            socket.gaierror(socket.EAI_AGAIN, "temporary"), True, id="temporary DNS failure"
        ),
        pytest.param(ConnectionResetError(errno.ECONNRESET, "reset"), True, id="connection reset"),
        pytest.param(httpx.ReadTimeout("network timeout"), True, id="network timeout"),
        pytest.param(ContextCanceledError(), False, id="caller cancellation"),
        pytest.param(asyncio.CancelledError(), False, id="task cancellation"),
        pytest.param(Exception("permanent"), False, id="unknown error"),
        pytest.param(ssl.SSLCertVerificationError(), False, id="untrusted certificate"),
    ],
)
def test_remote_error_retry_classification_is_conservative(
    error: BaseException, retryable: bool
) -> None:
    assert is_retryable_remote_error(error) is retryable


async def test_send_remote_request_does_not_retry_redirect_policy_failure() -> None:
    attempts: list[int] = []
    url: list[str] = []

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        attempts.append(1)
        redirect(writer, url[0], 302)

    with serve(handler) as server:
        url.append(server.url)
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.retry_policy.initial_backoff = 0
        request.retry_policy.max_backoff = 0
        event = single(await send_test(request), PrimitiveEventType.FAILED)
    assert len(attempts) == 10
    assert "stopped after 10 redirects" in event.result.error


async def test_send_remote_request_cancellation_interrupts_outstanding_request() -> None:
    request_started = threading.Event()

    def handler(request: ServerRequest, _: ResponseWriter) -> None:
        request_started.set()
        request.wait_for_disconnect()

    with serve(handler) as server:
        client = RemoteClient()
        try:
            request = default_remote_request("operation-1", "remote-1", server.url)
            events: Channel[PrimitiveEvent] = Channel()
            task = client.send_request(request, events)
            assert await asyncio.to_thread(request_started.wait, 10)
            task.cancel()
            event = single(await collect(events), PrimitiveEventType.CANCELED)
        finally:
            await client.aclose()
    assert (event.source, event.correlation_id) == (request.source, request.correlation_id)


async def test_send_remote_request_cancellation_interrupts_retry_backoff() -> None:
    attempts: list[int] = []

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        attempts.append(1)
        writer.write_header(503)

    with serve(handler) as server:
        client = RemoteClient()
        try:
            request = default_remote_request("operation-1", "remote-1", server.url)
            request.retry_policy.max_attempts = 2
            request.retry_policy.initial_backoff = 3600
            request.retry_policy.max_backoff = 3600
            events: Channel[PrimitiveEvent] = Channel()
            task = client.send_request(request, events)
            started = await asyncio.wait_for(events.get(), 10)
            assert started.type == PrimitiveEventType.REMOTE_RESPONSE_STARTED
            retry_event = await asyncio.wait_for(events.get(), 10)
            assert retry_event.type == PrimitiveEventType.REMOTE_RETRY_SCHEDULED
            retry: RemoteRetryScheduledResult = retry_event.result
            assert (retry.attempt, retry.delay) == (1, 3600)
            task.cancel()
            single(await collect(events), PrimitiveEventType.CANCELED)
        finally:
            await client.aclose()
    assert len(attempts) == 1


async def test_send_remote_request_reports_stream_failure() -> None:
    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        writer.set_header("Content-Length", "100")
        writer.write_header(200)
        writer.write(b"short")

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.retry_policy = RemoteRetryPolicy(max_attempts=1)
        events = await send_test(request)
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_STREAM_FAILED,
            PrimitiveEventType.FAILED,
        ],
    )
    assert attempt_output(events, 1) == b"short"
    failure: RemoteStreamFailureResult = of_type(events, PrimitiveEventType.REMOTE_STREAM_FAILED)[
        0
    ].result
    assert failure.attempt == 1
    assert "unexpected EOF" in failure.error
    assert "unexpected EOF" in events[-1].result.error


async def test_send_remote_request_retries_stream_failure() -> None:
    attempts: list[int] = []

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            writer.set_header("Content-Length", "100")
            writer.write_header(200)
            writer.write(b"partial")
            return
        writer.write_header(200)
        writer.write(b"recovered")

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.retry_policy = RemoteRetryPolicy(max_attempts=2)
        events = await send_test(request)
    assert len(attempts) == 2
    assert_types(
        events,
        [
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_STREAM_FAILED,
            PrimitiveEventType.REMOTE_RETRY_SCHEDULED,
            PrimitiveEventType.REMOTE_RESPONSE_STARTED,
            PrimitiveEventType.REMOTE_OUTPUT,
            PrimitiveEventType.REMOTE_COMPLETED,
        ],
    )
    assert attempt_output(events, 1) == b"partial"
    failure: RemoteStreamFailureResult = of_type(events, PrimitiveEventType.REMOTE_STREAM_FAILED)[
        0
    ].result
    assert failure.attempt == 1
    assert "unexpected EOF" in failure.error
    retry: RemoteRetryScheduledResult = of_type(events, PrimitiveEventType.REMOTE_RETRY_SCHEDULED)[
        0
    ].result
    assert retry.attempt == 1
    assert "unexpected EOF" in retry.reason
    assert attempt_output(events, 2) == b"recovered"
    assert events[-1].result == RemoteCompletedResult(2)


def _set_method(request: RemoteRequest, value: str) -> None:
    request.method = value


def _set_url(request: RemoteRequest, value: str) -> None:
    request.url = value


def _set_idle(request: RemoteRequest) -> None:
    request.response_idle_timeout = 0


def _set_sse(request: RemoteRequest, options: RemoteSSEOptions) -> None:
    request.sse = options


def _set_attempts(request: RemoteRequest) -> None:
    request.retry_policy.max_attempts = 0


def _set_initial_backoff(request: RemoteRequest) -> None:
    request.retry_policy.initial_backoff = -1


def _set_max_below_initial(request: RemoteRequest) -> None:
    request.retry_policy.max_backoff = request.retry_policy.initial_backoff - 1


def _set_positive_max_with_zero_initial(request: RemoteRequest) -> None:
    request.retry_policy.initial_backoff = 0
    request.retry_policy.max_backoff = 1


def _set_status_codes(request: RemoteRequest) -> None:
    request.retry_policy.retryable_status_codes = [99]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(lambda r: _set_method(r, ""), "method must be set", id="method"),
        pytest.param(lambda r: _set_url(r, "/relative"), "absolute HTTP or HTTPS URL", id="URL"),
        pytest.param(
            lambda r: _set_method(r, "invalid\nmethod"),
            "invalid method",
            id="malformed HTTP request",
        ),
        pytest.param(
            _set_idle, "response idle timeout must be positive", id="response idle timeout"
        ),
        pytest.param(
            lambda r: _set_sse(r, RemoteSSEOptions()),
            "SSE maximum frame size must be positive",
            id="SSE frame size",
        ),
        pytest.param(
            lambda r: _set_sse(r, RemoteSSEOptions(max_frame_size=1, frame_delimiter=2)),
            "invalid SSE frame delimiter mode",
            id="SSE frame delimiter",
        ),
        pytest.param(_set_attempts, "maximum attempts must be positive", id="attempts"),
        pytest.param(
            _set_initial_backoff, "initial backoff must not be negative", id="initial backoff"
        ),
        pytest.param(
            _set_max_below_initial,
            "maximum backoff must not be less than initial backoff",
            id="maximum backoff",
        ),
        pytest.param(
            _set_positive_max_with_zero_initial,
            "maximum backoff must be zero when initial backoff is zero",
            id="positive maximum with zero initial backoff",
        ),
        pytest.param(
            _set_status_codes,
            "retryable status code 99 must be between 100 and 599",
            id="retryable status code",
        ),
    ],
)
async def test_send_remote_request_rejects_invalid_request(
    mutate: Callable[[RemoteRequest], None], message: str
) -> None:
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    mutate(request)
    event = single(await send_test(request), PrimitiveEventType.FAILED)
    assert message in event.result.error


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"": ["value"]}, id="empty name"),
        pytest.param({"Invalid\nName": ["value"]}, id="name"),
        pytest.param({"Header": ["invalid\rvalue"]}, id="value"),
    ],
)
async def test_send_remote_request_rejects_invalid_headers_before_attempt(
    headers: dict[str, list[str]],
) -> None:
    attempts: list[int] = []

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        attempts.append(1)
        writer.write_header(204)

    with serve(handler) as server:
        request = default_remote_request("operation-1", "remote-1", server.url)
        request.headers = headers
        event = single(await send_test(request), PrimitiveEventType.FAILED)
    assert "invalid header field" in event.result.error
    assert attempts == []


def test_prepare_remote_request_canonicalizes_headers_and_forwards_host() -> None:
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.headers = {"x-request-id": ["one", "two"], "host": ["tenant.example.com"]}
    prepared = prepare_remote_request(request)
    names = [name for name, _ in prepared.headers]
    assert "x-request-id" not in names
    assert [value for name, value in prepared.headers if name == "X-Request-Id"] == ["one", "two"]
    assert prepared.host == "tenant.example.com"
    assert all(name.lower() != "host" for name in names)


def test_prepare_remote_request_rejects_multiple_host_values() -> None:
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.headers = {"Host": ["one.example.com", "two.example.com"]}
    with pytest.raises(Exception, match="Host header must have at most one value"):
        prepare_remote_request(request)


async def test_send_remote_request_canceled_before_start() -> None:
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    client = RemoteClient(mock_http_client(lambda _: httpx.Response(200)))
    events: Channel[PrimitiveEvent] = Channel()
    task = client.send_request(request, events)
    task.cancel()
    event = single(await collect(events), PrimitiveEventType.CANCELED)
    assert (event.source, event.correlation_id) == (request.source, request.correlation_id)
    await client.aclose()


async def test_send_remote_event_canceled_before_send() -> None:
    events: Channel[PrimitiveEvent] = Channel()
    sent = await with_cancel_requested(
        lambda: send_event(events, PrimitiveEvent(PrimitiveEventType.REMOTE_OUTPUT))
    )
    assert sent is False


@pytest.mark.looptime
async def test_send_remote_event_cancellation_interrupts_blocked_send() -> None:
    events: Channel[PrimitiveEvent] = Channel()
    task = asyncio.create_task(send_event(events, PrimitiveEvent(PrimitiveEventType.REMOTE_OUTPUT)))
    await settle()
    task.cancel()
    assert await task is False
    assert events.empty()


def _response(body: Body) -> httpx.Response:
    return httpx.Response(200, stream=body)


async def test_stream_remote_response_canceled_before_start_event() -> None:
    body = Body([b"unused"])
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel()
    error = await with_cancel_requested(
        lambda: stream_remote_response(request, _response(body), events, 1)
    )
    assert isinstance(error, ContextCanceledError)
    assert body.closed


@pytest.mark.looptime
async def test_stream_remote_response_cancellation_interrupts_output() -> None:
    body = Body([b"output"])
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel(1)
    task = asyncio.create_task(stream_remote_response(request, _response(body), events, 1))
    await settle()
    task.cancel()
    assert isinstance(await task, ContextCanceledError)
    assert body.closed
    assert (await events.get()).type == PrimitiveEventType.REMOTE_RESPONSE_STARTED


async def _start_run(
    client: httpx.AsyncClient, request: RemoteRequest, events: EventSink
) -> asyncio.Task[None]:
    return asyncio.create_task(run_remote_request(client, request, events))


async def _finish(task: asyncio.Task[None]) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()


async def test_run_remote_request_cancellation_wins_over_read_and_close_errors() -> None:
    tasks: list[asyncio.Task[None]] = []
    body = Body(
        error=RuntimeError("read failed"),
        close_error=RuntimeError("close failed"),
        on_read=lambda: _cancel(tasks[0]),
    )
    client = body_client(body)
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel(2)
    tasks.append(await _start_run(client, request, events))
    await _finish(tasks[0])
    collected = await collect(events)
    assert [event.type for event in collected] == [
        PrimitiveEventType.REMOTE_RESPONSE_STARTED,
        PrimitiveEventType.CANCELED,
    ]


@pytest.mark.looptime
async def test_run_remote_request_cancellation_preserves_buffered_response_event() -> None:
    client = body_client(Body([b"output"]))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel(1)
    task = await _start_run(client, request, events)
    await settle()
    task.cancel()
    assert (await events.get()).type == PrimitiveEventType.REMOTE_RESPONSE_STARTED
    await _finish(task)
    assert (await events.get()).type == PrimitiveEventType.CANCELED
    assert events.empty()


@pytest.mark.looptime
async def test_run_remote_request_cancellation_preserves_buffered_output_event() -> None:
    client = body_client(Body([b"partial"], error=UnexpectedEOFError()))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel(1)
    task = await _start_run(client, request, events)
    assert (await events.get()).type == PrimitiveEventType.REMOTE_RESPONSE_STARTED
    await settle()
    task.cancel()
    assert (await events.get()).type == PrimitiveEventType.REMOTE_OUTPUT
    await _finish(task)
    assert (await events.get()).type == PrimitiveEventType.CANCELED
    assert events.empty()


@pytest.mark.looptime
async def test_run_remote_request_completed_event_does_not_require_immediate_drain() -> None:
    client = body_client(Body([]))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel(1)
    task = await _start_run(client, request, events)
    assert (await events.get()).type == PrimitiveEventType.REMOTE_RESPONSE_STARTED
    await settle()
    await _finish(task)
    assert (await events.get()).type == PrimitiveEventType.REMOTE_COMPLETED
    assert events.empty()


@pytest.mark.looptime
async def test_run_remote_request_cancellation_preserves_queued_event() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise ConnectionResetError(errno.ECONNRESET, "connection reset by peer")

    client = mock_http_client(handler)
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    events: Channel[PrimitiveEvent] = Channel(1)
    await events.put(PrimitiveEvent(PrimitiveEventType.REMOTE_OUTPUT))
    task = await _start_run(client, request, events)
    await settle()
    task.cancel()
    assert (await events.get()).type == PrimitiveEventType.REMOTE_OUTPUT
    await _finish(task)
    assert (await events.get()).type == PrimitiveEventType.CANCELED
    assert events.empty()


@pytest.mark.looptime
async def test_wait_for_remote_retry_cancellation_interrupts_timer() -> None:
    task = asyncio.create_task(wait_for_remote_retry(3600))
    await settle()
    task.cancel()
    assert await task is False


async def test_wait_for_remote_retry_canceled_before_start() -> None:
    started = time.monotonic()
    assert await with_cancel_requested(lambda: wait_for_remote_retry(3600)) is False
    assert time.monotonic() - started < 10


def test_remote_retry_delay_is_exponential_and_bounded() -> None:
    policy = RemoteRetryPolicy(initial_backoff=2.0, max_backoff=5.0)
    assert [policy.backoff(attempts) for attempts in range(1, 5)] == [2.0, 4.0, 5.0, 5.0]


@dataclass
class _ContentTypeCase:
    content_type: str = ""
    status: int = 0
    request_sse: bool = False
    framed: bool = False
    extra: dict[str, str] = field(default_factory=dict)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_ContentTypeCase(request_sse=True, framed=True), id="missing"),
        pytest.param(_ContentTypeCase(status=299, request_sse=True, framed=True), id="missing 299"),
        pytest.param(_ContentTypeCase(status=300, request_sse=True), id="missing 300"),
        pytest.param(_ContentTypeCase(status=401, request_sse=True), id="missing 401"),
        pytest.param(_ContentTypeCase(status=503, request_sse=True), id="missing 503"),
        pytest.param(
            _ContentTypeCase("text/event-stream", request_sse=True, framed=True), id="SSE"
        ),
        pytest.param(
            _ContentTypeCase("text/event-stream; charset=utf-8", request_sse=True, framed=True),
            id="SSE parameters",
        ),
        pytest.param(
            _ContentTypeCase("Text/Event-Stream", request_sse=True, framed=True), id="SSE case"
        ),
        pytest.param(_ContentTypeCase("application/json", request_sse=True), id="JSON"),
        pytest.param(_ContentTypeCase("text/plain", request_sse=True), id="text"),
        pytest.param(
            _ContentTypeCase("text/event-stream; broken", request_sse=True), id="malformed"
        ),
        pytest.param(_ContentTypeCase(), id="missing without opt-in"),
        pytest.param(_ContentTypeCase("text/event-stream"), id="SSE without opt-in"),
    ],
)
async def test_remote_sse_content_type_selection(case: _ContentTypeCase) -> None:
    body = b"data: first\r\n\r\ndata: second\n\nunfinished"

    def handler(_: ServerRequest, writer: ResponseWriter) -> None:
        if case.content_type:
            writer.set_header("Content-Type", case.content_type)
        if case.status:
            writer.write_header(case.status)
        writer.write(body)

    with serve(handler) as server:
        request = default_remote_request("test", "content-type", server.url)
        request.retry_policy.max_attempts = 1
        if case.request_sse:
            request.sse = RemoteSSEOptions(
                max_frame_size=1024, frame_delimiter=SSEFrameDelimiter.STRIP
            )
        events = await send_test(request)
    started: RemoteResponseStartedResult = of_type(
        events, PrimitiveEventType.REMOTE_RESPONSE_STARTED
    )[0].result
    assert started.headers.get("Content-Type", "") == case.content_type
    if case.framed:
        outputs = of_type(events, PrimitiveEventType.REMOTE_OUTPUT)
        assert [event.result.data for event in outputs] == [b"data: first", b"data: second"]
    else:
        assert attempt_output(events, 1) == body
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED


async def test_send_remote_request_completes_empty_gzip_response() -> None:
    client = RemoteClient(
        mock_http_client(lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}))
    )
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert [event.type for event in events] == [
        PrimitiveEventType.REMOTE_RESPONSE_STARTED,
        PrimitiveEventType.REMOTE_COMPLETED,
    ]


async def test_send_remote_request_decodes_concatenated_gzip_members() -> None:
    body = gzip.compress(b"one") + gzip.compress(b"two")
    client = RemoteClient(
        mock_http_client(
            lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=body)
        )
    )
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert attempt_output(events, 1) == b"onetwo"
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED


@pytest.mark.parametrize(
    ("method", "headers"),
    [
        pytest.param("GET", {"Range": ["bytes=0-"]}, id="range"),
        pytest.param("HEAD", {}, id="head"),
    ],
)
async def test_send_remote_request_skips_transparent_gzip(
    method: str, headers: dict[str, list[str]]
) -> None:
    body = gzip.compress(b"raw")
    seen: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=body)

    client = RemoteClient(mock_http_client(handler))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.method = method
    request.headers = headers
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert "Accept-Encoding" not in seen[0].headers
    started: RemoteResponseStartedResult = events[0].result
    assert started.headers.get("Content-Encoding") == "gzip"
    assert attempt_output(events, 1) == body


async def test_send_remote_request_sends_utf8_header_values() -> None:
    seen: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return httpx.Response(204)

    client = RemoteClient(mock_http_client(handler))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    request.headers = {"X-Name": ["café"]}
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED, events
    assert (b"X-Name", "café".encode()) in seen[0].headers.raw


async def test_send_remote_request_reports_unexpected_transport_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport failed")

    client = RemoteClient(mock_http_client(handler))
    request = default_remote_request("operation-1", "remote-1", "https://example.com")
    event = single(await send(client, request), PrimitiveEventType.FAILED)
    assert event.result.error == (
        'send remote request GET "https://example.com": attempt 1: '
        'Get "https://example.com": transport failed'
    )


def redirecting_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[RemoteClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(http_request: httpx.Request) -> httpx.Response:
        http_request.read()
        seen.append(http_request)
        return handler(http_request)

    transport = httpx.MockTransport(record)
    return RemoteClient(httpx.AsyncClient(transport=transport, follow_redirects=True)), seen


@pytest.mark.parametrize(
    ("status", "method", "want_method", "want_body"),
    [
        pytest.param(301, "PUT", "GET", b"", id="301 PUT"),
        pytest.param(302, "DELETE", "GET", b"", id="302 DELETE"),
        pytest.param(303, "HEAD", "HEAD", b"", id="303 HEAD"),
        pytest.param(308, "PATCH", "PATCH", b"payload", id="308 PATCH"),
    ],
)
async def test_send_remote_request_follows_go_redirect_methods(
    status: int, method: str, want_method: str, want_body: bytes
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/start":
            return httpx.Response(status, headers={"Location": "/target"})
        return httpx.Response(204)

    client, seen = redirecting_client(handler)
    request = default_remote_request("operation-1", "remote-1", "https://example.com/start")
    request.method = method
    request.body = b"payload"
    request.headers = {"Content-Type": ["text/plain"]}
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED, events
    target = seen[1]
    assert (target.method, target.content) == (want_method, want_body)
    assert ("Content-Type" in target.headers) == bool(want_body)
    assert target.headers.get("Referer") == "https://example.com/start"


@pytest.mark.parametrize(
    ("location", "kept"),
    [
        pytest.param("https://example.com:8443/target", True, id="same host"),
        pytest.param("https://api.example.com/target", True, id="subdomain"),
        pytest.param("https://other.test/target", False, id="other domain"),
    ],
)
async def test_send_remote_request_strips_sensitive_headers_like_go(
    location: str, kept: bool
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/start":
            return httpx.Response(302, headers={"Location": location})
        return httpx.Response(204)

    client, seen = redirecting_client(handler)
    request = default_remote_request("operation-1", "remote-1", "https://example.com/start")
    request.headers = {"Authorization": ["Bearer token"], "Cookie": ["a=b"], "X-Other": ["1"]}
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED, events
    target = seen[1]
    assert ("Authorization" in target.headers) is kept
    assert ("Cookie" in target.headers) is kept
    assert target.headers.get("X-Other") == "1"


@pytest.mark.parametrize(
    ("location", "want_host"),
    [
        pytest.param("/target", "tenant.example.com", id="relative"),
        pytest.param("https://example.com/target", "example.com", id="absolute"),
    ],
)
async def test_send_remote_request_preserves_host_only_on_relative_redirect(
    location: str, want_host: str
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path == "/start":
            return httpx.Response(302, headers={"Location": location})
        return httpx.Response(204)

    client, seen = redirecting_client(handler)
    request = default_remote_request("operation-1", "remote-1", "https://example.com/start")
    request.headers = {"Host": ["tenant.example.com"]}
    request.retry_policy.max_attempts = 1
    events = await send(client, request)
    assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED, events
    assert seen[0].headers["Host"] == "tenant.example.com"
    assert seen[1].headers["Host"] == want_host


async def test_remote_client_does_not_persist_response_cookies() -> None:
    cookies: list[str | None] = []

    def handler(request: ServerRequest, writer: ResponseWriter) -> None:
        cookies.append(request.headers.get("Cookie"))
        writer.set_header("Set-Cookie", "session=1; Path=/")
        writer.write_header(204)

    with serve(handler) as server:
        client = RemoteClient()
        try:
            request = default_remote_request("operation-1", "remote-1", server.url)
            request.retry_policy.max_attempts = 1
            for _ in range(2):
                events = await send(client, request)
                assert events[-1].type == PrimitiveEventType.REMOTE_COMPLETED, events
        finally:
            await client.aclose()
    assert cookies == [None, None]
