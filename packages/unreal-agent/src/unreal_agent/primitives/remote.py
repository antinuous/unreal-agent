import asyncio
import contextlib
import errno
import re
import socket
import ssl
import zlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Final

import httpx

from unreal_agent.primitives._errors import (
    ContextCanceledError,
    DeadlineExceededError,
    PrimitiveError,
    UnexpectedEOFError,
    causes,
    describe,
    go_quote,
    has_errno,
    join,
    wrap,
)
from unreal_agent.primitives.file import IO_READ_CHUNK_SIZE
from unreal_agent.primitives.primitive_event import (
    NO_CORRELATION,
    NO_SOURCE,
    CorrelationID,
    EventSink,
    PrimitiveEvent,
    PrimitiveEventType,
    SourceID,
    cancel_requested,
    primitive_canceled,
    primitive_failure,
    send_event,
    send_terminal,
    start_primitive,
)
from unreal_agent.primitives.remote_sse import RemoteSSEFramer, SSEFrameDelimiter

DEFAULT_REMOTE_MAX_ATTEMPTS: Final = 5
DEFAULT_RETRYABLE_STATUS_CODES: Final = (
    408,
    425,
    429,
    500,
    502,
    503,
    504,
    520,
    521,
    522,
    523,
    524,
    529,
)
# Go's http.Client stops before sending an 11th request; httpx counts responses instead.
_MAX_REDIRECTS: Final = 9
_RETRYABLE_ERRNOS: Final = (
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTDOWN,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENETUNREACH,
    errno.EPIPE,
    errno.ETIMEDOUT,
)
_TOKEN: Final = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_QUOTED_STRING: Final = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass(slots=True)
class RemoteSSEOptions:
    max_frame_size: int = 0
    frame_delimiter: int = SSEFrameDelimiter.STRIP


@dataclass(slots=True)
class RemoteRetryPolicy:
    """Durations are seconds."""

    max_attempts: int = 0
    initial_backoff: float = 0.0
    max_backoff: float = 0.0
    retryable_status_codes: list[int] = field(default_factory=list)

    def backoff(self, failed_attempts: int) -> float:
        delay = self.initial_backoff
        for _ in range(failed_attempts - 1):
            if delay >= self.max_backoff or delay > self.max_backoff - delay:
                return self.max_backoff
            delay *= 2
        return min(delay, self.max_backoff)


@dataclass(slots=True)
class RemoteRequest:
    """Durations are seconds. Header names may repeat values; `Host` sets the request host."""

    source: SourceID = NO_SOURCE
    correlation_id: CorrelationID = NO_CORRELATION
    method: str = ""
    url: str = ""
    headers: dict[str, list[str]] = field(default_factory=dict)
    body: bytes = b""
    response_idle_timeout: float = 0.0
    sse: RemoteSSEOptions | None = None
    retry_policy: RemoteRetryPolicy = field(default_factory=RemoteRetryPolicy)


def default_remote_request(
    source: SourceID | str, correlation_id: CorrelationID | str, url: str
) -> RemoteRequest:
    return RemoteRequest(
        source=SourceID(source),
        correlation_id=CorrelationID(correlation_id),
        method="GET",
        url=url,
        headers={},
        response_idle_timeout=30.0,
        retry_policy=RemoteRetryPolicy(
            max_attempts=DEFAULT_REMOTE_MAX_ATTEMPTS,
            initial_backoff=2.0,
            max_backoff=30.0,
            retryable_status_codes=list(DEFAULT_RETRYABLE_STATUS_CODES),
        ),
    )


@dataclass(slots=True, frozen=True)
class RemoteResponseStartedResult:
    attempt: int
    status_code: int
    headers: httpx.Headers


@dataclass(slots=True, frozen=True)
class RemoteOutputResult:
    attempt: int
    offset: int
    data: bytes


@dataclass(slots=True, frozen=True)
class RemoteStreamFailureResult:
    attempt: int
    error: str


@dataclass(slots=True, frozen=True)
class RemoteRetryScheduledResult:
    attempt: int
    reason: str
    delay: float


@dataclass(slots=True, frozen=True)
class RemoteCompletedResult:
    attempt: int


def new_remote_http_client() -> httpx.AsyncClient:
    """Mirror the Go transport: env proxies, 30s dial, HTTP/2 via ALPN, 100 idle connections."""
    keepalive: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle_option = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
    if idle_option is not None:
        keepalive.append((socket.IPPROTO_TCP, idle_option, 30))
    if hasattr(socket, "TCP_KEEPINTVL"):
        keepalive.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30))
    limits = httpx.Limits(
        max_connections=None, max_keepalive_connections=100, keepalive_expiry=90.0
    )
    transport = httpx.AsyncHTTPTransport(http2=True, limits=limits, socket_options=keepalive)
    return httpx.AsyncClient(
        transport=transport,
        http2=True,
        limits=limits,
        timeout=httpx.Timeout(None, connect=30.0),
        follow_redirects=True,
        max_redirects=_MAX_REDIRECTS,
        trust_env=True,
    )


class RemoteClient:
    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self.http_client = http_client if http_client is not None else new_remote_http_client()

    async def aclose(self) -> None:
        await self.http_client.aclose()

    def send_request(self, request: RemoteRequest, events: EventSink) -> asyncio.Task[None]:
        return start_primitive(
            run_remote_request(self.http_client, request, events), name="remote.request"
        )


@dataclass(slots=True)
class PreparedRemoteRequest:
    method: str
    url: httpx.URL
    headers: list[tuple[str, str]]
    host: str | None
    decompress: bool


def prepare_remote_request(request: RemoteRequest) -> PreparedRemoteRequest:
    if not request.method:
        raise PrimitiveError("send remote request: method must be set")
    if not _TOKEN.fullmatch(request.method):
        raise PrimitiveError(
            f"send remote request: net/http: invalid method {go_quote(request.method)}"
        )
    try:
        url = httpx.URL(request.url)
    except Exception as error:
        raise wrap("send remote request", error) from None
    if not url.host or url.scheme not in ("http", "https"):
        raise PrimitiveError("send remote request: URL must be an absolute HTTP or HTTPS URL")
    if request.response_idle_timeout <= 0:
        raise PrimitiveError("send remote request: response idle timeout must be positive")
    if request.sse is not None:
        if request.sse.max_frame_size <= 0:
            raise PrimitiveError("send remote request: SSE maximum frame size must be positive")
        if request.sse.frame_delimiter not in (
            SSEFrameDelimiter.STRIP,
            SSEFrameDelimiter.PASS_THROUGH,
        ):
            raise PrimitiveError("send remote request: invalid SSE frame delimiter mode")
    policy = request.retry_policy
    if policy.max_attempts <= 0:
        raise PrimitiveError("send remote request: maximum attempts must be positive")
    if policy.initial_backoff < 0:
        raise PrimitiveError("send remote request: initial backoff must not be negative")
    if policy.max_backoff < policy.initial_backoff:
        raise PrimitiveError(
            "send remote request: maximum backoff must not be less than initial backoff"
        )
    if policy.initial_backoff == 0 and policy.max_backoff != 0:
        raise PrimitiveError(
            "send remote request: maximum backoff must be zero when initial backoff is zero"
        )
    for status_code in policy.retryable_status_codes:
        if status_code < 100 or status_code > 599:
            raise PrimitiveError(
                f"send remote request: retryable status code {status_code} "
                "must be between 100 and 599"
            )

    headers: list[tuple[str, str]] = []
    host_values: list[str] = []
    for name, values in request.headers.items():
        if not _TOKEN.fullmatch(name):
            raise PrimitiveError(
                f"send remote request: net/http: invalid header field name {go_quote(name)}"
            )
        for value in values:
            if not _valid_header_value(value):
                raise PrimitiveError(
                    "send remote request: net/http: invalid header field value for "
                    + go_quote(name)
                )
        if name.lower() == "host":
            host_values.extend(values)
            continue
        canonical = canonical_header_key(name)
        headers.extend((canonical, value) for value in values)
    if len(host_values) > 1:
        raise PrimitiveError("send remote request: Host header must have at most one value")
    decompress = request.method != "HEAD" and not any(
        name == "Accept-Encoding" for name, _ in headers
    )
    if decompress:
        headers.append(("Accept-Encoding", "gzip"))
    return PreparedRemoteRequest(
        method=request.method,
        url=url,
        headers=headers,
        host=host_values[0] if host_values else None,
        decompress=decompress,
    )


def canonical_header_key(name: str) -> str:
    """Go's `textproto.CanonicalMIMEHeaderKey` for valid token names."""
    return "-".join(part[:1].upper() + part[1:].lower() for part in name.split("-"))


def _valid_header_value(value: str) -> bool:
    return all(
        character == "\t" or (character >= " " and character != "\x7f") for character in value
    )


def is_retryable_remote_status(policy: RemoteRetryPolicy, status_code: int) -> bool:
    return status_code in policy.retryable_status_codes


def _connection_closed(error: BaseException) -> bool:
    # httpcore reports a peer closing the connection only through these messages.
    text = str(error).lower()
    return "disconnected" in text or "peer closed" in text or "connection closed" in text


def is_retryable_remote_error(error: BaseException) -> bool:
    for cause in causes(error):
        if isinstance(cause, ssl.SSLCertVerificationError):
            return False
        if isinstance(
            cause,
            DeadlineExceededError
            | UnexpectedEOFError
            | EOFError
            | httpx.TimeoutException
            | TimeoutError,
        ):
            return True
        if isinstance(cause, httpx.RemoteProtocolError) and _connection_closed(cause):
            return True
        if isinstance(cause, socket.gaierror) and cause.errno == socket.EAI_AGAIN:
            return True
        if isinstance(cause, httpx.TransportError) and _transport_cause_retryable(cause):
            return True
    return has_errno(error, *_RETRYABLE_ERRNOS)


def _transport_cause_retryable(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in _RETRYABLE_ERRNOS:
            return True
        if isinstance(current, socket.gaierror) and current.errno == socket.EAI_AGAIN:
            return True
        current = current.__cause__ or current.__context__
    return False


def _status_text(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""


def _url_error(method: str, url: str, error: BaseException) -> PrimitiveError:
    operation = method[:1] + method[1:].lower()
    wrapped = PrimitiveError(f"{operation} {go_quote(url)}: {describe(error)}")
    wrapped.__cause__ = error
    return wrapped


def _idle_timeout_error() -> PrimitiveError:
    return wrap("remote request response idle timeout", DeadlineExceededError())


async def run_remote_request(
    client: httpx.AsyncClient, request: RemoteRequest, events: EventSink
) -> None:
    try:
        prepared = prepare_remote_request(request)
    except PrimitiveError as error:
        await send_terminal(events, _remote_failure(request, error))
        return
    if cancel_requested():
        await send_terminal(events, _remote_canceled(request))
        return

    policy = request.retry_policy
    for attempt in range(1, policy.max_attempts + 1):
        response: httpx.Response | None = None
        attempt_error: BaseException | None = None
        stream_failed = False
        try:
            response = await _send_attempt(client, request, prepared)
        except asyncio.CancelledError:
            await send_terminal(events, _remote_canceled(request))
            return
        except TimeoutError:
            attempt_error = _idle_timeout_error()
        except httpx.TooManyRedirects as error:
            attempt_error = _url_error(
                request.method,
                str(error.request.url),
                PrimitiveError(f"stopped after {_MAX_REDIRECTS + 1} redirects"),
            )
        except (httpx.HTTPError, OSError, ValueError) as error:
            attempt_error = _url_error(request.method, request.url, error)
        else:
            attempt_error = await stream_remote_response(
                request, response, events, attempt, prepared.decompress
            )
            stream_failed = attempt_error is not None

        if cancel_requested():
            await send_terminal(events, _remote_canceled(request))
            return
        if stream_failed:
            assert attempt_error is not None
            if not await send_event(
                events, _remote_stream_failure(request, attempt, attempt_error)
            ):
                await send_terminal(events, _remote_canceled(request))
                return
        if attempt_error is not None and not is_retryable_remote_error(attempt_error):
            await send_terminal(
                events, _remote_failure(request, _attempt_failure(request, attempt, attempt_error))
            )
            return

        if attempt_error is None:
            assert response is not None
            status = response.status_code
            if not is_retryable_remote_status(policy, status) or attempt == policy.max_attempts:
                await send_terminal(
                    events,
                    PrimitiveEvent(
                        PrimitiveEventType.REMOTE_COMPLETED,
                        request.source,
                        request.correlation_id,
                        RemoteCompletedResult(attempt),
                    ),
                )
                return
            attempt_error = PrimitiveError(
                f"response status {status} {_status_text(status)} is retryable"
            )

        if attempt == policy.max_attempts:
            await send_terminal(
                events, _remote_failure(request, _attempt_failure(request, attempt, attempt_error))
            )
            return
        delay = policy.backoff(attempt)
        retry = PrimitiveEvent(
            PrimitiveEventType.REMOTE_RETRY_SCHEDULED,
            request.source,
            request.correlation_id,
            RemoteRetryScheduledResult(attempt, describe(attempt_error), delay),
        )
        if not await send_event(events, retry) or not await wait_for_remote_retry(delay):
            await send_terminal(events, _remote_canceled(request))
            return


async def _send_attempt(
    client: httpx.AsyncClient, request: RemoteRequest, prepared: PreparedRemoteRequest
) -> httpx.Response:
    headers = list(prepared.headers)
    if prepared.host is not None:
        headers.append(("Host", prepared.host))
    http_request = client.build_request(
        prepared.method,
        prepared.url,
        headers=headers,
        content=request.body if request.body else None,
    )
    async with asyncio.timeout(request.response_idle_timeout):
        return await client.send(http_request, stream=True)


async def wait_for_remote_retry(delay: float) -> bool:
    if cancel_requested():
        return False
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return False
    return True


async def _close_response(response: httpx.Response) -> BaseException | None:
    try:
        await response.aclose()
    except asyncio.CancelledError:
        return None
    except Exception as error:
        return error
    return None


async def _response_chunks(response: httpx.Response, decode: bool) -> AsyncGenerator[bytes]:
    # Iterate the stream directly: `aiter_raw` closes the response at EOF and
    # would surface a close error as a read error, which Go ignores after EOF.
    stream = response.stream
    if not isinstance(stream, httpx.AsyncByteStream):
        raise TypeError("remote response requires an asynchronous body stream")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if decode else None
    async for raw in stream:
        chunk = decoder.decompress(raw) if decoder is not None else raw
        for start in range(0, max(len(chunk), 1), IO_READ_CHUNK_SIZE):
            yield chunk[start : start + IO_READ_CHUNK_SIZE]
    if decoder is not None:
        tail = decoder.flush()
        if not decoder.eof:
            raise UnexpectedEOFError()
        for start in range(0, len(tail), IO_READ_CHUNK_SIZE):
            yield tail[start : start + IO_READ_CHUNK_SIZE]


async def stream_remote_response(
    request: RemoteRequest,
    response: httpx.Response,
    events: EventSink,
    attempt: int,
    decompress: bool = False,
) -> BaseException | None:
    """Deliver one response. Returns the stream error, or a cancellation error when cancelled.

    The idle timeout covers each body read and pauses while events are delivered.
    """
    decoded = decompress and response.headers.get("Content-Encoding", "").lower() == "gzip"
    headers = httpx.Headers(response.headers)
    if decoded:
        for name in ("Content-Encoding", "Content-Length"):
            if name in headers:
                del headers[name]
    started = PrimitiveEvent(
        PrimitiveEventType.REMOTE_RESPONSE_STARTED,
        request.source,
        request.correlation_id,
        RemoteResponseStartedResult(attempt, response.status_code, headers),
    )
    if not await send_event(events, started):
        await _close_response(response)
        return ContextCanceledError()

    framer: RemoteSSEFramer | None = None
    if request.sse is not None:
        content_type = response.headers.get("Content-Type", "")
        if content_type == "":
            frame_sse = 200 <= response.status_code < 300
        else:
            frame_sse = parse_media_type(content_type) == "text/event-stream"
        if frame_sse:
            framer = RemoteSSEFramer(
                request.sse.max_frame_size, SSEFrameDelimiter(request.sse.frame_delimiter)
            )

    loop = asyncio.get_running_loop()
    chunks = _response_chunks(response, decoded)
    offset = 0
    deadline = loop.time() + request.response_idle_timeout
    try:
        while True:
            read_error: BaseException | None = None
            chunk = b""
            try:
                async with asyncio.timeout_at(deadline):
                    chunk = await anext(chunks)
            except StopAsyncIteration:
                read_error = EOFError()
            except asyncio.CancelledError:
                await _close_response(response)
                return ContextCanceledError()
            except TimeoutError:
                read_error = _idle_timeout_error()
            except httpx.RemoteProtocolError as error:
                read_error = error
                if _connection_closed(error):
                    read_error = UnexpectedEOFError()
                    read_error.__cause__ = error
            except Exception as error:
                read_error = error
            if not chunk and read_error is None:
                continue

            if chunk:
                if framer is not None:
                    frames, frame_error = framer.append(chunk)
                    for frame in frames:
                        if not await _send_output(
                            request, events, attempt, frame.offset, frame.data
                        ):
                            await _close_response(response)
                            return ContextCanceledError()
                    if frame_error is not None:
                        return join(frame_error, await _close_response(response))
                else:
                    if not await _send_output(request, events, attempt, offset, chunk):
                        await _close_response(response)
                        return ContextCanceledError()
                    offset += len(chunk)

            if read_error is not None:
                if framer is not None:
                    for frame in framer.finish():
                        if not await _send_output(
                            request, events, attempt, frame.offset, frame.data
                        ):
                            await _close_response(response)
                            return ContextCanceledError()
                if isinstance(read_error, EOFError):
                    await _close_response(response)
                    return None
                return join(read_error, await _close_response(response))
            deadline = loop.time() + request.response_idle_timeout
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await chunks.aclose()


async def _send_output(
    request: RemoteRequest, events: EventSink, attempt: int, offset: int, data: bytes
) -> bool:
    return await send_event(
        events,
        PrimitiveEvent(
            PrimitiveEventType.REMOTE_OUTPUT,
            request.source,
            request.correlation_id,
            RemoteOutputResult(attempt, offset, data),
        ),
    )


def parse_media_type(value: str) -> str | None:
    """Return the lowercase media type, or None when Go's `mime.ParseMediaType` would fail."""
    media_type, _, parameters = value.partition(";")
    media_type = media_type.strip().lower()
    major, slash, minor = media_type.partition("/")
    if not _TOKEN.fullmatch(major) or (slash and not _TOKEN.fullmatch(minor)):
        return None
    rest = parameters.strip() if slash or parameters else ""
    while rest:
        name, equals, rest = rest.partition("=")
        if not equals or not _TOKEN.fullmatch(name.strip()):
            return None
        rest = rest.lstrip()
        quoted = _QUOTED_STRING.match(rest)
        if quoted is not None:
            rest = rest[quoted.end() :]
        else:
            token = _TOKEN.match(rest)
            if token is None:
                return None
            rest = rest[token.end() :]
        rest = rest.strip()
        if rest:
            if not rest.startswith(";"):
                return None
            rest = rest[1:].strip()
    return media_type


def _remote_stream_failure(
    request: RemoteRequest, attempt: int, error: BaseException
) -> PrimitiveEvent:
    return PrimitiveEvent(
        PrimitiveEventType.REMOTE_STREAM_FAILED,
        request.source,
        request.correlation_id,
        RemoteStreamFailureResult(attempt, describe(error)),
    )


def _attempt_failure(request: RemoteRequest, attempt: int, error: BaseException) -> PrimitiveError:
    return wrap(
        f"send remote request {request.method} {go_quote(request.url)}: attempt {attempt}", error
    )


def _remote_failure(request: RemoteRequest, error: BaseException) -> PrimitiveEvent:
    return primitive_failure(request.source, request.correlation_id, error)


def _remote_canceled(request: RemoteRequest) -> PrimitiveEvent:
    return primitive_canceled(request.source, request.correlation_id)
