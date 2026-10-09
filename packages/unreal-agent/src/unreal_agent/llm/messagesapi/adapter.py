import asyncio
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import httpx

from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.gojson import Timestamp
from unreal_agent.llm.adapter import RequestOptions
from unreal_agent.llm.messagesapi._wire import parse_media_type
from unreal_agent.llm.messagesapi.errors import StreamEndedError, TransportError
from unreal_agent.llm.messagesapi.request import request_body
from unreal_agent.llm.messagesapi.response import decode_response
from unreal_agent.llm.messagesapi.retry import response_retry_delay, retryable_response_error
from unreal_agent.llm.messagesapi.stream import StreamingState, header, provider_error
from unreal_agent.llm.model import Request, Response
from unreal_agent.llm.responsesapi._wire import trim_space
from unreal_agent.primitives import (
    DEFAULT_REMOTE_MAX_ATTEMPTS,
    TERMINAL_EVENT_TYPES,
    Channel,
    CorrelationID,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    RemoteClient,
    RemoteOutputResult,
    RemoteRequest,
    RemoteResponseStartedResult,
    RemoteSSEOptions,
    SourceID,
    SSEFrameDelimiter,
    TimerRequest,
    default_remote_request,
    schedule_timer,
    sse_data,
)
from unreal_agent.primitives import timer as _timer

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "MAX_SSE_FRAME_BYTES",
    "MODEL_RESPONSE_IDLE_TIMEOUT",
    "REMOTE_SOURCE",
    "CacheTTL",
    "Config",
    "Exchange",
    "MessagesAdapter",
    "check_config",
]

REMOTE_SOURCE: Final = SourceID("llm.messagesapi")
DEFAULT_MAX_ATTEMPTS: Final = DEFAULT_REMOTE_MAX_ATTEMPTS
MODEL_RESPONSE_IDLE_TIMEOUT: Final = 30 * 60.0
MAX_SSE_FRAME_BYTES: Final = 256 << 20
MAX_ERROR_BODY_BYTES: Final = 1 << 20

_FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"


class CacheTTL(StrEnum):
    NONE = ""
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"

    @staticmethod
    def is_valid(value: str) -> bool:
        return value in CacheTTL._value2member_map_


@dataclass(frozen=True, slots=True)
class Exchange:
    """One traced exchange: the request JSON and the assembled response JSON or provider error."""

    request_body: bytes
    status_code: int
    response_body: bytes


@dataclass(slots=True)
class Config:
    """Adapter configuration.

    `fallback` None enables server-side fallback. `max_attempts` None uses
    DEFAULT_MAX_ATTEMPTS. `clock` and `jitter` drive
    retry backoff and default to the timer primitive's wall clock and
    `random.random`.
    """

    endpoint: str = ""
    headers: Mapping[str, Sequence[str]] = field(default_factory=dict)
    fallback: bool | None = None
    cache_ttl: str = CacheTTL.NONE
    max_attempts: int | None = None
    trace: Callable[[Exchange], None] | None = None
    clock: Callable[[], Timestamp] | None = None
    jitter: Callable[[], float] | None = None


@dataclass(slots=True)
class _Attempt:
    status: int = 0
    headers: httpx.Headers | None = None
    body: bytes = b""
    err: BaseException | None = None
    retry: bool = False


def check_config(config: Config) -> int:
    """Validate the configuration and return the attempt limit. Raises ValueError."""
    if not go_trim_space(config.endpoint):
        raise ValueError("messages API endpoint must be set")
    if not CacheTTL.is_valid(config.cache_ttl):
        raise ValueError(f"unsupported cache TTL {go_quote(config.cache_ttl)}")
    max_attempts = DEFAULT_MAX_ATTEMPTS if config.max_attempts is None else config.max_attempts
    if max_attempts <= 0:
        raise ValueError("max attempts must be positive")
    return max_attempts


class MessagesAdapter:
    def __init__(self, remote: RemoteClient | None, config: Config) -> None:
        if remote is None:
            raise ValueError("remote client must be set")
        self.max_attempts = check_config(config)
        self.remote = remote
        self.endpoint = config.endpoint
        self.headers = {
            name: list(values)
            for name, values in config.headers.items()
            if name.lower() != "accept"
        }
        self.headers["Accept"] = ["text/event-stream"]
        self.fallback = True if config.fallback is None else config.fallback
        if self.fallback:
            name = next(
                (n for n in self.headers if n.lower() == "anthropic-beta"), "Anthropic-Beta"
            )
            self.headers.setdefault(name, []).append(_FALLBACK_BETA)
        self.cache_ttl = config.cache_ttl
        self.trace = config.trace
        self._clock = config.clock or (lambda: _timer.wall_clock())
        self._jitter = config.jitter or random.random

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        body = request_body(request, fallback=self.fallback, cache_ttl=self.cache_ttl)
        result = await self._exchange(body)
        if self.trace is not None and result.body:
            self.trace(Exchange(body, result.status, result.body))
        if result.err is not None:
            raise result.err
        return decode_response(result.body)

    def remote_request(self, body: bytes) -> RemoteRequest:
        request = default_remote_request(REMOTE_SOURCE, str(uuid.uuid4()), self.endpoint)
        request.method = "POST"
        request.headers = {name: list(values) for name, values in self.headers.items()}
        request.body = body
        request.response_idle_timeout = MODEL_RESPONSE_IDLE_TIMEOUT
        request.retry_policy.max_attempts = 1
        request.sse = RemoteSSEOptions(
            max_frame_size=MAX_SSE_FRAME_BYTES, frame_delimiter=SSEFrameDelimiter.STRIP
        )
        return request

    async def _exchange(self, body: bytes) -> _Attempt:
        attempt = 1
        while True:
            request = self.remote_request(body)
            result = await self._exchange_attempt(request)
            if not result.retry or attempt >= self.max_attempts:
                return result
            now = self._clock()
            delay = response_retry_delay(
                request.retry_policy, attempt, result.headers, now, self._jitter()
            )
            await _wait_backoff(request, Timestamp(now.unix_ns + delay))
            attempt += 1

    async def _exchange_attempt(self, request: RemoteRequest) -> _Attempt:
        events: Channel[PrimitiveEvent] = Channel()
        task = self.remote.send_request(request, events)
        result = _Attempt()
        state = StreamingState()
        streaming = False
        parse_error: BaseException | None = None
        body = bytearray()
        while True:
            event = await _receive(events, task)
            transport_error: BaseException | None = None
            match event.type:
                case PrimitiveEventType.REMOTE_RESPONSE_STARTED:
                    started: RemoteResponseStartedResult = event.result
                    result.status, result.headers = started.status_code, started.headers
                    success = 200 <= result.status < 300
                    content_type = header(result.headers, "Content-Type")
                    try:
                        media_type = parse_media_type(content_type)
                    except ValueError as error:
                        media_type = ""
                        if content_type:
                            parse_error = ValueError(f"invalid response content type: {error}")
                    if parse_error is None and success and media_type != "text/event-stream":
                        parse_error = ValueError(
                            "expected text/event-stream response, got content type "
                            + go_quote(content_type)
                        )
                    if parse_error is not None:
                        task.cancel()
                    streaming = success and media_type == "text/event-stream"
                    continue
                case PrimitiveEventType.REMOTE_OUTPUT:
                    if parse_error is not None or state.terminal or state.api_error is not None:
                        continue
                    output: RemoteOutputResult = event.result
                    if streaming:
                        data = trim_space(sse_data(output.data) or b"")
                        if data:
                            try:
                                state.observe(data)
                            except ValueError as error:
                                parse_error = error
                        if state.terminal or state.api_error is not None or parse_error:
                            # Drain the primitive's terminal event before returning or retrying.
                            task.cancel()
                    elif len(body) + len(output.data) > MAX_ERROR_BODY_BYTES:
                        parse_error = ValueError(
                            f"messages API response exceeds {MAX_ERROR_BODY_BYTES} bytes"
                        )
                        task.cancel()
                    else:
                        body += output.data
                    continue
                case PrimitiveEventType.REMOTE_COMPLETED:
                    pass
                case PrimitiveEventType.FAILED:
                    transport_error = _remote_failure_error(event)
                case PrimitiveEventType.CANCELED:
                    transport_error = asyncio.CancelledError()
                case _:
                    continue

            result.body = bytes(body)
            if parse_error is not None:
                result.err = parse_error
            elif state.terminal:
                result.body = state.unwrap()
            elif state.api_error is not None:
                result.body = state.body
                api_error = provider_error(result.status, result.body, result.headers)
                result.err = api_error
                result.retry = retryable_response_error(api_error, result.headers)
            elif result.status != 0 and not 200 <= result.status < 300:
                api_error = provider_error(result.status, result.body, result.headers)
                result.err = api_error
                result.retry = retryable_response_error(api_error, result.headers)
            elif transport_error is not None:
                result.err = transport_error
                result.retry = event.type == PrimitiveEventType.FAILED
            elif streaming:
                try:
                    result.body = state.unwrap()
                except StreamEndedError as error:
                    result.body, result.err, result.retry = b"", error, True
            return result


async def _wait_backoff(request: RemoteRequest, deadline: Timestamp) -> None:
    events: Channel[PrimitiveEvent] = Channel()
    task = schedule_timer(
        TimerRequest(
            source=request.source,
            correlation_id=CorrelationID(request.correlation_id + ":backoff"),
            deadline=deadline,
        ),
        events,
    )
    event = await _receive(events, task)
    match event.type:
        case PrimitiveEventType.TIMER_FIRED:
            return
        case PrimitiveEventType.CANCELED:
            raise asyncio.CancelledError
        case PrimitiveEventType.FAILED:
            raise _remote_failure_error(event)
        case _:
            raise TransportError(f"unexpected retry timer event {go_quote(event.type)}")


async def _receive(events: Channel[PrimitiveEvent], task: asyncio.Task[None]) -> PrimitiveEvent:
    """Take the next event. On cancellation, cancel the primitive and drain it before raising."""
    try:
        return await events.get()
    except asyncio.CancelledError:
        task.cancel()
        while True:
            try:
                event = await events.get()
            except asyncio.CancelledError:
                continue
            if event.type in TERMINAL_EVENT_TYPES:
                break
        raise


def _remote_failure_error(event: PrimitiveEvent) -> TransportError:
    if not isinstance(event.result, PrimitiveFailureResult):
        return TransportError("remote request failed with an invalid result")
    return TransportError(event.result.error)
