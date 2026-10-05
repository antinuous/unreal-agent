import asyncio
import hashlib
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import httpx

from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.gojson import Timestamp
from unreal_agent.llm.adapter import RequestOptions
from unreal_agent.llm.model import Request, Response
from unreal_agent.llm.responsesapi._wire import trim_space
from unreal_agent.llm.responsesapi.errors import APIError, StreamEndedError, TransportError
from unreal_agent.llm.responsesapi.request import request_body
from unreal_agent.llm.responsesapi.response import decode_response
from unreal_agent.llm.responsesapi.retry import response_retry_delay, retryable_response_error
from unreal_agent.llm.responsesapi.stream import ResponseState, provider_error
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
    "CacheKeyPlacement",
    "Config",
    "Exchange",
    "ResponsesAdapter",
]

REMOTE_SOURCE: Final = SourceID("llm.responsesapi")
DEFAULT_MAX_ATTEMPTS: Final = DEFAULT_REMOTE_MAX_ATTEMPTS
# Reasoning can produce multi-minute gaps between events.
MODEL_RESPONSE_IDLE_TIMEOUT: Final = 30 * 60.0
MAX_SSE_FRAME_BYTES: Final = 256 << 20
MAX_ERROR_BODY_BYTES: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class Exchange:
    """One traced exchange: the request JSON and the terminal response JSON or HTTP error body."""

    request_body: bytes
    status_code: int
    response_body: bytes


@dataclass(frozen=True, slots=True)
class CacheKeyPlacement:
    header: str = ""
    use_prompt_cache_key_field: bool = False


@dataclass(slots=True)
class Config:
    """Adapter configuration.

    `max_attempts` None uses DEFAULT_MAX_ATTEMPTS. `extensions` are provider
    top-level fields (raw JSON) merged into every request body, such as
    OpenRouter's prompt caching; a key that names a standard field is rejected
    per request. `clock` and `jitter` drive retry backoff and default to the
    timer primitive's wall clock and `random.random`.
    """

    endpoint: str = ""
    headers: Mapping[str, Sequence[str]] = field(default_factory=dict)
    cache_key_placement: CacheKeyPlacement = field(default_factory=CacheKeyPlacement)
    max_attempts: int | None = None
    trace: Callable[[Exchange], None] | None = None
    extensions: Mapping[str, bytes] = field(default_factory=dict)
    clock: Callable[[], Timestamp] | None = None
    jitter: Callable[[], float] | None = None


@dataclass(slots=True)
class _Attempt:
    status: int = 0
    headers: httpx.Headers | None = None
    body: bytes = b""
    err: BaseException | None = None
    api_error: APIError | None = None
    retry: bool = False


def check_config(config: Config) -> int:
    """Validate the configuration and return the attempt limit. Raises ValueError."""
    if not go_trim_space(config.endpoint):
        raise ValueError("responses API endpoint must be set")
    max_attempts = DEFAULT_MAX_ATTEMPTS if config.max_attempts is None else config.max_attempts
    if max_attempts <= 0:
        raise ValueError("max attempts must be positive")
    return max_attempts


class ResponsesAdapter:
    def __init__(self, remote: RemoteClient | None, config: Config) -> None:
        if remote is None:
            raise ValueError("remote client must be set")
        max_attempts = check_config(config)
        self.remote = remote
        self.endpoint = config.endpoint
        self.headers = {name: list(values) for name, values in config.headers.items()}
        self.trace = config.trace
        self.cache_key_placement = config.cache_key_placement
        self.max_attempts = max_attempts
        self.extensions = dict(config.extensions)
        self._clock = config.clock or (lambda: _timer.wall_clock())
        self._jitter = config.jitter or random.random

    async def respond(self, request: Request, options: RequestOptions) -> Response:
        key = ""
        if options.CacheKey:
            key = hashlib.sha256(options.CacheKey.encode("utf-8", "surrogateescape")).hexdigest()
        prompt_cache_key = key if self.cache_key_placement.use_prompt_cache_key_field else ""
        body = request_body(request, prompt_cache_key, self.extensions)
        status_code, response_body = await self._exchange(body, key)
        if self.trace is not None:
            self.trace(Exchange(body, status_code, response_body))
        if status_code < 200 or status_code >= 300:
            raise provider_error(status_code, response_body).wrap("create response")
        return decode_response(response_body)

    def remote_request(self, body: bytes, cache_key: str) -> RemoteRequest:
        request = default_remote_request(REMOTE_SOURCE, str(uuid.uuid4()), self.endpoint)
        request.method = "POST"
        request.body = body
        request.headers = {
            name: list(values) for name, values in self.headers.items() if name.lower() != "accept"
        }
        request.headers["Accept"] = ["text/event-stream"]
        if self.cache_key_placement.header and cache_key:
            request.headers[self.cache_key_placement.header] = [cache_key]
        request.response_idle_timeout = MODEL_RESPONSE_IDLE_TIMEOUT
        request.retry_policy.max_attempts = 1
        request.sse = RemoteSSEOptions(
            max_frame_size=MAX_SSE_FRAME_BYTES, frame_delimiter=SSEFrameDelimiter.STRIP
        )
        return request

    async def _exchange(self, body: bytes, cache_key: str) -> tuple[int, bytes]:
        attempt = 1
        while True:
            request = self.remote_request(body, cache_key)
            result = await self._exchange_attempt(request)
            if not result.retry or attempt >= self.max_attempts:
                if result.err is not None:
                    raise result.err
                return result.status, result.body
            now = self._clock()
            delay = response_retry_delay(
                request.retry_policy, attempt, result.api_error, result.headers, now, self._jitter()
            )
            await _wait_backoff(request, Timestamp(now.unix_ns + delay))
            attempt += 1

    async def _exchange_attempt(self, request: RemoteRequest) -> _Attempt:
        events: Channel[PrimitiveEvent] = Channel()
        task = self.remote.send_request(request, events)
        result = _Attempt()
        state = ResponseState()
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
                    streaming = 200 <= result.status < 300
                    continue
                case PrimitiveEventType.REMOTE_OUTPUT:
                    if parse_error is not None or state.terminal or state.err is not None:
                        continue
                    output: RemoteOutputResult = event.result
                    if streaming:
                        payload = trim_space(sse_data(output.data) or b"")
                        if payload and payload != b"[DONE]":
                            try:
                                state.observe(payload)
                            except (ValueError, APIError) as error:
                                parse_error = error
                    elif len(body) + len(output.data) > MAX_ERROR_BODY_BYTES:
                        parse_error = ValueError("responses API error response exceeds 1 MiB")
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
                return result
            if state.terminal:
                try:
                    result.body = state.unwrap()
                except (ValueError, APIError, TransportError) as error:
                    result.body, result.err = b"", error
                    return result
                result.api_error = state.failure
            elif state.err is not None:
                result.err = result.api_error = state.err
            elif result.status != 0 and not streaming:
                result.api_error = provider_error(result.status, result.body)
            elif transport_error is not None:
                result.err = transport_error
                result.retry = event.type == PrimitiveEventType.FAILED
                return result
            else:
                try:
                    result.body = state.unwrap()
                except (ValueError, APIError, TransportError) as error:
                    result.body, result.err = b"", error
                    result.retry = isinstance(error, StreamEndedError)
                return result
            result.retry = retryable_response_error(
                result.api_error, request.retry_policy.retryable_status_codes
            )
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
