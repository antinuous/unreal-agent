"""Retry classification and backoff. Durations are integer nanoseconds, like Go's time.Duration."""

import math
import re
from collections.abc import Mapping
from typing import Final

import httpx

from unreal_agent._go import go_trim_space
from unreal_agent.gojson import Timestamp
from unreal_agent.llm.messagesapi.errors import APIError
from unreal_agent.llm.messagesapi.stream import header
from unreal_agent.llm.responsesapi.retry import MILLISECOND, SECOND, _parse_http_time
from unreal_agent.primitives import RemoteRetryPolicy

__all__ = ["response_retry_delay", "retryable_response_error"]

_RETRYABLE_TYPES: Final = frozenset(
    {"rate_limit_error", "overloaded_error", "api_error", "timeout_error"}
)
_DECIMAL: Final = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_HEX: Final = re.compile(
    r"[+-]?0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][+-]?\d+", re.ASCII
)

type Headers = httpx.Headers | Mapping[str, str] | None


def retryable_response_error(error: APIError, headers: Headers) -> bool:
    match header(headers, "X-Should-Retry"):
        case "true":
            return True
        case "false":
            return False
    if 200 <= error.status_code < 300:
        return error.type in _RETRYABLE_TYPES
    return error.status_code in (408, 409, 429) or error.status_code >= 500


def _parse_float(value: str) -> float | None:
    """Go's `strconv.ParseFloat` for finite decimal and hexadecimal numbers."""
    if _DECIMAL.fullmatch(value):
        return float(value)
    if _HEX.fullmatch(value):
        return float.fromhex(value)
    return None


def response_retry_delay(
    policy: RemoteRetryPolicy, attempt: int, headers: Headers, now: Timestamp, jitter: float
) -> int:
    """Return the wait before the next attempt in nanoseconds.

    A server hint is capped by the policy and not jittered. Jitter in [0, 1)
    shortens a computed backoff by up to 20%.
    """
    maximum = round(policy.max_backoff * SECOND)
    for name, unit in (("Retry-After-Ms", MILLISECOND), ("Retry-After", SECOND)):
        number = _parse_float(go_trim_space(header(headers, name)))
        if number is not None and number > 0 and not math.isinf(number):
            return int(min(number * float(unit), float(maximum)))
    deadline = _parse_http_time(header(headers, "Retry-After"))
    if deadline is not None and deadline * SECOND > now.unix_ns:
        return min(deadline * SECOND - now.unix_ns, maximum)
    delay = _backoff(round(policy.initial_backoff * SECOND), maximum, attempt)
    return delay - int(float(delay // 5) * jitter)


def _backoff(initial: int, maximum: int, failed_attempts: int) -> int:
    delay = initial
    for _ in range(failed_attempts - 1):
        if delay >= maximum or delay > maximum - delay:
            return maximum
        delay *= 2
    return min(delay, maximum)
