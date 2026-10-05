"""Retry classification and backoff. Durations are integer nanoseconds, like Go's time.Duration."""

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Final

import httpx

from unreal_agent.gojson import Timestamp
from unreal_agent.llm.responsesapi._wire import trim_space_text
from unreal_agent.llm.responsesapi.errors import APIError
from unreal_agent.primitives import RemoteRetryPolicy

__all__ = [
    "response_retry_delay",
    "retry_after_header",
    "retry_after_message",
    "retryable_response_error",
]

SECOND: Final = 10**9
MILLISECOND: Final = 10**6
_MAX_DURATION: Final = 2**63 - 1

_RETRY_AFTER_MESSAGE: Final = re.compile(
    r"\btry again in[\t\n\f\r ]*(\d+(?:\.\d+)?)[\t\n\f\r ]*(ms|milliseconds?|s|seconds?)\b",
    re.IGNORECASE | re.ASCII,
)

_FINAL_CODES: Final = frozenset(
    {
        "context_length_exceeded",
        "insufficient_quota",
        "usage_not_included",
        "usage_limit_reached",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "cyber_policy",
        "misalignment_policy_violation",
        "invalid_prompt",
        "bio_policy",
        "invalid_api_key",
        "invalid_token",
    }
)
_FINAL_TYPES: Final = frozenset({"authentication_error", "permission_error", "insufficient_quota"})


def retryable_response_error(error: APIError | None, retryable_statuses: Sequence[int]) -> bool:
    """Providers report failures inconsistently, so this classifier fails open, favoring retries."""
    if error is None:
        return False
    if error.code in _FINAL_CODES or error.type in _FINAL_TYPES:
        return False
    if error.status_code < 200 or error.status_code >= 300:
        return error.status_code in retryable_statuses
    return True


def response_retry_delay(
    policy: RemoteRetryPolicy,
    attempt: int,
    error: APIError | None,
    headers: httpx.Headers | None,
    now: Timestamp,
    jitter: float,
) -> int:
    """Return the wait before the next attempt in nanoseconds.

    A server hint is capped by the policy and not jittered. Jitter in [0, 1)
    shortens a computed backoff by up to 20%.
    """
    retry_after = ""
    if headers is not None:
        values = headers.get_list("Retry-After")
        retry_after = values[0] if values else ""
    hint = retry_after_header(retry_after, now)
    if error is not None and error.code == "rate_limit_exceeded":
        hint = max(hint, retry_after_message(error.message))
    initial = round(policy.initial_backoff * SECOND)
    maximum = round(policy.max_backoff * SECOND)
    if hint > 0:
        return min(hint, maximum)
    if error is not None and error.code in ("server_is_overloaded", "slow_down"):
        # Unlike the Codex CLI, unattended runs retry overloads with a longer backoff.
        initial, maximum = 10 * SECOND, 60 * SECOND
    delay = _backoff(initial, maximum, attempt)
    return delay - int(float(delay // 5) * jitter)


def _backoff(initial: int, maximum: int, failed_attempts: int) -> int:
    delay = initial
    for _ in range(failed_attempts - 1):
        if delay >= maximum or delay > maximum - delay:
            return maximum
        delay *= 2
    return min(delay, maximum)


def retry_after_header(value: str, now: Timestamp) -> int:
    value = trim_space_text(value)
    if value.isascii() and value.isdigit():
        seconds = int(value)
        if seconds > 2**64 - 1 or seconds > _MAX_DURATION // SECOND:
            return 0
        return seconds * SECOND
    deadline = _parse_http_time(value)
    if deadline is None:
        return 0
    return max(0, min(deadline * SECOND - now.unix_ns, _MAX_DURATION))


_HTTP_TIME_FORMATS: Final = (
    re.compile(
        r"(?P<weekday>[A-Za-z]{3}), (?P<day>\d{2}) (?P<month>[A-Za-z]{3}) (?P<year>\d{4}) "
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2}) GMT",
        re.ASCII,
    ),
    re.compile(
        r"(?P<weekday>[A-Za-z]{6,9}), (?P<day>\d{2})-(?P<month>[A-Za-z]{3})-(?P<year>\d{2}) "
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2}) [A-Z]{3,5}",
        re.ASCII,
    ),
    re.compile(
        r"(?P<weekday>[A-Za-z]{3}) (?P<month>[A-Za-z]{3}) (?P<day> ?\d{1,2}) "
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2}) (?P<year>\d{4})",
        re.ASCII,
    ),
)
_MONTHS: Final = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)
_SHORT_DAYS: Final = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_LONG_DAYS: Final = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _parse_http_time(value: str) -> int | None:
    """Go's `http.ParseTime`: RFC 1123 (GMT), RFC 850, or asctime. Returns Unix seconds."""
    for index, pattern in enumerate(_HTTP_TIME_FORMATS):
        match = pattern.fullmatch(value)
        if match is None:
            continue
        days = _LONG_DAYS if index == 1 else _SHORT_DAYS
        if match["weekday"].lower() not in days or match["month"].lower() not in _MONTHS:
            continue
        year = int(match["year"])
        if index == 1:
            year += 1900 if year >= 69 else 2000
        try:
            moment = datetime(
                year,
                _MONTHS.index(match["month"].lower()) + 1,
                int(match["day"].strip()),
                int(match["hour"]),
                int(match["minute"]),
                int(match["second"]),
                tzinfo=UTC,
            )
        except ValueError:
            continue
        return int(moment.timestamp())
    return None


def retry_after_message(message: str) -> int:
    match = _RETRY_AFTER_MESSAGE.search(message)
    if match is None:
        return 0
    unit = MILLISECOND if match[2].lower().startswith("m") else SECOND
    return _parse_duration(match[1], unit)


def _parse_duration(number: str, unit: int) -> int:
    """Go's `time.ParseDuration` for one unsigned decimal quantity. Overflow gives 0."""
    whole, _, fraction = number.partition(".")
    value = int(whole)
    if value > (1 << 63) // unit:
        return 0
    value *= unit
    if fraction:
        digits = 0
        scale = 1.0
        overflow = False
        for character in fraction:
            if overflow:
                continue
            if digits > ((1 << 63) - 1) // 10:
                overflow = True
                continue
            next_digits = digits * 10 + int(character)
            if next_digits > 1 << 63:
                overflow = True
                continue
            digits = next_digits
            scale *= 10
        value += int(float(digits) * (float(unit) / scale))
        if value > 1 << 63:
            return 0
    if value > _MAX_DURATION:
        return 0
    return value
