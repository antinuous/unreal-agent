from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from unreal_agent.gojson import Timestamp
from unreal_agent.llm.responsesapi import APIError
from unreal_agent.llm.responsesapi.retry import (
    response_retry_delay,
    retry_after_header,
    retry_after_message,
    retryable_response_error,
)
from unreal_agent.llm.responsesapi.stream import provider_error
from unreal_agent.primitives import default_remote_request

SECOND = 10**9
MILLISECOND = 10**6
MICROSECOND = 10**3
POLICY = default_remote_request("test", "test", "http://example.invalid").retry_policy


@pytest.mark.parametrize(
    ("error", "want"),
    [
        pytest.param(None, False, id="no error"),
        pytest.param(
            APIError(status_code=200, type="error", code="server_error"),
            True,
            id="top-level server error",
        ),
        pytest.param(
            APIError(status_code=200, code="rate_limit_exceeded"), True, id="nested rate limit"
        ),
        pytest.param(
            APIError(status_code=200, code="server_is_overloaded"), True, id="nested overload"
        ),
        pytest.param(APIError(status_code=200, code="slow_down"), True, id="nested slow down"),
        pytest.param(
            APIError(status_code=503, code="server_is_overloaded"), True, id="HTTP overload"
        ),
        pytest.param(APIError(status_code=429, code="unknown"), True, id="HTTP transient status"),
        pytest.param(APIError(status_code=529), True, id="HTTP nonstandard status"),
        pytest.param(APIError(status_code=200, code="unknown"), True, id="unknown stream code"),
        pytest.param(APIError(status_code=200, code="429"), True, id="numeric stream code"),
        pytest.param(
            APIError(
                status_code=200,
                type="invalid_request_error",
                code="invalid_request_error",
                message="rate limit exceeded, please try again later",
            ),
            True,
            id="in-band rate limit labelled invalid request (Fireworks)",
        ),
        pytest.param(
            APIError(status_code=200, type="invalid_request_error", code="context_length_exceeded"),
            False,
            id="in-band final code keeps its label",
        ),
        pytest.param(
            APIError(status_code=400, type="invalid_request_error", code="invalid_request_error"),
            False,
            id="HTTP bad request stays final",
        ),
        pytest.param(
            APIError(status_code=429, code="insufficient_quota"),
            False,
            id="quota overrides HTTP status",
        ),
        pytest.param(
            APIError(status_code=503, code="bio_policy"), False, id="policy overrides HTTP status"
        ),
        pytest.param(
            APIError(status_code=200, code="server_error", type="invalid_request_error"),
            True,
            id="bad request type is not trusted in-band",
        ),
        pytest.param(
            APIError(status_code=503, type="authentication_error"), False, id="authentication type"
        ),
        pytest.param(
            APIError(status_code=429, type="permission_error"), False, id="permission type"
        ),
        pytest.param(
            APIError(status_code=401, code="rate_limit_exceeded"), False, id="unauthorized status"
        ),
        pytest.param(APIError(status_code=307), False, id="redirect"),
    ],
)
def test_retryable_response_error_codes(error: APIError | None, want: bool) -> None:
    assert retryable_response_error(error, POLICY.retryable_status_codes) == want


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            APIError(
                status_code=200,
                code="credit_balance_exhausted",
                message="You have no credits remaining.",
                type="insufficient_quota",
            ),
            id="credit_balance_exhausted code",
        ),
        pytest.param(
            APIError(status_code=200, message="quota", type="insufficient_quota"),
            id="insufficient_quota type",
        ),
    ],
)
def test_does_not_retry_terminal_in_band_errors(error: APIError) -> None:
    assert not retryable_response_error(error, [])


@pytest.mark.parametrize(
    ("message", "want"),
    [
        ("Please try again in 11.054s. Visit the docs.", 11054 * MILLISECOND),
        ("Please try again in 28ms.", 28 * MILLISECOND),
        ("Please try again in 1.5 milliseconds.", 1500 * MICROSECOND),
        ("TRY AGAIN IN 35 SECONDS.", 35 * SECOND),
        ("try again in 1 second", SECOND),
        ("try again in 0s", 0),
        ("try again in -1s", 0),
        ("try again in NaNs", 0),
        ("try again in 999999999999999999999s", 0),
        ("try again in 1 minute", 0),
        ("try again in 12snowflakes", 0),
        ("try again in 2", 0),
        ("try again in tomorrow", 0),
        ("quota exhausted", 0),
    ],
)
def test_retry_after_message(message: str, want: int) -> None:
    assert retry_after_message(message) == want


NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=UTC)


def http_date(moment: datetime) -> str:
    return format_datetime(moment, usegmt=True)


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (" 120 ", 120 * SECOND),
        (http_date(NOW + timedelta(minutes=1)), 60 * SECOND),
        (http_date(NOW - timedelta(minutes=1)), 0),
        ("0", 0),
        ("-1", 0),
        ("1.5", 0),
        ("9223372036", 9223372036 * SECOND),
        ("9223372037", 0),
        ("18446744073709551615", 0),
        ("999999999999999999999", 0),
        ("garbage", 0),
        ("", 0),
        ("Monday, 14-Sep-26 12:01:00 GMT", 60 * SECOND),
        ("Mon Sep 14 12:02:00 2026", 120 * SECOND),
        ("Mon Sep  4 12:00:00 2026", 0),
        ("Mon, 14 Sep 2026 12:01:00 UTC", 0),
        ("Mon, 31 Sep 2026 12:01:00 GMT", 0),
        ("mon, 14 sep 2026 12:01:00 GMT", 60 * SECOND),
        ("Mon Sep 15 12:00:00 2026", 86400 * SECOND),
    ],
)
def test_retry_after_header(value: str, want: int) -> None:
    now = Timestamp.from_datetime(NOW)
    assert retry_after_header(value, now) == want


@pytest.mark.parametrize(
    ("attempt", "code", "message", "header", "want"),
    [
        pytest.param(1, "", "", "", 2 * SECOND, id="initial"),
        pytest.param(3, "", "", "", 8 * SECOND, id="exponential"),
        pytest.param(100, "", "", "", 30 * SECOND, id="capped"),
        pytest.param(1, "server_is_overloaded", "", "", 10 * SECOND, id="overloaded"),
        pytest.param(2, "slow_down", "", "", 20 * SECOND, id="slow down"),
        pytest.param(100, "slow_down", "", "", 60 * SECOND, id="overload capped"),
        pytest.param(
            1,
            "rate_limit_exceeded",
            "try again in 3600s",
            "",
            30 * SECOND,
            id="message hint capped",
        ),
        pytest.param(1, "", "", "86400", 30 * SECOND, id="header hint capped"),
        pytest.param(
            1, "rate_limit_exceeded", "try again in 28ms", "", 28 * MILLISECOND, id="short hint"
        ),
        pytest.param(
            1, "rate_limit_exceeded", "try again in 20s", "10", 20 * SECOND, id="longer hint wins"
        ),
        pytest.param(
            1,
            "rate_limit_exceeded",
            "try again in 10s",
            "20",
            20 * SECOND,
            id="header beats message",
        ),
        pytest.param(
            1,
            "rate_limit_exceeded",
            "try again in 5s",
            "invalid",
            5 * SECOND,
            id="invalid header uses message",
        ),
        pytest.param(
            1,
            "server_error",
            "try again in 120s",
            "",
            2 * SECOND,
            id="ignore non-rate-limit message",
        ),
    ],
)
def test_response_retry_delay(
    attempt: int, code: str, message: str, header: str, want: int
) -> None:
    error = APIError(code=code, message=message)
    headers = httpx.Headers({"Retry-After": header})
    now = Timestamp.now()
    got = response_retry_delay(POLICY, attempt, error, headers, now, 0)
    assert got == want
    for jitter in (0.25, 0.5, 0.99):
        delay = response_retry_delay(POLICY, attempt, error, headers, now, jitter)
        assert want * 4 // 5 <= delay <= want
        if header or code == "rate_limit_exceeded":
            assert delay == got, "jitter shortened the server hint"


@pytest.mark.parametrize("status_code", [520, 521, 522, 523, 524, 529])
def test_provider_error_preserves_nonstandard_status_code(status_code: int) -> None:
    error = provider_error(status_code, b"")
    assert error.status_code == status_code
    assert f"status {status_code}" in str(error)


def test_provider_error_formats() -> None:
    assert (
        str(provider_error(418, b""))
        == "responses API request failed with status 418: I'm a teapot"
    )
    assert str(provider_error(413, b"null")) == "responses API request failed with status 413: null"
    error = provider_error(
        400, b'{"error":{"code":"bad","message":"broken","param":"model","type":"invalid"}}'
    )
    assert (error.status_code, error.code, error.message, error.param, error.type) == (
        400,
        "bad",
        "broken",
        "model",
        "invalid",
    )
    assert str(error) == "responses API error bad: broken"
    assert str(error.wrap("create response")) == "create response: responses API error bad: broken"
    assert str(APIError(message="x")) == "responses API request failed: x"
