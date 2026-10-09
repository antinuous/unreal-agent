from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from unreal_agent.gojson import Timestamp
from unreal_agent.llm.messagesapi import APIError
from unreal_agent.llm.messagesapi.retry import response_retry_delay, retryable_response_error
from unreal_agent.llm.messagesapi.stream import provider_error
from unreal_agent.primitives import default_remote_request

SECOND = 10**9
MILLISECOND = 10**6
POLICY = default_remote_request("test", "test", "http://example.invalid").retry_policy
MAX_BACKOFF = round(POLICY.max_backoff * SECOND)
NOW = datetime(2026, 9, 25, tzinfo=UTC)


@pytest.mark.parametrize(
    ("headers", "attempt", "want"),
    [
        ({"Retry-After": "2.5"}, 1, 2500 * MILLISECOND),
        ({"Retry-After-Ms": "150", "Retry-After": "5"}, 1, 150 * MILLISECOND),
        ({"Retry-After": "120"}, 1, MAX_BACKOFF),
        (
            {"Retry-After": format_datetime(NOW + timedelta(seconds=10), usegmt=True)},
            1,
            10 * SECOND,
        ),
        ({"Retry-After": "invalid"}, 1, 2 * SECOND),
        ({"Retry-After": "NaN"}, 1, 2 * SECOND),
        ({"Retry-After": "+Inf"}, 1, 2 * SECOND),
        (None, 2, 4 * SECOND),
        (None, 20, MAX_BACKOFF),
    ],
)
def test_retry_delay(headers: dict[str, str] | None, attempt: int, want: int) -> None:
    now = Timestamp.from_datetime(NOW)
    wire = httpx.Headers(headers) if headers is not None else None
    assert response_retry_delay(POLICY, attempt, wire, now, 0) == want


@pytest.mark.parametrize(
    ("status", "error_type", "should_retry", "want"),
    [
        (200, "overloaded_error", "", True),
        (200, "rate_limit_error", "", True),
        (200, "api_error", "", True),
        (200, "timeout_error", "", True),
        (200, "invalid_request_error", "", False),
        (200, "invalid_request_error", "true", True),
        (408, "", "", True),
        (409, "", "", True),
        (429, "", "", True),
        (500, "", "", True),
        (529, "", "", True),
        (503, "", "false", False),
        (400, "", "", False),
        (401, "", "", False),
    ],
)
def test_retryable_response_error(
    status: int, error_type: str, should_retry: str, want: bool
) -> None:
    headers = httpx.Headers({"X-Should-Retry": should_retry} if should_retry else {})
    error = APIError(status_code=status, type=error_type)
    assert retryable_response_error(error, headers) is want


@pytest.mark.parametrize(
    ("body", "want"), [("", "Bad Gateway"), ("<html>upstream failed</html>", "upstream failed")]
)
def test_provider_error_fallback(body: str, want: str) -> None:
    error = provider_error(502, body.encode(), None)
    assert error.status_code == 502
    assert want in error.message


def test_provider_error_preserves_metadata_with_empty_message() -> None:
    body = (
        b'{"type":"error","request_id":"req-body","error":{"type":"overloaded_error","message":""}}'
    )
    error = provider_error(200, body, httpx.Headers({"Request-Id": "req-header"}))
    assert (error.type, error.request_id, error.message) == (
        "overloaded_error",
        "req-body",
        body.decode(),
    )


def test_api_error_text() -> None:
    assert str(APIError(401, "authentication_error", "bad key")) == (
        "messages API error authentication_error: bad key"
    )
    assert str(APIError(502, "", "Bad Gateway")) == (
        "messages API request failed with status 502: Bad Gateway"
    )
