"""The OpenAI Responses API adapter, shared by the OpenAI-compatible clients.

`ResponsesAdapter.respond` streams one model response over SSE through the
remote primitive and returns only the terminal response. Each attempt is a
fresh remote request with a new UUID correlation ID and transport retries
off; the adapter retries itself, waiting through the timer primitive.

Errors: request encoding and response decoding raise ValueError. A provider
error (HTTP or in-band) raises APIError. A failed transport raises
TransportError (StreamEndedError when the stream ends early). Cancelling the
calling task cancels the request and raises asyncio.CancelledError.
"""

from unreal_agent.llm.responsesapi.adapter import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_SSE_FRAME_BYTES,
    MODEL_RESPONSE_IDLE_TIMEOUT,
    REMOTE_SOURCE,
    CacheKeyPlacement,
    Config,
    Exchange,
    ResponsesAdapter,
)
from unreal_agent.llm.responsesapi.errors import APIError, StreamEndedError, TransportError
from unreal_agent.llm.responsesapi.request import request_body
from unreal_agent.llm.responsesapi.response import decode_response

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "MAX_SSE_FRAME_BYTES",
    "MODEL_RESPONSE_IDLE_TIMEOUT",
    "REMOTE_SOURCE",
    "APIError",
    "CacheKeyPlacement",
    "Config",
    "Exchange",
    "ResponsesAdapter",
    "StreamEndedError",
    "TransportError",
    "decode_response",
    "request_body",
]
