"""The Anthropic Messages API adapter.

`MessagesAdapter.respond` streams one message over SSE through the remote
primitive, assembles it from its events, and returns the normalized response.
Each attempt is a fresh remote request with a new UUID correlation ID and
transport retries off; the adapter retries itself, waiting through the timer
primitive.

Errors: request encoding and response decoding raise ValueError. A provider
error (HTTP or in-band) raises APIError. A failed transport raises
TransportError (StreamEndedError when the stream ends before message_stop).
Cancelling the calling task cancels the request and raises
asyncio.CancelledError.
"""

from unreal_agent.llm.messagesapi.adapter import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_SSE_FRAME_BYTES,
    MODEL_RESPONSE_IDLE_TIMEOUT,
    REMOTE_SOURCE,
    CacheTTL,
    Config,
    Exchange,
    MessagesAdapter,
)
from unreal_agent.llm.messagesapi.errors import APIError, StreamEndedError, TransportError
from unreal_agent.llm.messagesapi.request import request_body
from unreal_agent.llm.messagesapi.response import decode_response

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "MAX_SSE_FRAME_BYTES",
    "MODEL_RESPONSE_IDLE_TIMEOUT",
    "REMOTE_SOURCE",
    "APIError",
    "CacheTTL",
    "Config",
    "Exchange",
    "MessagesAdapter",
    "StreamEndedError",
    "TransportError",
    "decode_response",
    "request_body",
]
