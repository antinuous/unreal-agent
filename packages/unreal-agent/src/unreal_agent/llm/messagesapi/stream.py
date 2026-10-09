from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from unreal_agent._go import go_trim_space
from unreal_agent.gojson import JSONError
from unreal_agent.llm.messagesapi._wire import ErrorEnvelope, StreamEvent
from unreal_agent.llm.messagesapi.builder import MessageBuilder
from unreal_agent.llm.messagesapi.errors import APIError, StreamEndedError
from unreal_agent.llm.responsesapi._wire import decode_object, status_text
from unreal_agent.primitives._errors import UnexpectedEOFError

__all__ = ["StreamingState", "header", "provider_error"]


def header(headers: httpx.Headers | Mapping[str, str] | None, name: str) -> str:
    """Go's `http.Header.Get`: the first value, or empty."""
    if headers is None:
        return ""
    if isinstance(headers, httpx.Headers):
        values = headers.get_list(name)
        return values[0] if values else ""
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return ""


def provider_error(
    status_code: int, body: bytes, headers: httpx.Headers | Mapping[str, str] | None
) -> APIError:
    result = APIError(status_code=status_code, request_id=header(headers, "Request-Id"))
    try:
        envelope = decode_object(body, ErrorEnvelope)
    except JSONError:
        envelope = None
    if envelope is not None:
        error = envelope.error
        if error is not None:
            result.type, result.message = error.type or "", error.message or ""
        if envelope.request_id:
            result.request_id = envelope.request_id
    if not result.message:
        result.message = go_trim_space(body.decode("utf-8", "replace"))
        if not result.message:
            result.message = status_text(status_code)
    return result


@dataclass(slots=True)
class StreamingState:
    """Accumulates one attempt's stream until message_stop or an error event."""

    message: MessageBuilder | None = None
    body: bytes = b""
    terminal: bool = False
    api_error: APIError | None = None

    def observe(self, data: bytes) -> None:
        """Apply one event payload. Raises ValueError for malformed or out-of-order events."""
        if self.terminal:
            return
        if self.api_error is not None:
            raise self.api_error
        try:
            event = decode_object(data, StreamEvent)
        except JSONError as error:
            raise ValueError(f"decode Messages stream event: {error}") from error
        kind = event.type or ""
        match kind:
            case "message_start":
                if self.message is not None:
                    raise ValueError("duplicate message_start")
                self.message = MessageBuilder.start(bytes(event.message))
            case "message_delta" | "message_stop":
                if self.message is None:
                    raise ValueError(f"{kind} before message_start")
                if kind == "message_delta":
                    self.message.update(bytes(event.delta), bytes(event.usage))
                    return
                try:
                    body = self.message.build()
                except JSONError as error:
                    raise ValueError(str(error)) from error
                self.body, self.terminal = body, True
                self.message = None
            case "content_block_start" | "content_block_delta" | "content_block_stop":
                if self.message is None:
                    raise ValueError(f"{kind} before message_start")
                index = event.index
                if index is None or index < 0:
                    raise ValueError(f"{kind} requires a nonnegative index")
                if kind == "content_block_start":
                    self.message.start_block(index, bytes(event.content_block))
                    return
                block = self.message.blocks.get(index)
                if block is None:
                    raise ValueError(f"{kind} for nonexistent block {index}")
                if kind == "content_block_delta":
                    try:
                        block.handle_delta(bytes(event.delta))
                    except (ValueError, JSONError) as error:
                        raise ValueError(f"content block {index}: {error}") from error
                else:
                    block.handle_stop()
            case "error":
                self.body = bytes(data)
                self.api_error = provider_error(200, data, None)
            case "":
                raise ValueError("messages stream event has no type")

    def unwrap(self) -> bytes:
        if self.api_error is not None:
            raise self.api_error
        if not self.terminal:
            raise StreamEndedError(
                "messages stream ended without message_stop: unexpected EOF"
            ) from UnexpectedEOFError()
        return self.body
