from dataclasses import dataclass, field
from typing import Final

import msgspec

from unreal_agent._go import go_trim_space
from unreal_agent.gojson import Int64, JSONError, Raw, encode
from unreal_agent.llm.responsesapi._wire import (
    decode_object,
    reformat,
    status_text,
    trim_space,
)
from unreal_agent.llm.responsesapi.errors import APIError, StreamEndedError
from unreal_agent.primitives._errors import UnexpectedEOFError

__all__ = ["ResponseState", "provider_error"]

_TERMINAL_TYPES: Final = frozenset({"response.completed", "response.failed", "response.incomplete"})
_EMPTY: Final = Raw(b"")


class _EventError(msgspec.Struct):
    code: str | None = None
    message: str | None = None
    param: str | None = None
    type: str | None = None


class _Event(msgspec.Struct):
    type: str | None = None
    response: Raw = _EMPTY
    item: Raw = _EMPTY
    output_index: Int64 | None = None
    code: str | None = None
    message: str | None = None
    param: str | None = None
    error: _EventError | None = None


class _ErrorBody(msgspec.Struct):
    code: str | None = None
    message: str | None = None
    param: str | None = None
    type: str | None = None


class _ErrorEnvelope(msgspec.Struct):
    error: _ErrorBody | None = None


def provider_error(status_code: int, body: bytes) -> APIError:
    try:
        envelope = decode_object(body, _ErrorEnvelope)
    except JSONError:
        envelope = None
    if envelope is not None:
        error = envelope.error or _ErrorBody()
        if error.message or error.code:
            return APIError(
                status_code=status_code,
                code=error.code or "",
                message=error.message or "",
                param=error.param or "",
                type=error.type or "",
            )
    message = go_trim_space(body.decode("utf-8", "replace"))
    if not message:
        message = status_text(status_code)
    return APIError(status_code=status_code, message=message)


@dataclass(slots=True)
class ResponseState:
    """Accumulates one attempt's stream until a terminal event or an error event."""

    terminal: bool = False
    response: bytes | None = None
    items: dict[int, bytes] = field(default_factory=dict)
    failure: APIError | None = None
    err: APIError | None = None

    def observe(self, data: bytes) -> None:
        """Apply one event payload. Raises ValueError for malformed event JSON."""
        if self.terminal or self.err is not None:
            if self.err is not None:
                raise self.err
            return
        try:
            event = decode_object(data, _Event)
        except JSONError as error:
            raise ValueError(f"invalid Responses stream event JSON: {error}") from error
        kind = event.type or ""
        if kind in _TERMINAL_TYPES:
            self.terminal = True
            self.response = bytes(event.response) if len(event.response) else None
            if kind == "response.failed":
                self.failure = provider_error(200, self.response or b"")
        if kind == "response.output_item.done":
            item = trim_space(bytes(event.item))
            if event.output_index is None or event.output_index < 0 or item[:1] != b"{":
                return
            self.items[event.output_index] = bytes(event.item)
        if kind == "error":
            # The provider nests code/message under an "error" object as often as it
            # puts them at the top level (e.g. an in-band credit_balance_exhausted
            # failure). Read both so the surfaced error is diagnosable and classifiable
            # instead of an opaque empty "status 200:".
            code, message, param = event.code or "", event.message or "", event.param or ""
            error_type = ""
            if event.error is not None:
                code = code or event.error.code or ""
                message = message or event.error.message or ""
                param = param or event.error.param or ""
                error_type = event.error.type or ""
            self.err = APIError(
                status_code=200,
                code=code,
                message=message,
                param=param,
                type=error_type or "error",
            )

    def unwrap(self) -> bytes:
        """Return the terminal response, with collected items spliced into an empty output."""
        if self.err is not None:
            raise self.err
        body = trim_space(self.response or b"")
        if body[:1] != b"{":
            raise StreamEndedError(
                "responses stream ended without a terminal response: unexpected EOF"
            ) from UnexpectedEOFError()
        if not self.items:
            return body
        # The ChatGPT subscription endpoint omits `output` from the terminal response,
        # so the items collected from the stream are spliced in.
        fields = decode_object(body, dict[str, Raw])
        output: list[Raw] = []
        raw = fields.get("output")
        if raw is not None and len(raw):
            output = decode_object(bytes(raw), list[Raw])
        if output:
            return body
        spliced = [Raw(reformat(self.items[index])) for index in sorted(self.items)]
        fields["output"] = Raw(encode(spliced))
        return encode(
            {name: Raw(reformat(bytes(value))) for name, value in fields.items()},
            deterministic=True,
        )
