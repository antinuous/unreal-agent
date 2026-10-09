"""Hand-written Messages API wire structs and the generic JSON model the builders use.

Generic JSON mirrors Go's `apijson`: numbers decode to their verbatim text (as
`Raw`) so re-encoding preserves precision, and encoding sorts object members.
"""

import json
from typing import Any, Final

import msgspec

from unreal_agent._go import go_trim_space, is_go_space
from unreal_agent.gojson import Int64, JSONError, Raw, validate

__all__ = ["parse_any", "parse_media_type"]

_EMPTY: Final = Raw(b"")


def _number(text: str) -> Raw:
    return Raw(text.encode())


def parse_any(data: bytes | str) -> Any:
    """Decode JSON with Go v2 strictness, keeping numbers as raw text."""
    validate(data)
    try:
        return json.loads(data, parse_int=_number, parse_float=_number)
    except json.JSONDecodeError as error:
        raise JSONError(str(error)) from error


_TSPECIALS: Final = frozenset('()<>@,;:\\"/[]?=')


def _lstrip(value: str) -> str:
    index = 0
    while index < len(value) and is_go_space(ord(value[index])):
        index += 1
    return value[index:]


def _consume_token(value: str) -> tuple[str, str]:
    end = 0
    while end < len(value) and " " < value[end] < "\x7f" and value[end] not in _TSPECIALS:
        end += 1
    return value[:end], value[end:]


def _consume_value(value: str) -> tuple[str, str]:
    if not value.startswith('"'):
        return _consume_token(value)
    out: list[str] = []
    index = 1
    while index < len(value):
        character = value[index]
        if character == '"':
            return "".join(out), value[index + 1 :]
        if character == "\\" and index + 1 < len(value) and value[index + 1] in _TSPECIALS:
            out.append(value[index + 1])
            index += 2
            continue
        if character in "\r\n":
            return "", value
        out.append(character)
        index += 1
    return "", value


def _consume_param(value: str) -> tuple[str, str, str]:
    rest = _lstrip(value)
    if not rest.startswith(";"):
        return "", "", value
    name, rest = _consume_token(_lstrip(rest[1:]))
    name = name.lower()
    if not name:
        return "", "", value
    rest = _lstrip(rest)
    if not rest.startswith("="):
        return "", "", value
    rest = _lstrip(rest[1:])
    parsed, after = _consume_value(rest)
    if not parsed and after == rest:
        return "", "", value
    return name, parsed, after


def parse_media_type(value: str) -> str:
    """Go's `mime.ParseMediaType`, returning only the media type. Raises ValueError."""
    base = value.partition(";")[0]
    media_type = go_trim_space(base.lower())
    kind, rest = _consume_token(media_type)
    if not kind:
        raise ValueError("mime: no media type")
    if rest:
        if not rest.startswith("/"):
            raise ValueError("mime: expected slash after first token")
        subtype, rest = _consume_token(rest[1:])
        if not subtype:
            raise ValueError("mime: expected token after slash")
        if rest:
            raise ValueError("mime: unexpected content after media subtype")
    params: dict[str, str] = {}
    remaining = value[len(base) :]
    while remaining:
        remaining = _lstrip(remaining)
        if not remaining:
            break
        name, parsed, after = _consume_param(remaining)
        if not name:
            if go_trim_space(after) == ";":
                break
            raise ValueError("mime: invalid media parameter")
        if params.get(name, parsed) != parsed:
            raise ValueError("mime: duplicate parameter name")
        params[name] = parsed
        remaining = after
    return media_type


class Wire(msgspec.Struct):
    pass


class Discriminator(Wire):
    type: str | None = None


class CacheCreation(Wire):
    ephemeral_1h_input_tokens: Int64 | None = None
    ephemeral_5m_input_tokens: Int64 | None = None


class ServerToolUsage(Wire):
    web_fetch_requests: Int64 | None = None
    web_search_requests: Int64 | None = None


class OutputTokensDetails(Wire):
    thinking_tokens: Int64 | None = None


class Usage(Wire):
    cache_creation: CacheCreation | None = None
    cache_creation_input_tokens: Int64 | None = None
    cache_read_input_tokens: Int64 | None = None
    inference_geo: str | None = None
    input_tokens: Int64 | None = None
    output_tokens: Int64 | None = None
    output_tokens_details: OutputTokensDetails | None = None
    server_tool_use: ServerToolUsage | None = None
    service_tier: str | None = None


class Container(Wire):
    expires_at: str | None = None
    id: str | None = None
    skills: list[Raw] | None = None


class RefusalStopDetails(Wire):
    category: str | None = None
    explanation: str | None = None
    type: str | None = None


class Message(Wire):
    container: Container | None = None
    content: list[Raw] | None = None
    id: str | None = None
    model: str | None = None
    role: str | None = None
    stop_details: RefusalStopDetails | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None
    type: str | None = None
    usage: Raw = _EMPTY


class ResponseTextBlock(Wire):
    citations: list[Raw] | None = None
    text: str | None = None
    type: str | None = None


class ResponseToolUseBlock(Wire):
    caller: Raw = _EMPTY
    id: str | None = None
    input: Raw = _EMPTY
    name: str | None = None
    toolset_name: str | None = None
    type: str | None = None


class ResponseThinkingBlock(Wire):
    signature: str | None = None
    thinking: str | None = None
    type: str | None = None


class ResponseRedactedThinkingBlock(Wire):
    data: str | None = None
    type: str | None = None


class StreamEvent(Wire):
    type: str | None = None
    index: Int64 | None = None
    message: Raw = _EMPTY
    content_block: Raw = _EMPTY
    delta: Raw = _EMPTY
    usage: Raw = _EMPTY


class BlockDelta(Wire):
    type: str | None = None
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    partial_json: str | None = None


class ErrorBody(Wire):
    type: str | None = None
    message: str | None = None


class ErrorEnvelope(Wire):
    error: ErrorBody | None = None
    request_id: str | None = None
