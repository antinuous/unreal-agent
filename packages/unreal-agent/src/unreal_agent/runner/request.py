"""The runner's JSON request: strict decoding with Go's error text, and validation."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

from unreal_agent.llm import ReasoningEffort
from unreal_agent.runner.errors import RunnerError, describe
from unreal_agent.runner.streams import Reader
from unreal_agent.tool._go import go_quote, go_trim_space
from unreal_agent.tool._jsonv2 import GoJSONError, decode_raw_object

__all__ = [
    "Request",
    "RequestMessage",
    "decode_request",
    "reasoning_effort",
    "validate_request",
]

_REQUEST_TYPE: Final = "agentrunner.Request"
_MESSAGE_TYPE: Final = "agentrunner.RequestMessage"
_MAX_INT64: Final = 2**63 - 1


@dataclass(slots=True)
class RequestMessage:
    role: str = ""
    content: str = ""
    message_id: str | None = None


@dataclass(slots=True)
class Request:
    """None marks a JSON member that was absent or null, like a nil Go pointer or slice."""

    messages: list[RequestMessage] | None = None
    prompt: str | None = None
    system_prompt: str | None = None
    model: str = ""
    max_attempts: int | None = None
    session_id: str | None = None
    thinking_level: str = ""
    include_partial_messages: bool | None = None
    extra_allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default=None)

    def enabled_tools(self, *names: str) -> list[str]:
        disallowed = self.disallowed_tools or []
        return [name for name in names if name not in disallowed]


class _Number(str):
    """A JSON number's literal text."""


def decode_request(input: Reader) -> Request:
    """Read one JSON request, rejecting unknown members like Go's json.RejectUnknownMembers."""
    try:
        raw = input.read()
    except Exception as error:
        raise RunnerError(f"read input: {describe(error)}") from error
    text = go_trim_space(raw.decode("utf-8", "surrogateescape"))
    if text == "":
        raise RunnerError("empty input")
    data = text.encode("utf-8", "surrogateescape")
    try:
        return _decode(data)
    except GoJSONError as error:
        raise RunnerError(f"invalid JSON: {error}") from error


def _decode(data: bytes) -> Request:
    try:
        members = decode_raw_object(data)
    except GoJSONError as error:
        raise GoJSONError(
            str(error).replace("Go map[string]jsontext.Value", "Go " + _REQUEST_TYPE)
        ) from None
    request = Request()
    for name, encoded in (members or {}).items():
        decoder = _REQUEST_FIELDS.get(name)
        if decoder is None:
            raise _unknown_member(_REQUEST_TYPE, name, "")
        value = json.loads(encoded, parse_int=_Number, parse_float=_Number)
        setattr(request, name, decoder(value, "/" + name))
    return request


def _kind(value: Any) -> str:
    match value:
        case _Number():
            return "number"
        case str():
            return "string"
        case bool():
            return "boolean"
        case list():
            return "array"
        case dict():
            return "object"
        case _:
            return "null"


def _mismatch(value: Any, go_type: str, pointer: str) -> GoJSONError:
    message = f"json: cannot unmarshal JSON {_kind(value)} into Go {go_type}"
    if pointer:
        message += f" within {go_quote(pointer)}"
    return GoJSONError(message)


def _unknown_member(go_type: str, name: str, pointer: str) -> GoJSONError:
    message = (
        f"json: cannot unmarshal JSON string into Go {go_type}: "
        f"unknown object member name {go_quote(name)}"
    )
    if pointer:
        message += f" within {go_quote(pointer)}"
    return GoJSONError(message)


def _string(value: Any, pointer: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and not isinstance(value, _Number):
        return value
    raise _mismatch(value, "string", pointer)


def _optional_string(value: Any, pointer: str) -> str | None:
    return None if value is None else _string(value, pointer)


def _optional_int(value: Any, pointer: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, _Number):
        raise _mismatch(value, "int", pointer)
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if not digits.isascii() or not digits.isdigit() or (digits[0] == "0" and digits != "0"):
        raise GoJSONError(
            f"json: cannot unmarshal JSON number {value} into Go int "
            f"within {go_quote(pointer)}: invalid syntax"
        )
    magnitude = int(digits)
    if magnitude > (_MAX_INT64 + 1 if negative else _MAX_INT64):
        raise GoJSONError(
            f"json: cannot unmarshal JSON number {value} into Go int "
            f"within {go_quote(pointer)}: value out of range"
        )
    return -magnitude if negative else magnitude


def _optional_bool(value: Any, pointer: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise _mismatch(value, "bool", pointer)


def _strings(value: Any, pointer: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise _mismatch(value, "[]string", pointer)
    return [_string(item, f"{pointer}/{index}") for index, item in enumerate(value)]


def _messages(value: Any, pointer: str) -> list[RequestMessage] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise _mismatch(value, "[]" + _MESSAGE_TYPE, pointer)
    return [_message(item, f"{pointer}/{index}") for index, item in enumerate(value)]


def _message(value: Any, pointer: str) -> RequestMessage:
    if value is None:
        return RequestMessage()
    if not isinstance(value, dict):
        raise _mismatch(value, _MESSAGE_TYPE, pointer)
    message = RequestMessage()
    for name, member in value.items():
        member_pointer = f"{pointer}/{name}"
        match name:
            case "role":
                message.role = _string(member, member_pointer)
            case "content":
                message.content = _string(member, member_pointer)
            case "message_id":
                message.message_id = _optional_string(member, member_pointer)
            case _:
                raise _unknown_member(_MESSAGE_TYPE, name, pointer)
    return message


_REQUEST_FIELDS: Final[dict[str, Callable[[Any, str], Any]]] = {
    "messages": _messages,
    "prompt": _optional_string,
    "system_prompt": _optional_string,
    "model": _string,
    "max_attempts": _optional_int,
    "session_id": _optional_string,
    "thinking_level": _string,
    "include_partial_messages": _optional_bool,
    "extra_allowed_tools": _strings,
    "disallowed_tools": _strings,
}

_THINKING_LEVELS: Final = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
    "xhigh": ReasoningEffort.XHIGH,
    "max": ReasoningEffort.MAX,
}


def validate_request(parsed: Request) -> list[RequestMessage]:
    """Return the user messages to deliver. Raises RunnerError."""
    if parsed.session_id is not None and go_trim_space(parsed.session_id) == "":
        raise RunnerError("session_id must not be empty")
    if parsed.thinking_level != "" and parsed.thinking_level not in _THINKING_LEVELS:
        raise RunnerError("thinking_level must be one of: low, medium, high, xhigh, max")
    for name in [*(parsed.extra_allowed_tools or []), *(parsed.disallowed_tools or [])]:
        if go_trim_space(name) == "":
            raise RunnerError("tool names must not be empty")

    if parsed.messages is None:
        if parsed.prompt is None:
            raise RunnerError("messages must be set")
        return [RequestMessage(content=parsed.prompt)]
    if not parsed.messages:
        raise RunnerError("messages must not be empty")
    for index, message in enumerate(parsed.messages):
        if message.role not in ("", "user"):
            raise RunnerError(f"messages[{index}].role must be user")
        if message.message_id is None:
            continue
        message_id = go_trim_space(message.message_id)
        if message_id == "":
            raise RunnerError(f"messages[{index}].message_id must not be empty")
        if not _is_uuid(message_id):
            raise RunnerError(f"messages[{index}].message_id must be a UUID")
    return parsed.messages


def _is_uuid(text: str) -> bool:
    """Go's uuid.Parse: canonical, `{...}`, `urn:uuid:...`, or 32 hex digits."""
    data = text.encode("utf-8", "surrogateescape")
    match len(data):
        case 45:
            data = data.removeprefix(b"urn:uuid:")
        case 38:
            data = data.removeprefix(b"{").removesuffix(b"}")
        case 32:
            return _is_hex(data)
    if len(data) != 36 or any(data[index] != 0x2D for index in (8, 13, 18, 23)):
        return False
    return all(_is_hex(data[start:end]) for start, end in _UUID_GROUPS)


_UUID_GROUPS: Final = ((0, 8), (9, 13), (14, 18), (19, 23), (24, 36))
_HEX_DIGITS: Final = frozenset(b"0123456789abcdefABCDEF")


def _is_hex(data: bytes) -> bool:
    return all(byte in _HEX_DIGITS for byte in data)


def reasoning_effort(level: str) -> ReasoningEffort:
    return _THINKING_LEVELS.get(level, ReasoningEffort.HIGH)
