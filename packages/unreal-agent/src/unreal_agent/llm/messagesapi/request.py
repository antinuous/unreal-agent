"""Messages API request encoding.

Bodies match the Go encoder byte for byte: object members are sorted at every
level, numbers in tool arguments and schemas keep their text, and raw thinking
blocks are replayed in their own member order.
"""

import base64
import binascii
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast
from urllib.parse import urlsplit

from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.gojson import JSONError, Raw, encode
from unreal_agent.llm.messagesapi._wire import parse_any
from unreal_agent.llm.model import (
    Item,
    ItemType,
    Message,
    Reasoning,
    ReasoningEffort,
    Request,
    Role,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
)
from unreal_agent.llm.responsesapi._wire import reformat_checked

__all__ = ["DEFAULT_MAX_OUTPUT_TOKENS", "request_body"]

DEFAULT_MAX_OUTPUT_TOKENS: Final = 128_000

_IMAGE_MEDIA_TYPES: Final = ("image/png", "image/jpeg", "image/gif", "image/webp")


def request_body(request: Request) -> bytes:
    """Encode the streaming request body. Raises ValueError for unsupported input."""
    try:
        params = request_params(request)
        return encode(params, deterministic=True)
    except JSONError as error:
        raise ValueError(f"encode messages request: {error}") from error


def request_params(request: Request) -> dict[str, Any]:
    if not go_trim_space(request.Model.ID):
        raise ValueError("model must be set")
    max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    if request.Model.MaxOutputTokens is not None:
        max_tokens = request.Model.MaxOutputTokens
        if max_tokens <= 0:
            raise ValueError(f"invalid max output tokens {max_tokens}")
    params: dict[str, Any] = {
        "cache_control": {"type": "ephemeral"},
        "max_tokens": max_tokens,
        "model": request.Model.ID,
        "stream": True,
    }
    effort = request.Model.ReasoningEffort
    if effort:
        if not ReasoningEffort.is_valid(effort):
            raise ValueError(f"unsupported reasoning effort {go_quote(effort)}")
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": str(effort)}
    messages, system = request_input(request.Input)
    params["messages"] = messages
    if system is not None:
        params["system"] = system
    tools = request_tools(request.Tools)
    if tools:
        params["tools"] = tools
    return params


@dataclass(slots=True)
class _Call:
    name: str
    answered: bool = False


def request_input(
    items: Sequence[Item],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Group items into alternating messages. Raises ValueError."""
    system: list[dict[str, Any]] | None = None
    start = 0
    if items and items[0].Type == ItemType.MESSAGE:
        first = items[0].Data
        if type(first) is Message and first.Role == Role.SYSTEM:
            system = [_text(first.Text)]
            start = 1
    messages: list[dict[str, Any]] = []
    calls: dict[str, _Call] = {}
    pending: list[str] = []
    while start < len(items):
        role = _item_role_at(items, start)
        end = start + 1
        while end < len(items) and _item_role_at(items, end) == role:
            end += 1
        if role == Role.ASSISTANT:
            blocks: list[Any] = []
            for index in range(start, end):
                item = items[index]
                block = _wrap_item(index, request_input_block, item)
                if item.Type == ItemType.TOOL_CALL:
                    call = cast(ToolCall, item.Data)
                    if call.CallID in calls:
                        raise ValueError(
                            f"input item {index}: duplicate tool call {go_quote(call.CallID)}"
                        )
                    calls[call.CallID] = _Call(call.Name)
                    pending.append(call.CallID)
                blocks.append(block)
        else:
            blocks = _request_user_items(items[start:end], start, calls, pending)
            pending = []
        messages.append({"content": blocks, "role": str(role)})
        start = end
    if pending:
        raise ValueError(
            f"missing immediately following results for tool calls {_quote_list(pending)}"
        )
    if not messages:
        raise ValueError("messages must not be empty")
    return messages, system


def _quote_list(values: Sequence[str]) -> str:
    return "[" + " ".join(go_quote(value) for value in values) + "]"


def _wrap_item[T, R](index: int, convert: Callable[[T], R], value: T) -> R:
    try:
        return convert(value)
    except JSONError:
        raise
    except ValueError as error:
        raise ValueError(f"input item {index}: {error}") from error


def _item_role_at(items: Sequence[Item], index: int) -> Role:
    return _wrap_item(index, _item_role, items[index])


def _item_role(item: Item) -> Role:
    item.validate()
    match item.Type:
        case ItemType.MESSAGE:
            role = cast(Message, item.Data).Role
            match role:
                case Role.USER | Role.ASSISTANT:
                    return Role(role)
                case Role.SYSTEM:
                    raise ValueError("system message must be the first input item")
                case _:
                    raise ValueError(f"unsupported role {go_quote(role)}")
        case ItemType.TOOL_RESULT:
            return Role.USER
        case _:
            return Role.ASSISTANT


def _request_user_items(
    items: Sequence[Item], offset: int, calls: dict[str, _Call], pending: Sequence[str]
) -> list[Any]:
    results: dict[str, dict[str, Any]] = {}
    ordinary: list[Any] = []
    for index, item in enumerate(items):
        if item.Type == ItemType.MESSAGE:
            ordinary.append(_text(cast(Message, item.Data).Text))
            continue
        result = cast(ToolResult, item.Data)
        call = calls.get(result.CallID)
        if call is None:
            raise ValueError(
                f"input item {offset + index}: result for unknown tool call "
                f"{go_quote(result.CallID)}"
            )
        blocks = _wrap_item(offset + index, request_tool_output, result.Output)
        if not call.answered:
            block: dict[str, Any] = {"tool_use_id": result.CallID, "type": "tool_result"}
            if blocks:
                block["content"] = blocks
            results[result.CallID] = block
            call.answered = True
            continue
        status = "running" if result.Running else "completed"
        header = f"{go_quote(call.name)} (call {go_quote(result.CallID)}, {status})\n"
        ordinary.append(_text(header))
        ordinary.extend(blocks)
    # The first result answers the call, even when it only reports running.
    # Emit those results in call order; later updates must not rewrite history.
    ordered: list[Any] = []
    for call_id in pending:
        answer = results.get(call_id)
        if answer is None:
            raise ValueError(
                f"missing immediately following result for tool call {go_quote(call_id)}"
            )
        ordered.append(answer)
    return ordered + ordinary


def request_input_block(item: Item) -> Any:
    match item.Type:
        case ItemType.MESSAGE:
            return _text(cast(Message, item.Data).Text)
        case ItemType.TOOL_CALL:
            call = cast(ToolCall, item.Data)
            if not call.CallID or not call.Name:
                raise ValueError("tool call must have an ID and name")
            try:
                arguments = _request_tool_arguments(call.Arguments)
            except JSONError as error:
                raise ValueError(
                    f"encode tool call {go_quote(call.CallID)} arguments: {error}"
                ) from error
            return {"id": call.CallID, "input": arguments, "name": call.Name, "type": "tool_use"}
        case ItemType.REASONING:
            raw = bytes(cast(Reasoning, item.Data).Raw)
            if not raw:
                raise ValueError("reasoning must carry an Anthropic thinking block in Raw")
            return Raw(reformat_checked(raw))
        case _:
            raise ValueError(f"unsupported assistant item type {go_quote(item.Type)}")


def _request_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        value = parse_any(arguments)
    except JSONError:
        value = None
    if not isinstance(value, dict):
        # Keep malformed arguments visible alongside the translator's error,
        # while satisfying the provider's object-only history format.
        return {"invalid_arguments": arguments}
    return value


def request_tool_output(output: Sequence[ToolResultOutput]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in output:
        match part.Kind:
            case ToolResultKind.TEXT:
                blocks.append(_text(part.Value))
            case ToolResultKind.IMAGE:
                blocks.append({"source": _request_image_source(part.Value), "type": "image"})
            case _:
                raise ValueError(f"unsupported tool result kind {go_quote(part.Kind)}")
    return blocks


def _request_image_source(value: str) -> dict[str, Any]:
    if value.startswith("data:"):
        header, found, data = value.removeprefix("data:").partition(",")
        media_type = header.removesuffix(";base64")
        if not found or media_type == header or media_type not in _IMAGE_MEDIA_TYPES:
            raise ValueError("unsupported image data URL")
        try:
            decoded = base64.b64decode(
                data.replace("\r", "").replace("\n", "").encode("ascii"), validate=True
            )
        except binascii.Error, UnicodeEncodeError, ValueError:
            decoded = b""
        if not decoded:
            raise ValueError("image data URL must contain valid nonempty base64 data")
        return {"data": decoded, "media_type": media_type, "type": "base64"}
    if not _is_http_url(value):
        raise ValueError("image must be an HTTP(S) URL or a supported base64 data URL")
    return {"type": "url", "url": value}


def _is_http_url(value: str) -> bool:
    if any(character < " " or character == "\x7f" for character in value):
        return False
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    host = parts.netloc.rpartition("@")[2]
    return parts.scheme in ("http", "https") and bool(host)


def request_tools(source: Sequence[Tool]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for index, tool in enumerate(source):
        match tool.Type:
            case ToolType.FUNCTION:
                if not tool.Name:
                    raise ValueError(f"tool {index}: name must be set")
                schema = _input_schema(index, tool.Parameters)
                function: dict[str, Any] = {
                    "input_schema": schema,
                    "name": tool.Name,
                    "strict": False,
                    "type": "custom",
                }
                if tool.Description:
                    function["description"] = tool.Description
                try:
                    encode(function)
                except JSONError as error:
                    raise ValueError(f"tool {index}: {error}") from error
                tools.append(function)
            case _:
                raise ValueError(f"tool {index}: unsupported tool type {go_quote(tool.Type)}")
    return tools


def _input_schema(index: int, parameters: Any) -> dict[str, Any]:
    try:
        encoded = encode(parameters if parameters is not None else {}, deterministic=True)
    except JSONError as error:
        raise ValueError(f"tool {index}: encode schema: {error}") from error
    try:
        schema = _decode_input_schema(encoded)
    except (JSONError, ValueError) as error:
        raise ValueError(f"tool {index}: decode schema: {error}") from error
    if schema.get("type") != "object":
        raise ValueError(f"tool {index}: input schema must have type object")
    return schema


def _decode_input_schema(encoded: bytes) -> dict[str, Any]:
    value = parse_any(encoded)
    if value is None:
        return {"type": ""}
    if not isinstance(value, dict):
        raise ValueError("input schema must be a JSON object")
    schema = cast(dict[str, Any], value)
    if "properties" in schema:
        properties = schema["properties"]
        if properties is None:
            del schema["properties"]
        elif not isinstance(properties, dict):
            raise ValueError("error reading 'properties': expected a JSON object")
    if "required" in schema:
        required = schema["required"]
        if required is None:
            del schema["required"]
        elif not isinstance(required, list) or not all(
            name is None or isinstance(name, str) for name in required
        ):
            raise ValueError("error reading 'required': expected an array of strings")
        else:
            schema["required"] = ["" if name is None else name for name in required]
    kind = schema.get("type")
    if kind is None:
        schema["type"] = ""
    elif not isinstance(kind, str):
        raise ValueError("error reading 'type': expected a string")
    return schema


def _text(text: str) -> dict[str, Any]:
    return {"text": text, "type": "text"}
