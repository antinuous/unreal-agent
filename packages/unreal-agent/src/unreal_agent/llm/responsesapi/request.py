"""Responses API request encoding.

Bodies match the Go encoder byte for byte: object members are sorted at every
level (the generated wire types declare properties alphabetically), Go's
`omitempty` drops empty strings, arrays, and objects, and raw provider JSON is
replayed in its own member order.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from unreal_agent.gojson import JSONError, Raw, encode, validate
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
    ToolType,
)
from unreal_agent.llm.responsesapi._wire import coerce_utf8, reformat_checked
from unreal_agent.primitives._errors import go_quote

__all__ = ["request_body"]

_INCLUDE: Final = ["reasoning.encrypted_content"]


def request_body(
    request: Request, prompt_cache_key: str, extensions: Mapping[str, bytes] | None = None
) -> bytes:
    """Encode the streaming request body. Raises ValueError for unsupported input."""
    input_items = _request_input(request.Input)
    tools = _request_tools(request.Tools)

    try:
        # Go encodes the model and plain message content with encoding/json v1, which
        # replaces invalid UTF-8 instead of rejecting it.
        model = Raw(encode(coerce_utf8(request.Model.ID)))
    except JSONError as error:
        raise ValueError(f"encode model: {error}") from error
    params: dict[str, Any] = {
        "include": _INCLUDE,
        "store": False,
        "stream": True,
    }
    if request.Model.ID:
        params["model"] = model
    if input_items:
        params["input"] = input_items
    if prompt_cache_key:
        params["prompt_cache_key"] = prompt_cache_key
    if request.Model.MaxOutputTokens is not None:
        params["max_output_tokens"] = request.Model.MaxOutputTokens
    effort = request.Model.ReasoningEffort
    if effort:
        if not ReasoningEffort.is_valid(effort):
            raise ValueError(f"unsupported reasoning effort {go_quote(effort)}")
        params["reasoning"] = {"effort": str(effort), "summary": "auto"}
    if tools:
        params["tools"] = tools
    if extensions:
        for name, value in extensions.items():
            if name in params:
                raise ValueError(
                    f"request extension {go_quote(name)} overrides a Responses API field"
                )
            try:
                params[name] = Raw(reformat_checked(bytes(value)))
            except JSONError as error:
                raise ValueError(f"request extension {go_quote(name)} is not valid JSON") from error
    try:
        return encode(params, deterministic=True)
    except JSONError as error:
        raise ValueError(f"encode response request: {error}") from error


def _request_input(items: Sequence[Item]) -> list[Raw]:
    converted: list[Raw] = []
    for index, item in enumerate(items):
        try:
            converted.append(request_input_item(item))
        except (ValueError, JSONError) as error:
            raise ValueError(f"input item {index}: {error}") from error
    return converted


def request_input_item(source: Item) -> Raw:
    """Encode one input item as Go's `openaiapi.InputItem` union value."""
    if source.Type == ItemType.MESSAGE and not source.ProviderID:
        message = _data(source, Message, "message")
        converted: dict[str, Any] = {
            "content": coerce_utf8(message.Text),
            "role": str(message.Role),
        }
        if message.Phase:
            converted["phase"] = message.Phase
        return _union(converted)
    return _request_item(source)


def _request_item(source: Item) -> Raw:
    match source.Type:
        case ItemType.MESSAGE:
            message = _data(source, Message, "message")
            if message.Role != Role.ASSISTANT:
                return _union(
                    {
                        "content": [{"text": message.Text, "type": "input_text"}],
                        "role": str(message.Role),
                        "type": "message",
                    }
                )
            content = []
            if message.Text:
                content.append(
                    {"annotations": [], "logprobs": [], "text": message.Text, "type": "output_text"}
                )
            output: dict[str, Any] = {
                "content": content,
                "id": source.ProviderID,
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
            if message.Phase:
                output["phase"] = message.Phase
            return _union(output)
        case ItemType.TOOL_CALL:
            call = _data(source, ToolCall, "tool_call")
            try:
                arguments = _request_tool_call_arguments(call.Arguments)
            except JSONError as error:
                raise ValueError(
                    f"encode tool call {go_quote(call.CallID)} arguments: {error}"
                ) from error
            function_call: dict[str, Any] = {
                "arguments": arguments,
                "call_id": call.CallID,
                "name": call.Name,
                "type": "function_call",
            }
            if source.ProviderID:
                function_call["id"] = source.ProviderID
            return _union(function_call)
        case ItemType.TOOL_RESULT:
            result = _data(source, ToolResult, "tool_result")
            contents: list[dict[str, Any]] = []
            for part in result.Output:
                match part.Kind:
                    case ToolResultKind.TEXT:
                        contents.append({"text": part.Value, "type": "input_text"})
                    case ToolResultKind.IMAGE:
                        image: dict[str, Any] = {"type": "input_image"}
                        if part.Value:
                            image["image_url"] = part.Value
                        contents.append(image)
                    case _:
                        raise ValueError(f"unsupported tool result kind {go_quote(part.Kind)}")
            output_item: dict[str, Any] = {
                "output": [_union(content) for content in contents],
                "type": "function_call_output",
            }
            if result.CallID:
                output_item["call_id"] = result.CallID
            return _union(output_item)
        case ItemType.REASONING:
            reasoning = _data(source, Reasoning, "reasoning")
            if len(reasoning.Raw) == 0:
                raise ValueError("reasoning item must carry the provider item in Raw")
            return Raw(reformat_checked(bytes(reasoning.Raw)))
        case _:
            raise ValueError(f"unsupported input item type {go_quote(source.Type)}")


def _request_tool_call_arguments(arguments: str) -> str:
    try:
        if isinstance(validate(arguments), dict):
            return arguments
    except JSONError:
        pass
    # Providers may reject their own malformed calls in history. Keep the raw
    # arguments visible alongside the tool error in a replayable JSON object.
    return encode({"invalid_arguments": arguments}).decode()


def _request_tools(source: Sequence[Tool]) -> list[Raw]:
    tools: list[Raw] = []
    for index, tool in enumerate(source):
        try:
            tools.append(_request_tool(tool))
        except (ValueError, JSONError) as error:
            raise ValueError(f"tool {index}: {error}") from error
    return tools


def _request_tool(source: Tool) -> Raw:
    match source.Type:
        case ToolType.FUNCTION:
            # The wire requires the field. The harness validates tool calls itself, and provider
            # schema enforcement would reject the loose schemas that external tools contribute.
            function: dict[str, Any] = {
                "name": source.Name,
                "parameters": source.Parameters or {},
                "strict": False,
                "type": "function",
            }
            if source.Description:
                function["description"] = source.Description
            return _union(function)
        case ToolType.HOSTED:
            if source.Name == "web_search":
                return _union({"type": "web_search"})
            raise ValueError(f"unsupported hosted tool name {go_quote(source.Name)}")
        case _:
            raise ValueError(f"unsupported tool type {go_quote(source.Type)}")


def _union(value: Mapping[str, Any]) -> Raw:
    return Raw(encode(value, deterministic=True))


_GO_NAMES: Final[dict[type, str]] = {
    Message: "llm.Message",
    ToolCall: "llm.ToolCall",
    ToolResult: "llm.ToolResult",
    Reasoning: "llm.Reasoning",
}


def _data[T](source: Item, kind: type[T], label: str) -> T:
    data = source.Data
    if type(data) is not kind:
        raise ValueError(f"{label} item data must be {_GO_NAMES[kind]}, got {_go_type(data)}")
    return cast(T, data)


def _go_type(value: object) -> str:
    if value is None:
        return "<nil>"
    name = _GO_NAMES.get(type(value))
    return name if name is not None else type(value).__qualname__
