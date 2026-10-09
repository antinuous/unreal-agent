"""Messages API response normalization.

The wire structs mirror the Messages API members so type mismatches fail
decoding. Content blocks are decoded by their `type` tag, and thinking blocks
and tool inputs keep their verbatim JSON.
"""

from typing import Final

from unreal_agent._go import go_quote
from unreal_agent.gojson import JSONError, Raw
from unreal_agent.llm.messagesapi import _wire as wire
from unreal_agent.llm.model import (
    Failure,
    Item,
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Response,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from unreal_agent.llm.responsesapi._wire import decode_object

__all__ = ["decode_response"]

_STOP_REASONS: Final = {
    "end_turn": StopReason.COMPLETE,
    "tool_use": StopReason.COMPLETE,
    "stop_sequence": StopReason.COMPLETE,
    "max_tokens": StopReason.MAX_OUTPUT_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_OUTPUT_TOKENS,
    "refusal": StopReason.REFUSED,
}


def decode_response(body: bytes) -> Response:
    """Normalize an assembled Messages API message. Raises ValueError."""
    try:
        source = decode_object(body, wire.Message)
    except JSONError as error:
        raise ValueError(f"decode messages response: {error}") from error
    usage = wire.Usage()
    if len(source.usage):
        try:
            usage = decode_object(bytes(source.usage), wire.Usage)
        except JSONError as error:
            raise ValueError(f"decode messages response usage: {error}") from error
    converted = _response(source, usage)
    if len(source.usage):
        converted.Usage.Raw = Raw(bytes(source.usage))
    return converted


def _response(source: wire.Message, usage: wire.Usage) -> Response:
    if source.type != "message" or source.role != "assistant":
        raise ValueError(
            f"expected an assistant message, got type {go_quote(source.type or '')} "
            f"and role {go_quote(source.role or '')}"
        )
    if source.stop_reason is None:
        raise ValueError("response must have a stop reason")
    output: list[Item] = []
    converted = Response(ID=source.id or "", Output=output, Usage=response_usage(usage))
    stop = _STOP_REASONS.get(source.stop_reason)
    if stop is None:
        raise ValueError(f"unsupported stop reason {go_quote(source.stop_reason)}")
    converted.Stop = stop
    if stop == StopReason.REFUSED:
        converted.Failure = _refusal(source.stop_details)
    calls: set[str] = set()
    for index, block in enumerate(source.content or []):
        try:
            item = _output_item(block)
        except (ValueError, JSONError) as error:
            raise ValueError(f"output block {index}: {error}") from error
        if item.Type == ItemType.TOOL_CALL:
            call_id = item.Data.CallID if isinstance(item.Data, ToolCall) else ""
            if call_id in calls:
                raise ValueError(f"output block {index}: duplicate tool call {go_quote(call_id)}")
            calls.add(call_id)
        output.append(item)
    return converted


def _refusal(details: wire.RefusalStopDetails | None) -> Failure:
    failure = Failure(Code="refusal", Message="response refused")
    if details is not None:
        if details.category:
            failure.Code = details.category
        if details.explanation:
            failure.Message = details.explanation
    return failure


def _as[T](raw: Raw, kind: type[T]) -> T:
    return decode_object(bytes(raw), kind)


def _output_item(source: Raw) -> Item:
    try:
        kind = _as(source, wire.Discriminator).type or ""
    except JSONError as error:
        raise ValueError(f"decode output block type: {error}") from error
    match kind:
        case "text":
            text = _as(source, wire.ResponseTextBlock)
            if text.citations:
                raise ValueError("citations are not supported")
            return Item(
                Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text=text.text or "")
            )
        case "tool_use":
            block = _as(source, wire.ResponseToolUseBlock)
            if not block.id or not block.name or bytes(block.input)[:1] != b"{":
                raise ValueError("tool call must have an ID, name, and input object")
            if block.toolset_name is not None:
                raise ValueError(f"unsupported toolset {go_quote(block.toolset_name)}")
            caller = bytes(block.caller)
            if caller and caller != b"null":
                caller_kind = _as(block.caller, wire.Discriminator).type or ""
                if caller_kind != "direct":
                    raise ValueError(f"unsupported tool caller {go_quote(caller_kind)}")
            return Item(
                ProviderID=block.id,
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(
                    CallID=block.id, Name=block.name, Arguments=bytes(block.input).decode()
                ),
            )
        case "thinking":
            thinking = _as(source, wire.ResponseThinkingBlock)
            if not thinking.signature:
                raise ValueError("thinking block must have a signature")
            provider = ProviderItem(Type=kind, Raw=Raw(bytes(source)))
            if thinking.thinking:
                provider.Display = ProviderDisplay(
                    Kind=ProviderDisplayKind.REASONING, Text=thinking.thinking
                )
            return Item(Type=ItemType.PROVIDER, Data=provider)
        case "redacted_thinking":
            if not _as(source, wire.ResponseRedactedThinkingBlock).data:
                raise ValueError("redacted thinking block must have data")
            return Item(
                Type=ItemType.PROVIDER, Data=ProviderItem(Type=kind, Raw=Raw(bytes(source)))
            )
        case _:
            raise ValueError(f"unsupported output block type {go_quote(kind)}")


def response_usage(source: wire.Usage) -> Usage:
    cached = source.cache_read_input_tokens or 0
    written = source.cache_creation_input_tokens or 0
    details = source.output_tokens_details
    return Usage(
        InputTokens=(source.input_tokens or 0) + cached + written,
        CachedInputTokens=cached,
        CacheWriteInputTokens=written,
        OutputTokens=source.output_tokens or 0,
        ReasoningTokens=(details.thinking_tokens or 0) if details is not None else 0,
    )
