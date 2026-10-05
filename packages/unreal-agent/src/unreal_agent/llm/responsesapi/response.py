"""Responses API response normalization.

The wire structs mirror the members of the generated `openaiapi.Response` so
type mismatches fail decoding as they do in Go. Output items are decoded by
their `type` tag. Field names match case-sensitively; Go's legacy decoder used
for output items also accepts other cases.
"""

from typing import Final

import msgspec

from unreal_agent._go import go_quote
from unreal_agent.gojson import Int64, JSONError, Raw
from unreal_agent.llm.model import (
    Failure,
    Item,
    ItemType,
    Message,
    Reasoning,
    Response,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from unreal_agent.llm.responsesapi._wire import decode_object

__all__ = ["decode_response"]

_EMPTY: Final = Raw(b"")


class _Wire(msgspec.Struct):
    pass


class _ResponseError(_Wire):
    code: str | None = None
    message: str | None = None


class _IncompleteDetails(_Wire):
    reason: str | None = None


class _Conversation(_Wire):
    id: str | None = None


class _Moderation(_Wire):
    input: Raw = _EMPTY
    output: Raw = _EMPTY


class _Prompt(_Wire):
    id: str | None = None
    variables: Raw = _EMPTY
    version: str | None = None


class _PromptCacheOptions(_Wire):
    mode: str | None = None
    ttl: str | None = None


class _Reasoning(_Wire):
    context: str | None = None
    effort: str | None = None
    generate_summary: str | None = None
    mode: Raw = _EMPTY
    summary: str | None = None


class _Text(_Wire):
    format: Raw = _EMPTY
    verbosity: str | None = None


class _Response(_Wire):
    background: bool | None = None
    completed_at: float | None = None
    conversation: _Conversation | None = None
    created_at: float | None = None
    error: _ResponseError | None = None
    id: str | None = None
    incomplete_details: _IncompleteDetails | None = None
    instructions: Raw = _EMPTY
    max_output_tokens: Int64 | None = None
    max_tool_calls: Int64 | None = None
    metadata: dict[str, str | None] | None = None
    model: Raw = _EMPTY
    moderation: _Moderation | None = None
    object: str | None = None
    output: list[Raw] | None = None
    output_text: str | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt: _Prompt | None = None
    prompt_cache_key: str | None = None
    prompt_cache_options: _PromptCacheOptions | None = None
    prompt_cache_retention: str | None = None
    reasoning: _Reasoning | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    status: str | None = None
    temperature: float | None = None
    text: _Text | None = None
    tool_choice: Raw = _EMPTY
    tools: list[Raw] | None = None
    top_logprobs: Int64 | None = None
    top_p: float | None = None
    truncation: str | None = None
    usage: Raw = _EMPTY
    user: str | None = None


class _InputTokensDetails(_Wire):
    cache_write_tokens: Int64 | None = None
    cached_tokens: Int64 | None = None


class _OutputTokensDetails(_Wire):
    reasoning_tokens: Int64 | None = None


class _ResponseUsage(_Wire):
    input_tokens: Int64 | None = None
    input_tokens_details: _InputTokensDetails | None = None
    output_tokens: Int64 | None = None
    output_tokens_details: _OutputTokensDetails | None = None
    total_tokens: Int64 | None = None


class _Discriminator(_Wire):
    type: str | None = None


class _OutputMessage(_Wire):
    content: list[Raw] | None = None
    id: str | None = None
    phase: str | None = None
    role: str | None = None
    status: str | None = None
    type: str | None = None


class _TopLogProb(_Wire):
    bytes: list[Int64] | None = None
    logprob: float | None = None
    token: str | None = None


class _LogProb(_Wire):
    bytes: list[Int64] | None = None
    logprob: float | None = None
    token: str | None = None
    top_logprobs: list[_TopLogProb | None] | None = None


class _OutputTextContent(_Wire):
    annotations: list[Raw] | None = None
    logprobs: list[_LogProb | None] | None = None
    text: str | None = None
    type: str | None = None


class _RefusalContent(_Wire):
    refusal: str | None = None
    type: str | None = None


class _FunctionToolCall(_Wire):
    arguments: str | None = None
    call_id: str | None = None
    caller: Raw = _EMPTY
    id: str | None = None
    name: str | None = None
    namespace: str | None = None
    status: str | None = None
    type: str | None = None


class _ReasoningTextContent(_Wire):
    text: str | None = None
    type: str | None = None


class _SummaryTextContent(_Wire):
    text: str | None = None
    type: str | None = None


class _ReasoningItem(_Wire):
    content: list[_ReasoningTextContent | None] | None = None
    encrypted_content: str | None = None
    id: str | None = None
    status: str | None = None
    summary: list[_SummaryTextContent | None] | None = None
    type: str | None = None


def decode_response(body: bytes) -> Response:
    """Normalize a terminal Responses API response. Raises ValueError."""
    try:
        source = decode_object(body, _Response)
        usage = decode_object(bytes(source.usage), _ResponseUsage) if source.usage else None
    except JSONError as error:
        raise ValueError(f"decode response: {error}") from error
    converted = _response(source, usage)
    if source.usage:
        converted.Usage.Raw = Raw(bytes(source.usage))
    return converted


def _response(source: _Response, usage: _ResponseUsage | None) -> Response:
    if source.status is None:
        raise ValueError(f"unsupported response status {go_quote('')}")
    items: list[Item] = []
    converted = Response(ID=source.id or "", Output=items, Usage=_usage(usage))
    match source.status:
        case "completed":
            converted.Stop = StopReason.COMPLETE
        case "incomplete":
            converted.Stop = _incomplete_stop_reason(source)
        case "failed":
            converted.Failure = _failure(source)
        case status:
            raise ValueError(f"unsupported response status {go_quote(status)}")
    for index, output in enumerate(source.output or []):
        try:
            item = _output_item(output)
        except (ValueError, JSONError) as error:
            raise ValueError(f"output item {index}: {error}") from error
        if item is not None:
            items.append(item)
    return converted


def _as[T](raw: Raw, kind: type[T]) -> T:
    return decode_object(bytes(raw), kind)


def _output_item(source: Raw) -> Item | None:
    try:
        item_type = _as(source, _Discriminator).type or ""
    except JSONError as error:
        raise ValueError(f"decode output item type: {error}") from error

    match item_type:
        case "web_search_call":
            return None
        case "message":
            message = _as(source, _OutputMessage)
            text: list[str] = []
            for content in message.content or []:
                try:
                    content_type = _as(content, _Discriminator).type or ""
                except JSONError as error:
                    raise ValueError(f"decode message content type: {error}") from error
                match content_type:
                    case "output_text":
                        text.append(_as(content, _OutputTextContent).text or "")
                    case "refusal":
                        text.append(_as(content, _RefusalContent).refusal or "")
                    case _:
                        raise ValueError(
                            f"unsupported message content type {go_quote(content_type)}"
                        )
            return Item(
                ProviderID=message.id or "",
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="".join(text), Phase=message.phase or ""),
            )
        case "function_call":
            call = _as(source, _FunctionToolCall)
            if call.status in ("in_progress", "incomplete"):
                return None
            return Item(
                ProviderID=call.id or "",
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(
                    CallID=call.call_id or "", Name=call.name or "", Arguments=call.arguments or ""
                ),
            )
        case "reasoning":
            reasoning = _as(source, _ReasoningItem)
            summary = [
                part.text or "" if part is not None else "" for part in reasoning.summary or []
            ]
            return Item(
                ProviderID=reasoning.id or "",
                Type=ItemType.REASONING,
                Data=Reasoning(Summary=summary, Raw=Raw(bytes(source))),
            )
        case _:
            raise ValueError(f"unsupported output item type {go_quote(item_type)}")


def _incomplete_stop_reason(source: _Response) -> StopReason:
    details = source.incomplete_details
    if details is None or details.reason is None:
        raise ValueError(f"unsupported incomplete reason {go_quote('')}")
    match details.reason:
        case "max_output_tokens":
            return StopReason.MAX_OUTPUT_TOKENS
        case "content_filter":
            return StopReason.REFUSED
        case reason:
            raise ValueError(f"unsupported incomplete reason {go_quote(reason)}")


def _failure(source: _Response) -> Failure:
    if source.error is None:
        return Failure(Message="response failed")
    return Failure(Code=source.error.code or "", Message=source.error.message or "")


def _usage(source: _ResponseUsage | None) -> Usage:
    if source is None:
        return Usage()
    input_details = source.input_tokens_details or _InputTokensDetails()
    output_details = source.output_tokens_details or _OutputTokensDetails()
    return Usage(
        InputTokens=source.input_tokens or 0,
        CachedInputTokens=input_details.cached_tokens or 0,
        CacheWriteInputTokens=input_details.cache_write_tokens or 0,
        OutputTokens=source.output_tokens or 0,
        ReasoningTokens=output_details.reasoning_tokens or 0,
    )
