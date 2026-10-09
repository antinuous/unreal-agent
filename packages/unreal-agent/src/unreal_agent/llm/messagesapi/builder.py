"""Assembly of a streamed message from its message and content block events."""

from dataclasses import dataclass, field
from typing import Any, Final

from unreal_agent._go import go_quote
from unreal_agent.gojson import JSONError, Raw, encode, validate
from unreal_agent.llm.messagesapi._wire import BlockDelta, parse_any
from unreal_agent.llm.responsesapi._wire import decode_object

__all__ = ["BlockContentBuilder", "MessageBuilder", "merge_usage"]

# A fallback block starts the next model's attempt; earlier thinking and tool
# calls belong to a superseded model and cannot be replayed.
_SUPERSEDED_KINDS: Final = frozenset({"thinking", "redacted_thinking", "tool_use"})


def _optional_string(fields: dict[str, Any], name: str) -> str:
    value = fields.get(name)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _object(raw: bytes) -> dict[str, Any] | None:
    value = parse_any(raw)
    if value is not None and not isinstance(value, dict):
        raise JSONError("cannot unmarshal JSON value into a Go map")
    return value


@dataclass(slots=True)
class BlockContentBuilder:
    kind: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    text: list[str] = field(default_factory=list)
    signature: list[str] = field(default_factory=list)
    input: list[str] = field(default_factory=list)
    finalized: bool = False

    @classmethod
    def start(cls, raw: bytes) -> BlockContentBuilder:
        fields = _object(raw) or {}
        kind = fields.get("type")
        block = cls(kind=kind if isinstance(kind, str) else "", fields=fields)
        match block.kind:
            case "text":
                block.text.append(_optional_string(fields, "text"))
            case "thinking":
                thinking = _optional_string(fields, "thinking")
                signature = _optional_string(fields, "signature")
                block.text.append(thinking)
                block.signature.append(signature)
            case "tool_use" | "redacted_thinking" | "fallback":
                pass
            case kind:
                raise ValueError(f"unsupported output block type {go_quote(kind)}")
        return block

    def handle_delta(self, raw: bytes) -> None:
        if self.finalized:
            raise ValueError(f"content block is finalized, cannot handle delta {go_quote(raw)}")
        delta = decode_object(raw, BlockDelta)
        kind = delta.type or ""
        match (kind, self.kind):
            case ("text_delta", "text"):
                self.text.append(delta.text or "")
            case ("thinking_delta", "thinking"):
                self.text.append(delta.thinking or "")
            case ("signature_delta", "thinking"):
                self.signature.append(delta.signature or "")
            case ("input_json_delta", "tool_use"):
                self.input.append(delta.partial_json or "")
            case _:
                raise ValueError(
                    f"unsupported delta {go_quote(kind)} for {go_quote(self.kind)} block"
                )

    def handle_stop(self) -> None:
        self.finalized = True

    def build(self) -> dict[str, Any]:
        fields = dict(self.fields)
        match self.kind:
            case "text":
                fields["text"] = "".join(self.text)
            case "thinking":
                fields["thinking"] = "".join(self.text)
                fields["signature"] = "".join(self.signature)
            case "tool_use":
                streamed = "".join(self.input)
                if streamed:
                    try:
                        value = validate(streamed)
                    except JSONError:
                        value = None
                    if not isinstance(value, dict):
                        raise ValueError("streamed tool input is not a complete JSON object")
                    fields["input"] = Raw(streamed.encode())
        return fields


@dataclass(slots=True)
class MessageBuilder:
    fields: dict[str, Any]
    blocks: dict[int, BlockContentBuilder] = field(default_factory=dict)

    @classmethod
    def start(cls, raw: bytes) -> MessageBuilder:
        try:
            fields = _object(raw)
        except JSONError as error:
            raise ValueError(f"decode message_start: {error}") from error
        if fields is None:
            raise ValueError("message_start requires a message object")
        return cls(fields)

    def update(self, raw: bytes, usage: bytes) -> None:
        if raw:
            try:
                delta = _object(raw) or {}
            except JSONError as error:
                raise ValueError(f"decode message_delta: {error}") from error
            for name, value in delta.items():
                if value is not None:
                    self.fields[name] = value
        if usage and usage.strip(b" \t\r\n") != b"null":
            try:
                update = _object(usage) or {}
            except JSONError as error:
                raise ValueError(f"decode usage delta: {error}") from error
            self.fields["usage"] = merge_usage(self.fields.get("usage"), update)

    def start_block(self, index: int, raw: bytes) -> None:
        if index in self.blocks:
            raise ValueError(f"content block {index} already exists")
        try:
            block = BlockContentBuilder.start(raw)
        except (ValueError, JSONError) as error:
            raise ValueError(f"content block {index}: {error}") from error
        self.blocks[index] = block

    def build(self) -> bytes:
        content: list[dict[str, Any]] = []
        superseded = False
        for index in sorted(self.blocks, reverse=True):
            builder = self.blocks[index]
            if not builder.finalized:
                continue
            if builder.kind == "fallback":
                superseded = True
            if superseded and builder.kind in _SUPERSEDED_KINDS:
                continue
            try:
                content.append(builder.build())
            except ValueError as error:
                raise ValueError(f"content block {index}: {error}") from error
        content.reverse()
        self.fields["content"] = content
        return encode(self.fields, deterministic=True)


def merge_usage(previous: Any, update: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = previous if isinstance(previous, dict) else {}
    for name, value in update.items():
        if value is None:
            continue
        if isinstance(value, dict):
            fields[name] = merge_usage(fields.get(name), value)
        else:
            fields[name] = value
    return fields
