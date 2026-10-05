import json
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any

from msgspec import field

from unreal_agent.gojson import Int64, JSONError, OmitZero, Raw, Struct, decode

__all__ = [
    "Failure",
    "Item",
    "ItemData",
    "ItemType",
    "Message",
    "Model",
    "Reasoning",
    "ReasoningEffort",
    "Request",
    "Response",
    "Role",
    "StopReason",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolResultKind",
    "ToolResultOutput",
    "ToolType",
    "Usage",
]


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ItemType(StrEnum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"


class ToolResultKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ToolType(StrEnum):
    FUNCTION = "function"
    HOSTED = "hosted"


class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._value2member_map_


class StopReason(StrEnum):
    COMPLETE = "complete"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    REFUSED = "refused"


_Role = Role
_RawJSON = Raw
_ReasoningEffort = ReasoningEffort


class Message(Struct):
    Role: _Role = ""
    Text: str = ""
    Phase: str = ""


class ToolCall(Struct):
    CallID: str = ""
    Name: str = ""
    Arguments: str = ""


class ToolResultOutput(Struct):
    Kind: ToolResultKind = ""
    Value: str = ""


class ToolResult(Struct):
    CallID: str = ""
    Output: list[ToolResultOutput] = []


class Reasoning(Struct):
    """A reasoning item.

    Raw is the provider's verbatim reasoning item. A provider may attach state to
    it that the harness cannot reconstruct, such as encrypted reasoning content,
    so adapters replay Raw unchanged instead of re-encoding Summary.
    """

    Summary: Annotated[list[str], OmitZero] = []
    Raw: Annotated[_RawJSON, OmitZero] = _RawJSON(b"")


type ItemData = Message | ToolCall | ToolResult | Reasoning

_DATA_TYPES: dict[str, tuple[type[Struct], str, str]] = {
    ItemType.MESSAGE: (Message, "message", "llm.Message"),
    ItemType.TOOL_CALL: (ToolCall, "tool call", "llm.ToolCall"),
    ItemType.TOOL_RESULT: (ToolResult, "tool result", "llm.ToolResult"),
    ItemType.REASONING: (Reasoning, "reasoning", "llm.Reasoning"),
}

if TYPE_CHECKING:
    _ItemDataField = ItemData | None
else:
    # Decoding captures Data verbatim so __post_init__ can decode it by Type.
    _ItemDataField = Raw


class Item(Struct):
    """A model input or output item whose Data type is selected by Type.

    Decoding rejects unknown types and null data. Encoding with
    `gojson.encode` calls `validate` first, like Go's MarshalJSON.
    """

    ProviderID: str = ""
    Type: ItemType = ""
    Data: _ItemDataField = None

    def __post_init__(self) -> None:
        if isinstance(self.Data, Raw):
            self.Data = _decode_item_data(self.Type, self.Data)

    def validate(self) -> None:
        """Raise ValueError when Data does not match Type."""
        entry = _DATA_TYPES.get(self.Type)
        if entry is None:
            raise ValueError(f"unsupported item type {_quote(self.Type)}")
        kind, label, go_name = entry
        if type(self.Data) is not kind:
            raise ValueError(f"{label} data must be {go_name}, got {_type_name(self.Data)}")

    def __gojson_validate__(self) -> None:
        try:
            self.validate()
        except ValueError as error:
            raise JSONError(str(error)) from error


def _decode_item_data(kind: str, encoded: Raw) -> ItemData:
    if bytes(encoded).strip() == b"null":
        raise ValueError(f"{kind} data must not be null")
    entry = _DATA_TYPES.get(kind)
    if entry is None:
        raise ValueError(f"unsupported item type {_quote(kind)}")
    data_type, label, _ = entry
    try:
        return decode(encoded, data_type)
    except JSONError as error:
        raise ValueError(f"decode {label} data: {error}") from error


class Tool(Struct):
    Type: ToolType = ""
    Name: str = ""
    Description: str = ""
    Parameters: dict[str, Any] = {}


class Model(Struct):
    ID: str = ""
    MaxOutputTokens: Int64 | None = None
    ReasoningEffort: _ReasoningEffort = ""


_Model = Model


class Request(Struct):
    Model: _Model = field(default_factory=Model)
    Input: list[Item] = []
    Tools: list[Tool] = []


class Usage(Struct):
    """Token usage.

    InputTokens includes CachedInputTokens and CacheWriteInputTokens.
    OutputTokens includes ReasoningTokens.
    """

    InputTokens: Int64 = 0
    CachedInputTokens: Int64 = 0
    CacheWriteInputTokens: Int64 = 0
    OutputTokens: Int64 = 0
    ReasoningTokens: Int64 = 0
    Raw: Annotated[_RawJSON, OmitZero] = _RawJSON(b"")


class Failure(Struct):
    Code: str = ""
    Message: str = ""


_Usage = Usage
_Failure = Failure


class Response(Struct):
    ID: str = ""
    Stop: StopReason = ""
    Output: Annotated[list[Item], OmitZero] = []
    Usage: _Usage = field(default_factory=Usage)
    Failure: _Failure | None = None


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _type_name(value: object) -> str:
    return "<nil>" if value is None else type(value).__qualname__
