from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

from msgspec import field

from unreal_agent._go import go_type_name, json_quote
from unreal_agent.gojson import Int64, JSONError, OmitZero, Raw, Struct, decode

__all__ = [
    "Failure",
    "Item",
    "ItemData",
    "ItemType",
    "Message",
    "Model",
    "ProviderDisplay",
    "ProviderDisplayKind",
    "ProviderItem",
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
    NONE = ""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ItemType(StrEnum):
    NONE = ""
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROVIDER = "provider"


_LEGACY_ITEM_REASONING: Final = "reasoning"


class ToolResultKind(StrEnum):
    NONE = ""
    TEXT = "text"
    IMAGE = "image"


class ToolType(StrEnum):
    NONE = ""
    FUNCTION = "function"
    HOSTED = "hosted"


class ReasoningEffort(StrEnum):
    NONE = ""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value != "" and value in cls._value2member_map_


class StopReason(StrEnum):
    NONE = ""
    COMPLETE = "complete"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    REFUSED = "refused"


_Role = Role
_RawJSON = Raw
_ReasoningEffort = ReasoningEffort


class Message(Struct):
    Role: _Role = _Role.NONE
    Text: str = ""
    Phase: str = ""


class ToolCall(Struct):
    CallID: str = ""
    Name: str = ""
    Arguments: str = ""


class ToolResultOutput(Struct):
    Kind: ToolResultKind = ToolResultKind.NONE
    Value: str = ""


class ToolResult(Struct):
    CallID: str = ""
    Output: list[ToolResultOutput] = []
    Running: bool = False


class ProviderDisplayKind(StrEnum):
    NONE = ""
    REASONING = "reasoning"


class ProviderDisplay(Struct):
    Kind: ProviderDisplayKind = ProviderDisplayKind.NONE
    Text: str = ""


class ProviderItem(Struct):
    """Raw is required and authoritative for replay. Display is a presentation projection."""

    Type: str = ""
    Raw: _RawJSON = _RawJSON(b"")
    Display: Annotated[ProviderDisplay | None, OmitZero] = None


class _LegacyReasoning(Struct):
    Summary: list[str] | None = None
    Raw: _RawJSON = _RawJSON(b"")


type ItemData = Message | ToolCall | ToolResult | ProviderItem

_DATA_TYPES: dict[str, tuple[type[Struct], str, str]] = {
    ItemType.MESSAGE: (Message, "message", "llm.Message"),
    ItemType.TOOL_CALL: (ToolCall, "tool call", "llm.ToolCall"),
    ItemType.TOOL_RESULT: (ToolResult, "tool result", "llm.ToolResult"),
    ItemType.PROVIDER: (ProviderItem, "provider", "llm.ProviderItem"),
}

_GO_NAMES: dict[type, str] = {kind: go_name for kind, _, go_name in _DATA_TYPES.values()}

if TYPE_CHECKING:
    _ItemTypeField = ItemType
    _ItemDataField = ItemData | None
else:
    # Decoding captures Type and Data verbatim so __post_init__ can migrate the
    # retired "reasoning" type and decode Data by Type.
    _ItemTypeField = str
    _ItemDataField = Raw


class Item(Struct, kw_only=True):
    """A model input or output item whose Data type is selected by Type.

    Decoding rejects unknown types and null or absent data, and validates every
    item except legacy "reasoning" items, which decode as provider items that
    may lack Raw. Encoding with `gojson.encode` calls `validate` first, like Go's
    MarshalJSON.
    """

    ProviderID: str = ""
    Type: _ItemTypeField = ItemType.NONE
    # Required, unlike other fields, because Go's UnmarshalJSON rejects an item without Data.
    Data: _ItemDataField

    def __post_init__(self) -> None:
        if isinstance(self.Data, Raw):
            kind = self.Type
            self.Data = _decode_item_data(kind, self.Data)
            if kind == _LEGACY_ITEM_REASONING:
                self.Type = ItemType.PROVIDER
            else:
                self.Type = ItemType(kind)
                self.validate()

    def validate(self) -> None:
        """Raise ValueError when Data does not match Type."""
        entry = _DATA_TYPES.get(self.Type)
        if entry is None:
            raise ValueError(f"unsupported item type {json_quote(self.Type)}")
        kind, label, go_name = entry
        if type(self.Data) is not kind:
            got = go_type_name(self.Data, _GO_NAMES)
            raise ValueError(f"{label} data must be {go_name}, got {got}")
        if isinstance(self.Data, ProviderItem):
            raw = bytes(self.Data.Raw).lstrip(b" \t\r\n")
            if not raw or raw.startswith(b"n"):
                raise ValueError("provider item Raw is required")

    def __gojson_validate__(self) -> None:
        try:
            self.validate()
        except ValueError as error:
            raise JSONError(str(error)) from error


def _decode_item_data(kind: str, encoded: Raw) -> ItemData:
    if bytes(encoded).strip() == b"null":
        raise ValueError(f"{kind} data must not be null")
    if kind == _LEGACY_ITEM_REASONING:
        return _decode_legacy_reasoning(encoded)
    entry = _DATA_TYPES.get(kind)
    if entry is None:
        raise ValueError(f"unsupported item type {json_quote(kind)}")
    data_type, label, _ = entry
    try:
        return cast(ItemData, decode(encoded, data_type))
    except JSONError as error:
        raise ValueError(f"decode {label} data: {error}") from error


def _decode_legacy_reasoning(encoded: Raw) -> ProviderItem:
    try:
        value = decode(encoded, _LegacyReasoning)
    except JSONError as error:
        raise ValueError(f"decode reasoning data: {error}") from error
    provider = ProviderItem(Raw=value.Raw)
    if bytes(value.Raw).lstrip(b" \t\r\n").startswith(b"{"):
        try:
            header = decode(value.Raw, dict[str, Raw])
        except JSONError as error:
            raise ValueError(f"decode legacy reasoning payload: {error}") from error
        kind = header.get("type")
        if kind is not None and bytes(kind).lstrip(b" \t\r\n").startswith(b'"'):
            try:
                provider.Type = decode(kind, str)
            except JSONError as error:
                raise ValueError(f"decode legacy reasoning type: {error}") from error
    if value.Summary:
        provider.Display = ProviderDisplay(
            Kind=ProviderDisplayKind.REASONING, Text="\n\n".join(value.Summary)
        )
    return provider


class Tool(Struct):
    Type: ToolType = ToolType.NONE
    Name: str = ""
    Description: str = ""
    Parameters: dict[str, Any] = {}


class Model(Struct):
    ID: str = ""
    MaxOutputTokens: Int64 | None = None
    ReasoningEffort: _ReasoningEffort = _ReasoningEffort.NONE


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
    Stop: StopReason = StopReason.NONE
    Output: Annotated[list[Item] | None, OmitZero] = None
    Usage: _Usage = field(default_factory=Usage)
    Failure: _Failure | None = None
