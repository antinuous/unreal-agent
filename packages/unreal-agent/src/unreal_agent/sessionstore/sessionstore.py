from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Final, Protocol, cast
from uuid import UUID

from msgspec import Meta
from msgspec import field as struct_field

from unreal_agent._go import go_quote
from unreal_agent.gojson import JSONError, OmitEmpty, Raw, Struct, Timestamp, decode
from unreal_agent.inbox import ID as InputID
from unreal_agent.inbox import Input
from unreal_agent.llm import Response
from unreal_agent.operation import Operation
from unreal_agent.session import ID as SessionID
from unreal_agent.session import Session, Turn, TurnID
from unreal_agent.tool import CallStatus

__all__ = [
    "BEFORE_FIRST",
    "NIL_OBSERVER_ID",
    "ExistError",
    "Fork",
    "Item",
    "ItemData",
    "ItemKind",
    "ModelResponse",
    "NotExistError",
    "Observer",
    "ObserverID",
    "Page",
    "ResumeState",
    "Sequence",
    "SessionInfo",
    "Snapshot",
    "Store",
    "StoreError",
    "ToolCallStatus",
]

type Sequence = int

BEFORE_FIRST: Final[Sequence] = 0
"""The cursor before the first assigned sequence."""


class StoreError(Exception):
    """A session store operation failed."""


class NotExistError(StoreError, FileNotFoundError):
    """The session, turn, or operation does not exist (Go `fs.ErrNotExist`)."""


class ExistError(StoreError, FileExistsError):
    """The turn, response, or operation already exists (Go `fs.ErrExist`)."""


class ItemKind(StrEnum):
    NONE = ""
    FORK = "fork"
    INPUT = "input"
    TURN = "turn"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL_STATUS = "tool_call_status"


class Fork(Struct):
    ParentID: SessionID = ""
    PreviousTurnID: TurnID = ""


_Response = Response
_CallStatus = CallStatus


class ModelResponse(Struct):
    TurnID: str = ""
    Response: _Response = struct_field(default_factory=Response)


class ToolCallStatus(Struct):
    TurnID: str = ""
    CallID: str = ""
    Status: _CallStatus = struct_field(default_factory=CallStatus)
    Operations: Annotated[list[Operation], OmitEmpty] = []


type ItemData = Fork | Input | Turn | ModelResponse | ToolCallStatus

_DATA_TYPES: Final[dict[str, tuple[type[Struct], str, str]]] = {
    ItemKind.FORK: (Fork, "fork", "sessionstore.Fork"),
    ItemKind.INPUT: (Input, "input", "inbox.Input"),
    ItemKind.TURN: (Turn, "turn", "session.Turn"),
    ItemKind.MODEL_RESPONSE: (ModelResponse, "model response", "sessionstore.ModelResponse"),
    ItemKind.TOOL_CALL_STATUS: (ToolCallStatus, "tool-call status", "sessionstore.ToolCallStatus"),
}

_GO_TYPE_NAMES: Final[dict[type, str]] = {
    kind: go_name for kind, _, go_name in _DATA_TYPES.values()
}

if TYPE_CHECKING:
    _ItemDataField = ItemData | None
else:
    # Decoding captures Data verbatim so __post_init__ can decode it by Kind.
    _ItemDataField = Raw


class Item(Struct, kw_only=True):
    """A history item whose Data type is selected by Kind.

    Decoding rejects unknown kinds, null or absent data, and invalid inputs. Encoding
    with `gojson.encode` calls `validate_data` first, like Go's MarshalJSON.
    """

    Sequence: Annotated[int, Meta(ge=0)] = 0
    RecordedAt: Timestamp = Timestamp()
    Kind: ItemKind = ItemKind.NONE
    # Required, unlike other fields, because Go's UnmarshalJSON rejects an item without Data.
    Data: _ItemDataField

    def __post_init__(self) -> None:
        if isinstance(self.Data, Raw):
            self.Data = _decode_item_data(self.Kind, self.Data)

    def validate_data(self) -> None:
        """Raise ValueError when Data does not match Kind."""
        entry = _DATA_TYPES.get(self.Kind)
        if entry is None:
            raise ValueError(f"unsupported item kind {go_quote(self.Kind)}")
        kind, label, go_name = entry
        if type(self.Data) is not kind:
            raise ValueError(f"{label} data must be {go_name}, got {_go_type(self.Data)}")
        if isinstance(self.Data, Input):
            try:
                self.Data.validate()
            except ValueError as error:
                raise ValueError(f"invalid input data: {error}") from error

    def __gojson_validate__(self) -> None:
        try:
            self.validate_data()
        except ValueError as error:
            raise JSONError(str(error)) from error


def _decode_item_data(kind: str, encoded: Raw) -> ItemData:
    if bytes(encoded).strip() == b"null":
        raise ValueError(f"{kind} data must not be null")
    entry = _DATA_TYPES.get(kind)
    if entry is None:
        raise ValueError(f"unsupported item kind {go_quote(kind)}")
    data_type, label, _ = entry
    try:
        value = cast(ItemData, decode(encoded, data_type))
        if isinstance(value, Input):
            value.validate()
    except (JSONError, ValueError) as error:
        raise ValueError(f"decode {label} data: {error}") from error
    return value


def _go_type(value: object) -> str:
    if value is None:
        return "<nil>"
    return _GO_TYPE_NAMES.get(type(value), type(value).__qualname__)


type ObserverID = UUID

NIL_OBSERVER_ID: Final[ObserverID] = UUID(int=0)

type Observer = Callable[[SessionID, Item], None]
"""Called synchronously after an item is persisted, with the ID of the session holding it."""


@dataclass(slots=True)
class Snapshot:
    session: Session = field(default_factory=Session)


@dataclass(slots=True)
class SessionInfo:
    id: SessionID = ""
    last_updated_at: Timestamp = field(default_factory=Timestamp)


@dataclass(slots=True)
class Page:
    items: list[Item] = field(default_factory=list)
    next_after: Sequence = BEFORE_FIRST
    more: bool = False


@dataclass(slots=True)
class ResumeState:
    snapshot: Snapshot = field(default_factory=Snapshot)
    operations: list[Operation] = field(default_factory=list)
    """Unfinished operations and terminal states missing from tool-call history."""
    external_input_ids: list[InputID] = field(default_factory=list)


class Store(Protocol):
    """Append-only session history and operation state.

    Failures raise `StoreError`; a missing session, turn, or operation raises
    `NotExistError` and a duplicate raises `ExistError`. Writes for one session
    run in call order.
    """

    def add_observer(self, observer: Observer | None, /) -> ObserverID:
        """Not safe to call while an item write is notifying observers."""
        ...

    def remove_observer(self, id: ObserverID, /) -> None: ...

    async def create(self, id: SessionID, /) -> Snapshot: ...

    async def list_sessions(self) -> list[SessionInfo]: ...

    async def inspect(self, id: SessionID, /) -> Snapshot: ...

    async def items(self, id: SessionID, after: Sequence, limit: int, /) -> Page: ...

    async def append_input(self, id: SessionID, input: Input, /) -> None: ...

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None: ...

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None: ...

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        """Append the status and its operation snapshots.

        The first append for a call also initializes those operations.
        """
        ...

    async def save_operation(self, id: SessionID, operation: Operation, /) -> None:
        """Store a complete state; the latest state for its ID wins."""
        ...

    async def resume(self, id: SessionID, /) -> ResumeState: ...

    async def fork(
        self, id: SessionID, parent_id: SessionID, previous_turn_id: TurnID, /
    ) -> Snapshot: ...
