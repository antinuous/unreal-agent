from collections.abc import Sequence
from dataclasses import dataclass, field

from msgspec.structs import replace

from unreal_agent.gojson import Timestamp, is_valid_json
from unreal_agent.inbox import Input, InputKind
from unreal_agent.operation import Operation, Status
from unreal_agent.session import ID as SessionID
from unreal_agent.session import Session, Turn, TurnID
from unreal_agent.sessionstore import (
    ExistError,
    Fork,
    Item,
    ItemData,
    ItemKind,
    ModelResponse,
    NotExistError,
    ResumeState,
    Snapshot,
    StoreError,
    ToolCallStatus,
)
from unreal_agent.tool import CallStatus
from unreal_agent.tool._go import go_quote

_TERMINAL_STATUSES = frozenset({Status.COMPLETED, Status.FAILED, Status.CANCELED})
_VALID_STATUSES = frozenset(
    {
        Status.READY,
        Status.AWAITING,
        Status.CANCELING,
        Status.COMPLETED,
        Status.FAILED,
        Status.CANCELED,
    }
)


@dataclass(slots=True)
class SessionHead:
    """The validated write state of one session.

    Every transition validates before it mutates, so a rejected transition
    leaves the head unchanged.
    """

    snapshot: Snapshot
    operations: list[Operation] = field(default_factory=list)
    item_sequence: int = 0
    latest_turn_id: TurnID = ""
    turns: set[TurnID] = field(default_factory=set)
    owned_turns: set[TurnID] = field(default_factory=set)
    responded_turns: set[TurnID] = field(default_factory=set)
    tool_call_statuses: set[tuple[TurnID, str]] = field(default_factory=set)
    operation_positions: dict[str, int] = field(default_factory=dict)

    @property
    def id(self) -> SessionID:
        return self.snapshot.session.ID

    def append_input(self, input: Input, recorded_at: Timestamp) -> Item:
        try:
            input.validate()
        except ValueError as error:
            raise StoreError(f"append input to session {go_quote(self.id)}: {error}") from error
        return self._append_item(ItemKind.INPUT, input, recorded_at)

    def append_turn(self, turn: Turn, recorded_at: Timestamp) -> Item:
        if turn.ID == "":
            raise StoreError(f"append turn to session {go_quote(self.id)}: turn ID is empty")
        if turn.PreviousTurnID != self.latest_turn_id:
            raise StoreError(
                f"append turn {go_quote(turn.ID)} to session {go_quote(self.id)}: "
                f"previous turn is {go_quote(turn.PreviousTurnID)}, "
                f"want {go_quote(self.latest_turn_id)}"
            )
        if turn.ID in self.turns:
            raise ExistError(f"append turn {go_quote(turn.ID)}: file already exists")
        self.turns.add(turn.ID)
        self.owned_turns.add(turn.ID)
        self.latest_turn_id = turn.ID
        return self._append_item(ItemKind.TURN, turn, recorded_at)

    def append_model_response(self, response: ModelResponse, recorded_at: Timestamp) -> Item:
        if response.TurnID not in self.owned_turns:
            raise NotExistError(
                f"append model response to session {go_quote(self.id)} "
                f"for turn {go_quote(response.TurnID)}: file does not exist"
            )
        if response.TurnID in self.responded_turns:
            raise ExistError(
                f"append model response for turn {go_quote(response.TurnID)}: file already exists"
            )
        self.responded_turns.add(response.TurnID)
        return self._append_item(ItemKind.MODEL_RESPONSE, response, recorded_at)

    def append_tool_call_status(
        self, status: ToolCallStatus, operations: Sequence[Operation], recorded_at: Timestamp
    ) -> Item:
        """Append `status`, which must not carry operations, initializing `operations` once."""
        if status.TurnID not in self.owned_turns:
            raise NotExistError(
                f"append tool-call status to session {go_quote(self.id)} "
                f"for turn {go_quote(status.TurnID)}: file does not exist"
            )
        if status.CallID == "":
            raise StoreError(
                f"append tool-call status for turn {go_quote(status.TurnID)}: call ID is empty"
            )
        key = (status.TurnID, status.CallID)
        if key in self.tool_call_statuses:
            return self._append_item(ItemKind.TOOL_CALL_STATUS, status, recorded_at)
        seen: set[str] = set()
        for index, value in enumerate(operations):
            try:
                validate_operation(value)
            except StoreError as error:
                raise StoreError(f"initialize operation {index}: {error}") from error
            if value.ID in seen or value.ID in self.operation_positions:
                raise ExistError(f"initialize operation {go_quote(value.ID)}: file already exists")
            seen.add(value.ID)
        try:
            _validate_status_operation_references(status.Status, operations)
        except StoreError as error:
            raise StoreError(
                f"append tool-call status for turn {go_quote(status.TurnID)} "
                f"and call {go_quote(status.CallID)}: {error}"
            ) from error

        self.tool_call_statuses.add(key)
        for value in operations:
            self.operation_positions[value.ID] = len(self.operations)
            self.operations.append(replace(value))
        return self._append_item(ItemKind.TOOL_CALL_STATUS, status, recorded_at)

    def save_operation(self, value: Operation) -> None:
        validate_operation(value)
        index = self.operation_positions.get(value.ID)
        if index is None:
            raise NotExistError(
                f"save operation {go_quote(value.ID)} in session {go_quote(self.id)}: "
                "file does not exist"
            )
        existing = self.operations[index]
        if existing.Type != value.Type or existing.Version != value.Version:
            raise StoreError(f"save operation {go_quote(value.ID)}: type and version cannot change")
        self.operations[index] = replace(value)

    def reset_owned_state(self) -> None:
        self.owned_turns = set()
        self.responded_turns = set()
        self.tool_call_statuses = set()
        self.operations = []
        self.operation_positions = {}

    def _append_item(self, kind: ItemKind, data: ItemData, recorded_at: Timestamp) -> Item:
        self.item_sequence += 1
        return Item(Sequence=self.item_sequence, RecordedAt=recorded_at, Kind=kind, Data=data)


def new_session_head(id: SessionID, created_at: Timestamp) -> SessionHead:
    return SessionHead(snapshot=Snapshot(session=Session(ID=id, CreatedAt=created_at)))


@dataclass(slots=True)
class StoredState:
    head: SessionHead
    items: list[Item] = field(default_factory=list)

    def append_input(self, input: Input, recorded_at: Timestamp) -> None:
        self.items.append(self.head.append_input(input, recorded_at))

    def append_turn(self, turn: Turn, recorded_at: Timestamp) -> None:
        self.items.append(self.head.append_turn(turn, recorded_at))

    def append_model_response(self, response: ModelResponse, recorded_at: Timestamp) -> None:
        self.items.append(self.head.append_model_response(response, recorded_at))

    def append_tool_call_status(
        self, status: ToolCallStatus, operations: Sequence[Operation], recorded_at: Timestamp
    ) -> None:
        item = self.head.append_tool_call_status(
            replace(status, Operations=[]), operations, recorded_at
        )
        item.Data = replace(status, Operations=list(operations))
        self.items.append(item)

    def save_operation(self, value: Operation) -> None:
        self.head.save_operation(value)

    def resume(self) -> ResumeState:
        external_input_ids: list[str] = []
        pending: set[str] = set()
        for item in self.items:
            match item.Data:
                case Input(Kind=InputKind.EXTERNAL, ID=input_id):
                    external_input_ids.append(input_id)
                case ToolCallStatus(Operations=snapshots):
                    for value in snapshots:
                        if value.Status in _TERMINAL_STATUSES:
                            pending.discard(value.ID)
                        else:
                            pending.add(value.ID)
                case _:
                    pass
        operations = [
            value
            for value in self.head.operations
            if value.Status not in _TERMINAL_STATUSES or value.ID in pending
        ]
        return ResumeState(
            snapshot=self.head.snapshot,
            operations=operations,
            external_input_ids=external_input_ids,
        )

    def inherit_item(self, item: Item) -> None:
        """Copy an item from a parent history without validating it."""
        head = self.head
        head.item_sequence = item.Sequence
        match item.Data:
            case Turn(ID=turn_id):
                head.turns.add(turn_id)
                head.owned_turns.add(turn_id)
                head.latest_turn_id = turn_id
            case ModelResponse(TurnID=turn_id):
                head.responded_turns.add(turn_id)
            case ToolCallStatus() as status:
                # TODO: Preserve status snapshots in forked history without making
                # inherited operations dispatchable.
                item = replace(item, Data=replace(status, Operations=[]))
                head.tool_call_statuses.add((status.TurnID, status.CallID))
            case Fork():
                head.reset_owned_state()
            case _:
                pass
        self.items.append(item)

    def append_fork(self, value: Fork, recorded_at: Timestamp) -> None:
        self.items.append(self.head._append_item(ItemKind.FORK, value, recorded_at))
        self.head.reset_owned_state()


def new_stored_state(id: SessionID, created_at: Timestamp) -> StoredState:
    return StoredState(head=new_session_head(id, created_at))


def fork_stored_state(
    parent: StoredState, id: SessionID, previous_turn_id: TurnID, created_at: Timestamp
) -> StoredState:
    boundary = -1
    for index, item in enumerate(parent.items):
        if boundary >= 0 and item.Kind == ItemKind.TURN:
            break
        match item.Data:
            case (
                Turn(ID=turn_id) | ModelResponse(TurnID=turn_id) | ToolCallStatus(TurnID=turn_id)
            ) if turn_id == previous_turn_id:
                boundary = index
            case _:
                pass
    if boundary < 0:
        raise NotExistError(
            f"fork session {go_quote(parent.head.id)} at turn {go_quote(previous_turn_id)}: "
            "file does not exist"
        )

    state = new_stored_state(id, created_at)
    for item in parent.items[: boundary + 1]:
        state.inherit_item(item)
    state.append_fork(Fork(ParentID=parent.head.id, PreviousTurnID=previous_turn_id), created_at)
    return state


def _validate_status_operation_references(
    status: CallStatus, operations: Sequence[Operation]
) -> None:
    if status.Error != "":
        if status.WaitingFor:
            raise StoreError("error status cannot wait for operations")
        if operations:
            raise StoreError("error status cannot initialize operations")
        return
    if not operations:
        raise StoreError("successful status must initialize at least one operation")
    if len(status.WaitingFor) != len(operations):
        raise StoreError(
            f"status waits for {len(status.WaitingFor)} operations, initialized {len(operations)}"
        )
    initialized = {value.ID for value in operations}
    waiting: set[str] = set()
    for id in status.WaitingFor:
        if id in waiting:
            raise StoreError(f"status repeats operation {go_quote(id)}")
        waiting.add(id)
        if id not in initialized:
            raise StoreError(f"status waits for operation {go_quote(id)} that was not initialized")


def validate_operation(value: Operation) -> None:
    if value.ID == "":
        raise StoreError("operation ID is empty")
    if value.Type == "":
        raise StoreError(f"operation {go_quote(value.ID)} type is empty")
    if value.Version == 0:
        raise StoreError(f"operation {go_quote(value.ID)} version is zero")
    if value.Status not in _VALID_STATUSES:
        raise StoreError(
            f"operation {go_quote(value.ID)} has unsupported status {go_quote(value.Status)}"
        )
    if len(value.State) != 0 and not is_valid_json(bytes(value.State)):
        raise StoreError(f"operation {go_quote(value.ID)} state is not valid JSON")
    if len(value.Idempotency) != 0 and not is_valid_json(bytes(value.Idempotency)):
        raise StoreError(f"operation {go_quote(value.ID)} idempotency data is not valid JSON")
