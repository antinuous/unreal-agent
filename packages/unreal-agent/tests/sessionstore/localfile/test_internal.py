import dataclasses
import errno
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from msgspec import field as struct_field
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent.gojson import Raw, Timestamp
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import (
    Failure,
    ItemType,
    Message,
    Reasoning,
    Response,
    Role,
    StopReason,
    ToolCall,
    Usage,
)
from unreal_agent.llm import Item as LLMItem
from unreal_agent.operation import Operation, Status
from unreal_agent.session import Session, Turn, TurnType
from unreal_agent.sessionstore import (
    ExistError,
    Fork,
    Item,
    ItemKind,
    ModelResponse,
    NotExistError,
    Snapshot,
    StoreError,
    ToolCallStatus,
)
from unreal_agent.sessionstore.localfile import Store
from unreal_agent.sessionstore.localfile.codec import (
    FORMAT_VERSION,
    ItemRecord,
    OperationRecord,
    RecordType,
    SessionRecord,
    decode_log,
    encode_initial_log,
    encode_record,
)
from unreal_agent.sessionstore.localfile.state import (
    StoredState,
    fork_stored_state,
    new_stored_state,
    validate_operation,
)
from unreal_agent.sessionstore.localfile.store import (
    append_file,
    list_session_entries,
    persist_temporary,
    publish_file,
    sync_directory,
    sync_directory_fd,
)
from unreal_agent.tool import CallStatus

TESTDATA = Path(__file__).parent / "testdata"
STATE_CREATED_AT = Timestamp.parse("2026-08-27T09:00:00Z")
STATE_UPDATED_AT = Timestamp.parse("2026-08-27T09:01:00Z")
MINUTE = 60 * 10**9


def later(minutes: int) -> Timestamp:
    return Timestamp(STATE_UPDATED_AT.unix_ns + minutes * MINUTE)


def valid_response(turn_id: str, *call_ids: str) -> ModelResponse:
    return ModelResponse(
        TurnID=turn_id,
        Response=Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID=call_id, Name="test", Arguments="{}"),
                )
                for call_id in call_ids
            ]
        ),
    )


def valid_status(turn_id: str, call_id: str, *waiting_for: str) -> ToolCallStatus:
    return ToolCallStatus(
        TurnID=turn_id, CallID=call_id, Status=CallStatus(WaitingFor=list(waiting_for))
    )


def valid_operation(id: str, status: Status) -> Operation:
    return Operation(ID=id, Type="test", Version=1, Status=status, State=Raw(b'{"step":1}'))


def clone_state(state: StoredState) -> StoredState:
    head = state.head
    return StoredState(
        head=dataclasses.replace(
            head,
            operations=list(head.operations),
            turns=set(head.turns),
            owned_turns=set(head.owned_turns),
            responded_turns=set(head.responded_turns),
            tool_call_statuses=set(head.tool_call_statuses),
            operation_positions=dict(head.operation_positions),
        ),
        items=list(state.items),
    )


def empty_test_state() -> StoredState:
    return new_stored_state("session-1", STATE_CREATED_AT)


def state_with_turn() -> StoredState:
    state = empty_test_state()
    state.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    return state


def state_with_response() -> StoredState:
    state = state_with_turn()
    state.append_model_response(valid_response("turn-1"), STATE_UPDATED_AT)
    return state


def state_with_status() -> StoredState:
    state = state_with_turn()
    state.append_tool_call_status(
        valid_status("turn-1", "call-1", "operation-1"),
        [valid_operation("operation-1", Status.READY)],
        STATE_UPDATED_AT,
    )
    return state


def assert_item_kinds(items: list[Item], *want: ItemKind) -> None:
    assert [item.Kind for item in items] == list(want), f"items = {items!r}"


def read_file(path: str) -> bytes:
    return Path(path).read_bytes()


def write_file(path: str, data: bytes) -> None:
    Path(path).write_bytes(data)


def empty_log() -> bytes:
    return encode_record(
        RecordType.SESSION,
        SessionRecord(
            Version=FORMAT_VERSION, Session=Session(ID="session-1", CreatedAt=STATE_CREATED_AT)
        ),
    )


def item_record(item: Item, *operations: Operation) -> bytes:
    return encode_record(RecordType.ITEM, ItemRecord(Item=item, Operations=list(operations)))


def turn_item(sequence: int, turn: Turn) -> Item:
    return Item(Sequence=sequence, RecordedAt=STATE_UPDATED_AT, Kind=ItemKind.TURN, Data=turn)


# State transitions.


def test_stored_state_lifecycle() -> None:
    state = new_stored_state("session-1", STATE_CREATED_AT)
    input = Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"message":"hello"}'))
    state.append_input(input, STATE_UPDATED_AT)
    state.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), later(1))
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="hello"))]
        ),
    )
    state.append_model_response(response, later(2))
    operations = [
        valid_operation("ready", Status.READY),
        valid_operation("complete", Status.COMPLETED),
    ]
    state.append_tool_call_status(
        valid_status("turn-1", "call-1", "ready", "complete"), operations, later(3)
    )
    updated = replace(operations[0], Status=Status.AWAITING, State=Raw(b'{"step":2}'))
    state.save_operation(updated)

    assert_item_kinds(
        state.items,
        ItemKind.INPUT,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.TOOL_CALL_STATUS,
    )
    assert [item.Sequence for item in state.items] == [1, 2, 3, 4]
    assert state.items[0].RecordedAt == STATE_UPDATED_AT
    resume = state.resume()
    assert resume.operations == [updated]
    assert resume.external_input_ids == ["input-1"]


def test_resume_returns_only_external_input_ids() -> None:
    state = new_stored_state("session-1", STATE_CREATED_AT)
    for input in [
        Input(ID="external-1", Kind=InputKind.EXTERNAL),
        Input(ID="control-1", Kind=InputKind.CONTROL, Payload=Raw(b'{"Mode":"hard"}')),
        Input(ID="crash-1", Kind=InputKind.CRASH),
        Input(ID="external-2", Kind=InputKind.EXTERNAL),
    ]:
        state.append_input(input, STATE_UPDATED_AT)
    assert state.resume().external_input_ids == ["external-1", "external-2"]


def test_resume_includes_inherited_external_input_ids() -> None:
    parent = new_stored_state("parent", STATE_CREATED_AT)
    parent.append_input(Input(ID="parent-input", Kind=InputKind.EXTERNAL), STATE_UPDATED_AT)
    parent.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    child = fork_stored_state(parent, "child", "turn-1", STATE_UPDATED_AT)
    child.append_input(Input(ID="child-input", Kind=InputKind.EXTERNAL), STATE_UPDATED_AT)
    assert child.resume().external_input_ids == ["parent-input", "child-input"]


def test_status_operation_references_accept_error() -> None:
    state = state_with_turn()
    state.append_tool_call_status(
        ToolCallStatus(
            TurnID="turn-1", CallID="call-1", Status=CallStatus(Error="invalid arguments")
        ),
        [],
        STATE_UPDATED_AT,
    )
    assert len(state.items) == 2
    assert state.head.operations == []


def test_status_operation_references_accept_exact_set_in_different_order() -> None:
    state = state_with_turn()
    operations = [
        valid_operation("operation-1", Status.READY),
        valid_operation("operation-2", Status.READY),
    ]
    state.append_tool_call_status(
        valid_status("turn-1", "call-1", "operation-2", "operation-1"),
        operations,
        STATE_UPDATED_AT,
    )
    assert len(state.items) == 2
    assert state.head.operations == operations


@pytest.mark.parametrize(
    ("status", "operations", "want"),
    [
        pytest.param(
            CallStatus(), [], "must initialize at least one operation", id="empty success"
        ),
        pytest.param(
            CallStatus(Error="invalid", WaitingFor=["operation-1"]),
            [],
            "error status cannot wait for operations",
            id="error waits for operation",
        ),
        pytest.param(
            CallStatus(Error="invalid"),
            [valid_operation("operation-1", Status.READY)],
            "error status cannot initialize operations",
            id="error initializes operation",
        ),
        pytest.param(
            CallStatus(WaitingFor=["operation-1"]),
            [
                valid_operation("operation-1", Status.READY),
                valid_operation("operation-2", Status.READY),
            ],
            "waits for 1 operations, initialized 2",
            id="reference count differs",
        ),
        pytest.param(
            CallStatus(WaitingFor=["operation-2"]),
            [valid_operation("operation-1", Status.READY)],
            'waits for operation "operation-2" that was not initialized',
            id="reference was not initialized",
        ),
        pytest.param(
            CallStatus(WaitingFor=["operation-1", "operation-1"]),
            [
                valid_operation("operation-1", Status.READY),
                valid_operation("operation-2", Status.READY),
            ],
            'repeats operation "operation-1"',
            id="duplicate reference",
        ),
    ],
)
def test_status_operation_references_reject(
    status: CallStatus, operations: list[Operation], want: str
) -> None:
    state = state_with_turn()
    before = clone_state(state)
    with pytest.raises(StoreError, match=want):
        state.append_tool_call_status(
            ToolCallStatus(TurnID="turn-1", CallID="call-1", Status=status),
            operations,
            STATE_UPDATED_AT,
        )
    assert state == before


def test_repeated_tool_call_status_adds_item() -> None:
    state = state_with_status()
    status = valid_status("turn-1", "call-1", "operation-1")
    operations = list(state.head.operations)
    state.append_tool_call_status(status, [], later(1))
    assert len(state.items) == 3
    assert state.items[2].Data == status
    assert state.head.operations == operations


@pytest.mark.parametrize(
    ("build", "apply", "kind"),
    [
        pytest.param(
            empty_test_state,
            lambda state: state.append_input(Input(), STATE_UPDATED_AT),
            None,
            id="empty input ID",
        ),
        pytest.param(
            empty_test_state,
            lambda state: state.append_input(
                Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b"{")), STATE_UPDATED_AT
            ),
            None,
            id="invalid input JSON",
        ),
        pytest.param(
            empty_test_state,
            lambda state: state.append_turn(Turn(), STATE_UPDATED_AT),
            None,
            id="empty turn ID",
        ),
        pytest.param(
            state_with_turn,
            lambda state: state.append_turn(
                Turn(ID="turn-2", PreviousTurnID="other", Type=TurnType.REGULAR), STATE_UPDATED_AT
            ),
            None,
            id="wrong previous turn",
        ),
        pytest.param(
            state_with_turn,
            lambda state: state.append_turn(
                Turn(ID="turn-1", PreviousTurnID="turn-1", Type=TurnType.REGULAR),
                STATE_UPDATED_AT,
            ),
            ExistError,
            id="duplicate turn",
        ),
        pytest.param(
            empty_test_state,
            lambda state: state.append_model_response(valid_response("missing"), STATE_UPDATED_AT),
            NotExistError,
            id="response for missing turn",
        ),
        pytest.param(
            state_with_response,
            lambda state: state.append_model_response(valid_response("turn-1"), STATE_UPDATED_AT),
            ExistError,
            id="duplicate response",
        ),
        pytest.param(
            empty_test_state,
            lambda state: state.append_tool_call_status(
                valid_status("missing", "call-1"), [], STATE_UPDATED_AT
            ),
            NotExistError,
            id="status for missing turn",
        ),
        pytest.param(
            state_with_turn,
            lambda state: state.append_tool_call_status(
                valid_status("turn-1", ""), [], STATE_UPDATED_AT
            ),
            None,
            id="empty call ID",
        ),
        pytest.param(
            state_with_turn,
            lambda state: state.append_tool_call_status(
                valid_status("turn-1", "call-1", "operation-1", "operation-2"),
                [
                    valid_operation("operation-1", Status.READY),
                    Operation(ID="operation-2", Status=Status.READY),
                ],
                STATE_UPDATED_AT,
            ),
            None,
            id="invalid operation after valid operation",
        ),
        pytest.param(
            state_with_turn,
            lambda state: state.append_tool_call_status(
                valid_status("turn-1", "call-1", "operation-1", "operation-1"),
                [
                    valid_operation("operation-1", Status.READY),
                    valid_operation("operation-1", Status.READY),
                ],
                STATE_UPDATED_AT,
            ),
            ExistError,
            id="duplicate operation in batch",
        ),
        pytest.param(
            state_with_status,
            lambda state: state.append_tool_call_status(
                valid_status("turn-1", "call-2", "operation-1"),
                [valid_operation("operation-1", Status.READY)],
                STATE_UPDATED_AT,
            ),
            ExistError,
            id="operation already exists",
        ),
        pytest.param(
            state_with_status,
            lambda state: state.save_operation(
                Operation(ID="operation-1", Status=Status.COMPLETED, Version=1)
            ),
            None,
            id="invalid operation update",
        ),
        pytest.param(
            state_with_status,
            lambda state: state.save_operation(valid_operation("missing", Status.COMPLETED)),
            NotExistError,
            id="missing operation update",
        ),
        pytest.param(
            state_with_status,
            lambda state: state.save_operation(
                replace(valid_operation("operation-1", Status.COMPLETED), Type="changed")
            ),
            None,
            id="operation type changes",
        ),
        pytest.param(
            state_with_status,
            lambda state: state.save_operation(
                replace(valid_operation("operation-1", Status.COMPLETED), Version=2)
            ),
            None,
            id="operation version changes",
        ),
    ],
)
def test_rejected_state_transitions_do_not_mutate_state(
    build: Callable[[], StoredState],
    apply: Callable[[StoredState], None],
    kind: type[StoreError] | None,
) -> None:
    state = build()
    before = clone_state(state)
    with pytest.raises(kind or StoreError):
        apply(state)
    assert state == before


@pytest.mark.parametrize(
    ("value", "want"),
    [
        pytest.param(Operation(), "operation ID is empty", id="empty ID"),
        pytest.param(Operation(ID="operation-1"), "type is empty", id="empty type"),
        pytest.param(
            Operation(ID="operation-1", Type="test"), "version is zero", id="zero version"
        ),
        pytest.param(
            Operation(ID="operation-1", Type="test", Version=1, Status=cast(Status, "unknown")),
            "unsupported status",
            id="unsupported status",
        ),
        pytest.param(
            Operation(
                ID="operation-1", Type="test", Version=1, Status=Status.READY, State=Raw(b"{")
            ),
            "state is not valid JSON",
            id="invalid state",
        ),
        pytest.param(
            Operation(
                ID="operation-1",
                Type="test",
                Version=1,
                Status=Status.READY,
                Idempotency=Raw(b"{"),
            ),
            "idempotency data is not valid JSON",
            id="invalid idempotency",
        ),
        pytest.param(valid_operation("operation-1", Status.READY), "", id="valid"),
    ],
)
def test_validate_operation(value: Operation, want: str) -> None:
    if not want:
        validate_operation(value)
        return
    with pytest.raises(StoreError, match=want):
        validate_operation(value)


def test_fork_stored_state_uses_last_record_for_turn() -> None:
    parent = new_stored_state("parent", STATE_CREATED_AT)
    parent.append_input(
        Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b"{}")), STATE_UPDATED_AT
    )
    parent.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    parent.append_model_response(valid_response("turn-1", "call-1"), STATE_UPDATED_AT)
    parent.append_tool_call_status(
        valid_status("turn-1", "call-1", "operation-1"),
        [valid_operation("operation-1", Status.READY)],
        STATE_UPDATED_AT,
    )
    parent.append_input(
        Input(ID="input-2", Kind=InputKind.EXTERNAL, Payload=Raw(b"{}")), STATE_UPDATED_AT
    )
    parent.append_turn(
        Turn(ID="turn-2", PreviousTurnID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT
    )
    before = clone_state(parent)

    child = fork_stored_state(parent, "child", "turn-1", STATE_UPDATED_AT)
    assert parent == before, "fork changed parent state"
    parent_status = parent.items[3].Data
    assert isinstance(parent_status, ToolCallStatus)
    assert len(parent_status.Operations) == 1
    assert child.head.snapshot.session.ID == "child"
    assert len(child.items) == 5
    assert child.items[3].Kind == ItemKind.TOOL_CALL_STATUS
    assert child.items[4].Kind == ItemKind.FORK
    child_status = child.items[3].Data
    assert isinstance(child_status, ToolCallStatus)
    assert child_status.Operations == []
    assert child.head.operations == []
    with pytest.raises(NotExistError):
        fork_stored_state(parent, "missing", "unknown", STATE_UPDATED_AT)


def test_fork_stored_state_ignores_delayed_status_beyond_newer_turn() -> None:
    parent = new_stored_state("parent", STATE_CREATED_AT)
    parent.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    parent.append_model_response(valid_response("turn-1", "call-1"), STATE_UPDATED_AT)
    parent.append_turn(
        Turn(ID="turn-2", PreviousTurnID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT
    )
    parent.append_model_response(valid_response("turn-2"), STATE_UPDATED_AT)
    parent.append_tool_call_status(
        valid_status("turn-1", "call-1", "operation-1"),
        [valid_operation("operation-1", Status.READY)],
        STATE_UPDATED_AT,
    )

    child = fork_stored_state(parent, "child", "turn-1", STATE_UPDATED_AT)
    assert_item_kinds(child.items, ItemKind.TURN, ItemKind.MODEL_RESPONSE, ItemKind.FORK)
    child = fork_stored_state(parent, "child-2", "turn-2", STATE_UPDATED_AT)
    assert_item_kinds(
        child.items,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.FORK,
    )


def test_fork_stored_state_preserves_interleaved_prefix() -> None:
    parent = new_stored_state("parent", STATE_CREATED_AT)
    parent.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    parent.append_model_response(valid_response("turn-1"), STATE_UPDATED_AT)
    parent.append_input(
        Input(ID="steering", Kind=InputKind.EXTERNAL, Payload=Raw(b"{}")), STATE_UPDATED_AT
    )
    parent.append_tool_call_status(
        valid_status("turn-1", "call-1", "operation-1"),
        [valid_operation("operation-1", Status.READY)],
        STATE_UPDATED_AT,
    )
    parent.append_turn(
        Turn(ID="turn-2", PreviousTurnID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT
    )

    child = fork_stored_state(parent, "child", "turn-1", STATE_UPDATED_AT)
    assert_item_kinds(
        child.items,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.INPUT,
        ItemKind.TOOL_CALL_STATUS,
        ItemKind.FORK,
    )
    decode_log(encode_initial_log(child.head.snapshot.session, child.items))


def test_fork_stored_state_scopes_delayed_statuses_to_owning_branch() -> None:
    root = new_stored_state("root", STATE_CREATED_AT)
    root.append_turn(Turn(ID="root-turn", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    root.append_model_response(valid_response("root-turn"), STATE_UPDATED_AT)

    child = fork_stored_state(root, "child", "root-turn", STATE_UPDATED_AT)
    before = clone_state(child)
    with pytest.raises(NotExistError):
        child.append_tool_call_status(
            valid_status("root-turn", "late-root-call", "root-operation"),
            [valid_operation("root-operation", Status.READY)],
            STATE_UPDATED_AT,
        )
    assert child == before, "rejected inherited status changed child"

    child.append_turn(
        Turn(ID="child-turn-1", PreviousTurnID="root-turn", Type=TurnType.REGULAR),
        STATE_UPDATED_AT,
    )
    child.append_model_response(valid_response("child-turn-1"), STATE_UPDATED_AT)
    child.append_turn(
        Turn(ID="child-turn-2", PreviousTurnID="child-turn-1", Type=TurnType.REGULAR),
        STATE_UPDATED_AT,
    )
    child.append_model_response(valid_response("child-turn-2"), STATE_UPDATED_AT)
    child.append_tool_call_status(
        valid_status("child-turn-1", "late-child-call", "child-operation"),
        [valid_operation("child-operation", Status.READY)],
        STATE_UPDATED_AT,
    )

    grandchild = fork_stored_state(child, "grandchild", "child-turn-1", STATE_UPDATED_AT)
    assert_item_kinds(
        grandchild.items,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.FORK,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.FORK,
    )
    decode_log(encode_initial_log(grandchild.head.snapshot.session, grandchild.items))


# Log encoding and decoding.


@pytest.mark.parametrize(
    ("state", "want"),
    [
        pytest.param(
            new_stored_state("invalid/session", STATE_CREATED_AT),
            "must contain only ASCII letters, digits, and dashes",
            id="invalid session ID",
        ),
        pytest.param(
            new_stored_state("session-1", Timestamp(253402300800 * 10**9)),
            "encode session",
            id="unencodable creation time",
        ),
    ],
)
def test_encode_initial_log_rejects_invalid_state(state: StoredState, want: str) -> None:
    with pytest.raises(StoreError, match=want):
        encode_initial_log(state.head.snapshot.session, state.items)


@pytest.mark.parametrize(
    ("item", "want"),
    [
        pytest.param(
            Item(
                Sequence=2,
                RecordedAt=STATE_UPDATED_AT,
                Kind=ItemKind.TURN,
                Data=Turn(ID="turn-1", Type=TurnType.REGULAR),
            ),
            "non-contiguous sequence",
            id="non-contiguous sequence",
        ),
        pytest.param(
            Item(Sequence=1, Kind=ItemKind.TURN, Data=Turn(ID="turn-1", Type=TurnType.REGULAR)),
            "recorded time is zero",
            id="zero recorded time",
        ),
        pytest.param(
            Item(Sequence=1, RecordedAt=STATE_UPDATED_AT, Kind=ItemKind.TURN, Data=Input()),
            "turn data must be session.Turn",
            id="wrong data type",
        ),
        pytest.param(
            Item(
                Sequence=1,
                RecordedAt=STATE_UPDATED_AT,
                Kind=ItemKind.MODEL_RESPONSE,
                Data=ModelResponse(TurnID="turn-1", Response=Response(Usage=Usage(Raw=Raw(b"{")))),
            ),
            "encode item 0",
            id="invalid model response",
        ),
    ],
)
def test_encode_initial_log_rejects_invalid_items(item: Item, want: str) -> None:
    with pytest.raises(StoreError, match=want):
        encode_initial_log(Session(ID="session-1", CreatedAt=STATE_CREATED_AT), [item])


def test_encode_initial_log_embeds_json_values() -> None:
    state = new_stored_state("session-1", STATE_CREATED_AT)
    state.append_input(
        Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"input":true}')),
        STATE_UPDATED_AT,
    )
    state.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    state.append_model_response(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(
                Output=[
                    LLMItem(Type=ItemType.REASONING, Data=Reasoning(Raw=Raw(b'{"reasoning":true}')))
                ],
                Usage=Usage(Raw=Raw(b'{"usage":true}')),
            ),
        ),
        STATE_UPDATED_AT,
    )
    encoded = encode_initial_log(state.head.snapshot.session, state.items)
    for want in (b'"Payload":{"input":true}', b'"Raw":{"reasoning":true}', b'"Raw":{"usage":true}'):
        assert want in encoded


def test_encode_record_keeps_embedded_json_on_one_line() -> None:
    encoded = encode_record(RecordType.OPERATION, Raw(b'{\n  "step": 1\n}'))
    assert encoded.count(b"\n") == 1
    assert encoded.endswith(b"\n")


def test_decode_log_folds_operation_updates() -> None:
    initial = valid_operation("operation-1", Status.READY)
    status = Item(
        Sequence=2,
        RecordedAt=STATE_UPDATED_AT,
        Kind=ItemKind.TOOL_CALL_STATUS,
        Data=valid_status("turn-1", "call-1", "operation-1"),
    )
    updated = replace(initial, Status=Status.AWAITING, State=Raw(b'{"step":2}'))
    encoded = (
        empty_log()
        + item_record(turn_item(1, Turn(ID="turn-1", Type=TurnType.REGULAR)))
        + item_record(status, initial)
        + encode_record(RecordType.OPERATION, OperationRecord(Operation=updated))
    )

    state, committed_size = decode_log(encoded)
    assert committed_size == len(encoded)
    assert state.head.operations == [updated]


def test_decode_log_ignores_incomplete_tail() -> None:
    committed = empty_log()
    state, committed_size = decode_log(committed + b'{"type":"item"')
    assert state.head.id == "session-1"
    assert committed_size == len(committed)

    with pytest.raises(StoreError):
        decode_log(committed + b"{\n")


def test_decode_log_preserves_inherited_input() -> None:
    encoded = (
        empty_log()
        + item_record(turn_item(1, Turn(ID="parent-turn", Type=TurnType.REGULAR)))
        + item_record(
            Item(
                Sequence=2,
                RecordedAt=STATE_UPDATED_AT,
                Kind=ItemKind.INPUT,
                Data=Input(ID="parent-input", Kind=InputKind.EXTERNAL, Payload=Raw(b"{}")),
            )
        )
        + item_record(
            Item(
                Sequence=3,
                RecordedAt=STATE_UPDATED_AT,
                Kind=ItemKind.FORK,
                Data=Fork(ParentID="parent", PreviousTurnID="parent-turn"),
            )
        )
    )
    state, _ = decode_log(encoded)
    input = state.items[1].Data
    assert isinstance(input, Input)
    assert input.ID == "parent-input"


def _invalid_logs() -> list[tuple[str, bytes, str]]:
    header = empty_log()
    turn = turn_item(1, Turn(ID="turn-1", Type=TurnType.REGULAR))
    status = Item(
        Sequence=2,
        RecordedAt=STATE_UPDATED_AT,
        Kind=ItemKind.TOOL_CALL_STATUS,
        Data=valid_status("turn-1", "call-1", "operation-1"),
    )
    with_turn = header + item_record(turn)
    initial = valid_operation("operation-1", Status.READY)
    with_status = with_turn + item_record(status, initial)

    def session_record(id: str, created_at: Timestamp, version: int = FORMAT_VERSION) -> bytes:
        return encode_record(
            RecordType.SESSION,
            SessionRecord(Version=version, Session=Session(ID=id, CreatedAt=created_at)),
        )

    def fork_item(sequence: int) -> Item:
        return Item(
            Sequence=sequence,
            RecordedAt=STATE_UPDATED_AT,
            Kind=ItemKind.FORK,
            Data=Fork(ParentID="parent", PreviousTurnID="turn-1"),
        )

    return [
        ("no committed record", b"{}", "no committed records"),
        ("unsupported record", encode_record("unknown", Raw(b"{}")), "unsupported record type"),
        ("item first", item_record(turn), "session record must be first"),
        (
            "operation first",
            encode_record(RecordType.OPERATION, OperationRecord(Operation=initial)),
            "session record must be first",
        ),
        (
            "session not first",
            header + session_record("other", STATE_CREATED_AT),
            "session record must be first",
        ),
        (
            "malformed session data",
            encode_record(RecordType.SESSION, Raw(b"[]")),
            "decode session record",
        ),
        (
            "legacy version",
            session_record("session-1", STATE_CREATED_AT, 1),
            "legacy session format version 1 cannot be resumed",
        ),
        (
            "unsupported version",
            session_record("session-1", STATE_CREATED_AT, FORMAT_VERSION + 1),
            "unsupported session format version",
        ),
        (
            "invalid session ID",
            session_record("invalid/id", STATE_CREATED_AT),
            "must contain only ASCII letters, digits, and dashes",
        ),
        (
            "zero creation time",
            session_record("session-1", Timestamp()),
            "creation time is zero",
        ),
        (
            "malformed item data",
            header + encode_record(RecordType.ITEM, Raw(b"[]")),
            "decode item record",
        ),
        (
            "missing item data",
            header + encode_record(RecordType.ITEM, Raw(b'{"Item":{"Sequence":1,"Kind":"turn"}}')),
            "decode item record",
        ),
        (
            "non-contiguous item",
            header + item_record(turn_item(2, Turn(ID="turn-1", Type=TurnType.REGULAR))),
            "has sequence 2, want 1",
        ),
        (
            "zero item time",
            header
            + item_record(
                Item(Sequence=1, Kind=ItemKind.TURN, Data=Turn(ID="turn-1", Type=TurnType.REGULAR))
            ),
            "recorded time is zero",
        ),
        (
            "wrong previous turn",
            with_turn
            + item_record(
                turn_item(2, Turn(ID="turn-2", PreviousTurnID="other", Type=TurnType.REGULAR))
            ),
            'previous turn is "other", want "turn-1"',
        ),
        (
            "duplicate turn",
            with_turn
            + item_record(
                turn_item(2, Turn(ID="turn-1", PreviousTurnID="turn-1", Type=TurnType.REGULAR))
            ),
            'append turn "turn-1": file already exists',
        ),
        (
            "operations on turn",
            header + item_record(turn, valid_operation("operation-1", Status.READY)),
            "initializes operations",
        ),
        (
            "status without operation",
            with_turn + item_record(status),
            "must initialize at least one operation",
        ),
        (
            "invalid initialized operation",
            with_turn + item_record(status, Operation(Type="test", Version=1, Status=Status.READY)),
            "operation ID is empty",
        ),
        (
            "duplicate initialized operation",
            with_status
            + item_record(
                Item(
                    Sequence=3,
                    RecordedAt=STATE_UPDATED_AT,
                    Kind=ItemKind.TOOL_CALL_STATUS,
                    Data=valid_status("turn-1", "call-2", "operation-1"),
                ),
                initial,
            ),
            'initialize operation "operation-1": file already exists',
        ),
        (
            "operation inherited across fork",
            with_status + item_record(fork_item(3)),
            "inherited item 1 initializes operations",
        ),
        (
            "operation update inside inherited prefix",
            header
            + item_record(turn)
            + encode_record(RecordType.OPERATION, OperationRecord(Operation=initial))
            + item_record(fork_item(2)),
            "appears in inherited history",
        ),
        (
            "malformed operation update",
            header + encode_record(RecordType.OPERATION, Raw(b"[]")),
            "decode operation record",
        ),
        (
            "missing operation update",
            header
            + encode_record(
                RecordType.OPERATION,
                OperationRecord(Operation=valid_operation("missing", Status.READY)),
            ),
            "file does not exist",
        ),
    ]


@pytest.mark.parametrize(
    ("encoded", "want"),
    [pytest.param(encoded, want, id=name) for name, encoded, want in _invalid_logs()],
)
def test_decode_log_rejects_invalid_records(encoded: bytes, want: str) -> None:
    with pytest.raises(StoreError) as caught:
        decode_log(encoded)
    assert want in str(caught.value)


def test_decode_log_keeps_missing_and_duplicate_kinds() -> None:
    logs = {name: encoded for name, encoded, _ in _invalid_logs()}
    with pytest.raises(NotExistError):
        decode_log(logs["missing operation update"])
    with pytest.raises(ExistError):
        decode_log(logs["duplicate turn"])


@pytest.mark.parametrize("outcome", ["incomplete", "response"])
def test_turn_types_survive_reopen(outcome: str) -> None:
    state = new_stored_state("session-1", STATE_CREATED_AT)
    state.append_turn(Turn(ID="turn-1", Type=TurnType.REGULAR), STATE_UPDATED_AT)
    response = valid_response("turn-1")
    if outcome == "response":
        state.append_model_response(response, STATE_UPDATED_AT)
    reopened, _ = decode_log(encode_initial_log(state.head.snapshot.session, state.items))
    assert reopened == state, "reopen changed state"
    if outcome == "response":
        with pytest.raises(ExistError):
            reopened.append_model_response(response, STATE_UPDATED_AT)
    else:
        reopened.append_model_response(response, STATE_UPDATED_AT)


# Golden fixtures.


def test_store_loads_golden_log() -> None:
    store = Store(TESTDATA)
    got, committed_size = store._read_state("golden-session")
    assert committed_size == len(Path(store.session_path("golden-session")).read_bytes())

    created_at = Timestamp.parse("2026-08-27T10:00:00Z")
    recorded_at = Timestamp.parse("2026-08-27T10:01:00Z")
    assert got.head.snapshot == Snapshot(session=Session(ID="golden-session", CreatedAt=created_at))
    assert got.items == [
        Item(
            Sequence=1,
            RecordedAt=recorded_at,
            Kind=ItemKind.INPUT,
            Data=Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"message":"hello"}')),
        ),
        Item(
            Sequence=2,
            RecordedAt=recorded_at,
            Kind=ItemKind.TURN,
            Data=Turn(ID="turn-1", Type=TurnType.REGULAR),
        ),
        Item(
            Sequence=3,
            RecordedAt=recorded_at,
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    ID="response-1",
                    Stop=StopReason.COMPLETE,
                    Output=[
                        LLMItem(
                            ProviderID="message-1",
                            Type=ItemType.MESSAGE,
                            Data=Message(Role=Role.ASSISTANT, Text="working", Phase="commentary"),
                        ),
                        LLMItem(
                            ProviderID="reasoning-1",
                            Type=ItemType.REASONING,
                            Data=Reasoning(Summary=["inspect"], Raw=Raw(b'{"encrypted":"opaque"}')),
                        ),
                    ],
                    Usage=Usage(
                        InputTokens=10,
                        CachedInputTokens=2,
                        CacheWriteInputTokens=1,
                        OutputTokens=4,
                        ReasoningTokens=3,
                        Raw=Raw(b'{"provider_total":14}'),
                    ),
                ),
            ),
        ),
        Item(
            Sequence=4,
            RecordedAt=recorded_at,
            Kind=ItemKind.TOOL_CALL_STATUS,
            Data=ToolCallStatus(
                TurnID="turn-1",
                CallID="call-1",
                Status=CallStatus(WaitingFor=["operation-1"]),
                Operations=[
                    Operation(
                        ID="operation-1",
                        Type="test",
                        Version=1,
                        Status=Status.READY,
                        State=Raw(b'{"step":1}'),
                        Idempotency=Raw(b'{"key":"one"}'),
                    )
                ],
            ),
        ),
    ]
    assert got.head.operations == [
        Operation(
            ID="operation-1",
            Type="test",
            Version=1,
            Status=Status.AWAITING,
            State=Raw(b'{"step":2}'),
            Idempotency=Raw(b'{"key":"one"}'),
        )
    ]


def test_store_loads_golden_fork_log() -> None:
    store = Store(TESTDATA)
    got, committed_size = store._read_state("golden-fork")
    assert committed_size == len(Path(store.session_path("golden-fork")).read_bytes())

    created_at = Timestamp.parse("2026-08-27T11:00:00Z")
    recorded_at = Timestamp.parse("2026-08-27T11:01:00Z")
    assert got.head.snapshot == Snapshot(session=Session(ID="golden-fork", CreatedAt=created_at))
    assert got.items == [
        Item(
            Sequence=1,
            RecordedAt=recorded_at,
            Kind=ItemKind.TURN,
            Data=Turn(ID="parent-turn", Type=TurnType.REGULAR),
        ),
        Item(
            Sequence=2,
            RecordedAt=recorded_at,
            Kind=ItemKind.FORK,
            Data=Fork(ParentID="parent-session", PreviousTurnID="parent-turn"),
        ),
        Item(
            Sequence=3,
            RecordedAt=recorded_at,
            Kind=ItemKind.INPUT,
            Data=Input(
                ID="child-input", Kind=InputKind.EXTERNAL, Payload=Raw(b'{"message":"child"}')
            ),
        ),
        Item(
            Sequence=4,
            RecordedAt=recorded_at,
            Kind=ItemKind.TURN,
            Data=Turn(ID="child-turn-1", PreviousTurnID="parent-turn", Type=TurnType.REGULAR),
        ),
        Item(
            Sequence=5,
            RecordedAt=recorded_at,
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(
                TurnID="child-turn-1",
                Response=Response(
                    ID="response-1",
                    Stop=StopReason.COMPLETE,
                    Output=[
                        LLMItem(
                            ProviderID="message-1",
                            Type=ItemType.MESSAGE,
                            Data=Message(Role=Role.ASSISTANT, Text="working", Phase="commentary"),
                        ),
                        LLMItem(
                            ProviderID="call-1",
                            Type=ItemType.TOOL_CALL,
                            Data=ToolCall(CallID="call-1", Name="test", Arguments="{}"),
                        ),
                        LLMItem(
                            ProviderID="reasoning-1",
                            Type=ItemType.REASONING,
                            Data=Reasoning(Summary=["inspect"], Raw=Raw(b'{"encrypted":"opaque"}')),
                        ),
                    ],
                    Usage=Usage(
                        InputTokens=10,
                        CachedInputTokens=2,
                        CacheWriteInputTokens=1,
                        OutputTokens=4,
                        ReasoningTokens=3,
                        Raw=Raw(b'{"provider_total":14}'),
                    ),
                ),
            ),
        ),
        Item(
            Sequence=6,
            RecordedAt=recorded_at,
            Kind=ItemKind.TURN,
            Data=Turn(ID="child-turn-2", PreviousTurnID="child-turn-1", Type=TurnType.REGULAR),
        ),
        Item(
            Sequence=7,
            RecordedAt=recorded_at,
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(
                TurnID="child-turn-2",
                Response=Response(
                    ID="response-2",
                    Stop=StopReason.REFUSED,
                    Failure=Failure(Code="server_error", Message="failed"),
                ),
            ),
        ),
        Item(
            Sequence=8,
            RecordedAt=recorded_at,
            Kind=ItemKind.TOOL_CALL_STATUS,
            Data=ToolCallStatus(
                TurnID="child-turn-1",
                CallID="call-error",
                Status=CallStatus(Error="invalid arguments"),
            ),
        ),
    ]
    assert got.head.operations == []


def test_golden_session_re_encodes_byte_identical() -> None:
    store = Store(TESTDATA)
    state, _ = store._read_state("golden-session")
    encoded = encode_initial_log(state.head.snapshot.session, state.items) + encode_record(
        RecordType.OPERATION, OperationRecord(Operation=state.head.operations[0])
    )
    assert encoded == (TESTDATA / "golden-session.session.jsonl").read_bytes()


def test_golden_fork_re_encodes_byte_identical() -> None:
    state, _ = Store(TESTDATA)._read_state("golden-fork")
    encoded = encode_initial_log(state.head.snapshot.session, state.items)
    assert encoded == (TESTDATA / "golden-fork.session.jsonl").read_bytes()


class _Record(gojson.Struct):
    type: str = ""
    data: Raw = Raw(b"")


class _UncheckedItem(gojson.Struct):
    Sequence: int = 0
    RecordedAt: Timestamp = Timestamp()
    Kind: str = ""
    Data: Raw = Raw(b"")


class _UncheckedItemRecord(gojson.Struct):
    Item: _UncheckedItem = struct_field(default_factory=_UncheckedItem)


def test_legacy_session_re_encodes_byte_identical() -> None:
    legacy = (TESTDATA / "legacy-session.session.jsonl").read_bytes()
    header = gojson.decode(gojson.decode(legacy, _Record).data, SessionRecord)
    assert header.Version == 1
    assert encode_record(RecordType.SESSION, header) == legacy


def test_soft_stop_session_re_encodes_byte_identical() -> None:
    header_line, item_line = (
        (TESTDATA / "soft-stop-session.session.jsonl").read_bytes().splitlines(keepends=True)
    )
    header = gojson.decode(gojson.decode(header_line, _Record).data, SessionRecord)
    assert encode_record(RecordType.SESSION, header) == header_line

    data = gojson.decode(item_line, _Record).data
    with pytest.raises(gojson.JSONError, match='unsupported control mode "soft"'):
        gojson.decode(data, ItemRecord)
    record = gojson.decode(data, _UncheckedItemRecord)
    input = gojson.decode(record.Item.Data, Input)
    assert input.Kind == InputKind.CONTROL
    record.Item.Data = Raw(gojson.encode(input))
    assert encode_record(RecordType.ITEM, record) == item_line


# Store internals.


async def test_store_caches_write_state_for_created_and_loaded_sessions(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    created = store._get_cached_write_state("session-1")
    assert created is not None
    assert created.head.id == "session-1"
    assert created.committed_size == len(read_file(store.session_path("session-1")))

    reopened = Store(tmp_path)
    assert reopened._get_cached_write_state("session-1") is None
    await reopened.inspect("session-1")
    assert reopened._get_cached_write_state("session-1") is None, "read cached the write state"
    await reopened.append_turn("session-1", Turn(ID="turn-1", Type=TurnType.REGULAR))
    assert reopened._get_cached_write_state("session-1") is not None


async def test_store_load_rejects_invalid_completed_record(tmp_path: Path) -> None:
    store = Store(tmp_path)
    write_file(store.session_path("session-1"), empty_log() + b"{\n")
    with pytest.raises(StoreError, match='read session "session-1": decode record 2'):
        await store.inspect("session-1")


async def test_store_load_rejects_mismatched_filename_and_header(tmp_path: Path) -> None:
    store = Store(tmp_path)
    other = new_stored_state("other", STATE_CREATED_AT)
    encoded = encode_initial_log(other.head.snapshot.session, other.items)
    write_file(store.session_path("session-1"), encoded)
    with pytest.raises(StoreError, match='file contains session "other"'):
        await store.inspect("session-1")


def test_append_file_reports_open_failure(tmp_path: Path) -> None:
    with pytest.raises(NotExistError, match="open session log for append"):
        append_file(str(tmp_path / "missing" / "session.jsonl"), 0, b"{}\n")


def test_publish_file_cleans_temporary_file_when_target_is_a_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (target / "entry").write_bytes(b"")
    with pytest.raises(StoreError, match="publish session file"):
        publish_file(str(tmp_path), str(target), b"replacement")
    assert (target / "entry").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".session-")]


def test_publish_file_reports_temporary_file_creation_failure(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_bytes(b"")
    with pytest.raises(StoreError, match="create temporary session file"):
        publish_file(str(file), str(file / "target"), b"")


def test_persist_temporary_reports_closed_file_failures(tmp_path: Path) -> None:
    fd, _ = tempfile.mkstemp(prefix="closed-", dir=tmp_path)
    os.close(fd)
    with pytest.raises(StoreError) as caught:
        persist_temporary(fd, b"state")
    assert "write temporary session file" in str(caught.value)
    assert "close temporary session file" in str(caught.value)
    cause = caught.value.__cause__
    assert isinstance(cause, StoreError)
    assert isinstance(cause.__cause__, OSError)
    assert cause.__cause__.errno == errno.EBADF


def test_persist_temporary_reports_sync_failure() -> None:
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        persist_temporary(fd, b"state")
    except StoreError as error:
        message = str(error)
    else:
        pytest.skip("platform permits syncing the null device")
    assert "sync temporary session file" in message


def test_sync_directory_fd_reports_closed_handle(tmp_path: Path) -> None:
    fd = os.open(tmp_path, os.O_RDONLY)
    os.close(fd)
    with pytest.raises(StoreError) as caught:
        sync_directory_fd(fd)
    assert "sync session store directory" in str(caught.value)
    assert "close session store directory" in str(caught.value)


def test_sync_directory_reports_open_failure(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="open session store directory for sync"):
        sync_directory(str(tmp_path / "missing"))


def test_new_rejects_invalid_directories(tmp_path: Path) -> None:
    with pytest.raises(StoreError):
        Store("  ")
    path = tmp_path / "file"
    path.write_bytes(b"")
    with pytest.raises(StoreError, match="create session store directory"):
        Store(path)


def test_new_creates_private_directory(tmp_path: Path) -> None:
    directory = tmp_path / "parent" / "sessions"
    Store(directory)
    for path in (directory.parent, directory):
        assert path.stat().st_mode & 0o777 == 0o700 & ~_umask()


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


@pytest.mark.parametrize("outcome", ["incomplete", "response"])
async def test_compaction_history_reopens_and_forks(tmp_path: Path, outcome: str) -> None:
    store = Store(tmp_path)
    await store.create("parent")
    observed: list[Item] = []
    store.add_observer(lambda _, item: observed.append(item))
    await store.append_turn("parent", Turn(ID="compact", Type=TurnType.COMPACTION))
    if outcome != "incomplete":
        await store.append_input(
            "parent", Input(ID="late", Kind=InputKind.EXTERNAL, Payload=Raw(b'"hello"'))
        )
        await store.append_model_response("parent", valid_response("compact"))

    store = Store(tmp_path)
    page = await store.items("parent", 0, 100)
    assert page.items == observed
    # A later turn must not extend the selected fork boundary.
    await store.append_turn(
        "parent", Turn(ID="next", PreviousTurnID="compact", Type=TurnType.REGULAR)
    )
    await store.fork("child", "parent", "compact")

    store = Store(tmp_path)
    child = await store.items("child", 0, 100)
    assert len(child.items) == len(observed) + 1
    assert child.items[: len(observed)] == observed
    assert child.items[-1].Kind == ItemKind.FORK
    with pytest.raises(NotExistError):
        await store.append_model_response("child", valid_response("compact"))
    await store.append_turn(
        "child", Turn(ID="child-turn", PreviousTurnID="compact", Type=TurnType.COMPACTION)
    )
    await store.append_model_response("child", valid_response("child-turn"))
    await store.resume("child")


# Listing internals.


@pytest.mark.parametrize(
    ("removed", "want"),
    [
        pytest.param(["middle"], ["first", "last"], id="one removed"),
        pytest.param(["first", "middle", "last"], [], id="all removed"),
    ],
)
async def test_list_session_entries_skips_removed_files(
    tmp_path: Path, removed: list[str], want: list[str]
) -> None:
    store = Store(tmp_path)
    for id in ("first", "middle", "last"):
        await store.create(id)
    with os.scandir(tmp_path) as iterator:
        entries = list(iterator)
    for id in removed:
        os.remove(store.session_path(id))
    assert [info.id for info in list_session_entries(entries)] == want


async def test_list_session_entries_skips_files_with_stat_errors(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    store = Store(directory)
    await store.create("session-1")
    with os.scandir(directory) as iterator:
        entries = list(iterator)
    os.remove(store.session_path("session-1"))
    directory.rmdir()
    directory.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        entries[0].stat(follow_symlinks=False)
    assert list_session_entries(entries) == []
