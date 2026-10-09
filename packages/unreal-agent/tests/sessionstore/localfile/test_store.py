import asyncio
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent.gojson import Raw, Timestamp
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Response,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
    Usage,
)
from unreal_agent.operation import Operation, Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import (
    BEFORE_FIRST,
    ExistError,
    Fork,
    Item,
    ItemKind,
    ModelResponse,
    NotExistError,
    SessionInfo,
    StoreError,
    ToolCallStatus,
)
from unreal_agent.sessionstore.localfile import Store
from unreal_agent.sessionstore.localfile import store as store_module
from unreal_agent.tool import CallStatus

TESTDATA = Path(__file__).parent / "testdata"
INVALID_BYTE = b"\xff".decode("utf-8", "surrogateescape")

type StoreCall = Callable[[Store], Awaitable[object]]


def session_path(directory: Path, id: str) -> Path:
    return directory / f"{id}.session.jsonl"


async def create_turn(store: Store, session_id: str, turn_id: str, previous_turn_id: str) -> None:
    await store.append_turn(
        session_id, Turn(ID=turn_id, PreviousTurnID=previous_turn_id, Type=TurnType.REGULAR)
    )


async def create_session_with_turn(
    store: Store, session_id: str, turn_id: str, previous_turn_id: str
) -> None:
    await store.create(session_id)
    await create_turn(store, session_id, turn_id, previous_turn_id)


def session_file_contents(directory: Path) -> bytes:
    paths = [path for path in directory.iterdir() if path.name.endswith(".session.jsonl")]
    assert len(paths) == 1, "expected exactly one session file"
    return paths[0].read_bytes()


async def assert_stored_item_kinds(store: Store, id: str, *want: ItemKind) -> None:
    page = await store.items(id, BEFORE_FIRST, 100)
    assert [item.Kind for item in page.items] == list(want), f"{id} items = {page.items!r}"


def block_session_target(directory: Path, id: str) -> Path:
    target = session_path(directory, id)
    target.mkdir(mode=0o700)
    (target / "sentinel").write_bytes(b"unchanged")
    return target


def assert_blocked_session_target(target: Path) -> None:
    assert (target / "sentinel").read_bytes() == b"unchanged"


def assert_no_session_temporary_files(directory: Path) -> None:
    for path in directory.iterdir():
        assert not (path.name.startswith(".session-") and path.name.endswith(".tmp")), (
            f"temporary file remains: {path.name!r}"
        )


def external_input(id: str, payload: bytes = b"{}") -> Input:
    return Input(ID=id, Kind=InputKind.EXTERNAL, Payload=Raw(payload))


def ready_operation(id: str, status: Status = Status.READY) -> Operation:
    return Operation(ID=id, Type="test", Version=1, Status=status)


def waiting_status(turn_id: str, call_id: str, *operations: Operation) -> ToolCallStatus:
    return ToolCallStatus(
        TurnID=turn_id,
        CallID=call_id,
        Status=CallStatus(WaitingFor=[value.ID for value in operations]),
        Operations=list(operations),
    )


def tool_call_response(turn_id: str, *call_ids: str) -> ModelResponse:
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


def invalid_response_call(item: LLMItem) -> StoreCall:
    def call(store: Store) -> Awaitable[None]:
        return store.append_model_response(
            "session-1", ModelResponse(TurnID="turn-1", Response=Response(Output=[item]))
        )

    return call


async def test_store_persists_typed_history_and_pagination(tmp_path: Path) -> None:
    store = Store(tmp_path)
    snapshot = await store.create("session-1")
    assert session_path(tmp_path, "session-1").exists()

    input_item = LLMItem(
        ProviderID="input-provider-id",
        Type=ItemType.MESSAGE,
        Data=Message(Role=Role.USER, Text="hello"),
    )
    input = Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(gojson.encode(input_item)))
    await store.append_input(snapshot.session.ID, input)
    turn = Turn(ID="turn-1", Type=TurnType.REGULAR)
    await store.append_turn(snapshot.session.ID, turn)

    response = ModelResponse(
        TurnID=turn.ID,
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
                    Data=ToolCall(CallID="call-1", Name="bash", Arguments='{"command":"pwd"}'),
                ),
                LLMItem(
                    ProviderID="reasoning-1",
                    Type=ItemType.PROVIDER,
                    Data=ProviderItem(
                        Type="reasoning",
                        Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text="inspect"),
                        Raw=Raw(b'{"encrypted":"opaque"}'),
                    ),
                ),
            ],
            Usage=Usage(
                InputTokens=10,
                OutputTokens=4,
                ByModel={
                    "primary": TokenUsage(InputTokens=8, OutputTokens=2),
                    "fallback": TokenUsage(
                        InputTokens=10,
                        CachedInputTokens=3,
                        CacheWriteInputTokens=2,
                        OutputTokens=4,
                        ReasoningTokens=1,
                    ),
                },
                Raw=Raw(b'{"provider":14}'),
            ),
        ),
    )
    await store.append_model_response(snapshot.session.ID, response)
    initial = Operation(
        ID="operation-1",
        Type="test",
        Version=1,
        Status=Status.AWAITING,
        State=Raw(b'{"step":1}'),
        Idempotency=Raw(b'{"key":"one"}'),
    )
    status = waiting_status(turn.ID, "call-1", initial)
    await store.append_tool_call_status(snapshot.session.ID, status)

    reopened = Store(tmp_path)
    inspected = await reopened.inspect(snapshot.session.ID)
    assert inspected.session.ID == snapshot.session.ID
    assert inspected.session.CreatedAt == snapshot.session.CreatedAt

    first = await reopened.items(snapshot.session.ID, BEFORE_FIRST, 2)
    assert (len(first.items), first.next_after, first.more) == (2, 2, True)
    second = await reopened.items(snapshot.session.ID, first.next_after, 2)
    assert (len(second.items), second.next_after, second.more) == (2, 4, False)
    items = first.items + second.items
    want_kinds = [
        ItemKind.INPUT,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.TOOL_CALL_STATUS,
    ]
    for index, item in enumerate(items):
        assert item.Sequence == index + 1
        assert item.Kind == want_kinds[index]
        assert not item.RecordedAt.is_zero()
    assert items[0].Data == input
    assert items[2].Data == response
    assert items[3].Data == status

    resume = await reopened.resume(snapshot.session.ID)
    assert resume.operations == [initial]
    assert resume.external_input_ids == ["input-1"]

    after_end = await reopened.items(snapshot.session.ID, 100, 10)
    assert (after_end.items, after_end.next_after, after_end.more) == ([], 100, False)


async def test_store_notifies_observers_synchronously_after_item_persistence(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    notifications: list[str] = []

    def observe(name: str) -> Callable[[str, Item], None]:
        def observer(id: str, item: Item) -> None:
            state, _ = Store(tmp_path)._read_state(id)
            assert state.items[item.Sequence - 1 : item.Sequence] == [item]
            notifications.append(f"{name}:{item.Kind}")

        return observer

    first_id = store.add_observer(observe("first"))
    second_id = store.add_observer(observe("second"))
    await store.append_turn("session-1", Turn(ID="turn-1", Type=TurnType.REGULAR))
    store.remove_observer(first_id)
    await store.append_model_response(
        "session-1", ModelResponse(TurnID="turn-1", Response=Response(ID="response-1"))
    )
    store.remove_observer(second_id)
    await store.append_input("session-1", external_input("input-1"))

    assert notifications == ["first:turn", "second:turn", "second:model_response"]


async def test_store_observer_receives_canonical_tool_call_status_item(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    observed: list[Item] = []

    def observer(id: str, item: Item) -> None:
        assert id == "session-1"
        observed.append(item)

    store.add_observer(observer)
    initial = ready_operation("operation-1")
    status = waiting_status("turn-1", "call-1", initial)
    await store.append_tool_call_status("session-1", status)
    await store.save_operation("session-1", replace(initial, Status=Status.AWAITING))

    assert len(observed) == 1
    assert observed[0].Kind == ItemKind.TOOL_CALL_STATUS
    assert observed[0].Data == status


async def test_store_does_not_notify_observer_when_item_persistence_fails(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    observed: list[Item] = []
    store.add_observer(lambda _, item: observed.append(item))

    with pytest.raises(StoreError):
        await store.append_turn("missing", Turn(ID="turn-1", Type=TurnType.REGULAR))
    assert observed == []


async def test_store_notifies_observer_of_fork_item(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "parent", "turn-1", "")
    observed: list[tuple[str, Item]] = []
    store.add_observer(lambda id, item: observed.append((id, item)))

    await store.fork("child", "parent", "turn-1")

    assert len(observed) == 1
    assert observed[0][0] == "child"
    assert observed[0][1].Kind == ItemKind.FORK


async def test_store_rejects_legacy_session_resume() -> None:
    with pytest.raises(StoreError, match="legacy session format version 1 cannot be resumed"):
        await Store(TESTDATA).resume("legacy-session")


async def test_store_rejects_soft_stop_history() -> None:
    with pytest.raises(StoreError, match='unsupported control mode "soft"'):
        await Store(TESTDATA).resume("soft-stop-session")


async def test_store_appends_jsonl_records(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    path = session_path(tmp_path, "session-1")
    created_info = path.stat()
    created = path.read_bytes()

    await create_turn(store, "session-1", "turn-1", "")
    initial = ready_operation("operation-1")
    await store.append_tool_call_status("session-1", waiting_status("turn-1", "call-1", initial))
    await store.save_operation("session-1", replace(initial, Status=Status.AWAITING))

    appended_info = path.stat()
    assert os.path.samestat(created_info, appended_info), "mutation replaced the session log"
    appended = path.read_bytes()
    assert appended.startswith(created), "mutation rewrote the committed log prefix"
    lines = appended.removesuffix(b"\n").split(b"\n")
    assert [gojson.validate(line)["type"] for line in lines] == [
        "session",
        "item",
        "item",
        "operation",
    ]
    assert b'"Operations":[' in lines[2]

    resume = await Store(tmp_path).resume("session-1")
    assert [value.Status for value in resume.operations] == [Status.AWAITING]


async def test_store_persists_repeated_tool_call_statuses(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    initial = ready_operation("operation-1")
    status = waiting_status("turn-1", "call-1", initial)
    await store.append_tool_call_status("session-1", status)
    completed = replace(initial, Status=Status.COMPLETED)
    await store.save_operation("session-1", completed)
    await store.append_tool_call_status("session-1", replace(status, Operations=[completed]))

    reopened = Store(tmp_path)
    await assert_stored_item_kinds(
        reopened,
        "session-1",
        ItemKind.TURN,
        ItemKind.TOOL_CALL_STATUS,
        ItemKind.TOOL_CALL_STATUS,
    )
    page = await reopened.items("session-1", BEFORE_FIRST, 10)
    first, last = page.items[1].Data, page.items[2].Data
    assert isinstance(first, ToolCallStatus)
    assert isinstance(last, ToolCallStatus)
    assert first.Operations == [initial]
    assert last.Operations == [completed]
    assert (await reopened.resume("session-1")).operations == []


async def test_reopened_store_continues_from_derived_write_state(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    await store.append_input("session-1", external_input("input-1"))
    await create_turn(store, "session-1", "turn-1", "")
    initial = ready_operation("operation-1")
    await store.append_tool_call_status("session-1", waiting_status("turn-1", "call-1", initial))

    reopened = Store(tmp_path)
    await reopened.save_operation("session-1", replace(initial, Status=Status.AWAITING))
    await reopened.append_input("session-1", external_input("input-2"))
    await create_turn(reopened, "session-1", "turn-2", "turn-1")

    verifier = Store(tmp_path)
    await assert_stored_item_kinds(
        verifier,
        "session-1",
        ItemKind.INPUT,
        ItemKind.TURN,
        ItemKind.TOOL_CALL_STATUS,
        ItemKind.INPUT,
        ItemKind.TURN,
    )
    await verifier.inspect("session-1")
    resume = await verifier.resume("session-1")
    assert [value.Status for value in resume.operations] == [Status.AWAITING]


async def test_known_session_append_does_not_read_log(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    path = session_path(tmp_path, "session-1")
    path.chmod(0o200)
    try:
        try:
            path.read_bytes()
        except PermissionError:
            pass
        else:
            pytest.skip("platform permits reading a write-only file")

        await store.append_turn("session-1", Turn(ID="turn-1", Type=TurnType.REGULAR))
    finally:
        path.chmod(0o600)
    await assert_stored_item_kinds(Store(tmp_path), "session-1", ItemKind.TURN)


async def test_store_recovers_incomplete_jsonl_tail_before_append(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    path = session_path(tmp_path, "session-1")
    with path.open("ab") as file:
        file.write(b"incomplete-tail")

    await assert_stored_item_kinds(store, "session-1", ItemKind.TURN)
    await store.append_model_response(
        "session-1", ModelResponse(TurnID="turn-1", Response=Response(Output=[]))
    )

    contents = path.read_bytes()
    assert b"incomplete-tail" not in contents
    assert contents.endswith(b"\n")
    await assert_stored_item_kinds(
        Store(tmp_path), "session-1", ItemKind.TURN, ItemKind.MODEL_RESPONSE
    )


async def test_store_recovers_from_append_failure(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    path = session_path(tmp_path, "session-1")
    committed = path.read_bytes()
    path.unlink()
    path.mkdir(mode=0o700)

    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[
                LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="complete"))
            ]
        ),
    )
    with pytest.raises(StoreError):
        await store.append_model_response("session-1", response)

    path.rmdir()
    path.write_bytes(committed)
    await store.append_model_response("session-1", response)

    page = await Store(tmp_path).items("session-1", BEFORE_FIRST, 10)
    assert len(page.items) == 2
    assert page.items[1].Kind == ItemKind.MODEL_RESPONSE
    assert page.items[1].Data == response


async def test_create_replaces_populated_session(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "old-turn", "")
    await store.append_input("session-1", external_input("old-input"))
    await store.append_tool_call_status(
        "session-1",
        waiting_status("old-turn", "old-call", ready_operation("old-operation", Status.AWAITING)),
    )

    replaced = await store.create("session-1")
    reopened = Store(tmp_path)
    assert await reopened.inspect("session-1") == replaced
    assert (await reopened.items("session-1", BEFORE_FIRST, 100)).items == []
    assert (await reopened.resume("session-1")).operations == []
    await create_turn(store, "session-1", "fresh-turn", "")
    await assert_stored_item_kinds(Store(tmp_path), "session-1", ItemKind.TURN)
    assert_no_session_temporary_files(tmp_path)


async def test_failed_replacement_evicts_cached_write_state(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    target = session_path(tmp_path, "session-1")
    target.unlink()
    block_session_target(tmp_path, "session-1")

    with pytest.raises(StoreError):
        await store.create("session-1")
    (target / "sentinel").unlink()
    target.rmdir()
    await Store(tmp_path).create("session-1")
    await store.append_turn("session-1", Turn(ID="fresh-turn", Type=TurnType.REGULAR))


async def test_store_resumes_unsettled_operations_and_saves_latest_state(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    operations = [
        replace(ready_operation(id, status), State=Raw(f'{{"n":{index + 1}}}'.encode()))
        for index, (id, status) in enumerate(
            [
                ("ready", Status.READY),
                ("awaiting", Status.AWAITING),
                ("canceling", Status.CANCELING),
                ("complete", Status.COMPLETED),
                ("failed", Status.FAILED),
                ("canceled", Status.CANCELED),
            ]
        )
    ]
    status = waiting_status("turn-1", "call-1", *operations)
    await store.append_tool_call_status("session-1", status)

    want_ids = ["ready", "awaiting", "canceling"]
    resume = await store.resume("session-1")
    assert [value.ID for value in resume.operations] == want_ids

    updated = replace(operations[1], Status=Status.COMPLETED, State=Raw(b'{ "n": 20 }'))
    await store.save_operation("session-1", updated)
    resume = await store.resume("session-1")
    assert resume.operations[1].Status == Status.COMPLETED
    assert bytes(resume.operations[1].State) == b'{"n":20}'
    assert [value.ID for value in resume.operations] == want_ids

    operations[1] = updated
    status = replace(status, Operations=list(operations))
    await store.append_tool_call_status("session-1", status)
    resume = await store.resume("session-1")
    assert resume.operations == [operations[0], operations[2]]

    for index in (0, 2):
        operations[index] = replace(operations[index], Status=Status.COMPLETED)
        await store.save_operation("session-1", operations[index])
    await store.append_tool_call_status("session-1", replace(status, Operations=operations))
    assert (await store.resume("session-1")).operations == []


async def test_fork_copies_history_through_turn_without_operations(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "parent", "turn-1", "")
    response = tool_call_response("turn-1", "call-1")
    response.Response.ID = "response-1"
    response.Response.Stop = StopReason.COMPLETE
    await store.append_model_response("parent", response)
    parent_operation = ready_operation("parent-operation", Status.AWAITING)
    await store.append_tool_call_status(
        "parent", waiting_status("turn-1", "call-1", parent_operation)
    )
    await store.append_input("parent", external_input("next-input"))
    await create_turn(store, "parent", "turn-2", "turn-1")
    await store.append_model_response(
        "parent", ModelResponse(TurnID="turn-2", Response=Response(ID="response-2", Output=[]))
    )

    snapshot = await store.fork("child", "parent", "turn-1")
    assert snapshot.session.ID == "child"
    page = await store.items("child", BEFORE_FIRST, 100)
    assert [item.Kind for item in page.items] == [
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.TOOL_CALL_STATUS,
        ItemKind.FORK,
    ]
    assert page.items[-1].Data == Fork(ParentID="parent", PreviousTurnID="turn-1")
    assert (await store.resume("child")).operations == []

    await store.append_input("child", external_input("child-input"))
    await create_turn(store, "child", "child-turn", "turn-1")


async def test_fork_persists_delayed_status_boundaries(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "parent", "turn-1", "")
    await store.append_model_response("parent", tool_call_response("turn-1", "call-1"))
    await create_turn(store, "parent", "turn-2", "turn-1")
    await store.append_model_response(
        "parent", ModelResponse(TurnID="turn-2", Response=Response(Output=[]))
    )
    await store.append_tool_call_status(
        "parent", waiting_status("turn-1", "call-1", ready_operation("operation-1"))
    )

    await store.fork("child-1", "parent", "turn-1")
    await store.fork("child-2", "parent", "turn-2")

    reopened = Store(tmp_path)
    await assert_stored_item_kinds(
        reopened, "child-1", ItemKind.TURN, ItemKind.MODEL_RESPONSE, ItemKind.FORK
    )
    await assert_stored_item_kinds(
        reopened,
        "child-2",
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.TURN,
        ItemKind.MODEL_RESPONSE,
        ItemKind.FORK,
    )
    for id in ("child-1", "child-2"):
        assert (await reopened.resume(id)).operations == [], f"{id} inherited operations"


async def test_store_reports_missing_values_and_cancellation(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    await store.create("session-1")
    with pytest.raises(FileNotFoundError):
        await store.inspect("missing")
    with pytest.raises(NotExistError):
        await store.save_operation("session-1", ready_operation("missing"))
    with pytest.raises(StoreError):
        await store.items("session-1", 0, 0)

    task = asyncio.create_task(store.inspect("session-1"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_mutations_use_the_supplied_session_id(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    await create_session_with_turn(store, "session-2", "turn-2", "")

    with pytest.raises(NotExistError):
        await store.append_model_response(
            "session-1", ModelResponse(TurnID="turn-2", Response=Response(Output=[]))
        )
    operation = ready_operation("operation-2")
    await store.append_tool_call_status("session-2", waiting_status("turn-2", "call-2", operation))
    completed = replace(operation, Status=Status.COMPLETED)
    with pytest.raises(NotExistError):
        await store.save_operation("session-1", completed)
    await store.save_operation("session-2", completed)


async def test_separate_stores_support_concurrent_creates(tmp_path: Path) -> None:
    stores = [Store(tmp_path), Store(tmp_path)]
    await asyncio.gather(
        *(
            stores[index % len(stores)].create(f"session-{chr(ord('a') + index)}")
            for index in range(24)
        )
    )


async def test_separate_stores_can_replace_the_same_session(tmp_path: Path) -> None:
    stores = [Store(tmp_path), Store(tmp_path)]
    await asyncio.gather(*(store.create("session-1") for store in stores))
    await stores[0].inspect("session-1")


async def test_same_store_serializes_writes_for_one_session(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    await asyncio.gather(
        *(store.append_input("session-1", external_input(f"input-{index}")) for index in range(20))
    )
    page = await Store(tmp_path).items("session-1", BEFORE_FIRST, 100)
    assert [item.Sequence for item in page.items] == list(range(1, 21))
    assert [item.Data.ID for item in page.items if isinstance(item.Data, Input)] == [
        f"input-{index}" for index in range(20)
    ]


@pytest.mark.parametrize(
    ("status", "operations"),
    [
        pytest.param(
            CallStatus(WaitingFor=["valid", "invalid"]),
            [ready_operation("valid"), Operation(ID="invalid", Status=Status.READY)],
            id="invalid operation",
        ),
        pytest.param(
            CallStatus(WaitingFor=["operation-a"]),
            [ready_operation("operation-b")],
            id="mismatched reference",
        ),
        pytest.param(
            CallStatus(Error="invalid arguments"),
            [ready_operation("operation-1")],
            id="error with operation",
        ),
    ],
)
async def test_rejected_tool_call_statuses_do_not_change_persisted_state(
    tmp_path: Path, status: CallStatus, operations: list[Operation]
) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    before = session_file_contents(tmp_path)

    with pytest.raises(StoreError):
        await store.append_tool_call_status(
            "session-1",
            ToolCallStatus(TurnID="turn-1", CallID="call-1", Status=status, Operations=operations),
        )
    assert session_file_contents(tmp_path) == before
    await assert_stored_item_kinds(store, "session-1", ItemKind.TURN)
    assert (await store.resume("session-1")).operations == []


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda store: store.items("missing", BEFORE_FIRST, 1), id="items"),
        pytest.param(
            lambda store: store.append_input("missing", external_input("input-1")),
            id="append input",
        ),
        pytest.param(
            lambda store: store.append_turn("missing", Turn(ID="turn-1", Type=TurnType.REGULAR)),
            id="append turn",
        ),
        pytest.param(
            lambda store: store.append_model_response("missing", ModelResponse(TurnID="turn-1")),
            id="append model response",
        ),
        pytest.param(
            lambda store: store.append_tool_call_status(
                "missing", ToolCallStatus(TurnID="turn-1", CallID="call-1")
            ),
            id="append tool-call status",
        ),
        pytest.param(
            lambda store: store.save_operation("missing", ready_operation("operation-1")),
            id="save operation",
        ),
        pytest.param(lambda store: store.resume("missing"), id="resume"),
    ],
)
async def test_missing_sessions_fail_every_read_modify_write_method(
    tmp_path: Path, call: StoreCall
) -> None:
    with pytest.raises(FileNotFoundError):
        await call(Store(tmp_path))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda store: store.append_input("session-1", Input()),
            id="empty input ID",
        ),
        pytest.param(
            lambda store: store.append_turn(
                "session-1", Turn(ID="turn-2", PreviousTurnID="wrong", Type=TurnType.REGULAR)
            ),
            id="wrong previous turn",
        ),
        pytest.param(
            lambda store: store.append_model_response(
                "session-1",
                ModelResponse(TurnID="turn-1", Response=Response(Usage=Usage(Raw=Raw(b"{")))),
            ),
            id="invalid response usage",
        ),
        pytest.param(
            invalid_response_call(LLMItem(Type=ItemType.MESSAGE, Data=ToolCall())),
            id="message data has wrong type",
        ),
        pytest.param(
            invalid_response_call(LLMItem(Type=ItemType.PROVIDER, Data=Message())),
            id="provider data has wrong type",
        ),
        pytest.param(
            invalid_response_call(
                LLMItem(Type=ItemType.PROVIDER, Data=ProviderItem(Type="reasoning", Raw=Raw(b"{")))
            ),
            id="provider raw data is invalid JSON",
        ),
        pytest.param(
            invalid_response_call(LLMItem(Type=cast(ItemType, "unknown"), Data=Message())),
            id="unsupported response item type",
        ),
        pytest.param(
            invalid_response_call(
                LLMItem(
                    Type=ItemType.MESSAGE,
                    Data=Message(Role=Role.ASSISTANT, Text=INVALID_BYTE),
                )
            ),
            id="invalid UTF-8 in response",
        ),
        pytest.param(
            lambda store: store.save_operation(
                "session-1", Operation(ID="operation-1", Status=Status.READY)
            ),
            id="invalid operation update",
        ),
    ],
)
async def test_rejected_public_transitions_do_not_change_persisted_state(
    tmp_path: Path, call: StoreCall
) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    before = session_file_contents(tmp_path)
    with pytest.raises(StoreError):
        await call(store)
    assert session_file_contents(tmp_path) == before


@pytest.mark.parametrize(
    "id",
    [
        "",
        ".",
        "..",
        "nested/session",
        "/absolute",
        "with_underscore",
        "with space",
        "unicode-ø",
        INVALID_BYTE,
    ],
)
async def test_store_rejects_invalid_session_ids(tmp_path: Path, id: str) -> None:
    with pytest.raises(StoreError):
        await Store(tmp_path).create(id)


async def test_store_rejects_invalid_session_ids_and_cancellation(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with pytest.raises(StoreError):
        await store.inspect("")
    with pytest.raises(StoreError):
        await store.fork("", "parent", "turn-1")

    task = asyncio.create_task(store.create("session-1"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await store.create("cached-session")
    before = session_file_contents(tmp_path)
    task = asyncio.create_task(
        store.append_turn("cached-session", Turn(ID="turn-1", Type=TurnType.REGULAR))
    )
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session_file_contents(tmp_path) == before


async def test_cancelled_write_finishes_and_notifies_before_cancelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    observed: list[Item] = []
    store.add_observer(lambda _, item: observed.append(item))
    started = threading.Event()
    release = threading.Event()
    append_file = store_module.append_file

    def blocking_append(path: str, committed_size: int, data: bytes) -> None:
        started.set()
        release.wait()
        append_file(path, committed_size, data)

    monkeypatch.setattr(store_module, "append_file", blocking_append)
    task = asyncio.create_task(
        store.append_turn("session-1", Turn(ID="turn-1", Type=TurnType.REGULAR))
    )
    await asyncio.to_thread(started.wait)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [item.Kind for item in observed] == [ItemKind.TURN]
    await assert_stored_item_kinds(Store(tmp_path), "session-1", ItemKind.TURN)
    await create_turn(store, "session-1", "turn-2", "turn-1")


async def test_write_after_cancellation_from_observer_is_not_persisted(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    observed: list[Item] = []

    def cancel_writer(_: str, item: Item) -> None:
        observed.append(item)
        if task is not None:
            task.cancel()

    task: asyncio.Task[None] | None = None
    store.add_observer(cancel_writer)

    async def append_turns() -> None:
        await create_turn(store, "session-1", "turn-1", "")
        await create_turn(store, "session-1", "turn-2", "turn-1")

    task = asyncio.create_task(append_turns())
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [item.Kind for item in observed] == [ItemKind.TURN]
    await assert_stored_item_kinds(Store(tmp_path), "session-1", ItemKind.TURN)


async def test_fork_rejects_self(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    before = session_file_contents(tmp_path)
    with pytest.raises(StoreError, match="onto itself"):
        await store.fork("session-1", "session-1", "turn-1")
    assert session_file_contents(tmp_path) == before


async def test_fork_reports_missing_parent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await Store(tmp_path).fork("child", "missing", "turn-1")


async def test_fork_reports_missing_turn(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("parent")
    with pytest.raises(NotExistError):
        await store.fork("child", "parent", "missing")


async def test_fork_replaces_existing_destination(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "parent", "turn-1", "")
    await create_session_with_turn(store, "child", "old-child-turn", "")
    await store.append_tool_call_status(
        "child",
        waiting_status(
            "old-child-turn",
            "old-child-call",
            ready_operation("old-child-operation", Status.AWAITING),
        ),
    )
    forked = await store.fork("child", "parent", "turn-1")

    reopened = Store(tmp_path)
    assert await reopened.inspect("child") == forked
    page = await reopened.items("child", BEFORE_FIRST, 100)
    assert [item.Kind for item in page.items] == [ItemKind.TURN, ItemKind.FORK]
    assert page.items[0].Data == Turn(ID="turn-1", Type=TurnType.REGULAR)
    assert (await reopened.resume("child")).operations == []

    await store.append_input("child", external_input("child-input"))
    await create_turn(store, "child", "child-turn", "turn-1")
    await assert_stored_item_kinds(
        Store(tmp_path),
        "child",
        ItemKind.TURN,
        ItemKind.FORK,
        ItemKind.INPUT,
        ItemKind.TURN,
    )


async def test_create_reports_publication_failure(tmp_path: Path) -> None:
    store = Store(tmp_path)
    target = block_session_target(tmp_path, "session-1")
    with pytest.raises(StoreError) as caught:
        await store.create("session-1")
    assert 'create session "session-1"' in str(caught.value)
    assert "publish session file" in str(caught.value)
    assert_blocked_session_target(target)
    assert_no_session_temporary_files(tmp_path)


async def test_fork_reports_publication_failure(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "parent", "turn-1", "")
    parent_path = session_path(tmp_path, "parent")
    parent_before = parent_path.read_bytes()
    target = block_session_target(tmp_path, "child")

    with pytest.raises(StoreError) as caught:
        await store.fork("child", "parent", "turn-1")
    assert 'fork session "child"' in str(caught.value)
    assert "publish session file" in str(caught.value)
    assert parent_path.read_bytes() == parent_before
    assert_blocked_session_target(target)
    assert_no_session_temporary_files(tmp_path)


async def test_store_recovers_from_encoding_failure(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    before = session_file_contents(tmp_path)
    with pytest.raises(StoreError):
        await store.append_model_response(
            "session-1",
            ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    Output=[
                        LLMItem(
                            Type=ItemType.MESSAGE,
                            Data=Message(Role=Role.ASSISTANT, Text=INVALID_BYTE),
                        )
                    ]
                ),
            ),
        )
    assert session_file_contents(tmp_path) == before

    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[
                LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="complete"))
            ]
        ),
    )
    await store.append_model_response("session-1", response)
    page = await Store(tmp_path).items("session-1", BEFORE_FIRST, 10)
    assert [item.Kind for item in page.items] == [ItemKind.TURN, ItemKind.MODEL_RESPONSE]
    assert page.items[1].Data == response


async def test_store_errors_mark_missing_and_duplicate_values(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "session-1", "turn-1", "")
    with pytest.raises(FileExistsError) as caught:
        await create_turn(store, "session-1", "turn-1", "turn-1")
    assert isinstance(caught.value, ExistError)
    assert str(caught.value) == 'append turn "turn-1": file already exists'


async def test_list_sessions(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert await store.list_sessions() == []

    want = [
        SessionInfo(id="a", last_updated_at=Timestamp(0)),
        SessionInfo(id="a-b", last_updated_at=Timestamp.parse("2026-08-27T12:00:00Z")),
        SessionInfo(id="z", last_updated_at=Timestamp.parse("2026-08-27T13:00:00Z")),
    ]
    for info in want:
        await store.create(info.id)
        nanoseconds = info.last_updated_at.unix_ns
        os.utime(session_path(tmp_path, info.id), ns=(nanoseconds, nanoseconds))
    for name in ("README", ".session-pending.tmp", "old.session.json"):
        (tmp_path / name).write_bytes(b"unrelated")
    (tmp_path / "nested.session.jsonl").mkdir(mode=0o700)
    os.symlink("a.session.jsonl", tmp_path / "link.session.jsonl")

    assert await Store(tmp_path).list_sessions() == want


async def test_list_sessions_tracks_history_and_operation_writes(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await store.create("session-1")
    path = session_path(tmp_path, "session-1")
    future = Timestamp.parse("2100-01-01T00:00:00Z")
    writes: list[tuple[str, Callable[[], Awaitable[None]]]] = [
        ("turn", lambda: store.append_turn("session-1", Turn(ID="turn-1", Type=TurnType.REGULAR))),
        (
            "tool status",
            lambda: store.append_tool_call_status(
                "session-1", waiting_status("turn-1", "call-1", ready_operation("operation-1"))
            ),
        ),
        (
            "operation",
            lambda: store.save_operation(
                "session-1", ready_operation("operation-1", Status.COMPLETED)
            ),
        ),
    ]
    for name, write in writes:
        os.utime(path, ns=(future.unix_ns, future.unix_ns))
        await write()
        listed = await store.list_sessions()
        modified = Timestamp(path.stat().st_mtime_ns)
        assert len(listed) == 1, name
        assert listed[0].id == "session-1", name
        assert listed[0].last_updated_at == modified, name
        assert listed[0].last_updated_at < future, name


async def test_list_sessions_includes_forks_and_replacements(tmp_path: Path) -> None:
    store = Store(tmp_path)
    await create_session_with_turn(store, "parent", "turn-1", "")
    await store.fork("child", "parent", "turn-1")
    await store.create("parent")
    listed = await Store(tmp_path).list_sessions()
    assert listed == [
        SessionInfo(id=id, last_updated_at=Timestamp(session_path(tmp_path, id).stat().st_mtime_ns))
        for id in ("child", "parent")
    ]


async def test_list_sessions_does_not_read_files(tmp_path: Path) -> None:
    store = Store(tmp_path)
    path = session_path(tmp_path, "session-1")
    path.write_bytes(b"")
    path.chmod(0)
    try:
        listed = await store.list_sessions()
    finally:
        path.chmod(0o600)
    assert [info.id for info in listed] == ["session-1"]


async def test_list_sessions_ignores_invalid_filenames(tmp_path: Path) -> None:
    store = Store(tmp_path)
    for name in (
        ".session.jsonl",
        "invalid_id.session.jsonl",
        "with space.session.jsonl",
        "unicode-ø.session.jsonl",
    ):
        (tmp_path / name).write_bytes(b"")
    assert await store.list_sessions() == []
    await store.create("session-1")
    assert [info.id for info in await store.list_sessions()] == ["session-1"]


async def test_list_sessions_reports_cancellation_and_directory_errors(tmp_path: Path) -> None:
    directory = tmp_path / "sessions"
    store = Store(directory)
    task = asyncio.create_task(store.list_sessions())
    task.cancel("stop listing")
    with pytest.raises(asyncio.CancelledError, match="stop listing"):
        await task
    directory.rmdir()
    with pytest.raises(FileNotFoundError):
        await store.list_sessions()


def test_store_cleans_directory_like_go(tmp_path: Path) -> None:
    store = Store("/" + str(tmp_path) + "//sessions/./")
    assert store.directory == str(tmp_path / "sessions")
    assert store.session_path("s") == str(tmp_path / "sessions" / "s.session.jsonl")
