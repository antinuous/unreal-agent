import asyncio
import contextlib
from collections.abc import Callable
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent._go import go_quote
from unreal_agent.contextbuilder import new_builder
from unreal_agent.coordinator import Coordinator, CoordinatorError
from unreal_agent.coordinator.loop import (
    HISTORY_PAGE_SIZE,
    ToolCallKey,
    ToolCallState,
    _closed_input_error,
)
from unreal_agent.inbox import Inbox, Input, InputKind
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import (
    ItemType,
    Message,
    Request,
    Response,
    Role,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.operation import TYPE_SHELL, Operation, Status, UnsupportedError
from unreal_agent.session import Session, Turn, TurnType
from unreal_agent.sessionstore import (
    BEFORE_FIRST,
    Fork,
    Item,
    ItemKind,
    ModelResponse,
    Page,
    ResumeState,
    Snapshot,
    ToolCallStatus,
)
from unreal_agent.tool import (
    BASH_NAME,
    VIEW_IMAGE_NAME,
    CallStatus,
    Registry,
    StaticTranslators,
)

from tests.coordinator._helpers import (
    FailingBuilder,
    FailingResultTranslator,
    FakeAdapter,
    FakeOperationManager,
    FakeStore,
    ItemRequest,
    OperationStatusTranslator,
    SubmittingTranslator,
    TerminalResultTranslator,
    TestTranslator,
    as_builder,
    assert_stop_result,
    caused_by,
    deps,
    empty_fake_store,
    external_event,
    fake_time,
    new_stop_test_run,
    new_test_coordinator,
    new_test_inbox,
    stored_item,
    submit_test_input,
    text_response,
    value_spec,
    wait,
    with_preamble,
)


def loop_state_store(*items: Item, operations: list[Operation] | None = None) -> FakeStore:
    return FakeStore(
        resume=ResumeState(
            snapshot=Snapshot(session=Session(ID="session-1")),
            operations=list(operations or []),
        ),
        history=list(items),
    )


def loop_state_user_message(text: str) -> LLMItem:
    return LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=text))


def loop_state_text_result(call_id: str, value: str) -> LLMItem:
    return LLMItem(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(
            CallID=call_id, Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]
        ),
    )


async def test_coordinator_restores_session() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    input = external_event("input-1", "hello")
    turn = Turn(ID="turn-1", Type=TurnType.REGULAR)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    response = Response(
        ID="response-1",
        Output=[
            LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="working")),
            LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        ],
    )
    status = ToolCallStatus(
        TurnID=turn.ID, CallID=call.CallID, Status=CallStatus(Error="invalid arguments")
    )
    resumed_operation = Operation(
        ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING
    )
    store = loop_state_store(
        stored_item(1, ItemKind.INPUT, input),
        stored_item(2, ItemKind.TURN, turn),
        stored_item(3, ItemKind.MODEL_RESPONSE, ModelResponse(TurnID=turn.ID, Response=response)),
        stored_item(4, ItemKind.TOOL_CALL_STATUS, status),
        operations=[resumed_operation],
    )
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )

    await current._restore()

    state = current._state
    assert state.current_turn_id == turn.ID, f"restored state = {state!r}"
    assert state.current_turn_type == TurnType.REGULAR, f"restored state = {state!r}"
    assert state.operations.get(resumed_operation.ID) == resumed_operation, (
        f"restored state = {state!r}"
    )

    built = builder.build()
    want_input = with_preamble(
        loop_state_user_message("hello"),
        (response.Output or [])[0],
        (response.Output or [])[1],
        loop_state_text_result(call.CallID, "error:invalid arguments"),
    )
    assert built.request.Input == want_input, (
        f"built request = {built.request!r}, want input {want_input!r}"
    )
    assert store.item_requests == [ItemRequest(0, HISTORY_PAGE_SIZE)], (
        f"item requests = {store.item_requests!r}"
    )


async def test_coordinator_restores_ordinary_response_during_compaction() -> None:
    run = new_stop_test_run(0)
    response = text_response("Ordinary response")
    assert response.Output is not None
    response.Output.append(
        LLMItem(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="old-call", Name="unknown"))
    )
    run.store.history = [
        stored_item(1, ItemKind.INPUT, external_event("input", "hello")),
        stored_item(2, ItemKind.TURN, Turn(ID="ordinary", Type=TurnType.REGULAR)),
        stored_item(
            3,
            ItemKind.TURN,
            Turn(ID="compact", PreviousTurnID="ordinary", Type=TurnType.COMPACTION),
        ),
        stored_item(
            4, ItemKind.MODEL_RESPONSE, ModelResponse(TurnID="ordinary", Response=response)
        ),
    ]
    await run.current._load_history()
    built = run.dependencies.context_builder.build()
    want = [*with_preamble(loop_state_user_message("hello")), *(response.Output or [])]
    message = "earlier ordinary response was lost or treated as a compaction response"
    assert built.request.Input == want, message
    assert run.current._state.current_turn_type == TurnType.COMPACTION, message
    assert run.current._pending_inputs() == 1, message
    assert len(run.current._state.tool_calls) == 1, message


async def test_coordinator_restores_paginated_fork_history() -> None:
    parent_input = external_event("parent-input", "parent")
    items = [stored_item(1, ItemKind.INPUT, parent_input)]
    for sequence in range(2, HISTORY_PAGE_SIZE + 1):
        items.append(
            stored_item(sequence, ItemKind.TURN, Turn(ID=f"turn-{sequence}", Type=TurnType.REGULAR))
        )
    items.append(
        stored_item(
            HISTORY_PAGE_SIZE + 1,
            ItemKind.FORK,
            Fork(ParentID="parent", PreviousTurnID="turn-256"),
        )
    )
    items.append(
        stored_item(HISTORY_PAGE_SIZE + 2, ItemKind.INPUT, external_event("child-input", "child"))
    )
    store = loop_state_store(*items)
    builder = new_builder()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        builder,
        Registry(StaticTranslators()),
    )

    await current._restore()

    assert store.item_requests == [
        ItemRequest(0, HISTORY_PAGE_SIZE),
        ItemRequest(HISTORY_PAGE_SIZE, HISTORY_PAGE_SIZE),
    ], f"item requests = {store.item_requests!r}"
    assert current._state.current_turn_id == "turn-256", (
        f"current turn ID = {current._state.current_turn_id!r}, want 'turn-256'"
    )
    built = builder.build()
    want = with_preamble(loop_state_user_message("parent"), loop_state_user_message("child"))
    assert built.request.Input == want, f"replayed input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_returns_session_history_error() -> None:
    store = empty_fake_store()
    store.items_error = RuntimeError("disk unavailable")
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )

    with pytest.raises(CoordinatorError) as caught:
        await current.run()
    assert str(caught.value) == "load session history after 0: disk unavailable", (
        f"restore error = {caught.value!r}"
    )


async def test_coordinator_rejects_session_history_without_progress() -> None:
    store = empty_fake_store()
    store.items_page = Page(more=True, next_after=BEFORE_FIRST)
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )

    with pytest.raises(CoordinatorError) as caught:
        await current._restore()
    assert str(caught.value) == "load session history did not advance after 0", (
        f"restore error = {caught.value!r}"
    )


async def test_coordinator_keeps_unreplayable_tool_status_in_local_state() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    turn = Turn(ID="turn-1", Type=TurnType.REGULAR)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    status = ToolCallStatus(
        TurnID=turn.ID,
        CallID=call.CallID,
        Status=CallStatus(WaitingFor=["terminal-operation"]),
    )
    store = loop_state_store(
        stored_item(1, ItemKind.TURN, turn),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID=turn.ID,
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(3, ItemKind.TOOL_CALL_STATUS, status),
    )
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )

    await current._restore()
    call_state = current._state.tool_calls.get(ToolCallKey(status.TurnID, status.CallID))
    assert call_state is not None, f"tool call state = {call_state!r}"
    assert call_state.status is not None, f"tool call state = {call_state!r}"
    assert call_state.status == status.Status, f"tool call state = {call_state!r}"
    built = builder.build()
    assert built.request.Input == with_preamble(LLMItem(Type=ItemType.TOOL_CALL, Data=call)), (
        f"replayed input = {built.request.Input!r}"
    )


async def test_coordinator_restores_completed_tool_call_from_status_snapshots() -> None:
    registry = Registry(StaticTranslators(view_image=OperationStatusTranslator()), VIEW_IMAGE_NAME)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    initial = Operation(ID="operation-1", Type="test", Version=1, Status=Status.READY)
    completed = replace(initial, Status=Status.COMPLETED)
    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=[initial.ID])
    )
    initial_status = replace(status, Operations=[initial])
    completed_status = replace(status, Operations=[completed])
    store = loop_state_store(
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(3, ItemKind.TOOL_CALL_STATUS, initial_status),
        stored_item(4, ItemKind.TOOL_CALL_STATUS, completed_status),
    )
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )

    await current._restore()
    assert ToolCallKey("turn-1", call.CallID) not in current._state.tool_calls, (
        "completed tool call remains in local state"
    )
    assert current._state.operations.get(completed.ID) == completed, (
        f"restored operation = {current._state.operations.get(completed.ID)!r}, want {completed!r}"
    )
    built = builder.build()
    want = with_preamble(
        LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        loop_state_text_result(call.CallID, "completed"),
    )
    assert built.request.Input == want, f"replayed input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_overlays_resumed_operations_after_history_snapshots() -> None:
    registry = Registry(StaticTranslators(view_image=OperationStatusTranslator()), VIEW_IMAGE_NAME)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    first = Operation(ID="operation-1", Type="test", Version=1, Status=Status.READY)
    second = Operation(ID="operation-2", Type="test", Version=1, Status=Status.READY)
    resumed_second = replace(second, Status=Status.AWAITING)
    status = ToolCallStatus(
        TurnID="turn-1",
        CallID=call.CallID,
        Status=CallStatus(WaitingFor=[first.ID, second.ID]),
        Operations=[replace(first), replace(second)],
    )
    store = loop_state_store(
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(3, ItemKind.TOOL_CALL_STATUS, status),
        operations=[resumed_second],
    )
    operations = FakeOperationManager()
    current = new_test_coordinator(store, new_test_inbox(), operations, new_builder(), registry)

    await current._restore()
    restored = current._state.operations
    assert restored.get(first.ID) == first, f"restored operations = {restored!r}"
    assert restored.get(second.ID) == resumed_second, f"restored operations = {restored!r}"
    await current._dispatch_operations_to_manager()
    want = {first.ID: first, second.ID: resumed_second}
    got = {value.ID: value for value in operations.adds}
    assert got == want, f"dispatched operations = {got!r}, want {want!r}"


def test_coordinator_tracks_tool_calls() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    first = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    second = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments='{"second":true}')
    for response in (
        ModelResponse(
            TurnID="turn-1",
            Response=Response(
                Output=[
                    LLMItem(
                        Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="working")
                    ),
                    LLMItem(Type=ItemType.TOOL_CALL, Data=first),
                ]
            ),
        ),
        ModelResponse(
            TurnID="turn-2",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=second)]),
        ),
    ):
        current._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=response))

    want = {
        ToolCallKey("turn-1", first.CallID): ToolCallState(tool_call=first, operations=set()),
        ToolCallKey("turn-2", second.CallID): ToolCallState(tool_call=second, operations=set()),
    }
    assert current._state.tool_calls == want, (
        f"tool calls = {current._state.tool_calls!r}, want {want!r}"
    )

    current._add_item_to_local_state(
        Item(
            Kind=ItemKind.TOOL_CALL_STATUS,
            Data=ToolCallStatus(
                TurnID="turn-1",
                CallID=first.CallID,
                Status=CallStatus(Error="invalid arguments"),
            ),
        )
    )
    want = {
        ToolCallKey("turn-2", second.CallID): ToolCallState(tool_call=second, operations=set()),
    }
    assert current._state.tool_calls == want, (
        f"tool calls = {current._state.tool_calls!r}, want {want!r}"
    )

    waiting_for = ["operation-1", "operation-2"]
    waiting_status = CallStatus(WaitingFor=waiting_for)
    current._add_item_to_local_state(
        Item(
            Kind=ItemKind.TOOL_CALL_STATUS,
            Data=ToolCallStatus(TurnID="turn-2", CallID=second.CallID, Status=waiting_status),
        )
    )
    want = {
        ToolCallKey("turn-2", second.CallID): ToolCallState(
            tool_call=second,
            status=waiting_status,
            operations={waiting_for[0], waiting_for[1]},
        ),
    }
    assert current._state.tool_calls == want, (
        f"tool calls = {current._state.tool_calls!r}, want {want!r}"
    )


def test_coordinator_tool_call_operations_are_terminal() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    assert current._tool_call_operations_are_terminal("missing-turn", "missing-call"), (
        "tool call without operations was not recognized as terminal"
    )

    key = ToolCallKey("turn-1", "call-1")
    current._state.tool_calls[key] = ToolCallState(
        tool_call=ToolCall(), operations={"completed", "failed", "canceled"}
    )
    for id, status in {
        "completed": Status.COMPLETED,
        "failed": Status.FAILED,
        "canceled": Status.CANCELED,
    }.items():
        current._add_operation_to_local_state(Operation(ID=id, Status=status))

    assert current._tool_call_operations_are_terminal(key.turn_id, key.call_id), (
        "terminal operations were not recognized"
    )

    current._add_operation_to_local_state(Operation(ID="failed", Status=Status.AWAITING))
    assert not current._tool_call_operations_are_terminal(key.turn_id, key.call_id), (
        "non-terminal operation was recognized as terminal"
    )


def test_coordinator_adds_tool_result_from_tracked_tool_call() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    builder = new_builder()
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), builder, registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )

    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(Error="invalid arguments")
    )
    current._add_tool_result_to_local_state(status)

    built = builder.build()
    want = with_preamble(loop_state_text_result(call.CallID, "error:invalid arguments"))
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"


def test_coordinator_accepts_orphaned_tool_status() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    status = ToolCallStatus(
        TurnID="turn-1", CallID="missing-call", Status=CallStatus(WaitingFor=["operation-1"])
    )

    item = current._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))
    assert item.Data == status, f"returned item = {item.Data!r}, want {status!r}"


def test_coordinator_returns_tool_result_translation_error() -> None:
    registry = Registry(
        StaticTranslators(view_image=FailingResultTranslator(RuntimeError("translation failed"))),
        VIEW_IMAGE_NAME,
    )
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )

    with pytest.raises(CoordinatorError) as caught:
        current._add_item_to_local_state(
            Item(
                Kind=ItemKind.TOOL_CALL_STATUS,
                Data=ToolCallStatus(TurnID="turn-1", CallID=call.CallID),
            )
        )
    assert str(caught.value) == 'add tool call "call-1" result to context: translation failed', (
        f"add status error = {caught.value!r}"
    )


def test_coordinator_skips_tool_result_without_available_translator() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    call = ToolCall(CallID="call-1", Name="unavailable-tool")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )

    current._add_tool_result_to_local_state(ToolCallStatus(TurnID="turn-1", CallID=call.CallID))
    assert ToolCallKey("turn-1", call.CallID) in current._state.tool_calls, (
        "tool call with unavailable translator was removed"
    )


def test_coordinator_skips_tool_result_with_untracked_operation() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    current = new_test_coordinator(
        empty_fake_store(), new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME)
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )
    current._add_operation_to_local_state(Operation(ID="operation-1", Status=Status.COMPLETED))

    current._add_tool_result_to_local_state(
        ToolCallStatus(
            TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=["operation-1"])
        )
    )
    assert ToolCallKey("turn-1", call.CallID) in current._state.tool_calls, (
        "tool call with an untracked operation was removed"
    )


async def test_coordinator_keeps_control_input_without_context_projection() -> None:
    input = Input(
        ID="control-input", Kind=InputKind.CONTROL, Payload=gojson.Raw(b'{"Mode":"when_idle"}')
    )
    store = loop_state_store(stored_item(1, ItemKind.INPUT, input))
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, Registry(StaticTranslators())
    )

    await current._restore()
    built = builder.build()
    assert built.request.Input == with_preamble(), f"replayed input = {built.request.Input!r}"


@pytest.mark.parametrize(
    ("item", "want"),
    [
        pytest.param(stored_item(1, ItemKind.FORK, Turn()), "want sessionstore.Fork", id="fork"),
        pytest.param(stored_item(1, ItemKind.INPUT, Turn()), "want inbox.Input", id="input"),
        pytest.param(
            stored_item(1, ItemKind.INPUT, Input(ID="input-1", Kind=cast(InputKind, "unknown"))),
            "invalid input",
            id="invalid input",
        ),
        pytest.param(stored_item(1, ItemKind.TURN, Input()), "want session.Turn", id="turn"),
        pytest.param(
            stored_item(1, ItemKind.MODEL_RESPONSE, Turn()),
            "want sessionstore.ModelResponse",
            id="response",
        ),
        pytest.param(
            stored_item(1, ItemKind.TOOL_CALL_STATUS, Turn()),
            "want sessionstore.ToolCallStatus",
            id="status",
        ),
        pytest.param(stored_item(1, "unknown", None), 'unsupported item kind "unknown"', id="kind"),
    ],
)
async def test_coordinator_rejects_invalid_session_item_data(item: Item, want: str) -> None:
    store = loop_state_store(item)
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    with pytest.raises(CoordinatorError) as caught:
        await current._restore()
    assert want in str(caught.value), f"error = {caught.value!r}, want substring {want!r}"


@fake_time
async def test_coordinator_run_returns_context_cancellation() -> None:
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    task = asyncio.create_task(current.run())
    await wait()
    task.cancel()
    await asyncio.wait((task,))
    error: BaseException | None = asyncio.CancelledError() if task.cancelled() else task.exception()
    assert isinstance(error, asyncio.CancelledError), (
        f"Run error = {error!r}, want context cancellation"
    )


async def test_closed_input_error_prefers_context_cancellation() -> None:
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    try:
        error = _closed_input_error("input")
    finally:
        task.uncancel()
    assert isinstance(error, asyncio.CancelledError), (
        f"closed input error = {error!r}, want context cancellation"
    )


async def loop_run_receive[T](values: asyncio.Queue[T]) -> T:
    try:
        return await asyncio.wait_for(values.get(), 1)
    except TimeoutError:
        raise AssertionError("timed out waiting for value") from None


def loop_run_start(current: Coordinator) -> asyncio.Task[None]:
    return asyncio.create_task(current.run())


async def loop_run_error(task: asyncio.Task[None]) -> BaseException | None:
    done, _ = await asyncio.wait((task,), timeout=1)
    assert done, "timed out waiting for value"
    if task.cancelled():
        return asyncio.CancelledError()
    return task.exception()


def loop_run_closed_inbox() -> Inbox:
    inputs = new_test_inbox()
    inputs.close()
    return inputs


async def loop_run_returned(current: Coordinator) -> BaseException | None:
    try:
        await current.run()
    except Exception as error:
        return error
    return None


async def test_coordinator_run_calls_model_after_persisted_external_input() -> None:
    store = empty_fake_store()
    store.history = [stored_item(1, ItemKind.TURN, Turn(ID="previous-turn", Type=TurnType.REGULAR))]
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    started: asyncio.Queue[Request] = asyncio.Queue()
    request_canceled: asyncio.Queue[BaseException] = asyncio.Queue()
    order: list[str] = []
    store.on_append_input = lambda _: order.append("input")
    store.on_append_turn = lambda _: order.append("turn")

    async def respond(request: Request) -> Response:
        order.append("respond")
        started.put_nowait(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            request_canceled.put_nowait(error)
            raise
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    builder = new_builder()
    current = new_test_coordinator(
        store, inputs, operations, builder, Registry(StaticTranslators()), adapter
    )
    task = loop_run_start(current)

    event = external_event("input-1", "hello")
    await submit_test_input(inputs, event)
    request = await loop_run_receive(started)
    built = builder.build()
    want = with_preamble(LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello")))
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"
    assert request == built.request, f"model request = {request!r}, want {built.request!r}"
    assert store.appended_inputs == [event], (
        f"appended inputs = {store.appended_inputs!r}, want {[event]!r}"
    )
    assert len(store.appended_turns) == 1, f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[0].ID != "", f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[0].PreviousTurnID == "previous-turn", (
        f"appended turns = {store.appended_turns!r}"
    )
    assert order == ["input", "turn", "respond"], f"effect order = {order}"
    assert len(operations.adds) == 0, (
        f"unexpected operation effects: adds={operations.adds} cancels={operations.cancels}"
    )
    assert len(operations.cancels) == 0, (
        f"unexpected operation effects: adds={operations.adds} cancels={operations.cancels}"
    )

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    canceled = await loop_run_receive(request_canceled)
    assert isinstance(canceled, asyncio.CancelledError), (
        f"model request error = {canceled!r}, want context cancellation"
    )


async def test_coordinator_run_slurps_queued_inputs_before_calling_model() -> None:
    store = empty_fake_store()
    inputs = new_test_inbox()
    first = external_event("input-1", "first")
    second = external_event("input-2", "second")
    await submit_test_input(inputs, first)
    await submit_test_input(inputs, second)
    started: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    request = await loop_run_receive(started)
    want = with_preamble(
        LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="first")),
        LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="second")),
    )
    assert request.Input == want, f"model input = {request.Input!r}, want {want!r}"
    assert store.appended_inputs == [first, second], f"appended inputs = {store.appended_inputs!r}"
    assert len(adapter.requests) == 1, f"model requests = {len(adapter.requests)}, want 1"

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )


async def test_coordinator_run_does_not_call_model_when_request_build_fails() -> None:
    store = empty_fake_store()
    inputs = new_test_inbox()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        as_builder(FailingBuilder(new_builder(), RuntimeError("context unavailable"))),
        Registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    event = external_event("input-1", "hello")
    await submit_test_input(inputs, event)
    err = await loop_run_error(task)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == "build model request: context unavailable", f"Run error = {err!r}"
    assert store.appended_inputs == [event], (
        f"effects after build failure: inputs={store.appended_inputs} "
        f"turns={store.appended_turns} requests={adapter.requests}"
    )
    assert len(store.appended_turns) == 0, (
        f"effects after build failure: inputs={store.appended_inputs} "
        f"turns={store.appended_turns} requests={adapter.requests}"
    )
    assert len(adapter.requests) == 0, (
        f"effects after build failure: inputs={store.appended_inputs} "
        f"turns={store.appended_turns} requests={adapter.requests}"
    )


async def test_coordinator_run_does_not_call_model_when_turn_store_fails() -> None:
    store = empty_fake_store()
    store.append_turn_error = RuntimeError("disk unavailable")
    inputs = new_test_inbox()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    err = await loop_run_error(task)
    assert err is not None, f"Run error = {err!r}"
    assert "disk unavailable" in str(err), f"Run error = {err!r}"
    assert len(store.appended_turns) == 1, (
        f"effects after turn store failure: turns={store.appended_turns} "
        f"requests={adapter.requests}"
    )
    assert len(adapter.requests) == 0, (
        f"effects after turn store failure: turns={store.appended_turns} "
        f"requests={adapter.requests}"
    )


async def test_coordinator_run_persists_model_response_for_originating_turn() -> None:
    store = empty_fake_store()
    response_stored: asyncio.Queue[ModelResponse] = asyncio.Queue()
    store.on_append_model_response = response_stored.put_nowait
    response = Response(
        ID="response-1",
        Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="done"))],
    )

    async def respond(request: Request) -> Response:
        return response

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    stored = await loop_run_receive(response_stored)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_turns) == 1, (
        f"stored response = {stored!r}, turns = {store.appended_turns!r}"
    )
    assert stored.TurnID == store.appended_turns[0].ID, (
        f"stored response = {stored!r}, turns = {store.appended_turns!r}"
    )
    assert stored.Response == response, (
        f"stored response = {stored!r}, turns = {store.appended_turns!r}"
    )
    assert len(adapter.requests) == 1, (
        f"message response effects: requests={adapter.requests} "
        f"responses={store.appended_responses} turns={store.appended_turns}"
    )
    assert len(store.appended_responses) == 1, (
        f"message response effects: requests={adapter.requests} "
        f"responses={store.appended_responses} turns={store.appended_turns}"
    )
    assert len(store.appended_turns) == 1, (
        f"message response effects: requests={adapter.requests} "
        f"responses={store.appended_responses} turns={store.appended_turns}"
    )


async def test_coordinator_run_persists_tool_call_before_dispatch() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    store = empty_fake_store()
    dispatched: asyncio.Queue[Operation] = asyncio.Queue()
    operations = FakeOperationManager()

    def add_error(value: Operation) -> Exception | None:
        if len(store.appended_responses) != 1 or len(store.appended_statuses) != 1:
            return RuntimeError("operation dispatched before response and status were stored")
        dispatched.put_nowait(value)
        return None

    operations.add_error = add_error
    response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )

    async def respond(request: Request) -> Response:
        return response

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        operations,
        new_builder(),
        Registry(StaticTranslators(bash=translator), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "run"))
    operation_value = await loop_run_receive(dispatched)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 1, (
        f"appended statuses = {store.appended_statuses!r}, operation = {operation_value!r}"
    )
    assert store.appended_statuses[0].Operations == [operation_value], (
        f"appended statuses = {store.appended_statuses!r}, operation = {operation_value!r}"
    )
    assert store.appended_statuses[0].Status.WaitingFor == [operation_value.ID], (
        f"appended statuses = {store.appended_statuses!r}, operation = {operation_value!r}"
    )
    assert len(adapter.requests) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )
    assert len(store.appended_turns) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )


async def test_coordinator_run_starts_corrective_turn_for_validation_error() -> None:
    first_response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )
    started: asyncio.Queue[Request] = asyncio.Queue()
    call_count = 0

    async def respond(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        started.put_nowait(request)
        if call_count == 1:
            return first_response
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators(), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "run"))
    first_request = await loop_run_receive(started)
    second_request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 1, f"appended statuses = {store.appended_statuses!r}"
    assert store.appended_statuses[0].Status.Error != "", (
        f"appended statuses = {store.appended_statuses!r}"
    )
    assert len(store.appended_turns) == 2, f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[1].PreviousTurnID == store.appended_turns[0].ID, (
        f"appended turns = {store.appended_turns!r}"
    )
    assert len(first_request.Input) == 2, f"model requests = {[first_request, second_request]!r}"
    assert len(second_request.Input) == 4, f"model requests = {[first_request, second_request]!r}"
    assert second_request.Input[3].Type == ItemType.TOOL_RESULT, (
        f"model requests = {[first_request, second_request]!r}"
    )


async def test_coordinator_run_starts_continuation_turn_for_completed_tool_call() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    first_response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )
    started: asyncio.Queue[Request] = asyncio.Queue()
    call_count = 0

    async def respond(request: Request) -> Response:
        nonlocal call_count
        call_count += 1
        started.put_nowait(request)
        if call_count == 1:
            return first_response
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    operations = FakeOperationManager()
    dispatched: asyncio.Queue[Operation] = asyncio.Queue()

    def add_error(value: Operation) -> Exception | None:
        dispatched.put_nowait(value)
        return None

    operations.add_error = add_error
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        operations,
        new_builder(),
        Registry(StaticTranslators(bash=translator), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "run"))
    await loop_run_receive(started)
    operation_value = replace(await loop_run_receive(dispatched), Status=Status.COMPLETED)
    operations.send(operation_value)
    continuation_request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert store.saved_operations == [operation_value], (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    assert len(store.appended_statuses) == 2, (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    assert store.appended_statuses[1].Operations == [operation_value], (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    options = list(adapter.request_options)
    assert len(options) == 2, (
        f"request options = {options!r}, want session-1 cache keys for both turns"
    )
    assert options[0].CacheKey == "session-1", (
        f"request options = {options!r}, want session-1 cache keys for both turns"
    )
    assert options[1].CacheKey == "session-1", (
        f"request options = {options!r}, want session-1 cache keys for both turns"
    )
    assert len(continuation_request.Input) == 4, f"continuation request = {continuation_request!r}"
    assert continuation_request.Input[3].Type == ItemType.TOOL_RESULT, (
        f"continuation request = {continuation_request!r}"
    )
    assert len(adapter.requests) == 2, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )
    assert len(store.appended_turns) == 2, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )


async def test_coordinator_run_batches_completed_tool_calls_into_one_turn() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    operation_value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    calls = [
        ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}"),
        ToolCall(CallID="call-2", Name=VIEW_IMAGE_NAME, Arguments="{}"),
    ]
    status = CallStatus(WaitingFor=[operation_value.ID])
    store = empty_fake_store()
    store.resume.operations = [replace(operation_value)]
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    Output=[
                        LLMItem(Type=ItemType.TOOL_CALL, Data=calls[0]),
                        LLMItem(Type=ItemType.TOOL_CALL, Data=calls[1]),
                    ]
                ),
            ),
        ),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(
                TurnID="turn-1",
                CallID=calls[0].CallID,
                Status=status,
                Operations=[replace(operation_value)],
            ),
        ),
        stored_item(
            4,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(
                TurnID="turn-1",
                CallID=calls[1].CallID,
                Status=status,
                Operations=[replace(operation_value)],
            ),
        ),
    ]
    started: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry, adapter)
    task = loop_run_start(current)

    operations.send(replace(operation_value, Status=Status.COMPLETED))
    request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 2, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(store.appended_turns) == 1, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert store.appended_turns[0].PreviousTurnID == "turn-1", (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(adapter.requests) == 1, f"requests = {adapter.requests!r}"
    assert len(request.Input) == 5, f"requests = {adapter.requests!r}"


async def test_coordinator_run_steers_active_model_request() -> None:
    started: asyncio.Queue[Request] = asyncio.Queue()
    first_canceled: asyncio.Queue[None] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if len(request.Input) == 2:
                first_canceled.put_nowait(None)
            raise
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "first"))
    first_request = await loop_run_receive(started)
    await submit_test_input(inputs, external_event("input-2", "second"))
    second_request = await loop_run_receive(started)
    await loop_run_receive(first_canceled)
    done, _ = await asyncio.wait((task,), timeout=0.02)
    assert not done, f"Run stopped after superseded cancellation: {await loop_run_error(task)!r}"
    assert len(first_request.Input) == 2, f"steered requests = {[first_request, second_request]!r}"
    assert len(second_request.Input) == 3, f"steered requests = {[first_request, second_request]!r}"
    assert len(store.appended_turns) == 2, f"appended turns = {store.appended_turns!r}"
    assert store.appended_turns[1].PreviousTurnID == store.appended_turns[0].ID, (
        f"appended turns = {store.appended_turns!r}"
    )
    assert len(store.appended_responses) == 0, f"appended responses = {store.appended_responses!r}"

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )


async def test_coordinator_run_handles_operation_update_while_model_is_running() -> None:
    started: asyncio.Queue[None] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(None)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    initial = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    store = empty_fake_store()
    store.resume.operations = [replace(initial)]
    stored_update: asyncio.Queue[Operation] = asyncio.Queue()
    store.on_save_operation = stored_update.put_nowait
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), Registry(StaticTranslators()), adapter
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    await loop_run_receive(started)
    update = replace(initial, Status=Status.AWAITING)
    operations.send(update)
    stored = await loop_run_receive(stored_update)
    assert stored == update, f"stored operation = {stored!r}, want {update!r}"

    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(adapter.requests) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )
    assert len(store.appended_turns) == 1, (
        f"requests = {adapter.requests}, turns = {store.appended_turns}"
    )


async def test_coordinator_run_drops_successful_response_from_superseded_turn() -> None:
    spec = value_spec(b'{"value":1}')
    translator = SubmittingTranslator([spec])
    first_response = Response(
        ID="response-1",
        Output=[
            LLMItem(
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="stale-call", Name=BASH_NAME, Arguments="{}"),
            )
        ],
    )
    second_response = Response(
        ID="response-2",
        Output=[
            LLMItem(
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="current answer"),
            )
        ],
    )
    started: asyncio.Queue[Request] = asyncio.Queue()
    first_returned: asyncio.Queue[None] = asyncio.Queue()
    release_second = asyncio.Event()
    response_stored: asyncio.Queue[ModelResponse] = asyncio.Queue()
    store = empty_fake_store()
    store.on_append_model_response = response_stored.put_nowait

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        if len(request.Input) == 2:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
            first_returned.put_nowait(None)
            return first_response
        await release_second.wait()
        return second_response

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators(bash=translator), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)
    try:
        await submit_test_input(inputs, external_event("input-1", "first"))
        await loop_run_receive(started)
        await submit_test_input(inputs, external_event("input-2", "second"))
        await loop_run_receive(started)
        await loop_run_receive(first_returned)
        await asyncio.sleep(0.02)
        release_second.set()
        stored = await loop_run_receive(response_stored)
        assert len(store.appended_turns) == 2, (
            f"stored response = {stored!r}, turns = {store.appended_turns!r}"
        )
        assert stored.TurnID == store.appended_turns[1].ID, (
            f"stored response = {stored!r}, turns = {store.appended_turns!r}"
        )
        assert stored.Response == second_response, (
            f"stored response = {stored!r}, turns = {store.appended_turns!r}"
        )
        try:
            unexpected = await asyncio.wait_for(response_stored.get(), 0.02)
        except TimeoutError:
            pass
        else:
            raise AssertionError(f"stored superseded response = {unexpected!r}")

        task.cancel()
        err = await loop_run_error(task)
        assert isinstance(err, asyncio.CancelledError), (
            f"Run error = {err!r}, want context cancellation"
        )
        assert store.appended_responses == [stored], (
            f"superseded effects: responses={store.appended_responses!r} "
            f"translations={translator.calls!r} statuses={store.appended_statuses!r}"
        )
        assert len(translator.calls) == 0, (
            f"superseded effects: responses={store.appended_responses!r} "
            f"translations={translator.calls!r} statuses={store.appended_statuses!r}"
        )
        assert len(store.appended_statuses) == 0, (
            f"superseded effects: responses={store.appended_responses!r} "
            f"translations={translator.calls!r} statuses={store.appended_statuses!r}"
        )
    finally:
        task.cancel()


async def test_coordinator_run_returns_current_model_error() -> None:
    provider_error = RuntimeError("provider unavailable")

    async def respond(request: Request) -> Response:
        raise provider_error

    adapter = FakeAdapter(respond)
    store = empty_fake_store()
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
        adapter,
    )
    task = loop_run_start(current)

    await submit_test_input(inputs, external_event("input-1", "hello"))
    err = await loop_run_error(task)
    assert caused_by(err, provider_error), f"Run error = {err!r}"
    assert "call model for turn" in str(err), f"Run error = {err!r}"
    assert len(store.appended_turns) == 1, (
        f"turns = {store.appended_turns!r}, responses = {store.appended_responses!r}"
    )
    assert len(store.appended_responses) == 0, (
        f"turns = {store.appended_turns!r}, responses = {store.appended_responses!r}"
    )


async def test_coordinator_handles_model_response_before_scheduling_tool_calls() -> None:
    spec = value_spec(b'{"value":1}')
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            ID="response-1",
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                )
            ],
        ),
    )
    store = empty_fake_store()
    stored_before_translation = False

    def on_translate() -> None:
        nonlocal stored_before_translation
        stored_before_translation = store.appended_responses == [response]

    translator = SubmittingTranslator([spec], on_translate)
    operations = FakeOperationManager()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        operations,
        new_builder(),
        Registry(StaticTranslators(bash=translator), BASH_NAME),
    )

    await current._handle_model_response(response)
    assert stored_before_translation, (
        "tool call was translated before its model response was stored"
    )
    assert len(store.appended_statuses) == 1, f"appended statuses = {store.appended_statuses!r}"
    assert len(store.appended_statuses[0].Operations) == 1, (
        f"appended statuses = {store.appended_statuses!r}"
    )
    assert len(operations.adds) == 0, f"handler dispatched operations = {operations.adds!r}"


async def test_coordinator_does_not_schedule_tool_calls_when_model_response_store_fails() -> None:
    store = empty_fake_store()
    store.append_model_response_error = RuntimeError("disk unavailable")
    translator = SubmittingTranslator()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators(bash=translator), BASH_NAME),
    )
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                )
            ]
        ),
    )

    with pytest.raises(CoordinatorError) as raised:
        await current._handle_model_response(response)
    assert str(raised.value) == 'store turn "turn-1" response: disk unavailable', (
        f"handle model response error = {raised.value!r}"
    )
    assert store.appended_responses == [response], (
        f"appended responses = {store.appended_responses!r}"
    )
    assert len(translator.calls) == 0, (
        f"scheduled after response store failure: calls={translator.calls!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(store.appended_statuses) == 0, (
        f"scheduled after response store failure: calls={translator.calls!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(current._state.operations) == 0, (
        f"scheduled after response store failure: calls={translator.calls!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )


async def test_coordinator_handles_model_response_without_tool_calls() -> None:
    store = empty_fake_store()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    response = ModelResponse(
        TurnID="turn-1",
        Response=Response(
            Output=[LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="done"))]
        ),
    )

    await current._handle_model_response(response)
    assert store.appended_responses == [response], (
        f"response effects: responses={store.appended_responses!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(store.appended_statuses) == 0, (
        f"response effects: responses={store.appended_responses!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )
    assert len(current._state.operations) == 0, (
        f"response effects: responses={store.appended_responses!r} "
        f"statuses={store.appended_statuses!r} operations={current._state.operations!r}"
    )


async def test_coordinator_run_schedules_tool_calls_without_status_before_dispatch() -> None:
    first_spec = replace(value_spec(b'{"value":1}'), MaxOutputLength=3)
    second_spec = replace(value_spec(b'{"value":2}'), MaxOutputLength=7)
    translator = SubmittingTranslator([first_spec, second_spec])
    registry = Registry(StaticTranslators(bash=translator), BASH_NAME)
    handled = ToolCall(CallID="call-handled", Name=BASH_NAME, Arguments="{}")
    missing = ToolCall(CallID="call-missing", Name=BASH_NAME, Arguments="{}")
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(
                    Output=[
                        LLMItem(Type=ItemType.TOOL_CALL, Data=handled),
                        LLMItem(Type=ItemType.TOOL_CALL, Data=missing),
                    ]
                ),
            ),
        ),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(
                TurnID="turn-1",
                CallID=handled.CallID,
                Status=CallStatus(Error="already handled"),
            ),
        ),
    ]
    inputs = loop_run_closed_inbox()
    operations = FakeOperationManager()

    def add_error(value: Operation) -> Exception | None:
        if len(store.appended_statuses) != 1 or store.appended_statuses[0].CallID != missing.CallID:
            return RuntimeError("operation dispatched before tool-call status was stored")
        return None

    operations.add_error = add_error
    requests: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        requests.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry, adapter)

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}, want closed inbox error"
    assert str(err) == "inbox output closed", f"Run error = {err!r}, want closed inbox error"
    assert translator.calls == [missing], (
        f"translated calls = {translator.calls!r}, want {[missing]!r}"
    )
    assert len(store.appended_statuses) == 1, f"appended statuses = {store.appended_statuses!r}"
    status = store.appended_statuses[0]
    assert status.TurnID == "turn-1", f"appended status = {status!r}"
    assert status.CallID == missing.CallID, f"appended status = {status!r}"
    assert len(status.Status.WaitingFor or []) == 2, f"appended status = {status!r}"
    assert len(status.Operations) == 2, f"appended status = {status!r}"
    want_operations: dict[str, Operation] = {}
    for index, value in enumerate(status.Operations):
        assert value.ID != "", f"scheduled operation {index} = {value!r}"
        assert (status.Status.WaitingFor or [])[index] == value.ID, (
            f"scheduled operation {index} = {value!r}"
        )
        assert value.Status == Status.READY, f"scheduled operation {index} = {value!r}"
        assert value.MaxOutputLength == translator.specs[index].MaxOutputLength, (
            f"scheduled operation {index} = {value!r}"
        )
        want_operations[value.ID] = value
        local = current._state.operations.get(value.ID)
        assert local == value, f"local operation {value.ID!r} = {local!r}, want {value!r}"
    got_operations = {value.ID: value for value in operations.adds}
    assert got_operations == want_operations, (
        f"dispatched operations = {got_operations!r}, want {want_operations!r}"
    )
    await current._schedule_tool_calls()
    assert len(translator.calls) == 1, (
        f"rescheduled call: calls={translator.calls!r} statuses={store.appended_statuses!r}"
    )
    assert len(store.appended_statuses) == 1, (
        f"rescheduled call: calls={translator.calls!r} statuses={store.appended_statuses!r}"
    )
    assert_stop_result(await loop_run_receive(requests), handled.CallID, "already handled")


async def test_coordinator_run_starts_corrective_turn_for_recovered_validation_error() -> None:
    call = ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}")
    store = empty_fake_store()
    store.history = [
        stored_item(1, ItemKind.TURN, Turn(ID="turn-1", Type=TurnType.REGULAR)),
        stored_item(
            2,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
    ]
    started: asyncio.Queue[Request] = asyncio.Queue()

    async def respond(request: Request) -> Response:
        started.put_nowait(request)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    inputs = new_test_inbox()
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators(), BASH_NAME),
        adapter,
    )
    task = loop_run_start(current)

    request = await loop_run_receive(started)
    task.cancel()
    err = await loop_run_error(task)
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(store.appended_statuses) == 1, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert store.appended_statuses[0].Status.Error != "", (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(store.appended_turns) == 1, (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert store.appended_turns[0].PreviousTurnID == "turn-1", (
        f"statuses = {store.appended_statuses!r}, turns = {store.appended_turns!r}"
    )
    assert len(request.Input) == 3, f"corrective request = {request!r}"
    assert request.Input[2].Type == ItemType.TOOL_RESULT, f"corrective request = {request!r}"


async def test_coordinator_run_rejects_external_input_without_text_payload() -> None:
    store = empty_fake_store()
    inputs = new_test_inbox()
    await submit_test_input(
        inputs, Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=gojson.Raw(b"{}"))
    )
    current = new_test_coordinator(
        store,
        inputs,
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert 'add input "input-1" to context' in str(err), f"Run error = {err!r}"
    assert len(store.appended_inputs) == 0, f"stored inputs = {store.appended_inputs!r}"


async def test_coordinator_run_stores_operation_updates_without_redispatch() -> None:
    store = empty_fake_store()
    initial = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    store.resume.operations = [replace(initial)]
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    updated = replace(initial, Status=Status.AWAITING)
    operations.send(updated)
    operations.close()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), Registry(StaticTranslators()), adapter
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}, want closed operation updates error"
    assert str(err) == "operation updates closed", (
        f"Run error = {err!r}, want closed operation updates error"
    )
    local = current._state.operations.get(initial.ID)
    assert local == updated, f"operation = {local!r}, want {updated!r}"
    assert store.saved_operations == [updated], (
        f"saved operations = {store.saved_operations!r}, want {[updated]!r}"
    )
    assert operations.adds == [initial], (
        f"dispatched operations = {operations.adds!r}, want only the initial operation"
    )
    assert len(operations.cancels) == 0, (
        f"unexpected effects: cancels={operations.cancels} model={adapter.requests}"
    )
    assert len(adapter.requests) == 0, (
        f"unexpected effects: cancels={operations.cancels} model={adapter.requests}"
    )


async def test_coordinator_reconciles_tool_calls_from_persisted_operation_updates() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    store = empty_fake_store()
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_item_to_local_state(
        Item(
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        )
    )
    operations = [
        Operation(ID="operation-1", Status=Status.READY),
        Operation(ID="operation-2", Status=Status.AWAITING),
    ]
    for value in operations:
        current._add_operation_to_local_state(value)
    status = ToolCallStatus(
        TurnID="turn-1",
        CallID=call.CallID,
        Status=CallStatus(WaitingFor=[operations[0].ID, operations[1].ID]),
    )
    current._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))

    first = replace(operations[0], Status=Status.COMPLETED)
    await current._handle_operation_update(first)
    await current._reconcile_tool_calls()
    key = ToolCallKey(status.TurnID, status.CallID)
    assert key in current._state.tool_calls, (
        "tool call was removed before every operation became terminal"
    )

    second = replace(operations[1], Status=Status.FAILED)
    await current._handle_operation_update(second)
    await current._reconcile_tool_calls()
    assert key not in current._state.tool_calls, "completed tool call remains in local state"
    assert store.saved_operations == [first, second], (
        f"saved operations = {store.saved_operations!r}, want {[first, second]!r}"
    )
    assert store.appended_statuses == [
        ToolCallStatus(
            TurnID=status.TurnID,
            CallID=status.CallID,
            Status=status.Status,
            Operations=[first, second],
        )
    ], f"appended statuses = {store.appended_statuses!r}"
    built = builder.build()
    want = with_preamble(
        LLMItem(Type=ItemType.TOOL_CALL, Data=call),
        LLMItem(
            Type=ItemType.TOOL_RESULT,
            Data=ToolResult(
                CallID=call.CallID,
                Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="error:")],
            ),
        ),
    )
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_run_returns_reconciliation_error() -> None:
    registry = Registry(StaticTranslators(view_image=TerminalResultTranslator()), VIEW_IMAGE_NAME)
    value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=[value.ID])
    )
    store = empty_fake_store()
    store.resume.operations = [replace(value)]
    store.history = [
        stored_item(
            1,
            ItemKind.MODEL_RESPONSE,
            ModelResponse(
                TurnID="turn-1",
                Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
            ),
        ),
        stored_item(2, ItemKind.TOOL_CALL_STATUS, status),
        stored_item(
            3,
            ItemKind.TOOL_CALL_STATUS,
            ToolCallStatus(TurnID="another-turn", CallID="another-call"),
        ),
        stored_item(4, ItemKind.TURN, Turn(ID="turn-2", Type=TurnType.REGULAR)),
    ]
    completed = replace(value, Status=Status.COMPLETED)
    operations = FakeOperationManager()
    operations.send(completed)
    current = new_test_coordinator(store, new_test_inbox(), operations, new_builder(), registry)

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == ('add tool call "call-1" result to context: terminal result failed'), (
        f"Run error = {err!r}"
    )
    assert store.saved_operations == [completed], (
        f"saved operations = {store.saved_operations!r}, want {[completed]!r}"
    )


async def test_coordinator_does_not_complete_tool_call_before_operation_is_stored() -> None:
    registry = Registry(StaticTranslators(view_image=TestTranslator()), VIEW_IMAGE_NAME)
    store = empty_fake_store()
    store.save_operation_error = RuntimeError("disk unavailable")
    builder = new_builder()
    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), builder, registry
    )
    call = ToolCall(CallID="call-1", Name=VIEW_IMAGE_NAME, Arguments="{}")
    current._add_tool_calls_to_local_state(
        ModelResponse(
            TurnID="turn-1",
            Response=Response(Output=[LLMItem(Type=ItemType.TOOL_CALL, Data=call)]),
        )
    )
    value = Operation(ID="operation-1", Status=Status.READY)
    current._add_operation_to_local_state(value)
    status = ToolCallStatus(
        TurnID="turn-1", CallID=call.CallID, Status=CallStatus(WaitingFor=[value.ID])
    )
    current._add_item_to_local_state(Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status))

    value = replace(value, Status=Status.COMPLETED)
    with pytest.raises(CoordinatorError) as raised:
        await current._handle_operation_update(value)
    assert str(raised.value) == 'store operation "operation-1": disk unavailable', (
        f"handle operation update error = {raised.value!r}"
    )
    assert ToolCallKey(status.TurnID, status.CallID) in current._state.tool_calls, (
        "tool call was removed before its operation was stored"
    )
    built = builder.build()
    assert len(built.request.Input) == 2, (
        f"tool results = {built.request.Input!r}, want only the initial result"
    )


async def test_coordinator_run_dispatches_restored_non_terminal_operations() -> None:
    store = empty_fake_store()
    statuses = [
        Status.READY,
        Status.AWAITING,
        Status.CANCELING,
        Status.COMPLETED,
        Status.FAILED,
        Status.CANCELED,
    ]
    for status in statuses:
        store.resume.operations.append(
            Operation(ID=status.value, Type=TYPE_SHELL, Version=1, Status=status)
        )
    inputs = loop_run_closed_inbox()
    operations = FakeOperationManager()
    adapter = FakeAdapter()
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), Registry(StaticTranslators()), adapter
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}, want closed inbox error"
    assert str(err) == "inbox output closed", f"Run error = {err!r}, want closed inbox error"
    adds = sorted(operations.adds, key=lambda value: value.ID)
    want = [
        Operation(ID="awaiting", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING),
        Operation(ID="canceling", Type=TYPE_SHELL, Version=1, Status=Status.CANCELING),
        Operation(ID="ready", Type=TYPE_SHELL, Version=1, Status=Status.READY),
    ]
    assert adds == want, f"dispatched operations = {adds!r}, want {want!r}"
    assert len(adapter.requests) == 0, f"model requests = {adapter.requests!r}"


async def test_coordinator_run_returns_operation_dispatch_error() -> None:
    store = empty_fake_store()
    value = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.READY)
    store.resume.operations = [value]
    operations = FakeOperationManager()
    operations.add_error = lambda _: RuntimeError("dispatch failed")
    current = new_test_coordinator(
        store, new_test_inbox(), operations, new_builder(), Registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == 'dispatch operation "operation-1": dispatch failed', f"Run error = {err!r}"


async def test_coordinator_run_returns_unsupported_recovered_operation() -> None:
    store = empty_fake_store()
    store.resume.operations = [
        Operation(ID="unsupported", Type="remote", Version=1, Status=Status.READY),
        Operation(ID="supported", Type=TYPE_SHELL, Version=1, Status=Status.READY),
    ]
    inputs = loop_run_closed_inbox()
    operations = FakeOperationManager()

    def add_error(value: Operation) -> Exception | None:
        if value.ID == "unsupported":
            return UnsupportedError(
                f"manager does not support {go_quote(value.Type)}: unsupported operation"
            )
        return None

    operations.add_error = add_error
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), Registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert caused_by(err, UnsupportedError), f"Run error = {err!r}, want unsupported operation"
    assert "unsupported" in current._state.operations, "unsupported operation was not retained"


async def test_coordinator_clones_operation_data_before_dispatch() -> None:
    operations = FakeOperationManager()
    operations.mutate_adds = True
    current = new_test_coordinator(
        empty_fake_store(),
        new_test_inbox(),
        operations,
        new_builder(),
        Registry(StaticTranslators()),
    )
    value = current._add_operation_to_local_state(
        Operation(
            ID="operation-1",
            Type=TYPE_SHELL,
            Version=1,
            Status=Status.READY,
            State=gojson.Raw(b'{"state":"original"}'),
            Idempotency=gojson.Raw(b'{"key":"original"}'),
        )
    )

    await current._dispatch_operations_to_manager()
    stored = current._state.operations[value.ID]
    assert bytes(stored.State) == b'{"state":"original"}', (
        f"stored operation was mutated: {stored!r}"
    )
    assert bytes(stored.Idempotency) == b'{"key":"original"}', (
        f"stored operation was mutated: {stored!r}"
    )


async def test_coordinator_run_returns_input_store_error_after_updating_local_state() -> None:
    store = empty_fake_store()
    store.append_input_error = RuntimeError("disk unavailable")
    inputs = new_test_inbox()
    event = external_event("input-1", "hello")
    await submit_test_input(inputs, event)
    current = new_test_coordinator(
        store, inputs, FakeOperationManager(), new_builder(), Registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == 'store input "input-1": disk unavailable', f"Run error = {err!r}"
    built = deps(current).context_builder.build()
    want = with_preamble(LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello")))
    assert built.request.Input == want, f"built input = {built.request.Input!r}, want {want!r}"


async def test_coordinator_run_returns_operation_store_error_after_updating_local_state() -> None:
    store = empty_fake_store()
    store.save_operation_error = RuntimeError("disk unavailable")
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    update = Operation(ID="operation-1", Type=TYPE_SHELL, Version=1, Status=Status.AWAITING)
    operations.send(update)
    current = new_test_coordinator(
        store, inputs, operations, new_builder(), Registry(StaticTranslators())
    )

    err = await loop_run_returned(current)
    assert err is not None, f"Run error = {err!r}"
    assert str(err) == 'store operation "operation-1": disk unavailable', f"Run error = {err!r}"
    local = current._state.operations.get(update.ID)
    assert local == update, f"operation = {local!r}, want {update!r}"
    assert len(operations.adds) == 0, (
        f"operations dispatched before store commit = {operations.adds!r}"
    )


def _loop_run_fail_turn(store: FakeStore, _: Coordinator) -> None:
    store.append_turn_error = RuntimeError("disk unavailable")


def _loop_run_fail_model_response(store: FakeStore, _: Coordinator) -> None:
    store.append_model_response_error = RuntimeError("disk unavailable")


def _loop_run_fail_status(store: FakeStore, _: Coordinator) -> None:
    store.append_status_error = RuntimeError("disk unavailable")


def _loop_run_configure_nothing(store: FakeStore, _: Coordinator) -> None:
    pass


_LOOP_RUN_TURN = Turn(ID="turn-1", Type=TurnType.REGULAR)


@pytest.mark.parametrize(
    ("item", "configure", "want"),
    [
        pytest.param(
            Item(Kind=ItemKind.TURN, Data=_LOOP_RUN_TURN),
            _loop_run_fail_turn,
            'store turn "turn-1": disk unavailable',
            id="turn",
        ),
        pytest.param(
            Item(Kind=ItemKind.MODEL_RESPONSE, Data=ModelResponse(TurnID=_LOOP_RUN_TURN.ID)),
            _loop_run_fail_model_response,
            'store turn "turn-1" response: disk unavailable',
            id="model response",
        ),
        pytest.param(
            Item(
                Kind=ItemKind.TOOL_CALL_STATUS,
                Data=ToolCallStatus(
                    TurnID=_LOOP_RUN_TURN.ID,
                    CallID="call-1",
                    Status=CallStatus(WaitingFor=["operation-1"]),
                ),
            ),
            _loop_run_fail_status,
            'store tool call "call-1" status: disk unavailable',
            id="tool status",
        ),
        pytest.param(
            Item(Kind=ItemKind.FORK, Data=Fork()),
            _loop_run_configure_nothing,
            'unsupported local item kind "fork"',
            id="unsupported item",
        ),
    ],
)
async def test_coordinator_returns_session_item_store_errors(
    item: Item, configure: Callable[[FakeStore, Coordinator], None], want: str
) -> None:
    store = empty_fake_store()
    current = new_test_coordinator(
        store,
        new_test_inbox(),
        FakeOperationManager(),
        new_builder(),
        Registry(StaticTranslators()),
    )
    configure(store, current)

    with pytest.raises(CoordinatorError) as raised:
        await current._store_item_in_session_store(item)
    assert str(raised.value) == want, f"store error = {raised.value!r}, want {want!r}"
