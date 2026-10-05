import dataclasses
from pathlib import Path
from typing import Any, cast

import pytest
from msgspec.structs import replace
from unreal_agent.contextbuilder import Builder, new_builder
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, ToolCall, ToolResult
from unreal_agent.operation import Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import Item, ItemKind, ModelResponse, ToolCallStatus
from unreal_agent.sessionstore.localfile import Store as LocalFileStore

from tests.coordinator._helpers import (
    FakeOperationManager,
    external_event,
    independent_tool_calls,
    new_stop_test_run,
    new_test_coordinator,
    new_test_inbox,
    stored_item,
    text_response,
)

DELIVERY_STEPS: list[tuple[str, ItemKind, str, int, int, int, int]] = [
    ("first completion", ItemKind.TOOL_CALL_STATUS, "first", 1, 1, 0, 1),
    ("request starts", ItemKind.TURN, "turn-1", 1, 1, 0, 1),
    ("second completes during request", ItemKind.TOOL_CALL_STATUS, "second", 0, 2, 0, 2),
    ("duplicate completion", ItemKind.TOOL_CALL_STATUS, "second", 0, 2, 0, 2),
    ("request completes", ItemKind.MODEL_RESPONSE, "response-1", 0, 2, 1, 1),
    ("next request starts", ItemKind.TURN, "turn-2", 0, 2, 1, 1),
    ("next request completes", ItemKind.MODEL_RESPONSE, "response-2", 0, 2, 2, 0),
]


@pytest.mark.parametrize(
    "step_index", range(len(DELIVERY_STEPS)), ids=[step[0] for step in DELIVERY_STEPS]
)
async def test_coordinator_replays_tool_result_balance(step_index: int) -> None:
    store, registry = independent_tool_calls(2)
    store.resume.operations = []

    def completion(index: int) -> ToolCallStatus:
        status = cast(ToolCallStatus, store.history[index + 2].Data)
        value = replace(status.Operations[0], Status=Status.COMPLETED)
        return replace(status, Operations=[value])

    first, second = completion(0), completion(1)
    data: dict[str, Any] = {
        "first": first,
        "second": second,
        "turn-1": Turn(ID="delivery-1", PreviousTurnID="turn-1", Type=TurnType.REGULAR),
        "response-1": ModelResponse(TurnID="delivery-1", Response=text_response("Checking.")),
        "turn-2": Turn(ID="delivery-2", PreviousTurnID="delivery-1", Type=TurnType.REGULAR),
        "response-2": ModelResponse(TurnID="delivery-2", Response=text_response("Done.")),
    }
    for _, kind, key, *_counts in DELIVERY_STEPS[: step_index + 1]:
        store.history.append(stored_item(len(store.history) + 1, kind, data[key]))
    _, _, _, pending_calls, completed, delivered, pending_results = DELIVERY_STEPS[step_index]

    current = new_test_coordinator(
        store, new_test_inbox(), FakeOperationManager(), new_builder(), registry
    )
    await current._restore()
    got = len(current._state.tool_calls)
    assert got == pending_calls, f"pending calls = {got}, want {pending_calls}"
    got = current._state.available_inputs
    assert got == completed, f"completed results = {got}, want {completed}"
    built = current._dependencies.context_builder.build()
    appended = 0
    for item in built.request.Input:
        if (
            item.Type == ItemType.TOOL_RESULT
            and cast(ToolResult, item.Data).Output[0].Value == Status.COMPLETED
        ):
            appended += 1
    assert appended == current._state.available_inputs, (
        f"appended completions = {appended}, available inputs = {current._state.available_inputs}"
    )
    got = current._state.delivered_inputs
    assert got == delivered, f"delivered results = {got}, want {delivered}"
    got = current._pending_inputs()
    assert got == pending_results, f"pending results = {got}, want {pending_results}"


async def test_coordinator_compaction_preserves_pending_inputs_on_replay(tmp_path: Path) -> None:
    run = new_stop_test_run(2)
    store = LocalFileStore(tmp_path)
    await store.create("session-1")
    current = run.current
    current._dependencies.sessions = store
    status = cast(ToolCallStatus, run.store.history[2].Data)
    completed = replace(status.Operations[0], Status=Status.COMPLETED)
    status = replace(status, Operations=[completed])
    items = [
        *run.store.history,
        Item(Kind=ItemKind.INPUT, Data=external_event("before", "Before compaction")),
        Item(
            Kind=ItemKind.TURN,
            Data=Turn(ID="compact", PreviousTurnID="turn-1", Type=TurnType.COMPACTION),
        ),
        Item(Kind=ItemKind.INPUT, Data=external_event("during", "During compaction")),
        Item(Kind=ItemKind.TOOL_CALL_STATUS, Data=status),
    ]
    for item in items:
        current._add_item_to_local_state(item)
        await current._store_item_in_session_store(item)
    before = current._dependencies.context_builder.build()
    response = ModelResponse(TurnID="compact", Response=text_response("Summary"))
    response.Response.Output.append(
        LLMItem(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="summary-call", Name="unknown"))
    )
    current._state.call_model = True
    statuses = await current._handle_model_response(response)
    message = "compaction response changed ordinary scheduling"
    assert len(statuses) == 0, message
    assert current._state.call_model, message
    current._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=response))
    state = current._state
    message = f"compaction changed pending work: {state}"
    assert state.current_turn_type == TurnType.COMPACTION, message
    assert state.delivered_inputs == 0, message
    assert current._pending_inputs() == 3, message
    assert len(state.tool_calls) == 1, message
    assert len(state.operations) == 2, message
    reopened = LocalFileStore(tmp_path)
    page = await reopened.items("session-1", 0, 100)
    assert page.items[-1].Data == response, "compaction response was not persisted"
    replayed = new_stop_test_run(0).current
    replayed._dependencies.sessions = reopened
    await replayed._load_history()
    want_state = dataclasses.replace(current._state, call_model=False)
    assert replayed._state == want_state, f"replayed state = {replayed._state}, want {want_state}"
    builders: list[Builder] = [
        current._dependencies.context_builder,
        replayed._dependencies.context_builder,
    ]
    for builder in builders:
        built = builder.build()
        assert built.request == before.request, "compaction changed active context"
