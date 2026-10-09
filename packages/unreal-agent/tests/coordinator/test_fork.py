from pathlib import Path
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Response, ToolCall
from unreal_agent.operation import Status
from unreal_agent.session import Turn, TurnID, TurnType
from unreal_agent.sessionstore import BEFORE_FIRST, ModelResponse, Store, StoreError, ToolCallStatus
from unreal_agent.sessionstore.localfile import Store as LocalFileStore
from unreal_agent.tool import BASH_NAME, VIEW_IMAGE_NAME, Registry, StaticTranslators

from tests.coordinator._helpers import (
    SLURP_IDLE_TIMEOUT_NS,
    MemoryStore,
    SubmittingTranslator,
    TestTranslator,
    external_event,
    fake_time,
    new_stop_test_run,
    persist_test_run,
    sleep_ns,
    stop_input,
    text_response,
    value_spec,
    wait,
)

FORK_PARENT_STAGES = ["unscheduled", "running", "completed", "compaction", "compaction response"]


@pytest.mark.parametrize(
    ("parent_stage", "child_tool"),
    [(stage, tool) for stage in FORK_PARENT_STAGES for tool in (False, True)],
    ids=[
        f"parent={stage}/child-tool={str(tool).lower()}"
        for stage in FORK_PARENT_STAGES
        for tool in (False, True)
    ],
)
@fake_time
async def test_coordinator_fork_stops_when_child_is_idle(
    parent_stage: str, child_tool: bool
) -> None:
    parent = new_stop_test_run(1)
    if parent_stage == "unscheduled":
        parent.store.history = parent.store.history[:2]
    store = await persist_test_run(parent)
    previous_turn: TurnID = "turn-1"
    if parent_stage == "completed":
        status = cast(ToolCallStatus, parent.store.history[2].Data)
        status.Operations[0].Status = Status.COMPLETED
        await store.save_operation("session-1", status.Operations[0])
        await store.append_tool_call_status("session-1", status)
        previous_turn = "parent-final"
        await store.append_turn(
            "session-1", Turn(ID=previous_turn, PreviousTurnID="turn-1", Type=TurnType.REGULAR)
        )
        await store.append_model_response(
            "session-1", ModelResponse(TurnID=previous_turn, Response=text_response("Parent done."))
        )
    compaction = parent_stage in ("compaction", "compaction response")
    if compaction:
        previous_turn = "compact"
        await store.append_turn(
            "session-1",
            Turn(ID=previous_turn, PreviousTurnID="turn-1", Type=TurnType.COMPACTION),
        )
        if parent_stage == "compaction response":
            await store.append_model_response(
                "session-1", ModelResponse(TurnID=previous_turn, Response=text_response("Summary"))
            )
    await store.fork("child", "session-1", previous_turn)
    restored = await store.resume("child")
    child = new_stop_test_run(0)
    child.dependencies.session_id = "child"
    child.dependencies.sessions = cast(Store, store)
    child.dependencies.restored = restored
    spec = value_spec(b'{"value":1}')
    registry = Registry(
        StaticTranslators(bash=SubmittingTranslator([spec]), view_image=TestTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    child.dependencies.tools = registry
    await child.start()
    child.assert_running()
    message = "inherited tool call became active child work"
    assert len(child.operations.adds) == 0, message
    assert len(child.current._state.tool_calls) == 0, message
    assert child.current._state.current_turn_id == previous_turn, (
        "fork lost the parent turn boundary"
    )
    message = "fork changed the parent turn kind or started inherited work"
    assert (child.current._state.current_turn_type == TurnType.COMPACTION) == compaction, message
    assert child.current._pending_inputs() == 0, message
    assert len(child.calls) == 0, message
    await child.input(
        external_event("child-user", "hello"),
        stop_input("child-stop", ControlMode.STOP_WHEN_IDLE),
    )
    assert child.current._state.current_turn_type == TurnType.REGULAR, (
        "child turn did not establish its own type"
    )
    inherited_call = False
    for item in child.calls[0].request.Input:
        if item.Type == ItemType.TOOL_CALL and cast(ToolCall, item.Data).CallID == "call-0":
            inherited_call = True
    assert inherited_call, "fork lost its parent context"
    if child_tool:
        await child.respond(
            0,
            Response(
                Output=[
                    LLMItem(
                        Type=ItemType.TOOL_CALL,
                        Data=ToolCall(CallID="child-call", Name=BASH_NAME, Arguments="{}"),
                    )
                ]
            ),
        )
        child.assert_running()
        message = "child tool call was not tracked"
        assert len(child.operations.adds) == 1, message
        assert len(child.current._state.tool_calls) == 1, message
        completed = replace(child.operations.adds[0], Status=Status.COMPLETED)
        child.operations.send(completed)
        await wait()
        await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        child.assert_running()
        await child.respond(1, text_response("Child tool done."))
    else:
        await child.respond(0, text_response("Child done."))
    child.assert_stopped()


async def _fork_store_outcome(
    store: MemoryStore | LocalFileStore, parent_stage: str
) -> list[object]:
    """The history and resume state a fork test sees, without timestamps."""
    parent = new_stop_test_run(1)
    if parent_stage == "unscheduled":
        parent.store.history = parent.store.history[:2]
    await store.create("session-1")
    await store.append_turn("session-1", cast(Turn, parent.store.history[0].Data))
    await store.append_model_response(
        "session-1", cast(ModelResponse, parent.store.history[1].Data)
    )
    for item in parent.store.history[2:]:
        await store.append_tool_call_status("session-1", cast(ToolCallStatus, item.Data))
    await store.append_input("session-1", external_event("parent-user", "hello"))
    previous_turn: TurnID = "turn-1"
    if parent_stage == "completed":
        status = cast(ToolCallStatus, parent.store.history[2].Data)
        status.Operations[0].Status = Status.COMPLETED
        await store.save_operation("session-1", status.Operations[0])
        await store.append_tool_call_status("session-1", status)
        previous_turn = "parent-final"
        await store.append_turn(
            "session-1", Turn(ID=previous_turn, PreviousTurnID="turn-1", Type=TurnType.REGULAR)
        )
        await store.append_model_response(
            "session-1", ModelResponse(TurnID=previous_turn, Response=text_response("Parent done."))
        )
    if parent_stage in ("compaction", "compaction response"):
        previous_turn = "compact"
        await store.append_turn(
            "session-1",
            Turn(ID=previous_turn, PreviousTurnID="turn-1", Type=TurnType.COMPACTION),
        )
        if parent_stage == "compaction response":
            await store.append_model_response(
                "session-1", ModelResponse(TurnID=previous_turn, Response=text_response("Summary"))
            )
    outcome: list[object] = []
    for target in ("missing-turn", previous_turn):
        try:
            await store.fork(f"child-{target}", "session-1", target)
        except StoreError as error:
            outcome.append(("fork error", type(error), str(error)))
    for id in ("session-1", f"child-{previous_turn}"):
        page = await store.items(id, BEFORE_FIRST, 100)
        outcome.append([(item.Sequence, item.Kind, item.Data) for item in page.items])
        outcome.append((page.next_after, page.more))
        resumed = await store.resume(id)
        outcome.append((resumed.snapshot.session.ID, resumed.operations))
        outcome.append(resumed.input_ids)
    return outcome


@pytest.mark.parametrize("parent_stage", FORK_PARENT_STAGES)
async def test_memory_store_matches_localfile_fork(parent_stage: str, tmp_path: Path) -> None:
    want = await _fork_store_outcome(LocalFileStore(tmp_path), parent_stage)
    got = await _fork_store_outcome(MemoryStore(), parent_stage)
    assert got == want, f"memory store = {got!r}, localfile store = {want!r}"
