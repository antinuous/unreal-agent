from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Role
from unreal_agent.operation import LocalOperationManager, Status, UnsupportedError
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import BEFORE_FIRST, ItemKind, ModelResponse, ToolCallStatus
from unreal_agent.tool import CallStatus, StaticTranslators, new_registry

from tests.coordinator._helpers import (
    MemoryStore,
    assert_stop_result,
    caused_by,
    external_event,
    fake_time,
    new_stop_test_run,
    persist_test_run,
    restore_test_run,
    stop_input,
    stored_item,
    submit_test_input,
    text_response,
    with_preamble,
)


@pytest.mark.parametrize("stage", ["untranslated", "rejected", "accepted"])
@fake_time
async def test_coordinator_resumes_unavailable_tool(stage: str) -> None:
    run = new_stop_test_run(1)
    want_error = 'tool "ViewImage" is not available'
    if stage == "untranslated":
        run.store.history = run.store.history[:2]
    elif stage == "rejected":
        status = cast(ToolCallStatus, run.store.history[2].Data)
        run.store.history[2].Data = replace(
            status, Status=CallStatus(Error=want_error), Operations=[]
        )
    store = await persist_test_run(run)
    before = await store.items("session-1", BEFORE_FIRST, 100)
    await restore_test_run(run, store)
    run.dependencies.tools = new_registry(StaticTranslators())
    await run.start()
    if stage != "accepted":
        await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
        assert len(run.calls) == 1, (
            "unavailable tool did not produce one corrective request without dispatch"
        )
        assert len(run.operations.adds) == 0, (
            "unavailable tool did not produce one corrective request without dispatch"
        )
        assert_stop_result(run.calls[0].request, "call-0", want_error)
        await run.respond(0, text_response("Corrected."))
        run.assert_stopped()

        resumed = new_stop_test_run(0)
        resumed.dependencies.tools = new_registry(StaticTranslators())
        await restore_test_run(resumed, store)
        await resumed.start()
        await resumed.input(stop_input("stop-again", ControlMode.STOP_WHEN_IDLE))
        resumed.assert_stopped()
        assert len(resumed.calls) == 0, "settled validation error started work on resume"
        assert len(resumed.operations.adds) == 0, "settled validation error started work on resume"
        return
    err = run.finished_error()
    assert err is not None, f"Run error = {err!r}, want unavailable-tool error"
    assert 'tool "ViewImage" required by recorded call "call-0" is not available' in str(err), (
        f"Run error = {err!r}, want unavailable-tool error"
    )
    assert len(run.calls) == 0, "unresolvable tool started work"
    assert len(run.operations.adds) == 0, "unresolvable tool started work"
    after = await store.items("session-1", BEFORE_FIRST, 100)
    assert before == after, "failed resume changed persisted history"


@pytest.mark.parametrize(
    "stage", ["input", "turn", "response", "compaction", "compaction response"]
)
@fake_time
async def test_coordinator_resumes_unanswered_input(stage: str) -> None:
    run = new_stop_test_run(0)
    run.store.history = [stored_item(1, ItemKind.INPUT, external_event("user", "hello"))]
    if stage != "input":
        turn = Turn(ID="first", Type=TurnType.REGULAR)
        if stage in ("compaction", "compaction response"):
            turn.Type = TurnType.COMPACTION
        run.store.history.append(stored_item(2, ItemKind.TURN, turn))
    if stage in ("response", "compaction response"):
        run.store.history.append(
            stored_item(
                3,
                ItemKind.MODEL_RESPONSE,
                ModelResponse(TurnID="first", Response=text_response("Hello.")),
            )
        )
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    if stage == "response":
        assert len(run.calls) == 0, "answered input caused another request"
    else:
        assert len(run.calls) == 1, f"requests = {len(run.calls)}, want resumed input delivery"
        assert run.current._state.current_turn_type == TurnType.REGULAR, (
            "resume did not start ordinary delivery of pending input"
        )
        assert run.current._state.current_turn_id != "first", (
            "resume did not start ordinary delivery of pending input"
        )
        assert run.current._pending_inputs() == 1, (
            "resume did not start ordinary delivery of pending input"
        )
        want = with_preamble(
            LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text="hello"))
        )
        assert run.calls[0].request.Input == want, "resume changed pending input context"
        await run.respond(0, text_response("Hello."))
    assert run.current._pending_inputs() == 0, "ordinary response did not deliver pending input"
    run.assert_stopped()


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(ControlMode.STOP_HARD, id=str(ControlMode.STOP_HARD)),
        pytest.param(ControlMode.STOP_WHEN_IDLE, id=str(ControlMode.STOP_WHEN_IDLE)),
    ],
)
@fake_time
async def test_coordinator_stop_rejects_unsupported_operation(mode: ControlMode) -> None:
    run = new_stop_test_run(1)
    manager = LocalOperationManager()
    try:
        run.dependencies.operations = manager
        await submit_test_input(run.inputs, stop_input("stop", mode))
        await run.start()
        err = run.finished_error()
        assert caused_by(err, UnsupportedError), f"Run error = {err!r}, want unsupported operation"
        assert len(run.store.saved_operations) == 0, (
            "unsupported operation changed state or started a model request"
        )
        assert len(run.calls) == 0, "unsupported operation changed state or started a model request"
    finally:
        await manager.aclose()


@pytest.mark.parametrize(
    "terminal",
    [
        pytest.param(Status.COMPLETED, id=str(Status.COMPLETED)),
        pytest.param(Status.FAILED, id=str(Status.FAILED)),
        pytest.param(Status.CANCELED, id=str(Status.CANCELED)),
    ],
)
@fake_time
async def test_coordinator_resumes_saved_terminal_operation(terminal: Status) -> None:
    run = new_stop_test_run(1)
    store = MemoryStore()
    await store.create("session-1")
    await store.append_turn("session-1", cast(Turn, run.store.history[0].Data))
    await store.append_model_response("session-1", cast(ModelResponse, run.store.history[1].Data))
    await store.append_tool_call_status(
        "session-1", cast(ToolCallStatus, run.store.history[2].Data)
    )
    completed = replace(run.store.resume.operations[0], Status=terminal)
    await store.save_operation("session-1", completed)
    restored = await store.resume("session-1")
    run.dependencies.sessions, run.dependencies.restored = store, restored
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(run.operations.adds) == 0, "saved terminal operation was dispatched again"
    assert len(run.calls) == 1, f"requests = {len(run.calls)}, want restored result delivery"
    assert_stop_result(run.calls[0].request, "call-0", str(terminal))
    await run.respond(0, text_response("Done."))
    run.assert_stopped()

    resumed = new_stop_test_run(0)
    restored = await store.resume("session-1")
    assert len(restored.operations) == 0, (
        f"settled operations retained for resume: {restored.operations!r}"
    )
    resumed.dependencies.sessions, resumed.dependencies.restored = store, restored
    await resumed.start()
    await resumed.input(stop_input("stop-again", ControlMode.STOP_WHEN_IDLE))
    assert len(resumed.calls) == 0, "delivered terminal result caused work on the next resume"
    assert len(resumed.operations.adds) == 0, (
        "delivered terminal result caused work on the next resume"
    )
    resumed.assert_stopped()
