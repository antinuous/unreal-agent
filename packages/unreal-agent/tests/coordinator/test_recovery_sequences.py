import asyncio
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import ItemType, Message, Request, Response, ToolResult
from unreal_agent.operation import Status
from unreal_agent.sessionstore import BEFORE_FIRST, ItemKind, ToolCallStatus
from unreal_agent.tool import VIEW_IMAGE_NAME, StaticTranslators, new_registry

from tests.coordinator._helpers import (
    SLURP_IDLE_TIMEOUT_NS,
    FailingBuilder,
    FailingStore,
    FakeAdapter,
    StopTestCall,
    SubmittingTranslator,
    as_builder,
    assert_completed_results,
    assert_stop_result,
    caused_by,
    external_event,
    new_stop_test_run,
    persist_test_run,
    restore_test_run,
    sleep_ns,
    stop_input,
    synctest,
    text_response,
    value_spec,
    wait,
)


@pytest.mark.parametrize("stage", ["steered-request", "followup-request"])
@synctest
async def test_coordinator_resumes_mixed_delivery_after_steering(stage: str) -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    await restore_test_run(run, store)
    run_canceled = False

    async def respond(request: Request) -> Response:
        task = asyncio.current_task()
        assert task is not None
        call = StopTestCall(task, request, asyncio.get_running_loop().create_future())
        run.calls.append(call)
        if len(run.calls) == 1:
            while True:
                try:
                    return await asyncio.shield(call.response)
                except asyncio.CancelledError:
                    if call.response.done():
                        return call.response.result()
                    if run_canceled:
                        raise
        return await call.response

    run.dependencies.llm = FakeAdapter(respond)
    run.go()
    await wait()
    await run.input(external_event("first", "first input"))
    await run.update(0, Status.COMPLETED)
    await run.input(external_event("steer", "steering input"))
    await run.update(1, Status.COMPLETED)
    assert len(run.calls) == 2, "steering did not replace the first request"
    assert run.calls[0].canceled, "steering did not replace the first request"
    await run.respond(0, text_response("Stale response."))
    assert run.current._state.delivered_inputs == 0, "stale response consumed pending inputs"
    pending = 4
    if stage == "followup-request":
        await run.respond(1, text_response("Current response."))
        pending = 1
        assert len(run.calls) == 3, "completion during the steered request did not trigger delivery"
    run_canceled = True
    run.cancel()
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), f"Run error = {err!r}"

    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    got = resumed.current._pending_inputs()
    assert got == pending, f"replayed balance = {got}, want {pending}"
    assert len(resumed.calls) == 1, f"resumed requests = {len(resumed.calls)}, want 1"
    request = resumed.calls[0].request
    assert_completed_results(request, 2)
    messages: dict[str, int] = {}
    for item in request.Input:
        if item.Type == ItemType.MESSAGE:
            text = cast(Message, item.Data).Text
            messages[text] = messages.get(text, 0) + 1
    assert messages.get("first input", 0) == 1, f"replayed messages = {messages}"
    assert messages.get("steering input", 0) == 1, f"replayed messages = {messages}"
    assert messages.get("Stale response.", 0) == 0, f"replayed messages = {messages}"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, "resumed delivery did not settle exactly once"
    assert len(resumed.calls) == 1, "resumed delivery did not settle exactly once"


@synctest
async def test_coordinator_resumes_partially_completed_tool_call() -> None:
    run = new_stop_test_run(1)
    second = replace(run.store.resume.operations[0], ID="operation-1")
    run.store.resume.operations.append(second)
    status = cast(ToolCallStatus, run.store.history[2].Data)
    status = replace(
        status,
        Status=replace(status.Status, WaitingFor=[*status.Status.WaitingFor, second.ID]),
        Operations=[*status.Operations, second],
    )
    run.store.history[2].Data = status
    store = await persist_test_run(run)
    first = replace(status.Operations[0], Status=Status.COMPLETED)
    await store.save_operation("session-1", first)
    await restore_test_run(run, store)
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_running()
    assert len(run.calls) == 0, "partial completion produced a finished tool result"
    assert run.current._pending_inputs() == 0, "partial completion produced a finished tool result"
    assert len(run.operations.adds) != 0, "unfinished operation was not dispatched"
    for dispatched in run.operations.adds:
        assert dispatched == second, f"redispatched operation = {dispatched!r}, want {second!r}"
    await run.update(1, Status.COMPLETED)
    assert len(run.calls) == 1, f"requests = {len(run.calls)}, want one completed tool result"
    assert_stop_result(run.calls[0].request, "call-0", "completed,completed")
    completed_results = 0
    for item in run.calls[0].request.Input:
        if (
            item.Type == ItemType.TOOL_RESULT
            and cast(ToolResult, item.Data).Output[0].Value == "completed,completed"
        ):
            completed_results += 1
    assert completed_results == 1, f"completed tool results = {completed_results}, want 1"
    await run.update(1, Status.COMPLETED)
    assert run.current._pending_inputs() == 1, (
        "duplicate terminal update changed the delivery balance"
    )
    await run.respond(0, text_response("Done."))
    run.assert_stopped()
    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    assert len(resumed.dependencies.restored.operations) == 0, (
        "settled operations were retained for recovery"
    )
    await resumed.start()
    await resumed.input(stop_input("stop-again", ControlMode.STOP_WHEN_IDLE))
    resumed.assert_stopped()
    assert len(resumed.calls) == 0, "settled tool call ran again on resume"
    assert len(resumed.operations.adds) == 0, "settled tool call ran again on resume"


@pytest.mark.parametrize("failure", ["reconciliation", "initial-build", "initial-turn"])
@synctest
async def test_coordinator_retries_recovery_persistence_failures(failure: str) -> None:
    run = new_stop_test_run(1)
    store = await persist_test_run(run)
    want = RuntimeError("injected failure")
    faults = FailingStore(store)
    terminal = Status.COMPLETED
    value = replace(run.store.resume.operations[0], Status=terminal)
    await store.save_operation("session-1", value)
    if failure == "reconciliation":
        faults.status_error = want
    elif failure == "initial-build":
        run.dependencies.context_builder = as_builder(
            FailingBuilder(run.dependencies.context_builder, want)
        )
    elif failure == "initial-turn":
        faults.turn_error = want
    await restore_test_run(run, faults)
    await run.start()
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(run.calls) == 0, "model request started before its prerequisites were committed"
    page = await store.items("session-1", BEFORE_FIRST, 100)
    turns = sum(1 for item in page.items if item.Kind == ItemKind.TURN)
    assert turns == 1, "failed recovery persisted an extra turn"

    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(resumed.calls) == 1, "retry did not deliver the saved result without redispatch"
    assert len(resumed.operations.adds) == 0, (
        "retry did not deliver the saved result without redispatch"
    )
    assert_stop_result(resumed.calls[0].request, "call-0", str(terminal))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, "retry did not settle delivery exactly once"
    assert len(resumed.calls) == 1, "retry did not settle delivery exactly once"


@synctest
async def test_coordinator_retries_initial_tool_status_persistence_failure() -> None:
    run = new_stop_test_run(1)
    run.store.history = run.store.history[:2]
    store = await persist_test_run(run)
    translator = SubmittingTranslator([value_spec(b"1")])
    registry = new_registry(StaticTranslators(view_image=translator), VIEW_IMAGE_NAME)
    run.dependencies.tools = registry
    want = RuntimeError("cannot persist initial tool status")
    faults = FailingStore(store)
    faults.status_error = want
    await restore_test_run(run, faults)
    await run.start()
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(translator.calls) == 1, (
        "failed status commit did not prevent dispatch and model requests"
    )
    assert len(run.operations.adds) == 0, (
        "failed status commit did not prevent dispatch and model requests"
    )
    assert len(run.calls) == 0, "failed status commit did not prevent dispatch and model requests"
    page = await store.items("session-1", BEFORE_FIRST, 100)
    assert len(page.items) == 2, f"history after failed commit = {page.items!r}"
    assert page.items[1].Kind == ItemKind.MODEL_RESPONSE, (
        f"history after failed commit = {page.items!r}"
    )

    resumed = new_stop_test_run(0)
    resumed.dependencies.tools = registry
    await restore_test_run(resumed, store)
    assert len(resumed.dependencies.restored.operations) == 0, (
        "failed status commit persisted operations"
    )
    await resumed.start()
    assert len(translator.calls) == 2, "retry did not translate and dispatch the unfinished call"
    assert len(resumed.operations.adds) == 1, (
        "retry did not translate and dispatch the unfinished call"
    )
    restored = await store.resume("session-1")
    assert restored.operations == resumed.operations.adds, (
        "dispatched operation differs from committed operation"
    )
    resumed.operations.send(replace(resumed.operations.adds[0], Status=Status.COMPLETED))
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(resumed.calls) == 1, (
        f"requests = {len(resumed.calls)}, want completed result delivery"
    )
    assert_stop_result(resumed.calls[0].request, "call-0", "")
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
