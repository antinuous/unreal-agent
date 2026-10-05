import asyncio
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.contextbuilder import new_builder
from unreal_agent.coordinator import CoordinatorError
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Request, Response, Role, ToolCall
from unreal_agent.operation import Operation, Status
from unreal_agent.sessionstore import ToolCallStatus
from unreal_agent.tool import BASH_NAME, StaticTranslators, new_registry

from tests.coordinator._helpers import (
    SLURP_IDLE_TIMEOUT_NS,
    FakeAdapter,
    FakeOperationManager,
    StopTestCall,
    SubmittingResultErrorTranslator,
    assert_completed_results,
    caused_by,
    external_event,
    fake_time,
    heartbeat_input,
    independent_tool_calls,
    new_stop_test_run,
    new_test_coordinator,
    new_test_inbox,
    new_tool_grace_test_run,
    sleep_ns,
    stop_input,
    submit_test_input,
    text_response,
    tool_grace_response,
    update_tool_grace_call,
    value_spec,
    wait,
)


@fake_time
async def test_coordinator_run_returns_dispatch_error_for_new_operation() -> None:
    run = new_tool_grace_test_run()
    want = RuntimeError("dispatch failed")
    run.operations.add_error = lambda _: want
    await run.start()
    await run.input(external_event("input", "run tool"))
    await run.respond(0, tool_grace_response("A"))
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(run.calls) == 1, (
        "dispatch failure started another turn or failed to preserve the committed operation"
    )
    assert len(run.store.appended_statuses) == 1, (
        "dispatch failure started another turn or failed to preserve the committed operation"
    )
    assert len(run.operations.adds) == 1, (
        "dispatch failure started another turn or failed to preserve the committed operation"
    )
    assert run.store.appended_statuses[0].Operations == run.operations.adds, (
        "dispatched operation differs from its committed state"
    )


@fake_time
async def test_coordinator_dispatches_only_new_tool_operations() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("first", "run tool"))
    await run.respond(0, tool_grace_response("A"))
    await run.input(external_event("second", "run another tool"))
    await run.respond(1, tool_grace_response("B"))
    await update_tool_grace_call(run, "A", Status.AWAITING)
    await run.input(heartbeat_input("heartbeat"))
    await run.respond(2, text_response("Waiting."))
    assert len(run.operations.adds) == 2, (
        f"operation dispatches = {run.operations.adds!r}, want each new operation once"
    )
    assert run.operations.adds[0].ID != run.operations.adds[1].ID, (
        f"operation dispatches = {run.operations.adds!r}, want each new operation once"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    await run.respond(3, text_response("A completed."))
    await run.respond(4, text_response("B completed."))
    run.assert_stopped()
    assert len(run.operations.adds) == 2, "result delivery redispatched operations"


@fake_time
async def test_coordinator_scheduling_result_failure_prevents_dispatch() -> None:
    run = new_stop_test_run(0)
    want = RuntimeError("cannot translate result")
    translator = SubmittingResultErrorTranslator(want, [value_spec()])
    run.dependencies.tools = new_registry(StaticTranslators(bash=translator), BASH_NAME)

    async def respond(request: Request) -> Response:
        return Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="call-1", Name=BASH_NAME, Arguments="{}"),
                )
            ]
        )

    adapter = FakeAdapter(respond)
    run.dependencies.llm = adapter
    await run.start()
    await run.input(external_event("input", "run it"))
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}, want result translation error"
    assert len(translator.calls) == 1, "model response was not persisted and translated"
    assert len(run.store.appended_responses) == 1, "model response was not persisted and translated"
    assert len(run.store.appended_statuses) == 0, (
        "failed result translation committed a status, dispatched work, or started another request"
    )
    assert len(run.operations.adds) == 0, (
        "failed result translation committed a status, dispatched work, or started another request"
    )
    assert len(adapter.requests) == 1, (
        "failed result translation committed a status, dispatched work, or started another request"
    )


async def test_coordinator_reconciliation_rejects_untranslated_call() -> None:
    run = new_stop_test_run(1)
    run.store.history = run.store.history[:2]
    await run.current._load_history()
    statuses: list[ToolCallStatus] = []
    with pytest.raises(CoordinatorError) as caught:
        statuses = await run.current._reconcile_tool_calls()
    assert str(caught.value) == 'reconcile untranslated tool call "call-0" in turn "turn-1"', (
        f"reconciliation error = {caught.value!r}, want untranslated-call error"
    )
    assert len(statuses) == 0, "reconciliation completed an untranslated call"
    assert len(run.store.appended_statuses) == 0, "reconciliation completed an untranslated call"
    assert len(run.current._state.tool_calls) == 1, "reconciliation completed an untranslated call"
    assert run.current._pending_inputs() == 0, "reconciliation completed an untranslated call"


@pytest.mark.parametrize(
    ("start_with_input", "steer"),
    [(False, False), (False, True), (True, False), (True, True)],
    ids=[
        "input=false/steer=false",
        "input=false/steer=true",
        "input=true/steer=false",
        "input=true/steer=true",
    ],
)
@fake_time
async def test_coordinator_run_defers_completions_until_model_finishes(
    start_with_input: bool, steer: bool
) -> None:
    store, registry = independent_tool_calls(3)
    inputs = new_test_inbox()
    operations = FakeOperationManager()
    calls: list[StopTestCall] = []

    async def respond(request: Request) -> Response:
        task = asyncio.current_task()
        assert task is not None
        call = StopTestCall(task, request, asyncio.get_running_loop().create_future())
        calls.append(call)
        return await call.response

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(store, inputs, operations, new_builder(), registry, adapter)
    run_task = asyncio.create_task(current.run())
    try:

        async def finish_operation(index: int) -> None:
            operations.send(replace(store.resume.operations[index], Status=Status.COMPLETED))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
            assert len(store.appended_statuses) == index + 1, (
                f"completed tool statuses = {len(store.appended_statuses)}, want {index + 1}"
            )
            status = store.appended_statuses[index]
            assert status.CallID == f"call-{index}", f"completed call = {status.CallID!r}"

        first_pending = 0
        if start_with_input:
            await submit_test_input(inputs, external_event("progress", "check progress"))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        else:
            await finish_operation(0)
            first_pending = 1
        assert len(calls) == 1, f"model requests = {len(calls)}, want 1"
        first = calls[0]
        for index in range(first_pending, 3):
            await finish_operation(index)
            assert not first.canceled, f"operation {index} canceled active model request"
            assert len(calls) == 1, f"model requests = {len(calls)}, want 1"
        response = Response(
            ID="progress-response",
            Output=[
                LLMItem(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="progress"))
            ],
        )
        if steer:
            await submit_test_input(inputs, external_event("steering", "summarize results"))
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        else:
            first.response.set_result(response)
            await wait()
            await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        assert len(calls) == 2, f"model requests = {len(calls)}, want 2"
        next = calls[1]
        assert_completed_results(next.request, 3)
        if steer:
            assert first.canceled, "steering did not cancel active model request"
            last = next.request.Input[-1]
            assert last.Type == ItemType.MESSAGE, f"steering missing from request: {next.request!r}"
            assert cast(Message, last.Data).Text == "summarize results", (
                f"steering missing from request: {next.request!r}"
            )
        next.response.set_result(Response(ID="final-response"))
        await wait()
        assert len(calls) == 2, (
            f"model requests = {len(calls)}, turns = {len(store.appended_turns)}, want 2 each"
        )
        assert len(store.appended_turns) == 2, (
            f"model requests = {len(calls)}, turns = {len(store.appended_turns)}, want 2 each"
        )
        if not steer:
            assert store.appended_responses[0].Response == response, (
                f"active model response was not preserved: {store.appended_responses!r}"
            )
        run_task.cancel()
        await wait()
        assert run_task.cancelled(), f"Run error = {run_task!r}, want context cancellation"
    finally:
        run_task.cancel()


@fake_time
async def test_coordinator_run_slurps_independent_operation_completions() -> None:
    store, registry = independent_tool_calls(3)
    operations = FakeOperationManager()
    for value in store.resume.operations:
        operations.send(replace(value, Status=Status.COMPLETED))

    async def respond(request: Request) -> Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    current = new_test_coordinator(
        store, new_test_inbox(), operations, new_builder(), registry, adapter
    )
    run_task = asyncio.create_task(current.run())
    try:
        await wait()
        await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
        requests = adapter.requests
        assert len(requests) == 1, f"model requests = {len(requests)}, want 1"
        assert_completed_results(requests[0], 3)
        assert len(store.saved_operations) == 3, (
            f"completion effects: operations={len(store.saved_operations)} "
            f"statuses={len(store.appended_statuses)} turns={len(store.appended_turns)}"
        )
        assert len(store.appended_statuses) == 3, (
            f"completion effects: operations={len(store.saved_operations)} "
            f"statuses={len(store.appended_statuses)} turns={len(store.appended_turns)}"
        )
        assert len(store.appended_turns) == 1, (
            f"completion effects: operations={len(store.saved_operations)} "
            f"statuses={len(store.appended_statuses)} turns={len(store.appended_turns)}"
        )
        run_task.cancel()
        await wait()
        assert run_task.cancelled(), f"Run error = {run_task!r}, want context cancellation"
    finally:
        run_task.cancel()


async def test_coordinator_run_persists_completed_updates_before_closure() -> None:
    store, registry = independent_tool_calls(3)
    operations = FakeOperationManager()
    completed: list[Operation] = []
    for value in store.resume.operations:
        value = replace(value, Status=Status.COMPLETED)
        completed.append(value)
        operations.send(value)
    operations.close()

    async def respond(request: Request) -> Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    adapter = FakeAdapter(respond)
    builder = new_builder()
    current = new_test_coordinator(store, new_test_inbox(), operations, builder, registry, adapter)
    with pytest.raises(CoordinatorError) as caught:
        await current.run()
    assert str(caught.value) == "operation updates closed", (
        f"Run error = {caught.value!r}, want closed operation updates error"
    )
    assert store.saved_operations == completed, (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    assert len(store.appended_statuses) == 3, (
        f"completion effects: operations={store.saved_operations!r} "
        f"statuses={store.appended_statuses!r}"
    )
    built = builder.build()
    assert_completed_results(built.request, 3)
