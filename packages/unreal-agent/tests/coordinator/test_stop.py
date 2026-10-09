import asyncio
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Message, Request, Response, ToolCall
from unreal_agent.operation import Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import ItemKind, ModelResponse, ToolCallStatus
from unreal_agent.tool import VIEW_IMAGE_NAME

from tests.coordinator._helpers import (
    SLURP_IDLE_TIMEOUT_NS,
    FakeAdapter,
    assert_completed_results,
    assert_stop_result,
    caused_by,
    external_event,
    fake_time,
    heartbeat_input,
    new_stop_test_run,
    sleep_ns,
    stop_input,
    stop_test_adapter,
    stored_item,
    submit_test_input,
    text_response,
    wait,
)


@fake_time
async def test_coordinator_remains_available_until_stop() -> None:
    run = new_stop_test_run(0)
    await run.start()
    run.assert_running()
    for index in range(2):
        await run.input(external_event(f"input-{index}", "hello"))
        await run.respond(index, text_response("hello"))
        run.assert_running()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()
    assert len(run.calls) == 2, f"requests = {len(run.calls)}, want 2"


@pytest.mark.parametrize(
    ("terminal", "during_request"),
    [
        pytest.param(terminal, during, id=f"{terminal}/during_request={str(during).lower()}")
        for terminal, during in (
            (Status.COMPLETED, False),
            (Status.COMPLETED, True),
            (Status.CANCELED, True),
        )
    ],
)
@fake_time
async def test_coordinator_when_idle_delivers_pending_results(
    terminal: Status, during_request: bool
) -> None:
    run = new_stop_test_run(1)
    await run.start()
    await run.input(external_event("progress", "check progress"))
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    assert len(run.calls) == 1, "when_idle interrupted the model request"
    assert not run.calls[0].canceled, "when_idle interrupted the model request"
    if during_request:
        await run.update(0, terminal)
    await run.respond(0, text_response("The check is still running."))
    run.assert_running()
    if not during_request:
        assert len(run.calls) == 1, "requested model response before tool completed"
        await run.update(0, terminal)
    assert len(run.calls) == 2, f"requests = {len(run.calls)}, want 2"
    assert_stop_result(run.calls[1].request, "call-0", str(terminal))
    await run.respond(1, text_response("All results received."))
    run.assert_stopped()
    assert len(run.operations.cancels) == 0, "when_idle canceled an operation"


@fake_time
async def test_coordinator_stops_after_cancellation_is_recorded() -> None:
    run = new_stop_test_run(2)
    await run.start()
    await run.input(external_event("progress", "check progress"))
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    assert run.calls[0].canceled, "stop did not interrupt active model request"
    assert len(run.operations.cancels) == 2, (
        f"cancellations = {run.operations.cancels}, want both operations"
    )
    for reason in run.operations.cancel_reasons:
        assert reason == "user requested stop", f"cancellation reason = {reason!r}"
    run.assert_running()
    await run.update(0, Status.CANCELED)
    run.assert_running()
    await run.update(1, Status.COMPLETED)
    assert len(run.store.saved_operations) == 2, (
        "stop did not persist terminal updates and tool results"
    )
    assert len(run.store.appended_statuses) == 2, (
        "stop did not persist terminal updates and tool results"
    )
    assert len(run.calls) == 1, "hard stop started another request"
    run.assert_stopped()
    assert len(run.operations.cancels) == 2, "operation cancellation was requested more than once"


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, "steer"], ids=[ControlMode.STOP_HARD, "steer"]
)
@fake_time
async def test_coordinator_cancels_compaction_without_recording_a_response(mode: str) -> None:
    run = new_stop_test_run(0)
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("first", "hello"))
    run.current._state.current_turn_type = TurnType.COMPACTION
    if mode == "steer":
        await run.input(external_event("second", "change course"))
    else:
        await run.input(stop_input("stop", mode))
    want_type = TurnType.REGULAR
    if mode == ControlMode.STOP_HARD:
        want_type = TurnType.COMPACTION
    assert run.calls[0].canceled, (
        "cancellation did not preserve the recorded turn kind until the next turn"
    )
    assert run.current._state.current_turn_type == want_type, (
        "cancellation did not preserve the recorded turn kind until the next turn"
    )
    await run.respond(0, text_response("Late summary"))
    assert len(run.store.appended_responses) == 0, (
        "canceled compaction recorded a response or delivered input"
    )
    assert run.current._state.delivered_inputs == 0, (
        "canceled compaction recorded a response or delivered input"
    )
    if mode != ControlMode.STOP_HARD:
        assert len(run.calls) == 2, "late response affected the replacement request"
        assert run.current._cancel_model is not None, (
            "late response affected the replacement request"
        )
        await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
        await run.respond(1, text_response("Done"))
    run.assert_stopped()


@fake_time
async def test_coordinator_hard_stop_overrides_when_idle() -> None:
    run = new_stop_test_run(1)
    await run.start()
    await run.input(external_event("first", "hello"))
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    assert not run.calls[0].canceled, "when_idle interrupted the model"
    await run.input(stop_input("hard", ControlMode.STOP_HARD))
    await run.input(
        stop_input("idle-again", ControlMode.STOP_WHEN_IDLE),
        stop_input("hard-again", ControlMode.STOP_HARD),
        heartbeat_input("heartbeat"),
        external_event("late", "more work"),
    )
    run.assert_running()
    assert run.current._stop.request.Mode == ControlMode.STOP_HARD, (
        "later inputs changed the stop or repeated cancellation"
    )
    assert len(run.operations.cancels) == 1, (
        "later inputs changed the stop or repeated cancellation"
    )
    assert len(run.calls) == 1, (
        "hard stop did not interrupt the model or later input started another request"
    )
    assert run.calls[0].canceled, (
        "hard stop did not interrupt the model or later input started another request"
    )
    await run.update(0, Status.CANCELED)
    run.assert_stopped()
    assert len(run.operations.cancels) == 1, (
        "stop repeated cancellation or started another model request"
    )
    assert len(run.calls) == 1, "stop repeated cancellation or started another model request"


@fake_time
async def test_coordinator_hard_stop_discards_late_model_response() -> None:
    run = new_stop_test_run(1)
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("first", "hello"))
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    run.assert_running()
    await run.respond(
        0,
        Response(
            Output=[
                LLMItem(
                    Type=ItemType.TOOL_CALL,
                    Data=ToolCall(CallID="late", Name=VIEW_IMAGE_NAME, Arguments="{}"),
                )
            ]
        ),
    )
    assert len(run.store.appended_responses) == 0, (
        "late response was recorded or translated during cancellation"
    )
    assert len(run.store.appended_statuses) == 0, (
        "late response was recorded or translated during cancellation"
    )
    assert len(run.calls) == 1, "late response was recorded or translated during cancellation"
    await run.update(0, Status.CANCELED)
    run.assert_stopped()


@pytest.mark.parametrize("failure", ["input", "cancel", "operation", "tool-result"])
@fake_time
async def test_coordinator_stop_propagates_errors(failure: str) -> None:
    run = new_stop_test_run(2)
    want = RuntimeError("test failure")
    match failure:
        case "input":
            run.store.append_input_error = want
        case "cancel":
            run.operations.cancel_error = want
        case "operation":
            run.store.save_operation_error = want
        case "tool-result":
            run.store.append_status_error = want
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    if failure in ("operation", "tool-result"):
        await run.update(0, Status.CANCELED)
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    if failure == "cancel":
        assert len(run.operations.cancels) == 2, (
            "cancellation error prevented attempting the other operation"
        )


@pytest.mark.parametrize("source", ["inbox", "operation updates"])
@fake_time
async def test_coordinator_cancellation_while_collecting_updates(source: str) -> None:
    run = new_stop_test_run(1)
    run.go()
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await wait()
    if source == "inbox":
        await submit_test_input(run.inputs, external_event("input", "hello"))
    else:
        run.operations.send(replace(run.store.resume.operations[0], Status=Status.COMPLETED))
    await wait()
    run.cancel("caller disconnected")
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want collection cancellation cause"
    )
    want_inputs, want_updates = 1, 0
    if source == "operation updates":
        want_inputs, want_updates = 0, 1
    assert len(run.store.appended_inputs) == want_inputs, (
        "selected event was not persisted before draining, "
        "or canceled draining started a model request"
    )
    assert len(run.store.saved_operations) == want_updates, (
        "selected event was not persisted before draining, "
        "or canceled draining started a model request"
    )
    assert len(run.calls) == 0, (
        "selected event was not persisted before draining, "
        "or canceled draining started a model request"
    )


@fake_time
async def test_coordinator_drains_inbox_and_operations_before_calling_model() -> None:
    run = new_stop_test_run(1)
    await run.start()
    await submit_test_input(run.inputs, external_event("input", "check completed work"))
    run.operations.send(replace(run.store.resume.operations[0], Status=Status.COMPLETED))
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await wait()

    assert len(run.calls) == 1, (
        f"model requests = {len(run.calls)}, want one request containing both events"
    )
    assert_completed_results(run.calls[0].request, 1)
    found_input = False
    for item in run.calls[0].request.Input:
        if (
            item.Type == ItemType.MESSAGE
            and cast(Message, item.Data).Text == "check completed work"
        ):
            found_input = True
    assert found_input, (
        "request omitted the input, or selected and drained events were not persisted exactly once"
    )
    assert len(run.store.appended_inputs) == 1, (
        "request omitted the input, or selected and drained events were not persisted exactly once"
    )
    assert len(run.store.saved_operations) == 1, (
        "request omitted the input, or selected and drained events were not persisted exactly once"
    )


@fake_time
async def test_coordinator_cancellation_takes_precedence_over_model_error() -> None:
    run = new_stop_test_run(0)
    results: asyncio.Future[Exception] = asyncio.get_running_loop().create_future()

    async def respond(request: Request) -> Response:
        raise await results

    run.dependencies.llm = FakeAdapter(respond)
    run.go()
    await wait()
    await run.input(external_event("input", "hello"))
    cancel_model = run.current._cancel_model
    assert cancel_model is not None, "model request did not start"

    def cancel_run_and_model() -> object:
        run.cancel()
        return cancel_model()

    run.current._cancel_model = cancel_run_and_model
    results.set_result(RuntimeError("model failed"))
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), (
        f"Run error = {err!r}, want context cancellation"
    )
    assert len(run.store.appended_responses) == 0, "failed model response was persisted"


@pytest.mark.parametrize(
    "mode",
    [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE],
    ids=[ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE],
)
@fake_time
async def test_coordinator_stop_controls_do_not_repeat_on_resume(mode: ControlMode) -> None:
    run = new_stop_test_run(0)
    run.store.history.append(stored_item(3, ItemKind.INPUT, stop_input("old-stop", mode)))
    await run.start()
    run.assert_running()
    await run.input(external_event("new", "continue"))
    await run.respond(0, text_response("Continuing."))
    run.assert_running()
    await run.input(stop_input("new-stop", ControlMode.STOP_HARD))
    run.assert_stopped()


@pytest.mark.parametrize("stage", ["completed", "request-started", "response-recorded"])
@fake_time
async def test_coordinator_when_idle_delivers_restored_results(stage: str) -> None:
    run = new_stop_test_run(1)
    status = cast(ToolCallStatus, run.store.history[2].Data)
    status.Operations[0] = replace(status.Operations[0], Status=Status.COMPLETED)
    run.store.history[2].Data = status
    run.store.resume.operations = []
    run.dependencies.restored = run.store.resume
    if stage != "completed":
        run.store.history.append(
            stored_item(
                4,
                ItemKind.TURN,
                Turn(ID="delivered", PreviousTurnID="turn-1", Type=TurnType.REGULAR),
            )
        )
    if stage == "response-recorded":
        run.store.history.append(
            stored_item(
                5,
                ItemKind.MODEL_RESPONSE,
                ModelResponse(TurnID="delivered", Response=text_response("Done.")),
            )
        )
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    if stage != "response-recorded":
        assert len(run.calls) == 1, f"requests = {len(run.calls)}, want restored result delivery"
        assert_stop_result(run.calls[0].request, "call-0", str(Status.COMPLETED))
        await run.respond(0, text_response("Done."))
    else:
        assert len(run.calls) == 0, "already delivered result caused another turn"
    run.assert_stopped()
