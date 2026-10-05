from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.coordinator import CoordinatorError
from unreal_agent.inbox import ClosedError, ControlMode, Inbox
from unreal_agent.operation import Operation, Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import ItemKind, ModelResponse, ToolCallStatus

from tests.coordinator._helpers import (
    HOUR,
    MINUTE,
    NANOSECOND,
    SECOND,
    SLURP_IDLE_TIMEOUT_NS,
    advance_heartbeat_time,
    assert_completed_results,
    assert_heartbeat_count,
    assert_stop_result,
    caused_by,
    count_heartbeat_messages,
    external_event,
    fake_time,
    heartbeat_input,
    latest_heartbeat_reason,
    new_heartbeat_test_run,
    now_ns,
    stop_input,
    stored_item,
    text_response,
)


@fake_time
async def test_coordinator_heartbeats_while_waiting_for_tools() -> None:
    run = new_heartbeat_test_run(2)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    await advance_heartbeat_time(30 * SECOND)
    await run.update(0, Status.AWAITING)
    await advance_heartbeat_time(30 * SECOND - (2 * SLURP_IDLE_TIMEOUT_NS) - NANOSECOND)
    assert run.request_count() == 0, "heartbeat arrived before the interval elapsed"
    await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)
    want_reason = (
        "Heartbeat: waited 60 seconds for tool calls.\nRunning: "
        '[{"CallID":"call-0","Name":"ViewImage","Arguments":"{}"},'
        '{"CallID":"call-1","Name":"ViewImage","Arguments":"{}"}]'
    )
    reason = latest_heartbeat_reason(run)
    assert reason == want_reason, f"heartbeat reason = {reason!r}, want {want_reason!r}"
    assert run.request_count() == 1, (
        "heartbeat did not produce one model request with a check-in message"
    )
    assert count_heartbeat_messages(run.calls[0].request) == 1, (
        "heartbeat did not produce one model request with a check-in message"
    )
    await advance_heartbeat_time(3 * MINUTE)
    assert_heartbeat_count(run, 1)
    assert run.request_count() == 1, "heartbeat interrupted or overlapped an active model request"
    assert not run.calls[0].canceled, "heartbeat interrupted or overlapped an active model request"
    await run.respond(0, text_response("Still waiting."))
    await advance_heartbeat_time(MINUTE - NANOSECOND)
    assert_heartbeat_count(run, 1)
    await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 2)
    assert run.request_count() == 2, "next heartbeat did not append exactly one check-in"
    assert count_heartbeat_messages(run.calls[1].request) == 2, (
        "next heartbeat did not append exactly one check-in"
    )
    await run.update(0, Status.COMPLETED)
    await run.update(1, Status.COMPLETED)
    assert run.request_count() == 2, "tool completion interrupted a heartbeat response"
    assert not run.calls[1].canceled, "tool completion interrupted a heartbeat response"
    await run.respond(1, text_response("Waiting for results."))
    assert run.request_count() == 3, "completed results were not delivered after the model response"
    assert_completed_results(run.calls[2].request, 2)
    await run.respond(2, text_response("Done."))
    await advance_heartbeat_time(3 * MINUTE)
    assert_heartbeat_count(run, 2)
    assert run.request_count() == 3, (
        "heartbeats continued without pending calls or canceled operations"
    )
    assert len(run.operations.cancels) == 0, (
        "heartbeats continued without pending calls or canceled operations"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()


@pytest.mark.parametrize(
    ("pending", "interval"),
    [
        pytest.param(1, 0, id="disabled"),
        pytest.param(0, SECOND, id="idle"),
    ],
)
@fake_time
async def test_coordinator_heartbeat_disabled_or_idle(pending: int, interval: int) -> None:
    run = new_heartbeat_test_run(pending)
    run.dependencies.tool_heartbeat_interval = interval / SECOND
    await run.start()
    await advance_heartbeat_time(HOUR)
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 0, "unexpected model request"


@fake_time
async def test_coordinator_heartbeat_yields_to_steering_and_results() -> None:
    run = new_heartbeat_test_run(2)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    await advance_heartbeat_time(30 * SECOND)
    await run.input(external_event("steering", "check progress"))
    await advance_heartbeat_time(MINUTE)
    assert_heartbeat_count(run, 0)
    await run.respond(0, text_response("Waiting."))
    await advance_heartbeat_time(30 * SECOND)
    await run.update(0, Status.COMPLETED)
    assert run.request_count() == 2, "result waited for a heartbeat"
    await run.respond(1, text_response("One remains."))
    await advance_heartbeat_time(MINUTE - NANOSECOND)
    assert_heartbeat_count(run, 0)
    await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)
    want_reason = (
        "Heartbeat: waited 60 seconds for tool calls.\nRunning: "
        '[{"CallID":"call-1","Name":"ViewImage","Arguments":"{}"}]'
    )
    reason = latest_heartbeat_reason(run)
    assert reason == want_reason, (
        f"heartbeat reason = {reason!r}, want only the remaining call: {want_reason!r}"
    )


@fake_time
async def test_heartbeat_discards_previous_deadline() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    old_deadline = now_ns() + MINUTE
    await advance_heartbeat_time(20 * SECOND)
    await run.input(external_event("steering", "continue"))
    await run.respond(0, text_response("Waiting."))
    new_deadline = now_ns() + MINUTE
    await advance_heartbeat_time(old_deadline - now_ns() + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 1, "discarded deadline started a turn"
    await advance_heartbeat_time(new_deadline - now_ns() + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)


@fake_time
async def test_heartbeat_interval_shorter_than_inbox_batch() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = NANOSECOND / SECOND
    await run.start()
    for index in range(3):
        await advance_heartbeat_time(NANOSECOND + (2 * SLURP_IDLE_TIMEOUT_NS))
        assert_heartbeat_count(run, index + 1)
        assert run.request_count() == index + 1, "heartbeat prevented inbox delivery"
        await advance_heartbeat_time(SECOND)
        assert_heartbeat_count(run, index + 1)
        await run.respond(index, text_response("Waiting."))
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    await run.update(0, Status.CANCELED)
    run.assert_stopped()


@fake_time
async def test_queued_heartbeat_preserves_active_model_response() -> None:
    run = new_heartbeat_test_run(1)
    await run.start()
    await run.input(external_event("steering", "check progress"))
    await run.input(heartbeat_input("queued-heartbeat"))
    assert run.request_count() == 1, "queued heartbeat interrupted the model"
    assert not run.calls[0].canceled, "queued heartbeat interrupted the model"
    await run.respond(0, text_response("Working."))
    assert run.request_count() == 2, "queued heartbeat was not delivered in the next turn"
    assert count_heartbeat_messages(run.calls[1].request) == 1, (
        "queued heartbeat was not delivered in the next turn"
    )
    await run.respond(1, text_response("Waiting."))
    await advance_heartbeat_time(HOUR)
    assert run.request_count() == 2, "queued heartbeat was delivered twice"


@fake_time
async def test_heartbeat_waits_for_all_operations_of_call() -> None:
    run = new_heartbeat_test_run(2)
    response = cast(ModelResponse, run.store.history[1].Data)
    run.store.history[1].Data = replace(
        response, Response=replace(response.Response, Output=(response.Response.Output or [])[:1])
    )
    status = cast(ToolCallStatus, run.store.history[2].Data)
    operations = list(run.store.resume.operations)
    run.store.history[2].Data = replace(
        status,
        Operations=operations,
        Status=replace(status.Status, WaitingFor=[operations[0].ID, operations[1].ID]),
    )
    run.store.history = run.store.history[:3]
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    deadline = now_ns() + MINUTE
    await advance_heartbeat_time(30 * SECOND)
    await run.update(0, Status.COMPLETED)
    assert run.request_count() == 0, "partial completion started a turn"
    await advance_heartbeat_time(deadline - now_ns() + (2 * SLURP_IDLE_TIMEOUT_NS))
    assert_heartbeat_count(run, 1)
    await run.respond(0, text_response("Waiting."))
    await run.update(1, Status.COMPLETED)
    assert run.request_count() == 2, "last operation did not complete the call"
    assert_stop_result(run.calls[1].request, "call-0", "completed,completed")
    await run.respond(1, text_response("Done."))
    await advance_heartbeat_time(MINUTE)
    assert_heartbeat_count(run, 1)


@fake_time
async def test_coordinator_heartbeat_stops_during_cancellation() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = MINUTE / SECOND
    await run.start()
    await advance_heartbeat_time(30 * SECOND)
    await run.input(stop_input("stop", ControlMode.STOP_HARD))
    await advance_heartbeat_time(3 * MINUTE)
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 0, "heartbeat ran during cancellation"
    await run.update(0, Status.CANCELED)
    run.assert_stopped()


@fake_time
async def test_heartbeat_control_preserves_when_idle_stop() -> None:
    run = new_heartbeat_test_run(1)
    await run.start()
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE), heartbeat_input("heartbeat"))
    assert run.current._stop.request.Mode == ControlMode.STOP_WHEN_IDLE, (
        "heartbeat changed the stop or failed to wake the model"
    )
    assert run.request_count() == 1, "heartbeat changed the stop or failed to wake the model"
    await run.respond(0, text_response("Waiting."))
    await run.update(0, Status.COMPLETED)
    await run.respond(1, text_response("Done."))
    run.assert_stopped()


@fake_time
async def test_coordinator_heartbeat_propagates_submission_failure() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = SECOND / SECOND
    stopped_inbox = Inbox()
    stopped_inbox.close()

    def swap_inbox(_: Operation) -> None:
        run.dependencies.inbox = stopped_inbox

    run.store.on_save_operation = swap_inbox
    await run.start()
    await run.update(0, Status.AWAITING)

    await advance_heartbeat_time(SECOND)
    err = run.finished_error()
    assert caused_by(err, ClosedError), f"Run error = {err!r}, want {ClosedError.__name__}"
    assert_heartbeat_count(run, 0)
    assert run.request_count() == 0, "model ran after heartbeat submission failed"


@fake_time
async def test_coordinator_heartbeat_requires_persistence() -> None:
    run = new_heartbeat_test_run(1)
    run.dependencies.tool_heartbeat_interval = SECOND / SECOND
    want = RuntimeError("heartbeat storage failed")
    run.store.append_input_error = want
    await run.start()
    await advance_heartbeat_time(SECOND)
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert run.request_count() == 0, "model ran before heartbeat was persisted"


@pytest.mark.parametrize("stage", ["input", "turn", "response"])
@fake_time
async def test_coordinator_replays_heartbeat(stage: str) -> None:
    run = new_heartbeat_test_run(1)
    run.store.history.append(stored_item(4, ItemKind.INPUT, heartbeat_input("heartbeat-1")))
    if stage != "input":
        run.store.history.append(
            stored_item(5, ItemKind.TURN, Turn(ID="heartbeat-turn", Type=TurnType.REGULAR))
        )
    if stage == "response":
        run.store.history.append(
            stored_item(
                6,
                ItemKind.MODEL_RESPONSE,
                ModelResponse(TurnID="heartbeat-turn", Response=text_response("Waiting.")),
            )
        )
    await run.start()
    built = run.dependencies.context_builder.build()
    assert count_heartbeat_messages(built.request) == 1, "heartbeat was not replayed exactly once"
    if stage == "response":
        assert run.request_count() == 0, "delivered heartbeat woke the model again on resume"
    else:
        assert run.request_count() == 1, "undelivered heartbeat was not resumed"
        assert built.request == run.calls[0].request, "undelivered heartbeat was not resumed"
    assert_heartbeat_count(run, 0)


async def test_coordinator_rejects_negative_heartbeat_interval() -> None:
    run = new_heartbeat_test_run(0)
    run.dependencies.tool_heartbeat_interval = -SECOND / SECOND
    with pytest.raises(CoordinatorError, match="heartbeat interval"):
        await run.current.run()
