import pytest
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, ToolCall
from unreal_agent.operation import Status
from unreal_agent.tool import VIEW_IMAGE_NAME

from tests.coordinator._helpers import (
    MILLISECOND,
    NANOSECOND,
    SECOND,
    TOOL_CALL_RUN_GRACE_PERIOD_NS,
    assert_stop_result,
    external_event,
    fake_time,
    heartbeat_input,
    new_tool_grace_test_run,
    now_ns,
    sleep_ns,
    stop_input,
    text_response,
    tool_grace_response,
    update_tool_grace_call,
)


@pytest.mark.parametrize(
    "terminal", [Status.COMPLETED, Status.CANCELED], ids=lambda value: str(value)
)
@fake_time
async def test_coordinator_tool_grace_batches_completions_until_all_calls_finish(
    terminal: Status,
) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    deadline = now_ns() + SECOND
    await update_tool_grace_call(run, "A", terminal)
    assert run.request_count() == 1, "first completion was not persisted and deferred"
    assert len(run.store.appended_statuses) == 3, "first completion was not persisted and deferred"
    await sleep_ns(100 * MILLISECOND)
    await update_tool_grace_call(run, "B", terminal)
    assert run.request_count() == 2, "last completion did not end the grace period early"
    assert now_ns() < deadline, "last completion did not end the grace period early"
    assert_stop_result(run.calls[1].request, "A", str(terminal))
    assert_stop_result(run.calls[1].request, "B", str(terminal))
    await run.respond(1, text_response("Done."))
    await sleep_ns(deadline - now_ns() + SECOND)
    assert run.request_count() == 2, "expired grace period delivered results again"


@pytest.mark.parametrize("end_grace", ["steering", "expiry"])
@fake_time
async def test_coordinator_tool_grace_waits_only_for_latest_turn(end_grace: str) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run three tools"))
    await run.respond(0, tool_grace_response("A", "B", "C"))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    if end_grace == "steering":
        await run.input(external_event("steering", "run two more tools"))
    else:
        await sleep_ns(TOOL_CALL_RUN_GRACE_PERIOD_NS)
    await run.respond(1, tool_grace_response("D", "E"))
    deadline = now_ns() + TOOL_CALL_RUN_GRACE_PERIOD_NS
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    assert run.request_count() == 2, "older completion ended the new turn's grace period"
    await update_tool_grace_call(run, "D", Status.COMPLETED)
    assert run.request_count() == 2, "partial completion ended the new turn's grace period"
    await update_tool_grace_call(run, "E", Status.COMPLETED)
    assert run.request_count() == 3, (
        "older running call prevented the new turn's grace period from ending early"
    )
    assert now_ns() < deadline, (
        "older running call prevented the new turn's grace period from ending early"
    )
    for call_id in ["A", "B", "D", "E"]:
        assert_stop_result(run.calls[2].request, call_id, str(Status.COMPLETED))
    assert_stop_result(run.calls[2].request, "C", TOOL_CALL_RUNNING_PAYLOAD)
    await run.respond(2, text_response("Waiting for C."))
    await update_tool_grace_call(run, "C", Status.COMPLETED)
    assert run.request_count() == 4, (
        "older completion was deferred after the new turn's grace period ended"
    )
    assert_stop_result(run.calls[3].request, "C", str(Status.COMPLETED))


@fake_time
async def test_coordinator_tool_grace_deadline_does_not_reset_on_completion() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run three tools"))
    await run.respond(0, tool_grace_response("A", "B", "C"))
    deadline = now_ns() + SECOND
    await sleep_ns(100 * MILLISECOND)
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    await sleep_ns(400 * MILLISECOND)
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    await sleep_ns(deadline - now_ns() - NANOSECOND)
    assert run.request_count() == 1, "partial completions started a turn before the grace deadline"
    await sleep_ns(NANOSECOND)
    assert run.request_count() == 2, (
        "pending results were not delivered at the original grace deadline"
    )
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "C", TOOL_CALL_RUNNING_PAYLOAD)
    await sleep_ns(SECOND)
    assert run.request_count() == 2, "grace deadline repeated or interrupted the continuation"
    assert not run.calls[1].canceled, "grace deadline repeated or interrupted the continuation"


@fake_time
async def test_coordinator_tool_grace_expiry_without_results_does_not_start_turn() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    await sleep_ns(SECOND)
    assert run.request_count() == 1, "grace expiry started a turn without pending inputs"
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    assert run.request_count() == 2, "completion after grace expiry was deferred"
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", TOOL_CALL_RUNNING_PAYLOAD)


@pytest.mark.parametrize("kind", ["external", "heartbeat"])
@fake_time
async def test_coordinator_inbox_ends_tool_grace_period(kind: str) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    input = external_event("steering", "check progress")
    if kind == "heartbeat":
        input = heartbeat_input("heartbeat")
    await run.input(input)
    assert run.request_count() == 2, "inbox input did not end the tool grace period"
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", TOOL_CALL_RUNNING_PAYLOAD)
    await run.respond(1, text_response("Waiting."))
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    assert run.request_count() == 3, (
        "completion was deferred after inbox input ended the grace period"
    )
    assert_stop_result(run.calls[2].request, "B", str(Status.COMPLETED))


@fake_time
async def test_coordinator_tool_grace_discards_previous_deadline() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    old_deadline = now_ns() + SECOND
    await sleep_ns(250 * MILLISECOND)
    await run.input(external_event("steering", "run another tool"))
    await run.respond(1, tool_grace_response("C"))
    new_deadline = now_ns() + SECOND
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    await sleep_ns(old_deadline - now_ns() + NANOSECOND)
    assert run.request_count() == 2, "previous deadline ended the new grace period"
    await sleep_ns(new_deadline - now_ns())
    assert run.request_count() == 3, "new grace deadline did not deliver pending results"
    assert_stop_result(run.calls[2].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[2].request, "B", TOOL_CALL_RUNNING_PAYLOAD)
    assert_stop_result(run.calls[2].request, "C", TOOL_CALL_RUNNING_PAYLOAD)


@pytest.mark.parametrize("name", [VIEW_IMAGE_NAME, "unavailable"])
@fake_time
async def test_coordinator_immediate_tool_status_bypasses_grace(name: str) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run both tools"))
    response = tool_grace_response("A")
    assert response.Output is not None
    response.Output.append(
        LLMItem(
            Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="immediate", Name=name, Arguments="{}")
        )
    )
    await run.respond(0, response)
    assert run.request_count() == 2, (
        "immediate tool status was deferred or prevented valid work from dispatching"
    )
    assert len(run.operations.adds) == 1, (
        "immediate tool status was deferred or prevented valid work from dispatching"
    )
    want = ""
    if name == "unavailable":
        want = 'tool "unavailable" is not available'
    assert_stop_result(run.calls[1].request, "immediate", want)
    assert_stop_result(run.calls[1].request, "A", TOOL_CALL_RUNNING_PAYLOAD)


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE], ids=lambda value: str(value)
)
@fake_time
async def test_coordinator_stop_during_tool_grace_period(mode: ControlMode) -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("input", "run tool"))
    await run.respond(0, tool_grace_response("A"))
    await run.input(stop_input("stop", mode))
    terminal = Status.CANCELED
    if mode == ControlMode.STOP_WHEN_IDLE:
        terminal = Status.COMPLETED
        assert len(run.operations.cancels) == 0, "when-idle stop canceled pending work"
    else:
        assert len(run.operations.cancels) == 1, (
            "stop waited for grace expiry before canceling work"
        )
    assert run.request_count() == 1, "stop started a turn before the operation finished"
    await update_tool_grace_call(run, "A", terminal)
    if mode != ControlMode.STOP_HARD:
        assert run.request_count() == 2, (
            "when_idle waited for grace expiry before delivering the result"
        )
        assert_stop_result(run.calls[1].request, "A", str(terminal))
        await run.respond(1, text_response("Done."))
    else:
        assert run.request_count() == 1, "hard stop started a final turn"
    run.assert_stopped()
