import asyncio
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent import gojson
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Item as LLMItem
from unreal_agent.llm import ItemType, Reasoning, Response, ToolCall, ToolResult
from unreal_agent.operation import Status
from unreal_agent.tool import BASH_NAME, VIEW_IMAGE_NAME, StaticTranslators, new_registry

from tests.coordinator._helpers import (
    SECOND,
    SLURP_IDLE_TIMEOUT_NS,
    FailingStore,
    OperationStatusTranslator,
    SubmissionTranslator,
    assert_completed_results,
    assert_stop_result,
    caused_by,
    external_event,
    fake_time,
    new_stop_test_run,
    persist_test_run,
    restore_test_run,
    sleep_ns,
    stop_input,
    text_response,
    text_result,
    user_message,
    value_spec,
    wait,
)

COMPLETED = str(Status.COMPLETED)


@pytest.mark.parametrize(
    "stage", ["in flight", "response committed", "turn write failed", "response write failed"]
)
@fake_time
async def test_coordinator_submission_boundary_survives_recovery(stage: str) -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    faults = FailingStore(store)
    await restore_test_run(run, faults)
    run.go()
    await wait()
    await run.update(1, Status.COMPLETED)
    assert len(run.calls) == 1, "completion before submission was not included immediately"
    assert run.current._state.current_turn_inputs == 1, (
        "completion before submission was not included immediately"
    )
    sent = run.calls[0].request
    original = list(sent.Input)
    assert_stop_result(sent, "call-1", COMPLETED)
    assert_stop_result(sent, "call-0", TOOL_CALL_RUNNING_PAYLOAD)
    for item in sent.Input:
        if item.Type != ItemType.TOOL_RESULT:
            continue
        result = cast(ToolResult, item.Data)
        assert not (
            result.CallID == "call-1" and result.Output[0].Value == TOOL_CALL_RUNNING_PAYLOAD
        ), "completion before submission retained its running placeholder"
    await run.update(0, Status.COMPLETED)
    await run.input(stop_input("heartbeat", ControlMode.HEARTBEAT))
    assert len(run.calls) == 1, "late inputs interrupted or replaced the request"
    assert not run.calls[0].canceled, "late inputs interrupted or replaced the request"
    suffix = [text_result("call-0", COMPLETED), user_message("user requested stop")]
    want = replace(sent, Input=[*original, *suffix])
    response = text_response("Waiting for A.")
    failure = RuntimeError("injected write failure")
    if stage == "turn write failed":
        faults.turn_error = failure
    elif stage == "response write failed":
        faults.response_error = failure
    if stage != "in flight":
        await run.respond(0, response)
        if stage != "response write failed":
            want = replace(want, Input=[*original, *(response.Output or []), *suffix])
    if stage == "response committed":
        assert len(run.calls) == 2, "next request did not place the response before late inputs"
        assert run.calls[1].request == want, (
            "next request did not place the response before late inputs"
        )
        assert run.current._state.delivered_inputs == 1, (
            "late inputs were counted as delivered by the earlier response"
        )
        assert run.current._state.current_turn_inputs == 3, (
            "late inputs were counted as delivered by the earlier response"
        )
    assert sent.Input == original, "in-flight request was mutated"
    if stage in ("turn write failed", "response write failed"):
        err = await run.error()
        assert caused_by(err, failure), f"Run error = {err!r}, want {failure!r}"
        assert len(run.calls) == 1, "failed persistence started another model request"
    else:
        run.cancel()
        await wait()
        err = await run.error()
        assert isinstance(err, asyncio.CancelledError), f"Run error = {err!r}"

    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, (
        "resume did not reconstruct submitted history and include pending inputs"
    )
    assert resumed.calls[0].request == want, (
        "resume did not reconstruct submitted history and include pending inputs"
    )
    assert_completed_results(resumed.calls[0].request, 2)
    assert len(resumed.operations.adds) == 0, "resume redispatched completed operations"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, (
        "completion delivery did not settle in one response"
    )
    assert len(resumed.calls) == 1, "completion delivery did not settle in one response"


@fake_time
async def test_coordinator_steering_commits_pending_suffix_before_new_response() -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    await restore_test_run(run, store)
    await run.start()
    await run.input(external_event("first", "first input"))
    initial = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    await run.input(external_event("steer", "steering input"))
    assert len(run.calls) == 2, "steering did not replace the request"
    assert run.calls[0].canceled, "steering did not replace the request"
    want = replace(
        initial,
        Input=[*initial.Input, text_result("call-0", COMPLETED), user_message("steering input")],
    )
    assert run.calls[1].request == want, "replacement request omitted or reordered pending inputs"
    await run.update(1, Status.COMPLETED)
    response = text_response("Current response.")
    await run.respond(1, response)
    want = replace(
        want, Input=[*want.Input, *(response.Output or []), text_result("call-1", COMPLETED)]
    )
    assert len(run.calls) == 3, "new response did not land at the replacement request boundary"
    assert run.calls[2].request == want, (
        "new response did not land at the replacement request boundary"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await run.respond(2, text_response("Done."))
    run.assert_stopped()


@pytest.mark.parametrize("kind", ["external", "heartbeat", "hard stop"])
@fake_time
async def test_coordinator_recovers_pending_results_after_input_write_failure(kind: str) -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    faults = FailingStore(store)
    await restore_test_run(run, faults)
    await run.start()
    await run.update(1, Status.COMPLETED)
    want = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    want = replace(want, Input=[*want.Input, text_result("call-0", COMPLETED)])
    input = external_event("unpersisted", "unpersisted input")
    if kind == "heartbeat":
        input = stop_input("unpersisted", ControlMode.HEARTBEAT)
    elif kind == "hard stop":
        input = stop_input("unpersisted", ControlMode.STOP_HARD)
    failure = RuntimeError("input write failed")
    faults.input_error = failure
    await run.input(input)
    err = await run.error()
    assert caused_by(err, failure), f"Run error = {err!r}, want {failure!r}"
    assert len(run.calls) == 1, "unpersisted input triggered a request or cancellation"
    assert len(run.operations.cancels) == 0, "unpersisted input triggered a request or cancellation"
    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, "resume lost persisted results or retained the failed input"
    assert resumed.calls[0].request == want, (
        "resume lost persisted results or retained the failed input"
    )
    assert resumed.current._state.available_inputs == 2, (
        "resume miscounted results or redispatched completed operations"
    )
    assert len(resumed.operations.adds) == 0, (
        "resume miscounted results or redispatched completed operations"
    )
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, (
        "restored results were not delivered exactly once"
    )
    assert len(resumed.calls) == 1, "restored results were not delivered exactly once"


@fake_time
async def test_coordinator_starts_new_tool_while_delivering_previous_completion() -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    translator = SubmissionTranslator([value_spec(b"1")])
    run.dependencies.tools = new_registry(
        StaticTranslators(bash=translator, view_image=OperationStatusTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    await restore_test_run(run, store)
    run.go()
    await wait()
    await run.update(1, Status.COMPLETED)
    first = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    dispatches = len(run.operations.adds)
    response = Response(
        Output=[
            LLMItem(
                ProviderID="reasoning",
                Type=ItemType.REASONING,
                Data=Reasoning(
                    Raw=gojson.Raw(
                        b'{"id":"reasoning","type":"reasoning","summary":[],"encrypted_content":"opaque"}'
                    )
                ),
            ),
            LLMItem(
                ProviderID="item-C",
                Type=ItemType.TOOL_CALL,
                Data=ToolCall(CallID="C", Name=BASH_NAME, Arguments="{}"),
            ),
        ]
    )
    await run.respond(0, response)
    assert run.request_count() == 1, (
        "previous completion started a turn instead of waiting for the new tool"
    )
    await sleep_ns(SECOND - 1)
    assert run.request_count() == 1, (
        "previous completion started a turn before the tool grace period elapsed"
    )
    await sleep_ns(1)
    want = replace(
        first,
        Input=[
            *first.Input,
            *(response.Output or []),
            text_result("call-0", COMPLETED),
            text_result("C", TOOL_CALL_RUNNING_PAYLOAD),
        ],
    )
    assert len(run.calls) == 2, (
        "new tool response, old completion, and new running result were reordered"
    )
    assert run.calls[1].request == want, (
        "new tool response, old completion, and new running result were reordered"
    )
    work = (
        f"new work: translations={len(translator.calls)}, dispatches={len(run.operations.adds)}, "
        f"delivered={run.current._state.delivered_inputs}, "
        f"submitted={run.current._state.current_turn_inputs}"
    )
    assert len(translator.calls) == 1, work
    assert len(run.operations.adds) > dispatches, work
    assert run.current._state.delivered_inputs == 1, work
    assert run.current._state.current_turn_inputs == 2, work
    run.operations.send(replace(run.operations.adds[dispatches], Status=Status.COMPLETED))
    await wait()
    await sleep_ns(2 * SLURP_IDLE_TIMEOUT_NS)
    await wait()
    second_response = text_response("Waiting for C.")
    await run.respond(1, second_response)
    want = replace(
        want, Input=[*want.Input, *(second_response.Output or []), text_result("C", COMPLETED)]
    )
    assert len(run.calls) == 3, "new completion was not delivered after its in-flight response"
    assert run.calls[2].request == want, (
        "new completion was not delivered after its in-flight response"
    )
    assert run.current._state.delivered_inputs == 2, (
        "new completion was not delivered after its in-flight response"
    )
    run.cancel()
    await wait()
    err = await run.error()
    assert isinstance(err, asyncio.CancelledError), f"Run error = {err!r}"
    resumed = new_stop_test_run(0)
    resumed.dependencies.tools = new_registry(
        StaticTranslators(bash=translator, view_image=OperationStatusTranslator()),
        BASH_NAME,
        VIEW_IMAGE_NAME,
    )
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, "replay changed the request or repeated completed work"
    assert resumed.calls[0].request == want, "replay changed the request or repeated completed work"
    assert len(resumed.operations.adds) == 0, (
        "replay changed the request or repeated completed work"
    )
    assert len(translator.calls) == 1, "replay changed the request or repeated completed work"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Done."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, "completed calls were not delivered exactly once"
    assert resumed.current._state.available_inputs == 3, (
        "completed calls were not delivered exactly once"
    )
    assert len(resumed.calls) == 1, "completed calls were not delivered exactly once"


@fake_time
async def test_coordinator_hard_stop_preserves_pending_inputs_on_replay() -> None:
    run = new_stop_test_run(2)
    store = await persist_test_run(run)
    await restore_test_run(run, store)
    await run.start()
    await run.input(external_event("first", "check progress"))
    want = run.calls[0].request
    await run.update(0, Status.COMPLETED)
    await run.input(stop_input("hard", ControlMode.STOP_HARD))
    assert len(run.calls) == 1, "hard stop failed to interrupt and cancel only pending work"
    assert run.calls[0].canceled, "hard stop failed to interrupt and cancel only pending work"
    assert len(run.operations.cancels) == 1, (
        "hard stop failed to interrupt and cancel only pending work"
    )
    await run.input(external_event("late", "follow up after stop"))
    await run.update(1, Status.CANCELED)
    run.assert_stopped()
    assert len(run.calls) == 1, "hard stop started another model request"
    want = replace(
        want,
        Input=[
            *want.Input,
            text_result("call-0", COMPLETED),
            user_message("follow up after stop"),
            text_result("call-1", str(Status.CANCELED)),
        ],
    )
    resumed = new_stop_test_run(0)
    await restore_test_run(resumed, store)
    await resumed.start()
    assert len(resumed.calls) == 1, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    assert resumed.calls[0].request == want, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    assert len(resumed.operations.adds) == 0, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    assert len(resumed.operations.cancels) == 0, (
        "resume changed stop history, lost late input, or repeated operations"
    )
    await resumed.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, text_response("Follow-up received."))
    resumed.assert_stopped()
    assert resumed.current._pending_inputs() == 0, (
        "resumed stop history did not settle in one response"
    )
    assert len(resumed.calls) == 1, "resumed stop history did not settle in one response"
