from pathlib import Path
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.coordinator import Coordinator, CoordinatorError
from unreal_agent.coordinator.loop import ModelResponseResult, inbox_source, iterator_source
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Failure, ItemType, Message, Model, Response, StopReason, Usage
from unreal_agent.operation import Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import Item, ItemKind, ModelResponse
from unreal_agent.sessionstore.localfile import Store as LocalFileStore

from tests.coordinator._helpers import (
    COMPACTION_THRESHOLD,
    FailingStore,
    MemoryStore,
    append_compaction_test_item,
    as_store,
    assert_compaction_turn,
    assert_stop_result,
    caused_by,
    external_event,
    fake_time,
    new_compaction_run,
    new_stop_test_run,
    recorded_compaction_history,
    stop_input,
    stop_test_adapter,
    text_response,
)


def completed_summary() -> Response:
    return replace(text_response("Summary"), Stop=StopReason.COMPLETE)


async def process_events(current: Coordinator) -> bool:
    current._inputs = inbox_source(current._dependencies.inbox, current._wake.set)
    current._updates = iterator_source(
        current._dependencies.operations.updates(), current._wake.set
    )
    try:
        return await current._process_events()
    finally:
        await current._close()


@pytest.mark.parametrize("startup", [False, True], ids=["input", "startup"])
@fake_time
async def test_automatic_compaction_continues_ordinary_work(startup: bool) -> None:
    run = await new_compaction_run()
    input = external_event("pending", "Continue work")
    if startup:
        await run.dependencies.sessions.append_input("session-1", input)
    await run.start()
    if not startup:
        await run.input(input)
    await assert_compaction_turn(run, 1, TurnType.COMPACTION)
    assert len(run.calls[0].request.Tools) == 0, "summary delivered input or exposed tools"
    assert run.current._pending_inputs() == 1, "summary delivered input or exposed tools"
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await run.respond(0, completed_summary())
    await assert_compaction_turn(run, 2, TurnType.REGULAR)
    assert run.current._pending_inputs() == 1, "summary answered pending input"
    summary_found = input_found = False
    for item in run.calls[1].request.Input:
        if item.Type == ItemType.MESSAGE:
            text = cast(Message, item.Data).Text
            summary_found = summary_found or text == "Summary"
            input_found = input_found or text == "Continue work"
    assert summary_found, f"continuation = {run.calls[1].request!r}"
    assert input_found, f"continuation = {run.calls[1].request!r}"
    await run.respond(1, text_response("Done"))
    run.assert_stopped()
    assert run.current._pending_inputs() == 0, "ordinary input delivery or request count differs"
    assert len(run.calls) == 2, "ordinary input delivery or request count differs"


@pytest.mark.parametrize(
    "response",
    [
        Response(Stop=StopReason.REFUSED),
        Response(Stop=StopReason.MAX_OUTPUT_TOKENS),
        Response(Failure=Failure(Code="failed")),
    ],
    ids=["refused", "max_output_tokens", "#00"],
)
@fake_time
async def test_automatic_compaction_rejected_response_retries(response: Response) -> None:
    run = await new_compaction_run()
    await run.start()
    before = run.dependencies.context_builder
    await run.input(external_event("pending", "Continue"))
    for attempt in range(2):
        await run.respond(attempt, response)
        await assert_compaction_turn(run, attempt + 2, TurnType.COMPACTION)
        message = "rejected summary replaced context or delivered input"
        assert before is run.dependencies.context_builder, message
        assert run.current._pending_inputs() == 1, message
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await run.respond(2, Response(Stop=StopReason.COMPLETE, Output=text_response("Summary").Output))
    await assert_compaction_turn(run, 4, TurnType.REGULAR)
    await run.respond(3, text_response("Done"))
    run.assert_stopped()


@pytest.mark.parametrize("mode", [ControlMode.STOP_HARD, "steer"], ids=["hard", "steer"])
@fake_time
async def test_automatic_compaction_stop_and_steering(mode: str) -> None:
    run = await new_compaction_run()
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("pending", "Continue"))
    old = run.dependencies.context_builder
    if mode == "steer":
        await run.input(external_event("steer", "New direction"))
    else:
        await run.input(stop_input("stop", mode))
    assert run.calls[0].canceled, "compaction not canceled"
    summary = replace(text_response("Late summary"), Stop=StopReason.COMPLETE)
    await run.respond(0, summary)
    assert old is run.dependencies.context_builder, "late summary replaced context"
    if mode != ControlMode.STOP_HARD:
        await assert_compaction_turn(run, 2, TurnType.COMPACTION)
        await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
        await run.respond(
            1, Response(Stop=StopReason.COMPLETE, Output=text_response("Summary").Output)
        )
        await assert_compaction_turn(run, 3, TurnType.REGULAR)
        await run.respond(2, text_response("Done"))
    run.assert_stopped()


@pytest.mark.parametrize("outcome", ["complete", "refused", "no cutoff"])
async def test_compaction_response_preserves_pending_work(outcome: str, tmp_path: Path) -> None:
    current = await recorded_compaction_history(LocalFileStore(tmp_path))
    await append_compaction_test_item(
        current, Item(Kind=ItemKind.INPUT, Data=external_event("pending", "Continue"))
    )
    await append_compaction_test_item(
        current,
        Item(
            Kind=ItemKind.TURN,
            Data=Turn(ID="compact", PreviousTurnID="B", Type=TurnType.COMPACTION),
        ),
    )
    if outcome == "no cutoff":
        current._dependencies.context_builder = new_stop_test_run(
            0
        ).current._dependencies.context_builder
    previous = current._dependencies.context_builder
    response = Response(Stop=StopReason.COMPLETE)
    if outcome == "refused":
        response.Stop = StopReason.REFUSED
    await current._process_model_response(ModelResponseResult("compact", response=response))
    assert current._pending_inputs() == 1, "compaction response consumed pending input"
    call_model = await process_events(current)
    assert call_model, "pending work did not schedule continuation"
    if outcome == "complete":
        assert previous is not current._dependencies.context_builder, (
            "completed compaction did not reconstruct context"
        )
        rebuilt = current._dependencies.context_builder
        with pytest.raises(CoordinatorError):
            await current._process_model_response(ModelResponseResult("compact", response=response))
        message = "duplicate response changed pending input or context"
        assert current._pending_inputs() == 1, message
        assert current._dependencies.context_builder is rebuilt, message
    else:
        assert previous is current._dependencies.context_builder, (
            "unsuccessful response changed context"
        )


@pytest.mark.parametrize("stage", ["turn", "persisted response", "reconstructed"])
@fake_time
async def test_compaction_crash_continuation(stage: str) -> None:
    store = MemoryStore()
    recorded = await recorded_compaction_history(store)
    await append_compaction_test_item(
        recorded, Item(Kind=ItemKind.INPUT, Data=external_event("pending", "Continue"))
    )
    await append_compaction_test_item(
        recorded,
        Item(
            Kind=ItemKind.TURN,
            Data=Turn(ID="compact", PreviousTurnID="B", Type=TurnType.COMPACTION),
        ),
    )
    summary = ModelResponse(
        TurnID="compact",
        Response=Response(Stop=StopReason.COMPLETE, Output=text_response("Summary").Output),
    )
    if stage == "persisted response":
        await store.append_model_response("session-1", summary)
    if stage == "reconstructed":
        await recorded._handle_model_response(summary)
    run = new_stop_test_run(0)
    run.dependencies.sessions = as_store(store)
    run.dependencies.context_builder.set_model(Model(CompactionThreshold=COMPACTION_THRESHOLD))
    await run.start()
    ordinary = 0
    if stage == "turn":
        await assert_compaction_turn(run, 1, TurnType.COMPACTION)
        await run.respond(0, summary.Response)
        recorded._apply_compaction_response(summary.Response)
        ordinary = 1
    await assert_compaction_turn(run, ordinary + 1, TurnType.REGULAR)
    if stage == "persisted response":
        recorded._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=summary))
    recorded._dependencies.context_builder.set_model(
        Model(CompactionThreshold=COMPACTION_THRESHOLD)
    )
    expected = recorded._dependencies.context_builder.build()
    assert expected.request == run.calls[ordinary].request, (
        "resumed request differs from uninterrupted context"
    )
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await run.respond(ordinary, text_response("Done"))
    run.assert_stopped()


@fake_time
async def test_compaction_keeps_operation_updates_live() -> None:
    run = new_stop_test_run(1)
    recorded = await recorded_compaction_history(MemoryStore())
    for item in run.store.history:
        data = item.Data
        if isinstance(data, Turn):
            data = replace(data, PreviousTurnID="B")
        if isinstance(data, ModelResponse):
            usage = replace(data.Response.Usage, InputTokens=40_000)
            data = replace(data, Response=replace(data.Response, Usage=usage))
        await append_compaction_test_item(recorded, replace(item, Data=data))
    run.dependencies.sessions = recorded._dependencies.sessions
    run.dependencies.context_builder.set_model(Model(CompactionThreshold=COMPACTION_THRESHOLD))
    await run.start()
    await run.input(external_event("pending", "Continue"))
    await assert_compaction_turn(run, 1, TurnType.COMPACTION)
    await run.update(0, Status.COMPLETED)
    assert len(run.calls) == 1, "operation update interrupted compaction"
    assert not run.calls[0].canceled, "operation update interrupted compaction"
    await run.respond(0, Response(Stop=StopReason.COMPLETE, Output=text_response("Summary").Output))
    await assert_compaction_turn(run, 2, TurnType.REGULAR)
    assert_stop_result(run.calls[1].request, "call-0", str(Status.COMPLETED))
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await run.respond(1, text_response("Done"))
    run.assert_stopped()


@fake_time
async def test_automatic_compaction_without_cutoff_uses_ordinary_request() -> None:
    recorded = await recorded_compaction_history(MemoryStore())
    await append_compaction_test_item(
        recorded,
        Item(Kind=ItemKind.TURN, Data=Turn(ID="small", PreviousTurnID="B", Type=TurnType.REGULAR)),
    )
    await append_compaction_test_item(
        recorded,
        Item(
            Kind=ItemKind.MODEL_RESPONSE,
            Data=ModelResponse(TurnID="small", Response=Response(Usage=Usage(InputTokens=10_000))),
        ),
    )
    run = new_stop_test_run(0)
    run.dependencies.sessions = recorded._dependencies.sessions
    run.dependencies.context_builder.set_model(Model(CompactionThreshold=10_000))
    await run.start()
    assert run.dependencies.context_builder.needs_compaction(), (
        "expected pressure despite missing cutoff"
    )
    await run.input(external_event("pending", "Continue"))
    await assert_compaction_turn(run, 1, TurnType.REGULAR)
    await run.input(stop_input("idle", ControlMode.STOP_WHEN_IDLE))
    await run.respond(0, text_response("Done"))
    run.assert_stopped()


@fake_time
async def test_compaction_turn_persistence_failure_does_not_submit() -> None:
    run = await new_compaction_run()
    want = RuntimeError("turn write failed")
    store = FailingStore(run.dependencies.sessions)
    store.turn_error = want
    run.dependencies.sessions = as_store(store)
    await run.start()
    await run.input(external_event("pending", "Continue"))
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}"
    assert len(run.calls) == 0, "failed turn persistence submitted a request"
    assert run.current._cancel_model is None, "failed turn persistence submitted a request"
