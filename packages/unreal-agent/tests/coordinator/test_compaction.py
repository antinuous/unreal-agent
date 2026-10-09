from pathlib import Path

import pytest
from msgspec.structs import replace
from unreal_agent.inbox import ControlMode
from unreal_agent.llm import Failure, Model, Response, StopReason, Usage
from unreal_agent.llm import Item as LLMItem
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import BEFORE_FIRST, Item, ItemKind, ModelResponse
from unreal_agent.sessionstore.localfile import Store as LocalFileStore

from tests.coordinator._helpers import (
    COMPACTION_THRESHOLD,
    FailingStore,
    MemoryStore,
    append_compaction_test_item,
    as_store,
    assert_compaction_turn,
    caused_by,
    external_event,
    fake_time,
    new_stop_test_run,
    recorded_compaction_history,
    stop_input,
    text_response,
)


@pytest.mark.parametrize("stage", ["turn", "persisted response", "applied response"])
async def test_compaction_reconstruction_matches_local_file_replay(
    stage: str, tmp_path: Path
) -> None:
    current = await recorded_compaction_history(LocalFileStore(tmp_path))
    for item in [
        Item(Kind=ItemKind.INPUT, Data=external_event("before", "Before compaction")),
        Item(
            Kind=ItemKind.TURN,
            Data=Turn(ID="compact", PreviousTurnID="B", Type=TurnType.COMPACTION),
        ),
        Item(Kind=ItemKind.INPUT, Data=external_event("during", "During compaction")),
    ]:
        await append_compaction_test_item(current, item)
    old_builder = current._dependencies.context_builder
    before = old_builder.build()
    response = ModelResponse(
        TurnID="compact", Response=replace(text_response("Summary"), Stop=StopReason.COMPLETE)
    )
    match stage:
        case "persisted response":
            await current._dependencies.sessions.append_model_response(
                current._dependencies.session_id, response
            )
        case "applied response":
            await current._handle_model_response(response)
    reopened = LocalFileStore(tmp_path)
    replayed = new_stop_test_run(0).current
    replayed._dependencies.sessions = reopened
    await replayed._load_history()
    if stage == "persisted response":
        current._add_item_to_local_state(Item(Kind=ItemKind.MODEL_RESPONSE, Data=response))
    live = current._dependencies.context_builder.build()
    restored = replayed._dependencies.context_builder.build()
    assert live == restored, "live reconstruction and replay differ"
    assert current._state == replayed._state, "live reconstruction and replay differ"
    state = current._state
    message = f"compaction changed pending work: {state}"
    assert current._pending_inputs() == 2, message
    assert len(state.tool_calls) == 0, message
    assert len(state.operations) == 0, message
    if stage == "turn":
        assert live == before, "unanswered compaction changed context"
    else:
        message = "usable summary did not replace the builder"
        assert current._dependencies.context_builder is not old_builder, message
        assert live != before, message
    if stage == "applied response":
        live_builder = current._dependencies.context_builder
        replayed_builder = replayed._dependencies.context_builder
        continuation = replace(
            text_response("Continue"), Usage=Usage(InputTokens=29_000, OutputTokens=1_000)
        )
        for item in [
            Item(
                Kind=ItemKind.TURN,
                Data=Turn(ID="next", PreviousTurnID="compact", Type=TurnType.REGULAR),
            ),
            Item(
                Kind=ItemKind.MODEL_RESPONSE,
                Data=ModelResponse(TurnID="next", Response=continuation),
            ),
            Item(
                Kind=ItemKind.TURN,
                Data=Turn(ID="compact-again", PreviousTurnID="next", Type=TurnType.COMPACTION),
            ),
            Item(
                Kind=ItemKind.MODEL_RESPONSE,
                Data=ModelResponse(TurnID="compact-again", Response=response.Response),
            ),
        ]:
            await append_compaction_test_item(current, item)
            replayed._add_item_to_local_state(item)
        message = "subsequent compaction did not replace both builders"
        assert current._dependencies.context_builder is not live_builder, message
        assert replayed._dependencies.context_builder is not replayed_builder, message
        live = current._dependencies.context_builder.build()
        restored = replayed._dependencies.context_builder.build()
        message = "new checkpoints and subsequent compaction differ after replay"
        assert live == restored, message
        assert current._state == replayed._state, message


COMPACTION_RESPONSE_STATUSES: list[tuple[str, StopReason, Failure | None, list[LLMItem] | None]] = [
    (
        "truncated summary",
        StopReason.MAX_OUTPUT_TOKENS,
        None,
        text_response("Partial summary").Output,
    ),
    ("refusal", StopReason.REFUSED, None, text_response("Cannot summarize").Output),
    (
        "failure",
        StopReason.NONE,
        Failure(Code="server_error", Message="failed"),
        text_response("Summary").Output,
    ),
    ("empty output", StopReason.COMPLETE, None, None),
]


@pytest.mark.parametrize(
    ("stop", "failure", "output"),
    [case[1:] for case in COMPACTION_RESPONSE_STATUSES],
    ids=[case[0] for case in COMPACTION_RESPONSE_STATUSES],
)
async def test_compaction_response_status_on_live_and_replay(
    stop: StopReason, failure: Failure | None, output: list[LLMItem] | None, tmp_path: Path
) -> None:
    current = await recorded_compaction_history(LocalFileStore(tmp_path))
    await append_compaction_test_item(
        current,
        Item(
            Kind=ItemKind.TURN,
            Data=Turn(ID="compact", PreviousTurnID="B", Type=TurnType.COMPACTION),
        ),
    )
    old_builder = current._dependencies.context_builder
    before = old_builder.build()
    response = ModelResponse(
        TurnID="compact", Response=Response(Stop=stop, Failure=failure, Output=output)
    )
    await current._handle_model_response(response)
    live = current._dependencies.context_builder.build()
    if stop != StopReason.COMPLETE:
        message = "unsuccessful compaction changed live context"
        assert current._dependencies.context_builder is old_builder, message
        assert live == before, message
    else:
        assert current._dependencies.context_builder is not old_builder, (
            "completed response did not replace context"
        )
    reopened = LocalFileStore(tmp_path)
    page = await reopened.items("session-1", 0, 100)
    last = page.items[-1]
    assert last.Kind == ItemKind.MODEL_RESPONSE, "response was not preserved in history"
    assert last.Data == response, "response was not preserved in history"
    replayed = new_stop_test_run(0).current
    replayed._dependencies.sessions = reopened
    await replayed._load_history()
    restored = replayed._dependencies.context_builder.build()
    assert restored == live, "live and replayed context or state differ"
    assert current._state == replayed._state, "live and replayed context or state differ"


@fake_time
async def test_compaction_persistence_failure_stops_and_retries_on_resume() -> None:
    store = MemoryStore()
    await recorded_compaction_history(store)
    run = new_stop_test_run(0)
    want = RuntimeError("response write failed")
    failing = FailingStore(as_store(store))
    failing.response_error = want
    run.dependencies.sessions = as_store(failing)
    run.dependencies.context_builder.set_model(Model(CompactionThreshold=COMPACTION_THRESHOLD))
    await run.start()
    await run.input(external_event("pending", "Continue work"))
    await assert_compaction_turn(run, 1, TurnType.COMPACTION)
    request = run.calls[0].request
    response = replace(text_response("Summary"), Stop=StopReason.COMPLETE)
    await run.respond(0, response)
    assert run.task is not None
    assert run.task.done(), "Run continued after failed summary persistence"
    err = run.finished_error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert len(run.calls) == 1, "failed persistence started another model request"
    page = await store.items("session-1", BEFORE_FIRST, 100)
    assert page.items[-1].Kind == ItemKind.TURN, "failed response was persisted"
    resumed = new_stop_test_run(0)
    resumed.dependencies.sessions = as_store(store)
    resumed.dependencies.context_builder.set_model(Model(CompactionThreshold=COMPACTION_THRESHOLD))
    await resumed.start()
    await assert_compaction_turn(resumed, 1, TurnType.COMPACTION)
    assert resumed.calls[0].request == request, (
        "resumed compaction request differs from the failed attempt"
    )
    assert resumed.current._pending_inputs() == 1, "pending input was lost during recovery"
    await resumed.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await resumed.respond(0, response)
    await assert_compaction_turn(resumed, 2, TurnType.REGULAR)
    await resumed.respond(1, text_response("Done"))
    resumed.assert_stopped()
