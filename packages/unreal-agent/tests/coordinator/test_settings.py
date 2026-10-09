import pytest
from msgspec.structs import replace
from unreal_agent.inbox import ControlMode, Settings
from unreal_agent.llm import Model, ReasoningEffort, Usage
from unreal_agent.operation import Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import ModelResponse

from tests.coordinator._helpers import (
    as_store,
    assert_compaction_turn,
    assert_stop_result,
    caused_by,
    external_event,
    fake_time,
    new_compaction_run,
    new_heartbeat_test_run,
    new_stop_test_run,
    new_tool_grace_test_run,
    persist_test_run,
    settings_input,
    stop_input,
    stop_test_adapter,
    text_response,
    tool_grace_response,
    update_tool_grace_call,
)


@fake_time
async def test_coordinator_settings_do_not_wake_idle_model_and_restart_active_request() -> None:
    run = new_stop_test_run(0)
    await run.start()
    initial_model = Model(
        ID="initial",
        CompactionThreshold=200_000,
        MaxOutputTokens=2048,
        ReasoningEffort=ReasoningEffort.LOW,
    )
    await run.input(
        settings_input(
            "initial",
            Settings(
                Model=initial_model.ID,
                CompactionThreshold=initial_model.CompactionThreshold,
                MaxOutputTokens=2048,
                ReasoningEffort=initial_model.ReasoningEffort,
            ),
        )
    )
    assert run.request_count() == 0, "settings woke an idle model"
    assert run.current._pending_inputs() == 0, "settings woke an idle model"
    run.assert_running()
    await run.input(external_event("prompt", "hello"))
    assert run.request_count() == 1, "next turn did not use settings"
    assert run.calls[0].request.Model == initial_model, "next turn did not use settings"
    await run.input(
        settings_input(
            "next",
            Settings(
                Model="next", CompactionThreshold=80_000, ReasoningEffort=ReasoningEffort.HIGH
            ),
        )
    )
    assert run.request_count() == 2, "settings did not restart the active request"
    assert run.calls[0].canceled, "settings did not restart the active request"
    assert not run.calls[1].canceled, "settings did not restart the active request"
    next_model = Model(
        ID="next",
        CompactionThreshold=80_000,
        MaxOutputTokens=2048,
        ReasoningEffort=ReasoningEffort.HIGH,
    )
    want = replace(run.calls[0].request, Model=next_model)
    assert run.calls[1].request == want, "restart did not preserve context and apply settings"
    assert run.calls[0].request.Model == initial_model, (
        "restart did not preserve context and apply settings"
    )
    await run.input(settings_input("limit", Settings(MaxOutputTokens=4096)))
    assert run.request_count() == 3, "output token limit did not restart the active request"
    assert run.calls[1].canceled, "output token limit did not restart the active request"
    assert not run.calls[2].canceled, "output token limit did not restart the active request"
    next_model = replace(next_model, MaxOutputTokens=4096)
    want = replace(want, Model=next_model)
    assert run.calls[2].request == want, (
        "restart did not preserve context and apply the output token limit"
    )
    await run.respond(2, text_response("Hello."))
    assert run.request_count() == 3, "settings caused an extra turn after the response"
    assert run.current._pending_inputs() == 0, "settings caused an extra turn after the response"
    await run.input(external_event("follow-up", "continue"))
    assert run.request_count() == 4, "following turn did not use updated model settings"
    assert run.calls[3].request.Model == next_model, (
        "following turn did not use updated model settings"
    )
    await run.respond(3, text_response("Done."))
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()


@fake_time
async def test_coordinator_settings_restart_without_waiting_for_canceled_response() -> None:
    run = new_stop_test_run(0)
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("prompt", "hello"))
    await run.input(settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.HIGH)))
    assert run.request_count() == 2, "restart waited for the canceled request to return"
    assert run.calls[0].canceled, "restart waited for the canceled request to return"
    await run.respond(0, text_response("Stale response."))
    assert run.store.appended_responses == [], "canceled response affected the replacement request"
    assert run.current._pending_inputs() == 1, "canceled response affected the replacement request"
    assert not run.calls[1].canceled, "canceled response affected the replacement request"
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    await run.respond(1, text_response("Done."))
    run.assert_stopped()


@fake_time
async def test_coordinator_model_switch_discards_previous_usage() -> None:
    run = await new_compaction_run()
    run.dependencies.context_builder.set_model(Model(ID="initial", CompactionThreshold=100_000))
    run.dependencies.llm = stop_test_adapter(run, ignore_cancellation=True)
    await run.start()
    await run.input(external_event("prompt", "continue"))
    await assert_compaction_turn(run, 1, TurnType.REGULAR)
    await run.input(settings_input("settings", Settings(Model="next", CompactionThreshold=25_000)))
    await assert_compaction_turn(run, 2, TurnType.REGULAR)
    assert run.calls[0].canceled, "model switch did not replace the active request"
    assert run.calls[1].request.Model.ID == "next", (
        "model switch did not replace the active request"
    )
    stale = replace(text_response("Old model response"), Usage=Usage(InputTokens=90_000))
    await run.respond(0, stale)
    assert not run.dependencies.context_builder.needs_compaction(), (
        "old model usage survived the model switch"
    )
    replayed = new_stop_test_run(0).current
    replayed._dependencies.sessions = run.dependencies.sessions
    await replayed._load_history()
    assert not replayed._dependencies.context_builder.needs_compaction(), (
        "replay restored checkpoints from the old model"
    )
    fresh = replace(text_response("New model response"), Usage=Usage(InputTokens=30_000))
    await run.respond(1, fresh)
    assert run.dependencies.context_builder.needs_compaction(), (
        "new model response did not establish measured usage"
    )
    await run.input(stop_input("stop", ControlMode.STOP_WHEN_IDLE))
    run.assert_stopped()


@fake_time
async def test_coordinator_settings_preserve_tool_grace_and_apply_to_continuation() -> None:
    run = new_tool_grace_test_run()
    await run.start()
    await run.input(external_event("prompt", "run both tools"))
    await run.respond(0, tool_grace_response("A", "B"))
    await update_tool_grace_call(run, "A", Status.COMPLETED)
    grace = run.current._state.grace
    await run.input(settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.MAX)))
    assert run.request_count() == 1, (
        "settings disturbed pending operations or ended their grace period"
    )
    assert len(run.current._state.grace_tool_calls) == 1, (
        "settings disturbed pending operations or ended their grace period"
    )
    assert run.current._state.grace is grace, (
        "settings disturbed pending operations or ended their grace period"
    )
    assert len(run.operations.cancels) == 0, (
        "settings disturbed pending operations or ended their grace period"
    )
    await update_tool_grace_call(run, "B", Status.COMPLETED)
    assert run.request_count() == 2, "automatic continuation did not use updated settings"
    assert run.calls[1].request.Model.ReasoningEffort == ReasoningEffort.MAX, (
        "automatic continuation did not use updated settings"
    )
    assert_stop_result(run.calls[1].request, "A", str(Status.COMPLETED))
    assert_stop_result(run.calls[1].request, "B", str(Status.COMPLETED))


@fake_time
async def test_coordinator_settings_follow_inbox_order_and_deduplicate() -> None:
    run = new_heartbeat_test_run(0)
    await run.start()
    first = settings_input(
        "first",
        Settings(
            Model="initial",
            MaxOutputTokens=123,
            ReasoningEffort=ReasoningEffort.LOW,
        ),
    )
    last = settings_input("last", Settings(Model="next", ReasoningEffort=ReasoningEffort.HIGH))
    repeated = settings_input("repeated", Settings(Model="next"))
    await run.input(first, last, first, repeated)
    assert run.recorded_inputs == [first, last, repeated], (
        f"recorded settings = {run.recorded_inputs!r}"
    )
    await run.input(external_event("prompt", "hello"))
    assert run.calls[0].request.Model == Model(
        ID="next", MaxOutputTokens=123, ReasoningEffort=ReasoningEffort.HIGH
    ), "duplicate rolled back the latest settings"
    await run.input(first, last, repeated)
    assert run.request_count() == 1, "duplicate settings restarted the active request"
    assert not run.calls[0].canceled, "duplicate settings restarted the active request"
    await run.input(
        settings_input("new-first", Settings(Model="final", ReasoningEffort=ReasoningEffort.LOW)),
        settings_input("new-last", Settings(ReasoningEffort=ReasoningEffort.MAX)),
    )
    assert run.request_count() == 2, "restart did not apply batched settings in inbox order"
    assert run.calls[1].request.Model == Model(
        ID="final", MaxOutputTokens=123, ReasoningEffort=ReasoningEffort.MAX
    ), "restart did not apply batched settings in inbox order"


@pytest.mark.parametrize(
    "mode", [ControlMode.STOP_HARD, ControlMode.STOP_WHEN_IDLE], ids=lambda value: str(value)
)
@fake_time
async def test_coordinator_settings_preserve_pending_stop(mode: ControlMode) -> None:
    run = new_stop_test_run(1)
    await run.start()
    await run.input(
        stop_input("stop", mode),
        settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.HIGH)),
    )
    run.assert_running()
    assert run.current._stop.request.Mode == mode, (
        "settings changed the pending stop or started a turn"
    )
    assert run.request_count() == 0, "settings changed the pending stop or started a turn"
    if mode == ControlMode.STOP_HARD:
        assert len(run.operations.cancels) == 1, "settings prevented operation cancellation"
        await run.update(0, Status.CANCELED)
        assert run.request_count() == 0, "hard stop started a final turn"
    else:
        assert len(run.operations.cancels) == 0, "settings canceled pending work"
        await run.update(0, Status.COMPLETED)
        assert run.request_count() == 1, "final turn did not use updated settings"
        assert run.calls[0].request.Model.ReasoningEffort == ReasoningEffort.HIGH, (
            "final turn did not use updated settings"
        )
        await run.respond(0, text_response("Done."))
    run.assert_stopped()


@fake_time
async def test_coordinator_settings_require_persistence() -> None:
    run = new_stop_test_run(0)
    want = RuntimeError("settings storage failed")
    run.store.append_input_error = want
    await run.start()
    await run.input(settings_input("settings", Settings(ReasoningEffort=ReasoningEffort.HIGH)))
    err = await run.error()
    assert caused_by(err, want), f"Run error = {err!r}, want {want!r}"
    assert run.request_count() == 0, "model ran after settings persistence failed"


@pytest.mark.parametrize("mode", ["resume", "fork"])
@fake_time
async def test_coordinator_settings_replay_on_resume_and_fork(mode: str) -> None:
    parent = new_stop_test_run(0)
    store = await persist_test_run(parent)
    for input in [
        settings_input(
            "first",
            Settings(
                Model="initial-model",
                CompactionThreshold=200_000,
                MaxOutputTokens=123,
                ReasoningEffort=ReasoningEffort.LOW,
            ),
        ),
        settings_input("effort", Settings(ReasoningEffort=ReasoningEffort.HIGH)),
        settings_input("model", Settings(Model="saved-model")),
        settings_input("threshold", Settings(CompactionThreshold=80_000)),
    ]:
        await store.append_input("session-1", input)
    turn = Turn(ID="parent-turn", PreviousTurnID="turn-1", Type=TurnType.REGULAR)
    await store.append_turn("session-1", turn)
    await store.append_model_response(
        "session-1", ModelResponse(TurnID=turn.ID, Response=text_response("Done."))
    )
    id = "session-1"
    if mode == "fork":
        id = "child"
        await store.fork(id, "session-1", turn.ID)
    restored = await store.resume(id)
    run = new_stop_test_run(0)
    run.dependencies.session_id = id
    run.dependencies.sessions = as_store(store)
    run.dependencies.restored = restored
    run.dependencies.context_builder.set_model(
        Model(ID="model", CompactionThreshold=250_000, ReasoningEffort=ReasoningEffort.MEDIUM)
    )
    await run.start()
    assert run.request_count() == 0, "replayed settings started a turn"
    await run.input(external_event("prompt", "continue"))
    assert run.calls[0].request.Model == Model(
        ID="saved-model",
        CompactionThreshold=80_000,
        MaxOutputTokens=123,
        ReasoningEffort=ReasoningEffort.HIGH,
    ), "latest recorded settings did not override initial configuration"
