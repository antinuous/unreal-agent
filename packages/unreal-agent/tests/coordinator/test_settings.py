import pytest
from unreal_agent.inbox import ControlMode, Settings
from unreal_agent.llm import Model, ReasoningEffort
from unreal_agent.operation import Status
from unreal_agent.session import Turn, TurnType
from unreal_agent.sessionstore import ModelResponse

from tests.coordinator._helpers import (
    as_store,
    assert_stop_result,
    caused_by,
    external_event,
    fake_time,
    new_heartbeat_test_run,
    new_stop_test_run,
    new_tool_grace_test_run,
    persist_test_run,
    settings_input,
    stop_input,
    text_response,
    tool_grace_response,
    update_tool_grace_call,
)


@fake_time
async def test_coordinator_settings_do_not_wake_or_interrupt_model() -> None:
    run = new_stop_test_run(0)
    await run.start()
    initial_model = Model(ID="initial", MaxOutputTokens=2048, ReasoningEffort=ReasoningEffort.LOW)
    await run.input(
        settings_input(
            "initial",
            Settings(
                Model=initial_model.ID,
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
            Settings(Model="next", ReasoningEffort=ReasoningEffort.HIGH),
        )
    )
    assert run.request_count() == 1, "settings interrupted or altered the active request"
    assert not run.calls[0].canceled, "settings interrupted or altered the active request"
    assert run.calls[0].request.Model == initial_model, (
        "settings interrupted or altered the active request"
    )
    await run.respond(0, text_response("Hello."))
    assert run.request_count() == 1, "settings caused an extra turn after the response"
    assert run.current._pending_inputs() == 0, "settings caused an extra turn after the response"
    await run.input(external_event("follow-up", "continue"))
    assert run.request_count() == 2, "following turn did not use updated settings"
    assert run.calls[1].request.Model == Model(
        ID="next", MaxOutputTokens=2048, ReasoningEffort=ReasoningEffort.HIGH
    ), "following turn did not use updated model settings"
    await run.respond(1, text_response("Done."))
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
                Model="initial-model", MaxOutputTokens=123, ReasoningEffort=ReasoningEffort.LOW
            ),
        ),
        settings_input("effort", Settings(ReasoningEffort=ReasoningEffort.HIGH)),
        settings_input("model", Settings(Model="saved-model")),
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
        Model(ID="model", ReasoningEffort=ReasoningEffort.MEDIUM)
    )
    await run.start()
    assert run.request_count() == 0, "replayed settings started a turn"
    await run.input(external_event("prompt", "continue"))
    assert run.calls[0].request.Model == Model(
        ID="saved-model", MaxOutputTokens=123, ReasoningEffort=ReasoningEffort.HIGH
    ), "latest recorded settings did not override initial configuration"
