from collections.abc import Callable
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import Raw
from unreal_agent.operation import (
    MAX_OUTPUT_LENGTH,
    TYPE_REMOTE_JOB,
    TYPE_SHELL,
    VERSION_REMOTE_JOB,
    Operation,
    OperationError,
    RemoteJobPlan,
    RemoteJobState,
    Status,
    Step,
    UnsupportedError,
    cancel_remote_job,
    decode_remote_job_state,
    fail_remote_job,
    new_remote_job_spec,
    update_remote_job,
)
from unreal_agent.operation.remote_job_handler import valid_remote_job_status_transition


def _plan(version: int = 1, data: bytes = b"{}") -> RemoteJobPlan:
    return RemoteJobPlan(Type="test", Version=version, Data=Raw(data))


def _operation(max_output_length: int | None = None) -> Operation:
    spec = new_remote_job_spec(_plan())
    return Operation(
        ID="remote",
        Type=spec.Type,
        Version=spec.Version,
        Status=Status.READY,
        State=spec.State,
        MaxOutputLength=spec.MaxOutputLength if max_output_length is None else max_output_length,
    )


def _state(step: Step) -> RemoteJobState:
    assert step.operation is not None
    return decode_remote_job_state(step.operation)


def test_new_remote_job_spec_round_trips_plan() -> None:
    plan = _plan(2, b'{"value":"preserved"}')
    spec = new_remote_job_spec(plan)
    assert (spec.Type, spec.Version) == (TYPE_REMOTE_JOB, VERSION_REMOTE_JOB)
    state = decode_remote_job_state(
        Operation(
            ID="remote-1",
            Type=spec.Type,
            Version=spec.Version,
            Status=Status.READY,
            MaxOutputLength=spec.MaxOutputLength,
            State=spec.State,
        )
    )
    assert (state.Plan.Type, state.Plan.Version) == (plan.Type, plan.Version)
    assert bytes(state.Plan.Data) == bytes(plan.Data)


@pytest.mark.parametrize(
    ("apply", "status"),
    [
        (lambda current: fail_remote_job(current, OperationError("failed")), Status.FAILED),
        (cancel_remote_job, Status.CANCELED),
    ],
    ids=["failed", "canceled"],
)
def test_remote_job_terminal_transitions_clear_subscription_and_truncated_result(
    apply: Callable[[Operation], Step], status: Status
) -> None:
    current = _operation()
    state = decode_remote_job_state(current)
    state.Subscription = Raw(b'{"partial":"response"}')
    state.TerminalResult = '{"stale":true}'
    current = replace(current, MaxOutputLength=3)
    awaiting = update_remote_job(current, state, Status.AWAITING)
    prepared = _state(awaiting)
    assert prepared.ResultTruncated
    assert prepared.ResultBytes == len(state.TerminalResult)

    assert awaiting.operation is not None
    terminal = apply(awaiting.operation)
    terminal_state = _state(terminal)
    assert terminal.operation is not None
    assert terminal.operation.Status == status
    assert len(terminal_state.Subscription) == 0
    assert terminal_state.TerminalResult == ""
    assert terminal_state.ResultBytes == 0
    assert not terminal_state.ResultTruncated


@pytest.mark.parametrize(
    ("message", "want", "truncated"),
    [("é", "é", False), ("éééé", "é...2 bytes truncated...éé", True)],
    ids=["short", "oversized"],
)
def test_fail_remote_job_replaces_truncated_error(message: str, want: str, truncated: bool) -> None:
    current = _operation(3)
    state = decode_remote_job_state(current)
    state.TerminalError = "previous error"
    awaiting = update_remote_job(current, state, Status.AWAITING)
    prepared = _state(awaiting)
    assert prepared.ErrorTruncated
    assert prepared.ErrorBytes == len(state.TerminalError)
    assert awaiting.operation is not None
    failed = _state(fail_remote_job(awaiting.operation, OperationError(message)))
    assert failed.TerminalError == want
    assert failed.ErrorBytes == len(message.encode())
    assert failed.ErrorTruncated == truncated


@pytest.mark.parametrize(
    ("plan", "want"),
    [
        (RemoteJobPlan(Version=1, Data=Raw(b"{}")), "type must be set"),
        (RemoteJobPlan(Type="test", Data=Raw(b"{}")), "version must be positive"),
        (RemoteJobPlan(Type="test", Version=1), "data must be valid JSON"),
        (RemoteJobPlan(Type="test", Version=1, Data=Raw(b"{")), "data must be valid JSON"),
    ],
    ids=["missing type", "zero version", "missing data", "invalid data"],
)
def test_new_remote_job_spec_rejects_invalid_plans(plan: RemoteJobPlan, want: str) -> None:
    with pytest.raises(OperationError, match=want):
        new_remote_job_spec(plan)


@pytest.mark.parametrize(
    "current",
    [
        Operation(ID="remote-1", Type=TYPE_SHELL, Version=VERSION_REMOTE_JOB),
        Operation(ID="remote-1", Type=TYPE_REMOTE_JOB, Version=VERSION_REMOTE_JOB + 1),
    ],
)
def test_decode_remote_job_state_reports_unsupported_envelope(current: Operation) -> None:
    with pytest.raises(UnsupportedError):
        decode_remote_job_state(current)


@pytest.mark.parametrize("failure", [False, True], ids=["result", "error"])
def test_remote_job_output_remains_truncated_after_round_trip(failure: bool) -> None:
    current = _operation(3)
    state = decode_remote_job_state(current)
    status = Status.COMPLETED
    if failure:
        state.TerminalError = "éééé"
        status = Status.FAILED
    else:
        state.TerminalResult = '"éééé"'
    for _ in range(2):
        step = update_remote_job(current, state, status)
        assert step.operation is not None
        current = step.operation
        state = decode_remote_job_state(current)
        if failure:
            assert state.TerminalError == "é...2 bytes truncated...éé"
            assert state.ErrorTruncated
            assert state.ErrorBytes == 8
        else:
            assert state.TerminalResult == '"...6 bytes truncated...é"'
            assert state.ResultTruncated
            assert state.ResultBytes == 10


@pytest.mark.parametrize("limit", [0, MAX_OUTPUT_LENGTH + 1])
def test_remote_job_rejects_invalid_output_limits(limit: int) -> None:
    spec = new_remote_job_spec(_plan())
    with pytest.raises(OperationError):
        decode_remote_job_state(
            Operation(Type=spec.Type, Version=spec.Version, State=spec.State, MaxOutputLength=limit)
        )


def test_remote_job_accepts_maximum_output_length() -> None:
    spec = new_remote_job_spec(_plan())
    decode_remote_job_state(
        Operation(
            Type=spec.Type,
            Version=spec.Version,
            State=spec.State,
            MaxOutputLength=MAX_OUTPUT_LENGTH,
        )
    )


def test_fail_remote_job_requires_an_error() -> None:
    with pytest.raises(OperationError, match="error must be set"):
        fail_remote_job(_operation(), None)


@pytest.mark.parametrize(
    ("current", "updated", "valid"),
    [
        pytest.param(Status.READY, Status.AWAITING, True, id="ready awaiting"),
        pytest.param(Status.READY, Status.CANCELING, True, id="ready canceling"),
        pytest.param(Status.READY, Status.COMPLETED, True, id="ready completed"),
        pytest.param(Status.READY, Status.FAILED, True, id="ready failed"),
        pytest.param(Status.READY, Status.CANCELED, True, id="ready canceled"),
        pytest.param(Status.READY, Status.READY, False, id="ready remains ready"),
        pytest.param(Status.AWAITING, Status.AWAITING, True, id="awaiting progress"),
        pytest.param(Status.AWAITING, Status.READY, False, id="awaiting regresses"),
        pytest.param(Status.CANCELING, Status.CANCELING, True, id="canceling progress"),
        pytest.param(Status.CANCELING, Status.FAILED, True, id="canceling failed"),
        pytest.param(Status.CANCELING, Status.CANCELED, True, id="canceling canceled"),
        pytest.param(Status.CANCELING, Status.COMPLETED, False, id="canceling completed"),
        pytest.param(Status.COMPLETED, Status.COMPLETED, False, id="terminal update"),
        pytest.param(cast(Status, "unknown"), Status.FAILED, False, id="unknown current"),
    ],
)
def test_remote_job_status_transitions(current: Status, updated: Status, valid: bool) -> None:
    assert valid_remote_job_status_transition(current, updated) == valid
