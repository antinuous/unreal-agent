from collections.abc import AsyncGenerator
from typing import Annotated, Final, Protocol

import msgspec
from msgspec.structs import replace

from unreal_agent._go import go_quote
from unreal_agent.gojson import (
    Int64,
    JSONError,
    OmitZero,
    Raw,
    Struct,
    Timestamp,
    Uint32,
    decode,
    encode,
    is_valid_json,
)
from unreal_agent.operation.operation import (
    ID,
    Operation,
    OperationError,
    PrimitiveDispatch,
    Spec,
    Status,
    Step,
    unsupported,
)
from unreal_agent.operation.output import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    bound_output,
    utf8_len,
)

__all__ = [
    "TYPE_REMOTE_JOB",
    "VERSION_REMOTE_JOB",
    "RemoteJobHandler",
    "RemoteJobPlan",
    "RemoteJobState",
    "cancel_remote_job",
    "decode_remote_job_state",
    "fail_remote_job",
    "new_remote_job_spec",
    "update_remote_job",
]

TYPE_REMOTE_JOB: Final = "remote_job"
VERSION_REMOTE_JOB: Final = 2

type RemoteJobPlanType = str
type RemoteJobPlanVersion = int


class RemoteJobPlan(Struct):
    Type: str = ""
    Version: Uint32 = 0
    Data: Raw = Raw(b"")


class RemoteJobHandler(Protocol):
    """Runs remote jobs for one plan type and version.

    `remote_job_updates` is iterated once for the manager's lifetime; ending the
    iteration means the handler stopped, and its jobs fail.
    """

    def remote_job_plan_type(self) -> RemoteJobPlanType: ...

    def remote_job_plan_version(self) -> RemoteJobPlanVersion: ...

    async def add_remote_job(self, operation: Operation, /) -> None: ...

    async def cancel_remote_job(self, id: ID, reason: str, /) -> None: ...

    def remote_job_updates(self) -> AsyncGenerator[Operation]: ...


class RemoteJobState(Struct):
    Plan: RemoteJobPlan = msgspec.field(default_factory=RemoteJobPlan)
    Handle: Annotated[Raw, OmitZero] = Raw(b"")
    NextInspectionAt: Annotated[Timestamp, OmitZero] = Timestamp()
    OutstandingInput: Annotated[Raw, OmitZero] = Raw(b"")
    Subscription: Annotated[Raw, OmitZero] = Raw(b"")
    TerminalResult: Annotated[str, OmitZero] = ""
    TerminalError: Annotated[str, OmitZero] = ""
    ResultBytes: Annotated[Int64, OmitZero] = 0
    ResultTruncated: Annotated[bool, OmitZero] = False
    ErrorBytes: Annotated[Int64, OmitZero] = 0
    ErrorTruncated: Annotated[bool, OmitZero] = False


def new_remote_job_spec(plan: RemoteJobPlan) -> Spec:
    validate_remote_job_plan(plan)
    return Spec(
        MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
        Type=TYPE_REMOTE_JOB,
        Version=VERSION_REMOTE_JOB,
        State=Raw(encode(RemoteJobState(Plan=plan))),
    )


def decode_remote_job_state(current: Operation) -> RemoteJobState:
    operation_id = go_quote(current.ID)
    if current.Type != TYPE_REMOTE_JOB:
        raise unsupported(
            f"advance remote job operation {operation_id}: type {go_quote(current.Type)}"
        )
    if current.Version != VERSION_REMOTE_JOB:
        raise unsupported(f"advance remote job operation {operation_id}: version {current.Version}")
    if current.MaxOutputLength <= 0:
        raise OperationError("max output length must be positive")
    if current.MaxOutputLength > MAX_OUTPUT_LENGTH:
        raise OperationError(f"max output length must not exceed {MAX_OUTPUT_LENGTH}")
    try:
        state = decode(bytes(current.State), RemoteJobState)
    except JSONError as error:
        raise OperationError(
            f"decode remote job operation {operation_id} state: {error}"
        ) from error
    try:
        validate_remote_job_plan(state.Plan)
    except OperationError as error:
        raise OperationError(
            f"validate remote job operation {operation_id} state: {error}"
        ) from error
    return state


def update_remote_job(
    current: Operation, state: RemoteJobState, status: Status, *dispatches: PrimitiveDispatch
) -> Step:
    decode_remote_job_state(current)
    try:
        validate_remote_job_plan(state.Plan)
    except OperationError as error:
        raise OperationError(
            f"validate remote job operation {go_quote(current.ID)} state: {error}"
        ) from error
    current = replace(current, Status=status)
    state = replace(state)
    _prepare_remote_job_output(current, state)
    return Step(operation=replace(current, State=Raw(encode(state))), dispatches=list(dispatches))


def fail_remote_job(current: Operation, error: BaseException | str | None) -> Step:
    if error is None:
        raise OperationError("fail remote job operation: error must be set")
    state = decode_remote_job_state(current)
    state.TerminalError = str(error)
    state.ErrorBytes, state.ErrorTruncated = 0, False
    state.TerminalResult = ""
    state.Subscription = Raw(b"")
    return update_remote_job(current, state, Status.FAILED)


def cancel_remote_job(current: Operation) -> Step:
    state = decode_remote_job_state(current)
    state.TerminalResult = ""
    state.Subscription = Raw(b"")
    return update_remote_job(current, state, Status.CANCELED)


def validate_remote_job_plan(plan: RemoteJobPlan) -> None:
    if not plan.Type:
        raise OperationError("remote job plan type must be set")
    if plan.Version == 0:
        raise OperationError("remote job plan version must be positive")
    if len(plan.Data) == 0 or not is_valid_json(bytes(plan.Data)):
        raise OperationError("remote job plan data must be valid JSON")


def _prepare_remote_job_output(current: Operation, state: RemoteJobState) -> None:
    # TODO: Capture the full result in a file before truncating and expose its path.
    limit = current.MaxOutputLength
    if not state.ResultTruncated or not state.TerminalResult:
        state.ResultBytes = utf8_len(state.TerminalResult)
        state.TerminalResult, state.ResultTruncated = bound_output(state.TerminalResult, limit)
    if not state.ErrorTruncated or not state.TerminalError:
        state.ErrorBytes = utf8_len(state.TerminalError)
        state.TerminalError, state.ErrorTruncated = bound_output(state.TerminalError, limit)
