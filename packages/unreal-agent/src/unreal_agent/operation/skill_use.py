from typing import Final

from msgspec.structs import replace

from unreal_agent._go import go_quote
from unreal_agent.gojson import JSONError, Raw, Struct, decode, encode
from unreal_agent.operation.operation import (
    Operation,
    OperationError,
    PrimitiveDispatch,
    Spec,
    Status,
    Step,
    unsupported,
)
from unreal_agent.primitives import (
    CorrelationID,
    IOReadCompletedResult,
    IOReadOutputResult,
    IOReadRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
)

__all__ = [
    "TYPE_SKILL_USE",
    "VERSION_SKILL_USE",
    "SkillUseState",
    "advance_skill_use",
    "decode_skill_use",
    "new_skill_use_spec",
]

TYPE_SKILL_USE: Final = "skill_use"
VERSION_SKILL_USE: Final = 1
_MAX_INT64: Final = 2**63 - 1

SKILL_USE_READ_CORRELATION: Final = CorrelationID("skill-use-read")


class SkillUseState(Struct):
    Path: str = ""
    Content: bytes = b""
    TerminalError: str = ""


def new_skill_use_spec(path: str) -> Spec:
    state = SkillUseState(Path=path)
    _validate(state)
    return Spec(Type=TYPE_SKILL_USE, Version=VERSION_SKILL_USE, State=Raw(encode(state)))


def decode_skill_use(current: Operation) -> SkillUseState:
    return skill_use_operation_state(current)


def advance_skill_use(current: Operation, event: PrimitiveEvent | None) -> Step:
    state = skill_use_operation_state(current)
    operation_id = go_quote(current.ID)
    match current.Status:
        case Status.READY:
            if event is not None:
                raise OperationError(
                    f"advance ready skill-use operation {operation_id}: unexpected primitive event"
                )
            if state.Content or state.TerminalError:
                raise OperationError(
                    f"advance ready skill-use operation {operation_id}: state is not initial"
                )
            return _dispatch_read(current, state)
        case Status.AWAITING:
            if state.TerminalError:
                raise OperationError(
                    f"advance awaiting skill-use operation {operation_id}: state is terminal"
                )
            if event is None:
                return _dispatch_read(current, state)
            return _advance_awaiting(current, state, event)
        case Status.CANCELING:
            if event is not None:
                raise OperationError(
                    f"advance canceling skill-use operation {operation_id}: "
                    "unexpected primitive event"
                )
            return cancel_skill_use(current, state)
        case _:
            raise OperationError(
                f"advance skill-use operation {operation_id}: "
                f"terminal status {go_quote(current.Status)}"
            )


def skill_use_operation_state(current: Operation) -> SkillUseState:
    operation_id = go_quote(current.ID)
    if current.Type != TYPE_SKILL_USE:
        raise unsupported(
            f"decode skill-use operation {operation_id}: type {go_quote(current.Type)}"
        )
    if current.Version != VERSION_SKILL_USE:
        raise unsupported(f"decode skill-use operation {operation_id}: version {current.Version}")
    try:
        state = decode(bytes(current.State), SkillUseState)
    except JSONError as error:
        raise OperationError(f"decode skill-use operation {operation_id} state: {error}") from error
    try:
        _validate(state)
    except OperationError as error:
        raise OperationError(
            f"validate skill-use operation {operation_id} state: {error}"
        ) from error
    return state


def _validate(state: SkillUseState) -> None:
    if not state.Path:
        raise OperationError("skill path must be set")


def _advance_awaiting(current: Operation, state: SkillUseState, event: PrimitiveEvent) -> Step:
    if event.source != current.ID:
        return fail_skill_use(
            current,
            state,
            f"skill-use primitive event source is {go_quote(event.source)}, "
            f"want {go_quote(current.ID)}",
        )
    if event.type == PrimitiveEventType.CANCELED:
        return cancel_skill_use(current, state)
    if event.type == PrimitiveEventType.FAILED:
        if not isinstance(event.result, PrimitiveFailureResult):
            return fail_skill_use(
                current, state, "skill-use primitive failed with an invalid result"
            )
        return fail_skill_use(current, state, event.result.error)
    if event.correlation_id != SKILL_USE_READ_CORRELATION:
        return fail_skill_use(
            current,
            state,
            f"skill-use primitive event correlation is {go_quote(event.correlation_id)}, "
            f"want {go_quote(SKILL_USE_READ_CORRELATION)}",
        )
    match event.type:
        case PrimitiveEventType.IO_READ_OUTPUT:
            output = event.result
            if not isinstance(output, IOReadOutputResult) or output.offset != len(state.Content):
                return fail_skill_use(current, state, "skill read returned invalid output")
            state.Content += output.data
            return _await(current, state)
        case PrimitiveEventType.IO_READ_COMPLETED:
            result = event.result
            if not isinstance(result, IOReadCompletedResult) or result.size != len(state.Content):
                return fail_skill_use(current, state, "skill read returned an invalid completion")
            return _step(replace(current, Status=Status.COMPLETED), state)
        case _:
            return fail_skill_use(
                current, state, f"skill read returned unexpected event {go_quote(event.type)}"
            )


def _dispatch_read(current: Operation, state: SkillUseState) -> Step:
    step = _await(current, state)
    step.dispatches = [
        PrimitiveDispatch(
            PrimitiveDispatchType.IO_READ,
            IOReadRequest(
                source=SourceID(current.ID),
                correlation_id=SKILL_USE_READ_CORRELATION,
                path=state.Path,
                offset=len(state.Content),
                count=_MAX_INT64,
            ),
        )
    ]
    return step


def _await(current: Operation, state: SkillUseState) -> Step:
    return _step(replace(current, Status=Status.AWAITING), state)


def _step(current: Operation, state: SkillUseState) -> Step:
    return Step(operation=replace(current, State=Raw(encode(state))))


def fail_skill_use(current: Operation, state: SkillUseState, error: BaseException | str) -> Step:
    state = replace(state, TerminalError=str(error))
    return _step(replace(current, Status=Status.FAILED), state)


def cancel_skill_use(current: Operation, state: SkillUseState) -> Step:
    state = replace(state, TerminalError="skill-use operation canceled")
    return _step(replace(current, Status=Status.CANCELED), state)
