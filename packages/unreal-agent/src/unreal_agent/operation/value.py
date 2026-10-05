from typing import Final

from msgspec.structs import replace

from unreal_agent._go import go_quote
from unreal_agent.gojson import JSONError, Raw, Struct, decode, encode, is_valid_json
from unreal_agent.operation.operation import (
    Operation,
    OperationError,
    Spec,
    Status,
    Step,
    unsupported,
)
from unreal_agent.primitives import PrimitiveEvent

__all__ = [
    "TYPE_VALUE",
    "VERSION_VALUE",
    "ValueState",
    "advance_value",
    "decode_value",
    "new_value_spec",
]

TYPE_VALUE: Final = "value"
VERSION_VALUE: Final = 1


class ValueState(Struct):
    Value: Raw = Raw(b"")


def new_value_spec(value: Raw | bytes) -> Spec:
    if len(value) == 0 or not is_valid_json(bytes(value)):
        raise OperationError("value operation result must be valid JSON")
    encoded = encode(ValueState(Value=Raw(bytes(value))))
    return Spec(Type=TYPE_VALUE, Version=VERSION_VALUE, State=Raw(encoded))


def decode_value(current: Operation) -> Raw:
    if current.Type != TYPE_VALUE:
        raise unsupported(
            f"decode value operation {go_quote(current.ID)}: type {go_quote(current.Type)}"
        )
    if current.Version != VERSION_VALUE:
        raise unsupported(
            f"decode value operation {go_quote(current.ID)}: version {current.Version}"
        )
    try:
        state = decode(bytes(current.State), ValueState)
    except JSONError as error:
        raise OperationError(
            f"decode value operation {go_quote(current.ID)} state: {error}"
        ) from error
    if len(state.Value) == 0 or not is_valid_json(bytes(state.Value)):
        raise OperationError(
            f"decode value operation {go_quote(current.ID)} state: result must be valid JSON"
        )
    return Raw(bytes(state.Value))


def advance_value(current: Operation, event: PrimitiveEvent | None) -> Step:
    decode_value(current)
    if event is not None:
        raise OperationError(
            f"advance value operation {go_quote(current.ID)}: unexpected primitive event"
        )
    match current.Status:
        case Status.READY:
            status = Status.COMPLETED
        case Status.CANCELING:
            status = Status.CANCELED
        case _:
            raise unsupported(
                f"advance value operation {go_quote(current.ID)}: status {go_quote(current.Status)}"
            )
    return Step(operation=replace(current, Status=status))
