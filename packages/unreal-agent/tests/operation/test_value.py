import asyncio
from collections.abc import Callable

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import Raw
from unreal_agent.operation import (
    TYPE_SHELL,
    TYPE_VALUE,
    VERSION_VALUE,
    LocalOperationManager,
    Operation,
    OperationError,
    Status,
    UnsupportedError,
    advance_value,
    decode_value,
    new_value_spec,
)
from unreal_agent.primitives import PrimitiveEvent, PrimitiveEventType

type NewManager = Callable[..., LocalOperationManager]


def _value_operation(value: bytes, status: Status = Status.READY) -> Operation:
    spec = new_value_spec(Raw(value))
    return Operation(
        ID="value-1", Type=spec.Type, Version=spec.Version, Status=status, State=spec.State
    )


def test_value_spec_carries_inline_json_result() -> None:
    want = b'{"tools":[{"name":"search"}]}'
    current = _value_operation(want)
    got = decode_value(current)
    assert bytes(got) == want
    # Raw values are immutable, so the stored value cannot change through the result.
    assert bytes(decode_value(current)) == b'{"tools":[{"name":"search"}]}'


@pytest.mark.parametrize("value", [b"", b"{"])
def test_new_value_spec_rejects_invalid_json(value: bytes) -> None:
    with pytest.raises(OperationError, match=r"^value operation result must be valid JSON$"):
        new_value_spec(Raw(value))


def test_advance_value_completes_without_primitive_dispatch() -> None:
    current = _value_operation(b'{"result":42}')
    step = advance_value(current, None)
    assert step.operation is not None
    assert step.operation.Status == Status.COMPLETED
    assert not step.dispatches

    canceled = advance_value(replace(current, Status=Status.CANCELING), None)
    assert canceled.operation is not None
    assert canceled.operation.Status == Status.CANCELED
    assert not canceled.dispatches

    with pytest.raises(OperationError, match="unexpected primitive event"):
        advance_value(current, PrimitiveEvent(PrimitiveEventType.FAILED))

    with pytest.raises(UnsupportedError):
        advance_value(replace(current, Status=Status.AWAITING), None)


async def test_local_operation_manager_completes_value(new_manager: NewManager) -> None:
    current = _value_operation(b'{"result":42}')
    manager = new_manager()
    await manager.add(current)
    async with asyncio.timeout(15):
        completed = await anext(manager.updates())
    assert completed.Status == Status.COMPLETED
    assert bytes(decode_value(completed)) == b'{"result":42}'


@pytest.mark.parametrize(
    ("operation_id", "type_", "version_offset"),
    [("wrong-type", TYPE_SHELL, 0), ("wrong-version", TYPE_VALUE, 1)],
)
def test_decode_value_rejects_other_types_and_versions(
    operation_id: str, type_: str, version_offset: int
) -> None:
    spec = new_value_spec(Raw(b"null"))
    current = Operation(
        ID=operation_id, Type=type_, Version=spec.Version + version_offset, State=spec.State
    )
    with pytest.raises(UnsupportedError):
        decode_value(current)


@pytest.mark.parametrize("state", [b"[]", b"{}", b'{"Value":'])
def test_decode_and_advance_value_reject_malformed_state(state: bytes) -> None:
    current = Operation(
        ID="value-1",
        Type=TYPE_VALUE,
        Version=VERSION_VALUE,
        Status=Status.READY,
        State=Raw(state),
    )
    with pytest.raises(OperationError, match='decode value operation "value-1" state'):
        decode_value(current)
    with pytest.raises(OperationError, match='decode value operation "value-1" state'):
        advance_value(current, None)


async def test_local_operation_manager_delivers_canceled_value(new_manager: NewManager) -> None:
    current = _value_operation(b'{"result":42}', Status.CANCELING)
    manager = new_manager()
    await manager.add(current)
    async with asyncio.timeout(15):
        canceled = await anext(manager.updates())
    assert canceled.Status == Status.CANCELED
    assert bytes(decode_value(canceled)) == b'{"result":42}'
