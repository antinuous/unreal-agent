import asyncio
import errno
import os
from collections.abc import AsyncIterator, Callable
from typing import Final

import pytest
from unreal_agent.gojson import Raw, decode, encode
from unreal_agent.operation import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    SHELL_ERR_FILENAME,
    SHELL_OUT_FILENAME,
    TYPE_SHELL,
    VERSION_SHELL,
    Operation,
    PrimitiveDispatch,
    Shell,
    ShellInput,
    ShellState,
    Status,
    Step,
    new_shell_spec,
)
from unreal_agent.primitives import PrimitiveDispatchType, PrimitiveEvent


class Panic(BaseException):
    pass


def assert_process_group_gone(process_group_id: int) -> None:
    with pytest.raises(ProcessLookupError) as error:
        os.killpg(process_group_id, 0)
    assert error.value.errno == errno.ESRCH


def advance_shell_once(current: Operation, event: PrimitiveEvent | None) -> Step:
    return Shell(current).handle(event)


def create_shell_artifacts(directory: str, out: bytes, err: bytes) -> None:
    os.mkdir(directory, 0o700)
    for name, contents in ((SHELL_OUT_FILENAME, out), (SHELL_ERR_FILENAME, err)):
        with open(os.path.join(directory, name), "wb") as file:
            file.write(contents)
        os.chmod(os.path.join(directory, name), 0o600)


def dispatch_data[T](dispatch: PrimitiveDispatch, kind: type[T]) -> T:
    assert isinstance(dispatch.data, kind), dispatch.data
    return dispatch.data


def one_dispatch_data[T](step: Step, dispatch_type: PrimitiveDispatchType, kind: type[T]) -> T:
    assert len(step.dispatches) == 1, step.dispatches
    assert step.dispatches[0].type == dispatch_type, step.dispatches
    return dispatch_data(step.dispatches[0], kind)


def new_shell_operation(
    operation_id: str, input: ShellInput, base_directory: str, max_output_length: int
) -> Operation:
    spec = new_shell_spec(input, base_directory, max_output_length)
    return Operation(
        ID=operation_id,
        Type=spec.Type,
        Version=spec.Version,
        Status=Status.READY,
        MaxOutputLength=spec.MaxOutputLength,
        State=spec.State,
        Idempotency=spec.Idempotency,
    )


async def receive_operation(
    updates: AsyncIterator[Operation],
    operation_id: str,
    accept: Callable[[Operation], bool],
) -> Operation:
    async with asyncio.timeout(15):
        async for current in updates:
            if operation_id == current.ID and accept(current):
                return current
    pytest.fail("updates closed before operation completed")


async def receive_terminal_operation(
    updates: AsyncIterator[Operation], operation_id: str
) -> Operation:
    return await receive_operation(
        updates, operation_id, lambda current: current.Status in TERMINAL_STATUSES
    )


def shell_state(current: Operation) -> ShellState:
    return decode(bytes(current.State), ShellState)


def shell_operation_with_state(operation_id: str, state: ShellState) -> Operation:
    return Operation(
        ID=operation_id,
        MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
        Type=TYPE_SHELL,
        Version=VERSION_SHELL,
        Status=Status.AWAITING,
        State=Raw(encode(state)),
    )


TERMINAL_STATUSES: Final = frozenset(
    {Status.COMPLETED, Status.FAILED, Status.CANCELED, Status.UNSUPPORTED}
)
