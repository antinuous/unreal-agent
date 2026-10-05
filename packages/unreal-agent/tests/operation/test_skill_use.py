import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Final

import pytest
from unreal_agent.gojson import Raw, encode
from unreal_agent.operation import (
    TYPE_SKILL_USE,
    VERSION_SKILL_USE,
    LocalOperationManager,
    Operation,
    PrimitiveDispatch,
    SkillUseState,
    Status,
    Step,
    advance_skill_use,
    decode_skill_use,
    new_skill_use_spec,
)
from unreal_agent.primitives import (
    IOReadRequest,
    PrimitiveDispatchType,
)

TERMINAL_STATUSES: Final = frozenset({Status.COMPLETED, Status.FAILED, Status.CANCELED})


def one_dispatch_data[T](step: Step, dispatch_type: PrimitiveDispatchType, kind: type[T]) -> T:
    assert len(step.dispatches) == 1, step.dispatches
    assert step.dispatches[0].type == dispatch_type, step.dispatches
    return dispatch_data(step.dispatches[0], kind)


def dispatch_data[T](dispatch: PrimitiveDispatch, kind: type[T]) -> T:
    assert isinstance(dispatch.data, kind), dispatch.data
    return dispatch.data


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


type NewManager = Callable[..., LocalOperationManager]


def new_skill_use_operation(operation_id: str, path: str) -> Operation:
    spec = new_skill_use_spec(path)
    return Operation(
        ID=operation_id, Type=spec.Type, Version=spec.Version, Status=Status.READY, State=spec.State
    )


def skill_use_operation_with_state(
    operation_id: str, status: Status, state: SkillUseState
) -> Operation:
    return Operation(
        ID=operation_id,
        Type=TYPE_SKILL_USE,
        Version=VERSION_SKILL_USE,
        Status=status,
        State=Raw(encode(state)),
    )


async def test_local_operation_manager_loads_entire_skill(
    tmp_path: Path, new_manager: NewManager
) -> None:
    path = tmp_path / "SKILL.md"
    want = "---\nname: review\ndescription: Review code.\n---\n\n" + "instruction\n" * 6_000
    path.write_text(want)
    current = new_skill_use_operation("skill-use", str(path))
    manager = new_manager()
    await manager.add(current)
    completed = await receive_terminal_operation(manager.updates(), current.ID)
    state = decode_skill_use(completed)
    assert completed.Status == Status.COMPLETED
    assert state.Content == want.encode()
    assert state.TerminalError == ""


async def test_skill_use_operation_resumes_partial_read(
    tmp_path: Path, new_manager: NewManager
) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("complete skill")
    current = skill_use_operation_with_state(
        "skill-use-resume", Status.AWAITING, SkillUseState(Path=str(path), Content=b"complete ")
    )
    step = advance_skill_use(current, None)
    request = one_dispatch_data(step, PrimitiveDispatchType.IO_READ, IOReadRequest)
    assert request.path == str(path)
    assert request.offset == len("complete ")
    assert request.count == 2**63 - 1

    manager = new_manager()
    await manager.add(current)
    completed = await receive_terminal_operation(manager.updates(), current.ID)
    state = decode_skill_use(completed)
    assert completed.Status == Status.COMPLETED
    assert state.Content == b"complete skill"


async def test_skill_use_operation_reports_read_failure(
    tmp_path: Path, new_manager: NewManager
) -> None:
    current = new_skill_use_operation("skill-use-missing", str(tmp_path / "missing" / "SKILL.md"))
    manager = new_manager()
    await manager.add(current)
    failed = await receive_terminal_operation(manager.updates(), current.ID)
    state = decode_skill_use(failed)
    assert failed.Status == Status.FAILED
    assert "open" in state.TerminalError
