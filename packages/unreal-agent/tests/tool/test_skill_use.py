import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from unreal_agent.llm import ToolCall
from unreal_agent.operation import LocalOperationManager, Operation, Status
from unreal_agent.tool import SKILL_USE_NAME, Skill, StaticTranslators, new_registry

from tests.tool._helpers import RecordingContext

TERMINAL = frozenset({Status.COMPLETED, Status.FAILED, Status.CANCELED})


async def receive_terminal(manager: LocalOperationManager, operation_id: str) -> Operation:
    async with asyncio.timeout(15):
        async for current in manager.updates():
            if operation_id == current.ID and current.Status in TERMINAL:
                return current
    pytest.fail("updates closed before the skill-use operation finished")


async def test_skill_use_loads_registered_skill_by_name(
    tmp_path: Path, new_manager: Callable[[], LocalOperationManager]
) -> None:
    path = tmp_path / "SKILL.md"
    want = "---\nname: review\ndescription: Review code.\n---\n\nRead all instructions.\n"
    path.write_text(want)
    path.chmod(0o600)
    registry = new_registry(StaticTranslators(), SKILL_USE_NAME)
    registry.register_skill(Skill(name="review", description="Review code.", path=str(path)))
    translator = registry.resolve(SKILL_USE_NAME)
    assert translator is not None
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Name=SKILL_USE_NAME, Arguments='{"name":"review"}'))
    assert status.Error == ""
    assert len(ctx.specs) == 1
    assert status.WaitingFor == ["operation-1"]

    spec = ctx.specs[0]
    current = Operation(
        ID="operation-1",
        Type=spec.Type,
        Version=spec.Version,
        Status=Status.READY,
        State=spec.State,
    )
    manager = new_manager()
    await manager.add(current)
    completed = await receive_terminal(manager, current.ID)
    result = translator.translate_result("call-1", status, [completed])
    assert result.CallID == "call-1"
    assert result.Output[0].Value == want


@pytest.mark.parametrize(
    ("call", "want_error"),
    [
        pytest.param(ToolCall(Name="Other"), "does not match", id="wrong tool"),
        pytest.param(ToolCall(Arguments="{"), "decode skill-use arguments", id="invalid JSON"),
        pytest.param(ToolCall(Arguments="{}"), 'argument "name" must be set', id="missing name"),
        pytest.param(
            ToolCall(Arguments='{"name":"missing"}'),
            'skill "missing" is not registered',
            id="unknown skill",
        ),
    ],
)
def test_skill_use_rejects_invalid_selection(call: ToolCall, want_error: str) -> None:
    registry = new_registry(StaticTranslators(), SKILL_USE_NAME)
    translator = registry.resolve(SKILL_USE_NAME)
    assert translator is not None
    status = translator.translate(RecordingContext(), call)
    assert want_error in status.Error
    assert not status.WaitingFor
