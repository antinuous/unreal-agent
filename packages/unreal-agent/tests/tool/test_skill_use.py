import asyncio
import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest
from unreal_agent.gojson import Raw, encode
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    TYPE_SKILL_USE,
    VERSION_SKILL_USE,
    LocalOperationManager,
    Operation,
    SkillUseState,
    Status,
    advance_skill_use,
)
from unreal_agent.primitives import (
    IOReadCompletedResult,
    IOReadRequest,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
)
from unreal_agent.tool import (
    SKILL_USE_NAME,
    CallStatus,
    Registry,
    Skill,
    SkillUseResult,
    SkillUseTranslator,
    StaticTranslators,
)

from tests.tool._helpers import RecordingContext

_PARTIAL = b"partial skill instructions"

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
    registry = Registry(StaticTranslators(), SKILL_USE_NAME)
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
    assert result == SkillUseResult(CallID="call-1", Content=want.encode())
    model = result.to_llm_result()
    assert model.CallID == "call-1"
    assert model.Output[0].Value == want


@pytest.mark.parametrize(
    ("event", "want"),
    [
        pytest.param(
            PrimitiveEvent(type=PrimitiveEventType.CANCELED),
            "skill-use operation canceled",
            id="canceled",
        ),
        pytest.param(
            PrimitiveEvent(
                type=PrimitiveEventType.FAILED, result=PrimitiveFailureResult(error="read failed")
            ),
            "read failed",
            id="read failed",
        ),
        pytest.param(
            PrimitiveEvent(
                type=PrimitiveEventType.IO_READ_COMPLETED,
                result=IOReadCompletedResult(size=len(_PARTIAL) + 1),
            ),
            "skill read returned an invalid completion",
            id="size mismatch",
        ),
    ],
)
def test_skill_use_failure_keeps_partial_content_out_of_model_output(
    event: PrimitiveEvent, want: str
) -> None:
    current = Operation(
        ID="skill-operation",
        Type=TYPE_SKILL_USE,
        Version=VERSION_SKILL_USE,
        Status=Status.AWAITING,
        State=Raw(encode(SkillUseState(Path="SKILL.md", Content=_PARTIAL))),
    )
    step = advance_skill_use(current, None)
    assert step.operation is not None
    read = step.dispatches[0].data
    assert isinstance(read, IOReadRequest)
    terminal = advance_skill_use(
        step.operation,
        dataclasses.replace(event, source=read.source, correlation_id=read.correlation_id),
    )
    assert terminal.operation is not None
    result = SkillUseTranslator(lambda _: None).translate_result(
        "skill-call", CallStatus(), [terminal.operation]
    )
    assert isinstance(result, SkillUseResult)
    assert (result.Content, result.Error) == (_PARTIAL, want)
    assert result.to_llm_result() == ToolResult(
        CallID="skill-call", Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=want)]
    ), "model result exposed partial instructions"


def test_skill_use_rejects_other_tool_name() -> None:
    registry = Registry(StaticTranslators(), SKILL_USE_NAME)
    translator = registry.resolve(SKILL_USE_NAME)
    assert translator is not None
    status = translator.translate(RecordingContext(), ToolCall(Name="Other"))
    assert "does not match" in status.Error
    assert not status.WaitingFor
