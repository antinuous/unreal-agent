import pytest
from unreal_agent.gojson import Raw, encode
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    TYPE_SKILL_USE,
    VERSION_SKILL_USE,
    Operation,
    SkillUseState,
    Status,
)
from unreal_agent.tool import (
    BASH_NAME,
    SKILL_USE_NAME,
    VIEW_IMAGE_NAME,
    CallStatus,
    Registry,
    StaticTranslators,
    TranslationError,
    Translator,
)

from tests.tool._helpers import RecordingContext


def text(value: str) -> ToolResult:
    return ToolResult(
        CallID="call", Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]
    )


def skill_use() -> Translator:
    translator = Registry(StaticTranslators(), SKILL_USE_NAME).resolve(SKILL_USE_NAME)
    assert translator is not None
    return translator


def skill_operation(status: Status, state: SkillUseState) -> Operation:
    return Operation(
        ID="operation-1",
        Type=TYPE_SKILL_USE,
        Version=VERSION_SKILL_USE,
        Status=status,
        State=Raw(encode(state)),
    )


@pytest.mark.parametrize("name", [BASH_NAME, VIEW_IMAGE_NAME])
def test_unconfigured_static_tools_report_errors_without_label(name: str) -> None:
    translator = Registry(StaticTranslators(), name).resolve(name)
    assert translator is not None
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Arguments="{}"))
    want = f'static tool "{name}" is not configured'
    assert status == CallStatus(Error=want)
    assert ctx.specs == []
    assert translator.translate_result("call", status, []) == text(want)


def test_skill_use_validation_error_has_no_label() -> None:
    assert skill_use().translate_result("call", CallStatus(Error="bad"), []) == text("bad")


@pytest.mark.parametrize(
    ("status", "state", "want"),
    [
        pytest.param(
            Status.COMPLETED,
            SkillUseState(Path="/s", Content=b"ok\xff\xfe!"),
            "ok\ufffd!",
            id="completed",
        ),
        pytest.param(Status.READY, SkillUseState(Path="/s"), "Skill is loading.", id="ready"),
        pytest.param(Status.AWAITING, SkillUseState(Path="/s"), "Skill is loading.", id="awaiting"),
        pytest.param(
            Status.CANCELING, SkillUseState(Path="/s"), "Skill is loading.", id="canceling"
        ),
        pytest.param(
            Status.FAILED,
            SkillUseState(Path="/s", TerminalError="open /s: denied"),
            "open /s: denied",
            id="failed",
        ),
        pytest.param(
            Status.CANCELED,
            SkillUseState(Path="/s", TerminalError="skill-use operation canceled"),
            "skill-use operation canceled",
            id="canceled",
        ),
    ],
)
def test_skill_use_results(status: Status, state: SkillUseState, want: str) -> None:
    operation = skill_operation(status, state)
    assert skill_use().translate_result("call", CallStatus(), [operation]) == text(want)


@pytest.mark.parametrize(
    ("operations", "want"),
    [
        pytest.param([], 'skill-use call "call" has 0 operations, want 1', id="none"),
        pytest.param(
            [
                skill_operation(Status.FAILED, SkillUseState(Path="/s")),
            ],
            'skill-use call "call" terminal operation "operation-1" has no error',
            id="terminal without error",
        ),
        pytest.param(
            [skill_operation("unknown", SkillUseState(Path="/s"))],  # type: ignore[arg-type]
            'skill-use call "call" operation "operation-1" has unsupported status "unknown"',
            id="unsupported status",
        ),
        pytest.param(
            [Operation(ID="operation-1", Type="shell", Version=VERSION_SKILL_USE)],
            'decode skill-use call "call" result: ',
            id="wrong type",
        ),
    ],
)
def test_skill_use_rejects_invalid_results(operations: list[Operation], want: str) -> None:
    with pytest.raises(TranslationError) as raised:
        skill_use().translate_result("call", CallStatus(), operations)
    assert str(raised.value).startswith(want)


@pytest.mark.parametrize("arguments", ["", "  \n"])
def test_skill_use_treats_blank_arguments_as_empty_object(arguments: str) -> None:
    status = skill_use().translate(RecordingContext(), ToolCall(Arguments=arguments))
    assert status == CallStatus(Error='skill-use argument "name" must be set')
