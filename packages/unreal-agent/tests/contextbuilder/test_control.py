import pytest
from msgspec.structs import replace
from unreal_agent.contextbuilder import PREAMBLE, Result, new_builder
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ControlMessage, ControlMode, Input, InputKind, Settings
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    ReasoningEffort,
    Response,
    Role,
    Tool,
)

from tests.contextbuilder._helpers import reasoning


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        (ControlMode.HEARTBEAT, "requested"),
        (ControlMode.STOP_HARD, ""),
        (ControlMode.STOP_WHEN_IDLE, ""),
    ],
)
def test_builder_control_messages(mode: ControlMode, text: str) -> None:
    builder = new_builder()
    builder.add_control_message(ControlMessage(Mode=mode, Reason="requested"))
    input = builder.build().request.Input[1:]
    if not text:
        assert input == []
        return
    assert len(input) == 1
    assert input[0].Type == ItemType.MESSAGE
    assert input[0].Data == Message(Role=Role.USER, Text=text)


_INITIAL_MODEL = Model(
    ID="initial",
    CompactionThreshold=200_000,
    MaxOutputTokens=123,
    ReasoningEffort=ReasoningEffort.HIGH,
)


@pytest.mark.parametrize(
    ("settings", "model"),
    [
        pytest.param(Settings(), _INITIAL_MODEL, id="omitted"),
        pytest.param(
            Settings(Model="next", CompactionThreshold=80_000),
            replace(_INITIAL_MODEL, ID="next", CompactionThreshold=80_000),
            id="model",
        ),
        pytest.param(
            Settings(Model="next"), replace(_INITIAL_MODEL, ID="next"), id="model without threshold"
        ),
        pytest.param(
            Settings(CompactionThreshold=80_000),
            replace(_INITIAL_MODEL, CompactionThreshold=80_000),
            id="threshold without model",
        ),
        pytest.param(
            Settings(CompactionThreshold=0),
            replace(_INITIAL_MODEL, CompactionThreshold=0),
            id="zero threshold",
        ),
        pytest.param(
            Settings(MaxOutputTokens=456), replace(_INITIAL_MODEL, MaxOutputTokens=456), id="limit"
        ),
    ],
)
def test_builder_settings_apply_supplied_fields(settings: Settings, model: Model) -> None:
    builder = new_builder()
    builder.set_model(_INITIAL_MODEL)
    builder.set_system_prompt("initial prompt")
    builder.add_tool(Tool(Name="tool"))
    builder.add_external_input(Input(ID="prompt", Kind=InputKind.EXTERNAL, Payload=Raw(b'"hello"')))
    original = builder.build()
    builder.add_control_message(
        ControlMessage(Mode=ControlMode.UPDATE_SETTINGS, Parameters=settings)
    )
    built = builder.build()
    assert built == Result(request=replace(original.request, Model=model), report=original.report)
    assert original.request.Model == _INITIAL_MODEL, "settings mutated an already built request"
    first = original.request.Input[0].Data
    assert isinstance(first, Message)
    assert first.Text == PREAMBLE + "\n\ninitial prompt", (
        "settings mutated an already built request"
    )


def test_builder_settings_only_change_effort_in_subsequent_requests() -> None:
    builder = new_builder()
    builder.set_model(
        Model(
            ID="model",
            CompactionThreshold=200_000,
            MaxOutputTokens=123,
            ReasoningEffort=ReasoningEffort.HIGH,
        )
    )
    builder.add_tool(Tool(Name="tool"))
    builder.add_model_response(
        Response(Output=[Item(Type=ItemType.PROVIDER, Data=reasoning("thought"))])
    )
    original = builder.build()
    builder.add_control_message(
        ControlMessage(
            Mode=ControlMode.UPDATE_SETTINGS,
            Parameters=Settings(ReasoningEffort=ReasoningEffort.LOW),
        )
    )
    built = builder.build()
    model = replace(original.request.Model, ReasoningEffort=ReasoningEffort.LOW)
    assert built == Result(request=replace(original.request, Model=model), report=original.report)
    assert original.request.Model.ReasoningEffort == ReasoningEffort.HIGH


def test_builder_settings_without_parameters_are_rejected() -> None:
    with pytest.raises(TypeError):
        new_builder().add_control_message(ControlMessage(Mode=ControlMode.UPDATE_SETTINGS))
