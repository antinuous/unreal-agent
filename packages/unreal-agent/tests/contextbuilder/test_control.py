import pytest
from msgspec.structs import replace
from unreal_agent.contextbuilder import Result, new_builder
from unreal_agent.inbox import ControlMessage, ControlMode, Settings
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


def test_builder_settings_only_change_effort_in_subsequent_requests() -> None:
    builder = new_builder()
    builder.set_model(Model(ID="model", MaxOutputTokens=123, ReasoningEffort=ReasoningEffort.HIGH))
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
