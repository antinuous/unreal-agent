import pytest
from unreal_agent.contextbuilder import PREAMBLE, new_builder
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ControlMessage, ControlMode, Input, InputKind
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Reasoning,
    Response,
    Role,
    ToolCall,
)

from tests.contextbuilder._helpers import text, tool_result


def message(role: Role, value: str, provider_id: str = "") -> Item:
    return Item(ProviderID=provider_id, Type=ItemType.MESSAGE, Data=Message(Role=role, Text=value))


def external(id: str, payload: bytes) -> Input:
    return Input(ID=id, Kind=InputKind.EXTERNAL, Payload=Raw(payload))


@pytest.mark.parametrize("empty_response", [False, True], ids=["full response", "empty response"])
def test_builder_places_response_before_unsubmitted_inputs(empty_response: bool) -> None:
    current = new_builder()
    current.add_external_input(external("first", b'"start"'))
    current.commit()
    current.add_model_response(
        Response(
            Output=[
                Item(
                    Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="A", Name="test", Arguments="{}")
                ),
                Item(
                    Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="B", Name="test", Arguments="{}")
                ),
            ]
        )
    )
    current.add_tool_result("A", text(""), True)
    current.add_tool_result("B", text("done B"), False)
    sent = current.build()
    original = list(sent.request.Input)
    current.commit()
    current.add_tool_result("A", text(""), True)
    current.add_tool_result("A", text("done A"), False)
    current.add_external_input(external("second", b'"continue"'))
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="heartbeat"))
    current.add_reasoning(Reasoning(Summary=["added reasoning"]))
    preview = current.build()
    suffix = [
        tool_result("A", "done A"),
        message(Role.USER, "continue"),
        message(Role.USER, "heartbeat"),
        Item(Type=ItemType.REASONING, Data=Reasoning(Summary=["added reasoning"])),
    ]
    want_preview = [*original, *suffix]
    assert preview.request.Input == want_preview

    output: list[Item] = [
        Item(
            ProviderID="reasoning",
            Type=ItemType.REASONING,
            Data=Reasoning(Summary=["response reasoning"]),
        ),
        message(Role.ASSISTANT, "working", "message"),
        Item(
            ProviderID="call",
            Type=ItemType.TOOL_CALL,
            Data=ToolCall(CallID="C", Name="test", Arguments="{}"),
        ),
    ]
    if empty_response:
        output = []
    current.add_model_response(Response(Output=output))
    after = current.build()
    want = [*original, *output, *suffix]
    assert after.request.Input == want
    assert sent.request.Input == original
    assert preview.request.Input == want_preview

    current.commit()
    current.commit()
    current.add_tool_result("C", text("done C"), False)
    next_output = message(Role.ASSISTANT, "next response")
    current.add_model_response(Response(Output=[next_output]))
    following = current.build()
    want = [*want, next_output, tool_result("C", "done C")]
    assert following.request.Input == want
    following.request.Input[1] = Item(Data=None)
    assert current.build().request.Input == want


def test_builder_submits_already_completed_results_without_delay() -> None:
    current = new_builder()
    current.add_tool_result("A", text(""), True)
    current.add_tool_result("A", text("done A"), False)
    want = [message(Role.SYSTEM, PREAMBLE), tool_result("A", "done A")]
    assert current.build().request.Input == want
    current.commit()
    response = message(Role.ASSISTANT, "done")
    current.add_model_response(Response(Output=[response]))
    assert current.build().request.Input == [*want, response]


def test_builder_updates_system_prompt_without_changing_conversation() -> None:
    current = new_builder()
    current.set_system_prompt("First instructions.")
    current.add_external_input(external("input", b'"start"'))
    current.commit()
    current.add_tool_result("A", text("done A"), False)
    before = current.build()
    for prompt in ("Updated instructions.", ""):
        current.set_system_prompt(prompt)
        after = current.build()
        want = PREAMBLE + ("\n\n" + prompt if prompt else "")
        assert after.request.Input[0].Data == Message(Role=Role.SYSTEM, Text=want)
        assert after.request.Input[1:] == before.request.Input[1:]
        first = before.request.Input[0].Data
        assert isinstance(first, Message)
        assert first.Text == PREAMBLE + "\n\nFirst instructions."
