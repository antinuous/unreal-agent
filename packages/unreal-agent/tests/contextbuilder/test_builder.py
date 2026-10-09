import pytest
from unreal_agent.contextbuilder import (
    PREAMBLE,
    TOOL_CALL_RUNNING_PAYLOAD,
    Report,
    Result,
    new_builder,
)
from unreal_agent.gojson import Raw, encode
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Request,
    Response,
    Role,
    StopReason,
    Tool,
    ToolCall,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    ToolType,
    Usage,
)
from unreal_agent.tool import Skill

from tests.contextbuilder._helpers import reasoning, text, tool_result


def with_preamble(*items: Item) -> list[Item]:
    """The items after the preamble every builder starts with."""
    return [Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text=PREAMBLE)), *items]


def user(value: str) -> Item:
    return Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=value))


def test_builder_adds_external_input_as_user_message() -> None:
    current = new_builder()
    current.add_external_input(
        Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(encode("Hello")))
    )
    assert current.build().request.Input == with_preamble(user("Hello"))


@pytest.mark.parametrize(
    ("input", "want"),
    [
        pytest.param(
            Input(ID="input-1", Kind=InputKind.CONTROL),
            'external input "input-1" has input kind "control"',
            id="wrong kind",
        ),
        pytest.param(
            Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(b"{")),
            'decode external input "input-1"',
            id="invalid payload",
        ),
    ],
)
def test_builder_rejects_invalid_external_input(input: Input, want: str) -> None:
    current = new_builder()
    with pytest.raises(ValueError, match=want):
        current.add_external_input(input)
    assert current.build().request.Input == with_preamble()


def test_builder_adds_model_response_output() -> None:
    output = [
        Item(
            ProviderID="reasoning-1",
            Type=ItemType.PROVIDER,
            Data=reasoning("Need current weather."),
        ),
        Item(
            ProviderID="message-1",
            Type=ItemType.MESSAGE,
            Data=Message(Role=Role.ASSISTANT, Text="Checking."),
        ),
        Item(
            ProviderID="call-item-1",
            Type=ItemType.TOOL_CALL,
            Data=ToolCall(CallID="call-1", Name="weather", Arguments='{"city":"London"}'),
        ),
    ]
    current = new_builder()
    current.add_model_response(
        Response(
            ID="response-1",
            Stop=StopReason.COMPLETE,
            Output=output,
            Usage=Usage(InputTokens=12, OutputTokens=8),
        )
    )
    assert current.build().request.Input == with_preamble(*output)


def test_builder_builds_request_from_added_values() -> None:
    model = Model(ID="gpt-test")
    weather = Tool(Type=ToolType.FUNCTION, Name="weather", Description="Get weather")
    thought = reasoning("Need current weather.")
    call = ToolCall(CallID="call-1", Name="weather", Arguments='{"city":"London"}')

    current = new_builder()
    current.set_model(model)
    current.add_tool(weather)
    current.add_model_response(
        Response(
            Output=[
                Item(Type=ItemType.PROVIDER, Data=thought),
                Item(Type=ItemType.TOOL_CALL, Data=call),
            ]
        )
    )
    current.add_tool_result("call-1", text("completed:operation-1"), False)
    result = current.build()

    assert result == Result(
        request=Request(
            Model=model,
            Tools=[weather],
            Input=with_preamble(
                Item(Type=ItemType.PROVIDER, Data=thought),
                Item(Type=ItemType.TOOL_CALL, Data=call),
                tool_result("call-1", "completed:operation-1"),
            ),
        ),
        report=Report(),
    )
    assert result.report.changes == []


def test_builder_preserves_tool_result_payload() -> None:
    output = [
        ToolResultOutput(Kind=ToolResultKind.TEXT, Value="界" * 4001 + "\udcff"),
        ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,aGVsbG8="),
        ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Dimensions: 2000x1500"),
    ]
    current = new_builder()
    current.add_tool_result("call-1", output, False)
    result = current.build()
    data = result.request.Input[1].Data
    assert isinstance(data, ToolResult)
    assert data.Output == output
    assert result.report.changes == []


@pytest.mark.parametrize(
    ("running", "output"),
    [
        pytest.param(False, "done A", id="completed"),
        pytest.param(True, TOOL_CALL_RUNNING_PAYLOAD, id="still running"),
    ],
)
def test_builder_replaces_only_staged_running_result_in_place(running: bool, output: str) -> None:
    current = new_builder()
    current.add_tool_result("A", text(""), True)
    current.commit()
    current.add_tool_result("A", text(""), True)
    current.add_tool_result("B", text(""), True)
    current.add_tool_result("C", text("done C"), False)
    current.add_external_input(
        Input(ID="input", Kind=InputKind.EXTERNAL, Payload=Raw(b'"continue"'))
    )
    before = current.build()
    original = list(before.request.Input)
    current.add_tool_result("A", text("done A"), running)
    result = current.build()
    assert result.request.Input == with_preamble(
        tool_result("A", TOOL_CALL_RUNNING_PAYLOAD, running=True),
        tool_result("A", output, running=running),
        tool_result("B", TOOL_CALL_RUNNING_PAYLOAD, running=True),
        tool_result("C", "done C"),
        user("continue"),
    )
    assert before.request.Input == original


def test_builder_preserves_completed_result_matching_running_payload() -> None:
    current = new_builder()
    current.add_tool_result("A", text(TOOL_CALL_RUNNING_PAYLOAD), False)
    current.add_tool_result("A", [], True)
    current.add_tool_result("A", text("done A"), False)
    assert current.build().request.Input == with_preamble(
        tool_result("A", TOOL_CALL_RUNNING_PAYLOAD),
        tool_result("A", "done A"),
    )


def test_builder_leads_system_prompt_with_preamble() -> None:
    current = new_builder()
    current.set_system_prompt("Be concise.")
    current.add_external_input(
        Input(ID="input-1", Kind=InputKind.EXTERNAL, Payload=Raw(encode("hello")))
    )
    assert current.build().request.Input == [
        Item(
            Type=ItemType.MESSAGE,
            Data=Message(Role=Role.SYSTEM, Text=PREAMBLE + "\n\nBe concise."),
        ),
        user("hello"),
    ]


def test_builder_appends_skills_to_preamble() -> None:
    skills = [
        Skill(
            name="go-review",
            description='Review <Go> & "tests"',
            path="/skills/reviewer's/SKILL.md",
        ),
        Skill(name="documents", description="Edit documents", path="/skills/documents/SKILL.md"),
    ]
    current = new_builder(*skills)
    skills[0] = Skill(name="changed")
    current.set_system_prompt("Be concise.")
    want = (
        PREAMBLE + "\n\nThe following skills provide specialized instructions for specific tasks.\n"
        "Use SkillUse to load a skill's file when the task matches its description.\n"
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool calls.\n"
        "\n"
        "<available_skills><skill><name>go-review</name><description>Review &lt;Go&gt; &amp; "
        "&#34;tests&#34;</description><location>/skills/reviewer&#39;s/SKILL.md</location>"
        "</skill><skill><name>documents</name><description>Edit documents</description>"
        "<location>/skills/documents/SKILL.md</location></skill></available_skills>\n"
        "\n"
        "Be concise."
    )
    message = current.build().request.Input[0].Data
    assert isinstance(message, Message)
    assert message.Text == want
