import copy
from typing import cast

import pytest
from msgspec.structs import replace
from unreal_agent.contextbuilder import new_builder
from unreal_agent.contextbuilder.builder import _Builder
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ControlMessage, ControlMode
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    ReasoningEffort,
    Role,
    Tool,
    ToolCall,
    ToolType,
    Usage,
)
from unreal_agent.tool import Skill

from tests.contextbuilder._helpers import (
    add_compaction_test_turn,
    build_compaction_context,
    new_compaction_history,
    new_internal_builder,
    reasoning,
    text,
    usable_compaction_response,
    user,
    with_preamble,
)


@pytest.mark.parametrize("completion", ["during compaction", "after compaction"])
def test_compaction_rebuilds_context_and_preserves_staging(completion: str) -> None:
    current = cast(
        _Builder,
        new_builder(Skill(name="review", description="Review code", path="/skills/review")),
    )
    current.set_model(Model(ID="model", MaxOutputTokens=4096, ReasoningEffort=ReasoningEffort.HIGH))
    current.set_system_prompt("System instructions")
    current.add_tool(Tool(Name="test", Type=ToolType.FUNCTION))
    for turn, total in enumerate([10_000, 18_000, 35_000, 43_000, 55_000]):
        output: list[Item] = []
        if turn == 3:
            output = [Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="call", Name="test"))]
        if turn == 4:
            current.add_tool_result("call", text(""), True)
        add_compaction_test_turn(current, turn, total, *output)
    current.add_tool_result("call", text(""), True)
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="before"))
    before = build_compaction_context(current)
    original = copy.deepcopy(before.request.Input)
    assert current._compaction_cutoff() is not None, "expected eligible prefix"
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="during"))
    if completion == "during compaction":
        current.add_tool_result("call", text("done"), False)
    current.set_system_prompt("Updated instructions")
    response = usable_compaction_response("First summary")
    assert response.Output is not None
    response.Output = [
        Item(
            ProviderID="reasoning",
            Type=ItemType.PROVIDER,
            Data=ProviderItem(
                Type="reasoning",
                Raw=Raw(b'{"type":"reasoning"}'),
                Display=ProviderDisplay(
                    Kind=ProviderDisplayKind.REASONING, Text="summary reasoning"
                ),
            ),
        ),
        *response.Output,
    ]
    response.Output[1].ProviderID = "summary-message"
    response.Output.append(
        Item(
            ProviderID="summary-continuation",
            Type=ItemType.MESSAGE,
            Data=Message(Role=Role.ASSISTANT, Text="Next steps"),
        )
    )
    response.Usage = Usage(InputTokens=1_000_000, OutputTokens=50_000)
    rebuilt = current.compact(response)
    assert rebuilt is not None, "summary did not produce a fresh builder"
    assert rebuilt is not current, "summary did not produce a fresh builder"
    if completion == "after compaction":
        rebuilt.add_tool_result("call", text("done"), False)
        current.add_tool_result("call", text("done"), False)
    accepted = build_compaction_context(current)
    built = build_compaction_context(rebuilt)
    start = 0
    for index, item in enumerate(accepted.request.Input):
        if (
            item.Type == ItemType.MESSAGE
            and isinstance(item.Data, Message)
            and item.Data.Text == "D input"
        ):
            start = index
            break
    want = replace(
        accepted.request,
        Input=[
            accepted.request.Input[0],
            user("Next steps"),
            *accepted.request.Input[start:],
        ],
    )
    assert start != 0
    assert built.request == want, f"rebuilt request = {built.request!r}, want {want!r}"
    assert before.request.Input == original, "compaction changed an earlier request snapshot"
    assert isinstance(rebuilt, _Builder)
    assert rebuilt._compaction_cutoff() is None, "old-context or summary-request usage was reused"
    assert rebuilt.compact(response) is None, "duplicate response reused old-context usage"


def test_compaction_replaces_previous_summary_using_new_usage() -> None:
    current = new_compaction_history()
    assert current._compaction_cutoff() is not None, "expected eligible prefix"
    fresh = current.compact(usable_compaction_response("First summary"))
    assert fresh is not None, "first compaction failed"
    rebuilt = cast(_Builder, fresh)
    add_compaction_test_turn(rebuilt, 5, 20_000)
    assert rebuilt._compaction_cutoff() is None, "context within the retention budget was compacted"
    add_compaction_test_turn(rebuilt, 6, 35_000)
    assert rebuilt._compaction_cutoff() is not None, (
        "two comparable new checkpoints did not allow compaction"
    )
    second = rebuilt.compact(usable_compaction_response("Second summary"))
    assert second is not None, "second compaction failed"
    want = with_preamble(
        user("Second summary"),
        user("G input"),
        Item(Type=ItemType.PROVIDER, Data=reasoning("G reasoning")),
        Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text="G output")),
    )
    assert build_compaction_context(second).request.Input == want


@pytest.mark.parametrize(
    "phase",
    [
        pytest.param("final_answer", id="final_answer"),
        pytest.param("commentary", id="commentary"),
        pytest.param("", id=""),
    ],
)
def test_compaction_uses_last_assistant_message(phase: str) -> None:
    current = new_compaction_history()
    response = usable_compaction_response("I am summarizing the conversation.")
    assert response.Output is not None
    response.Output[0].Data = Message(
        Role=Role.ASSISTANT, Text="I am summarizing the conversation.", Phase="commentary"
    )
    response.Output.extend(
        [
            Item(
                Type=ItemType.MESSAGE,
                Data=Message(Role=Role.ASSISTANT, Text="Last summary", Phase=phase),
            ),
            Item(Type=ItemType.PROVIDER, Data=reasoning("reasoning")),
            user("not an assistant summary"),
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text=" \n")),
        ]
    )
    rebuilt = current.compact(response)
    assert rebuilt is not None, "compaction failed"
    got = build_compaction_context(rebuilt).request.Input[1].Data
    assert got == Message(Role=Role.USER, Text="Last summary"), f"summary = {got!r}"


def test_compaction_selection_stays_stable_across_staged_arrivals() -> None:
    current = new_internal_builder()
    for turn, total in enumerate([10_000, 18_000, 35_000, 43_000, 55_000]):
        output: list[Item] = []
        if turn == 1:
            output = [Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="pending", Name="test"))]
        add_compaction_test_turn(current, turn, total, *output)
    before = current._compaction_cutoff()
    assert before is not None, "expected eligible prefix"
    current.add_tool_result("pending", text(""), True)
    current.add_control_message(ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="during"))
    current.add_tool_result("pending", text("done"), False)
    after = current._compaction_cutoff()
    assert after == before, f"staged arrivals changed the cut from {before} to {after}"
