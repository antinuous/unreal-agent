import copy
from typing import cast

import pytest
from unreal_agent._go import go_trim_space
from unreal_agent.contextbuilder import TOOL_CALL_RUNNING_PAYLOAD
from unreal_agent.contextbuilder.builder import _Builder
from unreal_agent.contextbuilder.compaction import COMPACTION_PROMPT
from unreal_agent.gojson import Raw
from unreal_agent.inbox import ControlMessage, ControlMode, Settings
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
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)

from tests.contextbuilder._helpers import (
    add_compaction_test_turn,
    build_compaction_context,
    new_compaction_history,
    new_internal_builder,
    text,
    tool_result,
    usable_compaction_response,
    user,
)

HANDOFF = (
    "This is system message. You are performing context compaction. "
    "Return only the text of a handoff summary for another LLM assistant "
    "to resume the original task."
)


def _system(value: str) -> Item:
    return Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text=value))


def test_build_compaction_selects_same_prefix_as_reconstruction() -> None:
    current = new_internal_builder()
    for turn, total in enumerate([1_000, 101_000, 104_000, 109_000, 119_000]):
        add_compaction_test_turn(current, turn, total)
    current.add_control_message(
        ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="before compaction")
    )
    before = build_compaction_context(current)
    checkpoints = list(current._prefix_tokens)
    staged = list(current._staged_suffix)
    result = current.build_compaction()
    message = "building compaction mutated ordinary context or accounting"
    assert before == build_compaction_context(current), message
    assert checkpoints == current._prefix_tokens, message
    assert staged == current._staged_suffix, message
    history = result.request.Input
    assert len(history) == 8, (
        f"history has {len(history)} items, want system, whole turns A and B, "
        "and summary instruction"
    )
    for index, want in {
        1: "A input",
        3: "A output",
        4: "B input",
        6: "B output",
        7: HANDOFF,
    }.items():
        message = history[index].Data
        assert isinstance(message, Message)
        assert message.Text == want, f"history[{index}] = {message!r}, want {want!r}"
    current.add_tool_result("pending", text("completed during compaction"), False)
    current.add_control_message(
        ControlMessage(Mode=ControlMode.HEARTBEAT, Reason="during compaction")
    )
    # Even a commit without a response checkpoint must leave the selected prefix unchanged.
    current.commit()
    again = current.build_compaction()
    assert result == again, "arrivals changed summary request"
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction failed"
    accepted = build_compaction_context(current).request.Input
    want_input = [accepted[0], user("summary"), *accepted[7:]]
    assert build_compaction_context(rebuilt).request.Input == want_input, (
        "reconstruction did not replace exactly the summarized prefix"
    )


def test_build_compaction_preserves_native_history() -> None:
    current = new_internal_builder()
    current.set_system_prompt("Project constraint: preserve /tmp/project.go")
    call = Item(
        ProviderID="fc-123",
        Type=ItemType.TOOL_CALL,
        Data=ToolCall(
            CallID="call-123", Name="bash", Arguments='{"command":"cat /tmp/project.go"}'
        ),
    )
    reasoning = Item(
        ProviderID="rs-123",
        Type=ItemType.PROVIDER,
        Data=ProviderItem(
            Type="reasoning",
            Raw=Raw(b'{"type":"reasoning","encrypted":"opaque-state"}'),
            Display=ProviderDisplay(
                Kind=ProviderDisplayKind.REASONING, Text="Need to inspect the file"
            ),
        ),
    )
    add_compaction_test_turn(current, 0, 10_000, reasoning, call)
    current.add_tool_result("call-123", text(""), True)
    add_compaction_test_turn(current, 1, 20_000)
    current.add_tool_result(
        "call-123", text("cat: /tmp/project.go: No such file or directory"), False
    )
    add_compaction_test_turn(current, 2, 50_000)
    before = build_compaction_context(current)
    snapshot = copy.deepcopy(before)
    result = current.build_compaction()
    want = list(before.request.Input)
    want[0] = _system(go_trim_space(COMPACTION_PROMPT))
    want.append(user(HANDOFF))
    assert result.request.Input == want, "native history changed"
    assert snapshot == build_compaction_context(current), (
        "building compaction changed ordinary provider replay data"
    )
    result.request.Input[1] = Item(Data=None)
    assert snapshot == build_compaction_context(current), (
        "compaction request shares its input slice with the original builder"
    )


def test_build_compaction_completes_pending_calls_at_cutoff() -> None:
    current = new_internal_builder()
    calls = [
        Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID=call_id, Name="test"))
        for call_id in ("finished", "running", "z", "a")
    ]
    add_compaction_test_turn(current, 0, 10_000, *calls)
    current.add_tool_result("finished", text("done"), False)
    current.add_tool_result("running", text(""), True)
    add_compaction_test_turn(current, 1, 35_000)
    current.add_tool_result("z", text("finished in retained history"), False)
    add_compaction_test_turn(current, 2, 50_000)
    current.add_tool_result("a", text("finished after latest checkpoint"), False)
    before = build_compaction_context(current)
    snapshot = copy.deepcopy(before)
    result = current.build_compaction()
    cutoff = current._compaction_cutoff()
    assert cutoff is not None
    end = current._prefix_tokens[cutoff].committed_prefix_index
    want = list(before.request.Input[:end])
    want[0] = _system(go_trim_space(COMPACTION_PROMPT))
    for call_id in ("z", "a"):
        want.append(tool_result(call_id, TOOL_CALL_RUNNING_PAYLOAD, running=True))
    want.append(user(HANDOFF))
    assert result.request.Input == want, "history with pending results"
    assert snapshot == build_compaction_context(current), (
        "completing native history changed the original builder"
    )
    current.commit()
    again = current.build_compaction()
    assert result == again, "committing newer results changed summary request"


def test_build_compaction_includes_previous_summary() -> None:
    fresh = new_compaction_history().compact(
        usable_compaction_response("Previous summary: keep /src/main.go")
    )
    assert fresh is not None, "first compaction failed"
    current = cast(_Builder, fresh)
    add_compaction_test_turn(current, 5, 20_000)
    add_compaction_test_turn(current, 6, 35_000)
    history = current.build_compaction().request.Input
    summary = history[1].Data
    assert isinstance(summary, Message)
    assert summary.Text == "Previous summary: keep /src/main.go", (
        f"previous summary missing: {summary!r}"
    )
    last = history[-2].Data
    assert isinstance(last, Message)
    assert last.Text == "F output", f"summary must include history through F and retain G: {last!r}"


@pytest.mark.parametrize(
    "limit", [pytest.param(None, id="unset"), pytest.param(32_768, id="configured")]
)
def test_build_compaction_preserves_model(limit: int | None) -> None:
    current = new_compaction_history()
    model = Model(
        ID="current-model",
        CompactionThreshold=64_000,
        MaxOutputTokens=limit,
        ReasoningEffort=ReasoningEffort.HIGH,
    )
    current.set_model(model)
    current.add_control_message(
        ControlMessage(
            Mode=ControlMode.UPDATE_SETTINGS,
            Parameters=Settings(ReasoningEffort=ReasoningEffort.LOW),
        )
    )
    model.ReasoningEffort = ReasoningEffort.LOW
    current.add_tool(Tool(Name="test"))
    result = current.build_compaction()
    assert result.request.Model == model, "building summary changed model settings"
    assert build_compaction_context(current).request.Model == model, (
        "building summary changed model settings"
    )
    assert len(result.request.Tools) == 0, "summary request includes tools"
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction lost updated model settings"
    assert build_compaction_context(rebuilt).request.Model == model, (
        "compaction lost updated model settings"
    )


@pytest.mark.parametrize("location", ["summarized", "retained", "staged"])
def test_compaction_preserves_structured_tool_results(location: str) -> None:
    current = new_internal_builder()
    call = Item(Type=ItemType.TOOL_CALL, Data=ToolCall(CallID="image", Name="ViewImage"))
    add_compaction_test_turn(current, 0, 10_000, call)
    output = [
        ToolResultOutput(Kind=ToolResultKind.TEXT, Value="Image preview"),
        ToolResultOutput(Kind=ToolResultKind.IMAGE, Value="data:image/png;base64,aGVsbG8="),
    ]
    if location == "summarized":
        current.add_tool_result("image", output, False)
    add_compaction_test_turn(current, 1, 35_000)
    if location == "retained":
        current.add_tool_result("image", output, False)
    add_compaction_test_turn(current, 2, 50_000)
    if location == "staged":
        current.add_tool_result("image", output, False)
    result = current.build_compaction()
    rebuilt = current.compact(usable_compaction_response("summary"))
    assert rebuilt is not None, "compaction failed"
    if location != "summarized":
        result = build_compaction_context(rebuilt)
    want = Item(Type=ItemType.TOOL_RESULT, Data=ToolResult(CallID="image", Output=output))
    assert want in result.request.Input, f"{location} structured tool result was lost"


@pytest.mark.parametrize("total", [0, 20_000])
def test_build_compaction_without_eligible_history(total: int) -> None:
    current = new_internal_builder()
    if total != 0:
        add_compaction_test_turn(current, 0, total)
    with pytest.raises(ValueError, match="no history eligible for compaction"):
        current.build_compaction()
