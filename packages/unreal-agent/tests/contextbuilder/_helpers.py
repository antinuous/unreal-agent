from typing import cast

from unreal_agent.contextbuilder import PREAMBLE, Builder, Result, new_builder
from unreal_agent.contextbuilder.builder import _Builder
from unreal_agent.gojson import Raw, encode
from unreal_agent.inbox import Input, InputKind
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    Response,
    Role,
    StopReason,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
    Usage,
)


def text(value: str) -> list[ToolResultOutput]:
    return [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]


def tool_result(call_id: str, value: str, running: bool = False) -> Item:
    return Item(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(CallID=call_id, Output=text(value), Running=running),
    )


def reasoning(value: str) -> ProviderItem:
    return ProviderItem(
        Type="reasoning",
        Raw=Raw(b'{"type":"reasoning"}'),
        Display=ProviderDisplay(Kind=ProviderDisplayKind.REASONING, Text=value),
    )


def with_preamble(*items: Item) -> list[Item]:
    return [Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text=PREAMBLE)), *items]


def user(value: str) -> Item:
    return Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=value))


def new_internal_builder() -> _Builder:
    return cast(_Builder, new_builder())


def add_compaction_test_turn(current: _Builder, turn: int, total: int, *output: Item) -> None:
    label = chr(ord("A") + turn)
    current.add_external_input(
        Input(ID=label, Kind=InputKind.EXTERNAL, Payload=Raw(encode(label + " input")))
    )
    current.commit()
    response = Response(
        Output=[
            Item(Type=ItemType.PROVIDER, Data=reasoning(label + " reasoning")),
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text=label + " output")),
            *output,
        ]
    )
    if total != 0:
        response.Usage = Usage(
            InputTokens=total - 1_000,
            OutputTokens=1_000,
            CachedInputTokens=2_000,
            CacheWriteInputTokens=500,
            ReasoningTokens=400,
        )
    current.add_model_response(response)


def assert_compaction_turn(current: _Builder, turn: int) -> None:
    start = current._compaction_cutoff()
    if turn < 0:
        assert start is None, f"selection = {start}, want no compaction"
        return
    assert start == turn - 1, f"selection = {start}, want turn {turn} at {turn - 1}"


def new_compaction_history() -> _Builder:
    current = new_internal_builder()
    for turn, total in enumerate([10_000, 18_000, 35_000, 43_000, 55_000]):
        add_compaction_test_turn(current, turn, total)
    return current


def usable_compaction_response(value: str) -> Response:
    return Response(
        Stop=StopReason.COMPLETE,
        Output=[Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.ASSISTANT, Text=value))],
    )


def build_compaction_context(current: Builder) -> Result:
    return current.build()
