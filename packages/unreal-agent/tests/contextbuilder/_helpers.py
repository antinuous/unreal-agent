from unreal_agent.gojson import Raw
from unreal_agent.llm import (
    Item,
    ItemType,
    ProviderDisplay,
    ProviderDisplayKind,
    ProviderItem,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
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
