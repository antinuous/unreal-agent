from unreal_agent.llm import Item, ItemType, ToolResult, ToolResultKind, ToolResultOutput


def text(value: str) -> list[ToolResultOutput]:
    return [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]


def tool_result(call_id: str, value: str, running: bool = False) -> Item:
    return Item(
        Type=ItemType.TOOL_RESULT,
        Data=ToolResult(CallID=call_id, Output=text(value), Running=running),
    )
