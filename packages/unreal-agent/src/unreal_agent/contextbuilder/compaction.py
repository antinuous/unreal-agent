from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Final, cast

from unreal_agent.contextbuilder.skills import read_prompt
from unreal_agent.llm import Item, ItemType, ToolCall, ToolResult

__all__ = ["COMPACTION_PROMPT", "COMPACTION_RETAINED_TOKENS", "PendingToolCalls"]

# This fixed budget is part of the replay contract.
COMPACTION_RETAINED_TOKENS: Final = 20_000

COMPACTION_PROMPT: Final = read_prompt("compaction.md")


@dataclass(slots=True)
class PendingToolCall:
    item: Item
    result: ToolResult | None = None


@dataclass(slots=True)
class PendingToolCalls:
    """Tool calls without a terminal result, in call order, with their latest running result."""

    calls: dict[str, PendingToolCall] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    @classmethod
    def collect(cls, prefix: Iterable[Item]) -> PendingToolCalls:
        pending = cls()
        for item in prefix:
            match item.Type:
                case ItemType.TOOL_CALL:
                    call = cast(ToolCall, item.Data)
                    pending.calls[call.CallID] = PendingToolCall(item)
                    pending.order.append(call.CallID)
                case ItemType.TOOL_RESULT:
                    pending._add_result(cast(ToolResult, item.Data))
                case _:
                    pass
        return pending

    def _add_result(self, result: ToolResult) -> None:
        call = self.calls.get(result.CallID)
        if call is None:
            return
        if not result.Running:
            del self.calls[result.CallID]
            return
        call.result = result

    def __iter__(self) -> Iterator[tuple[str, PendingToolCall]]:
        for call_id in self.order:
            call = self.calls.get(call_id)
            if call is not None:
                yield call_id, call
