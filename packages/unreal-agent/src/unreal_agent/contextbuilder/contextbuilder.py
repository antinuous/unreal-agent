from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from unreal_agent.inbox import ControlMessage, Input
from unreal_agent.llm import Model, Request, Response, Tool, ToolResultOutput

__all__ = ["Builder", "Change", "ChangeKind", "Report", "Result"]


class ChangeKind(StrEnum):
    NONE = ""
    OMITTED = "omitted"
    TRUNCATED = "truncated"
    COMPACTED = "compacted"


@dataclass(slots=True)
class Change:
    kind: ChangeKind = ChangeKind.NONE
    source: str = ""
    reason: str = ""


@dataclass(slots=True)
class Report:
    changes: list[Change] = field(default_factory=list)


@dataclass(slots=True)
class Result:
    request: Request
    report: Report = field(default_factory=Report)


class Builder(Protocol):
    """Retains model request state without performing I/O."""

    def add_external_input(self, input: Input, /) -> None:
        """Raise ValueError when the input is not an external JSON string."""
        ...

    def add_control_message(self, message: ControlMessage, /) -> None: ...

    def set_model(self, model: Model, /) -> None: ...

    def set_system_prompt(self, prompt: str, /) -> None: ...

    def add_model_response(self, response: Response, /) -> None: ...

    def add_tool(self, tool: Tool, /) -> None: ...

    def add_tool_result(
        self, call_id: str, payload: list[ToolResultOutput], running: bool, /
    ) -> None: ...

    def commit(self) -> None: ...

    def build(self) -> Result: ...
