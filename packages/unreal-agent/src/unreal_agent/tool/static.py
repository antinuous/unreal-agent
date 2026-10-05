from collections.abc import Sequence
from typing import Any, Final

from unreal_agent._go import go_quote
from unreal_agent.llm import Tool, ToolCall, ToolResult, ToolResultKind, ToolResultOutput, ToolType
from unreal_agent.operation import DEFAULT_MAX_OUTPUT_LENGTH, MAX_OUTPUT_LENGTH, Operation
from unreal_agent.tool.tool import CallStatus, Context, Definition

__all__ = [
    "BASH_NAME",
    "SKILL_USE_NAME",
    "VIEW_IMAGE_NAME",
]

BASH_NAME: Final = "Bash"
VIEW_IMAGE_NAME: Final = "ViewImage"
SKILL_USE_NAME: Final = "SkillUse"


class UnavailableTranslator:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus(Error=self._error_message())

    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> ToolResult:
        return ToolResult(
            CallID=call_id,
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=self._error_message())],
        )

    def _error_message(self) -> str:
        return f"static tool {go_quote(self._name)} is not configured"


def static_definitions() -> list[Definition]:
    return [
        Definition(
            tool=Tool(
                Type=ToolType.FUNCTION,
                Name=BASH_NAME,
                Description=(
                    "Execute a shell command in background. Independent commands may be issued "
                    "as parallel tool calls in one turn. Command child processes are killed "
                    "when the shell exits."
                ),
                Parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        },
                        "max_output_length": _max_output_length_schema(),
                    },
                    "required": ["command"],
                },
            )
        ),
        Definition(
            tool=Tool(
                Type=ToolType.FUNCTION,
                Name=VIEW_IMAGE_NAME,
                Description="View a local JPEG, PNG, BMP, TIFF, or WebP image.",
                Parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Image file path, absolute or relative to the workspace."
                            ),
                        },
                    },
                    "required": ["path"],
                },
            )
        ),
        Definition(
            tool=Tool(
                Type=ToolType.FUNCTION,
                Name=SKILL_USE_NAME,
                Description="Load the instructions for a registered skill.",
                Parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The exact name of the skill to load.",
                        },
                    },
                    "required": ["name"],
                },
            )
        ),
    ]


def _max_output_length_schema() -> dict[str, Any]:
    return {
        "type": "integer",
        "description": (
            "Maximum characters per output text field. Truncated text keeps its head and tail, "
            "around a marker stating how much was omitted, and path to the file with the "
            f"complete stream. Defaults to {DEFAULT_MAX_OUTPUT_LENGTH}."
        ),
        "minimum": 1,
        "maximum": MAX_OUTPUT_LENGTH,
        "default": DEFAULT_MAX_OUTPUT_LENGTH,
    }
