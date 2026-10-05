from typing import Final

from msgspec.structs import replace

from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.contextbuilder.contextbuilder import Builder, Result
from unreal_agent.contextbuilder.skills import format_skills_for_prompt, read_prompt
from unreal_agent.inbox import ControlMessage, ControlMode, Input, InputKind, Settings
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
    Reasoning,
    Request,
    Response,
    Role,
    Tool,
    ToolResult,
    ToolResultKind,
    ToolResultOutput,
)
from unreal_agent.tool import Skill
from unreal_agent.tool._jsonv2 import GoJSONError, decode_string

__all__ = ["PREAMBLE", "TOOL_CALL_RUNNING_PAYLOAD", "new_builder"]

TOOL_CALL_RUNNING_PAYLOAD: Final = (
    "Tool call is still running. Its result arrives in a later turn: continue with "
    "independent work, or end your turn to wait for it."
)
"""The result a running call shows until it completes."""

PREAMBLE: Final = read_prompt("preamble.md")


def _running_output() -> list[ToolResultOutput]:
    return [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=TOOL_CALL_RUNNING_PAYLOAD)]


class _Builder:
    """The committed prefix is resent unchanged; the staged suffix holds unsubmitted inputs."""

    __slots__ = ("_committed_prefix", "_model", "_preamble", "_staged_suffix", "_tools")

    def __init__(self, preamble: str) -> None:
        self._preamble = preamble
        self._model = Model()
        self._tools: list[Tool] = []
        self._committed_prefix: list[Item] = [Item(Data=None)]
        self._staged_suffix: list[Item] = []
        self.set_system_prompt("")

    def add_external_input(self, input: Input, /) -> None:
        if input.Kind != InputKind.EXTERNAL:
            raise ValueError(
                f"external input {go_quote(input.ID)} has input kind {go_quote(input.Kind)}"
            )
        try:
            text = decode_string(bytes(input.Payload))
        except GoJSONError as error:
            raise ValueError(f"decode external input {go_quote(input.ID)}: {error}") from error
        self._staged_suffix.append(
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=text))
        )

    def set_model(self, model: Model, /) -> None:
        self._model = replace(model)

    def add_control_message(self, message: ControlMessage, /) -> None:
        match message.Mode:
            case ControlMode.UPDATE_SETTINGS:
                settings = message.Parameters
                if not isinstance(settings, Settings):
                    raise TypeError("settings control message has no settings parameters")
                self._model = replace(self._model, ReasoningEffort=settings.ReasoningEffort)
            case ControlMode.HEARTBEAT:
                self._staged_suffix.append(
                    Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=message.Reason))
                )
            case _:
                pass

    def set_system_prompt(self, prompt: str, /) -> None:
        text = go_trim_space(self._preamble + "\n\n" + prompt)
        self._committed_prefix[0] = Item(
            Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text=text)
        )

    def add_model_response(self, response: Response, /) -> None:
        self._committed_prefix.extend(response.Output or ())

    def add_reasoning(self, reasoning: Reasoning, /) -> None:
        self._staged_suffix.append(Item(Type=ItemType.REASONING, Data=reasoning))

    def add_tool(self, tool: Tool, /) -> None:
        self._tools.append(tool)

    def add_tool_result(
        self, call_id: str, payload: list[ToolResultOutput], running: bool, /
    ) -> None:
        running_output = _running_output()
        if running:
            payload = running_output
        self._staged_suffix = [
            item
            for item in self._staged_suffix
            if not (
                item.Type == ItemType.TOOL_RESULT
                and isinstance(item.Data, ToolResult)
                and item.Data.CallID == call_id
                and item.Data.Output == running_output
            )
        ]
        self._staged_suffix.append(
            Item(Type=ItemType.TOOL_RESULT, Data=ToolResult(CallID=call_id, Output=list(payload)))
        )

    def commit(self) -> None:
        self._committed_prefix.extend(self._staged_suffix)
        self._staged_suffix = []

    def build(self) -> Result:
        request = Request(
            Model=replace(self._model),
            Input=[*self._committed_prefix, *self._staged_suffix],
            Tools=list(self._tools),
        )
        return Result(request=request)


def new_builder(*skills: Skill) -> Builder:
    """Start with the harness preamble, followed by the skills block when skills are given."""
    preamble = PREAMBLE
    skill_prompt = format_skills_for_prompt(skills)
    if skill_prompt:
        preamble += "\n\n" + skill_prompt
    return _Builder(preamble)
