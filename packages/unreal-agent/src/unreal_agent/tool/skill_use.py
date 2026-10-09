from collections.abc import Callable, Sequence
from dataclasses import dataclass

from unreal_agent._go import go_bytes, go_quote, go_trim_space, to_valid_utf8
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    Operation,
    OperationError,
    Status,
    decode_skill_use,
    new_skill_use_spec,
)
from unreal_agent.tool._jsonv2 import GoJSONError, decode_string_field
from unreal_agent.tool.static import SKILL_USE_NAME
from unreal_agent.tool.tool import CallStatus, Context, Skill, TranslationError

__all__ = ["SkillUseResult", "SkillUseTranslator"]


@dataclass(frozen=True, slots=True)
class SkillUseResult:
    CallID: str = ""
    Content: bytes = b""
    Running: bool = False
    Error: str = ""

    def to_llm_result(self) -> ToolResult:
        text = to_valid_utf8(self.Content)
        if self.Running:
            text = "Skill is loading."
        elif self.Error:
            text = self.Error
        return ToolResult(
            CallID=self.CallID,
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=text)],
            Running=self.Running,
        )


class SkillUseTranslator:
    """Loads a registered skill's file. Owned by the registry that resolves skill names."""

    __slots__ = ("_resolve_skill",)

    def __init__(self, resolve_skill: Callable[[str], Skill | None]) -> None:
        self._resolve_skill = resolve_skill

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        if call.Name not in ("", SKILL_USE_NAME):
            return CallStatus(
                Error=f"skill-use call name {go_quote(call.Name)} does not match "
                f"static tool {go_quote(SKILL_USE_NAME)}"
            )
        encoded = call.Arguments
        if go_trim_space(encoded) == "":
            encoded = "{}"
        try:
            name = decode_string_field(go_bytes(encoded), "name", "tool.skillUseArguments")
        except GoJSONError as error:
            return CallStatus(Error=f"decode skill-use arguments: {error}")
        if go_trim_space(name) == "":
            return CallStatus(Error='skill-use argument "name" must be set')
        skill = self._resolve_skill(name)
        if skill is None:
            return CallStatus(Error=f"skill {go_quote(name)} is not registered")
        try:
            spec = new_skill_use_spec(skill.path)
        except OperationError as error:
            return CallStatus(Error=f"build skill-use operation: {error}")
        return CallStatus(WaitingFor=[ctx.submit(spec)])

    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> SkillUseResult:
        if status.Error:
            return SkillUseResult(CallID=call_id, Error=status.Error)
        if len(operations) != 1:
            raise TranslationError(
                f"skill-use call {go_quote(call_id)} has {len(operations)} operations, want 1"
            )
        current = operations[0]
        try:
            state = decode_skill_use(current)
        except OperationError as error:
            raise TranslationError(
                f"decode skill-use call {go_quote(call_id)} result: {error}"
            ) from error
        match current.Status:
            case Status.COMPLETED:
                return SkillUseResult(CallID=call_id, Content=state.Content)
            case Status.READY | Status.AWAITING | Status.CANCELING:
                return SkillUseResult(CallID=call_id, Running=True)
            case Status.CANCELED | Status.FAILED:
                if not state.TerminalError:
                    raise TranslationError(
                        f"skill-use call {go_quote(call_id)} terminal operation "
                        f"{go_quote(current.ID)} has no error"
                    )
                return SkillUseResult(
                    CallID=call_id, Content=state.Content, Error=state.TerminalError
                )
            case _:
                raise TranslationError(
                    f"skill-use call {go_quote(call_id)} operation {go_quote(current.ID)} "
                    f"has unsupported status {go_quote(current.Status)}"
                )
