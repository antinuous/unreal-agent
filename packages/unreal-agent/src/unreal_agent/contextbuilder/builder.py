from dataclasses import dataclass
from typing import Final

from msgspec.structs import replace

from unreal_agent._go import go_int64, go_quote, go_trim_space
from unreal_agent.contextbuilder.compaction import (
    COMPACTION_PROMPT,
    COMPACTION_RETAINED_TOKENS,
    PendingToolCalls,
)
from unreal_agent.contextbuilder.contextbuilder import Builder, Result
from unreal_agent.contextbuilder.skills import format_skills_for_prompt, read_prompt
from unreal_agent.inbox import ControlMessage, ControlMode, Input, InputKind, Settings
from unreal_agent.llm import (
    Item,
    ItemType,
    Message,
    Model,
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

COMPACTION_INSTRUCTION: Final = (
    "This is system message. You are performing context compaction. "
    "Return only the text of a handoff summary for another LLM assistant "
    "to resume the original task."
)


def _running_output() -> list[ToolResultOutput]:
    return [ToolResultOutput(Kind=ToolResultKind.TEXT, Value=TOOL_CALL_RUNNING_PAYLOAD)]


@dataclass(slots=True, frozen=True)
class _PrefixToken:
    committed_prefix_index: int
    """Exclusive end of the response in the committed prefix."""
    tokens: int


class _Builder:
    """The committed prefix is resent unchanged; the staged suffix holds unsubmitted inputs."""

    __slots__ = (
        "_committed_prefix",
        "_model",
        "_preamble",
        "_prefix_tokens",
        "_staged_suffix",
        "_system_prompt",
        "_tools",
    )

    def __init__(self, preamble: str) -> None:
        self._preamble = preamble
        self._model = Model()
        self._tools: list[Tool] = []
        self._system_prompt = ""
        self._committed_prefix: list[Item] = [Item(Data=None)]
        self._staged_suffix: list[Item] = []
        self._prefix_tokens: list[_PrefixToken] = []
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
        self._stage_items(Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=text)))

    def set_model(self, model: Model, /) -> None:
        self._model = replace(model)

    def add_control_message(self, message: ControlMessage, /) -> None:
        match message.Mode:
            case ControlMode.UPDATE_SETTINGS:
                settings = message.Parameters
                if not isinstance(settings, Settings):
                    raise TypeError("settings control message has no settings parameters")
                model = replace(self._model)
                if settings.Model and settings.Model != model.ID:
                    model.ID = settings.Model
                    self._prefix_tokens = []
                if settings.CompactionThreshold is not None:
                    model.CompactionThreshold = settings.CompactionThreshold
                if settings.MaxOutputTokens is not None:
                    model.MaxOutputTokens = settings.MaxOutputTokens
                if settings.ReasoningEffort:
                    model.ReasoningEffort = settings.ReasoningEffort
                self._model = model
            case ControlMode.HEARTBEAT:
                self._staged_suffix.append(
                    Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=message.Reason))
                )
            case _:
                pass

    def set_system_prompt(self, prompt: str, /) -> None:
        self._system_prompt = prompt
        text = go_trim_space(self._preamble + "\n\n" + prompt)
        self._committed_prefix[0] = Item(
            Type=ItemType.MESSAGE, Data=Message(Role=Role.SYSTEM, Text=text)
        )

    def add_model_response(self, response: Response, /) -> None:
        self._committed_prefix.extend(response.Output or ())
        self._prefix_tokens.append(
            _PrefixToken(
                len(self._committed_prefix),
                go_int64(response.Usage.InputTokens + response.Usage.OutputTokens),
            )
        )

    def add_tool(self, tool: Tool, /) -> None:
        self._tools.append(tool)

    def add_tool_result(
        self, call_id: str, payload: list[ToolResultOutput], running: bool, /
    ) -> None:
        running_index = next(
            (
                index
                for index, item in enumerate(self._staged_suffix)
                if item.Type == ItemType.TOOL_RESULT
                and isinstance(item.Data, ToolResult)
                and item.Data.CallID == call_id
                and item.Data.Running
            ),
            -1,
        )
        if running:
            payload = _running_output()
        result_item = Item(
            Type=ItemType.TOOL_RESULT,
            Data=ToolResult(CallID=call_id, Output=list(payload), Running=running),
        )
        if running_index != -1:
            self._staged_suffix[running_index] = result_item
            return
        self._staged_suffix.append(result_item)

    def _stage_items(self, *items: Item) -> None:
        self._staged_suffix.extend(items)

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

    def needs_compaction(self) -> bool:
        if not self._prefix_tokens:
            return False
        return self._prefix_tokens[-1].tokens >= self._model.CompactionThreshold

    def build_compaction(self) -> Result:
        cutoff = self._compaction_cutoff()
        if cutoff is None:
            raise ValueError("no history eligible for compaction")
        end = self._prefix_tokens[cutoff].committed_prefix_index
        history = self._committed_prefix[1:end]
        summarizer = _Builder(COMPACTION_PROMPT)
        summarizer.set_model(self._model)
        summarizer._stage_items(*history)
        for call_id, call in PendingToolCalls.collect(history):
            if call.result is None:
                # Native history needs a result even when the call is pending at the cutoff.
                summarizer.add_tool_result(call_id, [], True)
        # Some providers require the last message to be a user message if there are no
        # running tool calls. Add it without checking tool calls for consistency.
        summarizer._stage_items(
            Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=COMPACTION_INSTRUCTION))
        )
        return summarizer.build()

    def compact(self, response: Response, /) -> Builder | None:
        cutoff = self._compaction_cutoff()
        if cutoff is None:
            return None
        start = self._prefix_tokens[cutoff].committed_prefix_index
        rebuilt = _Builder(self._preamble)
        rebuilt.set_model(self._model)
        rebuilt.set_system_prompt(self._system_prompt)
        for tool in self._tools:
            rebuilt.add_tool(tool)
        rebuilt._add_compaction_summary(response)
        rebuilt._carry_forward_context(self, start)
        return rebuilt

    def _add_compaction_summary(self, response: Response) -> None:
        summary = ""
        for item in reversed(response.Output or ()):
            message = item.Data
            if (
                item.Type == ItemType.MESSAGE
                and isinstance(message, Message)
                and message.Role == Role.ASSISTANT
                and go_trim_space(message.Text) != ""
            ):
                summary = message.Text
                break
        self._stage_items(Item(Type=ItemType.MESSAGE, Data=Message(Role=Role.USER, Text=summary)))

    def _compaction_cutoff(self) -> int | None:
        """Return the last checkpoint to summarize, retaining whole turns within the budget."""
        if not self._prefix_tokens:
            return None
        latest = self._prefix_tokens[-1].tokens
        if latest <= COMPACTION_RETAINED_TOKENS:
            return None
        start = 0
        for index in range(len(self._prefix_tokens) - 1, -1, -1):
            if go_int64(latest - self._prefix_tokens[index].tokens) > COMPACTION_RETAINED_TOKENS:
                break
            start = index
        return start

    def _carry_forward_context(self, source: _Builder, start: int) -> None:
        """Keep calls pending at the cutoff, with their latest running results, then the rest."""
        for call_id, call in PendingToolCalls.collect(source._committed_prefix[1:start]):
            self._stage_items(call.item)
            if call.result is not None:
                self.add_tool_result(call_id, call.result.Output, call.result.Running)
        self.commit()
        self._committed_prefix.extend(source._committed_prefix[start:])
        self._staged_suffix.extend(source._staged_suffix)


def new_builder(*skills: Skill) -> Builder:
    """Start with the harness preamble, followed by the skills block when skills are given."""
    preamble = PREAMBLE
    skill_prompt = format_skills_for_prompt(skills)
    if skill_prompt:
        preamble += "\n\n" + skill_prompt
    return _Builder(preamble)
