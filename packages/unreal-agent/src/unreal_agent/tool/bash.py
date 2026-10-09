from collections.abc import Sequence
from dataclasses import dataclass

from unreal_agent._go import go_bytes, go_quote
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    TYPE_SHELL,
    Operation,
    OperationError,
    ShellInput,
    ShellResult,
    Spec,
    Status,
    decode_shell_state,
    new_shell_spec,
)
from unreal_agent.tool._jsonv2 import GoJSONError, decode_optional_string, decode_raw_object
from unreal_agent.tool.output import error_status, parse_max_output_length
from unreal_agent.tool.tool import CallStatus, Context, TranslationError

__all__ = ["BashTranslator", "Config", "Result"]


@dataclass(frozen=True, slots=True)
class Config:
    shell: str = ""
    directory: str = ""
    base_directory: str = ""


class _ArgumentError(Exception):
    def __init__(self, message: str, limit: int) -> None:
        super().__init__(message)
        self.limit = limit


@dataclass(frozen=True, slots=True)
class Result:
    CallID: str = ""
    Running: bool = False
    Output: ShellResult | None = None
    OutPath: str = ""
    ErrPath: str = ""
    Error: str = ""

    def to_llm_result(self) -> ToolResult:
        parts: list[str] = []
        if self.Running:
            parts.append("Command is still running.")
        else:
            if self.Output is not None:
                if self.Output.Out:
                    parts.append(self.Output.Out)
                if self.Output.Err:
                    parts.append("Stderr:\n" + self.Output.Err)
                if self.Output.ExitCode != 0:
                    parts.append(f"Exit code: {self.Output.ExitCode}")
            else:
                if self.OutPath:
                    parts.append("Stdout capture: " + self.OutPath)
                if self.ErrPath:
                    parts.append("Stderr capture: " + self.ErrPath)
            if self.Error:
                parts.append("Error: " + self.Error)
            if not parts:
                parts.append("(no output)")
        return ToolResult(
            CallID=self.CallID,
            Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value="\n".join(parts))],
            Running=self.Running,
        )


class BashTranslator:
    __slots__ = ("_config",)

    def __init__(self, config: Config) -> None:
        self._config = config

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        try:
            command, limit = _validate_arguments(call.Arguments)
        except _ArgumentError as error:
            return error_status(str(error), error.limit)
        try:
            spec = self._build_operation(command, limit)
        except OperationError as error:
            return error_status(f"build Bash operation: {error}", limit)
        return CallStatus(WaitingFor=[ctx.submit(spec)])

    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> Result:
        if status.Error:
            if operations:
                raise TranslationError(
                    f"bash tool call {go_quote(call_id)} has both a validation error and operations"
                )
            return Result(CallID=call_id, Error=status.Error)
        if len(operations) != 1:
            raise TranslationError(
                f"bash tool call {go_quote(call_id)} has {len(operations)} operations, want 1"
            )
        return _translate_operation_result(call_id, operations[0])

    def _build_operation(self, command: str, limit: int) -> Spec:
        shell_input = ShellInput(
            Command=command, Shell=self._config.shell, Directory=self._config.directory
        )
        return new_shell_spec(shell_input, self._config.base_directory, limit)


def _translate_operation_result(call_id: str, current: Operation) -> Result:
    if current.Type != TYPE_SHELL:
        raise TranslationError(
            f"bash tool call {go_quote(call_id)} operation {go_quote(current.ID)} has type "
            f"{go_quote(current.Type)}, want {go_quote(TYPE_SHELL)}"
        )
    try:
        state = decode_shell_state(current)
    except OperationError as error:
        raise TranslationError(
            f"decode Bash operation {go_quote(current.ID)} state: {error}"
        ) from error
    terminal_error = state.TerminalError
    match current.Status:
        case Status.READY | Status.AWAITING | Status.CANCELING:
            return Result(CallID=call_id, Running=True)
        case Status.COMPLETED:
            if state.Result is None:
                raise TranslationError(
                    f"bash tool call {go_quote(call_id)} completed operation "
                    f"{go_quote(current.ID)} has no result"
                )
        case Status.FAILED | Status.CANCELED:
            if not terminal_error:
                terminal_error = "shell operation " + current.Status
        case _:
            raise TranslationError(
                f"bash tool call {go_quote(call_id)} operation {go_quote(current.ID)} "
                f"has invalid status {go_quote(current.Status)}"
            )
    return Result(
        CallID=call_id,
        Output=state.Result,
        OutPath=state.OutPath,
        ErrPath=state.ErrPath,
        Error=terminal_error,
    )


def _validate_arguments(encoded: str) -> tuple[str, int]:
    try:
        arguments = decode_raw_object(go_bytes(encoded)) or {}
    except GoJSONError as error:
        raise _ArgumentError(f"decode Bash arguments: {error}", 0) from error
    try:
        limit = parse_max_output_length(arguments.get("max_output_length", b""))
    except ValueError as error:
        raise _ArgumentError(f"bash argument: {error}", 0) from error
    encoded_command = arguments.get("command")
    if encoded_command is None:
        raise _ArgumentError('bash argument "command" must be set', limit)
    try:
        command = decode_optional_string(encoded_command)
    except GoJSONError as error:
        raise _ArgumentError(f'decode Bash argument "command": {error}', limit) from error
    if command is None:
        raise _ArgumentError('bash argument "command" must be a string', limit)
    offset = go_bytes(command).find(b"\x00")
    if offset >= 0:
        raise _ArgumentError(
            f'bash argument "command" contains a NUL byte at offset {offset}', limit
        )
    return command, limit
