"""The Bash tool: runs a shell command as a durable shell operation."""

from collections.abc import Sequence
from dataclasses import dataclass

from unreal_agent._go import go_bytes, go_quote
from unreal_agent.llm import ToolCall, ToolResult, ToolResultKind, ToolResultOutput
from unreal_agent.operation import (
    TYPE_SHELL,
    Operation,
    OperationError,
    ShellInput,
    Spec,
    Status,
    decode_shell_state,
    new_shell_spec,
)
from unreal_agent.tool._jsonv2 import GoJSONError, decode_optional_string, decode_raw_object
from unreal_agent.tool.output import error_status, parse_max_output_length
from unreal_agent.tool.tool import CallStatus, Context, TranslationError

__all__ = ["BashTranslator", "Config"]


@dataclass(frozen=True, slots=True)
class Config:
    shell: str = ""
    directory: str = ""
    base_directory: str = ""


class _ArgumentError(Exception):
    def __init__(self, message: str, limit: int) -> None:
        super().__init__(message)
        self.limit = limit


def _text(call_id: str, value: str) -> ToolResult:
    return ToolResult(
        CallID=call_id, Output=[ToolResultOutput(Kind=ToolResultKind.TEXT, Value=value)]
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
    ) -> ToolResult:
        if status.Error:
            if operations:
                raise TranslationError(
                    f"bash tool call {go_quote(call_id)} has both a validation error and operations"
                )
            return _text(call_id, "Error: " + status.Error)
        if len(operations) != 1:
            raise TranslationError(
                f"bash tool call {go_quote(call_id)} has {len(operations)} operations, want 1"
            )
        return _text(call_id, _translate_operation_result(call_id, operations[0]))

    def _build_operation(self, command: str, limit: int) -> Spec:
        shell_input = ShellInput(
            Command=command, Shell=self._config.shell, Directory=self._config.directory
        )
        return new_shell_spec(shell_input, self._config.base_directory, limit)


def _translate_operation_result(call_id: str, current: Operation) -> str:
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
            return "Command is still running."
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

    parts: list[str] = []
    if state.Result is not None:
        if state.Result.Out:
            parts.append(state.Result.Out)
        if state.Result.Err:
            parts.append("Stderr:\n" + state.Result.Err)
        if state.Result.ExitCode != 0:
            parts.append(f"Exit code: {state.Result.ExitCode}")
    else:
        if state.OutPath:
            parts.append("Stdout capture: " + state.OutPath)
        if state.ErrPath:
            parts.append("Stderr capture: " + state.ErrPath)
    if terminal_error:
        parts.append("Error: " + terminal_error)
    if not parts:
        return "(no output)"
    return "\n".join(parts)


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
