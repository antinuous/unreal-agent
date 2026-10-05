import pytest
from unreal_agent.gojson import Raw, decode, encode
from unreal_agent.llm import ToolCall
from unreal_agent.operation import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    TYPE_SHELL,
    VERSION_SHELL,
    Operation,
    ShellInput,
    ShellResult,
    ShellState,
    Spec,
    Status,
)
from unreal_agent.tool import CallStatus, TranslationError, bash


class RecordingContext:
    def __init__(self) -> None:
        self.specs: list[Spec] = []

    def submit(self, spec: Spec, /) -> str:
        self.specs.append(spec)
        return "operation-1"


def test_translator_submits_shell_operation() -> None:
    config = bash.Config(shell="/bin/bash", directory="/workspace", base_directory="/operations")
    translator = bash.new(config)
    ctx = RecordingContext()
    status = translator.translate(
        ctx,
        ToolCall(
            CallID="call-1",
            Name="Bash",
            Arguments='{"command":"  printf \'%s\' \\"$HOME\\"; exit 7  "}',
        ),
    )
    assert status.Error == ""
    assert status.WaitingFor == ["operation-1"]
    assert len(ctx.specs) == 1
    spec = ctx.specs[0]
    assert (spec.Type, spec.Version) == (TYPE_SHELL, VERSION_SHELL)
    state = decode(bytes(spec.State), ShellState)
    assert state.Input == ShellInput(
        Command="  printf '%s' \"$HOME\"; exit 7  ", Shell=config.shell, Directory=config.directory
    )
    assert state.BaseDirectory == config.base_directory
    assert spec.MaxOutputLength == DEFAULT_MAX_OUTPUT_LENGTH


@pytest.mark.parametrize(
    ("status", "state", "want"),
    [
        pytest.param(
            Status.COMPLETED,
            ShellState(
                Input=ShellInput(Command="secret command"),
                BaseDirectory="/secret/path",
                Result=ShellResult(Out="ok\n", OutSize=3),
            ),
            "ok\n",
            id="completed",
        ),
        pytest.param(
            Status.COMPLETED, ShellState(Result=ShellResult()), "(no output)", id="empty output"
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(Result=ShellResult(Out='\n  "quoted" \\ output\t\n\n')),
            '\n  "quoted" \\ output\t\n\n',
            id="stdout preserves whitespace and quotes",
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(Result=ShellResult(Err="warning\n")),
            "Stderr:\nwarning\n",
            id="successful stderr",
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(Result=ShellResult(Out="out\n", Err="err\n")),
            "out\n\nStderr:\nerr\n",
            id="both streams",
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(Result=ShellResult(ExitCode=1)),
            "Exit code: 1",
            id="nonzero exit without output",
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(Result=ShellResult(Err="command failed\n", ErrSize=15, ExitCode=7)),
            "Stderr:\ncommand failed\n\nExit code: 7",
            id="nonzero exit with stderr",
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(
                OutTruncated=True,
                OutPath="/captures/out",
                Result=ShellResult(
                    Out="head...92 bytes truncated; complete output in /captures/out...tail",
                    OutSize=100,
                ),
            ),
            "head...92 bytes truncated; complete output in /captures/out...tail",
            id="truncated stdout",
        ),
        pytest.param(
            Status.COMPLETED,
            ShellState(
                ErrTruncated=True,
                ErrPath="/captures/err",
                Result=ShellResult(
                    Out="output",
                    OutSize=6,
                    Err="head...92 bytes truncated; complete output in /captures/err...tail",
                    ErrSize=100,
                    ExitCode=7,
                ),
            ),
            "output\nStderr:\nhead...92 bytes truncated; complete output in /captures/err...tail"
            "\nExit code: 7",
            id="truncated stderr",
        ),
        pytest.param(
            Status.READY,
            ShellState(OutTruncated=True, ErrTruncated=True),
            "Command is still running.",
            id="ready",
        ),
        pytest.param(
            Status.AWAITING,
            ShellState(OutTruncated=True, ErrTruncated=True),
            "Command is still running.",
            id="awaiting",
        ),
        pytest.param(
            Status.CANCELING,
            ShellState(OutTruncated=True, ErrTruncated=True),
            "Command is still running.",
            id="canceling",
        ),
        pytest.param(
            Status.CANCELED,
            ShellState(OutTruncated=True, ErrTruncated=True, TerminalError="canceled by user"),
            "Error: canceled by user",
            id="canceled",
        ),
        pytest.param(
            Status.FAILED,
            ShellState(OutTruncated=True, ErrTruncated=True, TerminalError="process failed"),
            "Error: process failed",
            id="failed",
        ),
        pytest.param(
            Status.FAILED,
            ShellState(),
            "Error: shell operation failed",
            id="failed without explanation",
        ),
        pytest.param(
            Status.CANCELED,
            ShellState(),
            "Error: shell operation canceled",
            id="canceled without explanation",
        ),
        pytest.param(
            Status.FAILED,
            ShellState(TerminalError="err...100 bytes truncated...end", ErrorTruncated=True),
            "Error: err...100 bytes truncated...end",
            id="truncated error",
        ),
        pytest.param(
            Status.FAILED,
            ShellState(
                TerminalError="read failed", OutPath="/captures/out", ErrPath="/captures/err"
            ),
            "Stdout capture: /captures/out\nStderr capture: /captures/err\nError: read failed",
            id="failed with captures",
        ),
    ],
)
def test_translator_translates_shell_operation_results(
    status: Status, state: ShellState, want: str
) -> None:
    result = bash.new(bash.Config()).translate_result(
        "call-1",
        CallStatus(),
        [
            Operation(
                ID="operation-1",
                Type=TYPE_SHELL,
                Version=VERSION_SHELL,
                MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
                Status=status,
                State=Raw(encode(state)),
            )
        ],
    )
    assert result.CallID == "call-1"
    assert result.Output[0].Value == want
    for internal in ("secret command", "/secret/path", "operation-1"):
        assert internal not in result.Output[0].Value


def test_translator_translates_validation_error_without_operations() -> None:
    translator = bash.new(bash.Config())
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(CallID="call-1", Arguments='{"command":42}'))
    assert status.Error
    assert ctx.specs == []
    result = translator.translate_result("call-1", status, [])
    assert result.CallID == "call-1"
    assert result.Output[0].Value == "Error: " + status.Error


@pytest.mark.parametrize(
    ("status", "count", "want"),
    [
        pytest.param(CallStatus(), 0, "has 0 operations, want 1", id="missing operation"),
        pytest.param(CallStatus(), 2, "has 2 operations, want 1", id="multiple operations"),
        pytest.param(
            CallStatus(Error="invalid"),
            1,
            "both a validation error and operations",
            id="validation error with operation",
        ),
    ],
)
def test_translator_rejects_invalid_operation_counts(
    status: CallStatus, count: int, want: str
) -> None:
    with pytest.raises(TranslationError, match=want):
        bash.new(bash.Config()).translate_result(
            "call-1", status, [Operation() for _ in range(count)]
        )


_VALID_STATE = Raw(encode(ShellState()))


@pytest.mark.parametrize(
    ("operation", "want"),
    [
        pytest.param(
            Operation(
                ID="operation-1",
                Type="other",
                Version=VERSION_SHELL,
                MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
            ),
            'has type "other", want "shell"',
            id="wrong type",
        ),
        pytest.param(
            Operation(
                ID="operation-1",
                Type=TYPE_SHELL,
                Version=VERSION_SHELL,
                MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
                State=Raw(b"{"),
            ),
            "decode Bash operation",
            id="malformed state",
        ),
        pytest.param(
            Operation(
                ID="operation-1",
                Type=TYPE_SHELL,
                Version=VERSION_SHELL,
                MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
                Status="unknown",  # type: ignore[arg-type]
                State=_VALID_STATE,
            ),
            "invalid status",
            id="invalid status",
        ),
        pytest.param(
            Operation(
                ID="operation-1",
                Type=TYPE_SHELL,
                Version=VERSION_SHELL,
                MaxOutputLength=DEFAULT_MAX_OUTPUT_LENGTH,
                Status=Status.COMPLETED,
                State=_VALID_STATE,
            ),
            "completed operation",
            id="completed without result",
        ),
    ],
)
def test_translator_rejects_invalid_shell_operation_results(
    operation: Operation, want: str
) -> None:
    with pytest.raises(TranslationError, match=want):
        bash.new(bash.Config()).translate_result("call-1", CallStatus(), [operation])


def test_translator_accepts_empty_command() -> None:
    translator = bash.new(bash.Config(shell="/bin/bash", base_directory="/operations"))
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Arguments='{"command":""}'))
    assert status.Error == ""
    assert len(status.WaitingFor) == 1
    assert len(ctx.specs) == 1


@pytest.mark.parametrize(
    ("arguments", "want"),
    [
        pytest.param("{", "decode Bash arguments: jsontext: unexpected EOF", id="malformed JSON"),
        pytest.param(
            '{"command":"pwd","command":"ls"}',
            'duplicate object member name "command"',
            id="duplicate command",
        ),
        pytest.param("{}", 'bash argument "command" must be set', id="missing command"),
        pytest.param(
            '{"COMMAND":"pwd"}', 'bash argument "command" must be set', id="uppercase command"
        ),
        pytest.param(
            '{"command":null}', 'bash argument "command" must be a string', id="null command"
        ),
        pytest.param(
            '{"command":42}', 'decode Bash argument "command": json:', id="non-string command"
        ),
        pytest.param(
            '{"command":"\\u0000pwd"}',
            'bash argument "command" contains a NUL byte at offset 0',
            id="NUL at start",
        ),
        pytest.param(
            '{"command":"pwd\\u0000"}',
            'bash argument "command" contains a NUL byte at offset 3',
            id="NUL at end",
        ),
        pytest.param(
            '{"command":"a\\u0000b\\u0000"}',
            'bash argument "command" contains a NUL byte at offset 1',
            id="first of multiple NULs",
        ),
        pytest.param(
            '{"command":"é雪\\u0000"}',
            'bash argument "command" contains a NUL byte at offset 5',
            id="NUL after multibyte text",
        ),
        pytest.param(
            '{"command":"cat <<\'EOF\'\\ncaf\\u0000e9\\nEOF\\n"}',
            'bash argument "command" contains a NUL byte at offset 15',
            id="NUL in heredoc",
        ),
    ],
)
def test_translator_rejects_invalid_arguments(arguments: str, want: str) -> None:
    translator = bash.new(bash.Config(shell="/bin/bash", base_directory="/operations"))
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Arguments=arguments))
    assert want in status.Error
    assert not status.WaitingFor
    assert ctx.specs == []


@pytest.mark.parametrize(
    "command", ["cat <<'EOF'\ncaf\\x00e9\nEOF\n", "printf 'a\\000b'", "printf '%s' '\\u0000'"]
)
def test_translator_preserves_escaped_nul_commands(command: str) -> None:
    translator = bash.new(bash.Config(shell="/bin/bash", base_directory="/operations"))
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Arguments=encode({"command": command}).decode()))
    assert status.Error == ""
    assert len(status.WaitingFor) == 1
    assert len(ctx.specs) == 1
    assert decode(bytes(ctx.specs[0].State), ShellState).Input.Command == command


def test_translator_returns_shell_configuration_error() -> None:
    translator = bash.new(bash.Config(shell="bash", base_directory="/operations"))
    ctx = RecordingContext()
    status = translator.translate(ctx, ToolCall(Arguments='{"command":"pwd"}'))
    assert status.Error == "build Bash operation: shell path must be absolute"
    assert ctx.specs == []
