import os
import re
from pathlib import Path

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import Raw, encode
from unreal_agent.operation import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    TYPE_SHELL,
    VERSION_SHELL,
    Operation,
    OperationError,
    Shell,
    ShellInput,
    ShellPhase,
    ShellState,
    Status,
    bound_output,
    decode_shell_state,
    new_shell_spec,
)
from unreal_agent.operation.output import _bound_output
from unreal_agent.primitives import (
    CorrelationID,
    IOReadCompletedResult,
    IOReadOutputResult,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
)

from tests.operation._helpers import advance_shell_once

_MARKER = re.compile(r"\.\.\.([0-9]+) bytes truncated\.\.\.")


def _rune_count(text: str) -> int:
    return len(text)


@pytest.mark.parametrize(
    ("text", "want", "limit", "truncated"),
    [
        pytest.param("", "", 1, False, id="empty"),
        pytest.param("界é🙂", "界é🙂", 3, False, id="exact"),
        pytest.param("abcdefghij", "abc...4 bytes truncated...hij", 6, True, id="even"),
        pytest.param("abcdefghij", "ab...5 bytes truncated...hij", 5, True, id="odd"),
        pytest.param("ab", "...1 bytes truncated...b", 1, True, id="one"),
        pytest.param("abc", "a...1 bytes truncated...c", 2, True, id="two"),
        pytest.param("界éab🙂好", "界...4 bytes truncated...🙂好", 3, True, id="unicode"),
        pytest.param("…abc…", "…...3 bytes truncated...…", 2, True, id="ellipsis"),
        pytest.param(b"a\xff\xfeb", "a��b", 4, False, id="invalid UTF8"),
        pytest.param(
            b"a\xffbc\xfed",
            "a�...2 bytes truncated...�d",
            4,
            True,
            id="invalid UTF8 truncated",
        ),
        pytest.param(
            "é\n🙂", "é...1 bytes truncated...🙂", 2, True, id="unicode and newline truncated"
        ),
        pytest.param(
            "é\n中🙂\t界", "é\n...7 bytes truncated...\t界", 4, True, id="unicode and whitespace"
        ),
        pytest.param("a\nbc\td", "a\n...2 bytes truncated...\td", 4, True, id="whitespace"),
        pytest.param(
            "\x00ab\x1f", "\x00...2 bytes truncated...\x1f", 2, True, id="control characters"
        ),
        pytest.param("abc", "...3 bytes truncated...", 0, True, id="zero"),
        pytest.param("abc", "...3 bytes truncated...", -1, True, id="negative"),
        pytest.param(
            "HEAD" + "x" * MAX_OUTPUT_LENGTH + "TAIL",
            "HEAD" + "x" * 499996 + "...8 bytes truncated..." + "x" * 499996 + "TAIL",
            MAX_OUTPUT_LENGTH,
            True,
            id="maximum",
        ),
    ],
)
def test_bound_output_preserves_head_and_tail(
    text: str | bytes, want: str, limit: int, truncated: bool
) -> None:
    got, got_truncated = bound_output(text, limit)
    assert (got, got_truncated) == (want, truncated)
    got.encode("utf-8")
    retained = _MARKER.sub("", got)
    assert _rune_count(retained) <= max(0, limit)


@pytest.mark.parametrize(
    "text",
    ['"\\/\b\f\n\r\t\x00\x1f', '\\u0000\\\\\\"abcd\\n', "é界🙂…<>&\u2028\u2029"],
)
def test_bound_output_character_boundaries(text: str) -> None:
    encoded_length = len(text.encode())
    for limit in range(_rune_count(text) + 2):
        got, truncated = bound_output(text, limit)
        if not truncated:
            assert got == text, limit
            assert _rune_count(text) <= limit, limit
            continue
        match = _MARKER.search(got)
        assert match is not None, got
        head, tail = got[: match.start()], got[match.end() :]
        assert text.startswith(head), (limit, got)
        assert text.endswith(tail), (limit, got)
        assert match.group(1) == str(encoded_length - len(head.encode()) - len(tail.encode()))
        for index, part in enumerate((head, tail)):
            budget = limit // 2 if index == 0 else limit - limit // 2
            assert _rune_count(part) == budget, (limit, part)


def test_bound_output_rejects_missing_tail() -> None:
    with pytest.raises(
        ValueError, match=r"^bound output: empty tail requires the complete output in head$"
    ):
        _bound_output(b"ab", b"", 10, 4, "/capture")


@pytest.mark.parametrize(
    ("head", "tail", "path", "want", "size"),
    [
        pytest.param(
            b"abcdef", b"", "", "ab...2 bytes truncated...ef", 6, id="complete without path"
        ),
        pytest.param(
            b"abcdef",
            b"",
            "/capture",
            "ab...2 bytes truncated; complete output in /capture...ef",
            6,
            id="complete with path",
        ),
        pytest.param(
            b"ab", b"yz", "", "ab...6 bytes truncated...yz", 10, id="partial without path"
        ),
        pytest.param(
            b"ab",
            b"yz",
            "/capture",
            "ab...6 bytes truncated; complete output in /capture...yz",
            10,
            id="partial with path",
        ),
        pytest.param(
            b"a\xff",
            b"\xfez",
            "",
            "a�...96 bytes truncated...�z",
            100,
            id="partial with invalid UTF8",
        ),
        pytest.param(
            "界é".encode(),
            "🙂好".encode(),
            "",
            "界é...88 bytes truncated...🙂好",
            100,
            id="partial with unicode",
        ),
    ],
)
def test_bound_output_with_optional_path(
    head: bytes, tail: bytes, path: str, want: str, size: int
) -> None:
    assert _bound_output(head, tail, size, 4, path) == (want, True)


@pytest.mark.parametrize("limit", [0, MAX_OUTPUT_LENGTH + 1])
def test_shell_rejects_invalid_output_limits(
    limit: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    with pytest.raises(OperationError):
        new_shell_spec(ShellInput(Shell="/bin/sh"), str(tmp_path_factory.mktemp("base")), limit)
    shell = new_shell_spec(ShellInput(Shell="/bin/sh"), str(tmp_path_factory.mktemp("base")), 1)
    with pytest.raises(OperationError):
        decode_shell_state(
            Operation(
                Type=shell.Type, Version=shell.Version, State=shell.State, MaxOutputLength=limit
            )
        )


def test_shell_accepts_maximum_output_length(tmp_path: Path) -> None:
    shell = new_shell_spec(ShellInput(Shell="/bin/sh"), str(tmp_path), MAX_OUTPUT_LENGTH)
    decode_shell_state(
        Operation(
            Type=shell.Type,
            Version=shell.Version,
            State=shell.State,
            MaxOutputLength=shell.MaxOutputLength,
        )
    )


@pytest.mark.parametrize(
    ("out", "size", "limit", "want", "truncated"),
    [
        pytest.param(
            ("界" * 40001).encode(),
            120003,
            DEFAULT_MAX_OUTPUT_LENGTH,
            "界" * 20000 + "...3 bytes truncated; complete output in {path}..." + "界" * 20000,
            True,
            id="default",
        ),
        pytest.param(
            "界éab".encode(),
            7,
            2,
            "界...3 bytes truncated; complete output in {path}...b",
            True,
            id="custom",
        ),
        pytest.param(("é" * 50000).encode(), 100000, 50000, "é" * 50000, False, id="above default"),
        pytest.param(b"a\xffb", 3, 3, "a�b", False, id="invalid UTF8"),
    ],
)
def test_shell_prepares_output_before_publishing_completion(
    tmp_path: Path, out: bytes, size: int, limit: int, want: str, truncated: bool
) -> None:
    base = str(tmp_path)
    encoded = encode(
        ShellState(
            Input=ShellInput(Shell="/bin/sh"),
            BaseDirectory=base,
            Phase=ShellPhase.READ_ERR,
            InlineOut=out,
            OutSize=size,
            PendingExitCode=0,
        )
    )
    current = Operation(
        ID="shell",
        Type=TYPE_SHELL,
        Version=VERSION_SHELL,
        Status=Status.AWAITING,
        State=Raw(encoded),
        MaxOutputLength=limit,
    )
    shell = Shell(current)
    shell.handle(
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_OUTPUT,
            SourceID("shell"),
            CorrelationID("read_err"),
            IOReadOutputResult(0, out),
        )
    )
    step = shell.handle(
        PrimitiveEvent(
            PrimitiveEventType.IO_READ_COMPLETED,
            SourceID("shell"),
            CorrelationID("read_err"),
            IOReadCompletedResult(size),
        )
    )
    assert step.operation is not None
    state = decode_shell_state(step.operation)
    out_path = os.path.join(base, "shell", "out")
    err_path = os.path.join(base, "shell", "err")
    assert step.operation.Status == Status.COMPLETED
    assert state.Result is not None
    assert state.Result.Out == want.replace("{path}", out_path)
    assert state.Result.Err == want.replace("{path}", err_path)
    assert (state.OutTruncated, state.ErrTruncated) == (truncated, truncated)
    assert (state.OutPath, state.ErrPath) == (out_path, err_path)


def test_shell_bounds_operation_errors(tmp_path: Path) -> None:
    spec = new_shell_spec(ShellInput(Shell="/bin/sh"), str(tmp_path), 100)
    current = Operation(
        ID="shell",
        Type=spec.Type,
        Version=spec.Version,
        Status=Status.AWAITING,
        State=spec.State,
        MaxOutputLength=3,
    )
    step = advance_shell_once(
        current,
        PrimitiveEvent(
            PrimitiveEventType.FAILED, SourceID("shell"), result=PrimitiveFailureResult("éééé")
        ),
    )
    assert step.operation is not None
    state = decode_shell_state(step.operation)
    assert state.TerminalError == "é...2 bytes truncated...éé"
    assert state.ErrorTruncated
    assert state.OutTruncated
    assert state.ErrTruncated


@pytest.mark.parametrize("cancel", [False, True])
def test_shell_replaces_previously_truncated_errors(tmp_path: Path, cancel: bool) -> None:
    state = ShellState(
        Input=ShellInput(Shell="/bin/sh"),
        BaseDirectory=str(tmp_path),
        TerminalError="o...10 bytes truncated...ld",
        ErrorTruncated=True,
    )
    current = Operation(
        ID="shell",
        Type=TYPE_SHELL,
        Version=VERSION_SHELL,
        Status=Status.AWAITING,
        State=Raw(encode(state)),
        MaxOutputLength=3,
    )
    event: PrimitiveEvent | None = PrimitiveEvent(
        PrimitiveEventType.FAILED, SourceID("shell"), result=PrimitiveFailureResult("abcdef")
    )
    want = "a...3 bytes truncated...ef"
    if cancel:
        current, event = replace(current, Status=Status.CANCELING), None
        want = "s...21 bytes truncated...ed"
    step = advance_shell_once(current, event)
    assert step.operation is not None
    for _ in range(2):
        state = decode_shell_state(step.operation)
        assert (state.TerminalError, state.ErrorTruncated) == (want, True)
