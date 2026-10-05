"""Persisted operation JSON is byte-identical to the Go implementation.

testdata/wire_golden.tsv holds `json.Marshal` output from the Go package for
the values built below.
"""

from pathlib import Path

import pytest
from msgspec import Struct
from unreal_agent.gojson import Raw, Timestamp, decode, encode
from unreal_agent.operation import (
    Operation,
    RemoteJobPlan,
    RemoteJobState,
    ShellInput,
    ShellPhase,
    ShellResult,
    ShellState,
    SkillUseState,
    Spec,
    Status,
    ValueState,
    ViewImageConfig,
    ViewImageResult,
    ViewImageState,
    new_remote_job_spec,
    new_shell_spec,
    new_skill_use_spec,
    new_value_spec,
    new_view_image_spec,
)

_GOLDEN = dict(
    line.split("\t", 1)
    for line in (Path(__file__).parent / "testdata" / "wire_golden.tsv")
    .read_text(encoding="utf-8")
    .split("\n")
    if line
)

_VALUES: dict[str, Struct] = {
    "spec_zero": Spec(),
    "op_zero": Operation(),
    "shell_spec": new_shell_spec(
        ShellInput(Command='echo "hi" <&>', Shell="/bin/sh", Directory="/tmp"), "/base", 100
    ),
    "value_spec": new_value_spec(Raw(b'{"a": [1, 2]}')),
    "skill_spec": new_skill_use_spec("/skills/SKILL.md"),
    "image_spec": new_view_image_spec(
        "/img.png", ViewImageConfig(MaxSize=10, MaxHeight=20, MaxWidth=30)
    ),
    "remote_spec": new_remote_job_spec(
        RemoteJobPlan(Type="test", Version=2, Data=Raw(b'{"value":"preserved"}'))
    ),
    "op_full": Operation(
        MaxOutputLength=7,
        ID="op-1",
        Type="shell",
        Version=3,
        Status=Status.AWAITING,
        State=Raw(b'{"x":1}'),
        Idempotency=Raw(b'"key"'),
    ),
    "shell_state_zero": ShellState(),
    "shell_state_full": ShellState(
        Input=ShellInput(Command="c", Shell="/bin/sh", Directory="/d"),
        BaseDirectory="/b",
        Phase=ShellPhase.READ_OUT_TAIL,
        ProcessGroupID=42,
        PendingExitCode=5,
        OutSize=10,
        ErrSize=11,
        InlineOut=b"out\xff",
        InlineErr=b"e",
        InlineOutTail=b"tail",
        InlineErrTail=b"",
        Result=ShellResult(Out="o\u2028", Err="e<>", OutSize=1, ErrSize=2, ExitCode=-1),
        TerminalError="boom",
        ErrorTruncated=True,
        OutTruncated=True,
        ErrTruncated=False,
        OutPath="/o",
        ErrPath="/e",
    ),
    "image_state": ViewImageState(
        Path="/p",
        Config=ViewImageConfig(MaxSize=1, MaxHeight=2, MaxWidth=3, MaxSourcePixels=4),
        Result=ViewImageResult(
            Content="abc",
            OriginalWidth=5,
            OriginalHeight=6,
            OriginalMIMEType="image/png",
            EncodedMIMEType="image/jpeg",
            ScaleRatio=1 / 3,
            Error="x",
        ),
    ),
    "image_state_one": ViewImageState(Path="/p", Result=ViewImageResult(ScaleRatio=1)),
    "remote_state_full": RemoteJobState(
        Plan=RemoteJobPlan(Type="t", Version=1, Data=Raw(b'{"k":"v"}')),
        Handle=Raw(b'{"h":1}'),
        NextInspectionAt=Timestamp.parse("2026-01-02T03:04:05.123456789Z"),
        OutstandingInput=Raw(b"[1]"),
        Subscription=Raw(b"null"),
        TerminalResult="r",
        TerminalError="e",
        ResultBytes=1,
        ResultTruncated=True,
        ErrorBytes=2,
        ErrorTruncated=True,
    ),
    "remote_state_zero": RemoteJobState(Plan=RemoteJobPlan(Data=Raw(b"{}"))),
    "skill_state": SkillUseState(Path="/s", Content=b"hello", TerminalError="t"),
    "value_state": ValueState(Value=Raw(b"true")),
}


@pytest.mark.parametrize("name", sorted(_VALUES))
def test_operation_json_matches_go(name: str) -> None:
    value = _VALUES[name]
    golden = _GOLDEN[name].encode()
    assert encode(value) == golden
    assert encode(decode(golden, type(value))) == golden


def test_golden_covers_every_value() -> None:
    assert set(_GOLDEN) == set(_VALUES)
