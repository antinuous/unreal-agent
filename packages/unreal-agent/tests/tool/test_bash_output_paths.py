# ruff: noqa: ASYNC240
import asyncio
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import encode
from unreal_agent.llm import ToolCall
from unreal_agent.operation import (
    DEFAULT_MAX_OUTPUT_LENGTH,
    SHELL_ERR_FILENAME,
    SHELL_OUT_FILENAME,
    LocalOperationManager,
    Operation,
    ShellInput,
    Status,
    decode_shell_state,
    new_shell_spec,
)
from unreal_agent.primitives import (
    Channel,
    IOCreateRequest,
    PrimitiveDispatchType,
    PrimitiveEvent,
    PrimitiveEventType,
    PrimitiveFailureResult,
    SourceID,
    create,
)
from unreal_agent.tool import CallStatus, bash

from tests.operation._helpers import advance_shell_once
from tests.tool._helpers import RecordingContext


@pytest.mark.parametrize("terminal", [Status.FAILED, Status.CANCELED])
@pytest.mark.parametrize(
    "creates",
    [
        pytest.param(1, id="before captures"),
        pytest.param(2, id="stdout exists"),
        pytest.param(3, id="both captures exist"),
    ],
)
async def test_capture_paths_survive_failure_and_cancellation(
    tmp_path: Path, creates: int, terminal: Status
) -> None:
    base = str(tmp_path)
    spec = new_shell_spec(ShellInput(Shell="/bin/sh"), base, 100)
    current = Operation(
        ID="captures",
        Type=spec.Type,
        Version=spec.Version,
        State=spec.State,
        MaxOutputLength=spec.MaxOutputLength,
        Status=Status.READY,
    )
    step = advance_shell_once(current, None)
    for _ in range(creates):
        assert len(step.dispatches) == 1
        assert step.dispatches[0].type == PrimitiveDispatchType.IO_CREATE
        request = step.dispatches[0].data
        assert isinstance(request, IOCreateRequest)
        events: Channel[PrimitiveEvent] = Channel()
        create(request, events)
        async with asyncio.timeout(15):
            event = await events.get()
        assert event.type == PrimitiveEventType.IO_CREATE_COMPLETED, event
        assert step.operation is not None
        step = advance_shell_once(step.operation, event)
    assert step.operation is not None
    if terminal == Status.FAILED:
        step = advance_shell_once(
            step.operation,
            PrimitiveEvent(
                source=SourceID(current.ID),
                type=PrimitiveEventType.FAILED,
                result=PrimitiveFailureResult(error="capture creation failed"),
            ),
        )
    else:
        step = advance_shell_once(replace(step.operation, Status=Status.CANCELING), None)
    assert step.operation is not None
    state = decode_shell_state(step.operation)
    assert step.operation.Status == terminal
    assert state.Result is None

    result = bash.new(bash.Config()).translate_result("call", CallStatus(), [step.operation])
    assert result.Output[0].Value.endswith("Error: " + state.TerminalError)
    for filename, recorded, exists in (
        (SHELL_OUT_FILENAME, state.OutPath, creates >= 2),
        (SHELL_ERR_FILENAME, state.ErrPath, creates >= 3),
    ):
        path = os.path.join(base, current.ID, filename)
        assert os.path.exists(path) == exists
        assert recorded == (path if exists else "")
        assert result.Output[0].Value.count(path) == (1 if exists else 0)


@pytest.mark.parametrize(
    ("stdout", "stderr", "limit"),
    [
        pytest.param("界" * 40001, "e" * 40001, 0, id="default character limit"),
        pytest.param("\n" * 20001, "\\" * 20001, 0, id="newlines and backslashes within default"),
        pytest.param("\n" * 40001, "\\" * 40001, 0, id="newlines and backslashes exceed default"),
        pytest.param("界" * 25000, "", 0, id="source read within default"),
        pytest.param("x" * 40000, "", 0, id="default boundary"),
        pytest.param("ok", "", 0, id="complete output"),
        pytest.param("界界界界", "éééé", 3, id="requested limit"),
        pytest.param("界" * 5000, "", 5000, id="requested limit below default"),
    ],
)
async def test_truncated_output_can_be_read_from_capture_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    new_manager: Callable[[], LocalOperationManager],
    stdout: str,
    stderr: str,
    limit: int,
) -> None:
    base = tmp_path / "captures with spaces"
    base.mkdir(mode=0o700)
    monkeypatch.setenv("OUT", stdout)
    monkeypatch.setenv("ERR", stderr)
    translator = bash.new(bash.Config(shell="/bin/sh", base_directory=str(base)))
    submitted = RecordingContext()
    arguments: dict[str, object] = {"command": "printf '%s' \"$OUT\"; printf '%s' \"$ERR\" >&2"}
    if limit:
        arguments["max_output_length"] = limit
    status = translator.translate(
        submitted, ToolCall(CallID="call-1", Arguments=encode(arguments).decode())
    )
    assert status.Error == ""
    assert len(submitted.specs) == 1
    spec = submitted.specs[0]
    manager = new_manager()
    await manager.add(
        Operation(
            ID=status.WaitingFor[0],
            Type=spec.Type,
            Version=spec.Version,
            Status=Status.READY,
            MaxOutputLength=spec.MaxOutputLength,
            State=spec.State,
        )
    )
    completed: Operation | None = None
    async with asyncio.timeout(5):
        async for update in manager.updates():
            assert update.Status not in (Status.FAILED, Status.CANCELED), update
            if update.Status == Status.COMPLETED:
                completed = update
                break
    assert completed is not None

    # A resumed translator must use the operation's capture directory.
    translator = bash.new(bash.Config(base_directory=str(tmp_path / "other")))
    result = translator.translate_result("call-1", status, [completed])
    state = decode_shell_state(completed)
    assert state.Result is not None
    want = state.Result.Out
    if state.Result.Err:
        want += "\nStderr:\n" + state.Result.Err
    assert result.Output[0].Value == want
    for capture_path, text, preview, truncated in (
        (state.OutPath, stdout, state.Result.Out, state.OutTruncated),
        (state.ErrPath, stderr, state.Result.Err, state.ErrTruncated),
    ):
        assert os.path.isabs(capture_path)
        assert capture_path.startswith(str(base) + os.sep)
        assert Path(capture_path).read_text() == text
        want_truncated = len(text) > (limit or DEFAULT_MAX_OUTPUT_LENGTH)
        assert truncated == want_truncated
        if not want_truncated:
            assert capture_path not in result.Output[0].Value
            assert preview == text
            continue
        assert result.Output[0].Value.count(capture_path) == 1
