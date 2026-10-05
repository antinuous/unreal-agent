import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from msgspec.structs import replace
from unreal_agent.gojson import encode
from unreal_agent.operation import (
    ClosedError,
    LocalOperationManager,
    Manager,
    Operation,
    OperationError,
    ShellInput,
    ShellPhase,
    ShellState,
    Status,
    UnsupportedError,
)

from tests.operation._helpers import (
    TERMINAL_STATUSES,
    assert_process_group_gone,
    create_shell_artifacts,
    new_shell_operation,
    receive_operation,
    receive_terminal_operation,
    shell_operation_with_state,
    shell_state,
)

TEST_SHELL_PATH: Final = "/bin/sh"


async def wait_for_file_contents(path: str, want: str) -> None:
    async with asyncio.timeout(15):
        while True:
            if _read_text(path) == want:
                return
            await asyncio.sleep(0.01)


def _read_text(path: str) -> str | None:
    try:
        with open(path) as file:
            return file.read()
    except FileNotFoundError:
        return None


type NewManager = Callable[..., LocalOperationManager]


def _conforms_to_manager(manager: LocalOperationManager) -> Manager:
    return manager


def _process_running(current: Operation) -> bool:
    state = shell_state(current)
    return state.Phase == ShellPhase.PROCESS and state.ProcessGroupID > 1


async def test_local_operation_manager_runs_shell_operation(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    new_manager: NewManager,
) -> None:
    manager = new_manager()
    monkeypatch.setenv("VALUE", "output")
    current = new_shell_operation(
        "local-shell",
        ShellInput(
            Shell=TEST_SHELL_PATH,
            Command="printf '%s' \"$VALUE\"; printf error >&2; exit 7",
            Directory=str(tmp_path_factory.mktemp("work")),
        ),
        str(tmp_path_factory.mktemp("base")),
        64,
    )
    await manager.add(current)
    completed = await receive_terminal_operation(manager.updates(), current.ID)
    state = shell_state(completed)
    assert completed.Status == Status.COMPLETED
    assert state.Result is not None
    assert state.TerminalError == ""
    result = state.Result
    assert (result.Out, result.Err, result.OutSize, result.ErrSize, result.ExitCode) == (
        "output",
        "error",
        6,
        5,
        7,
    )


async def test_local_operation_manager_runs_multiple_shell_operations(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    want = {"local-first": "first", "local-second": "second"}
    for operation_id, output in want.items():
        await manager.add(
            new_shell_operation(
                operation_id,
                ShellInput(Shell=TEST_SHELL_PATH, Command="printf " + output),
                str(tmp_path),
                64,
            )
        )
    completed: set[str] = set()
    async with asyncio.timeout(15):
        async for current in manager.updates():
            if current.Status != Status.COMPLETED:
                continue
            result = shell_state(current).Result
            assert result is not None
            assert result.Out == want[current.ID]
            completed.add(current.ID)
            if len(completed) == len(want):
                break


async def test_local_operation_manager_queues_updates_while_unread(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    new_manager: NewManager,
) -> None:
    manager = new_manager()
    marker = str(tmp_path_factory.mktemp("marker") / "ran")
    monkeypatch.setenv("MARKER", marker)
    current = new_shell_operation(
        "local-queued-updates",
        ShellInput(Shell=TEST_SHELL_PATH, Command='printf output; printf done > "$MARKER"'),
        str(tmp_path_factory.mktemp("base")),
        64,
    )
    await manager.add(current)
    await wait_for_file_contents(marker, "done")

    phases = [
        ShellPhase.CREATE_DIRECTORY,
        ShellPhase.CREATE_OUT,
        ShellPhase.CREATE_ERR,
        ShellPhase.PROCESS,
        ShellPhase.READ_OUT,
        ShellPhase.READ_ERR,
    ]
    ranks = {phase: rank for rank, phase in enumerate(phases)}
    last_rank = -1
    seen: set[tuple[Status, bytes, bytes]] = set()
    async with asyncio.timeout(15):
        async for update in manager.updates():
            if update.ID != current.ID:
                continue
            key = (update.Status, bytes(update.State), bytes(update.Idempotency))
            assert key not in seen, f"duplicate update {update!r}"
            seen.add(key)
            state = shell_state(update)
            if update.Status == Status.COMPLETED:
                assert last_rank == len(phases) - 1
                assert state.Result is not None
                assert (state.Result.Out, state.Result.ExitCode) == ("output", 0)
                manager.close()
                async for remaining in manager.updates():
                    assert remaining.ID != current.ID, f"update after terminal state {remaining!r}"
                return
            assert update.Status == Status.AWAITING
            rank = ranks[state.Phase]
            assert last_rank <= rank <= last_rank + 1, (state.Phase, last_rank)
            last_rank = max(last_rank, rank)
    pytest.fail("updates closed before operation completed")


async def test_local_operation_manager_resumes_shell_operation(
    tmp_path: Path, new_manager: NewManager
) -> None:
    base_directory = str(tmp_path)
    operation_id = "local-resume"
    create_shell_artifacts(os.path.join(base_directory, operation_id), b"output", b"error")
    current = shell_operation_with_state(
        operation_id,
        ShellState(
            Input=ShellInput(Shell=TEST_SHELL_PATH),
            BaseDirectory=base_directory,
            Phase=ShellPhase.READ_OUT,
            PendingExitCode=5,
            InlineOut=b"ou",
        ),
    )
    manager = new_manager()
    await manager.add(current)
    completed = await receive_terminal_operation(manager.updates(), operation_id)
    result = shell_state(completed).Result
    assert completed.Status == Status.COMPLETED
    assert result is not None
    assert (result.Out, result.Err, result.ExitCode) == ("output", "error", 5)


async def test_local_operation_manager_cancels_active_shell_operation(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    current = new_shell_operation(
        "local-cancel",
        ShellInput(Shell=TEST_SHELL_PATH, Command="exec sleep 30"),
        str(tmp_path),
        64,
    )
    await manager.add(current)
    running = await receive_operation(manager.updates(), current.ID, _process_running)
    process_group_id = shell_state(running).ProcessGroupID
    await manager.cancel(current.ID, "test cancellation")

    canceled = await receive_terminal_operation(manager.updates(), current.ID)
    state = shell_state(canceled)
    assert canceled.Status == Status.CANCELED
    assert state.Phase == ""
    assert state.ProcessGroupID == process_group_id
    assert state.Result is None
    assert_process_group_gone(process_group_id)


async def test_local_operation_manager_cancel_is_idempotent(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    current = new_shell_operation(
        "local-cancel-idempotent",
        ShellInput(Shell=TEST_SHELL_PATH, Command="exec sleep 30"),
        str(tmp_path),
        64,
    )
    await manager.add(current)
    await manager.cancel(current.ID, "first cancellation")
    await manager.cancel(current.ID, "second cancellation")
    canceled = await receive_terminal_operation(manager.updates(), current.ID)
    assert canceled.Status == Status.CANCELED
    await manager.cancel(current.ID, "cancellation after completion")


async def test_local_operation_manager_add_is_idempotent(
    tmp_path_factory: pytest.TempPathFactory, new_manager: NewManager
) -> None:
    manager = new_manager()
    current = new_shell_operation(
        "local-add-idempotent",
        ShellInput(Shell=TEST_SHELL_PATH, Command="exec sleep 30"),
        str(tmp_path_factory.mktemp("base")),
        64,
    )
    await manager.add(current)
    await manager.add(current)
    await manager.cancel(current.ID, "finish test operation")
    completed = await receive_terminal_operation(manager.updates(), current.ID)
    assert completed.Status == Status.CANCELED

    replacement = new_shell_operation(
        current.ID,
        ShellInput(Shell=TEST_SHELL_PATH, Command="printf duplicate"),
        str(tmp_path_factory.mktemp("base")),
        64,
    )
    await manager.add(replacement)
    barrier = new_shell_operation(
        "local-add-idempotent-barrier",
        ShellInput(Shell=TEST_SHELL_PATH, Command="true"),
        str(tmp_path_factory.mktemp("base")),
        64,
    )
    await manager.add(barrier)
    async with asyncio.timeout(15):
        async for update in manager.updates():
            assert update.ID != current.ID, f"duplicate operation update {update!r}"
            if update.ID == barrier.ID and update.Status == Status.COMPLETED:
                return
    pytest.fail("updates closed before the barrier completed")


async def test_local_operation_manager_shutdown_cancels_active_shell_operation(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    current = new_shell_operation(
        "local-shutdown-active",
        ShellInput(Shell=TEST_SHELL_PATH, Command="exec sleep 30"),
        str(tmp_path),
        64,
    )
    await manager.add(current)
    running = await receive_operation(manager.updates(), current.ID, _process_running)
    process_group_id = shell_state(running).ProcessGroupID
    manager.close()
    async with asyncio.timeout(15):
        async for _ in manager.updates():
            pass
    assert_process_group_gone(process_group_id)


async def test_local_operation_manager_isolates_primitive_failure(
    tmp_path_factory: pytest.TempPathFactory, new_manager: NewManager
) -> None:
    manager = new_manager()
    failed = new_shell_operation(
        "local-primitive-failure",
        ShellInput(Shell=TEST_SHELL_PATH),
        str(tmp_path_factory.mktemp("base") / "missing"),
        64,
    )
    succeeded = new_shell_operation(
        "local-after-primitive-failure",
        ShellInput(Shell=TEST_SHELL_PATH, Command="printf success"),
        str(tmp_path_factory.mktemp("base")),
        64,
    )
    await manager.add(failed)
    await manager.add(succeeded)
    terminal: dict[str, Operation] = {}
    async with asyncio.timeout(15):
        async for current in manager.updates():
            if current.Status in TERMINAL_STATUSES:
                terminal[current.ID] = current
            if len(terminal) == 2:
                break
    assert len(terminal) == 2, "updates closed before operations completed"

    failed_state = shell_state(terminal[failed.ID])
    assert terminal[failed.ID].Status == Status.FAILED
    assert failed_state.TerminalError
    assert failed_state.Result is None
    succeeded_state = shell_state(terminal[succeeded.ID])
    assert terminal[succeeded.ID].Status == Status.COMPLETED
    assert succeeded_state.Result is not None
    assert succeeded_state.Result.Out == "success"


async def test_local_operation_manager_rejects_invalid_operations(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    valid = new_shell_operation("local-valid", ShellInput(Shell=TEST_SHELL_PATH), str(tmp_path), 1)
    tests = [
        (
            "unsupported type",
            Operation(ID="local-unsupported", Type="unknown", Status=Status.READY),
            'does not support type "unknown"',
        ),
        (
            "unsupported version",
            replace(valid, ID="local-version", Version=valid.Version + 1),
            "unsupported version",
        ),
        (
            "terminal",
            replace(valid, ID="local-terminal", Status=Status.COMPLETED),
            "terminal status",
        ),
    ]
    for name, current, want in tests:
        with pytest.raises(OperationError, match=want) as error:
            await manager.add(current)
        if name != "terminal":
            assert isinstance(error.value, UnsupportedError)

    await manager.add(valid)
    await manager.add(valid)
    await manager.cancel("unknown", "test")


async def test_local_operation_manager_closes_updates_on_close(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    current = new_shell_operation(
        "local-shutdown", ShellInput(Shell=TEST_SHELL_PATH, Command="true"), str(tmp_path), 64
    )
    await manager.add(current)
    manager.close()
    async with asyncio.timeout(5):
        async for _ in manager.updates():
            pass


async def test_local_operation_manager_raises_closed_error_after_shutdown(
    tmp_path: Path, new_manager: NewManager
) -> None:
    manager = new_manager()
    manager.close()
    async for _ in manager.updates():
        pass
    current = new_shell_operation(
        "local-after-shutdown", ShellInput(Shell=TEST_SHELL_PATH), str(tmp_path), 64
    )
    with pytest.raises(ClosedError):
        await manager.add(current)
    with pytest.raises(ClosedError):
        await manager.cancel(current.ID, "after shutdown")


@pytest.mark.timeout(60)
async def test_shell_capture_update_volume_scales_linearly(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    previous_bytes = 0
    previous_updates = 0
    for limit in (250_000, 1_000_000):
        base = str(tmp_path_factory.mktemp("base"))
        operation_id = "large-capture"
        contents = b"x" * (4 * limit + 1)
        create_shell_artifacts(os.path.join(base, operation_id), contents, contents)
        current = shell_operation_with_state(
            operation_id,
            ShellState(
                Input=ShellInput(Shell=TEST_SHELL_PATH),
                BaseDirectory=base,
                Phase=ShellPhase.READ_OUT,
                PendingExitCode=0,
            ),
        )
        current = replace(current, MaxOutputLength=limit)
        serialized_bytes = 0
        updates = 0
        async with LocalOperationManager() as manager, asyncio.timeout(15):
            await manager.add(current)
            async for update in manager.updates():
                serialized_bytes += len(encode(update))
                updates += 1
                assert updates <= 6, (
                    f"limit {limit} produced {updates} updates totaling {serialized_bytes} bytes"
                )
                assert serialized_bytes <= 32 * limit, (
                    f"limit {limit} produced {updates} updates totaling {serialized_bytes} bytes"
                )
                assert update.Status not in (Status.FAILED, Status.CANCELED), bytes(update.State)
                if update.Status == Status.COMPLETED:
                    state = shell_state(update)
                    assert state.Result is not None
                    assert state.Result.OutSize == len(contents)
                    assert state.Result.ErrSize == len(contents)
                    assert state.OutTruncated
                    assert state.ErrTruncated
                    break
            else:
                pytest.fail("manager closed before the capture completed")
        if previous_bytes:
            assert serialized_bytes <= 4 * previous_bytes + 4096
            assert updates == previous_updates
        previous_bytes, previous_updates = serialized_bytes, updates
