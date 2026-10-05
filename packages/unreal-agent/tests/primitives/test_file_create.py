import asyncio
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from unreal_agent.primitives import (
    IO_CREATE_DEFAULT_MODE,
    CorrelationID,
    IOCreateKind,
    IOCreateRequest,
    IOCreateResult,
    PrimitiveEvent,
    PrimitiveEventType,
    SourceID,
    create,
)
from unreal_agent.primitives.channel import Channel

from tests.primitives._helpers import collect_events, failure_text, single_event, write_test_file


async def create_events(request: IOCreateRequest) -> list[PrimitiveEvent]:
    events: Channel[PrimitiveEvent] = Channel()
    create(request, events)
    return await collect_events(events)


def contents(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def exists(path: str | Path) -> bool:
    return os.path.exists(path)


def is_directory(path: str | Path) -> bool:
    return os.path.isdir(path)


def permissions(path: str) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


async def test_create_file_creates_empty_file(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    request = IOCreateRequest(
        source=SourceID("operation-1"),
        correlation_id=CorrelationID("create-1"),
        kind=IOCreateKind.REGULAR_FILE,
        path=path,
        mode=IO_CREATE_DEFAULT_MODE,
    )
    event = single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
    assert (event.source, event.correlation_id) == (request.source, request.correlation_id)
    assert event.result == IOCreateResult(IOCreateKind.REGULAR_FILE)
    assert contents(path) == b""
    assert permissions(path) & ~0o664 == 0


async def test_create_file_uses_requested_mode(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    single_event(
        await create_events(IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=path, mode=0o600)),
        PrimitiveEventType.IO_CREATE_COMPLETED,
    )
    assert permissions(path) == 0o600


async def test_create_file_with_zero_mode_is_replayable(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    request = IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=path)
    for _ in range(2):
        single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
    assert permissions(path) == 0


async def test_create_directory_creates_empty_directory(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    request = IOCreateRequest(
        source=SourceID("operation-1"),
        correlation_id=CorrelationID("create-1"),
        kind=IOCreateKind.DIRECTORY,
        path=path,
        mode=0o700,
    )
    event = single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
    assert event.result == IOCreateResult(IOCreateKind.DIRECTORY)
    assert is_directory(path)
    assert permissions(path) == 0o700
    assert os.listdir(path) == []


async def test_create_directory_accepts_trailing_separator(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    event = single_event(
        await create_events(
            IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=path + os.sep, mode=0o700)
        ),
        PrimitiveEventType.IO_CREATE_COMPLETED,
    )
    assert event.result == IOCreateResult(IOCreateKind.DIRECTORY)
    assert is_directory(path)


async def test_create_directory_with_zero_mode_is_replayable(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    request = IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=path)
    for _ in range(2):
        single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
    assert is_directory(path)
    assert permissions(path) == 0
    os.chmod(path, 0o700)


async def test_create_uses_process_umask(tmp_path: Path) -> None:
    old_umask = os.umask(0o027)
    try:
        for request, want in (
            (
                IOCreateRequest(
                    kind=IOCreateKind.REGULAR_FILE, path=str(tmp_path / "file"), mode=0o666
                ),
                0o640,
            ),
            (
                IOCreateRequest(
                    kind=IOCreateKind.DIRECTORY, path=str(tmp_path / "directory"), mode=0o777
                ),
                0o750,
            ),
        ):
            for _ in range(2):
                single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
            assert permissions(request.path) == want
    finally:
        os.umask(old_umask)


async def test_create_accepts_existing_path_with_restrictive_mode_without_changing_it(
    tmp_path: Path,
) -> None:
    file = tmp_path / "file"
    file.write_bytes(b"")
    directory = tmp_path / "directory"
    directory.mkdir()
    os.chmod(file, 0o400)
    os.chmod(directory, 0o500)
    try:
        for request, want in (
            (IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=str(file), mode=0o600), 0o400),
            (IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=str(directory), mode=0o700), 0o500),
        ):
            single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
            assert permissions(request.path) == want
    finally:
        os.chmod(directory, 0o700)


async def test_create_directory_preserves_path_resolution(tmp_path: Path) -> None:
    separator = os.sep
    path = f"{tmp_path}{separator}missing{separator}..{separator}created{separator}"
    event = single_event(
        await create_events(IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=path, mode=0o700)),
        PrimitiveEventType.FAILED,
    )
    assert failure_text(event)
    assert not exists(tmp_path / "created")


async def test_create_rejects_unsupported_kind(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    event = single_event(await create_events(IOCreateRequest(path=path)), PrimitiveEventType.FAILED)
    assert failure_text(event)
    assert not exists(path)


async def test_create_file_preserves_existing_file(tmp_path: Path) -> None:
    existing = b"existing contents"
    path = write_test_file(tmp_path, existing)
    request = IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=path, mode=0o600)
    event = single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
    assert event.result == IOCreateResult(request.kind)
    assert contents(path) == existing


async def test_create_directory_allows_existing_directory(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    request = IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=str(tmp_path), mode=0o700)
    event = single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
    assert event.result == IOCreateResult(request.kind)


async def test_create_directory_concurrent_calls_are_idempotent(tmp_path: Path) -> None:
    invocation_count = 16
    path = str(tmp_path / "created")
    request = IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=path, mode=0o700)
    start = asyncio.Event()

    async def invoke() -> list[PrimitiveEvent]:
        await start.wait()
        return await create_events(request)

    invocations = [asyncio.create_task(invoke()) for _ in range(invocation_count)]
    start.set()
    for events in await asyncio.gather(*invocations):
        event = single_event(events, PrimitiveEventType.IO_CREATE_COMPLETED)
        assert event.result == IOCreateResult(IOCreateKind.DIRECTORY)
    assert is_directory(path)


async def test_create_accepts_existing_path_with_broader_mode_without_changing_it(
    tmp_path: Path,
) -> None:
    file = write_test_file(tmp_path, b"")
    os.chmod(file, 0o644)
    directory = tmp_path / "directory"
    directory.mkdir()
    os.chmod(directory, 0o755)
    for request, want in (
        (IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=file, mode=0o600), 0o644),
        (IOCreateRequest(kind=IOCreateKind.DIRECTORY, path=str(directory), mode=0o700), 0o755),
    ):
        single_event(await create_events(request), PrimitiveEventType.IO_CREATE_COMPLETED)
        assert permissions(request.path) == want


async def test_create_rejects_existing_path_of_wrong_kind(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    for request, want in (
        (
            IOCreateRequest(
                kind=IOCreateKind.DIRECTORY, path=write_test_file(tmp_path, b""), mode=0o700
            ),
            "not a directory",
        ),
        (
            IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=str(directory), mode=0o600),
            "not a regular file",
        ),
    ):
        event = single_event(await create_events(request), PrimitiveEventType.FAILED)
        assert want in failure_text(event)


async def test_create_rejects_existing_symlink(tmp_path: Path) -> None:
    target = write_test_file(tmp_path, b"", "target")
    path = tmp_path / "link"
    path.symlink_to(target)
    single_event(
        await create_events(
            IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=str(path), mode=0o600)
        ),
        PrimitiveEventType.FAILED,
    )


async def test_create_file_canceled_before_start(tmp_path: Path) -> None:
    path = str(tmp_path / "canceled")
    events: Channel[PrimitiveEvent] = Channel()
    task = create(
        IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=path, mode=IO_CREATE_DEFAULT_MODE),
        events,
    )
    task.cancel()
    single_event(await collect_events(events), PrimitiveEventType.CANCELED)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not exists(path)


async def test_create_file_failure_is_terminal(tmp_path: Path) -> None:
    request = IOCreateRequest(
        source=SourceID("operation-1"),
        correlation_id=CorrelationID("create-1"),
        kind=IOCreateKind.REGULAR_FILE,
        path=str(tmp_path / "missing" / "created"),
        mode=IO_CREATE_DEFAULT_MODE,
    )
    event = single_event(await create_events(request), PrimitiveEventType.FAILED)
    assert failure_text(event)
    assert (event.source, event.correlation_id) == (request.source, request.correlation_id)


_CREATE_HELPER = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from unreal_agent.primitives import (
        TERMINAL_EVENT_TYPES, IOCreateKind, IOCreateRequest, PrimitiveEventType, create,
    )
    from unreal_agent.primitives.channel import Channel

    start, ready = int(sys.argv[1]), int(sys.argv[2])
    os.write(ready, b"\\x01")
    os.close(ready)
    while os.read(start, 4096):
        pass
    os.close(start)

    async def main():
        events = Channel()
        request = IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=sys.argv[3], mode=0o600)
        create(request, events)
        collected = []
        while True:
            event = await events.get()
            collected.append(event)
            if event.type in TERMINAL_EVENT_TYPES:
                return collected

    events = asyncio.run(main())
    result = {"Completed": False, "Kind": 0, "Error": ""}
    if len(events) != 1:
        result["Error"] = f"events = {len(events)}, want 1"
    elif events[0].type == PrimitiveEventType.IO_CREATE_COMPLETED:
        result["Completed"] = True
        result["Kind"] = int(events[0].result.kind)
    elif events[0].type == PrimitiveEventType.FAILED:
        result["Error"] = events[0].result.error
    else:
        result["Error"] = f"unexpected event type {events[0].type!r}"
    sys.stdout.write(json.dumps(result))
    """
)


def test_create_file_concurrent_processes_are_idempotent(tmp_path: Path) -> None:
    process_count = 6
    path = str(tmp_path / "created")
    start_reader, start_writer = os.pipe()
    ready_reader, ready_writer = os.pipe()
    children: list[subprocess.Popen[bytes]] = []
    try:
        for _ in range(process_count):
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _CREATE_HELPER,
                        str(start_reader),
                        str(ready_writer),
                        path,
                    ],
                    pass_fds=(start_reader, ready_writer),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        os.close(start_reader)
        os.close(ready_writer)
        ready = b""
        while len(ready) < process_count:
            chunk = os.read(ready_reader, process_count - len(ready))
            assert chunk, "a child exited before signaling readiness"
            ready += chunk
        os.close(ready_reader)
        os.close(start_writer)

        results = []
        for index, child in enumerate(children):
            stdout, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, f"child {index}: {stdout!r} {stderr!r}"
            results.append(json.loads(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()

    assert sum(result["Completed"] for result in results) == process_count
    assert sum(1 for result in results if result["Error"]) == 0
    assert contents(path) == b""
    for result in results:
        assert result["Completed"]
        assert result["Kind"] == IOCreateKind.REGULAR_FILE
    assert os.listdir(tmp_path) == ["created"]


@pytest.mark.parametrize(
    ("kind", "path", "want"),
    [
        pytest.param(
            IOCreateKind.REGULAR_FILE,
            "parent\x00name/created",
            'open parent directory "parent\\x00name/": open parent\x00name/: invalid argument',
            id="parent",
        ),
        pytest.param(
            IOCreateKind.REGULAR_FILE,
            "created\x00name",
            'create "created\\x00name": invalid argument',
            id="file",
        ),
        pytest.param(
            IOCreateKind.DIRECTORY,
            "created\x00name",
            'create "created\\x00name": invalid argument',
            id="directory",
        ),
    ],
)
async def test_create_reports_invalid_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: IOCreateKind, path: str, want: str
) -> None:
    monkeypatch.chdir(tmp_path)
    event = single_event(
        await create_events(IOCreateRequest(kind=kind, path=path, mode=0o700)),
        PrimitiveEventType.FAILED,
    )
    assert failure_text(event) == want
    assert os.listdir(tmp_path) == []


async def test_create_parent_open_failure_includes_path_error(tmp_path: Path) -> None:
    parent = f"{tmp_path}/missing/"
    event = single_event(
        await create_events(IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=parent + "x")),
        PrimitiveEventType.FAILED,
    )
    assert failure_text(event) == (
        f'open parent directory "{parent}": open {parent}: no such file or directory'
    )
