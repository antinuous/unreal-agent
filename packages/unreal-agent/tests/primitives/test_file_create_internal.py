import errno
import os
import stat
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from unreal_agent.primitives import IOCreateKind, IOCreateRequest
from unreal_agent.primitives._errors import (
    ContextCanceledError,
    FileClosedError,
    causes,
    is_error,
    is_only_context_cancellation,
)
from unreal_agent.primitives._fileio import File
from unreal_agent.primitives.file_create import (
    close_file,
    create_new_path,
    create_or_use_path,
    full_sync_result,
    make_new_directory_at,
    open_create_directory,
    open_new_file_at,
    persist_created_file,
    sync_created_directory,
    sync_created_file,
    unix_file_mode,
    use_existing_path,
)


@pytest.fixture
def parent_directory(tmp_path: Path) -> Iterator[tuple[str, File]]:
    parent = open_create_directory(str(tmp_path))
    yield str(tmp_path), parent
    if not parent.closed:
        parent.close()


def closed_temp_file(directory: Path) -> File:
    descriptor, path = tempfile.mkstemp(dir=directory, prefix="created-")
    file = File(descriptor, path)
    file.close()
    return file


def test_create_new_path_uses_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = create_new_path(
        IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path="created", mode=0o664),
        threading.Event(),
    )
    assert result.kind == IOCreateKind.REGULAR_FILE
    assert os.path.exists("created")


def test_create_new_path_uses_requested_mode(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    create_new_path(
        IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=path, mode=0o600), threading.Event()
    )
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_use_existing_path_requests_retry_for_missing_entry(
    parent_directory: tuple[str, File],
) -> None:
    parent_path, parent = parent_directory
    request = IOCreateRequest(
        kind=IOCreateKind.REGULAR_FILE, path=os.path.join(parent_path, "missing"), mode=0o600
    )
    assert use_existing_path(request, "missing", parent_path, parent) is None
    parent.stat()
    assert not os.path.exists(request.path)


def test_create_or_use_path_honors_cancellation_before_mutation(
    parent_directory: tuple[str, File],
) -> None:
    parent_path, parent = parent_directory
    request = IOCreateRequest(
        kind=IOCreateKind.REGULAR_FILE, path=os.path.join(parent_path, "created"), mode=0o600
    )
    canceled = threading.Event()
    canceled.set()
    with pytest.raises(Exception) as raised:  # noqa: PT011
        create_or_use_path(request, "created", parent_path, parent, unix_file_mode(0o600), canceled)
    assert is_error(raised.value, ContextCanceledError)
    assert not os.path.exists(request.path)


def test_create_or_use_path_preserves_close_failure_during_cancellation(
    parent_directory: tuple[str, File],
) -> None:
    parent_path, parent = parent_directory
    parent.close()
    canceled = threading.Event()
    canceled.set()
    with pytest.raises(Exception) as raised:  # noqa: PT011
        create_or_use_path(IOCreateRequest(), "created", parent_path, parent, 0o600, canceled)
    assert is_error(raised.value, ContextCanceledError)
    assert not is_only_context_cancellation(raised.value)


def test_unix_file_mode() -> None:
    mode = 0o640 | stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
    assert unix_file_mode(mode) == 0o640 | 0o4000 | 0o2000 | 0o1000


def test_create_new_path_rejects_non_unix_mode_bits(tmp_path: Path) -> None:
    path = str(tmp_path / "created")
    with pytest.raises(Exception, match="unsupported mode bits"):
        create_new_path(
            IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=path, mode=stat.S_IFDIR | 0o700),
            threading.Event(),
        )
    assert not os.path.exists(path)


def test_open_create_directory_rejects_file(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        open_create_directory(str(path))


def test_open_new_file_at_rejects_closed_directory(tmp_path: Path) -> None:
    parent = open_create_directory(str(tmp_path))
    parent.close()
    with pytest.raises(FileClosedError):
        open_new_file_at(parent, "created", unix_file_mode(0o664))
    assert not (tmp_path / "created").exists()


def test_make_new_directory_at_rejects_closed_directory(tmp_path: Path) -> None:
    parent = open_create_directory(str(tmp_path))
    parent.close()
    with pytest.raises(FileClosedError):
        make_new_directory_at(parent, "created", 0o700)
    assert not (tmp_path / "created").exists()


def test_sync_created_file_reports_file_sync_failure(
    tmp_path: Path, parent_directory: tuple[str, File]
) -> None:
    _, parent = parent_directory
    file = closed_temp_file(tmp_path)
    with pytest.raises(Exception, match="sync file"):
        sync_created_file(file, parent)


def test_sync_created_file_reports_parent_sync_failure(tmp_path: Path) -> None:
    descriptor, path = tempfile.mkstemp(dir=tmp_path, prefix="created-")
    file = File(descriptor, path)
    parent_path = tmp_path / "parent"
    parent_path.mkdir()
    parent = open_create_directory(str(parent_path))
    parent.close()
    try:
        with pytest.raises(Exception, match="sync parent directory"):
            sync_created_file(file, parent)
    finally:
        file.close()


def test_sync_created_directory_reports_parent_sync_failure(tmp_path: Path) -> None:
    parent = open_create_directory(str(tmp_path))
    parent.close()
    with pytest.raises(Exception, match="sync parent directory"):
        sync_created_directory(parent)


def test_persist_created_file_reports_sync_and_close_failures(
    tmp_path: Path, parent_directory: tuple[str, File]
) -> None:
    parent_path, parent = parent_directory
    file = closed_temp_file(tmp_path)
    with pytest.raises(Exception) as raised:  # noqa: PT011
        persist_created_file(
            IOCreateRequest(kind=IOCreateKind.REGULAR_FILE, path=file.name),
            parent_path,
            file,
            parent,
        )
    for want in ("sync file", "close file"):
        assert want in str(raised.value)


def test_close_file_reports_failure(tmp_path: Path) -> None:
    file = closed_temp_file(tmp_path)
    error = close_file("file", file.name, file)
    assert error is not None
    assert is_error(error, FileClosedError)


@pytest.mark.parametrize(
    ("error", "want_error"),
    [
        pytest.param(None, False, id="success"),
        pytest.param(OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP)), False, id="unsupported"),
        pytest.param(Exception("flush failed"), True, id="failure"),
    ],
)
def test_full_sync_result(error: BaseException | None, want_error: bool) -> None:
    result = full_sync_result(error)
    if not want_error:
        assert result is None
        return
    assert result is not None
    assert error is not None
    assert any(cause is error for cause in causes(result))


@pytest.mark.skip(
    reason="Go-only EINTR retry loop: Python retries EINTR inside system calls (PEP 475)"
)
def test_retry_eintr() -> None:
    pass
