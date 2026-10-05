import asyncio
import errno
import os
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Final

from msgspec.structs import replace

from unreal_agent._filepath import go_abs, go_join
from unreal_agent._go import go_quote
from unreal_agent.gojson import Timestamp
from unreal_agent.inbox import Input
from unreal_agent.operation import Operation
from unreal_agent.session import ID as SessionID
from unreal_agent.session import Turn, TurnID
from unreal_agent.sessionstore import (
    Item,
    ModelResponse,
    Observer,
    ObserverID,
    Page,
    ResumeState,
    SessionInfo,
    Snapshot,
    StoreError,
    ToolCallStatus,
)
from unreal_agent.sessionstore.localfile._errors import join, message, validate_session_id, wrap
from unreal_agent.sessionstore.localfile.codec import (
    ItemRecord,
    OperationRecord,
    RecordType,
    decode_log,
    encode_initial_log,
    encode_record,
)
from unreal_agent.sessionstore.localfile.state import (
    SessionHead,
    StoredState,
    fork_stored_state,
    new_stored_state,
    validate_operation,
)

__all__ = ["SESSION_FILE_SUFFIX", "Store"]

SESSION_FILE_SUFFIX: Final = ".session.jsonl"

type _Apply = Callable[[SessionHead], tuple[RecordType, Any, Item | None]]


@dataclass(slots=True)
class _CachedWriteState:
    head: SessionHead
    committed_size: int


class Store:
    """Session history and operation state in `<directory>/<id>.session.jsonl`.

    Blocking file work runs in worker threads. Writes for one session run one
    at a time in call order. A write that has started finishes even if the
    calling task is cancelled; its observers are notified before the
    cancellation propagates, so observers see every persisted item.
    """

    __slots__ = ("_cache", "_cache_lock", "_directory", "_locks", "_observers")

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        text = os.fspath(directory)
        if text.strip() == "":
            raise StoreError("session store directory is empty")
        absolute = go_abs(text)
        try:
            _make_directories(absolute)
        except OSError as error:
            raise StoreError(
                f"create session store directory: {message(error, 'mkdir')}"
            ) from error
        self._directory = absolute
        self._cache: dict[SessionID, _CachedWriteState] = {}
        self._cache_lock = threading.Lock()
        self._observers: dict[ObserverID, Observer] = {}
        self._locks: dict[SessionID, asyncio.Lock] = {}

    @property
    def directory(self) -> str:
        return self._directory

    def add_observer(self, observer: Observer, /) -> ObserverID:
        id = uuid.uuid4()
        self._observers[id] = observer
        return id

    def remove_observer(self, id: ObserverID, /) -> None:
        self._observers.pop(id, None)

    async def create(self, id: SessionID, /) -> Snapshot:
        validate_session_id(id)
        async with self._session_lock(id):
            state, cancelled = await _complete(partial(self._create, id))
        if cancelled is not None:
            raise cancelled
        return _copy_snapshot(state.head.snapshot)

    async def list_sessions(self) -> list[SessionInfo]:
        return await asyncio.to_thread(self._list_sessions)

    async def inspect(self, id: SessionID, /) -> Snapshot:
        validate_session_id(id)
        state, _ = await asyncio.to_thread(self._read_state, id)
        return state.head.snapshot

    async def items(self, id: SessionID, after: int, limit: int, /) -> Page:
        if limit <= 0:
            raise StoreError("item page limit must be positive")
        if after < 0:
            raise StoreError("item page cursor is negative")
        validate_session_id(id)
        state, _ = await asyncio.to_thread(self._read_state, id)
        count = len(state.items)
        start = min(after, count)
        end = min(start + limit, count)
        items = state.items[start:end]
        return Page(
            items=items,
            next_after=items[-1].Sequence if items else after,
            more=end < count,
        )

    async def append_input(self, id: SessionID, input: Input, /) -> None:
        def apply(head: SessionHead) -> tuple[RecordType, Any, Item | None]:
            item = head.append_input(input, Timestamp.now())
            return RecordType.ITEM, ItemRecord(Item=item), item

        await self._write(id, apply)

    async def append_turn(self, id: SessionID, turn: Turn, /) -> None:
        def apply(head: SessionHead) -> tuple[RecordType, Any, Item | None]:
            item = head.append_turn(turn, Timestamp.now())
            return RecordType.ITEM, ItemRecord(Item=item), item

        await self._write(id, apply)

    async def append_model_response(self, id: SessionID, response: ModelResponse, /) -> None:
        def apply(head: SessionHead) -> tuple[RecordType, Any, Item | None]:
            item = head.append_model_response(response, Timestamp.now())
            return RecordType.ITEM, ItemRecord(Item=item), item

        await self._write(id, apply)

    async def append_tool_call_status(self, id: SessionID, status: ToolCallStatus, /) -> None:
        operations = list(status.Operations)

        def apply(head: SessionHead) -> tuple[RecordType, Any, Item | None]:
            item = head.append_tool_call_status(
                replace(status, Operations=[]), operations, Timestamp.now()
            )
            record = ItemRecord(Item=item, Operations=operations)
            return RecordType.ITEM, record, replace(item, Data=replace(status))

        await self._write(id, apply)

    async def save_operation(self, id: SessionID, operation: Operation, /) -> None:
        validate_operation(operation)

        def apply(head: SessionHead) -> tuple[RecordType, Any, Item | None]:
            head.save_operation(operation)
            return RecordType.OPERATION, OperationRecord(Operation=operation), None

        await self._write(id, apply)

    async def resume(self, id: SessionID, /) -> ResumeState:
        validate_session_id(id)
        state, _ = await asyncio.to_thread(self._read_state, id)
        return state.resume()

    async def fork(
        self, id: SessionID, parent_id: SessionID, previous_turn_id: TurnID, /
    ) -> Snapshot:
        validate_session_id(id)
        if id == parent_id:
            raise StoreError(f"fork session {go_quote(id)} onto itself")
        state = await asyncio.to_thread(self._fork_state, id, parent_id, previous_turn_id)
        async with self._session_lock(id):
            _, cancelled = await _complete(partial(self._publish_fork, state))
        self._notify(id, state.items[-1])
        if cancelled is not None:
            raise cancelled
        return _copy_snapshot(state.head.snapshot)

    def session_path(self, id: SessionID) -> str:
        return go_join(self._directory, id + SESSION_FILE_SUFFIX)

    async def _write(self, id: SessionID, apply: _Apply) -> None:
        validate_session_id(id)
        async with self._session_lock(id):
            item, cancelled = await _complete(partial(self._write_record, id, apply))
        if item is not None:
            self._notify(id, item)
        if cancelled is not None:
            raise cancelled

    def _session_lock(self, id: SessionID) -> asyncio.Lock:
        lock = self._locks.get(id)
        if lock is None:
            lock = self._locks[id] = asyncio.Lock()
        return lock

    def _notify(self, id: SessionID, item: Item) -> None:
        for observer_id in list(self._observers):
            observer = self._observers.get(observer_id)
            if observer is not None:
                observer(id, item)

    def _create(self, id: SessionID) -> StoredState:
        state = new_stored_state(id, Timestamp.now())
        try:
            self._publish_initial_state(state)
        except StoreError as error:
            raise wrap(f"create session {go_quote(id)}", error) from error
        return state

    def _fork_state(
        self, id: SessionID, parent_id: SessionID, previous_turn_id: TurnID
    ) -> StoredState:
        try:
            parent, _ = self._read_state(parent_id)
        except StoreError as error:
            raise wrap(f"fork parent {go_quote(parent_id)}", error) from error
        return fork_stored_state(parent, id, previous_turn_id, Timestamp.now())

    def _publish_fork(self, state: StoredState) -> None:
        try:
            self._publish_initial_state(state)
        except StoreError as error:
            raise wrap(f"fork session {go_quote(state.head.id)}", error) from error

    def _read_state(self, id: SessionID) -> tuple[StoredState, int]:
        validate_session_id(id)
        prefix = f"read session {go_quote(id)}"
        path = self.session_path(id)
        try:
            with open(path, "rb") as file:
                encoded = file.read()
        except OSError as error:
            raise wrap(prefix, error, op="open") from error
        try:
            state, committed_size = decode_log(encoded)
        except StoreError as error:
            raise wrap(prefix, error) from error
        if state.head.id != id:
            raise StoreError(f"{prefix}: file contains session {go_quote(state.head.id)}")
        return state, committed_size

    def _load_write_state(self, id: SessionID) -> tuple[SessionHead, int]:
        cached = self._get_cached_write_state(id)
        if cached is not None:
            return cached.head, cached.committed_size
        state, committed_size = self._read_state(id)
        self._put_cached_write_state(id, state.head, committed_size)
        return state.head, committed_size

    def _write_record(self, id: SessionID, apply: _Apply) -> Item | None:
        head, committed_size = self._load_write_state(id)
        kind, value, item = apply(head)
        try:
            encoded = encode_record(kind, value)
            append_file(self.session_path(id), committed_size, encoded)
        except StoreError:
            self._evict_cached_write_state(id)
            raise
        self._put_cached_write_state(id, head, committed_size + len(encoded))
        return item

    def _publish_initial_state(self, state: StoredState) -> None:
        id = state.head.id
        encoded = encode_initial_log(state.head.snapshot.session, state.items)
        try:
            publish_file(self._directory, self.session_path(id), encoded)
        except StoreError:
            self._evict_cached_write_state(id)
            raise
        self._put_cached_write_state(id, state.head, len(encoded))

    def _get_cached_write_state(self, id: SessionID) -> _CachedWriteState | None:
        with self._cache_lock:
            return self._cache.get(id)

    def _put_cached_write_state(self, id: SessionID, head: SessionHead, size: int) -> None:
        with self._cache_lock:
            self._cache[id] = _CachedWriteState(head, size)

    def _evict_cached_write_state(self, id: SessionID) -> None:
        with self._cache_lock:
            self._cache.pop(id, None)

    def _list_sessions(self) -> list[SessionInfo]:
        try:
            with os.scandir(self._directory) as iterator:
                entries = list(iterator)
        except OSError as error:
            raise wrap("list sessions", error, op="open") from error
        return list_session_entries(entries)


def list_session_entries(entries: Iterable[os.DirEntry[str]]) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    for entry in entries:
        if not entry.name.endswith(SESSION_FILE_SUFFIX):
            continue
        if not entry.is_file(follow_symlinks=False):
            continue
        id = entry.name.removesuffix(SESSION_FILE_SUFFIX)
        try:
            validate_session_id(id)
            info = entry.stat(follow_symlinks=False)
        except StoreError, OSError:
            continue
        sessions.append(SessionInfo(id=id, last_updated_at=Timestamp(info.st_mtime_ns)))
    sessions.sort(key=lambda info: info.id)
    return sessions


async def _complete[T](function: Callable[[], T]) -> tuple[T, asyncio.CancelledError | None]:
    """Run a blocking write to completion even if the calling task is cancelled.

    Returns the result and the cancellation to raise once the caller has
    published the write's effects.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        # Deliver a cancellation requested while the task was running (for
        # example by an observer) before the write starts, as Go checks ctx.
        await asyncio.sleep(0)
    future = asyncio.ensure_future(asyncio.to_thread(function))
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as error:
            if future.cancelled():
                raise
            cancelled = error
            if not future.done():
                continue
            try:
                result = future.result()
            except BaseException as failure:
                raise cancelled from failure
        except BaseException as error:
            if cancelled is not None:
                raise cancelled from error
            raise
        return result, cancelled


def _make_directories(path: str) -> None:
    """Create `path` and missing parents with mode 0700, like Go's os.MkdirAll."""
    missing: list[str] = []
    current = path
    while not os.path.isdir(current):
        if os.path.lexists(current):
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), current)
        missing.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            if not os.path.isdir(directory):
                raise


def _copy_snapshot(snapshot: Snapshot) -> Snapshot:
    return Snapshot(session=replace(snapshot.session))


def publish_file(directory: str, target: str, data: bytes) -> None:
    """Atomically replace `target` with `data`: write a temporary file, sync, rename, sync."""
    try:
        fd, temporary = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=directory)
    except OSError as error:
        raise wrap("create temporary session file", error, op="open") from error
    try:
        persist_temporary(fd, data)
        try:
            os.rename(temporary, target)
        except OSError as error:
            raise wrap("publish session file", error, op="rename") from error
    except StoreError as error:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        except OSError as remove_error:
            raise join(error, StoreError(message(remove_error, "remove"))) from error
        raise
    sync_directory(directory)


def append_file(path: str, committed_size: int, data: bytes) -> None:
    """Discard any incomplete tail after `committed_size`, then append and sync `data`."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)
    except OSError as error:
        raise wrap("open session log for append", error, op="open") from error
    failure: StoreError | None = None
    try:
        os.ftruncate(fd, committed_size)
    except OSError as error:
        failure = wrap("discard incomplete session record", error, op="truncate")
    if failure is None:
        try:
            _write_all(fd, data)
        except OSError as error:
            failure = wrap("append session record", error, op="write")
    if failure is None:
        try:
            sync_fd(fd)
        except OSError as error:
            failure = wrap("sync session log", error, op="sync")
    _close(fd, "close session log", failure)


def persist_temporary(fd: int, data: bytes) -> None:
    failure: StoreError | None = None
    try:
        _write_all(fd, data)
    except OSError as error:
        failure = wrap("write temporary session file", error, op="write")
    if failure is None:
        try:
            sync_fd(fd)
        except OSError as error:
            failure = wrap("sync temporary session file", error, op="sync")
    _close(fd, "close temporary session file", failure)


def sync_directory(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise wrap("open session store directory for sync", error, op="open") from error
    sync_directory_fd(fd)


def sync_directory_fd(fd: int) -> None:
    failure: StoreError | None = None
    try:
        sync_fd(fd)
    except OSError as error:
        failure = wrap("sync session store directory", error, op="sync")
    _close(fd, "close session store directory", failure)


def sync_fd(fd: int) -> None:
    """Flush a file to stable storage the way Go's File.Sync does."""
    if sys.platform == "darwin":
        import fcntl

        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except OSError as error:
            if error.errno != errno.ENOTSUP:
                raise
        else:
            return
    os.fsync(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while True:
        written = os.write(fd, view)
        view = view[written:]
        if not view:
            return


def _close(fd: int, label: str, failure: StoreError | None) -> None:
    try:
        os.close(fd)
    except OSError as error:
        close_failure = wrap(label, error, op="close")
        failure = close_failure if failure is None else join(failure, close_failure)
    if failure is not None:
        raise failure
