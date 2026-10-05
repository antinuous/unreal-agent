"""Descriptor wrapper with Go `*os.File` close semantics."""

import os

from unreal_agent.primitives._errors import FileClosedError


class File:
    """Owns one descriptor. Use after close raises `FileClosedError`."""

    __slots__ = ("_fd", "name")

    def __init__(self, fd: int, name: str) -> None:
        self._fd = fd
        self.name = name

    @classmethod
    def open(cls, path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> File:
        return cls(os.open(path, flags | os.O_CLOEXEC, mode, dir_fd=dir_fd), path)

    @property
    def closed(self) -> bool:
        return self._fd < 0

    def fileno(self) -> int:
        if self._fd < 0:
            raise FileClosedError()
        return self._fd

    def close(self) -> None:
        fd = self.fileno()
        self._fd = -1
        os.close(fd)

    def stat(self) -> os.stat_result:
        return os.fstat(self.fileno())

    def sync(self) -> None:
        os.fsync(self.fileno())

    def truncate(self, size: int) -> None:
        os.ftruncate(self.fileno(), size)

    def read_at(self, size: int, offset: int) -> bytes:
        """Read up to `size` bytes at `offset`; a shorter result means end of file."""
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = os.pread(self.fileno(), remaining, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def write(self, data: bytes) -> int:
        return os.write(self.fileno(), data)

    def __repr__(self) -> str:
        return f"File({self.name!r}, fd={self._fd})"
