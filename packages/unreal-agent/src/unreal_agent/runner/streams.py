"""Byte streams consumed by the runner."""

from typing import Protocol

__all__ = ["Reader", "TeeWriter", "Writer", "write_flushed"]


class Reader(Protocol):
    def read(self) -> bytes: ...


class Writer(Protocol):
    def write(self, data: bytes, /) -> object: ...

    def flush(self) -> object: ...


def write_flushed(output: Writer, data: bytes) -> None:
    output.write(data)
    output.flush()


class TeeWriter:
    """Go's io.MultiWriter: write to each writer in order, stopping at the first error."""

    __slots__ = ("_writers",)

    def __init__(self, *writers: Writer) -> None:
        self._writers = writers

    def write(self, data: bytes, /) -> int:
        for writer in self._writers:
            write_flushed(writer, data)
        return len(data)

    def flush(self) -> None:
        for writer in self._writers:
            writer.flush()
