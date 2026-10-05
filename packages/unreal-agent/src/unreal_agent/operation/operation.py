from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, Protocol

from unreal_agent.gojson import Int64, OmitZero, Raw, Struct, Uint32
from unreal_agent.primitives import PrimitiveDispatchType

__all__ = [
    "ID",
    "Manager",
    "Operation",
    "OperationError",
    "PrimitiveDispatch",
    "Spec",
    "Status",
    "Step",
    "Type",
    "UnsupportedError",
    "Version",
]

type Type = str
type Version = int
type ID = str


class OperationError(Exception):
    """An operation could not be decoded, validated, or advanced."""


class UnsupportedError(OperationError):
    """The operation type, version, or status cannot be handled (Go `ErrUnsupported`)."""

    def __init__(self, message: str = "unsupported operation") -> None:
        super().__init__(message)


def unsupported(prefix: str) -> UnsupportedError:
    return UnsupportedError(f"{prefix}: unsupported operation")


class Status(StrEnum):
    NONE = ""
    READY = "ready"
    AWAITING = "awaiting"
    CANCELING = "canceling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


_Status = Status


class Spec(Struct):
    MaxOutputLength: Annotated[Int64, OmitZero] = 0
    Type: str = ""
    Version: Uint32 = 0
    State: Annotated[Raw, OmitZero] = Raw(b"")
    Idempotency: Annotated[Raw, OmitZero] = Raw(b"")


class Operation(Struct):
    MaxOutputLength: Annotated[Int64, OmitZero] = 0
    ID: str = ""
    Type: str = ""
    Version: Uint32 = 0
    Status: _Status = _Status.NONE
    State: Annotated[Raw, OmitZero] = Raw(b"")
    Idempotency: Annotated[Raw, OmitZero] = Raw(b"")


@dataclass(slots=True)
class PrimitiveDispatch:
    type: PrimitiveDispatchType
    data: Any = None


@dataclass(slots=True)
class Step:
    """`operation` is the next durable checkpoint; None leaves the checkpoint unchanged."""

    operation: Operation | None = None
    dispatches: list[PrimitiveDispatch] = field(default_factory=list)


class Manager(Protocol):
    async def add(self, operation: Operation, /) -> None:
        """Start an operation at most once for each ID during the manager's lifetime.

        Raises `UnsupportedError` when the operation type or version cannot be handled.
        """
        ...

    async def cancel(self, id: ID, reason: str, /) -> None: ...

    def updates(self) -> AsyncIterator[Operation]: ...
