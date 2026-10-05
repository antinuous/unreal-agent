"""Runner errors whose messages follow the Go implementation."""

from typing import overload

from unreal_agent.sessionstore.localfile._errors import message

__all__ = ["JoinedError", "RunnerError", "describe", "join", "wrap"]


class RunnerError(Exception):
    """A runner failure. `__cause__` is the wrapped error, like Go's `%w`."""


class JoinedError(RunnerError):
    """Go's errors.Join: the messages joined by newlines."""

    def __init__(self, errors: tuple[BaseException, ...]) -> None:
        super().__init__("\n".join(describe(error) for error in errors))
        self.errors = errors


def describe(error: BaseException, op: str = "") -> str:
    """The Go-style message, formatting OS errors like Go's *PathError."""
    return message(error, op) or type(error).__name__


def wrap(prefix: str, error: BaseException, op: str = "") -> RunnerError:
    wrapped = RunnerError(f"{prefix}: {describe(error, op)}")
    wrapped.__cause__ = error
    return wrapped


@overload
def join(first: BaseException, /, *errors: BaseException | None) -> BaseException: ...


@overload
def join(*errors: BaseException | None) -> BaseException | None: ...


def join(*errors: BaseException | None) -> BaseException | None:
    present = tuple(error for error in errors if error is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return JoinedError(present)
