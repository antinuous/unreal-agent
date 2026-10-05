"""Go-style error values: wrapped messages, joined errors, and `errors.Is` traversal."""

import os
from collections.abc import Iterator


class PrimitiveError(Exception):
    """An error whose message follows the Go implementation. `__cause__` is the wrapped error."""


class JoinedError(PrimitiveError):
    """The equivalent of Go's `errors.Join`: messages are joined by newlines."""

    def __init__(self, errors: tuple[BaseException, ...]) -> None:
        super().__init__("\n".join(describe(error) for error in errors))
        self.errors = errors


class ContextCanceledError(PrimitiveError):
    def __init__(self, message: str = "context canceled") -> None:
        super().__init__(message)


class DeadlineExceededError(PrimitiveError):
    def __init__(self, message: str = "context deadline exceeded") -> None:
        super().__init__(message)


class FileClosedError(PrimitiveError):
    def __init__(self, message: str = "file already closed") -> None:
        super().__init__(message)


class UnexpectedEOFError(PrimitiveError):
    def __init__(self, message: str = "unexpected EOF") -> None:
        super().__init__(message)


def describe(error: BaseException) -> str:
    """Return a Go-like message. OS errors use the lowercase strerror text."""
    if isinstance(error, OSError) and error.errno is not None:
        text = os.strerror(error.errno)
        return text[:1].lower() + text[1:]
    text = str(error)
    return text if text else type(error).__name__


def wrap(prefix: str, error: BaseException) -> PrimitiveError:
    wrapped = PrimitiveError(f"{prefix}: {describe(error)}")
    wrapped.__cause__ = error
    return wrapped


def join(*errors: BaseException | None) -> BaseException | None:
    present = tuple(error for error in errors if error is not None)
    if not present:
        return None
    return JoinedError(present)


def causes(error: BaseException) -> Iterator[BaseException]:
    """Walk the error tree depth first like Go's `errors.Is`."""
    stack = [error]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, JoinedError):
            stack.extend(reversed(current.errors))
        elif isinstance(current, BaseExceptionGroup):
            stack.extend(reversed(current.exceptions))
        elif current.__cause__ is not None:
            stack.append(current.__cause__)


def is_error(error: BaseException | None, kind: type[BaseException]) -> bool:
    return error is not None and any(isinstance(cause, kind) for cause in causes(error))


def has_errno(error: BaseException | None, *codes: int) -> bool:
    if error is None:
        return False
    return any(isinstance(cause, OSError) and cause.errno in codes for cause in causes(error))


def is_only_context_cancellation(error: BaseException) -> bool:
    if isinstance(error, JoinedError):
        return bool(error.errors) and all(is_only_context_cancellation(e) for e in error.errors)
    if isinstance(error, ContextCanceledError | DeadlineExceededError):
        return True
    if error.__cause__ is not None:
        return is_only_context_cancellation(error.__cause__)
    return False
