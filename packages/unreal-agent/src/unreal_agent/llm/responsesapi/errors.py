from typing import Self

__all__ = ["APIError", "StreamEndedError", "TransportError"]


class APIError(Exception):
    """An error reported by the provider, over HTTP or inside the event stream.

    `str()` includes the context added by `wrap`, like a wrapped Go error;
    `describe()` is the provider error alone.
    """

    def __init__(
        self,
        status_code: int = 0,
        code: str = "",
        message: str = "",
        param: str = "",
        type: str = "",
    ) -> None:
        super().__init__(status_code, code, message, param, type)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        self.type = type
        self.context: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.code:
            return f"responses API error {self.code}: {self.message}"
        if self.status_code:
            return f"responses API request failed with status {self.status_code}: {self.message}"
        return "responses API request failed: " + self.message

    def wrap(self, prefix: str) -> Self:
        """Return a copy whose message is prefixed, keeping the fields. The cause is `self`."""
        wrapped = type(self)(self.status_code, self.code, self.message, self.param, self.type)
        wrapped.context = (prefix, *self.context)
        wrapped.__cause__ = self
        return wrapped

    def __str__(self) -> str:
        return ": ".join((*self.context, self.describe()))

    def __repr__(self) -> str:
        return (
            f"APIError(status_code={self.status_code!r}, code={self.code!r}, "
            f"message={self.message!r}, param={self.param!r}, type={self.type!r})"
        )


class TransportError(Exception):
    """The remote request failed before a usable response arrived."""


class StreamEndedError(TransportError):
    """The event stream ended without a terminal response. `__cause__` is the unexpected EOF."""
