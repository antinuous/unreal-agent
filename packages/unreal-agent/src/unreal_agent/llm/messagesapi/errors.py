__all__ = ["APIError", "StreamEndedError", "TransportError"]


class APIError(Exception):
    """An error reported by the provider, over HTTP or inside the event stream."""

    def __init__(
        self, status_code: int = 0, type: str = "", message: str = "", request_id: str = ""
    ) -> None:
        super().__init__(status_code, type, message, request_id)
        self.status_code = status_code
        self.type = type
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        if self.type:
            return f"messages API error {self.type}: {self.message}"
        return f"messages API request failed with status {self.status_code}: {self.message}"

    def __repr__(self) -> str:
        return (
            f"APIError(status_code={self.status_code!r}, type={self.type!r}, "
            f"message={self.message!r}, request_id={self.request_id!r})"
        )


class TransportError(Exception):
    """The remote request failed before a usable response arrived."""


class StreamEndedError(TransportError):
    """The event stream ended without message_stop. `__cause__` is the unexpected EOF."""
