from unreal_agent.operation import DEFAULT_MAX_OUTPUT_LENGTH, MAX_OUTPUT_LENGTH, bound_output
from unreal_agent.tool._jsonv2 import GoJSONError, decode_int
from unreal_agent.tool.tool import CallStatus

__all__ = ["error_status", "parse_max_output_length"]


def error_status(message: str, limit: int) -> CallStatus:
    """A failed status whose error is bounded to `limit` characters (default when <= 0)."""
    if limit <= 0:
        limit = DEFAULT_MAX_OUTPUT_LENGTH
    text, truncated = bound_output(message, limit)
    return CallStatus(Error=text, ErrorTruncated=truncated)


def parse_max_output_length(encoded: bytes) -> int:
    """Parse a raw `max_output_length` argument. Raises ValueError with Go's message."""
    if len(encoded) == 0:
        return DEFAULT_MAX_OUTPUT_LENGTH
    try:
        limit = decode_int(encoded)
    except GoJSONError as error:
        raise ValueError(f"decode max_output_length: {error}") from error
    if limit <= 0:
        raise ValueError("max_output_length must be a positive integer")
    if limit > MAX_OUTPUT_LENGTH:
        raise ValueError(f"max_output_length must not exceed {MAX_OUTPUT_LENGTH}")
    return limit
