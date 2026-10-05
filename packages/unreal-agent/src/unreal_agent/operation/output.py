"""Bounded output text with Go UTF-8 semantics.

Text is measured in runes the way Go does: every byte of an invalid UTF-8
sequence counts as one rune and is rendered as U+FFFD. Decoding with
`surrogateescape` reproduces that split exactly, because each undecodable
byte becomes one lone surrogate in U+DC80..U+DCFF, a range valid UTF-8 never
produces.
"""

import re
from typing import Final

__all__ = ["DEFAULT_MAX_OUTPUT_LENGTH", "MAX_OUTPUT_LENGTH", "bound_output", "utf8_len"]

DEFAULT_MAX_OUTPUT_LENGTH: Final = 40_000
MAX_OUTPUT_LENGTH: Final = 1_000_000

_INVALID_BYTE: Final = re.compile("[\udc80-\udcff]")


def utf8_len(text: str) -> int:
    """Return the byte length of `text` as a Go string."""
    return len(_to_bytes(text))


def bound_output(text: str | bytes, limit: int) -> tuple[str, bool]:
    """Keep at most `limit` runes: a head of limit/2 and a tail of the rest, around a marker."""
    data = _to_bytes(text)
    return _bound_output(data, b"", len(data), limit, "")


def _bound_output(
    head: bytes, tail: bytes, full_size: int, limit: int, path: str
) -> tuple[str, bool]:
    limit = max(0, limit)
    if not tail:
        if full_size != len(head):
            raise ValueError("bound output: empty tail requires the complete output in head")
        decoded = _decode(head)
        if len(decoded) <= limit:
            return _sanitize(decoded), False
        tail = head

    head_text, head_size = _output_part(head, limit // 2, tail=False)
    tail_text, tail_size = _output_part(tail, limit - limit // 2, tail=True)
    marker = f"...{full_size - (head_size + tail_size)} bytes truncated"
    if path:
        marker += "; complete output in " + path
    return head_text + marker + "..." + tail_text, True


def _output_part(data: bytes, limit: int, *, tail: bool) -> tuple[str, int]:
    decoded = _decode(data)
    if limit <= 0:
        part = ""
    elif tail:
        part = decoded[-limit:]
    else:
        part = decoded[:limit]
    return _sanitize(part), len(part.encode("utf-8", "surrogateescape"))


def _decode(data: bytes) -> str:
    return data.decode("utf-8", "surrogateescape")


def _sanitize(text: str) -> str:
    return _INVALID_BYTE.sub("�", text)


def _to_bytes(text: str | bytes) -> bytes:
    if isinstance(text, bytes):
        return text
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        try:
            return text.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError:
            return text.encode("utf-8", "surrogatepass")
