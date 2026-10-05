"""Go string semantics needed for byte-identical tool and prompt text.

Python strings that came from Go-style byte strings carry invalid UTF-8 as
`surrogateescape` code points. `go_bytes` restores the original bytes so
quoting, trimming, and replacement match Go's rune decoding.
"""

import re
from typing import Final

from unreal_agent._filepath import go_clean, go_join

__all__ = [
    "decode_rune",
    "go_bytes",
    "go_clean",
    "go_join",
    "go_quote",
    "go_quote_rune",
    "go_trim_space",
    "is_truncated_rune",
    "to_valid_utf8",
]

RUNE_ERROR: Final = 0xFFFD

_SPACE: Final = (
    "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
_SPACE_SET: Final = frozenset(ord(character) for character in _SPACE)
_ESCAPED_BYTE: Final = re.compile("[\udc80-\udcff]+")
_ESCAPES: Final = {7: "\\a", 8: "\\b", 12: "\\f", 10: "\\n", 13: "\\r", 9: "\\t", 11: "\\v"}


def go_bytes(text: str) -> bytes:
    """Return the Go byte string a Python string stands for."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        try:
            return text.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError:
            return text.encode("utf-8", "surrogatepass")


def go_trim_space(text: str) -> str:
    """strings.TrimSpace: Python's str.strip also removes U+001C..U+001F."""
    return text.strip(_SPACE)


def is_go_space(rune: int) -> bool:
    return rune in _SPACE_SET


def to_valid_utf8(data: bytes) -> str:
    """strings.ToValidUTF8 with U+FFFD: each run of invalid bytes becomes one U+FFFD."""
    return _ESCAPED_BYTE.sub("\ufffd", data.decode("utf-8", "surrogateescape"))


def _continuation_range(first: int) -> tuple[int, int, int] | None:
    """Return (size, low, high) for the second byte of a sequence led by `first`."""
    if 0xC2 <= first <= 0xDF:
        return 2, 0x80, 0xBF
    if first == 0xE0:
        return 3, 0xA0, 0xBF
    if first == 0xED:
        return 3, 0x80, 0x9F
    if 0xE1 <= first <= 0xEF:
        return 3, 0x80, 0xBF
    if first == 0xF0:
        return 4, 0x90, 0xBF
    if 0xF1 <= first <= 0xF3:
        return 4, 0x80, 0xBF
    if first == 0xF4:
        return 4, 0x80, 0x8F
    return None


def decode_rune(data: bytes, index: int = 0) -> tuple[int, int]:
    """utf8.DecodeRune: (U+FFFD, 1) for an invalid byte, (U+FFFD, 0) at the end."""
    if index >= len(data):
        return RUNE_ERROR, 0
    first = data[index]
    if first < 0x80:
        return first, 1
    sequence = _continuation_range(first)
    if sequence is None:
        return RUNE_ERROR, 1
    size, low, high = sequence
    if index + size > len(data) or not low <= data[index + 1] <= high:
        return RUNE_ERROR, 1
    if any(not 0x80 <= byte <= 0xBF for byte in data[index + 2 : index + size]):
        return RUNE_ERROR, 1
    return ord(data[index : index + size].decode("utf-8")), size


def is_truncated_rune(data: bytes, index: int) -> bool:
    """Report whether data[index:] is a valid but incomplete UTF-8 prefix (not utf8.FullRune)."""
    if index >= len(data):
        return True
    sequence = _continuation_range(data[index])
    if sequence is None:
        return False
    size, low, high = sequence
    available = len(data) - index
    if available >= size:
        return False
    if available > 1 and not low <= data[index + 1] <= high:
        return False
    return all(0x80 <= byte <= 0xBF for byte in data[index + 2 :])


def _escaped_rune(rune: int, quote: str) -> str:
    if rune == ord(quote) or rune == 0x5C:
        return "\\" + chr(rune)
    if 0xD800 <= rune <= 0xDFFF or rune > 0x10FFFF:
        return "\\ufffd"
    if chr(rune).isprintable():
        return chr(rune)
    if rune in _ESCAPES:
        return _ESCAPES[rune]
    if rune < 0x20 or rune == 0x7F:
        return f"\\x{rune:02x}"
    if rune < 0x10000:
        return f"\\u{rune:04x}"
    return f"\\U{rune:08x}"


def go_quote(value: str | bytes) -> str:
    """strconv.Quote, the `%q` verb for strings."""
    data = go_bytes(value) if isinstance(value, str) else value
    out = ['"']
    index = 0
    while index < len(data):
        rune, size = decode_rune(data, index)
        if size == 1 and rune == RUNE_ERROR:
            out.append(f"\\x{data[index]:02x}")
        else:
            out.append(_escaped_rune(rune, '"'))
        index += size
    out.append('"')
    return "".join(out)


def go_quote_rune(data: bytes) -> str:
    """jsonwire.QuoteRune: quote the first rune of `data`, or its first invalid byte."""
    rune, size = decode_rune(data)
    if rune == RUNE_ERROR and size == 1:
        return f"'\\x{data[0]:x}'"
    return "'" + _escaped_rune(rune, "'") + "'"
