"""Go `encoding/json/v2` decoding of tool arguments, including Go's error text.

Argument errors are shown to the model, so the text matches Go byte for byte.
This mirrors jsontext's decoder state machine (the token path used to skip
unknown struct members and the value path used for raw values), the
`SyntacticError` format with JSON pointers and offsets, and the
`SemanticError` text for the Go types tools decode into.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from unreal_agent._go import (
    RUNE_ERROR,
    decode_rune,
    go_quote,
    go_quote_rune,
    is_go_space,
    is_truncated_rune,
)

__all__ = [
    "GoJSONError",
    "decode_int",
    "decode_optional_string",
    "decode_raw_object",
    "decode_string",
    "decode_string_field",
]

MAX_NESTING_DEPTH: Final = 10000
_WHITESPACE: Final = frozenset(b" \t\r\n")
_DIGITS: Final = frozenset(b"0123456789")
_HEX: Final = frozenset(b"0123456789abcdefABCDEF")
_KIND_NAMES: Final = {
    "n": "null",
    "f": "boolean",
    "t": "boolean",
    '"': "string",
    "0": "number",
    "{": "object",
    "}": "object",
    "[": "array",
    "]": "array",
}
_UNEXPECTED_EOF: Final = "unexpected EOF"
_MAX_INT64: Final = 2**63 - 1


class GoJSONError(ValueError):
    """A Go json.Unmarshal error; `str(error)` is Go's error text."""


class _EndOfInput(Exception):
    """io.EOF: no top-level value was present."""


@dataclass(slots=True)
class _Fault:
    text: str
    eof: bool = False
    mismatch: bool = False
    duplicate: bool = False
    suffix: list[str] = field(default_factory=list)


def _eof() -> _Fault:
    return _Fault(_UNEXPECTED_EOF)


@dataclass(slots=True)
class _Entry:
    is_object: bool
    length: int = 0
    active: bool = True
    seen: set[str] = field(default_factory=set)
    name: str = ""

    def need_object_name(self) -> bool:
        return self.is_object and self.length % 2 == 0

    def need_object_value(self) -> bool:
        return self.is_object and self.length % 2 == 1

    def need_implicit_comma(self, next_kind: str) -> bool:
        return not self.need_object_value() and self.length > 0 and next_kind not in "}]"


def _normalize(byte: int) -> str:
    if byte == 0x2D or 0x30 <= byte <= 0x39:
        return "0"
    character = chr(byte)
    return character if character in 'nft"{}[]' else "\0"


def _rune_count(data: bytes) -> int:
    count = index = 0
    while index < len(data):
        index += decode_rune(data, index)[1]
        count += 1
    return count


def _needs_escape(data: bytes) -> bool:
    index = 0
    while index < len(data):
        rune, size = decode_rune(data, index)
        if rune in (0x60, RUNE_ERROR) or is_go_space(rune) or not chr(rune).isprintable():
            return True
        index += size
    return False


def _invalid_text(label: str, what: bytes, where: str) -> str:
    if _rune_count(what) == 1:
        quoted = go_quote_rune(what)
    elif _needs_escape(what):
        quoted = go_quote(what)
    else:
        quoted = "`" + what.decode("utf-8") + "`"
    return f"invalid {label} {quoted} {where}".removesuffix(" ")


def _invalid_character(prefix: bytes, where: str) -> str:
    return _invalid_text("character", prefix[: decode_rune(prefix)[1]], where)


def _invalid_escape(what: bytes) -> _Fault:
    label = "surrogate pair" if len(what) > 6 else "escape sequence"
    return _Fault(_invalid_text(label, what, "in string"))


def _escape_pointer_name(name: str) -> str:
    return "/" + name.replace("~", "~0").replace("/", "~1")


def _pointer_parent(pointer: str) -> str:
    return pointer[: max(pointer.rfind("/"), 0)]


def _pointer_last_token(pointer: str) -> str:
    last = pointer[max(pointer.rfind("/"), 0) :].removeprefix("/")
    return last.replace("~1", "/").replace("~0", "~")


def _truncate_pointer(pointer: str, limit: int) -> str:
    data = pointer.encode("utf-8")
    if len(data) <= limit:
        return pointer
    head = limit // 2
    tail = len(data) - limit // 2
    slash = data.rfind(b"/", 0, head)
    if slash > 0:
        head = slash
    slash = data.find(b"/", tail)
    if slash >= 0:
        tail = slash + 1
    while head > 0 and 0x80 <= data[head] <= 0xBF:
        head -= 1
    while tail < len(data) and 0x80 <= data[tail] <= 0xBF:
        tail += 1
    middle_part = data[head:tail]
    match middle_part.count(b"/"):
        case 0:
            middle = "…"
        case 1:
            middle = "…/…"
        case _:
            middle = "…/…/…"
    if middle_part.startswith(b"/") and middle != "…":
        middle = middle.removeprefix("…")
    if middle_part.endswith(b"/") and middle != "…":
        middle = middle.removesuffix("…")
    return data[:head].decode("utf-8") + middle + data[tail:].decode("utf-8")


def _syntactic(text: str, offset: int, pointer: str, *, duplicate: bool = False) -> GoJSONError:
    message = "jsontext: " + text
    if duplicate:
        message += " " + go_quote(_pointer_last_token(pointer))
        pointer = _pointer_parent(pointer)
        offset = 0
    if pointer:
        message += " within " + go_quote(_truncate_pointer(pointer, 100))
    if offset > 0:
        message += f" after offset {offset}"
    return GoJSONError(message)


def _unquote(value: bytes) -> str:
    return json.loads(value.decode("utf-8"))


class _Decoder:
    __slots__ = ("buf", "prev_end", "prev_start", "tokens")

    def __init__(self, data: bytes) -> None:
        self.buf = data
        self.prev_start = 0
        self.prev_end = 0
        self.tokens = [_Entry(is_object=False)]

    def _skip_whitespace(self, pos: int) -> int:
        buf = self.buf
        while pos < len(buf) and buf[pos] in _WHITESPACE:
            pos += 1
        return pos

    def stack_pointer(self, where: int) -> str:
        pointer = ""
        depth = len(self.tokens)
        for index in range(1, depth):
            entry = self.tokens[index]
            array_delta = -1
            if index == depth - 1:
                if (
                    (where < 0 and entry.length == 0)
                    or (where == 0 and not entry.need_object_value())
                    or (where > 0 and entry.need_object_name())
                ):
                    return pointer
                if where > 0 and not entry.is_object:
                    array_delta = 0
            if entry.is_object:
                pointer += _escape_pointer_name(entry.name)
            else:
                pointer += f"/{entry.length + array_delta}"
        return pointer

    def wrap(self, fault: _Fault, pos: int, where: int) -> Exception:
        if fault.eof:
            return _EndOfInput()
        pointer = self.stack_pointer(where) + "".join(reversed(fault.suffix))
        text = fault.text
        if fault.mismatch:
            where_text = "at start of value"
            last = self.tokens[-1]
            if len(self.tokens) > 1 and last.length > 0:
                if last.is_object:
                    where_text = "after object value (expecting ',' or '}')"
                else:
                    where_text = "after array element (expecting ',' or ']')"
                pointer = _pointer_parent(pointer)
            text = _invalid_character(self.buf[pos:], where_text)
        return _syntactic(text, pos, pointer, duplicate=fault.duplicate)

    def need_delim(self, next_kind: str) -> str:
        last = self.tokens[-1]
        if last.need_object_value():
            return ":"
        if last.need_implicit_comma(next_kind) and len(self.tokens) > 1:
            return ","
        return ""

    def check_delim(self, delim: str, next_kind: str) -> Exception:
        where = "at start of value"
        match self.need_delim(next_kind):
            case ":":
                where = "after object name (expecting ':')"
            case ",":
                if self.tokens[-1].is_object:
                    where = "after object value (expecting ',' or '}')"
                else:
                    where = "after array element (expecting ',' or ']')"
        pos = self._skip_whitespace(self.prev_end)
        return self.wrap(_Fault(_invalid_character(self.buf[pos:], where)), pos, 0)

    def _next(self) -> tuple[int, str]:
        buf = self.buf
        pos = self._skip_whitespace(self.prev_end)
        if pos >= len(buf):
            fault = _Fault(_UNEXPECTED_EOF, eof=len(self.tokens) == 1)
            raise self.wrap(fault, pos, 0)
        delim = ""
        if buf[pos] in b":,":
            delim = chr(buf[pos])
            pos = self._skip_whitespace(pos + 1)
            if pos >= len(buf):
                error = self.wrap(_eof(), pos, 0)
                if self.need_delim('"') != delim:
                    error = self.check_delim(delim, '"')
                raise error
        next_kind = _normalize(buf[pos])
        if self.need_delim(next_kind) != delim:
            raise self.check_delim(delim, next_kind)
        return pos, next_kind

    def peek_kind(self) -> str:
        return self._next()[1]

    def _append_literal(self) -> _Fault | None:
        last = self.tokens[-1]
        if last.need_object_name():
            return _Fault("object member name must be a string")
        last.length += 1
        return None

    def _push(self, is_object: bool) -> _Fault | None:
        last = self.tokens[-1]
        if last.need_object_name():
            return _Fault("object member name must be a string")
        if len(self.tokens) - 1 == MAX_NESTING_DEPTH:
            return _Fault("exceeded max depth")
        last.length += 1
        self.tokens.append(_Entry(is_object=is_object))
        return None

    def _pop(self, is_object: bool) -> _Fault | None:
        last = self.tokens[-1]
        if is_object:
            if not last.is_object:
                return _Fault("", mismatch=True)
            if last.need_object_value():
                return _Fault("missing value after object name")
        elif last.is_object or len(self.tokens) == 1:
            return _Fault("", mismatch=True)
        self.tokens.pop()
        return None

    def _record_name(self, name: str) -> _Fault | None:
        last = self.tokens[-1]
        if not last.need_object_name():
            return None
        if last.active:
            if name in last.seen:
                return _Fault(
                    "duplicate object member name",
                    duplicate=True,
                    suffix=[_escape_pointer_name(name)],
                )
            last.seen.add(name)
        last.name = name
        return None

    def read_token(self) -> tuple[str, bytes]:
        pos, next_kind = self._next()
        start = pos
        match next_kind:
            case "n" | "f" | "t":
                literal = {"n": b"null", "f": b"false", "t": b"true"}[next_kind]
                pos, fault = self._consume_literal(pos, literal)
                if fault is not None:
                    raise self.wrap(fault, pos, +1)
                fault = self._append_literal()
            case '"':
                pos, fault = self._consume_string(pos)
                if fault is not None:
                    raise self.wrap(fault, pos, +1)
                fault = self._record_name(_unquote(self.buf[start:pos]))
                if fault is None:
                    self.tokens[-1].length += 1
            case "0":
                pos, fault = self._consume_number(pos)
                if fault is not None:
                    raise self.wrap(fault, pos, +1)
                fault = self._append_literal()
            case "{" | "[":
                fault = self._push(next_kind == "{")
                if fault is not None:
                    raise self.wrap(fault, pos, +1)
                pos += 1
            case "}" | "]":
                fault = self._pop(next_kind == "}")
                if fault is not None:
                    raise self.wrap(fault, pos, +1)
                pos += 1
            case _:
                fault = _Fault(_invalid_character(self.buf[pos:], "at start of value"))
                raise self.wrap(fault, pos, +1)
        if fault is not None:
            raise self.wrap(fault, start, +1)
        self.prev_start, self.prev_end = start, pos
        return next_kind, self.buf[start:pos]

    def read_value(self) -> bytes:
        pos, next_kind = self._next()
        start = pos
        pos, fault = self._consume_value(pos, len(self.tokens))
        if fault is not None:
            raise self.wrap(fault, pos, +1)
        match next_kind:
            case "n" | "t" | "f" | "0":
                fault = self._append_literal()
            case '"':
                fault = self._record_name(_unquote(self.buf[start:pos]))
                if fault is None:
                    self.tokens[-1].length += 1
            case "{" | "[":
                fault = self._push(next_kind == "{")
                if fault is None:
                    self.tokens.pop()
        if fault is not None:
            raise self.wrap(fault, start, +1)
        self.prev_start, self.prev_end = start, pos
        return self.buf[start:pos]

    def skip_value(self) -> None:
        if self.peek_kind() in "{[":
            depth = len(self.tokens)
            while True:
                self.read_token()
                if depth >= len(self.tokens):
                    return
        self.read_value()

    def check_eof(self) -> None:
        pos = self._skip_whitespace(self.prev_end)
        if pos < len(self.buf):
            fault = _Fault(_invalid_character(self.buf[pos:], "after top-level value"))
            raise self.wrap(fault, pos, 0)

    def _consume_literal(self, pos: int, literal: bytes) -> tuple[int, _Fault | None]:
        buf = self.buf
        for index, expected in enumerate(literal):
            if pos + index >= len(buf):
                return len(buf), _eof()
            if buf[pos + index] != expected:
                where = (
                    f"in literal {literal.decode()} (expecting {go_quote_rune(bytes([expected]))})"
                )
                return pos + index, _Fault(_invalid_character(buf[pos + index :], where))
        return pos + len(literal), None

    def _consume_string(self, pos: int) -> tuple[int, _Fault | None]:
        buf = self.buf
        end = len(buf)
        if pos >= end:
            return pos, _eof()
        if buf[pos] != 0x22:
            where = "at start of string (expecting '\"')"
            return pos, _Fault(_invalid_character(buf[pos:], where))
        n = pos + 1
        while True:
            while n < end and 0x20 <= buf[n] < 0x80 and buf[n] not in (0x22, 0x5C):
                n += 1
            if n >= end:
                return n, _eof()
            byte = buf[n]
            if byte == 0x22:
                return n + 1, None
            rune, size = decode_rune(buf, n)
            if size > 1:
                n += size
                continue
            if byte == 0x5C:
                resume = n
                if end < n + 2:
                    return resume, _eof()
                escape = buf[n + 1]
                if escape in b'/"\\bfnrt':
                    n += 2
                    continue
                if escape != 0x75:
                    return n, _invalid_escape(buf[n : n + 2])
                if end < n + 6:
                    if _has_escaped_utf16_prefix(buf[n:], lower=False):
                        return resume, _eof()
                    return n, _invalid_escape(buf[n:])
                high = _parse_hex(buf[n + 2 : n + 6])
                if high is None:
                    return n, _invalid_escape(buf[n : n + 6])
                n += 6
                if 0xD800 <= high < 0xE000:
                    if end < n + 6:
                        if _has_escaped_utf16_prefix(buf[n:], lower=True):
                            return resume, _eof()
                        return n - 6, _invalid_escape(buf[n - 6 :])
                    low = _parse_hex(buf[n + 2 : n + 6])
                    if (
                        buf[n] != 0x5C
                        or buf[n + 1] != 0x75
                        or low is None
                        or not (0xD800 <= high < 0xDC00 and 0xDC00 <= low < 0xE000)
                    ):
                        return n - 6, _invalid_escape(buf[n - 6 : n + 6])
                    n += 6
                continue
            if rune == RUNE_ERROR and byte >= 0x80:
                if is_truncated_rune(buf, n):
                    return n, _eof()
                return n, _Fault("invalid UTF-8")
            where = "in string (expecting non-control character)"
            return n, _Fault(_invalid_character(buf[n:], where))

    def _consume_number(self, pos: int) -> tuple[int, _Fault | None]:
        buf = self.buf
        end = len(buf)
        n = pos
        if buf[n] == 0x2D:
            n += 1
        if n >= end:
            return pos, _eof()
        if buf[n] == 0x30:
            n += 1
        elif 0x31 <= buf[n] <= 0x39:
            n += 1
            while n < end and buf[n] in _DIGITS:
                n += 1
        else:
            return n, _Fault(_invalid_character(buf[n:], "in number (expecting digit)"))
        if n < end and buf[n] == 0x2E:
            n += 1
            if n >= end:
                return pos, _eof()
            if buf[n] not in _DIGITS:
                return n, _Fault(_invalid_character(buf[n:], "in number (expecting digit)"))
            while n < end and buf[n] in _DIGITS:
                n += 1
        if n < end and buf[n] in b"eE":
            n += 1
            if n < end and buf[n] in b"+-":
                n += 1
            if n >= end:
                return pos, _eof()
            if buf[n] not in _DIGITS:
                return n, _Fault(_invalid_character(buf[n:], "in number (expecting digit)"))
            while n < end and buf[n] in _DIGITS:
                n += 1
        return n, None

    def _consume_value(self, pos: int, depth: int) -> tuple[int, _Fault | None]:
        """Consume one value without recursion, so nesting up to Go's limit works."""
        buf = self.buf
        end = len(buf)
        frames: list[_Frame] = []

        def fail(at: int, fault: _Fault, *, within_innermost: bool) -> tuple[int, _Fault]:
            tokens = [frame.token() for frame in frames]
            if frames and not within_innermost:
                tokens.pop()
            fault.suffix.extend(reversed(tokens))
            return at, fault

        state = _VALUE
        while True:
            if state == _VALUE:
                kind = _normalize(buf[pos])
                fault: _Fault | None
                match kind:
                    case "n" | "f" | "t":
                        literal = {"n": b"null", "f": b"false", "t": b"true"}[kind]
                        pos, fault = self._consume_literal(pos, literal)
                    case '"':
                        pos, fault = self._consume_string(pos)
                    case "0":
                        pos, fault = self._consume_number(pos)
                    case "{" | "[":
                        if depth == MAX_NESTING_DEPTH + 1:
                            return fail(pos, _Fault("exceeded max depth"), within_innermost=True)
                        pos = self._skip_whitespace(pos + 1)
                        if pos >= end:
                            return fail(pos, _eof(), within_innermost=True)
                        if buf[pos] == (0x7D if kind == "{" else 0x5D):
                            pos += 1
                            state = _AFTER_VALUE
                            continue
                        depth += 1
                        frames.append(_Frame(is_object=kind == "{"))
                        state = _NAME if kind == "{" else _VALUE
                        continue
                    case _:
                        last = self.tokens[-1]
                        if (last.is_object and kind == "]") or (not last.is_object and kind == "}"):
                            fault = _Fault("", mismatch=True)
                        else:
                            fault = _Fault(_invalid_character(buf[pos:], "at start of value"))
                if fault is not None:
                    return fail(pos, fault, within_innermost=True)
                state = _AFTER_VALUE
            elif state == _AFTER_VALUE:
                if not frames:
                    return pos, None
                frame = frames[-1]
                pos = self._skip_whitespace(pos)
                if pos >= end:
                    return fail(pos, _eof(), within_innermost=False)
                if buf[pos] == 0x2C:
                    pos += 1
                    if frame.is_object:
                        state = _NAME
                        continue
                    frame.index += 1
                    pos = self._skip_whitespace(pos)
                    if pos >= end:
                        return fail(pos, _eof(), within_innermost=False)
                    state = _VALUE
                elif buf[pos] == (0x7D if frame.is_object else 0x5D):
                    pos += 1
                    frames.pop()
                    depth -= 1
                else:
                    if frame.is_object:
                        where = "after object value (expecting ',' or '}')"
                    else:
                        where = "after array element (expecting ',' or ']')"
                    fault = _Fault(_invalid_character(buf[pos:], where))
                    return fail(pos, fault, within_innermost=False)
            else:
                frame = frames[-1]
                pos = self._skip_whitespace(pos)
                if pos >= end:
                    return fail(pos, _eof(), within_innermost=False)
                start = pos
                pos, fault = self._consume_string(pos)
                if fault is not None:
                    return fail(pos, fault, within_innermost=False)
                name = _unquote(buf[start:pos])
                frame.name = name
                if name in frame.seen:
                    fault = _Fault("duplicate object member name", duplicate=True)
                    return fail(start, fault, within_innermost=True)
                frame.seen.add(name)
                pos = self._skip_whitespace(pos)
                if pos >= end:
                    return fail(pos, _eof(), within_innermost=True)
                if buf[pos] != 0x3A:
                    where = "after object name (expecting ':')"
                    fault = _Fault(_invalid_character(buf[pos:], where))
                    return fail(pos, fault, within_innermost=True)
                pos = self._skip_whitespace(pos + 1)
                if pos >= end:
                    return fail(pos, _eof(), within_innermost=True)
                state = _VALUE


_VALUE: Final = 0
_AFTER_VALUE: Final = 1
_NAME: Final = 2


@dataclass(slots=True)
class _Frame:
    is_object: bool
    index: int = 0
    name: str = ""
    seen: set[str] = field(default_factory=set)

    def token(self) -> str:
        return _escape_pointer_name(self.name) if self.is_object else f"/{self.index}"


def _has_escaped_utf16_prefix(data: bytes, *, lower: bool) -> bool:
    for index, byte in enumerate(data):
        if index == 0 and byte != 0x5C:
            return False
        if index == 1 and byte != 0x75:
            return False
        if index == 2 and lower and byte not in b"dD":
            return False
        if index == 3 and lower and byte not in b"cdefCDEF":
            return False
        if 2 <= index < 6 and byte not in _HEX:
            return False
    return True


def _parse_hex(data: bytes) -> int | None:
    if len(data) != 4 or any(byte not in _HEX for byte in data):
        return None
    return int(data, 16)


def _semantic(
    decoder: _Decoder, go_type: str, *, value: bytes = b"", error: str = ""
) -> GoJSONError:
    """SemanticError for the previous token or value (newUnmarshalErrorAfter)."""
    previous = decoder.buf[decoder.prev_start : decoder.prev_end]
    message = "json: cannot unmarshal"
    kind = _KIND_NAMES.get(_normalize(previous[0])) if previous else None
    if kind is not None:
        message += " JSON " + kind
    if 0 < len(value) < 100:
        message += " " + value.decode("utf-8")
    message += " into Go " + go_type
    pointer = decoder.stack_pointer(-1)
    if pointer:
        message += " within " + go_quote(_truncate_pointer(pointer, 100))
    elif decoder.prev_start > 0:
        message += f" after offset {decoder.prev_start}"
    if error:
        message += ": " + error
    return GoJSONError(message)


def _duplicate(decoder: _Decoder) -> GoJSONError:
    return _syntactic("duplicate object member name", 0, decoder.stack_pointer(-1), duplicate=True)


def _unmarshal[T](data: bytes, read: Callable[[_Decoder], T]) -> T:
    decoder = _Decoder(data)
    try:
        result = read(decoder)
    except _EndOfInput:
        raise _syntactic(_UNEXPECTED_EOF, len(data), "") from None
    decoder.check_eof()
    return result


def _begin_object(decoder: _Decoder, go_type: str) -> bool:
    """Read the opening token of a map or struct. False means JSON null."""
    kind, _ = decoder.read_token()
    if kind == "n":
        return False
    if kind != "{":
        raise _semantic(decoder, go_type)
    decoder.tokens[-1].active = False
    return True


def decode_raw_object(data: bytes) -> dict[str, bytes] | None:
    """Unmarshal into Go `map[string]jsontext.Value`; None is a nil map."""

    def read(decoder: _Decoder) -> dict[str, bytes] | None:
        if not _begin_object(decoder, "map[string]jsontext.Value"):
            return None
        members: dict[str, bytes] = {}
        while decoder.peek_kind() != "}":
            name = _unquote(decoder.read_value())
            if name in members:
                raise _duplicate(decoder)
            members[name] = decoder.read_value()
        decoder.read_token()
        return members

    return _unmarshal(data, read)


def decode_string_field(data: bytes, field_name: str, go_type: str) -> str:
    """Unmarshal into a Go struct whose only field is the string `field_name`."""

    def read(decoder: _Decoder) -> str:
        if not _begin_object(decoder, go_type):
            return ""
        entry = decoder.tokens[-1]
        value = ""
        found = False
        while decoder.peek_kind() != "}":
            name = _unquote(decoder.read_value())
            if name != field_name:
                if name in entry.seen:
                    raise _duplicate(decoder)
                entry.seen.add(name)
                decoder.skip_value()
                continue
            if found:
                raise _duplicate(decoder)
            found = True
            value = _read_string(decoder) or ""
        decoder.read_token()
        return value

    return _unmarshal(data, read)


def _read_string(decoder: _Decoder) -> str | None:
    raw = decoder.read_value()
    match chr(raw[0]):
        case "n":
            return None
        case '"':
            return _unquote(raw)
        case _:
            raise _semantic(decoder, "string")


def decode_string(data: bytes) -> str:
    """Unmarshal into a Go `string`; JSON null leaves it empty."""
    return _unmarshal(data, lambda decoder: _read_string(decoder) or "")


def decode_optional_string(data: bytes) -> str | None:
    """Unmarshal into a Go `*string`; JSON null is nil."""
    return _unmarshal(data, _read_string)


def decode_int(data: bytes) -> int:
    """Unmarshal into a 64-bit Go `int`; JSON null is zero."""

    def read(decoder: _Decoder) -> int:
        raw = decoder.read_value()
        kind = _normalize(raw[0])
        if kind == "n":
            return 0
        if kind != "0":
            raise _semantic(decoder, "int")
        negative = raw[0] == 0x2D
        digits = raw[1:] if negative else raw
        if (
            not digits
            or any(byte not in _DIGITS for byte in digits)
            or (digits[0] == 0x30 and digits != b"0")
        ):
            raise _semantic(decoder, "int", value=raw, error="invalid syntax")
        magnitude = int(digits)
        limit = _MAX_INT64 + 1 if negative else _MAX_INT64
        if magnitude > limit:
            raise _semantic(decoder, "int", value=raw, error="value out of range")
        return -magnitude if negative else magnitude

    return _unmarshal(data, read)
