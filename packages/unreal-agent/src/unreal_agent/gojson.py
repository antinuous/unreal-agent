"""JSON encoding compatible with the Go `encoding/json/v2` wire format.

Persisted sessions, operations, and runner output keep the format produced by
the original Go implementation: struct fields in declaration order, PascalCase
names, `omitzero`/`omitempty` semantics, base64 bytes, RFC 3339 times with
nanosecond precision, and raw JSON values passed through without reparsing.
Decoding is strict in the same ways as Go: duplicate object members, invalid
UTF-8, unpaired surrogate escapes, non-finite numbers, and nesting deeper than
Go's limit of 10000 arrays and objects are rejected. Encoding enforces the same
depth limit. JSON null decodes into a non-optional field as its zero value.

Go's nil slices and maps are `None`: an `OmitZero` list or dict field is typed
`list[T] | None` and omitted only when it is `None`, so an empty but non-nil
value still encodes as `[]` or `{}`.
"""

import base64
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from functools import cache
from types import UnionType
from typing import Annotated, Any, Final, Self, Union, get_args, get_origin, get_type_hints

import msgspec

__all__ = [
    "MAX_DEPTH",
    "Int64",
    "JSONError",
    "OmitEmpty",
    "OmitZero",
    "Raw",
    "Struct",
    "Timestamp",
    "Uint32",
    "compact",
    "decode",
    "encode",
    "format_float",
    "is_valid_json",
    "validate",
]


class JSONError(ValueError):
    pass


class _Marker:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


OmitZero: Final = _Marker("OmitZero")
OmitEmpty: Final = _Marker("OmitEmpty")

Uint32 = Annotated[int, msgspec.Meta(ge=0, le=2**32 - 1)]
Int64 = Annotated[int, msgspec.Meta(ge=-(2**63), le=2**63 - 1)]

Raw = msgspec.Raw


class Struct(msgspec.Struct, kw_only=True):
    """Base for Go-compatible wire structs.

    Every field must have a default so decoding tolerates absent members the
    way Go does. Annotate a field with `OmitZero` or `OmitEmpty` to mirror the
    corresponding Go struct tag.

    Define `__gojson_validate__(self) -> None` to reject a value before it is
    encoded, like a Go MarshalJSON method that validates first.
    """


_TIME_RE: Final = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(?:(Z)|([+-])(\d{2}):(\d{2}))\Z",
    re.ASCII,
)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_GO_ZERO_UNIX_NS: Final = -62135596800 * 10**9
# datetime starts at year 1; year 0 is handled by shifting one 400-year
# Gregorian cycle, which preserves weekdays and leap years.
_CYCLE_YEARS: Final = 400
_CYCLE_SECONDS: Final = 146097 * 86400
_MIN_LOCAL_SECONDS: Final = _GO_ZERO_UNIX_NS // 10**9 - 366 * 86400
_MAX_LOCAL_SECONDS: Final = 253402300800


class Timestamp:
    """An instant with nanosecond precision and a fixed UTC offset, like Go's time.Time."""

    __slots__ = ("offset_seconds", "unix_ns")

    unix_ns: int
    offset_seconds: int

    def __init__(self, unix_ns: int = _GO_ZERO_UNIX_NS, offset_seconds: int = 0) -> None:
        object.__setattr__(self, "unix_ns", unix_ns)
        object.__setattr__(self, "offset_seconds", offset_seconds)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Timestamp is immutable")

    def __repr__(self) -> str:
        return f"Timestamp({self.format()!r})"

    @classmethod
    def now(cls) -> Self:
        import time

        return cls(time.time_ns())

    @classmethod
    def from_datetime(cls, value: datetime) -> Self:
        if value.tzinfo is None:
            raise ValueError("timestamp requires an aware datetime")
        offset = value.utcoffset() or timedelta()
        delta = value - _EPOCH
        unix_ns = (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000
        return cls(unix_ns, int(offset.total_seconds()))

    @classmethod
    def parse(cls, text: str) -> Self:
        match = _TIME_RE.fullmatch(text)
        if match is None:
            raise JSONError(f"invalid RFC 3339 timestamp {text!r}")
        year, month, day, hour, minute, second = (int(match.group(i)) for i in range(1, 7))
        fraction = match.group(7) or ""
        offset = 0
        if match.group(9):
            offset_hours, offset_minutes = int(match.group(10)), int(match.group(11))
            if offset_hours > 23 or offset_minutes > 59:
                raise JSONError(f"invalid RFC 3339 timestamp {text!r}")
            offset = offset_hours * 3600 + offset_minutes * 60
            if match.group(9) == "-":
                offset = -offset
        shift = _CYCLE_YEARS if year == 0 else 0
        try:
            base = datetime(
                year + shift,
                month,
                day,
                hour,
                minute,
                second,
                tzinfo=timezone(timedelta(seconds=offset)),
            )
        except ValueError as error:
            raise JSONError(f"invalid RFC 3339 timestamp {text!r}") from error
        nanos = int(fraction[:9].ljust(9, "0")) if fraction else 0
        unix_ns = cls.from_datetime(base).unix_ns - (_CYCLE_SECONDS if shift else 0) * 10**9
        return cls(unix_ns + nanos, offset)

    def is_zero(self) -> bool:
        return self.unix_ns == _GO_ZERO_UNIX_NS

    def format(self) -> str:
        """Format like Go's time.RFC3339Nano."""
        seconds, nanos = divmod(self.unix_ns, 10**9)
        local_seconds = seconds + self.offset_seconds
        shift = _CYCLE_YEARS if local_seconds < _GO_ZERO_UNIX_NS // 10**9 else 0
        local = _EPOCH + timedelta(seconds=local_seconds + (_CYCLE_SECONDS if shift else 0))
        text = local.strftime("%Y-%m-%dT%H:%M:%S")
        text = f"{local.year - shift:04d}{text[text.index('-') :]}"
        if nanos:
            text += "." + f"{nanos:09d}".rstrip("0")
        if self.offset_seconds == 0:
            return text + "Z"
        sign = "-" if self.offset_seconds < 0 else "+"
        hours, minutes = divmod(abs(self.offset_seconds) // 60, 60)
        return f"{text}{sign}{hours:02d}:{minutes:02d}"

    def check_year_range(self) -> None:
        """Raise JSONError for a local year Go's MarshalJSON cannot encode."""
        local_seconds = self.unix_ns // 10**9 + self.offset_seconds
        if not _MIN_LOCAL_SECONDS <= local_seconds < _MAX_LOCAL_SECONDS:
            raise JSONError("year outside of range [0,9999]")

    def __lt__(self, other: Timestamp) -> bool:
        return self.unix_ns < other.unix_ns

    def __le__(self, other: Timestamp) -> bool:
        return self.unix_ns <= other.unix_ns

    def __gt__(self, other: Timestamp) -> bool:
        return self.unix_ns > other.unix_ns

    def __ge__(self, other: Timestamp) -> bool:
        return self.unix_ns >= other.unix_ns

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timestamp) and self.unix_ns == other.unix_ns

    def __hash__(self) -> int:
        return hash(self.unix_ns)

    def __str__(self) -> str:
        return self.format()


def format_float(value: float) -> str:
    """Format a float64 the way Go's JSON encoder does (ES6 number formatting)."""
    if not math.isfinite(value):
        raise JSONError(f"unsupported value: {value!r}")
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, frac = mantissa.partition(".")
    digits = (whole + frac).lstrip("0")
    point = len(whole) + (int(exponent) if exponent else 0) - (len(whole + frac) - len(digits))
    digits = digits.rstrip("0")
    count = len(digits)
    if count <= point <= 21:
        return sign + digits + "0" * (point - count)
    if 0 < point <= 21:
        return f"{sign}{digits[:point]}.{digits[point:]}"
    if -6 < point <= 0:
        return f"{sign}0.{'0' * -point}{digits}"
    exp = point - 1
    body = digits[0] + ("." + digits[1:] if count > 1 else "")
    return f"{sign}{body}e{'+' if exp >= 0 else '-'}{abs(exp)}"


def _reject_constant(name: str) -> Any:
    raise JSONError(f"invalid JSON literal {name}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JSONError(f"duplicate object member name {json.dumps(key, ensure_ascii=False)}")
        result[key] = value
    return result


def _check_strings(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if not item.isascii():
                try:
                    item.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise JSONError("invalid surrogate pair in string") from error
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


MAX_DEPTH: Final = 10000
"""Go's `jsontext` limit on nested arrays and objects."""

_DEPTH_TOKEN: Final = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')
_DEPTH_ERROR: Final = "exceeded max depth"


def _check_depth(text: str) -> None:
    if text.count("[") + text.count("{") <= MAX_DEPTH:
        return
    depth = 0
    for match in _DEPTH_TOKEN.finditer(text):
        token = match.group()
        if token in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise JSONError(_DEPTH_ERROR)
        elif token in "]}":
            depth -= 1


def validate(data: bytes | bytearray | memoryview | str) -> Any:
    """Parse JSON with Go v2 strictness and return the generic value."""
    if isinstance(data, str):
        text = data
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise JSONError("invalid UTF-8") from error
    else:
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError as error:
            raise JSONError("invalid UTF-8") from error
    _check_depth(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
            parse_float=str,
            parse_int=str,
        )
    except json.JSONDecodeError as error:
        raise JSONError(str(error)) from error
    except RecursionError as error:
        raise JSONError(_DEPTH_ERROR) from error
    _check_strings(value)
    return value


def is_valid_json(data: bytes | bytearray | memoryview | str) -> bool:
    try:
        validate(data)
    except JSONError:
        return False
    return True


def _dec_hook(kind: type, value: Any) -> Any:
    if kind is Timestamp:
        if value is None:
            return Timestamp()
        if not isinstance(value, str):
            raise msgspec.ValidationError("expected an RFC 3339 string")
        return Timestamp.parse(value)
    raise NotImplementedError(kind)


@cache
def _decoder[T](kind: type[T]) -> msgspec.json.Decoder[T]:
    return msgspec.json.Decoder(kind, dec_hook=_dec_hook)


def decode[T](data: bytes | bytearray | memoryview | str, kind: type[T]) -> T:
    """Decode JSON into `kind` after Go-equivalent strictness checks."""
    validate(data)
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        try:
            return _decoder(kind).decode(raw)
        except msgspec.ValidationError:
            if b"null" not in raw:
                raise
            zeroed = _zero_nulls(raw, kind)
            if zeroed is raw:
                raise
            return _decoder(kind).decode(zeroed)
    except msgspec.DecodeError as error:
        raise JSONError(str(error)) from error
    except RecursionError as error:
        raise JSONError(_DEPTH_ERROR) from error


# Go v2 decodes JSON null into a non-pointer value as its zero value, which
# msgspec rejects. After a failed decode, nulls at positions whose type cannot
# hold None are rewritten to the zero literal of that type and decoding is
# retried. Raw values are never rewritten, so they keep their bytes.

_STRING: Final = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"')
_SCALAR: Final = re.compile(rb"[^,\]}\s]*")
_CONTAINER_TOKEN: Final = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')
_ZERO_TIME: Final = b'"0001-01-01T00:00:00Z"'

_inspect = msgspec.inspect


@cache
def _type_info(kind: Any) -> _inspect.Type:
    return _inspect.type_info(kind)


def _zero_literal(info: _inspect.Type) -> bytes | None:
    match info:
        case _inspect.StrType() | _inspect.BytesType():
            return b'""'
        case _inspect.IntType() | _inspect.FloatType():
            return b"0"
        case _inspect.BoolType():
            return b"false"
        case _inspect.ListType() | _inspect.VarTupleType() | _inspect.SetType():
            return b"[]"
        case _inspect.DictType():
            return b"{}"
        case _inspect.StructType():
            return b"{}"
        case _inspect.EnumType(cls=cls):
            values = {member.value for member in cls}
            if "" in values:
                return b'""'
            if 0 in values:
                return b"0"
            return None
        case _inspect.CustomType(cls=cls) if cls is Timestamp:
            return _ZERO_TIME
        case _:
            return None


def _non_null_type(info: _inspect.Type, first: int) -> _inspect.Type | None:
    if not isinstance(info, _inspect.UnionType):
        return info
    if first == 0x7B:
        wanted: tuple[type, ...] = (_inspect.StructType, _inspect.DictType)
    elif first == 0x5B:
        wanted = (_inspect.ListType, _inspect.VarTupleType, _inspect.SetType)
    else:
        return None
    candidates = [member for member in info.types if isinstance(member, wanted)]
    return candidates[0] if len(candidates) == 1 else None


class _NullZeroer:
    __slots__ = ("edits", "source")

    def __init__(self, source: bytes) -> None:
        self.source = source
        self.edits: list[tuple[int, bytes]] = []

    def ws(self, index: int) -> int:
        source = self.source
        while source[index] in _WS:
            index += 1
        return index

    def skip(self, index: int) -> int:
        source = self.source
        first = source[index]
        if first == 0x22:
            match = _STRING.match(source, index)
            assert match is not None
            return match.end()
        if first not in (0x5B, 0x7B):
            match = _SCALAR.match(source, index)
            assert match is not None
            return match.end()
        depth = 0
        for match in _CONTAINER_TOKEN.finditer(source, index):
            token = match.group()
            if token in (b"[", b"{"):
                depth += 1
            elif token in (b"]", b"}"):
                depth -= 1
                if depth == 0:
                    return match.end()
        raise AssertionError("unterminated JSON value")

    def walk(self, index: int, info: _inspect.Type) -> int:
        source = self.source
        first = source[index]
        if first == 0x6E:
            if not isinstance(info, _inspect.UnionType | _inspect.NoneType):
                literal = _zero_literal(info)
                if literal is not None:
                    self.edits.append((index, literal))
            return index + 4
        target = _non_null_type(info, first)
        if first == 0x7B and isinstance(target, _inspect.StructType):
            fields = {field.encode_name: field.type for field in target.fields}
            return self.members(index, fields.get, None)
        if first == 0x7B and isinstance(target, _inspect.DictType):
            value_type = target.value_type
            return self.members(index, None, value_type)
        if first == 0x5B and isinstance(
            target, _inspect.ListType | _inspect.VarTupleType | _inspect.SetType
        ):
            return self.elements(index, target.item_type)
        return self.skip(index)

    def members(
        self,
        index: int,
        lookup: Any,
        value_type: _inspect.Type | None,
    ) -> int:
        source = self.source
        index = self.ws(index + 1)
        if source[index] == 0x7D:
            return index + 1
        while True:
            key_end = self.skip(index)
            key = source[index:key_end]
            index = self.ws(self.ws(key_end) + 1)
            info = value_type
            if lookup is not None:
                name = json.loads(key) if b"\\" in key else key[1:-1].decode("utf-8")
                info = lookup(name)
            index = self.ws(self.skip(index) if info is None else self.walk(index, info))
            if source[index] == 0x7D:
                return index + 1
            index = self.ws(index + 1)

    def elements(self, index: int, item_type: _inspect.Type) -> int:
        source = self.source
        index = self.ws(index + 1)
        if source[index] == 0x5D:
            return index + 1
        while True:
            index = self.ws(self.walk(index, item_type))
            if source[index] == 0x5D:
                return index + 1
            index = self.ws(index + 1)


def _zero_nulls(source: bytes, kind: Any) -> bytes:
    zeroer = _NullZeroer(source)
    zeroer.walk(zeroer.ws(0), _type_info(kind))
    if not zeroer.edits:
        return source
    out = bytearray()
    start = 0
    for index, literal in zeroer.edits:
        out += source[start:index]
        out += literal
        start = index + 4
    out += source[start:]
    return bytes(out)


_WS: Final = frozenset(b" \t\r\n")


def compact(data: bytes | bytearray | memoryview) -> bytes:
    """Format a valid raw JSON value the way Go's encoder writes a `jsontext.Value`.

    Insignificant whitespace is removed and strings containing escapes are
    rewritten with minimal escaping. Numbers and member order are unchanged.
    """
    source = bytes(data)
    if b"\\" not in source and not any(byte in _WS for byte in source):
        return source
    out = bytearray()
    index = 0
    end = len(source)
    while index < end:
        byte = source[index]
        if byte == 0x22:
            close = index + 1
            escaped = False
            while source[close] != 0x22:
                if source[close] == 0x5C:
                    escaped = True
                    close += 2
                else:
                    close += 1
            token = source[index : close + 1]
            out += _encode_str(json.loads(token)) if escaped else token
            index = close + 1
            continue
        if byte not in _WS:
            out.append(byte)
        index += 1
    return bytes(out)


@dataclass(frozen=True, slots=True)
class _Field:
    attr: str
    name: str
    omit_zero: bool
    omit_empty: bool
    nilable: bool


def _is_nilable(hint: Any) -> bool:
    if get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return get_origin(hint) in (Union, UnionType) and type(None) in get_args(hint)


@cache
def _struct_fields(kind: type[msgspec.Struct]) -> tuple[_Field, ...]:
    hints = get_type_hints(kind, include_extras=True)
    names = kind.__struct_encode_fields__
    fields: list[_Field] = []
    for attr, name in zip(kind.__struct_fields__, names, strict=True):
        hint = hints[attr]
        extras = get_args(hint)[1:] if get_origin(hint) is Annotated else ()
        fields.append(
            _Field(attr, name, OmitZero in extras, OmitEmpty in extras, _is_nilable(hint))
        )
    return tuple(fields)


def _is_zero_field(value: Any, field: _Field) -> bool:
    # A field that can hold None stands for a Go pointer, interface, nil-able
    # slice, or nil-able map, which is zero only when nil. Other list and dict
    # fields cannot tell nil from empty, so empty counts as zero.
    return value is None if field.nilable else _is_zero(value)


def _is_zero(value: Any) -> bool:
    match value:
        case None | False:
            return True
        case bool():
            return False
        case Enum():
            return _is_zero(value.value)
        case int() | float():
            return value == 0
        case str() | bytes() | bytearray() | list() | tuple() | dict():
            return len(value) == 0
        case msgspec.Raw():
            return len(value) == 0
        case Timestamp():
            return value.is_zero()
        case msgspec.Struct():
            return all(
                _is_zero_field(getattr(value, f.attr), f) for f in _struct_fields(type(value))
            )
        case _:
            is_zero = getattr(value, "is_zero", None)
            return bool(is_zero()) if callable(is_zero) else False


_EMPTY_ENCODINGS: Final = frozenset({b'""', b"null", b"[]", b"{}"})


def _is_container(value: Any) -> bool:
    return isinstance(value, list | tuple | Mapping) and not isinstance(value, msgspec.Struct)


class _Encoder:
    __slots__ = ("depth", "deterministic", "out")

    def __init__(self, *, deterministic: bool) -> None:
        self.deterministic = deterministic
        self.out = bytearray()
        self.depth = 0

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise JSONError(_DEPTH_ERROR)

    def write(self, value: Any) -> None:
        out = self.out
        match value:
            case None:
                out += b"null"
            case True:
                out += b"true"
            case False:
                out += b"false"
            case Enum():
                self.write(value.value)
            case int():
                out += str(value).encode()
            case float():
                out += format_float(value).encode()
            case str():
                out += _encode_str(value)
            case bytes() | bytearray() | memoryview():
                out += b'"' + base64.b64encode(value) + b'"'
            case msgspec.Raw():
                out += compact(value) if len(value) else b"null"
            case Timestamp():
                value.check_year_range()
                out += b'"' + value.format().encode() + b'"'
            case msgspec.Struct():
                self.write_struct(value)
            case Mapping() | list() | tuple():
                self.write_container(value)
            case _:
                raise JSONError(f"unsupported type {type(value).__name__}")

    def write_struct(self, value: msgspec.Struct) -> None:
        validate = getattr(value, "__gojson_validate__", None)
        if validate is not None:
            validate()
        self.enter()
        out = self.out
        out += b"{"
        first = True
        for field in _struct_fields(type(value)):
            item = getattr(value, field.attr)
            if field.omit_zero and _is_zero_field(item, field):
                continue
            mark = len(out)
            if not first:
                out += b","
            out += _encode_str(field.name) + b":"
            start = len(out)
            self.write(item)
            if field.omit_empty and bytes(out[start:]) in _EMPTY_ENCODINGS:
                del out[mark:]
                continue
            first = False
        out += b"}"
        self.depth -= 1

    def open(self, value: Any) -> tuple[Iterator[tuple[bytes, Any]], bytes]:
        # Nested lists and maps use an explicit stack so depth is limited by
        # MAX_DEPTH rather than the interpreter's recursion limit.
        self.enter()
        if not isinstance(value, Mapping):
            self.out += b"["
            return ((b"," if index else b"", item) for index, item in enumerate(value)), b"]"
        items = list(value.items())
        for key, _ in items:
            if not isinstance(key, str):
                raise JSONError(f"unsupported map key type {type(key).__name__}")
        if self.deterministic:
            items.sort(key=lambda item: item[0].encode("utf-8"))
        self.out += b"{"
        members = (
            ((b"," if index else b"") + _encode_str(key) + b":", item)
            for index, (key, item) in enumerate(items)
        )
        return members, b"}"

    def write_container(self, value: Any) -> None:
        out = self.out
        stack = [self.open(value)]
        while stack:
            members, close = stack[-1]
            entry = next(members, None)
            if entry is None:
                out += close
                stack.pop()
                self.depth -= 1
                continue
            prefix, item = entry
            out += prefix
            if _is_container(item):
                stack.append(self.open(item))
            else:
                self.write(item)


def _encode_str(value: str) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as error:
        raise JSONError("invalid UTF-8") from error


def encode(value: Any, *, deterministic: bool = False) -> bytes:
    """Encode like Go's json.Marshal. `deterministic` sorts map keys."""
    encoder = _Encoder(deterministic=deterministic)
    try:
        encoder.write(value)
    except RecursionError as error:
        raise JSONError(_DEPTH_ERROR) from error
    return bytes(encoder.out)
