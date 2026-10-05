"""JSON encoding compatible with the Go `encoding/json/v2` wire format.

Persisted sessions, operations, and runner output keep the format produced by
the original Go implementation: struct fields in declaration order, PascalCase
names, `omitzero`/`omitempty` semantics, base64 bytes, RFC 3339 times with
nanosecond precision, and raw JSON values passed through without reparsing.
Decoding is strict in the same ways as Go: duplicate object members, invalid
UTF-8, unpaired surrogate escapes, and non-finite numbers are rejected.
"""

import base64
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from functools import cache
from typing import Annotated, Any, Final, Self, get_args, get_origin, get_type_hints

import msgspec

__all__ = [
    "Int64",
    "JSONError",
    "OmitEmpty",
    "OmitZero",
    "Raw",
    "Struct",
    "Timestamp",
    "Uint32",
    "Uint64",
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

Uint64 = Annotated[int, msgspec.Meta(ge=0, le=2**64 - 1)]
Uint32 = Annotated[int, msgspec.Meta(ge=0, le=2**32 - 1)]
Int64 = Annotated[int, msgspec.Meta(ge=-(2**63), le=2**63 - 1)]

Raw = msgspec.Raw


class Struct(msgspec.Struct, kw_only=True):
    """Base for Go-compatible wire structs.

    Every field must have a default so decoding tolerates absent members the
    way Go does. Annotate a field with `OmitZero` or `OmitEmpty` to mirror the
    corresponding Go struct tag.
    """


_TIME_RE: Final = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(?:([Zz])|([+-])(\d{2}):(\d{2}))\Z"
)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_GO_ZERO_UNIX_NS: Final = -62135596800 * 10**9


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
        try:
            base = datetime(
                year, month, day, hour, minute, second, tzinfo=timezone(timedelta(seconds=offset))
            )
        except ValueError as error:
            raise JSONError(f"invalid RFC 3339 timestamp {text!r}") from error
        nanos = int(fraction.ljust(9, "0")) if fraction else 0
        return cls(cls.from_datetime(base).unix_ns + nanos, offset)

    def is_zero(self) -> bool:
        return self.unix_ns == _GO_ZERO_UNIX_NS

    def to_datetime(self) -> datetime:
        """Return the instant truncated to microseconds."""
        tz = UTC if self.offset_seconds == 0 else timezone(timedelta(seconds=self.offset_seconds))
        return (_EPOCH + timedelta(microseconds=self.unix_ns // 1000)).astimezone(tz)

    def utc(self) -> Timestamp:
        return Timestamp(self.unix_ns, 0)

    def format(self) -> str:
        """Format like Go's time.RFC3339Nano."""
        seconds, nanos = divmod(self.unix_ns, 10**9)
        local = _EPOCH + timedelta(seconds=seconds + self.offset_seconds)
        text = local.strftime("%Y-%m-%dT%H:%M:%S")
        if local.year < 1000:
            text = f"{local.year:04d}{text[text.index('-') :]}"
        if nanos:
            text += "." + f"{nanos:09d}".rstrip("0")
        if self.offset_seconds == 0:
            return text + "Z"
        sign = "-" if self.offset_seconds < 0 else "+"
        hours, minutes = divmod(abs(self.offset_seconds) // 60, 60)
        return f"{text}{sign}{hours:02d}:{minutes:02d}"

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
    raw = data.encode("utf-8") if isinstance(data, str) else data
    try:
        return _decoder(kind).decode(raw)
    except msgspec.DecodeError as error:
        raise JSONError(str(error)) from error


_WS: Final = frozenset(b" \t\r\n")


def compact(data: bytes | bytearray | memoryview) -> bytes:
    """Remove insignificant whitespace without altering any token."""
    source = bytes(data)
    if not any(byte in _WS for byte in source):
        return source
    out = bytearray()
    in_string = False
    escaped = False
    for byte in source:
        if in_string:
            out.append(byte)
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
            out.append(byte)
        elif byte not in _WS:
            out.append(byte)
    return bytes(out)


@dataclass(frozen=True, slots=True)
class _Field:
    attr: str
    name: str
    omit_zero: bool
    omit_empty: bool


@cache
def _struct_fields(kind: type[msgspec.Struct]) -> tuple[_Field, ...]:
    hints = get_type_hints(kind, include_extras=True)
    names = kind.__struct_encode_fields__
    fields: list[_Field] = []
    for attr, name in zip(kind.__struct_fields__, names, strict=True):
        extras = get_args(hints[attr])[1:] if get_origin(hints[attr]) is Annotated else ()
        fields.append(_Field(attr, name, OmitZero in extras, OmitEmpty in extras))
    return tuple(fields)


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
            return all(_is_zero(getattr(value, f.attr)) for f in _struct_fields(type(value)))
        case _:
            is_zero = getattr(value, "is_zero", None)
            return bool(is_zero()) if callable(is_zero) else False


_EMPTY_ENCODINGS: Final = frozenset({b'""', b"null", b"[]", b"{}"})


class _Encoder:
    __slots__ = ("deterministic", "out")

    def __init__(self, *, deterministic: bool) -> None:
        self.deterministic = deterministic
        self.out = bytearray()

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
                if len(value) == 0:
                    raise JSONError("empty raw JSON value")
                out += compact(value)
            case Timestamp():
                out += b'"' + value.format().encode() + b'"'
            case msgspec.Struct():
                self.write_struct(value)
            case Mapping():
                self.write_mapping(value)
            case list() | tuple():
                out += b"["
                for index, item in enumerate(value):
                    if index:
                        out += b","
                    self.write(item)
                out += b"]"
            case _:
                raise JSONError(f"unsupported type {type(value).__name__}")

    def write_struct(self, value: msgspec.Struct) -> None:
        out = self.out
        out += b"{"
        first = True
        for field in _struct_fields(type(value)):
            item = getattr(value, field.attr)
            if field.omit_zero and _is_zero(item):
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

    def write_mapping(self, value: Mapping[Any, Any]) -> None:
        out = self.out
        items = list(value.items())
        for key, _ in items:
            if not isinstance(key, str):
                raise JSONError(f"unsupported map key type {type(key).__name__}")
        if self.deterministic:
            items.sort(key=lambda item: item[0].encode("utf-8"))
        out += b"{"
        for index, (key, item) in enumerate(items):
            if index:
                out += b","
            out += _encode_str(key) + b":"
            self.write(item)
        out += b"}"


def _encode_str(value: str) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as error:
        raise JSONError("invalid UTF-8") from error


def encode(value: Any, *, deterministic: bool = False) -> bytes:
    """Encode like Go's json.Marshal. `deterministic` sorts map keys."""
    encoder = _Encoder(deterministic=deterministic)
    encoder.write(value)
    return bytes(encoder.out)
