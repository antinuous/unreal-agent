from typing import Annotated, Any

import msgspec
import pytest
from unreal_agent.gojson import (
    Int64,
    JSONError,
    OmitEmpty,
    OmitZero,
    Raw,
    Struct,
    Timestamp,
    compact,
    decode,
    encode,
    format_float,
)


class Sample(Struct):
    A: Int64 = 0
    B: Annotated[Raw, OmitZero] = Raw(b"")
    C: list[str] = []
    D: bytes = b""
    E: int | None = None
    F: dict[str, Any] = {}
    G: float = 0.0
    T: Timestamp = Timestamp()


class Omits(Struct):
    Zero: Annotated[list[str] | None, OmitZero] = None
    Empty: Annotated[list[str], OmitEmpty] = []
    Nested: Annotated[Sample | None, OmitZero] = None
    Kept: str = ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1"),
        (0.1, "0.1"),
        (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (1e-6, "0.000001"),
        (1e-7, "1e-7"),
        (123456789.125, "123456789.125"),
        (2.5e-8, "2.5e-8"),
        (1.5e300, "1.5e+300"),
        (-0.25, "-0.25"),
    ],
)
def test_format_float_matches_go(value: float, expected: str) -> None:
    assert format_float(value) == expected


def test_encode_matches_go_struct_output() -> None:
    value = Sample(
        B=Raw(b' { "x" :  [1, 2.50] } '),
        G=1e21,
        T=Timestamp.parse("2026-01-02T03:04:05.12Z"),
    )
    assert encode(value) == (
        b'{"A":0,"B":{"x":[1,2.50]},"C":[],"D":"","E":null,"F":{},"G":1e+21,'
        b'"T":"2026-01-02T03:04:05.12Z"}'
    )


def test_encode_string_escaping_matches_go() -> None:
    source = '\u2028\u2029<>&\x7f\x00\x01\x1f\b\f\n\r\t"\\é\U0001f600'
    expected = '"\u2028\u2029<>&\x7f\\u0000\\u0001\\u001f\\b\\f\\n\\r\\t\\"\\\\é\U0001f600"'
    assert encode(source) == expected.encode()


def test_encode_rejects_lone_surrogate() -> None:
    with pytest.raises(JSONError):
        encode("\ud800")


def test_map_order() -> None:
    value = {"b": 1, "a": 2, "c": 3}
    assert encode(value) == b'{"b":1,"a":2,"c":3}'
    assert encode(value, deterministic=True) == b'{"a":2,"b":1,"c":3}'


def test_omit_rules() -> None:
    assert encode(Omits()) == b'{"Kept":""}'
    assert encode(Omits(Zero=["a"], Empty=["b"], Nested=Sample(A=1))).startswith(
        b'{"Zero":["a"],"Empty":["b"],"Nested":{"A":1,'
    )


@pytest.mark.parametrize(
    "data",
    [
        b'{"A":1,"A":2}',
        b'{"A":1.5}',
        b'{"A":1e2}',
        b'{"B":"\xff"}',
        b'{"B":"\\ud800"}',
        b'{"A":9223372036854775808}',
        b'{"A":NaN}',
        b'{"A":1} x',
    ],
)
def test_decode_rejects_like_go(data: bytes) -> None:
    with pytest.raises(JSONError):
        decode(data, Sample)


def test_decode_accepts_like_go() -> None:
    assert decode(b'{"a":1}', Sample).A == 0
    assert decode(b'{"A":-0}', Sample).A == 0
    assert bytes(decode(b'{"B":"\\ud800\\udc00"}', Sample).B) == b'"\\ud800\\udc00"'
    raw = decode(b'{"B":{"x" : 1 }}', Sample).B
    assert bytes(raw) == b'{"x" : 1 }'
    assert encode(Sample(B=raw)).startswith(b'{"A":0,"B":{"x":1},')


def test_timestamp_round_trip() -> None:
    for text in [
        "2026-01-01T00:00:00.123456789+02:00",
        "2026-08-27T10:01:00Z",
        "0001-01-01T00:00:00Z",
        "2026-01-02T03:04:05.1-07:30",
    ]:
        assert Timestamp.parse(text).format() == text
    decoded = decode(b'{"T":"2026-01-01T00:00:00.123456789+02:00"}', Sample)
    assert decoded.T.unix_ns % 10**9 == 123456789
    assert Timestamp().is_zero()
    with pytest.raises(JSONError):
        Timestamp.parse("2026-01-01 00:00:00Z")


def test_compact_preserves_tokens() -> None:
    assert compact(b' { "a b" : [ 1.50 , "\\" x" ] } ') == b'{"a b":[1.50,"\\" x"]}'


def test_compact_rewrites_string_escapes_like_go() -> None:
    source = b'{"a\\u0062" : ["\\u003c\\u00e9\\/\\ud83d\\ude00\\u2028\\n\\u0007x", 1.50e+3, -0]}'
    expected = ('{"ab":["<é/😀' + chr(0x2028) + '\\n\\u0007x",1.50e+3,-0]}').encode()
    assert compact(source) == expected
    assert encode(Sample(B=Raw(source))).startswith(b'{"A":0,"B":' + expected + b",")


def test_encode_empty_raw_as_null_like_go() -> None:
    class Value(Struct):
        V: Raw = Raw(b"")
        E: Annotated[Raw, OmitEmpty] = Raw(b"")

    assert encode(Raw(b"")) == b"null"
    assert encode([Raw(b"")]) == b"[null]"
    assert encode(Value()) == b'{"V":null}'


@pytest.mark.parametrize(
    "text",
    [
        "2026-01-01t00:00:00Z",
        "2026-01-01T00:00:00z",
        "2026-01-01T00:00:00,5Z",
        "2026-01-01T00:00:00.Z",
        "0001-02-29T00:00:00Z",
        "\u0662026-01-01T00:00:00Z",
    ],
)
def test_timestamp_parse_rejects_like_go(text: str) -> None:
    with pytest.raises(JSONError):
        Timestamp.parse(text)


@pytest.mark.parametrize(
    ("text", "formatted"),
    [
        ("2026-01-01T00:00:00.1234567891Z", "2026-01-01T00:00:00.123456789Z"),
        ("2026-01-01T00:00:00.9999999999Z", "2026-01-01T00:00:00.999999999Z"),
        ("0000-02-29T00:00:00Z", "0000-02-29T00:00:00Z"),
        ("0000-01-01T00:00:00+01:00", "0000-01-01T00:00:00+01:00"),
        ("9999-12-31T23:59:59.999999999-23:59", "9999-12-31T23:59:59.999999999-23:59"),
    ],
)
def test_timestamp_parse_accepts_like_go(text: str, formatted: str) -> None:
    assert Timestamp.parse(text).format() == formatted


def test_timestamp_year_zero_orders_before_go_zero() -> None:
    year_zero = Timestamp.parse("0000-12-31T23:59:59Z")
    assert year_zero.unix_ns == Timestamp().unix_ns - 10**9
    assert (
        Timestamp.parse("0000-01-01T00:00:00Z").unix_ns == Timestamp().unix_ns - 366 * 86400 * 10**9
    )


def test_encode_rejects_years_outside_go_range() -> None:
    last = Timestamp.parse("9999-12-31T23:59:59Z")
    with pytest.raises(JSONError, match=r"year outside of range \[0,9999\]"):
        encode(Timestamp(last.unix_ns + 10**9))
    first = Timestamp.parse("0000-01-01T00:00:00Z")
    with pytest.raises(JSONError, match=r"year outside of range \[0,9999\]"):
        encode(Timestamp(first.unix_ns - 1))


class NilSlices(Struct):
    L: Annotated[list[str] | None, OmitZero] = None
    M: Annotated[dict[str, int] | None, OmitZero] = None


class Wrapper(Struct):
    Inner: Annotated[NilSlices, OmitZero] = msgspec.field(default_factory=NilSlices)


def test_omitzero_omits_only_nil_slices_and_maps_like_go() -> None:
    assert encode(NilSlices()) == b"{}"
    assert encode(NilSlices(L=[], M={})) == b'{"L":[],"M":{}}'
    assert encode(Wrapper()) == b"{}"
    assert encode(Wrapper(Inner=NilSlices(L=[]))) == b'{"Inner":{"L":[]}}'
    assert encode(Omits(Nested=Sample())).startswith(b'{"Nested":{"A":0,')
    assert decode(b'{"L":[],"M":{}}', NilSlices) == NilSlices(L=[], M={})
    assert decode(b"{}", NilSlices) == NilSlices()
    assert decode(b'{"L":null,"M":null}', NilSlices) == NilSlices()


class Enumish(Struct):
    S: Sample = msgspec.field(default_factory=Sample)
    K: Omits = msgspec.field(default_factory=Omits)


def test_decode_null_into_non_pointer_is_zero_like_go() -> None:
    got = decode(
        b'{"A":null,"B":null,"C":null,"D":null,"E":null,"F":null,"G":null,"T":null}', Sample
    )
    assert got == Sample(B=Raw(b"null"))
    assert bytes(got.B) == b"null"
    assert decode(b'{"S":null,"K":{"Nested":null,"Kept":null}}', Enumish) == Enumish()
    assert decode(b'{"C":["a",null],"F":{"x":null}}', Sample) == Sample(C=["a", ""], F={"x": None})
    assert decode(b"[1,null,3]", list[int]) == [1, 0, 3]
    assert decode(b'{"a":null}', dict[str, int]) == {"a": 0}
    assert decode(b'{"a":null}', dict[str, bool]) == {"a": False}
    assert decode(b"null", Sample) == Sample()
    assert decode(b"null", str) == ""
    assert decode(b'{"E":null}', Sample).E is None
    assert decode(b'{"A":"null"}', Omits) == Omits()


def test_decode_null_keeps_go_errors() -> None:
    with pytest.raises(JSONError):
        decode(b'{"A":null,"C":1}', Sample)
    with pytest.raises(JSONError):
        decode(b'{"A":null,"A":null}', Sample)


def _nested(depth: int) -> bytes:
    return b"[" * depth + b"]" * depth


def test_nesting_limit_matches_go() -> None:
    assert decode(_nested(10000), Any) is not None
    assert decode(b'{"F":{"a":' + _nested(9998) + b"}}", Sample).F["a"] is not None
    for data in (_nested(10001), b'{"F":{"a":' + _nested(9999) + b"}}", _nested(200000)):
        with pytest.raises(JSONError, match="exceeded max depth"):
            decode(data, Any)
    assert decode(b'["' + b"[" * 20000 + b'"]', list[str]) == ["[" * 20000]
    value: Any = []
    for _ in range(9999):
        value = [value]
    assert encode(value) == _nested(10000)
    with pytest.raises(JSONError, match="exceeded max depth"):
        encode([value])
    assert encode({"x": Sample(F={"y": value[0][0][0]})}).startswith(b'{"x":{"A":0')
