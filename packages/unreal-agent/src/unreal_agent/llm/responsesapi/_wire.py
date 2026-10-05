"""Go `encoding/json/v2` and `net/http` behaviors that the wire code depends on."""

from typing import Final

from unreal_agent.gojson import JSONError, compact, decode, validate

_ASCII_SPACE: Final = b" \t\n\v\f\r"
_UNICODE_SPACE: Final = " \t\n\v\f\r" + "".join(
    chr(code)
    for code in (0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)
)
_UNICODE_SPACE_BYTES: Final = tuple(character.encode() for character in _UNICODE_SPACE[6:])

_STATUS_TEXT: Final = {
    100: "Continue",
    101: "Switching Protocols",
    102: "Processing",
    103: "Early Hints",
    200: "OK",
    201: "Created",
    202: "Accepted",
    203: "Non-Authoritative Information",
    204: "No Content",
    205: "Reset Content",
    206: "Partial Content",
    207: "Multi-Status",
    208: "Already Reported",
    226: "IM Used",
    300: "Multiple Choices",
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified",
    305: "Use Proxy",
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    407: "Proxy Authentication Required",
    408: "Request Timeout",
    409: "Conflict",
    410: "Gone",
    411: "Length Required",
    412: "Precondition Failed",
    413: "Request Entity Too Large",
    414: "Request URI Too Long",
    415: "Unsupported Media Type",
    416: "Requested Range Not Satisfiable",
    417: "Expectation Failed",
    418: "I'm a teapot",
    421: "Misdirected Request",
    422: "Unprocessable Entity",
    423: "Locked",
    424: "Failed Dependency",
    425: "Too Early",
    426: "Upgrade Required",
    428: "Precondition Required",
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    505: "HTTP Version Not Supported",
    506: "Variant Also Negotiates",
    507: "Insufficient Storage",
    508: "Loop Detected",
    510: "Not Extended",
    511: "Network Authentication Required",
}


def status_text(status_code: int) -> str:
    """Go's `http.StatusText`."""
    return _STATUS_TEXT.get(status_code, "")


def trim_space(data: bytes) -> bytes:
    """Go's `bytes.TrimSpace`, which also trims Unicode white space."""
    while True:
        data = data.strip(_ASCII_SPACE)
        for space in _UNICODE_SPACE_BYTES:
            if data.startswith(space):
                data = data[len(space) :]
                break
            if data.endswith(space):
                data = data[: -len(space)]
                break
        else:
            return data


def trim_space_text(text: str) -> str:
    """Go's `strings.TrimSpace`."""
    return text.strip(_UNICODE_SPACE)


def decode_object[T](data: bytes, kind: type[T]) -> T:
    """Decode a JSON object into a struct. JSON null leaves the zero value, as in Go."""
    if trim_space(data) == b"null":
        validate(data)
        return kind()
    return decode(data, kind)


def reformat(value: bytes) -> bytes:
    if not value:
        raise JSONError("empty raw JSON value")
    return compact(value)


def reformat_checked(value: bytes) -> bytes:
    validate(value)
    return reformat(value)
