"""Command-line parsing that matches Go's `flag` package for the runner's flags.

argparse cannot reproduce Go's rules (single-dash long names, `-name=value`,
a value taken from the next argument even when it starts with a dash, parsing
that stops at the first positional argument, and Go's error text), so this is
a small port of `flag.FlagSet.parseOne`.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Final

from unreal_agent._go import go_quote
from unreal_agent.runner.errors import RunnerError, describe, join
from unreal_agent.runner.streams import Writer, write_flushed

__all__ = ["DEFAULT_TOOL_HEARTBEAT_INTERVAL", "FlagError", "HelpError", "Options", "parse_flags"]

NANOSECOND: Final = 1
MICROSECOND: Final = 1000 * NANOSECOND
MILLISECOND: Final = 1000 * MICROSECOND
SECOND: Final = 1000 * MILLISECOND
MINUTE: Final = 60 * SECOND
HOUR: Final = 60 * MINUTE

DEFAULT_TOOL_HEARTBEAT_INTERVAL: Final = 10 * MINUTE
"""Nanoseconds, like Go's time.Duration."""

_UNITS: Final = {
    "ns": NANOSECOND,
    "us": MICROSECOND,
    "µs": MICROSECOND,  # U+00B5 micro sign
    "μs": MICROSECOND,  # U+03BC Greek mu
    "ms": MILLISECOND,
    "s": SECOND,
    "m": MINUTE,
    "h": HOUR,
}

_FLAGS: Final = frozenset(
    {"p", "session-directory", "workspace", "log-directory", "tool-heartbeat-interval"}
)

_OPTIONS: Final = """\
  -log-directory string
    \toptional session JSONL log directory; unset writes only to stdout
  -p prompt
    \tsend a request with the given prompt without reading stdin
  -session-directory string
    \tdirectory containing session files; defaults to $XDG_STATE_HOME/unreal-agent/sessions, \
or $HOME/.local/state/unreal-agent/sessions
  -tool-heartbeat-interval duration
    \ttool-wait heartbeat interval (0 disables) (default 10m0s)
  -workspace string
    \tagent workspace and Bash working directory (default ".")
"""

REQUEST_HELP: Final = """
Request schema (JSON object; unknown fields are rejected):
  messages: array of {role: "user", content: string, message_id?: UUID string}
    Non-empty array of user messages delivered in order. role defaults to "user";
    message_id defaults to a generated UUID.
  prompt: string
    Shorthand for one user message; used when messages is absent.
    Supply messages or prompt. messages takes precedence when both are present.
  model: string (optional)
    Provider model ID; defaults to UNREAL_HARNESS_LLM_MODEL or the provider default.
  max_attempts: positive integer (optional)
    Overrides UNREAL_HARNESS_LLM_MAX_ATTEMPTS (default 5); 1 disables retries.
  system_prompt: string (optional)
    Replaces the default system prompt.
  thinking_level: "low" | "medium" | "high" | "xhigh" | "max" (optional; default "high")
  session_id: non-empty string (optional)
    Creates or resumes a persisted session.
  disallowed_tools: array of non-empty strings (optional)
    Static tool names excluded from model context and execution.
  extra_allowed_tools: array of non-empty strings (optional; accepted but ignored)
  include_partial_messages: boolean (optional; accepted but ignored)
"""


class FlagError(RunnerError):
    """A command-line parsing error, reported after the usage text."""


class HelpError(RunnerError):
    """Go's flag.ErrHelp: -h or -help was given and the usage text was written."""

    def __init__(self) -> None:
        super().__init__("flag: help requested")


@dataclass(slots=True)
class Options:
    prompt: str | None = None
    session_directory: str = ""
    workspace: str = "."
    log_directory: str = ""
    tool_heartbeat_interval: int = DEFAULT_TOOL_HEARTBEAT_INTERVAL
    """Nanoseconds."""
    arguments: list[str] = field(default_factory=list)


def usage(name: str) -> str:
    return (
        "Usage:\n"
        f"  {name} [options] < request.json\n"
        f"  {name} [options] 'JSON request'\n"
        f"  {name} [options] -p 'prompt'\n"
        "\n"
        "Reads one JSON request from stdin unless a positional request or -p is supplied.\n"
        "Place options before the positional request. "
        "-p and a positional request are mutually exclusive.\n"
        "\n"
        "Options:\n"
    )


def _write_usage(name: str, output: Writer) -> RunnerError | None:
    try:
        write_flushed(output, usage(name).encode())
    except Exception as error:
        return RunnerError(f"write usage: {describe(error)}")
    with contextlib.suppress(Exception):  # Go's PrintDefaults ignores write errors.
        write_flushed(output, _OPTIONS.encode())
    try:
        write_flushed(output, REQUEST_HELP.encode())
    except Exception as error:
        return RunnerError(f"write request schema: {describe(error)}")
    return None


def parse_flags(name: str, args: list[str], output: Writer) -> Options:
    """Parse `args` like Go's flag.FlagSet with ContinueOnError.

    Errors are written to `output` followed by the usage text. Raises HelpError
    for -h and -help, and FlagError otherwise; a failed usage write replaces
    HelpError and is joined to FlagError.
    """
    options = Options()
    remaining = list(args)
    while remaining:
        argument = remaining[0]
        if len(argument) < 2 or argument[0] != "-":
            break
        minuses = 1
        if argument[1] == "-":
            minuses = 2
            if len(argument) == 2:
                remaining.pop(0)
                break
        flag = argument[minuses:]
        if flag == "" or flag[0] in "-=":
            _fail(name, output, f"bad flag syntax: {argument}")
        remaining.pop(0)
        value: str | None = None
        equals = flag.find("=", 1)
        if equals > 0:
            flag, value = flag[:equals], flag[equals + 1 :]
        if flag not in _FLAGS:
            if flag in ("help", "h"):
                usage_error = _write_usage(name, output)
                if usage_error is not None:
                    raise usage_error
                raise HelpError
            _fail(name, output, f"flag provided but not defined: -{flag}")
        if value is None:
            if not remaining:
                _fail(name, output, f"flag needs an argument: -{flag}")
            value = remaining.pop(0)
        match flag:
            case "p":
                options.prompt = value
            case "session-directory":
                options.session_directory = value
            case "workspace":
                options.workspace = value
            case "log-directory":
                options.log_directory = value
            case _:
                try:
                    options.tool_heartbeat_interval = parse_duration(value)
                except ValueError:
                    _fail(
                        name,
                        output,
                        f"invalid value {go_quote(value)} for flag -{flag}: parse error",
                    )
    options.arguments = remaining
    return options


def _fail(name: str, output: Writer, message: str) -> None:
    with contextlib.suppress(Exception):  # Go's flag package ignores this write error.
        write_flushed(output, message.encode("utf-8", "surrogateescape") + b"\n")
    error = FlagError(message)
    usage_error = _write_usage(name, output)
    raise error if usage_error is None else join(error, usage_error)


def parse_duration(text: str) -> int:
    """Go's time.ParseDuration, returning nanoseconds. Raises ValueError."""
    original = text
    invalid = ValueError(f"time: invalid duration {go_quote(original)}")
    overflow = invalid
    negative = False
    if text and text[0] in "+-":
        negative = text[0] == "-"
        text = text[1:]
    if text == "0":
        return 0
    if text == "":
        raise invalid
    total = 0
    while text:
        if not (text[0] == "." or "0" <= text[0] <= "9"):
            raise invalid
        start = len(text)
        whole, text = _leading_int(text)
        consumed_whole = start != len(text)
        fraction, scale, consumed_fraction = 0, 1.0, False
        if text and text[0] == ".":
            text = text[1:]
            start = len(text)
            fraction, scale, text = _leading_fraction(text)
            consumed_fraction = start != len(text)
        if not consumed_whole and not consumed_fraction:
            raise invalid
        index = 0
        while index < len(text) and not (text[index] == "." or "0" <= text[index] <= "9"):
            index += 1
        if index == 0:
            raise ValueError(f"time: missing unit in duration {go_quote(original)}")
        unit_name, text = text[:index], text[index:]
        unit = _UNITS.get(unit_name)
        if unit is None:
            raise ValueError(
                f"time: unknown unit {go_quote(unit_name)} in duration {go_quote(original)}"
            )
        if whole > (1 << 63) // unit:
            raise overflow
        whole *= unit
        if fraction > 0:
            whole += int(float(fraction) * (float(unit) / scale))
            if whole > 1 << 63:
                raise overflow
        total += whole
        if total > 1 << 63:
            raise overflow
    if negative:
        return -total
    if total > (1 << 63) - 1:
        raise overflow
    return total


def _leading_int(text: str) -> tuple[int, str]:
    index = 0
    value = 0
    while index < len(text) and "0" <= text[index] <= "9":
        if value > (1 << 63) // 10:
            raise ValueError("time: bad [0-9]*")
        value = value * 10 + ord(text[index]) - ord("0")
        if value > 1 << 63:
            raise ValueError("time: bad [0-9]*")
        index += 1
    return value, text[index:]


def _leading_fraction(text: str) -> tuple[int, float, str]:
    index = 0
    value = 0
    scale = 1.0
    overflow = False
    while index < len(text) and "0" <= text[index] <= "9":
        if not overflow:
            if value > ((1 << 63) - 1) // 10:
                overflow = True
            else:
                candidate = value * 10 + ord(text[index]) - ord("0")
                if candidate > 1 << 63:
                    overflow = True
                else:
                    value = candidate
                    scale *= 10
        index += 1
    return value, scale, text[index:]
