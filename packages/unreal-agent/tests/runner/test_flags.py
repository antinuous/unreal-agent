"""Go `flag` and `time.ParseDuration` parity for the runner's command line.

Expected values come from Go 1.27's time.ParseDuration and flag.FlagSet.
"""

import io

import pytest
from unreal_agent.runner import FlagError, HelpError
from unreal_agent.runner.flags import parse_duration, parse_flags, usage


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("0", 0),
        ("10m", 600000000000),
        ("10ms", 10000000),
        ("1.5h", 5400000000000),
        (".5s", 500000000),
        ("1.s", 1000000000),
        ("-1s", -1000000000),
        ("+2us", 2000),
        ("3µs", 3000),
        ("4μs", 4000),
        ("1h2m3s4ms5us6ns", 3723004005006),
        ("9223372036854775807ns", 9223372036854775807),
        ("-9223372036854775808ns", -9223372036854775808),
        ("2562047h47m16.854775807s", 9223372036854775807),
        ("1.0000000000000000000001s", 1000000000),
        ("0.1ns", 0),
    ],
)
def test_parse_duration(text: str, want: int) -> None:
    assert parse_duration(text) == want


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("9223372036854775808ns", 'time: invalid duration "9223372036854775808ns"'),
        ("2562048h", 'time: invalid duration "2562048h"'),
        ("", 'time: invalid duration ""'),
        (".", 'time: invalid duration "."'),
        ("1", 'time: missing unit in duration "1"'),
        ("1x", 'time: unknown unit "x" in duration "1x"'),
        ("s", 'time: invalid duration "s"'),
        ("-", 'time: invalid duration "-"'),
        ("1e3s", 'time: unknown unit "e" in duration "1e3s"'),
    ],
)
def test_parse_duration_rejects(text: str, message: str) -> None:
    with pytest.raises(ValueError, match="^" + message.replace(".", r"\.") + "$"):
        parse_duration(text)


def test_parse_flags_stops_at_first_positional() -> None:
    options = parse_flags("runner", ["-workspace", "w", "x", "-p", "y"], io.BytesIO())
    assert options.workspace == "w"
    assert options.prompt is None
    assert options.arguments == ["x", "-p", "y"]


@pytest.mark.parametrize(
    ("args", "rest"),
    [(["-", "-p", "y"], ["-", "-p", "y"]), (["--", "-p"], ["-p"]), (["--p=a", "b"], ["b"])],
)
def test_parse_flags_terminators(args: list[str], rest: list[str]) -> None:
    assert parse_flags("runner", args, io.BytesIO()).arguments == rest


def test_parse_flags_takes_next_argument_as_value() -> None:
    options = parse_flags("runner", ["-p", "-workspace", "-log-directory=-x"], io.BytesIO())
    assert options.prompt == "-workspace"
    assert options.log_directory == "-x"
    assert options.arguments == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-=x"], "bad flag syntax: -=x"),
        (["---p"], "bad flag syntax: ---p"),
        (["-nope"], "flag provided but not defined: -nope"),
        (["-workspace"], "flag needs an argument: -workspace"),
        (
            ["-tool-heartbeat-interval", "1x"],
            'invalid value "1x" for flag -tool-heartbeat-interval: parse error',
        ),
    ],
)
def test_parse_flags_errors_print_message_and_usage(args: list[str], message: str) -> None:
    output = io.BytesIO()
    with pytest.raises(FlagError) as raised:
        parse_flags("runner", args, output)
    assert str(raised.value) == message
    assert output.getvalue().startswith((message + "\n" + usage("runner")).encode())


def test_parse_flags_help_writes_usage() -> None:
    output = io.BytesIO()
    with pytest.raises(HelpError):
        parse_flags("runner", ["--help"], output)
    assert output.getvalue().startswith(usage("runner").encode())
