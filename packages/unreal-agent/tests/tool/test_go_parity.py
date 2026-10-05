"""Pins Go behavior captured from the reference implementation.

testdata/argument_cases.json holds the CallStatus and Spec bytes Go produced
for each argument string.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Final

import pytest
from unreal_agent.gojson import JSONError, decode, encode
from unreal_agent.llm import ToolCall
from unreal_agent.tool import (
    SKILL_USE_NAME,
    CallStatus,
    Skill,
    StaticTranslators,
    Translator,
    bash,
    discover_skills,
    new_registry,
    viewimage,
)
from unreal_agent.tool._glob import BadPatternError, match
from unreal_agent.tool._go import go_quote, go_trim_space, to_valid_utf8

from tests.tool._helpers import RecordingContext

_CASES: Final = json.loads((Path(__file__).parent / "testdata" / "argument_cases.json").read_text())


def translators() -> dict[str, Translator]:
    registry = new_registry(StaticTranslators(), SKILL_USE_NAME)
    registry.register_skill(
        Skill(name="review", description="Review code.", path="/skills/review/SKILL.md")
    )
    skill_use = registry.resolve(SKILL_USE_NAME)
    assert skill_use is not None
    return {
        "bash": bash.new(
            bash.Config(shell="/bin/sh", directory="/workspace", base_directory="/operations")
        ),
        "view_image": viewimage.new(viewimage.Config(directory="/workspace")),
        "skill_use": skill_use,
    }


def case_arguments(case: dict[str, Any]) -> str:
    if "arguments" in case:
        return case["arguments"]
    return base64.b64decode(case["arguments_base64"]).decode("utf-8", "surrogateescape")


@pytest.mark.parametrize("tool", ["bash", "view_image", "skill_use"])
@pytest.mark.parametrize("case", _CASES, ids=[str(index) for index in range(len(_CASES))])
def test_arguments_match_go(case: dict[str, Any], tool: str) -> None:
    ctx = RecordingContext()
    status = translators()[tool].translate(ctx, ToolCall(Arguments=case_arguments(case)))
    assert encode(status).decode() == case[tool]["status"]
    want_spec = case[tool].get("spec")
    if want_spec is None:
        assert ctx.specs == []
    else:
        assert [encode(spec).decode() for spec in ctx.specs] == [want_spec]


_ARRAY_DEPTH_ERROR: Final = (
    'jsontext: exceeded max depth within "/a/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/…'
    '/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0" after offset 10004'
)
_OBJECT_DEPTH_ERROR: Final = (
    'jsontext: exceeded max depth within "/a/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/…'
    '/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b/b" after offset 50000'
)


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        pytest.param('{"a":' + "[" * 9998 + "]" * 9998 + "}", "", id="arrays 9998"),
        pytest.param('{"a":' + "[" * 9999 + "]" * 9999 + "}", "", id="arrays 9999"),
        pytest.param(
            '{"a":' + "[" * 10000 + "]" * 10000 + "}", _ARRAY_DEPTH_ERROR, id="arrays 10000"
        ),
        pytest.param(
            '{"a":' + "[" * 10001 + "]" * 10001 + "}", _ARRAY_DEPTH_ERROR, id="arrays 10001"
        ),
        pytest.param('{"a":' + '{"b":' * 9999 + "1" + "}" * 9999 + "}", "", id="objects 9999"),
        pytest.param(
            '{"a":' + '{"b":' * 10000 + "1" + "}" * 10000 + "}",
            _OBJECT_DEPTH_ERROR,
            id="objects 10000",
        ),
    ],
)
def test_nesting_limit_matches_go(arguments: str, error: str) -> None:
    for tool, (prefix, missing) in {
        "bash": ("decode Bash arguments: ", 'bash argument "command" must be set'),
        "view_image": ("decode ViewImage arguments: ", 'ViewImage argument "path" must be set'),
        "skill_use": ("decode skill-use arguments: ", 'skill-use argument "name" must be set'),
    }.items():
        status = translators()[tool].translate(RecordingContext(), ToolCall(Arguments=arguments))
        assert status.Error == (prefix + error if error else missing), tool


@pytest.mark.parametrize(
    ("status", "encoded"),
    [
        pytest.param(CallStatus(), b'{"Error":""}', id="empty"),
        pytest.param(CallStatus(Error="bad"), b'{"Error":"bad"}', id="error"),
        pytest.param(
            CallStatus(Error="x...1 bytes truncated...y", ErrorTruncated=True),
            b'{"Error":"x...1 bytes truncated...y","ErrorTruncated":true}',
            id="truncated",
        ),
        pytest.param(
            CallStatus(WaitingFor=["a", "b"]), b'{"Error":"","WaitingFor":["a","b"]}', id="waiting"
        ),
    ],
)
def test_call_status_encodes_like_go(status: CallStatus, encoded: bytes) -> None:
    assert encode(status) == encoded
    assert decode(encoded, CallStatus) == status


def test_call_status_decoding_is_strict() -> None:
    assert decode(b"{}", CallStatus) == CallStatus()
    with pytest.raises(JSONError):
        decode(b'{"Error":"a","Error":"b"}', CallStatus)


@pytest.mark.parametrize(
    ("value", "quoted"),
    [
        ("plain", '"plain"'),
        ('a"b\\c', '"a\\"b\\\\c"'),
        ("\a\b\f\n\r\t\v", '"\\a\\b\\f\\n\\r\\t\\v"'),
        ("\x00\x1f\x7f", '"\\x00\\x1f\\x7f"'),
        ("\x85\xa0\u2028\ufeff😀\U000e0001", '"\\u0085\\u00a0\\u2028\\ufeff😀\\U000e0001"'),
        (b"\xff\xc3", '"\\xff\\xc3"'),
        ("é雪", '"é雪"'),
        ("'", '"\'"'),
    ],
)
def test_go_quote(value: str | bytes, quoted: str) -> None:
    assert go_quote(value) == quoted


@pytest.mark.parametrize(
    ("data", "text"),
    [
        (b"ok", "ok"),
        (b"\xff\xfea\xff", "\ufffda\ufffd"),
        (b"\xed\xa0\x80", "\ufffd"),
        ("\ufffd".encode() + b"\xff", "\ufffd\ufffd"),
        (b"\xe2\x82x\xe2\x82\xac", "\ufffdx€"),
    ],
)
def test_to_valid_utf8(data: bytes, text: str) -> None:
    assert to_valid_utf8(data) == text


def test_trim_space_keeps_characters_go_does_not_treat_as_space() -> None:
    assert go_trim_space("\x1c x\x1f \x85\xa0\u3000") == "\x1c x\x1f"


@pytest.mark.parametrize(
    ("pattern", "name", "matched"),
    [
        ("*", ".x", True),
        ("a?c", "abc", True),
        ("a?c", "a/c", False),
        ("[a-c]x", "bx", True),
        ("[^a-c]x", "bx", False),
        ("[^a-c]x", "dx", True),
        ("\\*", "*", True),
        ("*a*b", "xaxb", True),
        ("*.go", "a.go", True),
        ("*", "a/b", False),
        ("[\\]]", "]", True),
        ("é?", "éx", True),
        ("?", "é", True),
    ],
)
def test_glob_match(pattern: str, name: str, matched: bool) -> None:
    assert match(pattern.encode(), name.encode()) == matched


@pytest.mark.parametrize("pattern", ["[", "a[", "[]", "[a-]", "\\"])
def test_glob_match_rejects_bad_patterns(pattern: str) -> None:
    with pytest.raises(BadPatternError, match=r"^syntax error in pattern$"):
        match(pattern.encode(), b"a")


def build_skill_tree(root: Path) -> None:
    def write(relative: str, data: bytes) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    write(".hidden/SKILL.md", b"---\nname: hidden\ndescription: Hidden dir.\n---\n")
    write("crlf/SKILL.md", b"---\r\nname: crlf \r\ndescription:  Windows lines. \r\n---\r\n")
    write("long-before/SKILL.md", b"---\nname: long\n" + b"x" * 65536 + b"\ndescription: d\n---\n")
    write(
        "long-edge/SKILL.md", b"---\nname: edge\ndescription: Edge.\n" + b"y" * 65535 + b"\n---\n"
    )
    write("long-after/SKILL.md", b"---\nname: after\ndescription: After.\n---\n" + b"z" * 70000)
    write("long-first/SKILL.md", b"-" * 65536 + b"\n")
    (root / "dir-skill" / "SKILL.md").mkdir(parents=True)
    write("lower/skill.md", b"---\nname: lower\ndescription: Lower.\n---\n")
    write(
        "spaces/SKILL.md",
        b"  ---  \nname:   spaced\t\ndescription:\tTabs\x1c\n  name: ignored\nName: ignored\n---\n",
    )
    write("unclosed/SKILL.md", b"---\nname: unclosed\ndescription: Unclosed.\n")
    write("empty/SKILL.md", b"")
    write("invalid-utf8/SKILL.md", b"---\nname: bad\xff\ndescription: Bad bytes\xc3.\n---\n")
    write("colon/SKILL.md", b"---\nname: a:b\ndescription: c: d\n---\n")
    write("zz-dup/SKILL.md", b"---\nname: a:b\ndescription: dup\n---\n")
    os.symlink(root / "crlf", root / "link-crlf")
    write("plainfile", b"x")


def expected_discovery(root: str) -> tuple[list[Skill], list[str]]:
    def path(name: str) -> str:
        return f"{root}/{name}/SKILL.md"

    def quoted(name: str) -> str:
        return go_quote(path(name))

    skills = [
        Skill("hidden", "Hidden dir.", path(".hidden")),
        Skill("a:b", "c: d", path("colon")),
        Skill("crlf", "Windows lines.", path("crlf")),
        Skill("bad\udcff", "Bad bytes\udcc3.", path("invalid-utf8")),
        Skill("after", "After.", path("long-after")),
        Skill("edge", "Edge.", path("long-edge")),
        Skill("spaced", "Tabs\x1c", path("spaces")),
    ]
    errors = [
        f"read skill {quoted('dir-skill')}: read {path('dir-skill')}: is a directory",
        f"parse skill {quoted('empty')}: missing opening YAML frontmatter delimiter",
        f'skill {quoted("link-crlf")}: duplicate name "crlf"',
        f"parse skill {quoted('long-before')}: read YAML frontmatter: "
        "bufio.Scanner: token too long",
        f"parse skill {quoted('long-first')}: missing opening YAML frontmatter delimiter",
        f"parse skill {quoted('unclosed')}: missing closing YAML frontmatter delimiter",
        f'skill {quoted("zz-dup")}: duplicate name "a:b"',
    ]
    return skills, errors


def test_discover_skills_matches_go(tmp_path: Path) -> None:
    root = tmp_path / "sk[i]lls"
    build_skill_tree(root)
    want_skills, want_errors = expected_discovery(str(root))
    escaped = str(tmp_path) + "/sk\\[i]lls"
    for directory in (
        escaped,
        str(tmp_path) + "/sk*",
        str(tmp_path) + "/./sk\\[i\\]lls/../sk\\[i\\]lls",
    ):
        skills, errors = discover_skills(directory)
        assert skills == want_skills, directory
        assert [str(error) for error in errors] == want_errors, directory
    assert discover_skills(str(root)) == ([], [])
    skills, errors = discover_skills(str(tmp_path) + "/sk[i")
    assert skills == []
    assert [str(error) for error in errors] == ["find skills: syntax error in pattern"]
