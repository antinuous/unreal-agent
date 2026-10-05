"""System messages pinned to bytes captured from the Go implementation."""

import base64
import json
from pathlib import Path
from typing import Any, Final

import pytest
from unreal_agent.contextbuilder import PREAMBLE, new_builder
from unreal_agent.llm import Message
from unreal_agent.tool import Skill

_CASES: Final = json.loads((Path(__file__).parent / "testdata" / "system_prompts.json").read_text())
_SKILL_PREAMBLE: Final = (
    "The following skills provide specialized instructions for specific tasks.\n"
    "Use SkillUse to load a skill's file when the task matches its description.\n"
    "When a skill file references a relative path, resolve it against the skill directory "
    "(parent of SKILL.md / dirname of the path) and use that absolute path in tool calls."
)


def go_string(encoded: str) -> str:
    return base64.b64decode(encoded).decode("utf-8", "surrogateescape")


def system_text(*skills: Skill, prompt: str = "") -> str:
    builder = new_builder(*skills)
    builder.set_system_prompt(prompt)
    message = builder.build().request.Input[0].Data
    assert isinstance(message, Message)
    return message.Text


@pytest.mark.parametrize("case", _CASES, ids=[str(index) for index in range(len(_CASES))])
def test_system_message_matches_go(case: dict[str, Any]) -> None:
    fields = [go_string(value) for value in case["skills"] or []]
    skills = [Skill(*fields[index : index + 3]) for index in range(0, len(fields), 3)]
    got = system_text(*skills, prompt=go_string(case["prompt"]))
    assert got.encode("utf-8", "surrogateescape") == base64.b64decode(case["system"])


@pytest.mark.parametrize(
    ("value", "escaped"),
    [
        pytest.param("a\"b'c<d>e&f", "a&#34;b&#39;c&lt;d&gt;e&amp;f", id="markup"),
        pytest.param("line\nbreak\ttab\rcr", "line&#xA;break&#x9;tab&#xD;cr", id="whitespace"),
        pytest.param("ctl\x01\x7f\x0b\x0c end", "ctl\ufffd\x7f\ufffd\ufffd end", id="controls"),
        pytest.param(
            "uni é 雪 😀 \u2028 \x85 \ufffe \uffff \ufffd",
            "uni é 雪 😀 \u2028 \x85 \ufffd \ufffd \ufffd",
            id="unicode",
        ),
        pytest.param("\udcffx\udcc3", "\ufffdx\ufffd", id="invalid UTF-8"),
        pytest.param("]]> \x00 nul", "]]&gt; \ufffd nul", id="CDATA end and NUL"),
        pytest.param("  spaced  ", "  spaced  ", id="spaces"),
        pytest.param("", "", id="empty"),
    ],
)
def test_available_skills_escape_like_go_xml(value: str, escaped: str) -> None:
    want = (
        f"{PREAMBLE}\n\n{_SKILL_PREAMBLE}\n\n<available_skills><skill><name>{escaped}</name>"
        f"<description>{escaped}</description><location>{escaped}</location></skill>"
        "</available_skills>"
    )
    assert system_text(Skill(value, value, value)) == want


@pytest.mark.parametrize(
    ("prompt", "want"),
    [
        ("", PREAMBLE),
        ("  x  ", PREAMBLE + "\n\n  x"),
        ("\x1c", PREAMBLE + "\n\n\x1c"),
        ("\xa0x\xa0", PREAMBLE + "\n\n\xa0x"),
        ("\u3000x\u2029", PREAMBLE + "\n\n\u3000x"),
    ],
)
def test_system_prompt_is_trimmed_like_go(prompt: str, want: str) -> None:
    assert system_text(prompt=prompt) == want
