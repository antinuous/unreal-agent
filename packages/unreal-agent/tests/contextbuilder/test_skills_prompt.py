"""System messages pinned to bytes captured from the Go implementation."""

import base64
import json
from pathlib import Path
from typing import Any, Final

import pytest
from unreal_agent.contextbuilder import new_builder
from unreal_agent.llm import Message
from unreal_agent.tool import Skill

_CASES: Final = json.loads((Path(__file__).parent / "testdata" / "system_prompts.json").read_text())


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
