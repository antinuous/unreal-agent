from collections.abc import Sequence
from importlib.resources import files
from typing import Final

from unreal_agent.tool import Skill
from unreal_agent.tool._go import decode_rune, go_bytes, go_trim_space

__all__ = ["format_skills_for_prompt", "read_prompt"]

_XML_ESCAPES: Final = {
    0x22: "&#34;",
    0x27: "&#39;",
    0x26: "&amp;",
    0x3C: "&lt;",
    0x3E: "&gt;",
    0x09: "&#x9;",
    0x0A: "&#xA;",
    0x0D: "&#xD;",
}


def read_prompt(name: str) -> str:
    """Return a bundled prompt with surrounding whitespace removed, like Go's TrimSpace."""
    text = files("unreal_agent.contextbuilder").joinpath("prompts", name).read_text("utf-8")
    return go_trim_space(text)


def _in_character_range(rune: int) -> bool:
    return (
        rune in (0x09, 0x0A, 0x0D)
        or 0x20 <= rune <= 0xD7FF
        or 0xE000 <= rune <= 0xFFFD
        or 0x10000 <= rune <= 0x10FFFF
    )


def _escape_text(text: str) -> str:
    """encoding/xml EscapeText: invalid UTF-8 and non-XML characters become U+FFFD."""
    data = go_bytes(text)
    out: list[str] = []
    index = 0
    while index < len(data):
        rune, size = decode_rune(data, index)
        index += size
        escaped = _XML_ESCAPES.get(rune)
        if escaped is not None:
            out.append(escaped)
        elif not _in_character_range(rune) or (rune == 0xFFFD and size == 1):
            out.append("\ufffd")
        else:
            out.append(chr(rune))
    return "".join(out)


def _element(name: str, text: str) -> str:
    return f"<{name}>{_escape_text(text)}</{name}>"


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """The skill preamble and an `<available_skills>` block, byte-identical to Go's xml.Marshal."""
    if not skills:
        return ""
    entries = "".join(
        "<skill>"
        + _element("name", skill.name)
        + _element("description", skill.description)
        + _element("location", skill.path)
        + "</skill>"
        for skill in skills
    )
    return (
        read_prompt("skill-preamble.md")
        + "\n\n<available_skills>"
        + entries
        + "</available_skills>"
    )
