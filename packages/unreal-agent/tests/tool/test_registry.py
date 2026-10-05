from collections.abc import Sequence

import pytest
from unreal_agent.llm import Tool, ToolCall, ToolResult, ToolType
from unreal_agent.operation import Operation
from unreal_agent.tool import (
    BASH_NAME,
    NIL_REGISTRATION_ID,
    SKILL_USE_NAME,
    VIEW_IMAGE_NAME,
    CallStatus,
    Context,
    Registry,
    Skill,
    SkillUseTranslator,
    StaticTranslators,
)


class FixedTranslator:
    def __init__(self) -> None:
        self.status = CallStatus()
        self.result = ToolResult()

    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return self.status

    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> ToolResult:
        return self.result


class FalsyTranslator(FixedTranslator):
    def __len__(self) -> int:
        return 0


def test_registry_keeps_injected_translators_that_are_falsy() -> None:
    translators = StaticTranslators(bash=FalsyTranslator(), view_image=FalsyTranslator())
    registry = Registry(translators, BASH_NAME, VIEW_IMAGE_NAME)
    assert registry.resolve(BASH_NAME) is translators.bash
    assert registry.resolve(VIEW_IMAGE_NAME) is translators.view_image


def test_registry_has_fixed_definitions_and_injected_translators() -> None:
    translators = StaticTranslators(bash=FixedTranslator(), view_image=FixedTranslator())
    registry = Registry(translators, BASH_NAME, VIEW_IMAGE_NAME, SKILL_USE_NAME)
    names = [definition.tool.Name for definition in registry.static_definitions()]
    assert names == [BASH_NAME, VIEW_IMAGE_NAME, SKILL_USE_NAME]
    assert registry.resolve(BASH_NAME) is translators.bash
    assert registry.resolve(VIEW_IMAGE_NAME) is translators.view_image
    assert isinstance(registry.resolve(SKILL_USE_NAME), SkillUseTranslator)
    assert registry.resolve("unknown") is None


def test_registry_owns_canonical_bash_definition() -> None:
    got = Registry(StaticTranslators(), BASH_NAME).static_definitions()[0].tool
    want = Tool(
        Type=ToolType.FUNCTION,
        Name=BASH_NAME,
        Description=(
            "Execute a shell command in background. Independent commands may be issued as "
            "parallel tool calls in one turn. Command child processes are killed when the "
            "shell exits."
        ),
        Parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "max_output_length": {
                    "type": "integer",
                    "description": (
                        "Maximum characters per output text field. Truncated text keeps its "
                        "head and tail, around a marker stating how much was omitted, and path "
                        "to the file with the complete stream. Defaults to 40000."
                    ),
                    "minimum": 1,
                    "maximum": 1000000,
                    "default": 40000,
                },
            },
            "required": ["command"],
        },
    )
    assert got == want


def test_registry_owns_canonical_skill_use_definition() -> None:
    got = Registry(StaticTranslators(), SKILL_USE_NAME).static_definitions()[0].tool
    want = Tool(
        Type=ToolType.FUNCTION,
        Name=SKILL_USE_NAME,
        Description="Load the instructions for a registered skill.",
        Parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The exact name of the skill to load."},
            },
            "required": ["name"],
        },
    )
    assert got == want


def test_registry_static_definitions_returns_independent_values() -> None:
    registry = Registry(StaticTranslators(), BASH_NAME)
    definitions = registry.static_definitions()
    definitions[0].tool.Name = "changed"
    definitions[0].tool.Parameters["changed"] = True
    got = registry.static_definitions()[0].tool
    assert got.Name == BASH_NAME
    assert "changed" not in got.Parameters


def test_registry_registers_lists_and_unregisters_skills() -> None:
    registry = Registry(StaticTranslators())
    first = Skill(name="go-review", description="Review Go code", path="/skills/go/SKILL.md")
    second = Skill(name="documents", description="Edit documents", path="/skills/docs/SKILL.md")
    first_id = registry.register_skill(first)
    second_id = registry.register_skill(second)
    assert registry.skills() == [first, second]

    copy_of_skills = registry.skills()
    copy_of_skills[0] = Skill(name=first.name, description=first.description, path="changed")
    assert registry.skills()[0] == first

    registry.unregister_skill(first_id)
    registry.unregister_skill(first_id)
    registry.unregister_skill(NIL_REGISTRATION_ID)
    assert registry.skills() == [second]
    assert first_id != second_id


def test_registry_rejects_invalid_and_duplicate_skills() -> None:
    registry = Registry(StaticTranslators())
    with pytest.raises(ValueError, match=r"^skill path must be set$"):
        registry.register_skill(Skill())
    with pytest.raises(ValueError, match=r"^skill name must be set$"):
        registry.register_skill(Skill(path="/skill"))
    with pytest.raises(ValueError, match=r"^skill description must be set$"):
        registry.register_skill(Skill(name="review", path="/skill"))
    skill = Skill(name="review", description="Review code", path="/skill")
    registry.register_skill(skill)
    with pytest.raises(ValueError, match=r'^skill path "/skill" is already registered$'):
        registry.register_skill(skill)
    with pytest.raises(ValueError, match=r'^skill name "review" is already registered$'):
        registry.register_skill(
            Skill(name="review", description="Review other code", path="/other-skill")
        )
