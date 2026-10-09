from collections.abc import Sequence

import pytest
from unreal_agent.llm import ToolCall
from unreal_agent.operation import Operation
from unreal_agent.tool import (
    BASH_NAME,
    SKILL_USE_NAME,
    VIEW_IMAGE_NAME,
    CallStatus,
    Context,
    Registry,
    Result,
    Skill,
    SkillUseResult,
    StaticTranslators,
)

ALL_NAMES = [BASH_NAME, VIEW_IMAGE_NAME, SKILL_USE_NAME]


class FixedTranslator:
    def translate(self, ctx: Context, call: ToolCall, /) -> CallStatus:
        return CallStatus()

    def translate_result(
        self, call_id: str, status: CallStatus, operations: Sequence[Operation], /
    ) -> Result:
        return SkillUseResult()


@pytest.mark.parametrize("selection", [0, 1, 6, 7])
def test_registry_keeps_selected_static_tools_across_catalog_changes(selection: int) -> None:
    enabled: list[str] = []
    want: list[str] = []
    for index, name in enumerate(ALL_NAMES):
        if selection & (1 << index):
            enabled += [name, name]
            want.append(name)
    enabled.reverse()
    registry = Registry(StaticTranslators(), *enabled)

    def assert_selection() -> None:
        assert [definition.tool.Name for definition in registry.static_definitions()] == want
        for name in ALL_NAMES:
            assert (registry.resolve(name) is not None) == (name in want), name

    assert_selection()
    enabled.clear()
    assert_selection()
    skill_id = registry.register_skill(
        Skill(name="review", description="Review code.", path="/skills/review/SKILL.md")
    )
    assert_selection()
    registry.unregister_skill(skill_id)
    assert_selection()


def test_registry_does_not_resolve_unselected_translators() -> None:
    registry = Registry(StaticTranslators(bash=FixedTranslator(), view_image=FixedTranslator()))
    assert registry.static_definitions() == []
    for name in ALL_NAMES:
        assert registry.resolve(name) is None
