from unreal_agent.llm import TokenUsage


def test_token_usage_add_includes_every_counter() -> None:
    names = TokenUsage.__struct_fields__
    left = TokenUsage(**{name: index + 1 for index, name in enumerate(names)})
    right = TokenUsage(**{name: 10 * (index + 1) for index, name in enumerate(names)})
    want = TokenUsage(**{name: 11 * (index + 1) for index, name in enumerate(names)})
    assert left.add(right) == want
