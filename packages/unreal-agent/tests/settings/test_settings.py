from pathlib import Path

import pytest
from unreal_agent.settings import Model, Provider, ProviderInfo, Settings, SettingsError, load


def _model(id: str, name: str, window: int, threshold: int) -> Model:
    return Model(id=id, name=name, context_window=window, compaction_threshold=threshold)


def _openai_models() -> list[Model]:
    return [
        _model("gpt-6-astra", "GPT-6 Astra", 1_050_000, 244_800),
        _model("gpt-6.1-sol", "GPT-6.1 Sol", 1_050_000, 244_800),
        _model("gpt-6-luna", "GPT-6 Luna", 1_050_000, 244_800),
        _model("gpt-6-sol", "GPT-6 Sol", 1_050_000, 244_800),
        _model("gpt-5.6-sol", "GPT-5.6 Sol", 1_050_000, 244_800),
        _model("gpt-5.6-terra", "GPT-5.6 Terra", 1_050_000, 244_800),
        _model("gpt-5.6-luna", "GPT-5.6 Luna", 1_050_000, 244_800),
    ]


_OPENAI_ONLY_MODELS = [_model("gpt-5.6-cyber", "GPT-5.6 Cyber", 400_000, 244_800)]


def builtin() -> Settings:
    return Settings(
        providers={
            "anthropic": Provider(
                info=ProviderInfo(id="anthropic", name="Anthropic"),
                models=[
                    _model("claude-opus-5-5", "Claude Opus 5.5", 1_000_000, 500_000),
                    _model("claude-opus-5", "Claude Opus 5", 1_000_000, 500_000),
                    _model("claude-sonnet-5-5", "Claude Sonnet 5.5", 1_000_000, 500_000),
                    _model("claude-fable-5-1", "Claude Fable 5.1", 1_000_000, 500_000),
                    _model("claude-fable-5", "Claude Fable 5", 1_000_000, 500_000),
                    _model("claude-mythos-5-1", "Claude Mythos 5.1", 1_000_000, 500_000),
                    _model("claude-mythos-5", "Claude Mythos 5", 1_000_000, 500_000),
                ],
            ),
            "fireworks": Provider(
                info=ProviderInfo(id="fireworks", name="Fireworks"),
                models=[
                    _model("accounts/fireworks/models/glm-5p3", "GLM-5.3", 1_048_576, 524_288),
                    _model(
                        "accounts/fireworks/models/glm-5p3-flash",
                        "GLM-5.3 Flash",
                        1_048_576,
                        524_288,
                    ),
                ],
            ),
            "openai": Provider(
                info=ProviderInfo(id="openai", name="OpenAI"),
                models=_openai_models() + _OPENAI_ONLY_MODELS,
            ),
            "openai-codex": Provider(
                info=ProviderInfo(id="openai-codex", name="OpenAI Codex"),
                models=_openai_models(),
            ),
        }
    )


def test_load(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    assert load(str(path)) == builtin(), "missing settings"
    path.write_text(
        '{"providers":{"openai":{"info":{"id":"openai","name":"OpenAI"},"models":['
        '{"id":"gpt-6-astra","name":"Custom Astra","context_window":200000,'
        '"compaction_threshold":150000},'
        '{"id":"custom","name":"Custom model","context_window":150000},'
        '{"id":"zero","name":"Zero threshold","context_window":100000,"compaction_threshold":0}'
        "]}}}"
    )
    configured = load(str(path))
    want = builtin()
    assert want.providers is not None
    want.providers["openai"] = Provider(
        info=ProviderInfo(id="openai", name="OpenAI"),
        models=[
            _model("gpt-6-astra", "Custom Astra", 200_000, 150_000),
            *_openai_models()[1:],
            *_OPENAI_ONLY_MODELS,
            _model("custom", "Custom model", 150_000, 75_000),
            _model("zero", "Zero threshold", 100_000, 50_000),
        ],
    )
    assert configured == want
    assert configured.providers is not None
    models = configured.providers["anthropic"].models
    assert models is not None
    models[0] = Model(compaction_threshold=1)
    path.unlink()
    assert load(str(path)) == builtin(), "overrides changed built-in settings"


@pytest.mark.parametrize(
    ("window", "threshold"),
    [(1, 0), (100_000, 50_000), (100_001, 50_000), (2**63 - 1, 4_611_686_018_427_387_903)],
)
def test_load_defaults_threshold_to_half_context_window(
    window: int, threshold: int, tmp_path: Path
) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        '{"providers":{"openai":{"info":{"id":"openai","name":"OpenAI"},"models":[{"id":'
        f'"gpt-6-astra","name":"Custom Astra","context_window":{window}}}]}}}}}}'
    )
    assert load(str(path)).model("openai", "gpt-6-astra").compaction_threshold == threshold


_OPENAI = '{"providers":{"openai":{"info":{"id":"openai","name":"OpenAI"},"models":%s}}}'


@pytest.mark.parametrize(
    "contents",
    [
        "",
        '{"providers":{"openai":{"info":{"id":"openai","name":"OpenAI"},"models":}}',
        _OPENAI % '{"other":{"compaction_threshold":1000}}',
        _OPENAI % '[{"name":"Other","context_window":100000}]',
        _OPENAI % '[{"id":" ","name":"Other","context_window":100000}]',
        _OPENAI % '[{"id":"other","context_window":100000}]',
        _OPENAI % '[{"id":"other","name":" ","context_window":100000}]',
        _OPENAI % '[{"id":"other","name":"Other"}]',
        _OPENAI % '[{"id":"other","name":"Other","context_window":0}]',
        _OPENAI % '[{"id":"other","name":"Other","context_window":-1}]',
        _OPENAI % '[{"id":"other","name":"Other","context_window":1.5}]',
        _OPENAI % '[{"id":"other","name":"Other","context_window":"100000"}]',
        _OPENAI % '[{"id":"other","name":"Other","context_window":9223372036854775808}]',
        _OPENAI
        % '[{"id":"other","name":"Other","context_window":100000,"compaction_threshold":-1}]',
        _OPENAI
        % '[{"id":"other","name":"Other","context_window":100000,"compaction_threshold":1.5}]',
        _OPENAI
        % '[{"id":"other","name":"Other","context_window":100000,"compaction_threshold":"1000"}]',
        _OPENAI % '[{"id":"other","name":"Other","context_window":100000,'
        '"compaction_threshold":9223372036854775808}]',
        _OPENAI % '[{"id":"other","name":"First","context_window":100000},'
        '{"id":"other","name":"Second","context_window":200000}]',
    ],
)
def test_load_rejects_invalid_models(contents: str, tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(contents)
    with pytest.raises(SettingsError):
        load(str(path))


def test_load_merges_models_within_providers(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"providers":{'
        '"openai":{"info":{"id":"openai","name":"Custom OpenAI"},"models":['
        '{"id":"custom","name":"Custom","context_window":200000}]},'
        '"other":{"info":{"id":"other","name":"Other"},"models":['
        '{"id":"gpt-6-astra","name":"Other Astra","context_window":100000},'
        '{"id":"custom","name":"Other custom","context_window":80000}]}'
        "}}"
    )
    configured = load(str(path))
    for provider, model, threshold in [
        ("openai", "gpt-6-astra", 244_800),
        ("openai-codex", "gpt-6-astra", 244_800),
        ("openai", "gpt-6.1-sol", 244_800),
        ("openai-codex", "gpt-6.1-sol", 244_800),
        ("openai", "gpt-6-luna", 244_800),
        ("openai-codex", "gpt-6-luna", 244_800),
        ("other", "gpt-6-astra", 50_000),
        ("openai", "custom", 100_000),
        ("other", "custom", 40_000),
        ("missing", "gpt-6-astra", 0),
        ("openai", "missing", 0),
    ]:
        got = configured.model(provider, model).compaction_threshold
        assert got == threshold, f"{provider}/{model} threshold = {got}, want {threshold}"
    assert configured.providers is not None
    assert configured.providers["openai"].info.name == "Custom OpenAI"


@pytest.mark.parametrize(
    "contents",
    [
        '{"providers":[]}',
        '{"providers":{"": {"info":{"id":"","name":"Empty"}}}}',
        '{"providers":{"other": {"models":[]}}}',
        '{"providers":{"other": {"info":{"name":"Other"}}}}',
        '{"providers":{"other": {"info":{"id":"different","name":"Other"}}}}',
        '{"providers":{"other": {"info":{"id":"other"}}}}',
        '{"providers":{"other": {"info":{"id":"other","name":" "}}}}',
    ],
)
def test_load_rejects_invalid_providers(contents: str, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(contents)
    with pytest.raises(SettingsError):
        load(str(path))


def test_load_reports_read_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match=r"^read user settings .*: read .*: is a directory$"):
        load(str(tmp_path))
