"""Model settings shared by harness applications.

`load(path)` reads the bundled `default_settings.json` and merges user
overrides from `path`, replacing models by ID within each provider. A missing
file leaves the defaults. An omitted or zero `compaction_threshold` defaults to
half the model's `context_window`.
"""

from unreal_agent.settings.settings import (
    Model,
    Provider,
    ProviderInfo,
    Settings,
    SettingsError,
    load,
)

__all__ = ["Model", "Provider", "ProviderInfo", "Settings", "SettingsError", "load"]
