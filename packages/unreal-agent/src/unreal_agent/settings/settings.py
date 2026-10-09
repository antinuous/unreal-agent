from importlib.resources import files

from msgspec import field

from unreal_agent._go import go_quote, go_trim_space
from unreal_agent.gojson import Int64, JSONError, Struct, decode
from unreal_agent.sessionstore.localfile._errors import message

__all__ = ["Model", "Provider", "ProviderInfo", "Settings", "SettingsError", "load"]


class SettingsError(Exception):
    pass


class ProviderInfo(Struct):
    id: str = ""
    name: str = ""


class Model(Struct):
    id: str = ""
    name: str = ""
    context_window: Int64 = 0
    compaction_threshold: Int64 = 0


class Provider(Struct):
    info: ProviderInfo = field(default_factory=ProviderInfo)
    models: list[Model] | None = None


class Settings(Struct):
    providers: dict[str, Provider] | None = None

    def model(self, provider_id: str, model_id: str) -> Model:
        provider = (self.providers or {}).get(provider_id) or Provider()
        return next((model for model in provider.models or [] if model.id == model_id), Model())


def load(path: str) -> Settings:
    """Load the built-in settings with user overrides from `path`. Raises SettingsError."""
    try:
        configured = _decode(
            files("unreal_agent.settings").joinpath("default_settings.json").read_bytes()
        )
    except (JSONError, SettingsError) as error:
        raise SettingsError(f"decode built-in settings: {error}") from error
    try:
        with open(path, "rb") as file:
            data = file.read()
    except FileNotFoundError:
        return configured
    except OSError as error:
        op = "read" if isinstance(error, IsADirectoryError) else "open"
        raise SettingsError(f"read user settings {go_quote(path)}: {message(error, op)}") from error
    try:
        overrides = _decode(data)
    except (JSONError, SettingsError) as error:
        raise SettingsError(f"decode user settings {go_quote(path)}: {error}") from error
    providers = configured.providers if configured.providers is not None else {}
    for id, override in (overrides.providers or {}).items():
        provider = providers.get(id) or Provider()
        provider.info = override.info
        models = provider.models if provider.models is not None else []
        for model in override.models or []:
            index = next(
                (index for index, current in enumerate(models) if current.id == model.id), -1
            )
            if index == -1:
                models.append(model)
            else:
                models[index] = model
        provider.models = models
        providers[id] = provider
    configured.providers = providers
    return configured


def _decode(data: bytes) -> Settings:
    config = decode(data, Settings)
    for id, provider in (config.providers or {}).items():
        if not go_trim_space(id) or provider.info.id != id:
            raise SettingsError(f"provider {go_quote(id)} info.id must match its non-empty key")
        if not go_trim_space(provider.info.name):
            raise SettingsError(f"provider {go_quote(id)} name must not be empty")
        seen: set[str] = set()
        for model in provider.models or []:
            if not go_trim_space(model.id):
                raise SettingsError(f"provider {go_quote(id)} model id must not be empty")
            if not go_trim_space(model.name):
                raise SettingsError(
                    f"provider {go_quote(id)} model {go_quote(model.id)} name must not be empty"
                )
            if model.context_window <= 0:
                raise SettingsError(
                    f"provider {go_quote(id)} model {go_quote(model.id)} "
                    "context_window must be positive"
                )
            if model.id in seen:
                raise SettingsError(
                    f"provider {go_quote(id)} duplicate model id {go_quote(model.id)}"
                )
            seen.add(model.id)
            if model.compaction_threshold < 0:
                raise SettingsError(
                    f"provider {go_quote(id)} model {go_quote(model.id)} "
                    "compaction_threshold must not be negative"
                )
            if model.compaction_threshold == 0:
                model.compaction_threshold = model.context_window // 2
    return config
