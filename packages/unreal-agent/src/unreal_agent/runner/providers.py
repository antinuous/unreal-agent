import dataclasses
from collections.abc import Callable

from unreal_agent._go import go_quote
from unreal_agent.llm.clients import anthropic, fireworks, ollama, openai, openaicodex, openrouter
from unreal_agent.runner.config import Client, Getenv, Provider
from unreal_agent.runner.errors import RunnerError

__all__ = ["default_providers", "find_provider"]


def _anthropic(api_key: str, base_url: str, max_attempts: int, _: Getenv) -> Client:
    return anthropic.Client(
        anthropic.Config(api_key=api_key, base_url=base_url, max_attempts=max_attempts)
    )


def _ollama(_: str, base_url: str, max_attempts: int, __: Getenv) -> Client:
    return ollama.Client(ollama.Config(base_url=base_url, max_attempts=max_attempts))


def _openai(api_key: str, base_url: str, max_attempts: int, _: Getenv) -> Client:
    return openai.Client(
        openai.Config(api_key=api_key, base_url=base_url, max_attempts=max_attempts)
    )


def _openai_codex(_: str, base_url: str, max_attempts: int, getenv: Getenv) -> Client:
    config = openaicodex.environment_config(getenv)
    return openaicodex.Client(
        dataclasses.replace(config, base_url=base_url, max_attempts=max_attempts)
    )


def _openrouter(api_key: str, base_url: str, max_attempts: int, _: Getenv) -> Client:
    return openrouter.Client(
        openrouter.Config(api_key=api_key, base_url=base_url, max_attempts=max_attempts)
    )


def _fireworks(api_key: str, base_url: str, max_attempts: int, _: Getenv) -> Client:
    return fireworks.Client(
        fireworks.Config(api_key=api_key, base_url=base_url, max_attempts=max_attempts)
    )


def default_providers() -> list[Provider]:
    return [
        Provider(name="ollama", base_url=ollama.BASE_URL, new_client=_ollama),
        Provider(
            name="openai",
            base_url="https://api.openai.com/v1",
            default_model="gpt-6-astra",
            api_key_environment="OPENAI_API_KEY",
            new_client=_openai,
        ),
        Provider(
            name="openai-codex",
            base_url=openaicodex.BASE_URL,
            default_model="gpt-6-astra",
            new_client=_openai_codex,
        ),
        Provider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_environment="OPENROUTER_API_KEY",
            new_client=_openrouter,
        ),
        Provider(
            name="fireworks",
            base_url="https://api.fireworks.ai/inference/v1",
            api_key_environment="FIREWORKS_API_KEY",
            new_client=_fireworks,
        ),
        Provider(
            name="anthropic",
            base_url=anthropic.DEFAULT_BASE_URL,
            default_model="claude-opus-5-5",
            api_key_environment="ANTHROPIC_API_KEY",
            new_client=_anthropic,
        ),
    ]


def find_provider(
    available: list[Provider], name: str
) -> tuple[Provider, Callable[[str, str, int, Getenv], Client]]:
    """Raises RunnerError when `name` is unknown or has no client factory."""
    for provider in available:
        if provider.name == name:
            if provider.new_client is None:
                raise RunnerError(f"provider {go_quote(name)} has no client factory")
            return provider, provider.new_client
    names = ", ".join(provider.name for provider in available)
    raise RunnerError(f"unsupported provider {go_quote(name)}; available providers: {names}")
