import dataclasses

from unreal_agent.llm.clients import fireworks, ollama, openai, openaicodex, openrouter
from unreal_agent.runner.config import Client, Getenv, Provider

__all__ = ["default_providers"]


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
        Provider(name="openai-codex", base_url=openaicodex.BASE_URL, new_client=_openai_codex),
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
    ]
