import asyncio
from pathlib import Path
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harness_harbor.agent import APT_ARCHIVE_FALLBACK, UnrealAgent
from harness_harbor.bundle import ARCHIVE, LAUNCHER


class ScriptedEnvironment:
    """Debian bullseye image without curl whose default apt mirror has moved."""

    def __init__(self, mirror_broken: bool) -> None:
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.curl_installed = False
        self.mirror_broken = mirror_broken

    async def exec(self, command: str, user: str | None = None, **_: object) -> ExecResult:
        self.commands.append(command)
        if "command -v" in command and "curl" in command:
            return ExecResult(return_code=0 if self.curl_installed else 1)
        if "for manager in apt-get" in command:
            return ExecResult(stdout="apt-get", return_code=0)
        if "archive.debian.org" in command:
            self.mirror_broken = False
            return ExecResult(return_code=0)
        if "apt-get install" in command:
            if self.mirror_broken:
                return ExecResult(stderr="E: Unable to fetch some archives", return_code=100)
            self.curl_installed = True
            return ExecResult(return_code=0)
        if "mkdir -p" in command or "tar -xzf" in command:
            return ExecResult(return_code=0)
        raise AssertionError(f"unexpected command: {command}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads[target_path] = await asyncio.to_thread(Path(source_path).read_bytes)


def environment(mirror_broken: bool = False) -> tuple[ScriptedEnvironment, BaseEnvironment]:
    scripted = ScriptedEnvironment(mirror_broken)
    return scripted, cast(BaseEnvironment, scripted)


@pytest.fixture
def agent(make_bundle) -> UnrealAgent:
    path = make_bundle(data=b"bundle archive")
    return UnrealAgent(bundle=str(path), logs_dir=path, model_name="openai/test")


async def test_install_extracts_the_verified_archive_and_checks_the_runner(agent):
    scripted, env = environment()
    await agent.install(env)
    remote = "/installed-agent/unreal-agent"
    assert scripted.uploads == {f"{remote}/{ARCHIVE}": b"bundle archive"}
    [extract] = [c for c in scripted.commands if "tar -xzf" in c]
    assert f"tar -xzf {remote}/{ARCHIVE} -C {remote}" in extract
    assert extract.endswith(f"{remote}/{LAUNCHER} -h")


async def test_curl_is_installed_before_the_agent_runs(agent):
    scripted, env = environment()
    await agent.ensure_curl(env)
    assert scripted.curl_installed
    assert not any("archive.debian.org" in c for c in scripted.commands)


async def test_moved_apt_mirror_falls_back_to_the_archive(agent):
    scripted, env = environment(mirror_broken=True)
    await agent.ensure_curl(env)
    assert scripted.curl_installed
    installs = [i for i, c in enumerate(scripted.commands) if "apt-get install" in c]
    fallback = [i for i, c in enumerate(scripted.commands) if APT_ARCHIVE_FALLBACK in c]
    assert len(installs) == 2
    assert len(fallback) == 1
    assert installs[0] < fallback[0] < installs[1]
