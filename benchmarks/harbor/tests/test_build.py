import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from build import LAUNCHER_SCRIPT, entry_point


def wheel(path: Path, entry_points: str) -> Path:
    archive = path / "unreal_agent-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(archive, "w") as contents:
        contents.writestr("unreal_agent-0.1.0.dist-info/entry_points.txt", entry_points)
    return archive


def test_entry_point_is_read_from_the_wheel(tmp_path):
    archive = wheel(
        tmp_path, "[console_scripts]\nunreal-agent-runner = unreal_agent.runner.__main__:main\n"
    )
    assert entry_point(archive, "unreal-agent-runner") == ("unreal_agent.runner.__main__", "main")
    with pytest.raises(ValueError, match="no other console script"):
        entry_point(archive, "other")


def test_launcher_runs_the_entry_point_with_the_bundled_python(tmp_path):
    python = tmp_path / "python/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python.chmod(0o755)
    launcher = tmp_path / "unreal-agent-runner"
    launcher.write_text(
        LAUNCHER_SCRIPT.format(
            runner="unreal-agent-runner", module="unreal_agent.runner.__main__", function="main"
        )
    )
    launcher.chmod(0o755)
    result = subprocess.run(
        [launcher, "-h"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "/nonexistent"},
    )
    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout + result.stderr
