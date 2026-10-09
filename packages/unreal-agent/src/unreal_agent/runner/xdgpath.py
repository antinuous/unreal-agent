import os

from unreal_agent._filepath import go_join
from unreal_agent.runner.config import Getenv
from unreal_agent.runner.errors import RunnerError

__all__ = ["config_directory", "session_directory"]


def config_directory(getenv: Getenv) -> str:
    """The `unreal-agent` directory under the XDG config home. Raises RunnerError."""
    return _directory(getenv, "XDG_CONFIG_HOME", ".config", "unreal-agent")


def session_directory(getenv: Getenv) -> str:
    """The `unreal-agent/sessions` directory under the XDG state home. Raises RunnerError."""
    return _directory(getenv, "XDG_STATE_HOME", ".local/state", "unreal-agent/sessions")


def _directory(getenv: Getenv, variable: str, fallback: str, subdirectory: str) -> str:
    if os.path.isabs(root := getenv(variable)):
        return go_join(root, subdirectory)
    home = getenv("HOME") or os.environ.get("HOME", "")
    if home == "":
        raise RunnerError(
            f"find home directory: $HOME is not defined; set an absolute {variable} or HOME"
        )
    if not os.path.isabs(home):
        raise RunnerError(f"set an absolute {variable} or HOME")
    return go_join(home, fallback, subdirectory)
