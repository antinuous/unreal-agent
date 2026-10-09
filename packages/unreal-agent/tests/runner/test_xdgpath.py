import pytest
from unreal_agent.runner import RunnerError
from unreal_agent.runner.xdgpath import config_directory, session_directory

from tests.runner._helpers import environment


@pytest.mark.parametrize(
    ("xdg", "home", "want"),
    [
        pytest.param("/config", "/home/user", "/config/unreal-agent", id="XDG config home"),
        pytest.param("/config", "", "/config/unreal-agent", id="XDG without home"),
        pytest.param("", "/home/user", "/home/user/.config/unreal-agent", id="home fallback"),
        pytest.param("", "", "/process/home/.config/unreal-agent", id="process home fallback"),
        pytest.param(
            "relative/config",
            "/home/user",
            "/home/user/.config/unreal-agent",
            id="relative XDG ignored",
        ),
    ],
)
def test_config_directory(xdg: str, home: str, want: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/process/home")
    assert config_directory(environment({"XDG_CONFIG_HOME": xdg, "HOME": home})) == want


@pytest.mark.parametrize("home", ["", "relative/home"])
def test_config_directory_rejects_invalid_home(home: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "")
    with pytest.raises(RunnerError, match="XDG_CONFIG_HOME"):
        config_directory(environment({"HOME": home}))


@pytest.mark.parametrize(
    ("state_home", "user_home", "process_home", "want"),
    [
        pytest.param(
            "/state", "/home/user", "", "/state/unreal-agent/sessions", id="XDG state home"
        ),
        pytest.param("/state", "", "", "/state/unreal-agent/sessions", id="XDG without home"),
        pytest.param(
            "",
            "/home/user",
            "/process/home",
            "/home/user/.local/state/unreal-agent/sessions",
            id="home fallback",
        ),
        pytest.param(
            "",
            "",
            "/process/home",
            "/process/home/.local/state/unreal-agent/sessions",
            id="process home fallback",
        ),
        pytest.param(
            "relative/state",
            "/home/user",
            "",
            "/home/user/.local/state/unreal-agent/sessions",
            id="relative XDG ignored",
        ),
        pytest.param("relative/state", "", "", None, id="missing home"),
        pytest.param("", "relative/home", "/process/home", None, id="relative home"),
    ],
)
def test_session_directory(
    state_home: str,
    user_home: str,
    process_home: str,
    want: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", process_home)
    getenv = environment({"XDG_STATE_HOME": state_home, "HOME": user_home})
    if want is None:
        with pytest.raises(RunnerError, match="XDG_STATE_HOME"):
            session_directory(getenv)
    else:
        assert session_directory(getenv) == want
