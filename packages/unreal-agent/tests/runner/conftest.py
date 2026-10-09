from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_user_directories(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the runner's XDG fallbacks away from the user's real settings and sessions.

    Session scoped so Hypothesis tests do not see a function-scoped fixture.
    Tests that exercise the fallbacks set HOME themselves.
    """
    home = tmp_path_factory.mktemp("home")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(home))
        patch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
        patch.setenv("XDG_STATE_HOME", str(home / ".local" / "state"))
        yield
