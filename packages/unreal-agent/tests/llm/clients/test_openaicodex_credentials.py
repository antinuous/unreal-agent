import base64
import json
import time
from pathlib import Path

import pytest
from unreal_agent.llm.clients.openaicodex import Config, environment_config
from unreal_agent.llm.clients.openaicodex.credentials import credentials

from tests.llm.clients._helpers import write_test_auth


def subscription_token(account: str, expires: int) -> str:
    claims = json.dumps(
        {"exp": expires, "https://api.openai.com/auth": {"chatgpt_account_id": account}},
        separators=(",", ":"),
    )
    payload = base64.urlsafe_b64encode(claims.encode()).rstrip(b"=").decode()
    return f"e30.{payload}.signature"


FUTURE = int(time.time()) + 3600
PAST = int(time.time()) - 3600


@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param(Config(access_token="opaque", account_id="account"), "", id="explicit"),
        pytest.param(Config(access_token=subscription_token("account", FUTURE)), "", id="claims"),
        pytest.param(
            Config(access_token=subscription_token("account", FUTURE), account_id="account"),
            "",
            id="matching account",
        ),
        pytest.param(Config(), "access token must be set", id="missing token"),
        pytest.param(Config(access_token="opaque"), "account ID must be set", id="missing account"),
        pytest.param(
            Config(access_token="sk-secret", account_id="account"), "not an API key", id="API key"
        ),
        pytest.param(
            Config(access_token=subscription_token("account", PAST)), "expired", id="expired"
        ),
        pytest.param(
            Config(access_token=subscription_token("other", FUTURE), account_id="account"),
            "does not match",
            id="mismatched account",
        ),
        pytest.param(
            Config(access_token="e30.invalid.payload", account_id="account"),
            "invalid Codex",
            id="invalid claims",
        ),
        pytest.param(
            Config(access_token="opaque\r\nX-Bad: value", account_id="account"),
            "invalid header",
            id="header injection",
        ),
        pytest.param(
            Config(access_token="opaque", account_id="account\nX-Bad:value"),
            "account ID",
            id="account injection",
        ),
        pytest.param(
            Config(access_token="opaque", auth_file="unused"),
            "cannot be combined",
            id="conflicting sources",
        ),
    ],
)
def test_credentials(config: Config, match: str) -> None:
    if match:
        with pytest.raises(ValueError, match=match) as caught:
            credentials(config)
        assert "sk-secret" not in str(caught.value)
        assert "e30." not in str(caught.value)
        return
    got = credentials(config)
    assert got.account_id == "account"
    assert got.access_token == config.access_token


def test_read_auth_file(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    write_test_auth(path, "opaque", "account")
    before = path.read_bytes()
    got = credentials(Config(auth_file=str(path)))
    assert (got.access_token, got.account_id) == ("opaque", "account")
    assert path.read_bytes() == before, "changed external auth file"


@pytest.mark.parametrize(
    ("body", "match", "mode"),
    [
        pytest.param(None, "private permissions", 0o644, id="public"),
        pytest.param("{bad-secret}", "invalid Codex auth file", 0o600, id="invalid"),
        pytest.param(
            '{"auth_mode":"apikey","OPENAI_API_KEY":"sk-secret",'
            '"tokens":{"access_token":"opaque","account_id":"account"}}',
            "not a ChatGPT",
            0o600,
            id="API-key mode",
        ),
        pytest.param(
            '{"OPENAI_API_KEY":"sk-secret"}', "access token must be set", 0o600, id="missing tokens"
        ),
        pytest.param(" " * (1 << 20) + "{}", "invalid Codex auth file", 0o600, id="too large"),
    ],
)
def test_read_auth_file_rejects(tmp_path: Path, body: str | None, match: str, mode: int) -> None:
    path = tmp_path / "auth.json"
    write_test_auth(path, "opaque", "account")
    if body is not None:
        path.write_text(body)
    path.chmod(mode)
    with pytest.raises(ValueError, match=match) as caught:
        credentials(Config(auth_file=str(path)))
    assert "secret" not in str(caught.value)


def test_read_auth_file_rejects_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="open Codex auth file"):
        credentials(Config(auth_file=str(tmp_path / "missing")))
    with pytest.raises(ValueError, match="regular file"):
        credentials(Config(auth_file=str(tmp_path)))


@pytest.mark.parametrize(
    ("process_home", "env", "file", "token", "account", "invalid"),
    [
        pytest.param(
            "/process/home",
            {"HOME": " /injected/home "},
            "/injected/home/.codex/auth.json",
            "",
            "",
            False,
            id="injected home takes precedence",
        ),
        pytest.param(
            "",
            {
                "HOME": "/home/example",
                "OPENAI_API_KEY": "sk-api",
                "UNREAL_HARNESS_LLM_API_KEY": "sk-generic",
            },
            "/home/example/.codex/auth.json",
            "",
            "",
            False,
            id="home default",
        ),
        pytest.param(
            "/process/home",
            {},
            "/process/home/.codex/auth.json",
            "",
            "",
            False,
            id="process home fallback",
        ),
        pytest.param("", {}, "", "", "", True, id="missing home"),
        pytest.param(
            "",
            {"HOME": "//home/./user/"},
            "/home/user/.codex/auth.json",
            "",
            "",
            False,
            id="home cleaned like go",
        ),
        pytest.param(
            "",
            {"CODEX_HOME": "//custom/codex/"},
            "/custom/codex/auth.json",
            "",
            "",
            False,
            id="codex home cleaned like go",
        ),
        pytest.param(
            "",
            {"OPENAI_CODEX_AUTH_FILE": "/custom/auth.json"},
            "/custom/auth.json",
            "",
            "",
            False,
            id="explicit file",
        ),
        pytest.param(
            "",
            {"OPENAI_CODEX_ACCESS_TOKEN": " token ", "OPENAI_CODEX_ACCOUNT_ID": " account "},
            "",
            "token",
            "account",
            False,
            id="explicit token",
        ),
        pytest.param(
            "",
            {"OPENAI_CODEX_ACCOUNT_ID": "account"},
            "",
            "",
            "account",
            False,
            id="partial token configuration",
        ),
        pytest.param(
            "",
            {"OPENAI_CODEX_AUTH_FILE": "/file", "OPENAI_CODEX_ACCESS_TOKEN": "token"},
            "",
            "",
            "",
            True,
            id="conflict",
        ),
    ],
)
def test_environment_config(
    monkeypatch: pytest.MonkeyPatch,
    process_home: str,
    env: dict[str, str],
    file: str,
    token: str,
    account: str,
    invalid: bool,
) -> None:
    monkeypatch.setenv("HOME", process_home)
    if invalid:
        with pytest.raises(ValueError, match=r"choose OPENAI_CODEX_AUTH_FILE|find Codex home"):
            environment_config(env.get)
        return
    config = environment_config(env.get)
    assert (config.auth_file, config.access_token, config.account_id) == (file, token, account)
