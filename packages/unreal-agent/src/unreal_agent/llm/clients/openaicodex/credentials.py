import base64
import os
import re
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import msgspec

from unreal_agent._filepath import go_join
from unreal_agent.gojson import Int64, JSONError
from unreal_agent.llm.clients.openaicodex.config import Config
from unreal_agent.llm.responsesapi._wire import decode_object, trim_space_text
from unreal_agent.primitives._errors import describe

__all__ = ["Credentials", "environment_config"]

_AUTH_FILE_LIMIT: Final = 1 << 20
_BASE64_URL: Final = re.compile(r"[A-Za-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class Credentials:
    access_token: str
    account_id: str


def environment_config(getenv: Callable[[str], str | None] = os.getenv) -> Config:
    """Select a credential source without reading files or consulting API keys."""

    def env(key: str) -> str:
        return trim_space_text(getenv(key) or "")

    config = Config(
        access_token=env("OPENAI_CODEX_ACCESS_TOKEN"),
        account_id=env("OPENAI_CODEX_ACCOUNT_ID"),
        auth_file=env("OPENAI_CODEX_AUTH_FILE"),
    )
    if config.auth_file and (config.access_token or config.account_id):
        raise ValueError(
            "choose OPENAI_CODEX_AUTH_FILE or OPENAI_CODEX_ACCESS_TOKEN/OPENAI_CODEX_ACCOUNT_ID, "
            "not both"
        )
    if config.access_token or config.account_id or config.auth_file:
        return config
    home = env("CODEX_HOME")
    if not home:
        user_home = env("HOME")
        if not user_home:
            user_home = os.environ.get("HOME", "")
            if not user_home:
                raise ValueError("find Codex home: $HOME is not defined")
        home = go_join(user_home, ".codex")
    config.auth_file = go_join(home, "auth.json")
    return config


def credentials(config: Config) -> Credentials:
    token, account_id = trim_space_text(config.access_token), trim_space_text(config.account_id)
    if config.auth_file:
        if token or account_id:
            raise ValueError("codex AuthFile cannot be combined with AccessToken or AccountID")
        token, account_id = _read_auth_file(config.auth_file)
    if not token:
        raise ValueError(
            "codex access token must be set; use OPENAI_CODEX_ACCESS_TOKEN or a "
            "ChatGPT-authenticated Codex auth file"
        )
    if token.startswith("sk-") or not _header_value(token):
        raise ValueError(
            "codex requires a subscription access token, not an API key or invalid header value"
        )
    claim_account, expires = _token_claims(token)
    if expires != 0 and int(time.time()) >= expires:
        raise ValueError(
            "codex access token has expired; renew credentials externally "
            "(interactive login and token refresh are not implemented)"
        )
    if not account_id:
        account_id = claim_account
    elif claim_account and account_id != claim_account:
        raise ValueError("codex account ID does not match the access token")
    if not account_id or not _header_value(account_id):
        raise ValueError(
            "codex account ID must be set in OPENAI_CODEX_ACCOUNT_ID, the auth file, "
            "or the access token"
        )
    return Credentials(access_token=token, account_id=account_id)


class _Tokens(msgspec.Struct):
    access_token: str | None = None
    account_id: str | None = None


class _AuthFile(msgspec.Struct):
    auth_mode: str | None = None
    tokens: _Tokens | None = None


def _read_auth_file(path: str) -> tuple[str, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise ValueError(
            "open Codex auth file (provide existing ChatGPT credentials; login is not "
            f"implemented): open {path}: {describe(error)}"
        ) from error
    try:
        try:
            info = os.fstat(descriptor)
        except OSError as error:
            raise ValueError(f"inspect Codex auth file: {describe(error)}") from error
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError(
                "codex auth file must be a regular file with private permissions (chmod 600)"
            )
        data = bytearray()
        try:
            while len(data) <= _AUTH_FILE_LIMIT:
                chunk = os.read(descriptor, _AUTH_FILE_LIMIT + 1 - len(data))
                if not chunk:
                    break
                data += chunk
        except OSError as error:
            raise ValueError(f"read Codex auth file: {describe(error)}") from error
    finally:
        os.close(descriptor)
    auth: _AuthFile | None = None
    if len(data) <= _AUTH_FILE_LIMIT:
        try:
            auth = decode_object(bytes(data), _AuthFile)
        except JSONError:
            auth = None
    if auth is None:
        raise ValueError(
            "invalid Codex auth file; expected a JSON object with tokens.access_token "
            "and tokens.account_id"
        )
    if auth.auth_mode and auth.auth_mode != "chatgpt":
        raise ValueError("codex auth file is not a ChatGPT subscription login")
    tokens = auth.tokens or _Tokens()
    return trim_space_text(tokens.access_token or ""), trim_space_text(tokens.account_id or "")


class _AuthClaims(msgspec.Struct):
    chatgpt_account_id: str | None = None


class _Claims(msgspec.Struct):
    exp: Int64 | None = None
    auth: _AuthClaims | None = msgspec.field(default=None, name="https://api.openai.com/auth")


def _token_claims(token: str) -> tuple[str, int]:
    """Read unverified claims as routing and expiry hints; the server authenticates the token."""
    parts = token.split(".")
    if len(parts) != 3:
        return "", 0
    try:
        payload = _decode_raw_url_base64(parts[1])
        claims = decode_object(payload, _Claims)
    except ValueError, JSONError:
        raise ValueError("invalid Codex access token claims") from None
    account = claims.auth.chatgpt_account_id if claims.auth is not None else None
    return account or "", claims.exp or 0


def _decode_raw_url_base64(text: str) -> bytes:
    """Go's `base64.RawURLEncoding.DecodeString`, which skips CR and LF."""
    text = text.replace("\r", "").replace("\n", "")
    if not _BASE64_URL.fullmatch(text) or len(text) % 4 == 1:
        raise ValueError("illegal base64 data")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _header_value(value: str) -> bool:
    return value != "" and all(" " < character <= "~" for character in value)
