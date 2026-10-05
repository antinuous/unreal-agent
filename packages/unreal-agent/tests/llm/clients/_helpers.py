import json
from pathlib import Path


def write_test_auth(path: Path, token: str, account: str) -> None:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": token,
                    "account_id": account,
                    "refresh_token": "never-use-or-write",
                },
            }
        )
    )
    path.chmod(0o600)
