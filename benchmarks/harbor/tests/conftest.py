import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from harness_harbor.bundle import ARCHIVE

type MakeBundle = Callable[..., Path]


@pytest.fixture
def make_bundle(tmp_path: Path) -> MakeBundle:
    def make(revision: str = "a" * 40, data: bytes = b"runner", **fields: str) -> Path:
        (tmp_path / ARCHIVE).write_bytes(data)
        manifest = {
            "revision": revision,
            "runner": "unreal-agent-runner",
            "sha256": hashlib.sha256(data).hexdigest(),
            "os": "linux",
            "arch": "amd64",
            "python_version": "3.14.8",
            **fields,
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        return tmp_path

    return make
