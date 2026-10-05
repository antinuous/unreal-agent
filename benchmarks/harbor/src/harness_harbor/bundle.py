import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ARCHIVE = "unreal-agent-runner.tar.gz"
LAUNCHER = "unreal-agent-runner"
MACHINES = {"amd64": "x86_64", "arm64": "aarch64"}
PYTHON_VERSION = re.compile(r"3\.\d+\.\d+")


@dataclass(frozen=True)
class Bundle:
    directory: Path
    revision: str
    sha256: str
    arch: str
    python_version: str

    @classmethod
    def load(cls, directory: str) -> Bundle:
        path = Path(directory).expanduser().resolve()
        manifest = json.loads((path / "manifest.json").read_text())
        for name, length in (("revision", 40), ("sha256", 64)):
            value = manifest[name]
            if len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid bundle {name}")
        if manifest["os"] != "linux" or manifest["arch"] not in MACHINES:
            raise ValueError("Runner bundle must target Linux amd64 or arm64")
        if not PYTHON_VERSION.fullmatch(manifest["python_version"]):
            raise ValueError("Invalid bundle python_version")
        bundle = cls(
            path,
            manifest["revision"],
            manifest["sha256"],
            manifest["arch"],
            manifest["python_version"],
        )
        bundle.read_archive()
        return bundle

    def read_archive(self) -> bytes:
        data = (self.directory / ARCHIVE).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("Runner bundle checksum does not match its manifest")
        return data
