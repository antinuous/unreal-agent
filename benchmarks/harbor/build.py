"""Build a self-contained Linux runner bundle from a committed source tree."""

import argparse
import configparser
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from harness_harbor.bundle import ARCHIVE, LAUNCHER, MACHINES

PACKAGE = "unreal-agent"
PLATFORM = "{machine}-manylinux_2_28"
PRUNE = (
    "bin/idle3*",
    "bin/pip*",
    "bin/pydoc3*",
    "bin/python*-config",
    "include",
    "lib/itcl*",
    "lib/libpython*",
    "lib/libtcl*",
    "lib/libtk*",
    "lib/pkgconfig",
    "lib/tcl*",
    "lib/thread*",
    "lib/tk*",
    "lib/python3.*/ensurepip",
    "lib/python3.*/idlelib",
    "lib/python3.*/lib-dynload/_tkinter*",
    "lib/python3.*/site-packages/pip*",
    "lib/python3.*/test",
    "lib/python3.*/tkinter",
    "lib/python3.*/turtledemo",
    "share",
)

# PEP 538 locale coercion exports LC_CTYPE=C.UTF-8 when the caller runs in the C
# locale, and -I stops PYTHONCOERCECLOCALE from disabling it. The launcher passes
# the caller's LC_CTYPE through so tool processes inherit the original environment.
LAUNCHER_SCRIPT = """\
#!/bin/sh
case $0 in
*/*) root=${{0%/*}} ;;
*) root=$(dirname -- "$(command -v -- "$0")") ;;
esac
exec "$root/python/bin/python3" -I -c '
import os, sys
if sys.argv[1]:
    os.environ["LC_CTYPE"] = sys.argv[2]
else:
    os.environ.pop("LC_CTYPE", None)
sys.argv[:3] = ["{runner}"]
from {module} import {function}
sys.exit({function}())
' "${{LC_CTYPE+set}}" "${{LC_CTYPE-}}" "$@"
"""


def run(*command: str | Path, cwd: Path | None = None) -> str:
    return subprocess.run(
        [str(part) for part in command], cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def entry_point(wheel: Path, runner: str) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt"))
        entries = configparser.ConfigParser(delimiters=("=",))
        entries.read_string(archive.read(name).decode())
    if not entries.has_option("console_scripts", runner):
        raise ValueError(f"{wheel.name} has no {runner} console script")
    module, _, function = entries.get("console_scripts", runner).partition(":")
    if not re.fullmatch(r"[\w.]+", module) or not function.isidentifier():
        raise ValueError(f"Unsupported {runner} entry point")
    return module, function


def install_python(request: str, machine: str, directory: Path) -> tuple[Path, str]:
    run(
        "uv",
        "python",
        "install",
        f"cpython-{request}-linux-{machine}-gnu",
        "--install-dir",
        directory,
        "--no-bin",
        "--no-registry",
    )
    [home] = [
        path for path in directory.glob(f"cpython-*-linux-{machine}-gnu") if not path.is_symlink()
    ]
    match = re.fullmatch(rf"cpython-(3\.\d+\.\d+)-linux-{machine}-gnu", home.name)
    if not match:
        raise ValueError(f"Unexpected Python installation: {home.name}")
    return home, match[1]


def normalize(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = member.gid = 0
    member.uname = member.gname = "root"
    if member.isdir() or member.mode & 0o100:
        member.mode |= 0o555
    elif member.isfile():
        member.mode |= 0o444
    return member


def build(revision: str, output: Path, arch: str, runner: str, python: str) -> None:
    repo = Path(__file__).resolve().parents[2]
    commit = run("git", "rev-parse", "--verify", f"{revision}^{{commit}}", cwd=repo).strip()
    if output.exists():
        raise FileExistsError(f"Refusing to replace runner bundle: {output}")
    machine = MACHINES[arch]
    with TemporaryDirectory(prefix="harbor-build-") as temporary:
        work = Path(temporary)
        source, wheels, pythons, root = (
            work / name for name in ("source", "wheels", "pythons", "root")
        )
        source.mkdir()
        with (work / "source.tar").open("wb") as archive:
            subprocess.run(["git", "archive", commit], cwd=repo, stdout=archive, check=True)
        with tarfile.open(work / "source.tar") as tree:
            tree.extractall(source, filter="data")
        run("uv", "build", "--wheel", "--package", PACKAGE, "--out-dir", wheels, cwd=source)
        [wheel] = wheels.glob("*.whl")
        module, function = entry_point(wheel, runner)
        requirements = work / "requirements.txt"
        run(
            "uv",
            "export",
            "--frozen",
            "--package",
            PACKAGE,
            "--no-dev",
            "--no-emit-workspace",
            "--output-file",
            requirements,
            cwd=source,
        )

        home, python_version = install_python(python, machine, pythons)
        for pattern in PRUNE:
            for path in home.glob(pattern):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        [site_packages] = home.glob("lib/python3.*/site-packages")
        target = [
            "--target",
            str(site_packages),
            "--python",
            sys.executable,
            "--python-platform",
            PLATFORM.format(machine=machine),
            "--python-version",
            python_version,
            "--only-binary",
            ":all:",
            "--no-deps",
            "--compile-bytecode",
        ]
        run("uv", "pip", "install", *target, "--require-hashes", "-r", requirements)
        run("uv", "pip", "install", *target, wheel)
        if (site_packages / "bin").exists():
            shutil.rmtree(site_packages / "bin")

        root.mkdir()
        home.rename(root / "python")
        launcher = root / LAUNCHER
        launcher.write_text(LAUNCHER_SCRIPT.format(runner=runner, module=module, function=function))
        launcher.chmod(0o755)
        bundle = work / ARCHIVE
        with tarfile.open(bundle, "w:gz", compresslevel=9) as archive:
            for path in sorted(root.iterdir()):
                archive.add(path, arcname=path.name, filter=normalize)
        data = bundle.read_bytes()
        manifest = {
            "revision": commit,
            "runner": runner,
            "sha256": hashlib.sha256(data).hexdigest(),
            "os": "linux",
            "arch": arch,
            "python_version": python_version,
        }
        output.mkdir(parents=True)
        (output / ARCHIVE).write_bytes(data)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {commit} ({manifest['sha256']}, {len(data)} bytes) in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--runner",
        default="unreal-agent-runner",
        help="console script of the unreal-agent package (default: unreal-agent-runner)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=tuple(MACHINES), default="amd64")
    parser.add_argument("--python", default="3.14", help="CPython version request")
    args = parser.parse_args()
    build(args.revision, args.output, args.arch, args.runner, args.python)
