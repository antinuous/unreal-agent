"""Go path/filepath semantics captured from Go 1.27 on darwin."""

import os
from pathlib import Path

import pytest
from unreal_agent._filepath import go_abs, go_clean, go_getwd, go_join


@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("//x", "/x"),
        ("//x/../y", "/y"),
        ("///a//b/", "/a/b"),
        ("/../a", "/a"),
        ("", "."),
        (".", "."),
        ("a//b/../c/", "a/c"),
        ("//", "/"),
    ],
)
def test_go_clean(path: str, want: str) -> None:
    assert go_clean(path) == want


def test_go_join() -> None:
    assert go_join("//a", "b") == "/a/b"
    assert go_join("//a/", ".env") == "/a/.env"
    assert go_join("", "") == ""


@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("//x", "/x"),
        ("//x/../y", "/y"),
        ("", "{cwd}"),
        (".", "{cwd}"),
        ("a//b/../c/", "{cwd}/a/c"),
    ],
)
def test_go_abs(path: str, want: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PWD", raising=False)
    assert go_abs(path) == want.format(cwd=os.getcwd())


def test_go_getwd_prefers_matching_pwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    monkeypatch.chdir(link)
    monkeypatch.setenv("PWD", "/" + str(link))
    assert go_getwd() == "/" + str(link)
    assert go_abs("x") == str(link / "x")
    monkeypatch.setenv("PWD", str(tmp_path))
    assert go_getwd() == os.getcwd()
    monkeypatch.setenv("PWD", "relative")
    assert go_getwd() == os.getcwd()
