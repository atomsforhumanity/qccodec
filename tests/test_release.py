"""Release preparation tests; Git and dependency resolution are mocked."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("outcome", ["success", "lock_failure", "cancel"])
def test_release_updates_lock_before_git_and_restores_on_failure(
    tmp_path, monkeypatch, outcome
):
    spec = importlib.util.spec_from_file_location(
        "qccodec_release", Path(__file__).parents[1] / "scripts" / "release.py"
    )
    assert spec is not None and spec.loader is not None
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    monkeypatch.chdir(tmp_path)
    originals = {
        "pyproject.toml": '[project]\nversion = "0.11.2"\n[project.urls]\nSource = "https://example.com/qccodec"\n',
        "CHANGELOG.md": "# Changelog\n\n## [unreleased]\n\n- Changes.\n",
        "uv.lock": "original lockfile\n",
    }
    for filename, content in originals.items():
        Path(filename).write_text(content)
    monkeypatch.setattr(release.sys, "argv", ["release.py", "0.12.0"])
    monkeypatch.setattr(release, "confirm_version", lambda _: outcome != "cancel")
    calls = []

    def run(command, *, check):
        assert command == ["uv", "lock"] and check
        assert 'version = "0.12.0"' in Path("pyproject.toml").read_text()
        calls.append("lock")
        Path("uv.lock").write_text("updated lockfile\n")
        if outcome == "lock_failure":
            raise subprocess.CalledProcessError(1, command)

    def git(version):
        assert version == "0.12.0"
        assert calls == ["lock"]
        assert Path("uv.lock").read_text() == "updated lockfile\n"
        assert "## [0.12.0]" in Path("CHANGELOG.md").read_text()
        calls.append("git")

    monkeypatch.setattr(release.subprocess, "run", run)
    monkeypatch.setattr(release, "run_git_commands", git)
    if outcome == "success":
        release.main()
        assert calls == ["lock", "git"]
    else:
        with pytest.raises(SystemExit) as exc:
            release.main()
        assert exc.value.code == 1
        assert calls == (["lock"] if outcome == "lock_failure" else [])
        for filename, content in originals.items():
            assert Path(filename).read_text() == content
