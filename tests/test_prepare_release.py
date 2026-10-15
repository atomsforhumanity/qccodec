"""scripts/prepare_release.py dates the changelog and updates compare links."""

import importlib.util
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.py"


def load_script():
    """Import the release script as a module."""
    spec = importlib.util.spec_from_file_location("prepare_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changelog_gets_dated_heading_and_links(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [unreleased]\n\n- Added a thing.\n\n## [0.1.0] - 2026-01-01\n\n"
        "[unreleased]: https://github.com/org/repo/compare/0.1.0...HEAD\n"
        "[0.1.0]: https://github.com/org/repo/releases/tag/0.1.0\n"
    )
    load_script().update_changelog("0.2.0", "https://github.com/org/repo")
    text = (tmp_path / "CHANGELOG.md").read_text()
    today = date.today().isoformat()
    assert f"## [unreleased]\n\n## [0.2.0] - {today}\n\n- Added a thing." in text
    assert "[unreleased]: https://github.com/org/repo/compare/0.2.0...HEAD" in text
    assert "[0.2.0]: https://github.com/org/repo/releases/tag/0.2.0" in text
    assert "[0.1.0]: https://github.com/org/repo/releases/tag/0.1.0" in text


def test_changelog_without_unreleased_section_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n")
    with pytest.raises(SystemExit):
        load_script().update_changelog("0.2.0", "https://github.com/org/repo")


def test_repository_changelog_has_unreleased_section():
    changelog = SCRIPT.parents[1] / "CHANGELOG.md"
    assert "## [unreleased]" in changelog.read_text()


def test_set_version_rewrites_only_the_project_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.1.0"\n\n[tool.other]\nversion = "keep"\n'
    )
    load_script().set_version("0.2.0")
    text = (tmp_path / "pyproject.toml").read_text()
    assert 'version = "0.2.0"' in text and 'version = "keep"' in text
