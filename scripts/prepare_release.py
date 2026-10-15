"""Prepare a release on a branch: set the version, date the changelog, refresh uv.lock.

Usage:
    uv run python scripts/prepare_release.py 0.16.0

Then commit, open a pull request, merge it, and tag the merge commit on master
(see CONTRIBUTING.md). The tag triggers the release workflow.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path


def set_version(version: str) -> None:
    """Set the ``version`` in pyproject.toml's [project] table.

    Args:
        version: The version being released.

    Raises:
        SystemExit: pyproject.toml has no version line.
    """
    path = Path("pyproject.toml")
    text, count = re.subn(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        path.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        sys.exit("pyproject.toml has no 'version = \"...\"' line.")
    path.write_text(text)


def update_changelog(version: str, repo_url: str) -> None:
    """Move the [unreleased] notes under a dated heading and update compare links.

    Args:
        version: The version being released.
        repo_url: The GitHub repository URL used for compare links.

    Raises:
        SystemExit: The changelog has no [unreleased] section.
    """
    path = Path("CHANGELOG.md")
    text = path.read_text()
    if "## [unreleased]" not in text:
        sys.exit("CHANGELOG.md has no '## [unreleased]' section.")
    heading = f"## [unreleased]\n\n## [{version}] - {date.today().isoformat()}"
    text = text.replace("## [unreleased]", heading, 1)
    link = f"[unreleased]: {repo_url}/compare/{version}...HEAD\n"
    if re.search(r"^\[unreleased\]:", text, flags=re.MULTILINE):
        text = re.sub(
            r"^\[unreleased\]:.*\n",
            link + f"[{version}]: {repo_url}/releases/tag/{version}\n",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    path.write_text(text)


def main() -> None:
    """Set the version and changelog for a release."""
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+\S*", sys.argv[1]):
        sys.exit("Usage: uv run python scripts/prepare_release.py X.Y.Z")
    version = sys.argv[1]
    pyproject = Path("pyproject.toml").read_text()
    match = re.search(r'^Source\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE)
    repo_url = match.group(1).rstrip("/") if match else ""
    set_version(version)
    subprocess.run(["uv", "lock"], check=True)  # record the new version in uv.lock
    update_changelog(version, repo_url)
    print(f"Prepared {version}. Review the diff, then commit and open a pull request.")


if __name__ == "__main__":
    main()
