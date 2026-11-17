"""Publish a built MkDocs site to the gh-pages branch, replacing only its own part.

The branch serves two sites: the stable docs at its root, built from final release
tags, and the next branch's docs under ``next/``. A root deploy replaces everything but
``next/`` and ``CNAME``; a deploy with ``--prefix next`` replaces ``next/`` and touches
nothing else. (``mkdocs gh-deploy`` replaces the whole branch, so it can't do either.)

Usage, from the repository root after ``mkdocs build``:
    uv run python scripts/deploy_docs.py site                     # the stable docs
    uv run python scripts/deploy_docs.py site-next --prefix next  # the next docs

Each deploy is one commit on top of the remote's branch (or the branch's first commit),
pushed without force. If another deploy pushed first, it starts again from the new tip.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: What a root deploy leaves in place: the next docs and the custom domain.
KEPT_BY_ROOT = frozenset({".git", "next", "CNAME"})
ATTEMPTS = 3


def git(*args: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    """Run git and return its output, stripped."""
    done = subprocess.run(
        ["git", *args], cwd=cwd, input=stdin, text=True, check=True, capture_output=True
    )
    return done.stdout.strip()


def remote_tip(remote: str, branch: str) -> str | None:
    """Fetch the remote's branch and return its commit, or None if it has none."""
    found = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", remote, branch],
        capture_output=True,
    )
    if found.returncode == 2:  # no such branch
        return None
    found.check_returncode()
    git("fetch", "--quiet", "--depth=1", remote, f"refs/heads/{branch}")
    return git("rev-parse", "FETCH_HEAD")


def replace(work: Path, site: Path, prefix: str, new_branch: bool) -> None:
    """Replace this deploy's part of the checked-out branch with the built site."""
    if prefix:
        target = work / prefix
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(site, target)
        (target / "CNAME").unlink(missing_ok=True)  # only the root's counts
        if not new_branch:
            return
    else:
        for entry in work.iterdir():
            if entry.name in KEPT_BY_ROOT:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        shutil.copytree(site, work, dirs_exist_ok=True)
    (work / ".nojekyll").touch()  # serve files whose names start with _ as they are


def publish(site: Path, prefix: str, remote: str, branch: str, message: str) -> None:
    """Commit the site onto the remote's branch and push it."""
    target = f"{branch}:/{prefix}/" if prefix else f"{branch}:/"
    for attempt in range(1, ATTEMPTS + 1):
        tip = remote_tip(remote, branch)
        if tip is None:  # the branch's first commit: an empty one to start from
            empty_tree = git("mktree", stdin="")
            base = git("commit-tree", empty_tree, "-m", f"Start {branch}")
        else:
            base = tip
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / branch
            git("worktree", "add", "--quiet", "--detach", str(work), base)
            try:
                replace(work, site, prefix, new_branch=tip is None)
                git("add", "--all", cwd=work)
                if not git("status", "--porcelain", cwd=work):
                    print(f"{target} already holds this site.")
                    return
                git("commit", "--quiet", "--message", message, cwd=work)
                pushed = subprocess.run(
                    ["git", "push", "--quiet", remote, f"HEAD:refs/heads/{branch}"],
                    cwd=work,
                )
                if pushed.returncode == 0:
                    print(f"Published {site} to {target}")
                    return
            finally:
                git("worktree", "remove", "--force", str(work))
        print(f"Push rejected (attempt {attempt} of {ATTEMPTS}); retrying.")
        time.sleep(5 * attempt)
    sys.exit(f"Could not push {branch} after {ATTEMPTS} attempts.")


def main() -> None:
    """Parse the arguments and publish."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("site", type=Path, help="the built site (mkdocs build)")
    parser.add_argument(
        "--prefix", default="", help="the directory it replaces (default: the root)"
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="gh-pages")
    parser.add_argument("--message", help="the commit message")
    args = parser.parse_args()

    site: Path = args.site.resolve()
    prefix: str = args.prefix.strip("/")
    if not (site / "index.html").is_file():
        sys.exit(f"{site} is not a built site: it has no index.html.")
    if prefix and (prefix.startswith(".") or "/" in prefix):
        sys.exit("--prefix is one directory name, such as next.")
    if not prefix and (site / "next").exists():
        sys.exit(f"{site} has a next/ directory, which belongs to the next docs.")
    target = f"/{prefix}/" if prefix else "the root"
    commit = git("rev-parse", "--short", "HEAD")
    message = args.message or f"Deploy {commit} to {target}"
    publish(site, prefix, args.remote, args.branch, message)


if __name__ == "__main__":
    main()
