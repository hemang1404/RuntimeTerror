"""
RuntimeTerror -- GitHub PR Fetcher.

Fetches pull request data from the GitHub REST API:
  - PR metadata (title, description, state)
  - Changed files with diffs
  - Full file contents (both base and head)
  - Test files from the repository

Works without any third-party GitHub library -- pure ``requests``.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import requests


# ── Configuration ────────────────────────────────────────────────

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API = "https://api.github.com"


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class PRFile:
    """A single file changed in the pull request."""
    filename: str
    status: str  # added, modified, removed, renamed
    patch: str   # unified diff
    additions: int
    deletions: int
    raw_url: str  # URL to fetch raw content from HEAD
    contents_url: str  # API URL for file contents


@dataclass
class PRData:
    """All data needed to debug a pull request."""
    owner: str
    repo: str
    pr_number: int
    title: str
    description: str
    state: str  # open, closed, merged
    base_branch: str
    head_branch: str
    head_sha: str
    changed_files: list[PRFile] = field(default_factory=list)
    # Populated after fetch_file_contents()
    source_files: dict[str, str] = field(default_factory=dict)  # filename -> content
    test_files: dict[str, str] = field(default_factory=dict)    # filename -> content
    repo_test_files: dict[str, str] = field(default_factory=dict)  # all test files
    requirements: str = ""  # requirements.txt content
    pyproject: str = ""     # pyproject.toml content
    clone_path: str = ""    # local clone path


# ── API Helpers ──────────────────────────────────────────────────

def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def _get(url: str, params: dict | None = None) -> dict | list:
    """Make a GET request to the GitHub API."""
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get_raw(url: str) -> str:
    """Fetch raw file content."""
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.text


# ── URL Parsing ──────────────────────────────────────────────────

def parse_pr_url(url: str) -> tuple[str, str, int]:
    """Parse a GitHub PR URL into (owner, repo, pr_number).

    Accepts:
      - https://github.com/owner/repo/pull/42
      - github.com/owner/repo/pull/42
      - owner/repo#42
      - owner/repo/42
    """
    url = url.strip()

    # Full URL: https://github.com/owner/repo/pull/42
    m = re.match(r"(?:https?://)?github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    # Short: owner/repo#42
    m = re.match(r"([^/]+)/([^/#]+)#(\d+)", url)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    # Very short: owner/repo/42
    m = re.match(r"([^/]+)/([^/]+)/(\d+)$", url)
    if m:
        return m.group(1), m.group(2), int(m.group(3))

    raise ValueError(f"Could not parse PR URL: {url!r}")


# ── Core Fetchers ────────────────────────────────────────────────

def fetch_pr_metadata(owner: str, repo: str, pr_number: int) -> PRData:
    """Fetch PR metadata from GitHub API."""
    data = _get(f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}")

    return PRData(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        title=data["title"],
        description=data.get("body", "") or "",
        state=data["state"],
        base_branch=data["base"]["ref"],
        head_branch=data["head"]["ref"],
        head_sha=data["head"]["sha"],
    )


def fetch_pr_files(pr: PRData) -> None:
    """Fetch the list of changed files for a PR (populates pr.changed_files)."""
    files = _get(
        f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/pulls/{pr.pr_number}/files",
        params={"per_page": 100},
    )

    pr.changed_files = []
    for f in files:
        if not f["filename"].endswith(".py"):
            continue  # Python only
        pr.changed_files.append(PRFile(
            filename=f["filename"],
            status=f["status"],
            patch=f.get("patch", ""),
            additions=f["additions"],
            deletions=f["deletions"],
            raw_url=f.get("raw_url", ""),
            contents_url=f.get("contents_url", ""),
        ))


def fetch_file_contents(pr: PRData) -> None:
    """Fetch the actual content of changed Python files from the HEAD branch.

    Also tries to find related test files and requirements.
    """
    # Fetch changed files
    for f in pr.changed_files:
        if f.status == "removed":
            continue
        try:
            url = f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/contents/{f.filename}?ref={pr.head_sha}"
            data = _get(url)
            if data.get("encoding") == "base64":
                import base64
                content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            else:
                content = _get_raw(data["download_url"])

            # Sort into source or test files
            basename = os.path.basename(f.filename)
            if basename.startswith("test_") or basename.endswith("_test.py") or "/tests/" in f.filename:
                pr.test_files[f.filename] = content
            else:
                pr.source_files[f.filename] = content
        except Exception as e:
            print(f"  [WARN] Could not fetch {f.filename}: {e}")

    # Try to fetch related test files (not in changed files)
    _fetch_repo_tests(pr)

    # Try to fetch requirements
    for req_file in ["requirements.txt", "requirements-dev.txt"]:
        try:
            url = f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/contents/{req_file}?ref={pr.head_sha}"
            data = _get(url)
            import base64
            pr.requirements = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            break
        except Exception:
            pass

    # Try pyproject.toml
    try:
        url = f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/contents/pyproject.toml?ref={pr.head_sha}"
        data = _get(url)
        import base64
        pr.pyproject = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        pass


def _fetch_repo_tests(pr: PRData) -> None:
    """Try to find test files in common locations."""
    test_dirs = ["tests", "test"]

    for test_dir in test_dirs:
        try:
            url = f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/contents/{test_dir}?ref={pr.head_sha}"
            items = _get(url)
            if not isinstance(items, list):
                continue
            for item in items:
                if item["type"] != "file":
                    continue
                if not item["name"].endswith(".py"):
                    continue
                if item["name"].startswith("test_") or item["name"].endswith("_test.py"):
                    try:
                        import base64
                        file_data = _get(item["url"])
                        content = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
                        pr.repo_test_files[f"{test_dir}/{item['name']}"] = content
                    except Exception:
                        pass
        except Exception:
            pass


# ── Repository Cloning ───────────────────────────────────────────

def clone_repo(pr: PRData, target_dir: str | None = None) -> str:
    """Clone the repository and checkout the PR branch.

    Returns the path to the cloned directory.
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="runtime_terror_pr_")

    repo_url = f"https://github.com/{pr.owner}/{pr.repo}.git"

    # Shallow clone for speed
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", pr.head_branch, repo_url, target_dir],
        capture_output=True, text=True, timeout=120,
    )

    pr.clone_path = target_dir
    return target_dir


# ── Convenience: Full fetch ──────────────────────────────────────

def fetch_pr(url: str) -> PRData:
    """One-call convenience: parse URL -> fetch everything.

    Usage::

        pr = fetch_pr("https://github.com/owner/repo/pull/42")
        print(pr.title)
        print(pr.source_files)     # {filename: content}
        print(pr.test_files)       # {filename: test_content}
        print(pr.changed_files)    # [PRFile(...), ...]
    """
    owner, repo, pr_number = parse_pr_url(url)
    pr = fetch_pr_metadata(owner, repo, pr_number)
    fetch_pr_files(pr)
    fetch_file_contents(pr)
    return pr


# ── CLI test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m server.github_fetcher <PR_URL>")
        sys.exit(1)

    pr = fetch_pr(sys.argv[1])
    print(f"PR #{pr.pr_number}: {pr.title}")
    print(f"State: {pr.state}")
    print(f"Branch: {pr.head_branch} -> {pr.base_branch}")
    print(f"\nChanged Python files ({len(pr.changed_files)}):")
    for f in pr.changed_files:
        print(f"  {f.status:10s} {f.filename} (+{f.additions}/-{f.deletions})")
    print(f"\nSource files fetched: {len(pr.source_files)}")
    for name in pr.source_files:
        print(f"  {name} ({len(pr.source_files[name])} chars)")
    print(f"\nTest files in changed: {len(pr.test_files)}")
    print(f"Test files in repo:    {len(pr.repo_test_files)}")
    if pr.requirements:
        print(f"\nrequirements.txt: {len(pr.requirements)} chars")
