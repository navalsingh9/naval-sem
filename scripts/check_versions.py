#!/usr/bin/env python3
"""
Fail if the four places that declare a version disagree.

release.yml patches app/version.py from the git tag at build time but never
touches pyproject.toml or CITATION.cff, so they drift silently -- version.py
once sat at 2.0.0 while the project was shipping 2.0.1, and CITATION.cff (the
file Zenodo and GitHub's citation widget read) sat at 0.2.1. This turns that
into a failed check instead of something noticed months later.
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _search(path: str, pattern: str) -> str | None:
    m = re.search(pattern, (ROOT / path).read_text(), re.M)
    return m.group(1) if m else None


found = {
    "pyproject.toml": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
    "CITATION.cff":   _search("CITATION.cff", r'^version: "([^"]+)"'),
    "app/version.py": _search("app/version.py", r'^_DEFAULT_APP_VERSION = "([^"]+)"'),
    "uv.lock":        _search("uv.lock", r'name = "naval-sem"\nversion = "([^"]+)"'),
}

width = max(len(k) for k in found)
for name, value in found.items():
    print(f"  {name:<{width}}  {value}")

missing = [n for n, v in found.items() if v is None]
if missing:
    print(f"\nCould not read a version from: {', '.join(missing)}")
    sys.exit(1)

if len(set(found.values())) != 1:
    print("\nVersions disagree. Bump them together in the PR that precedes the tag.")
    sys.exit(1)

print("\nAll four agree.")
