"""
NAVAL-SEM — single source of truth for the application version.

Both app/main.py and app/engine.py import APP_VERSION from here.
The CI pipeline (release.yml) patches _DEFAULT_APP_VERSION in this file
at build time via sed/PowerShell so the compiled binaries report the
correct release tag.

Runtime override: set the APP_VERSION environment variable to bypass
the compiled default (useful for dev/staging deployments).
"""

import os

_DEFAULT_APP_VERSION = "1.0.0"


def _resolve() -> str:
    env_version = os.getenv("APP_VERSION", "").strip()
    if env_version:
        return env_version.removeprefix("v")
    return _DEFAULT_APP_VERSION


APP_VERSION = _resolve()
