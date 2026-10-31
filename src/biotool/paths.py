"""Where the tool keeps downloaded models and settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

RESOURCES = Path(__file__).parent / "resources"


def app_data_dir() -> Path:
    """Per-user data folder (models cache, settings)."""
    override = os.environ.get("BIOTOOL_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "BioTool"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "BioTool"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "biotool"
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    d = app_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
