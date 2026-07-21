"""Canonical project paths independent of the current working directory."""

import os
from pathlib import Path


def find_project_root() -> Path:
    """Locate the source checkout, with an environment override for installed use."""

    override = os.environ.get("FACTOR_MINING_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[1]
    candidates = (source_root, Path.cwd().resolve(), *Path.cwd().resolve().parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "data"
        ).is_dir():
            return candidate
    return source_root


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_MANIFEST_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FACTOR_OUTPUT_DIR = OUTPUT_DIR / "factor_evaluation"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
