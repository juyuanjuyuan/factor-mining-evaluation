"""Runtime configuration and project import bootstrap."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _find_project_root() -> Path:
    configured = os.getenv("FACTOR_MINING_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "src" / "engine.py").is_file():
            raise RuntimeError(f"Invalid FACTOR_MINING_ROOT: {root}")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "engine.py").is_file():
            return parent
    raise RuntimeError("Cannot locate the factor-mining project root")


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
# This is deliberately a Python path entry, not PYTHONPATH. The project path
# contains a colon, which is unsafe in the PYTHONPATH environment variable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    registry_dir: Path = PROJECT_ROOT / "factor_registry"
    state_dir: Path = PROJECT_ROOT / "outputs" / "webapp"
    frontend_dist: Path = PROJECT_ROOT / "webapp" / "frontend" / "dist"
    test_batch_id: str = "webapp_test_factors"
    submitted_batch_id: str = "webapp_factor_library"
    legacy_custom_batch_id: str = "webapp_custom_factors"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "platform.db"

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def genetic_mining_dir(self) -> Path:
        """Campaign roots shared by the CLI and the web mining interface."""

        return self.state_dir.parent / "gp_factor_mining"

    @property
    def test_registry_path(self) -> Path:
        return self.registry_dir / f"{self.test_batch_id}.json"

    @property
    def submitted_registry_path(self) -> Path:
        return self.registry_dir / f"{self.submitted_batch_id}.json"

    @property
    def custom_batch_id(self) -> str:
        """Compatibility alias for older webapp code/tests."""
        return self.test_batch_id

    @property
    def custom_registry_path(self) -> Path:
        """Compatibility alias for older webapp code/tests."""
        return self.test_registry_path


settings = Settings(
    data_dir=Path(os.getenv("FACTOR_WEBAPP_DATA_DIR", PROJECT_ROOT / "data")).resolve(),
    registry_dir=Path(
        os.getenv("FACTOR_WEBAPP_REGISTRY_DIR", PROJECT_ROOT / "factor_registry")
    ).resolve(),
    state_dir=Path(
        os.getenv("FACTOR_WEBAPP_STATE_DIR", PROJECT_ROOT / "outputs" / "webapp")
    ).resolve(),
)
