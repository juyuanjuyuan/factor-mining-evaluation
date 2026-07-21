#!/usr/bin/env python3
"""Project-root wrapper for validating factor batch JSON files."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factor_registry import main


if __name__ == "__main__":
    raise SystemExit(main())
