#!/usr/bin/env python3
"""Evaluate the runnable Guotai Junan GTJA191 factor batch."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from batch import main


if __name__ == "__main__":
    # Insert defaults before user-provided arguments, so callers can still
    # override --output-dir, --select, --methods, and other batch options.
    sys.argv[1:1] = [
        "--registry-file",
        str(PROJECT_ROOT / "factor_registry" / "gtja191_runnable_factors.json"),
        "--set",
        "runnable",
        "--output-dir",
        str(PROJECT_ROOT / "outputs" / "factor_evaluation" / "gtja191_runnable"),
    ]
    raise SystemExit(main())
