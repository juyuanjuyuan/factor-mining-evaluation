"""Convert numpy/pandas values to strict JSON-compatible values."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if pd.isna(value):
        return None
    return str(value)
