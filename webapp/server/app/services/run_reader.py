"""Lazy, cached conversion of evaluator CSV details to chart JSON."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..sanitize import sanitize


def _relative_map(metrics: Mapping[str, Any], key: str) -> dict[str, str]:
    value = metrics.get(key, {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


class RunReader:
    @staticmethod
    def detail_names(run: Mapping[str, Any]) -> list[str]:
        return sorted(_relative_map(run.get("result") or {}, "evaluation_details"))

    @staticmethod
    def artifact_names(run: Mapping[str, Any]) -> list[str]:
        return sorted(_relative_map(run.get("result") or {}, "evaluation_artifacts"))

    @staticmethod
    def split_version(name: str) -> tuple[str, int]:
        parts = name.rsplit("__", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return name, 1

    @classmethod
    def latest_detail_name(cls, names: list[str], base_name: str) -> str | None:
        matches = [
            (version, name)
            for name in names
            for base, version in [cls.split_version(name)]
            if base == base_name
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: item[0])[1]

    @classmethod
    def latest_artifact_name(
        cls, run: Mapping[str, Any], base_name: str
    ) -> str | None:
        return cls.latest_detail_name(cls.artifact_names(run), base_name)

    @lru_cache(maxsize=128)
    def read_detail(self, output_dir: str, relative_path: str, mtime_ns: int) -> dict[str, Any]:
        del mtime_ns
        root = Path(output_dir).resolve()
        path = (root / relative_path).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
        frame.index = frame.index.astype(str)
        result: dict[str, Any] = {
            "index": frame.index.tolist(),
            "columns": [str(column) for column in frame.columns],
            "data": sanitize(frame.to_numpy().tolist()),
        }
        if "group_returns" in path.name:
            result["summary"] = sanitize(frame.mean(axis=0, skipna=True).to_dict())
        return result

    def get_detail(self, run: Mapping[str, Any], name: str) -> dict[str, Any]:
        paths = _relative_map(run.get("result") or {}, "evaluation_details")
        if name not in paths:
            raise KeyError(name)
        root = Path(str(run["output_dir"])).resolve()
        path = (root / paths[name]).resolve()
        path.relative_to(root)
        return {"name": name, **self.read_detail(str(root), paths[name], path.stat().st_mtime_ns)}

    @staticmethod
    def artifact_path(run: Mapping[str, Any], name: str) -> Path:
        paths = _relative_map(run.get("result") or {}, "evaluation_artifacts")
        if name not in paths:
            raise KeyError(name)
        root = Path(str(run["output_dir"])).resolve()
        path = (root / paths[name]).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
