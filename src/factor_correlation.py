"""Incremental correlation checks for the submitted factor library.

The factor registry deliberately stores definitions only.  This module keeps the
derived factor-exposure cache and the full, square Pearson-correlation matrix
under the webapp state directory instead.  After the first build, admitting a
candidate evaluates only that candidate, loads the cached exposures for the
existing library, and calculates its new row and column.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import (
    DEFAULT_FILES,
    evaluate_expression,
    expression_data_symbols,
    load_market_data,
    parse_and_validate_expression,
)


CACHE_VERSION = 1
CORRELATION_METHOD = "pooled_pearson"
CORRELATION_THRESHOLD = 0.75
CORRELATION_WINDOW = 60
MIN_PAIRED_OBSERVATIONS = 3


class FactorCorrelationError(ValueError):
    """The library correlation matrix cannot be calculated reliably."""


class FactorCorrelationThresholdError(FactorCorrelationError):
    """A candidate would make the library exceed its correlation threshold."""

    def __init__(self, violations: Sequence[Mapping[str, Any]], threshold: float):
        self.violations = [dict(violation) for violation in violations]
        self.threshold = threshold
        examples = "；".join(
            f"{item['factor_a']} / {item['factor_b']} = {item['correlation']:.4f}"
            for item in self.violations[:3]
        )
        remainder = "" if len(self.violations) <= 3 else f"；另有 {len(self.violations) - 3} 对"
        super().__init__(
            "因子库要求所有非对角元素的绝对 Pearson 相关系数不超过 "
            f"{threshold:.2f}，当前不通过：{examples}{remainder}"
        )


@dataclass(frozen=True)
class FactorDefinition:
    """A normalized registry definition and its stable cache identity."""

    factor_name: str
    expression: str
    expression_hash: str


@dataclass(frozen=True)
class PreparedCandidate:
    """A validated candidate that can be committed after the registry write."""

    definitions: tuple[FactorDefinition, ...]
    candidate: FactorDefinition
    candidate_exposure: pd.DataFrame
    matrix: pd.DataFrame


def non_overlapping_pair_correlations(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    window: int = CORRELATION_WINDOW,
    threshold: float = CORRELATION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Calculate pooled Pearson correlations in consecutive full-date blocks."""

    if window < 1:
        raise ValueError("correlation window must be positive")
    if not 0 < threshold < 1:
        raise ValueError("correlation threshold must be between 0 and 1")
    right = right.reindex(index=left.index, columns=left.columns)
    full_window_count = len(left.index) // window
    rows: list[dict[str, Any]] = []

    for window_number in range(full_window_count):
        start_position = window_number * window
        stop_position = start_position + window
        left_window = left.iloc[start_position:stop_position]
        right_window = right.iloc[start_position:stop_position]
        left_values = left_window.to_numpy(dtype=float, copy=False).ravel()
        right_values = right_window.to_numpy(dtype=float, copy=False).ravel()
        paired = np.isfinite(left_values) & np.isfinite(right_values)
        paired_observations = int(paired.sum())
        correlation: float | None = None
        if paired_observations >= MIN_PAIRED_OBSERVATIONS:
            x = left_values[paired]
            y = right_values[paired]
            if not np.isclose(np.std(x), 0.0) and not np.isclose(np.std(y), 0.0):
                candidate = float(np.corrcoef(x, y)[0, 1])
                if np.isfinite(candidate):
                    correlation = candidate

        start_day = pd.Timestamp(left.index[start_position]).date().isoformat()
        end_day = pd.Timestamp(left.index[stop_position - 1]).date().isoformat()
        rows.append(
            {
                "window_number": window_number + 1,
                "start_day": start_day,
                "end_day": end_day,
                "correlation": correlation,
                "paired_observations": paired_observations,
                "exceeds_threshold": (
                    correlation is not None and abs(correlation) > threshold
                ),
            }
        )
    return rows


class FactorCorrelationService:
    """Maintain the submitted-library Pearson correlation matrix.

    Correlation is calculated over all aligned finite ``(date, security)``
    observations (pooled Pearson correlation).  This gives a single symmetric
    n x n matrix and avoids comparing factor values from different axes.
    """

    def __init__(self, settings: Any, *, threshold: float = CORRELATION_THRESHOLD):
        if not 0 < threshold < 1:
            raise ValueError("correlation threshold must be between 0 and 1")
        self.settings = settings
        self.threshold = float(threshold)
        self.root = Path(settings.state_dir) / "factor_correlation"
        self.exposure_dir = self.root / "exposures"
        self.matrix_path = self.root / "matrix.csv"
        self.metadata_path = self.root / "metadata.json"
        self._lock = threading.RLock()

    def matrix(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return the current library matrix, rebuilding a stale cache once."""
        definitions = self._definitions(records)
        with self._lock:
            matrix = self._ensure_base(definitions)
            return self._payload(matrix)

    def pair_detail(
        self,
        factor_a: str,
        factor_b: str,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return non-overlapping 60-day correlations for one library pair."""

        if factor_a == factor_b:
            raise FactorCorrelationError("分段相关性需要选择两个不同的因子")
        definitions = self._definitions(records)
        by_name = {item.factor_name: item for item in definitions}
        missing = [name for name in (factor_a, factor_b) if name not in by_name]
        if missing:
            raise FactorCorrelationError(
                "正式因子库中不存在以下因子：" + "、".join(missing)
            )

        with self._lock:
            self._ensure_base(definitions)
            left = self._read_exposure(by_name[factor_a])
            right = self._read_exposure(by_name[factor_b])
            windows = non_overlapping_pair_correlations(
                left,
                right,
                window=CORRELATION_WINDOW,
                threshold=self.threshold,
            )

        finite_correlations = [
            abs(float(row["correlation"]))
            for row in windows
            if row["correlation"] is not None
        ]
        return {
            "factor_a": factor_a,
            "factor_b": factor_b,
            "method": CORRELATION_METHOD,
            "window_size": CORRELATION_WINDOW,
            "step_size": CORRELATION_WINDOW,
            "threshold": self.threshold,
            "sample_definition": (
                "按连续 60 个交易日切分的非重叠窗口；每个窗口使用所有对齐且有限的"
                "日期-证券观测值计算 pooled Pearson 相关系数，尾部不足 60 日不纳入"
            ),
            "max_abs_correlation": (
                max(finite_correlations) if finite_correlations else None
            ),
            "violation_window_count": sum(
                bool(row["exceeds_threshold"]) for row in windows
            ),
            "windows": windows,
        }

    def prepare_candidate(
        self,
        candidate_record: Mapping[str, Any],
        existing_records: Sequence[Mapping[str, Any]],
    ) -> PreparedCandidate:
        """Calculate a candidate's row/column without yet altering the cache.

        The caller writes the factor registry first and then invokes
        :meth:`commit_candidate`, so a failed registry write cannot leave a
        matrix referring to a non-existent factor.
        """
        candidate = self._definition(candidate_record)
        existing = self._definitions(existing_records)
        if candidate.factor_name in {item.factor_name for item in existing}:
            raise FactorCorrelationError(f"因子名称已在相关性矩阵中存在: {candidate.factor_name}")

        with self._lock:
            base = self._ensure_base(existing)
            candidate_exposure = self._evaluate_factor(candidate)
            self._validate_exposure(candidate, candidate_exposure)

            matrix = base.copy()
            matrix.loc[candidate.factor_name, candidate.factor_name] = 1.0
            for item in existing:
                existing_exposure = self._read_exposure(item)
                correlation = self._pair_correlation(
                    item,
                    existing_exposure,
                    candidate,
                    candidate_exposure,
                )
                matrix.loc[item.factor_name, candidate.factor_name] = correlation
                matrix.loc[candidate.factor_name, item.factor_name] = correlation

            definitions = tuple(sorted((*existing, candidate), key=lambda item: item.factor_name))
            names = [item.factor_name for item in definitions]
            matrix = matrix.reindex(index=names, columns=names).astype(float)
            violations = self._violations(matrix)
            if violations:
                raise FactorCorrelationThresholdError(violations, self.threshold)
            return PreparedCandidate(definitions, candidate, candidate_exposure, matrix)

    def commit_candidate(self, prepared: PreparedCandidate) -> None:
        """Persist a successfully admitted candidate and the resulting matrix."""
        with self._lock:
            self._write_exposure(prepared.candidate, prepared.candidate_exposure)
            self._write_matrix(prepared.matrix)
            self._write_metadata(prepared.definitions)

    def invalidate_removed_factor(self, record: Mapping[str, Any]) -> None:
        """Discard derived cache entries for a factor removed from the library.

        The registry is the source of truth.  Removing the matrix metadata
        forces the next read or admission to rebuild a matrix for exactly the
        remaining definitions, without requiring market data during removal.
        """
        definition = self._definition(record)
        with self._lock:
            for path in (
                self._exposure_path(definition),
                self.matrix_path,
                self.metadata_path,
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _ensure_base(self, definitions: tuple[FactorDefinition, ...]) -> pd.DataFrame:
        if self._cache_matches(definitions):
            return self._read_matrix(definitions)
        return self._rebuild(definitions)

    def _rebuild(self, definitions: tuple[FactorDefinition, ...]) -> pd.DataFrame:
        for definition in definitions:
            exposure = self._evaluate_factor(definition)
            self._validate_exposure(definition, exposure)
            self._write_exposure(definition, exposure)
        matrix = self._build_matrix(definitions)
        self._write_matrix(matrix)
        self._write_metadata(definitions)
        return matrix

    def _build_matrix(
        self,
        definitions: tuple[FactorDefinition, ...],
    ) -> pd.DataFrame:
        names = [item.factor_name for item in definitions]
        matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
        for position, left in enumerate(definitions):
            left_exposure = self._read_exposure(left)
            for right in definitions[position + 1 :]:
                right_exposure = self._read_exposure(right)
                correlation = self._pair_correlation(
                    left,
                    left_exposure,
                    right,
                    right_exposure,
                )
                matrix.loc[left.factor_name, right.factor_name] = correlation
                matrix.loc[right.factor_name, left.factor_name] = correlation
        return matrix

    def _definitions(
        self, records: Sequence[Mapping[str, Any]]
    ) -> tuple[FactorDefinition, ...]:
        definitions = tuple(sorted((self._definition(record) for record in records), key=lambda item: item.factor_name))
        names = [item.factor_name for item in definitions]
        if len(names) != len(set(names)):
            raise FactorCorrelationError("因子库中存在重复名称，无法构建相关性矩阵")
        return definitions

    @staticmethod
    def _definition(record: Mapping[str, Any]) -> FactorDefinition:
        name = str(record.get("factor_name", "")).strip()
        expression = str(record.get("expression", "")).strip()
        if not name:
            raise FactorCorrelationError("相关性矩阵需要非空的 factor_name")
        try:
            parse_and_validate_expression(expression)
        except (SyntaxError, ValueError) as exc:
            raise FactorCorrelationError(f"因子 {name} 的表达式无效: {exc}") from exc
        return FactorDefinition(
            factor_name=name,
            expression=expression,
            expression_hash=hashlib.sha256(expression.encode("utf-8")).hexdigest(),
        )

    def _evaluate_factor(self, definition: FactorDefinition) -> pd.DataFrame:
        data = load_market_data(self.settings.data_dir, definition.expression)
        factor = evaluate_expression(definition.expression, data).replace([np.inf, -np.inf], np.nan)
        return factor.reindex(index=data["c"].index, columns=data["c"].columns).astype(float)

    @staticmethod
    def _validate_exposure(definition: FactorDefinition, exposure: pd.DataFrame) -> None:
        values = exposure.to_numpy(dtype=float, copy=False).ravel()
        values = values[np.isfinite(values)]
        if len(values) < MIN_PAIRED_OBSERVATIONS:
            raise FactorCorrelationError(
                f"因子 {definition.factor_name} 的有效暴露少于 {MIN_PAIRED_OBSERVATIONS} 个，无法计算相关性"
            )
        if np.isclose(np.std(values), 0.0):
            raise FactorCorrelationError(
                f"因子 {definition.factor_name} 的有效暴露为常数，无法计算相关性"
            )

    def _pair_correlation(
        self,
        left_definition: FactorDefinition,
        left: pd.DataFrame,
        right_definition: FactorDefinition,
        right: pd.DataFrame,
    ) -> float:
        right = right.reindex(index=left.index, columns=left.columns)
        left_values = left.to_numpy(dtype=float, copy=False).ravel()
        right_values = right.to_numpy(dtype=float, copy=False).ravel()
        paired = np.isfinite(left_values) & np.isfinite(right_values)
        if int(paired.sum()) < MIN_PAIRED_OBSERVATIONS:
            raise FactorCorrelationError(
                f"因子 {left_definition.factor_name} 与 {right_definition.factor_name} "
                f"的共同有效暴露少于 {MIN_PAIRED_OBSERVATIONS} 个"
            )
        x = left_values[paired]
        y = right_values[paired]
        if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
            raise FactorCorrelationError(
                f"因子 {left_definition.factor_name} 与 {right_definition.factor_name} "
                "在共同有效样本中存在常数暴露，无法计算相关性"
            )
        correlation = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(correlation):
            raise FactorCorrelationError(
                f"因子 {left_definition.factor_name} 与 {right_definition.factor_name} 的相关性不是有限数值"
            )
        return correlation

    def _cache_matches(self, definitions: tuple[FactorDefinition, ...]) -> bool:
        # An empty DataFrame serializes to an empty CSV.  Rebuild that trivial
        # matrix instead of trying to parse it when the first factor is later
        # submitted after somebody has opened an empty library page.
        if not definitions:
            return False
        if not self.matrix_path.is_file() or not self.metadata_path.is_file():
            return False
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        expected = self._metadata_factors(definitions)
        if (
            metadata.get("cache_version") != CACHE_VERSION
            or metadata.get("method") != CORRELATION_METHOD
            or metadata.get("factors") != expected
        ):
            return False
        return all(self._exposure_path(item).is_file() for item in definitions)

    def _metadata_factors(self, definitions: tuple[FactorDefinition, ...]) -> list[dict[str, Any]]:
        return [
            {
                "factor_name": definition.factor_name,
                "expression_hash": definition.expression_hash,
                "data_signature": self._data_signature(definition),
            }
            for definition in definitions
        ]

    def _data_signature(self, definition: FactorDefinition) -> list[dict[str, Any]]:
        data_dir = Path(self.settings.data_dir).expanduser().resolve()
        if not (data_dir / DEFAULT_FILES["c"]).is_file() and (
            data_dir / "data" / DEFAULT_FILES["c"]
        ).is_file():
            data_dir = data_dir / "data"
        signature: list[dict[str, Any]] = []
        # ``c`` is the canonical alignment axis even for expressions that do
        # not reference close explicitly, so a close-matrix revision must
        # invalidate every cached exposure.
        for symbol in sorted(expression_data_symbols(definition.expression) | {"c"}):
            path = data_dir / DEFAULT_FILES[symbol]
            if not path.is_file():
                raise FactorCorrelationError(f"缺少相关性计算所需数据: {path}")
            stat = path.stat()
            signature.append(
                {
                    "symbol": symbol,
                    "file": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return signature

    def _read_matrix(self, definitions: tuple[FactorDefinition, ...]) -> pd.DataFrame:
        try:
            matrix = pd.read_csv(self.matrix_path, index_col=0, encoding="utf-8-sig")
            matrix.index = matrix.index.astype(str)
            matrix.columns = matrix.columns.astype(str)
            names = [item.factor_name for item in definitions]
            matrix = matrix.reindex(index=names, columns=names).astype(float)
        except (OSError, ValueError, TypeError) as exc:
            raise FactorCorrelationError("已保存的因子相关性矩阵无法读取") from exc
        if matrix.shape != (len(definitions), len(definitions)) or not np.isfinite(
            matrix.to_numpy(dtype=float)
        ).all():
            raise FactorCorrelationError("已保存的因子相关性矩阵不完整")
        return matrix

    def _read_exposure(self, definition: FactorDefinition) -> pd.DataFrame:
        path = self._exposure_path(definition)
        try:
            exposure = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            raise FactorCorrelationError(f"无法读取因子 {definition.factor_name} 的缓存暴露") from exc
        if not isinstance(exposure, pd.DataFrame) or exposure.empty:
            raise FactorCorrelationError(f"因子 {definition.factor_name} 的缓存暴露为空")
        return exposure

    def _write_exposure(self, definition: FactorDefinition, exposure: pd.DataFrame) -> None:
        path = self._exposure_path(definition)
        self.exposure_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=self.exposure_dir)
        os.close(descriptor)
        try:
            exposure.to_parquet(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _write_matrix(self, matrix: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".matrix.", suffix=".tmp", dir=self.root)
        os.close(descriptor)
        try:
            matrix.to_csv(temporary, encoding="utf-8-sig")
            os.replace(temporary, self.matrix_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _write_metadata(self, definitions: tuple[FactorDefinition, ...]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": CACHE_VERSION,
            "method": CORRELATION_METHOD,
            "threshold": self.threshold,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "factors": self._metadata_factors(definitions),
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".metadata.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.metadata_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _exposure_path(self, definition: FactorDefinition) -> Path:
        key = hashlib.sha256(
            f"{definition.factor_name}\x00{definition.expression_hash}".encode("utf-8")
        ).hexdigest()
        return self.exposure_dir / f"{key}.parquet"

    def _violations(self, matrix: pd.DataFrame) -> list[dict[str, Any]]:
        names = list(matrix.index)
        violations: list[dict[str, Any]] = []
        for row, left in enumerate(names):
            for column in range(row + 1, len(names)):
                correlation = float(matrix.iat[row, column])
                if abs(correlation) > self.threshold:
                    violations.append(
                        {
                            "factor_a": left,
                            "factor_b": names[column],
                            "correlation": correlation,
                            "abs_correlation": abs(correlation),
                        }
                    )
        return violations

    def _payload(self, matrix: pd.DataFrame) -> dict[str, Any]:
        violations = self._violations(matrix)
        names = list(matrix.index)
        off_diagonal = [
            abs(float(matrix.iat[row, column]))
            for row in range(len(names))
            for column in range(row + 1, len(names))
        ]
        updated_at = None
        if self.metadata_path.is_file():
            try:
                updated_at = json.loads(self.metadata_path.read_text(encoding="utf-8")).get("updated_at")
            except (OSError, ValueError):
                pass
        return {
            "factor_names": names,
            "matrix": matrix.values.tolist(),
            "threshold": self.threshold,
            "method": CORRELATION_METHOD,
            "sample_definition": "所有对齐且有限的日期-证券观测值上的 pooled Pearson 相关系数",
            "max_abs_off_diagonal": max(off_diagonal, default=0.0),
            "passed": not violations,
            "violations": violations,
            "updated_at": updated_at,
        }
