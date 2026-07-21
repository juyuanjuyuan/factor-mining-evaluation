"""PyTorch MPS backend for training-only genetic-programming fitness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from evaluators.tradability import open_limit_entry_masks

from ..fitness import FitnessContext, FitnessResult
from ..tree import ExpressionTree


SUPPORTED_TERMINALS = frozenset(
    {"c", "o", "h", "l", "vol", "amt", "vwap", "cap", "pct(c, 1)"}
)


def _import_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "MPS 训练后端需要可正常导入的 PyTorch；"
            f"当前导入失败：{type(exc).__name__}: {exc}"
        ) from exc
    return torch


def mps_runtime_status() -> dict[str, Any]:
    """Return a JSON-safe MPS capability report without raising."""

    try:
        torch = _import_torch()
    except RuntimeError as exc:
        return {
            "name": "mps",
            "available": False,
            "reason": str(exc),
            "torch_version": None,
            "device_name": None,
        }
    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    reason = None
    if not built:
        reason = "当前 PyTorch 未构建 MPS 支持"
    elif not available:
        reason = "当前 macOS 或设备不可用 MPS"
    device_name = None
    if available:
        getter = getattr(torch.backends.mps, "get_name", None)
        device_name = str(getter()) if callable(getter) else "Apple GPU"
    return {
        "name": "mps",
        "available": available,
        "reason": reason,
        "torch_version": str(torch.__version__),
        "device_name": device_name,
    }


class MPSFitnessBackend:
    """Evaluate GP trees on Apple GPU while preserving the pandas contract."""

    name = "mps"
    dtype_name = "float32"
    rolling_column_chunk = 128

    def __init__(self, context: FitnessContext):
        self.context = context
        self.torch = _import_torch()
        if not self.torch.backends.mps.is_available():
            status = mps_runtime_status()
            raise RuntimeError(status.get("reason") or "MPS 不可用")
        self.device = self.torch.device("mps")
        self.dtype = self.torch.float32
        close = context.market_data["c"]
        self.index = close.index
        self.columns = close.columns
        positions = self.index.get_indexer(context.signal_index)
        if (positions < 0).any():
            raise ValueError("MPS 训练日期不存在于完整行情时间线")
        self.signal_positions = self.torch.as_tensor(
            positions,
            dtype=self.torch.int64,
            device=self.device,
        )
        self.data = {
            symbol: self._frame_tensor(frame, reference=close)
            for symbol, frame in context.market_data.items()
            if symbol != "st"
        }
        if "st" in context.market_data:
            st = (
                context.market_data["st"]
                .reindex(index=self.index, columns=self.columns)
                .fillna(False)
                .astype(bool)
            )
            self.st = self.torch.as_tensor(
                st.to_numpy(dtype=bool, copy=False),
                dtype=self.torch.bool,
                device=self.device,
            )
        else:
            self.st = None
        self.forward_return = self.torch.as_tensor(
            context.forward_return.to_numpy(dtype=np.float32, copy=True),
            dtype=self.dtype,
            device=self.device,
        )
        self.style_exposures = tuple(
            self.torch.as_tensor(
                frame.reindex(
                    index=context.signal_index,
                    columns=self.columns,
                ).to_numpy(dtype=np.float32, copy=True),
                dtype=self.dtype,
                device=self.device,
            )
            for frame in context.style_exposures.values()
        )
        self.untradeable = self._build_untradeable_mask()

    def _frame_tensor(self, frame: pd.DataFrame, *, reference: pd.DataFrame):
        aligned = frame.reindex(index=reference.index, columns=reference.columns)
        return self.torch.as_tensor(
            aligned.to_numpy(dtype=np.float32, copy=True),
            dtype=self.dtype,
            device=self.device,
        )

    def _build_untradeable_mask(self):
        if self.context.preprocess_mode != "paper_local":
            return None
        required = {"o", "c", "limit"}
        if not required <= set(self.context.market_data) or self.st is None:
            raise KeyError("paper_local MPS 预处理缺少 o/c/limit/st")
        limit_up, limit_down = open_limit_entry_masks(
            self.context.market_data["o"],
            self.context.market_data["c"],
            self.context.market_data["limit"],
        )
        positions = self.index.get_indexer(self.context.signal_index) + 1
        mask = np.zeros((len(positions), len(self.columns)), dtype=bool)
        valid = positions < len(self.index)
        limit_values = (
            limit_up.to_numpy(dtype=bool, copy=False)
            | limit_down.to_numpy(dtype=bool, copy=False)
        )
        st_values = (
            self.context.market_data["st"]
            .reindex(index=self.index, columns=self.columns)
            .fillna(False)
            .astype(bool)
            .to_numpy(dtype=bool, copy=False)
        )
        mask[valid] = limit_values[positions[valid]] | st_values[positions[valid]]
        return self.torch.as_tensor(mask, dtype=self.torch.bool, device=self.device)

    @property
    def metadata(self) -> dict[str, Any]:
        status = mps_runtime_status()
        return {
            "name": self.name,
            "device": status.get("device_name") or "mps",
            "dtype": self.dtype_name,
            "torch_version": status.get("torch_version"),
            "implicit_cpu_fallback": False,
        }

    def close(self) -> None:
        empty_cache = getattr(self.torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def _ensure_matrix(self, value, operator: str):
        if not isinstance(value, self.torch.Tensor) or value.ndim != 2:
            raise TypeError(f"{operator} 需要日期×股票矩阵")
        return value

    def _shift(self, value, periods: int):
        value = self._ensure_matrix(value, "delay")
        if periods < 1:
            raise ValueError("periods must be positive")
        result = self.torch.full_like(value, self.torch.nan)
        if periods < value.shape[0]:
            result[periods:] = value[:-periods]
        return result

    def _rolling_sum_count(self, value, window: int, *, min_periods: int):
        value = self._ensure_matrix(value, "rolling")
        finite = self.torch.isfinite(value)
        cleaned = self.torch.where(finite, value, self.torch.zeros_like(value))
        sums = self._rolling_raw_sum(cleaned, window)
        count = self._rolling_raw_sum(finite.to(self.dtype), window)
        valid = count >= float(min_periods)
        return sums, count, valid

    def _rolling_raw_sum(self, value, window: int):
        """Trailing sum as one MPS convolution, avoiding long-prefix cancellation."""

        functional = self.torch.nn.functional
        channels = value.transpose(0, 1)[:, None, :]
        padded = functional.pad(channels, (window - 1, 0))
        kernel = self.torch.ones(
            (1, 1, window),
            dtype=self.dtype,
            device=self.device,
        )
        return functional.conv1d(padded, kernel).squeeze(1).transpose(0, 1)

    def _rolling_mean(self, value, window: int):
        sums, count, valid = self._rolling_sum_count(
            value,
            window,
            min_periods=max(1, window // 2),
        )
        return self.torch.where(valid, sums / count.clamp_min(1), self.torch.nan)

    def _rolling_std(self, value, window: int):
        value = self._ensure_matrix(value, "ts_std")
        outputs = []
        for start in range(0, value.shape[1], self.rolling_column_chunk):
            chunk = value[:, start : start + self.rolling_column_chunk]
            windows = self._rolling_windows(chunk, window)
            finite = self.torch.isfinite(windows)
            count = finite.sum(dim=2).to(self.dtype)
            cleaned = self.torch.where(finite, windows, self.torch.zeros_like(windows))
            mean = cleaned.sum(dim=2) / count.clamp_min(1)
            variance = self.torch.where(
                finite,
                (windows - mean[:, :, None]).square(),
                self.torch.zeros_like(windows),
            ).sum(dim=2) / count.clamp_min(1)
            valid = count >= max(1, window // 2)
            outputs.append(
                self.torch.where(valid, self.torch.sqrt(variance.clamp_min(0)), self.torch.nan)
            )
        return self.torch.cat(outputs, dim=1)

    def _rolling_windows(self, value, window: int):
        padded = self.torch.nn.functional.pad(
            value,
            (0, 0, window - 1, 0),
            value=self.torch.nan,
        )
        return padded.unfold(0, window, 1)

    def _rolling_covariance(self, left, right, window: int, *, correlation: bool):
        left = self._ensure_matrix(left, "ts_corr" if correlation else "ts_cov")
        right = self._ensure_matrix(right, "ts_corr" if correlation else "ts_cov")
        min_periods = max(2, window // 2)
        outputs = []
        for start in range(0, left.shape[1], self.rolling_column_chunk):
            x_windows = self._rolling_windows(
                left[:, start : start + self.rolling_column_chunk],
                window,
            )
            y_windows = self._rolling_windows(
                right[:, start : start + self.rolling_column_chunk],
                window,
            )
            finite = self.torch.isfinite(x_windows) & self.torch.isfinite(y_windows)
            count = finite.sum(dim=2).to(self.dtype)
            denominator = count.clamp_min(1)
            clean_x = self.torch.where(finite, x_windows, self.torch.zeros_like(x_windows))
            clean_y = self.torch.where(finite, y_windows, self.torch.zeros_like(y_windows))
            mean_x = clean_x.sum(dim=2) / denominator
            mean_y = clean_y.sum(dim=2) / denominator
            centered_x = self.torch.where(
                finite,
                x_windows - mean_x[:, :, None],
                self.torch.zeros_like(x_windows),
            )
            centered_y = self.torch.where(
                finite,
                y_windows - mean_y[:, :, None],
                self.torch.zeros_like(y_windows),
            )
            covariance = (centered_x * centered_y).sum(dim=2) / denominator
            valid = count >= min_periods
            if correlation:
                scale = self.torch.sqrt(
                    centered_x.square().sum(dim=2) * centered_y.square().sum(dim=2)
                )
                # A constant float32 window can acquire a tiny nonzero centered
                # sum from mean-reduction rounding.  Pandas correctly defines
                # its correlation as NaN, so reject exact zero-range inputs
                # before using the numerical denominator.
                negative_inf = self.torch.full_like(x_windows, -self.torch.inf)
                positive_inf = self.torch.full_like(x_windows, self.torch.inf)
                x_range = (
                    self.torch.where(finite, x_windows, negative_inf).amax(dim=2)
                    - self.torch.where(finite, x_windows, positive_inf).amin(dim=2)
                )
                y_range = (
                    self.torch.where(finite, y_windows, negative_inf).amax(dim=2)
                    - self.torch.where(finite, y_windows, positive_inf).amin(dim=2)
                )
                usable = (
                    valid
                    & (x_range > 0)
                    & (y_range > 0)
                    & (scale > self.torch.finfo(self.dtype).eps)
                )
                result = self.torch.where(
                    usable,
                    (centered_x * centered_y).sum(dim=2) / scale,
                    self.torch.nan,
                )
            else:
                result = self.torch.where(valid, covariance, self.torch.nan)
            outputs.append(result)
        return self.torch.cat(outputs, dim=1)

    def _rolling_extreme(self, value, window: int, *, maximum: bool, argument: bool):
        value = self._ensure_matrix(value, "ts_argmax" if argument and maximum else "rolling")
        functional = self.torch.nn.functional
        finite = self.torch.isfinite(value)
        encoded = self.torch.where(
            finite,
            value if maximum else -value,
            self.torch.full_like(value, -self.torch.inf),
        )
        channels = encoded.transpose(0, 1)[:, None, :]
        padded = functional.pad(channels, (window - 1, 0), value=-self.torch.inf)
        pooled, indices = functional.max_pool1d(
            padded,
            window,
            stride=1,
            return_indices=True,
        )
        best = pooled.squeeze(1).transpose(0, 1)
        if not maximum:
            best = -best
        _, count, valid = self._rolling_sum_count(
            value,
            window,
            min_periods=max(1, window // 2),
        )
        if argument:
            source_positions = indices.squeeze(1).transpose(0, 1) - (window - 1)
            rows = self.torch.arange(
                value.shape[0],
                device=self.device,
                dtype=source_positions.dtype,
            )[:, None]
            starts = (rows - (window - 1)).clamp_min(0)
            output = (source_positions - starts + 1).to(self.dtype)
        else:
            output = best
        return self.torch.where(valid & (count > 0), output, self.torch.nan)

    def _rolling_rank(self, value, window: int):
        value = self._ensure_matrix(value, "ts_rank")
        outputs = []
        for start in range(0, value.shape[1], self.rolling_column_chunk):
            chunk = value[:, start : start + self.rolling_column_chunk]
            windows = self._rolling_windows(chunk, window)
            newest = windows[:, :, -1]
            finite = self.torch.isfinite(windows) & self.torch.isfinite(newest[:, :, None])
            count = finite.sum(dim=2).to(self.dtype)
            less = (finite & (windows < newest[:, :, None])).sum(dim=2).to(self.dtype)
            equal = (finite & (windows == newest[:, :, None])).sum(dim=2).to(self.dtype)
            rank = (less + (equal + 1.0) / 2.0) / count.clamp_min(1)
            outputs.append(
                self.torch.where(
                    self.torch.isfinite(newest) & (count >= max(1, window // 2)),
                    rank,
                    self.torch.nan,
                )
            )
        return self.torch.cat(outputs, dim=1)

    def _rolling_product(self, value, window: int):
        value = self._ensure_matrix(value, "ts_product")
        lengths = self.torch.arange(
            1,
            value.shape[0] + 1,
            device=self.device,
            dtype=self.dtype,
        ).clamp_max(window)[:, None]
        outputs = []
        for start in range(0, value.shape[1], self.rolling_column_chunk):
            windows = self._rolling_windows(
                value[:, start : start + self.rolling_column_chunk],
                window,
            )
            finite = self.torch.isfinite(windows)
            count = finite.sum(dim=2).to(self.dtype)
            product = self.torch.prod(
                self.torch.where(finite, windows, self.torch.ones_like(windows)),
                dim=2,
            )
            valid = (count >= max(1, window // 2)) & (count == lengths)
            outputs.append(self.torch.where(valid, product, self.torch.nan))
        return self.torch.cat(outputs, dim=1)

    def _decay_linear(self, value, window: int):
        value = self._ensure_matrix(value, "decay_linear")
        lengths = self.torch.arange(
            1,
            value.shape[0] + 1,
            device=self.device,
            dtype=self.dtype,
        ).clamp_max(window)[:, None]
        positions = self.torch.arange(
            window,
            device=self.device,
            dtype=self.dtype,
        )[None, None, :]
        leading = (float(window) - lengths)[:, :, None]
        weights = (positions - leading + 1.0).clamp_min(0)
        outputs = []
        for start in range(0, value.shape[1], self.rolling_column_chunk):
            windows = self._rolling_windows(
                value[:, start : start + self.rolling_column_chunk],
                window,
            )
            finite = self.torch.isfinite(windows)
            count = finite.sum(dim=2).to(self.dtype)
            numerator = self.torch.where(
                finite,
                windows * weights,
                self.torch.zeros_like(windows),
            ).sum(dim=2)
            denominator = self.torch.where(
                finite,
                weights,
                self.torch.zeros_like(windows),
            ).sum(dim=2)
            valid = (count >= max(1, window // 2)) & (denominator > 0)
            outputs.append(
                self.torch.where(valid, numerator / denominator, self.torch.nan)
            )
        return self.torch.cat(outputs, dim=1)

    def _rank_last_dimension(self, value):
        value = self._ensure_matrix(value, "rank_cs")
        finite = self.torch.isfinite(value)
        safe = self.torch.where(finite, value, self.torch.full_like(value, self.torch.inf))
        sorted_values, order = self.torch.sort(safe, dim=1, stable=True)
        sorted_finite = self.torch.gather(finite, 1, order)
        boundary = self.torch.zeros_like(sorted_finite)
        boundary[:, 0] = sorted_finite[:, 0]
        boundary[:, 1:] = sorted_finite[:, 1:] & (
            sorted_values[:, 1:] != sorted_values[:, :-1]
        )
        group = self.torch.cumsum(boundary.to(self.torch.int64), dim=1) - 1
        group = group.clamp_min(0)
        positions = self.torch.arange(
            1,
            value.shape[1] + 1,
            device=self.device,
            dtype=self.dtype,
        )[None, :].expand_as(value)
        group_sums = self.torch.zeros_like(value).scatter_add(
            1,
            group,
            self.torch.where(sorted_finite, positions, self.torch.zeros_like(positions)),
        )
        group_counts = self.torch.zeros_like(value).scatter_add(
            1,
            group,
            sorted_finite.to(self.dtype),
        )
        average = group_sums / group_counts.clamp_min(1)
        sorted_rank = self.torch.gather(average, 1, group)
        finite_count = finite.sum(dim=1, keepdim=True).to(self.dtype)
        sorted_rank = self.torch.where(
            sorted_finite,
            sorted_rank / finite_count.clamp_min(1),
            self.torch.nan,
        )
        result = self.torch.full_like(value, self.torch.nan)
        return result.scatter(1, order, sorted_rank)

    def _cross_sectional_median(self, value):
        finite = self.torch.isfinite(value)
        safe = self.torch.where(finite, value, self.torch.full_like(value, self.torch.inf))
        sorted_values = self.torch.sort(safe, dim=1).values
        count = finite.sum(dim=1)
        lower = ((count - 1).clamp_min(0) // 2).to(self.torch.int64)
        upper = (count.clamp_min(1) // 2).to(self.torch.int64)
        low_value = self.torch.gather(sorted_values, 1, lower[:, None]).squeeze(1)
        high_value = self.torch.gather(sorted_values, 1, upper[:, None]).squeeze(1)
        median = (low_value + high_value) / 2.0
        return self.torch.where(count > 0, median, self.torch.nan)

    def _zscore(self, value):
        finite = self.torch.isfinite(value)
        count = finite.sum(dim=1, keepdim=True).to(self.dtype)
        cleaned = self.torch.where(finite, value, self.torch.zeros_like(value))
        mean = cleaned.sum(dim=1, keepdim=True) / count.clamp_min(1)
        variance = self.torch.where(finite, (value - mean).square(), self.torch.zeros_like(value)).sum(
            dim=1,
            keepdim=True,
        ) / count.clamp_min(1)
        std = self.torch.sqrt(variance.clamp_min(0))
        valid = finite & (std > 0) & (count > 0)
        return self.torch.where(valid, (value - mean) / std, self.torch.nan)

    def _residualize(self, factor, exposures: Sequence[Any]):
        if not exposures:
            return factor
        x = self.torch.stack(tuple(exposures), dim=2)
        finite_x = self.torch.isfinite(x).all(dim=2)
        valid = self.torch.isfinite(factor) & finite_x
        count = valid.sum(dim=1).to(self.dtype)
        clean_y = self.torch.where(valid, factor, self.torch.zeros_like(factor))
        clean_x = self.torch.where(valid[:, :, None], x, self.torch.zeros_like(x))
        denominator = count.clamp_min(1)[:, None]
        y_mean = clean_y.sum(dim=1) / count.clamp_min(1)
        x_mean = clean_x.sum(dim=1) / denominator
        centered_y = self.torch.where(valid, factor - y_mean[:, None], self.torch.zeros_like(factor))
        centered_x = self.torch.where(
            valid[:, :, None],
            x - x_mean[:, None, :],
            self.torch.zeros_like(x),
        )
        xx = self.torch.einsum("tnk,tnl->tkl", centered_x, centered_x)
        xy = self.torch.einsum("tnk,tn->tk", centered_x, centered_y)
        xx_cpu = xx.detach().cpu().numpy().astype(np.float64, copy=False)
        xy_cpu = xy.detach().cpu().numpy().astype(np.float64, copy=False)
        beta_cpu = np.einsum(
            "tkl,tl->tk",
            np.linalg.pinv(xx_cpu, rcond=1e-10),
            xy_cpu,
        ).astype(np.float32)
        beta = self.torch.as_tensor(beta_cpu, dtype=self.dtype, device=self.device)
        fitted_centered = self.torch.einsum("tnk,tk->tn", centered_x, beta)
        residual = factor - y_mean[:, None] - fitted_centered
        enough = count >= float(len(exposures) + 3)
        return self.torch.where(valid & enough[:, None], residual, self.torch.nan)

    def _terminal(self, value: str):
        if value in self.data:
            return self.data[value]
        if value == "pct(c, 1)":
            close = self.data["c"]
            return close / self._shift(close, 1) - 1.0
        raise ValueError(
            f"MPS 后端不支持终端 {value!r}；支持：{sorted(SUPPORTED_TERMINALS)}"
        )

    def _evaluate_node(self, node: ExpressionTree):
        if node.kind == "terminal":
            return self._terminal(str(node.value))
        if node.kind == "constant":
            return self.torch.tensor(float(node.value), dtype=self.dtype, device=self.device)
        children = tuple(self._evaluate_node(child) for child in node.children)
        name = str(node.value)
        if name in {"add", "sub", "mul", "div"}:
            left, right = children
            if name == "add":
                return left + right
            if name == "sub":
                return left - right
            if name == "mul":
                return left * right
            denominator = self.torch.where(
                self.torch.abs(right) > 1e-6,
                right,
                self.torch.ones_like(right),
            )
            return left / denominator
        value = children[0]
        if name == "abs":
            return self.torch.abs(value)
        if name == "sqrt":
            return self.torch.sqrt(self.torch.abs(value))
        if name == "log":
            return self.torch.log(self.torch.abs(value) + 1e-6)
        if name == "inv":
            denominator = self.torch.where(
                self.torch.abs(value) > 1e-6,
                value,
                self.torch.ones_like(value),
            )
            return 1.0 / denominator
        if name == "neg":
            return -value
        if name == "rank_cs":
            return self._rank_last_dimension(value)
        if name == "scale_cs":
            value = self._ensure_matrix(value, "scale_cs")
            denominator = self.torch.where(
                self.torch.isfinite(value),
                self.torch.abs(value),
                self.torch.zeros_like(value),
            ).sum(dim=1, keepdim=True)
            return self.torch.where(denominator > 0, value / denominator, self.torch.nan)
        if name == "signed_power":
            return self.torch.sign(value) * self.torch.pow(
                self.torch.abs(value),
                float(node.parameter),
            )
        window = int(node.parameter)
        if name == "delay":
            return self._shift(value, window)
        if name == "delta":
            return value - self._shift(value, window)
        if name == "ts_corr":
            return self._rolling_covariance(value, children[1], window, correlation=True)
        if name == "ts_cov":
            return self._rolling_covariance(value, children[1], window, correlation=False)
        if name == "decay_linear":
            return self._decay_linear(value, window)
        if name == "ts_min":
            return self._rolling_extreme(value, window, maximum=False, argument=False)
        if name == "ts_max":
            return self._rolling_extreme(value, window, maximum=True, argument=False)
        if name == "ts_argmin":
            return self._rolling_extreme(value, window, maximum=False, argument=True)
        if name == "ts_argmax":
            return self._rolling_extreme(value, window, maximum=True, argument=True)
        if name == "ts_rank":
            return self._rolling_rank(value, window)
        if name == "ts_sum":
            sums, _, valid = self._rolling_sum_count(
                value,
                window,
                min_periods=max(1, window // 2),
            )
            return self.torch.where(valid, sums, self.torch.nan)
        if name == "ts_product":
            return self._rolling_product(value, window)
        if name == "ts_std":
            return self._rolling_std(value, window)
        raise ValueError(f"MPS 后端没有实现 GP 算子 {name!r}")

    def evaluate_raw_tree_tensor(self, tree: ExpressionTree):
        result = self._evaluate_node(tree)
        result = self._ensure_matrix(result, "GP expression")
        if tuple(result.shape) != (len(self.index), len(self.columns)):
            raise ValueError(
                f"MPS 因子矩阵形状 {tuple(result.shape)} 与行情轴不一致"
            )
        return self.torch.where(self.torch.isfinite(result), result, self.torch.nan)

    def _preprocess(self, raw):
        factor = self.torch.index_select(raw, 0, self.signal_positions)
        if self.context.preprocess_mode == "none":
            return factor
        if self.context.preprocess_mode == "paper_local":
            factor = self.torch.where(self.untradeable, self.torch.nan, factor)
        median = self._cross_sectional_median(factor)
        mad = self._cross_sectional_median(self.torch.abs(factor - median[:, None]))
        factor = self.torch.maximum(
            self.torch.minimum(factor, (median + 5.0 * mad)[:, None]),
            (median - 5.0 * mad)[:, None],
        )
        factor = self._residualize(factor, self.style_exposures)
        return self._zscore(factor)

    def _daily_rank_ic(self, factor):
        returns = self.forward_return
        valid = self.torch.isfinite(factor) & self.torch.isfinite(returns)
        factor_rank = self._rank_last_dimension(
            self.torch.where(valid, factor, self.torch.nan)
        )
        return_rank = self._rank_last_dimension(
            self.torch.where(valid, returns, self.torch.nan)
        )
        count = valid.sum(dim=1)
        count_float = count.to(self.dtype).clamp_min(1)[:, None]
        factor_mean = self.torch.where(
            valid,
            factor_rank,
            self.torch.zeros_like(factor_rank),
        ).sum(dim=1, keepdim=True) / count_float
        return_mean = self.torch.where(
            valid,
            return_rank,
            self.torch.zeros_like(return_rank),
        ).sum(dim=1, keepdim=True) / count_float
        x = self.torch.where(valid, factor_rank - factor_mean, self.torch.zeros_like(factor_rank))
        y = self.torch.where(valid, return_rank - return_mean, self.torch.zeros_like(return_rank))
        numerator = (x * y).sum(dim=1)
        denominator = self.torch.sqrt(x.square().sum(dim=1) * y.square().sum(dim=1))
        usable = (count >= 3) & (denominator > 0)
        ic = self.torch.where(usable, numerator / denominator, self.torch.nan)
        return ic, count

    def _fitness_result(
        self,
        tree: ExpressionTree,
        *,
        parsimony_coefficient: float,
    ) -> FitnessResult:
        expression = tree.to_expression()
        try:
            factor = self._preprocess(self.evaluate_raw_tree_tensor(tree))
            ic_tensor, counts = self._daily_rank_ic(factor)
            ic = ic_tensor.detach().cpu().numpy().astype(float, copy=False)
            finite = np.isfinite(ic)
            finite_ic = ic[finite]
            if len(finite_ic) < self.context.minimum_ic_days:
                raise ValueError(
                    f"only {len(finite_ic)} finite IC days; "
                    f"requires {self.context.minimum_ic_days}"
                )
            ic_mean = float(finite_ic.mean())
            ic_std = float(finite_ic.std(ddof=1))
            ir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan
            adjusted = ic_mean - float(parsimony_coefficient) * tree.node_count
            pair_count = int(
                counts.detach().cpu().numpy()[finite].sum(dtype=np.int64)
            )
            return FitnessResult(
                expression=expression,
                raw_fitness=ic_mean if np.isfinite(ic_mean) else None,
                adjusted_fitness=adjusted if np.isfinite(adjusted) else None,
                ic_mean=ic_mean if np.isfinite(ic_mean) else None,
                ic_std=ic_std if np.isfinite(ic_std) else None,
                ir=ir if np.isfinite(ir) else None,
                ic_count=int(len(finite_ic)),
                pair_count=pair_count,
                node_count=tree.node_count,
                depth=tree.depth,
            )
        except Exception as exc:
            return FitnessResult(
                expression=expression,
                raw_fitness=None,
                adjusted_fitness=None,
                ic_mean=None,
                ic_std=None,
                ir=None,
                ic_count=0,
                pair_count=0,
                node_count=tree.node_count,
                depth=tree.depth,
                error=f"{type(exc).__name__}: {exc}",
            )

    def evaluate_many(
        self,
        trees: Sequence[ExpressionTree],
        *,
        parsimony_coefficient: float,
    ) -> tuple[FitnessResult, ...]:
        return tuple(
            self._fitness_result(
                tree,
                parsimony_coefficient=parsimony_coefficient,
            )
            for tree in trees
        )

    def evaluate_processed_tree(self, tree: ExpressionTree) -> pd.DataFrame:
        processed = self._preprocess(self.evaluate_raw_tree_tensor(tree))
        values = processed.detach().cpu().numpy().astype(float, copy=False)
        return pd.DataFrame(
            values,
            index=self.context.signal_index,
            columns=self.columns,
        )
