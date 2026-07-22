"""Validate stored evaluation pipelines without changing their explicit order."""

from __future__ import annotations

from evaluators import evaluation_method_names, resolve_evaluation_methods
from evaluators.base import REGISTERED_EVALUATION_METHODS, method_metadata


def validate_ordered_pipeline(names: list[str] | None) -> list[str]:
    """Validate an ordered pipeline while preserving repeats.

    Duplicate-free shorthand keeps the historical dependency auto-resolution.
    A pipeline containing repeated methods is treated as an exact execution
    plan, so every dependency must already have been produced by an earlier
    step.
    """

    if not names:
        return list(evaluation_method_names(resolve_evaluation_methods(None)))
    normalized = [str(name).strip() for name in names]
    if any(not name for name in normalized):
        raise ValueError("评价方法名称不能为空")
    unknown = [
        name for name in normalized if name not in REGISTERED_EVALUATION_METHODS
    ]
    if unknown:
        raise ValueError(f"未知评价方法: {', '.join(unknown)}")

    executed: set[str] = set()
    problems: list[str] = []
    for position, name in enumerate(normalized, start=1):
        metadata = method_metadata(REGISTERED_EVALUATION_METHODS[name])
        missing = [required for required in metadata.requires if required not in executed]
        if missing:
            problems.append(
                f"第 {position} 步 {name} 需要先执行 {', '.join(missing)}"
            )
        executed.update(metadata.provides)
    if not problems:
        return normalized
    if len(set(normalized)) == len(normalized):
        return list(
            evaluation_method_names(resolve_evaluation_methods(normalized))
        )
    raise ValueError("；".join(problems))
