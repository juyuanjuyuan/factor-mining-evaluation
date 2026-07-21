#!/usr/bin/env python3
"""Validate the generic one-batch-per-JSON registry contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from _bootstrap import PROJECT_ROOT

from factor_registry import load_factor_batch, validate_factor_batch


def sample() -> dict:
    return {
        "schema_version": 1,
        "batch_id": "test_batch",
        "batch_name": "Test batch",
        "batch_version": 1,
        "generated_at": "2026-07-03T00:00:00+08:00",
        "source": {"type": "test", "name": "synthetic"},
        "formula_language": "engine expression",
        "factor_count": 1,
        "factors": [
            {
                "factor_name": "test_factor",
                "entered_at": "2026-07-03T00:00:00+08:00",
                "expression": "rank_cs(delta(c, 5))",
                "required_symbols": ["c"],
            }
        ],
    }


def main() -> None:
    payload = sample()
    validate_factor_batch(payload)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "test_batch.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_factor_batch(path)["factor_count"] == 1

    duplicate = copy.deepcopy(payload)
    duplicate["factors"].append(copy.deepcopy(duplicate["factors"][0]))
    duplicate["factor_count"] = 2
    try:
        validate_factor_batch(duplicate)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate factor names must fail")

    result_field = copy.deepcopy(payload)
    result_field["factors"][0]["ir"] = 1.5
    try:
        validate_factor_batch(result_field)
    except ValueError as exc:
        assert "evaluation-result" in str(exc)
    else:
        raise AssertionError("evaluation results must not enter factor registry")

    missing_timezone = copy.deepcopy(payload)
    missing_timezone["factors"][0]["entered_at"] = "2026-07-03T00:00:00"
    try:
        validate_factor_batch(missing_timezone)
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:
        raise AssertionError("entered_at without timezone must fail")

    template = PROJECT_ROOT / "factor_registry/templates/factor_batch.template.json"
    validate_factor_batch(json.loads(template.read_text(encoding="utf-8")))
    print("generic factor batch registry contract passed")


if __name__ == "__main__":
    main()
