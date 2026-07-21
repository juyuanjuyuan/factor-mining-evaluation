"""Validation contract for evaluator-independent factor batch JSON files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from engine import parse_and_validate_expression
from paths import PROJECT_ROOT


SCHEMA_VERSION = 1
FACTOR_DEFINITION_LANGUAGE = "engine expression"
DEFAULT_FACTOR_BATCH = (
    PROJECT_ROOT / "factor_registry" / "webapp_factor_library.json"
)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RESULT_FIELDS = {
    "abs_ir",
    "evaluated_at",
    "ic_mean",
    "ic_std",
    "ir",
    "metrics",
    "performance",
    "returns",
}


def _validate_project(value: Any, *, location: str) -> None:
    """Allow an optional human-readable project classification."""
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{location} project must be a nonempty string")


@dataclass(frozen=True)
class RegisteredFactor:
    """One executable factor loaded from the canonical JSON registry."""

    number: int
    factor_name: str
    entered_at: str
    implementation_set: str
    expression: str
    paper_expression: str
    required_symbols: tuple[str, ...]
    uses_proxy: bool
    proxy_description: str

    @property
    def name(self) -> str:
        """Compatibility alias used by the evaluation runners."""
        return self.factor_name


def validate_factor_batch(
    payload: Mapping[str, Any],
    *,
    validate_expressions: bool = True,
) -> None:
    """Validate one source batch without allowing evaluation results."""
    required_batch = {
        "schema_version",
        "batch_id",
        "batch_name",
        "batch_version",
        "generated_at",
        "source",
        "formula_language",
        "factor_count",
        "factors",
    }
    missing = required_batch - set(payload)
    if missing:
        raise ValueError(f"factor batch is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {payload['schema_version']!r}"
        )
    batch_id = payload["batch_id"]
    if not isinstance(batch_id, str) or not _IDENTIFIER.fullmatch(batch_id):
        raise ValueError(f"invalid batch_id: {batch_id!r}")
    if not isinstance(payload["batch_name"], str) or not payload["batch_name"].strip():
        raise ValueError("batch_name must be a nonempty string")
    _validate_project(payload.get("project"), location="factor batch")
    if not isinstance(payload["batch_version"], int) or payload["batch_version"] < 1:
        raise ValueError("batch_version must be a positive integer")
    source = payload["source"]
    if not isinstance(source, Mapping) or not source.get("name"):
        raise ValueError("source must be an object with a nonempty name")
    formula_language = payload["formula_language"]
    if not isinstance(formula_language, str) or not formula_language.strip():
        raise ValueError("formula_language must be a nonempty string")

    factors = payload["factors"]
    if not isinstance(factors, list) or not factors:
        raise ValueError("factors must be a nonempty list")
    if payload["factor_count"] != len(factors):
        raise ValueError(
            "factor_count does not match factors length: "
            f"{payload['factor_count']} != {len(factors)}"
        )

    names: list[str] = []
    for position, factor in enumerate(factors, start=1):
        if not isinstance(factor, Mapping):
            raise ValueError(f"factor {position} must be an object")
        missing_factor = {
            "factor_name",
            "entered_at",
            "expression",
            "required_symbols",
        } - set(factor)
        if missing_factor:
            raise ValueError(
                f"factor {position} is missing fields: {sorted(missing_factor)}"
            )
        _validate_project(factor.get("project"), location=f"factor {position}")
        forbidden = _RESULT_FIELDS & set(factor)
        if forbidden:
            raise ValueError(
                f"factor {position} contains evaluation-result fields: "
                f"{sorted(forbidden)}"
            )
        name = factor["factor_name"]
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"factor {position} has invalid factor_name: {name!r}")
        names.append(name)
        entered_at = factor["entered_at"]
        if not isinstance(entered_at, str):
            raise ValueError(f"factor {name} entered_at must be an ISO timestamp")
        try:
            parsed_entered_at = datetime.fromisoformat(entered_at)
        except ValueError as exc:
            raise ValueError(
                f"factor {name} entered_at must be an ISO timestamp"
            ) from exc
        if (
            parsed_entered_at.tzinfo is None
            or parsed_entered_at.utcoffset() is None
        ):
            raise ValueError(
                f"factor {name} entered_at must include a timezone offset"
            )
        expression = factor["expression"]
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"factor {name} has an empty expression")
        symbols = factor["required_symbols"]
        if (
            not isinstance(symbols, list)
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
            or len(symbols) != len(set(symbols))
        ):
            raise ValueError(
                f"factor {name} required_symbols must be unique strings"
            )
        if (
            validate_expressions
            and formula_language == FACTOR_DEFINITION_LANGUAGE
        ):
            parse_and_validate_expression(expression)
    if len(names) != len(set(names)):
        raise ValueError("factor batch has duplicate factor_name values")


def load_factor_batch(
    path: str | Path,
    *,
    validate_expressions: bool = True,
) -> dict[str, Any]:
    """Load and validate one batch, including its canonical filename."""
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"factor batch root must be an object: {resolved}")
    validate_factor_batch(
        payload,
        validate_expressions=validate_expressions,
    )
    if resolved.stem != payload["batch_id"]:
        raise ValueError(
            f"filename must match batch_id: {resolved.stem!r} != "
            f"{payload['batch_id']!r}"
        )
    return payload


def load_registered_factors(
    path: str | Path = DEFAULT_FACTOR_BATCH,
    *,
    validate_expressions: bool = True,
) -> tuple[RegisteredFactor, ...]:
    """Load executable factor definitions from the canonical registry JSON."""
    payload = load_factor_batch(
        path,
        validate_expressions=validate_expressions,
    )
    factors = tuple(
        RegisteredFactor(
            number=int(record["number"]),
            factor_name=str(record["factor_name"]),
            entered_at=str(record["entered_at"]),
            implementation_set=str(record["implementation_set"]),
            expression=str(record["expression"]),
            paper_expression=str(record.get("paper_expression", "")),
            required_symbols=tuple(record["required_symbols"]),
            uses_proxy=bool(record.get("uses_proxy", False)),
            proxy_description=str(record.get("proxy_description", "")),
        )
        for record in payload["factors"]
    )
    numbers = [factor.number for factor in factors]
    if len(numbers) != len(set(numbers)):
        raise ValueError("factor registry has duplicate factor numbers")
    return factors


def _selected_numbers(selector: str) -> set[int] | None:
    normalized = selector.strip().lower()
    if normalized in {"", "all", "runnable"}:
        return None
    selected: set[int] = set()
    for item in normalized.split(","):
        token = item.strip()
        if not token:
            continue
        token = token.removeprefix("alpha101_").removeprefix("alpha")
        token = token.split("_", 1)[0].lstrip("0") or "0"
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(
                    f"descending factor range is not allowed: {item!r}"
                )
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return selected


def select_registered_factors(
    selector: str = "all",
    *,
    implementation_set: str | None = None,
    path: str | Path = DEFAULT_FACTOR_BATCH,
) -> tuple[RegisteredFactor, ...]:
    """Select registry factors by number/range and optional implementation set."""
    factors = load_registered_factors(path)
    if implementation_set and implementation_set != "runnable":
        factors = tuple(
            factor
            for factor in factors
            if factor.implementation_set == implementation_set
        )
    selected_numbers = _selected_numbers(selector)
    if selected_numbers is None:
        return factors
    available = {factor.number for factor in factors}
    unavailable = sorted(selected_numbers - available)
    if unavailable:
        scope = implementation_set or "registry"
        raise ValueError(
            f"selected factors are not available in {scope}: {unavailable}"
        )
    return tuple(
        factor for factor in factors if factor.number in selected_numbers
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=PROJECT_ROOT / "factor_registry",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = args.paths or sorted(
        args.registry_dir.expanduser().resolve().glob("*.json")
    )
    if not paths:
        raise FileNotFoundError("no factor batch JSON files found")
    summaries = []
    for path in paths:
        payload = load_factor_batch(path)
        summaries.append(
            {
                "path": str(Path(path).expanduser().resolve()),
                "batch_id": payload["batch_id"],
                "factor_count": payload["factor_count"],
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
