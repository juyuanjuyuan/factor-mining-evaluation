#!/usr/bin/env python3
"""Local CLI parity for factor creation and Webapp evaluation controls.

The command opens the same FastAPI routes in-process.  Validation, registry
writes, templates, job schemas, worker execution, gates, and funnel behavior
therefore stay identical to the Webapp instead of being reimplemented here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote

from paths import DATA_DIR, PROJECT_ROOT
from research_journal import ResearchJournal


DEFAULT_STATE_DIR = PROJECT_ROOT / "outputs" / "webapp"
DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "factor_registry"
TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
GATE_OPERATORS = {"gte", "lte", "gt", "lt", "between"}
RESEARCH_PHASES = ("exploration", "training", "testing")


class CliError(RuntimeError):
    """Expected command or API failure suitable for JSON output."""


class LocalApi:
    def __init__(self, client: Any):
        self.client = client

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        response = self.client.request(method, path, json=payload)
        if response.status_code >= 400:
            try:
                body = response.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except Exception:
                detail = response.text
            if not isinstance(detail, str):
                detail = json.dumps(detail, ensure_ascii=False)
            raise CliError(f"API {response.status_code}: {detail}")
        if response.status_code == 204:
            return {"ok": True}
        return response.json()


def _runtime_path(value: Path) -> str:
    return str(value.expanduser().resolve())


@contextmanager
def _local_api(args: argparse.Namespace, *, worker_enabled: bool) -> Iterator[LocalApi]:
    """Start the local API lifespan after configuring its runtime paths."""

    os.environ["FACTOR_WEBAPP_DATA_DIR"] = _runtime_path(args.data_dir)
    os.environ["FACTOR_WEBAPP_STATE_DIR"] = _runtime_path(args.state_dir)
    os.environ["FACTOR_WEBAPP_REGISTRY_DIR"] = _runtime_path(args.registry_dir)
    os.environ["FACTOR_WEBAPP_DISABLE_WORKER"] = "0" if worker_enabled else "1"
    # Factor research commands must not start or mutate a genetic campaign.
    os.environ["FACTOR_WEBAPP_DISABLE_GENETIC_MINING"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

    server_dir = PROJECT_ROOT / "webapp" / "server"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))
    try:
        from fastapi.testclient import TestClient
        from app.main import app
    except ImportError as exc:
        raise CliError(
            "factor-research requires the Webapp runtime; use the project's "
            "rdagent interpreter"
        ) from exc

    with TestClient(app) as client:
        yield LocalApi(client)


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))


def _encoded(value: str) -> str:
    return quote(value, safe="")


def _factor_ref(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError("factor reference must be BATCH_ID/FACTOR_NAME")
    batch_id, factor_name = value.split("/", 1)
    if not batch_id.strip() or not factor_name.strip():
        raise argparse.ArgumentTypeError("factor reference must be BATCH_ID/FACTOR_NAME")
    return batch_id.strip(), factor_name.strip()


def _method_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    methods = [item.strip() for item in value.split(",")]
    if not methods or any(not item for item in methods):
        raise CliError("--methods must be a non-empty comma-separated ordered list")
    return methods


def _tag_list(values: Sequence[str] | None) -> list[str]:
    return [value.strip() for value in values or [] if value.strip()]


def _gate_condition(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "gate must be METRIC:OP:VALUE or METRIC:between:LOW:HIGH"
        )
    metric, operator = parts[:2]
    if not metric or operator not in GATE_OPERATORS:
        raise argparse.ArgumentTypeError(
            f"gate operator must be one of {', '.join(sorted(GATE_OPERATORS))}"
        )
    if operator == "between" and len(parts) != 4:
        raise argparse.ArgumentTypeError("between gate requires lower and upper values")
    if operator != "between" and len(parts) != 3:
        raise argparse.ArgumentTypeError("only between accepts a second value")
    try:
        condition: dict[str, Any] = {
            "metric": metric,
            "op": operator,
            "value": float(parts[2]),
        }
        if len(parts) == 4:
            condition["value2"] = float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gate thresholds must be numbers") from exc
    return condition


def _stage_override(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("stage must be STAGE_NAME=METHOD1,METHOD2")
    name, raw_methods = value.split("=", 1)
    methods = _method_list(raw_methods)
    if not name.strip() or not methods:
        raise argparse.ArgumentTypeError("stage must include a name and methods")
    return name.strip(), methods


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _required_research_text(value: str | None, flag: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise CliError(f"{flag} is required when --research-id is provided")
    return normalized


def _research_context(args: argparse.Namespace) -> dict[str, Any] | None:
    """Validate enough lineage to make a later journal review meaningful."""

    research_id = getattr(args, "research_id", None)
    direction = getattr(args, "research_direction", None)
    hypothesis = getattr(args, "hypothesis", None)
    iteration_note = getattr(args, "iteration_note", None)
    parents = _tag_list(getattr(args, "parent_candidates", None))
    supplied = any((direction, hypothesis, iteration_note, parents))
    if research_id is None:
        if supplied:
            raise CliError(
                "--research-direction, --hypothesis, --iteration-note, and "
                "--parent-candidate require --research-id"
            )
        return None
    normalized_id = _required_research_text(research_id, "--research-id")
    return {
        "research_id": normalized_id,
        "research_phase": getattr(args, "research_phase", "exploration"),
        "research_direction": _required_research_text(
            direction, "--research-direction"
        ),
        "hypothesis": _required_research_text(hypothesis, "--hypothesis"),
        "iteration_note": (iteration_note or "").strip() or None,
        "parent_candidates": parents,
    }


def _note_research_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "research_id": _required_research_text(args.research_id, "--research-id"),
        "research_phase": args.research_phase,
        "research_direction": _required_research_text(
            args.research_direction, "--research-direction"
        ),
        "hypothesis": (args.hypothesis or "").strip() or None,
        "iteration_note": None,
        "parent_candidates": _tag_list(args.parent_candidates),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read JSON request file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliError("request JSON must contain one object")
    return payload


def _resolve_template(api: LocalApi, reference: str) -> dict[str, Any]:
    templates = api.request("GET", "/api/templates")
    matched = None
    if reference.isdigit():
        template_id = int(reference)
        matched = next((item for item in templates if item["id"] == template_id), None)
    if matched is None:
        matched = next((item for item in templates if item["name"] == reference), None)
    if matched is None:
        raise CliError(f"template not found: {reference}")
    return matched


def _wait_for_job(
    api: LocalApi,
    job_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        job = api.request("GET", f"/api/jobs/{_encoded(job_id)}")
        if job["status"] in TERMINAL_JOB_STATUSES:
            return job
        if time.monotonic() >= deadline:
            raise CliError(
                f"timed out waiting for job {job_id}; current status={job['status']}"
            )
        time.sleep(poll_interval)


def _build_run_payload(args: argparse.Namespace, api: LocalApi) -> dict[str, Any]:
    if args.request_file:
        return _load_json(args.request_file)

    if args.tags and (args.factors or args.expression or args.factor_name):
        raise CliError("tag selection cannot be combined with explicit factors")
    payload: dict[str, Any] = {"kind": args.kind}
    if args.title:
        payload["title"] = args.title

    if args.tags:
        payload.update(
            {
                "tags": _tag_list(args.tags),
                "tag_match": args.tag_match,
                "library": args.library,
            }
        )
    else:
        factors = [
            {"batch_id": batch_id, "factor_name": factor_name}
            for batch_id, factor_name in args.factors or []
        ]
        if args.expression or args.factor_name:
            if not args.expression or not args.factor_name:
                raise CliError("inline evaluation requires both --factor-name and --expression")
            factors.append(
                {
                    "batch_id": args.batch_id,
                    "factor_name": args.factor_name,
                    "expression": args.expression,
                }
            )
        if not factors:
            raise CliError(
                "run requires an inline expression, at least one --factor, or at least one --tag"
            )
        payload["factors"] = factors

    template = _resolve_template(api, args.template) if args.template else None
    if template:
        payload["template_id"] = template["id"]
        payload["kind"] = "funnel" if template["kind"] == "funnel" else "evaluate"
    params = dict(template.get("params") or {}) if template else {}
    payload["horizon"] = args.horizon if args.horizon is not None else int(params.get("horizon", 1))
    payload["n_quantiles"] = (
        args.quantiles if args.quantiles is not None else int(params.get("n_quantiles", 10))
    )
    payload["significance_level"] = (
        args.significance_level
        if args.significance_level is not None
        else float(params.get("significance_level", 0.05))
    )
    if (args.signal_start is None) != (args.signal_end is None):
        raise CliError("--signal-start and --signal-end must be provided together")
    if args.signal_start is not None:
        payload["signal_start"] = args.signal_start
        payload["signal_end"] = args.signal_end
    methods = _method_list(args.methods)
    if methods is not None:
        payload["methods"] = methods
    if args.gates:
        if payload["kind"] != "evaluate":
            raise CliError("custom metric gates apply only to evaluate jobs")
        payload["gate"] = {
            "conditions": list(args.gates),
            "match": args.gate_match,
        }
    return payload


def _template_stages(
    api: LocalApi,
    current: Mapping[str, Any] | None,
    overrides: Sequence[tuple[str, list[str]]] | None,
) -> list[dict[str, Any]]:
    current_params = dict(current.get("params") or {}) if current else {}
    raw_stages = current_params.get("stages")
    if isinstance(raw_stages, list):
        stages = [
            {"name": str(stage["name"]), "methods": list(stage["methods"])}
            for stage in raw_stages
        ]
    else:
        stages = [
            {"name": stage["name"], "methods": list(stage["methods"])}
            for stage in api.request("GET", "/api/funnel-stages")
        ]
    by_name = {stage["name"]: stage for stage in stages}
    for name, methods in overrides or []:
        if name not in by_name:
            raise CliError(f"unknown funnel stage: {name}")
        by_name[name]["methods"] = methods
    return stages


def _build_template_payload(
    args: argparse.Namespace,
    api: LocalApi,
    *,
    current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind = str(current["kind"]) if current else args.kind
    if current and args.kind and args.kind != kind:
        raise CliError("an existing template's kind cannot be changed")
    name = args.name or (str(current["name"]) if current else None)
    if not name:
        raise CliError("template name is required")
    if kind == "methods":
        methods = _method_list(args.methods)
        if methods is None and current:
            methods = list(current["methods"])
        if not methods:
            raise CliError("a methods template requires --methods")
        return {"name": name, "kind": kind, "methods": methods, "params": {}}

    current_params = dict(current.get("params") or {}) if current else {}
    return {
        "name": name,
        "kind": kind,
        "methods": [],
        "params": {
            "horizon": args.horizon if args.horizon is not None else int(current_params.get("horizon", 1)),
            "n_quantiles": args.quantiles if args.quantiles is not None else int(current_params.get("n_quantiles", 10)),
            "significance_level": (
                args.significance_level
                if args.significance_level is not None
                else float(current_params.get("significance_level", 0.05))
            ),
            "stages": _template_stages(api, current, args.stages),
        },
    }


def _factor_body(args: argparse.Namespace, current: Mapping[str, Any] | None = None) -> dict[str, Any]:
    def chosen(name: str, fallback: Any) -> Any:
        value = getattr(args, name, None)
        return fallback if value is None else value

    if current is None:
        if not args.factor_name or not args.expression:
            raise CliError("factor add requires --factor-name and --expression")
        tags = _tag_list(args.tags)
        return {
            "factor_name": args.factor_name,
            "expression": args.expression,
            "project": args.project or "自定义因子",
            "paper_expression": args.paper_expression or "",
            "uses_proxy": bool(args.uses_proxy),
            "proxy_description": args.proxy_description or "",
            "tags": tags,
        }

    tags = list(current.get("tags") or [])
    if getattr(args, "clear_tags", False):
        tags = []
    elif args.tags is not None:
        tags = _tag_list(args.tags)
    return {
        "factor_name": chosen("new_name", current["factor_name"]),
        "expression": chosen("expression", current["expression"]),
        "project": chosen("project", current.get("project") or "自定义因子"),
        "paper_expression": chosen("paper_expression", current.get("paper_expression") or ""),
        "uses_proxy": chosen("uses_proxy", bool(current.get("uses_proxy"))),
        "proxy_description": chosen(
            "proxy_description", current.get("proxy_description") or ""
        ),
        "tags": tags,
    }


def _dispatch(args: argparse.Namespace, api: LocalApi | None) -> tuple[Any, int]:
    if args.command == "journal":
        journal = ResearchJournal(args.state_dir)
        if args.journal_command == "note":
            return journal.add_note(_note_research_context(args), args.note.strip()), 0
        if args.journal_command == "history":
            return {
                "journal_path": str(journal.path),
                "research_id": args.research_id,
                "records": journal.records(
                    research_id=args.research_id,
                    limit=args.limit,
                ),
            }, 0
        if args.journal_command == "summary":
            return journal.summary(research_id=args.research_id, limit=args.limit), 0
        raise CliError("unsupported journal command")

    if api is None:
        raise CliError("this command requires the local Webapp API")

    if args.command == "catalog":
        endpoints = {
            "methods": "/api/methods",
            "operators": "/api/operators/causality",
            "funnel-stages": "/api/funnel-stages",
            "templates": "/api/templates",
            "market-data": "/api/market-data/timeline",
        }
        if args.catalog_name == "all":
            return (
                {
                    name: api.request("GET", endpoint)
                    for name, endpoint in endpoints.items()
                },
                0,
            )
        return api.request("GET", endpoints[args.catalog_name]), 0

    if args.command == "validate":
        return (
            api.request(
                "POST",
                "/api/expressions/validate",
                payload={"expression": args.expression},
            ),
            0,
        )

    if args.command == "factor":
        if args.factor_command == "list":
            endpoint = "/api/test-factors" if args.library == "test" else "/api/factors"
            return api.request("GET", endpoint), 0
        if args.factor_command == "add":
            return api.request("POST", "/api/test-factors", payload=_factor_body(args)), 0
        batch_id, factor_name = args.factor
        factor_path = f"/api/factors/{_encoded(batch_id)}/{_encoded(factor_name)}"
        if args.factor_command == "show":
            return api.request("GET", factor_path), 0
        if args.factor_command == "update":
            current = api.request("GET", factor_path)
            return api.request("PUT", factor_path, payload=_factor_body(args, current)), 0
        if args.factor_command == "tags":
            return (
                api.request(
                    "PUT",
                    f"{factor_path}/tags",
                    payload={"tags": _tag_list(args.tags)},
                ),
                0,
            )
        if args.factor_command == "project":
            return (
                api.request(
                    "PUT", f"{factor_path}/project", payload={"project": args.project}
                ),
                0,
            )
        if args.factor_command == "submit":
            endpoint = (
                f"/api/test-factors/{_encoded(batch_id)}/{_encoded(factor_name)}/submit"
            )
            return api.request("POST", endpoint), 0
        if args.factor_command == "delete":
            return api.request("DELETE", factor_path), 0

    if args.command == "template":
        if args.template_command == "list":
            return api.request("GET", "/api/templates"), 0
        if args.template_command == "create":
            payload = _build_template_payload(args, api)
            return api.request("POST", "/api/templates", payload=payload), 0
        template = _resolve_template(api, args.template)
        endpoint = f"/api/templates/{template['id']}"
        if args.template_command == "show":
            return template, 0
        if args.template_command == "clone":
            payload = {
                "name": args.name,
                "kind": template["kind"],
                "methods": template["methods"],
                "params": template["params"],
            }
            return api.request("POST", "/api/templates", payload=payload), 0
        if args.template_command == "update":
            payload = _build_template_payload(args, api, current=template)
            return api.request("PUT", endpoint, payload=payload), 0
        if args.template_command == "delete":
            return api.request("DELETE", endpoint), 0

    if args.command == "run":
        research = _research_context(args)
        journal = ResearchJournal(args.state_dir) if research else None
        try:
            job = api.request("POST", "/api/jobs", payload=_build_run_payload(args, api))
        except CliError as exc:
            if journal and research:
                journal.record_submission_failure(research, str(exc))
            raise
        if journal and research:
            journal.record_submission(research, job)
        if not args.submit_only:
            job = _wait_for_job(
                api,
                job["id"],
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            if journal:
                journal.record_completion_if_known(job)
        return job, 0 if job["status"] not in {"failed", "cancelled"} else 2

    if args.command == "job":
        if args.job_command == "list":
            return api.request("GET", f"/api/jobs?limit={args.limit}"), 0
        endpoint = f"/api/jobs/{_encoded(args.job_id)}"
        if args.job_command == "show":
            return api.request("GET", endpoint), 0
        if args.job_command == "wait":
            job = _wait_for_job(
                api,
                args.job_id,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
            ResearchJournal(args.state_dir).record_completion_if_known(job)
            return job, 0 if job["status"] == "succeeded" else 2
        if args.job_command == "cancel":
            job = api.request("POST", f"{endpoint}/cancel?force=false")
            ResearchJournal(args.state_dir).record_completion_if_known(job)
            return job, 0
        if args.job_command == "delete":
            return api.request("DELETE", endpoint), 0

    if args.command == "compare":
        return (
            api.request("POST", "/api/compare", payload={"run_ids": args.run_ids}),
            0,
        )

    raise CliError("unsupported command")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)


def _add_template_definition_arguments(
    parser: argparse.ArgumentParser,
    *,
    create: bool,
) -> None:
    parser.add_argument("--name", required=create)
    parser.add_argument("--kind", choices=("methods", "funnel"), default="methods" if create else None)
    parser.add_argument("--methods", help="ordered comma-separated methods; repeats are preserved")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--quantiles", type=int)
    parser.add_argument("--significance-level", type=float)
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        type=_stage_override,
        help="override one funnel stage: STAGE=METHOD1,METHOD2",
    )


def _add_research_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--research-id",
        help="stable research-thread identifier; enables append-only journaling",
    )
    parser.add_argument(
        "--research-phase",
        choices=RESEARCH_PHASES,
        default="exploration",
        help="research stage recorded with the candidate",
    )
    parser.add_argument(
        "--research-direction",
        help="economic direction or theme; required with --research-id",
    )
    parser.add_argument(
        "--hypothesis",
        help="candidate mechanism and expected relation; required with --research-id",
    )
    parser.add_argument(
        "--iteration-note",
        help="why this iteration differs from prior candidates",
    )
    parser.add_argument(
        "--parent-candidate",
        dest="parent_candidates",
        action="append",
        help="prior candidate this iteration extends or corrects; repeatable",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_runtime_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="show Webapp research capabilities")
    catalog.add_argument(
        "catalog_name",
        choices=("all", "methods", "operators", "funnel-stages", "templates", "market-data"),
        default="all",
        nargs="?",
    )

    validate = commands.add_parser("validate", help="validate an expression with the AST whitelist")
    validate.add_argument("--expression", required=True)

    factor = commands.add_parser("factor", help="manage test/formal factor records")
    factor_commands = factor.add_subparsers(dest="factor_command", required=True)
    factor_list = factor_commands.add_parser("list")
    factor_list.add_argument("--library", choices=("test", "factor"), default="test")
    factor_show = factor_commands.add_parser("show")
    factor_show.add_argument("--factor", type=_factor_ref, required=True)
    factor_add = factor_commands.add_parser("add")
    factor_add.add_argument("--factor-name", required=True)
    factor_add.add_argument("--expression", required=True)
    factor_add.add_argument("--project", default="自定义因子")
    factor_add.add_argument("--paper-expression", default="")
    factor_add.add_argument("--tag", dest="tags", action="append")
    factor_add.add_argument("--uses-proxy", action="store_true")
    factor_add.add_argument("--proxy-description", default="")
    factor_update = factor_commands.add_parser("update")
    factor_update.add_argument("--factor", type=_factor_ref, required=True)
    factor_update.add_argument("--new-name")
    factor_update.add_argument("--expression")
    factor_update.add_argument("--project")
    factor_update.add_argument("--paper-expression")
    factor_update.add_argument("--tag", dest="tags", action="append")
    factor_update.add_argument("--clear-tags", action="store_true")
    factor_update.add_argument("--uses-proxy", action=argparse.BooleanOptionalAction, default=None)
    factor_update.add_argument("--proxy-description")
    factor_tags = factor_commands.add_parser("tags")
    factor_tags.add_argument("--factor", type=_factor_ref, required=True)
    factor_tags.add_argument("--tag", dest="tags", action="append", default=[])
    factor_project = factor_commands.add_parser("project")
    factor_project.add_argument("--factor", type=_factor_ref, required=True)
    factor_project.add_argument("--project", required=True)
    for name in ("submit", "delete"):
        command = factor_commands.add_parser(name)
        command.add_argument("--factor", type=_factor_ref, required=True)

    template = commands.add_parser("template", help="manage evaluation/funnel templates")
    template_commands = template.add_subparsers(dest="template_command", required=True)
    template_commands.add_parser("list")
    template_show = template_commands.add_parser("show")
    template_show.add_argument("--template", required=True, help="template ID or exact name")
    template_create = template_commands.add_parser("create")
    _add_template_definition_arguments(template_create, create=True)
    template_update = template_commands.add_parser("update")
    template_update.add_argument("--template", required=True)
    _add_template_definition_arguments(template_update, create=False)
    template_clone = template_commands.add_parser("clone")
    template_clone.add_argument("--template", required=True)
    template_clone.add_argument("--name", required=True)
    template_delete = template_commands.add_parser("delete")
    template_delete.add_argument("--template", required=True)

    run = commands.add_parser("run", help="submit and optionally execute an evaluation job")
    run.add_argument("--request-file", type=Path, help="exact Webapp JobCreate JSON payload")
    run.add_argument("--kind", choices=("evaluate", "funnel"), default="evaluate")
    run.add_argument("--title")
    run.add_argument("--factor-name")
    run.add_argument("--expression")
    run.add_argument("--batch-id", default="temporary")
    run.add_argument("--factor", dest="factors", action="append", type=_factor_ref)
    run.add_argument("--tag", dest="tags", action="append")
    run.add_argument("--tag-match", choices=("any", "all"), default="any")
    run.add_argument("--library", choices=("test", "factor"), default="test")
    run.add_argument("--template", help="template ID or exact name")
    run.add_argument("--methods", help="ordered comma-separated methods; repeats are preserved")
    run.add_argument("--horizon", type=int)
    run.add_argument("--quantiles", type=int)
    run.add_argument("--significance-level", type=float)
    run.add_argument(
        "--signal-start",
        help="inclusive signal-date window start (YYYY-MM-DD); requires --signal-end",
    )
    run.add_argument(
        "--signal-end",
        help="inclusive signal-date window end (YYYY-MM-DD); requires --signal-start",
    )
    run.add_argument("--gate", dest="gates", action="append", type=_gate_condition)
    run.add_argument("--gate-match", choices=("all", "any"), default="all")
    _add_research_metadata_arguments(run)
    run.add_argument("--submit-only", action="store_true", help="queue without starting a local worker")
    run.add_argument("--timeout", type=_positive_float, default=1800.0)
    run.add_argument("--poll-interval", type=_positive_float, default=0.5)

    job = commands.add_parser("job", help="inspect and control evaluation jobs")
    job_commands = job.add_subparsers(dest="job_command", required=True)
    job_list = job_commands.add_parser("list")
    job_list.add_argument("--limit", type=int, default=100)
    for name in ("show", "cancel", "delete"):
        command = job_commands.add_parser(name)
        command.add_argument("--job-id", required=True)
    job_wait = job_commands.add_parser("wait")
    job_wait.add_argument("--job-id", required=True)
    job_wait.add_argument("--timeout", type=_positive_float, default=1800.0)
    job_wait.add_argument("--poll-interval", type=_positive_float, default=0.5)

    journal = commands.add_parser("journal", help="record and review agent research history")
    journal_commands = journal.add_subparsers(dest="journal_command", required=True)
    journal_note = journal_commands.add_parser("note", help="append a research decision or observation")
    journal_note.add_argument("--research-id", required=True)
    journal_note.add_argument("--research-phase", choices=RESEARCH_PHASES, default="exploration")
    journal_note.add_argument("--research-direction", required=True)
    journal_note.add_argument("--hypothesis")
    journal_note.add_argument("--parent-candidate", dest="parent_candidates", action="append")
    journal_note.add_argument("--note", required=True)
    journal_history = journal_commands.add_parser("history", help="read raw journal evidence")
    journal_history.add_argument("--research-id")
    journal_history.add_argument("--limit", type=_positive_int, default=50)
    journal_summary = journal_commands.add_parser("summary", help="build compact context for the next agent round")
    journal_summary.add_argument("--research-id")
    journal_summary.add_argument("--limit", type=_positive_int, default=200)

    compare = commands.add_parser("compare", help="compare completed run IDs")
    compare.add_argument("--run-id", dest="run_ids", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    worker_enabled = (
        args.command == "run" and not args.submit_only
    ) or (args.command == "job" and args.job_command == "wait")
    try:
        if args.command == "journal":
            payload, exit_code = _dispatch(args, None)
            _emit(payload)
            return exit_code
        with _local_api(args, worker_enabled=worker_enabled) as api:
            payload, exit_code = _dispatch(args, api)
        _emit(payload)
        return exit_code
    except (CliError, ValueError) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
