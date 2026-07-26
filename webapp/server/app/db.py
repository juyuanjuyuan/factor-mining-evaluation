"""Small SQLite data-access layer for jobs, immutable runs, and templates."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from evaluation_standards import PROFITABILITY_METHOD_NAMES

from .sanitize import sanitize


DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('evaluate', 'funnel')),
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    params_json TEXT NOT NULL,
    total_runs INTEGER NOT NULL DEFAULT 0,
    finished_runs INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    factor_name TEXT NOT NULL,
    batch_id TEXT,
    expression TEXT NOT NULL,
    stage TEXT,
    gate_outcome TEXT,
    gate_value TEXT,
    gate_explanation TEXT,
    status TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    n_quantiles INTEGER NOT NULL,
    methods_json TEXT NOT NULL,
    run_params_json TEXT NOT NULL DEFAULT '{}',
    output_dir TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_factor ON runs(batch_id, factor_name, created_at);
CREATE TABLE IF NOT EXISTS model_tests (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    terms_json TEXT NOT NULL,
    expression TEXT NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    test_start TEXT NOT NULL,
    test_end TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    n_quantiles INTEGER NOT NULL,
    methods_json TEXT NOT NULL,
    training_method TEXT NOT NULL DEFAULT 'winsorized_zscore_ridge',
    training_params_json TEXT NOT NULL DEFAULT '{}',
    fit_result_json TEXT,
    training_run_id TEXT REFERENCES runs(id),
    testing_run_id TEXT REFERENCES runs(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    locked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_tests_updated ON model_tests(updated_at DESC);
CREATE TABLE IF NOT EXISTS factor_tags (
    batch_id TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    tag TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (batch_id, factor_name, tag)
);
CREATE INDEX IF NOT EXISTS idx_factor_tags_tag ON factor_tags(tag, batch_id, factor_name);
CREATE TABLE IF NOT EXISTS pipeline_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('methods', 'funnel')),
    methods_json TEXT NOT NULL,
    params_json TEXT NOT NULL,
    is_builtin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


BUILTIN_TEMPLATES = (
    (
        "默认流水线",
        "methods",
        [
            "rank_ic",
            "rank_icir",
            "quantile_returns",
            "quantile_cumulative",
            "quantile_plot",
        ],
        {},
    ),
    (
        "盈利能力测试",
        "methods",
        list(PROFITABILITY_METHOD_NAMES),
        {},
    ),
    (
        "完整漏斗",
        "funnel",
        [],
        {"horizon": 1, "n_quantiles": 10, "significance_level": 0.05},
    ),
)
REMOVED_BUILTIN_TEMPLATE_NAMES = ("快速IC筛查", "未来数据校验", "完整组合评估")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._write_lock = threading.RLock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(DDL)
            self._migrate_schema(connection)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            orphaned_jobs = [
                row["job_id"]
                for row in connection.execute(
                    "SELECT DISTINCT job_id FROM runs WHERE status = 'running'"
                ).fetchall()
            ]
            if orphaned_jobs:
                connection.execute(
                    """
                    UPDATE runs SET status = 'failed',
                        error = 'API 进程重启，原 worker 状态已丢失',
                        finished_at = ?
                    WHERE status = 'running'
                    """,
                    (now_iso(),),
                )
                for job_id in orphaned_jobs:
                    self._refresh_job(connection, job_id)
        self._seed_templates()

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        """Apply additive migrations for state databases created before model tests."""

        run_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "run_params_json" not in run_columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN run_params_json TEXT NOT NULL DEFAULT '{}'"
            )
        model_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(model_tests)").fetchall()
        }
        if "training_method" not in model_columns:
            connection.execute(
                "ALTER TABLE model_tests ADD COLUMN training_method TEXT NOT NULL DEFAULT 'manual_weights'"
            )
        if "training_params_json" not in model_columns:
            connection.execute(
                "ALTER TABLE model_tests ADD COLUMN training_params_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "fit_result_json" not in model_columns:
            connection.execute("ALTER TABLE model_tests ADD COLUMN fit_result_json TEXT")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _seed_templates(self) -> None:
        timestamp = now_iso()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                f"""
                DELETE FROM pipeline_templates
                WHERE is_builtin = 1
                  AND name IN ({",".join("?" for _ in REMOVED_BUILTIN_TEMPLATE_NAMES)})
                """,
                REMOVED_BUILTIN_TEMPLATE_NAMES,
            )
            for name, kind, methods, params in BUILTIN_TEMPLATES:
                connection.execute(
                    """
                    INSERT INTO pipeline_templates
                    (name, kind, methods_json, params_json, is_builtin, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        kind = excluded.kind,
                        methods_json = excluded.methods_json,
                        params_json = excluded.params_json,
                        is_builtin = 1,
                        updated_at = excluded.updated_at
                    """,
                    (
                        name,
                        kind,
                        json.dumps(methods, ensure_ascii=False),
                        json.dumps(params, ensure_ascii=False),
                        timestamp,
                        timestamp,
                    ),
                )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("params_json", "methods_json", "run_params_json", "result_json"):
            if key in result:
                raw = result.pop(key)
                result[key.removesuffix("_json")] = json.loads(raw) if raw else None
        # gate_value is stored as JSON (a scalar for the funnel, a per-condition
        # list for custom gates); hand the frontend the parsed structure.
        raw_gate_value = result.get("gate_value")
        if isinstance(raw_gate_value, str) and raw_gate_value:
            try:
                result["gate_value"] = json.loads(raw_gate_value)
            except (TypeError, ValueError):
                pass
        if "is_builtin" in result:
            result["is_builtin"] = bool(result["is_builtin"])
        return result

    def create_job(
        self,
        job: Mapping[str, Any],
        runs: Iterable[Mapping[str, Any]],
    ) -> None:
        run_rows = list(runs)
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs
                (id, kind, title, status, params_json, total_runs, finished_runs, created_at)
                VALUES (?, ?, ?, 'queued', ?, ?, 0, ?)
                """,
                (
                    job["id"],
                    job["kind"],
                    job["title"],
                    json.dumps(sanitize(job.get("params", {})), ensure_ascii=False),
                    len(run_rows),
                    job["created_at"],
                ),
            )
            self._insert_runs(connection, run_rows)

    @staticmethod
    def _decode_model_test(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in (
            "terms_json",
            "methods_json",
            "training_params_json",
            "fit_result_json",
        ):
            raw = result.pop(key)
            if key in {"terms_json", "methods_json"}:
                fallback: Any = []
            elif key == "training_params_json":
                fallback = {}
            else:
                fallback = None
            result[key.removesuffix("_json")] = json.loads(raw) if raw else fallback
        result["locked"] = bool(result.get("locked_at"))
        return result

    def create_model_test(self, model: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO model_tests
                (id, model_name, terms_json, expression, train_start, train_end,
                 test_start, test_end, horizon, n_quantiles, methods_json,
                 training_method, training_params_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model["id"],
                    model["model_name"],
                    json.dumps(sanitize(model["terms"]), ensure_ascii=False),
                    model["expression"],
                    model["train_start"],
                    model["train_end"],
                    model["test_start"],
                    model["test_end"],
                    model["horizon"],
                    model["n_quantiles"],
                    json.dumps(model["methods"], ensure_ascii=False),
                    model["training_method"],
                    json.dumps(sanitize(model["training_params"]), ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_model_test(str(model["id"]))  # type: ignore[return-value]

    def list_model_tests(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM model_tests ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._decode_model_test(row) for row in rows]

    def delete_model_test(self, model_id: str) -> bool:
        """Delete only the model-session record; linked evaluation jobs remain auditable."""

        with self._write_lock, self.connection() as connection:
            return bool(
                connection.execute(
                    "DELETE FROM model_tests WHERE id = ?", (model_id,)
                ).rowcount
            )

    def get_model_test(self, model_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            return self._decode_model_test(
                connection.execute(
                    "SELECT * FROM model_tests WHERE id = ?", (model_id,)
                ).fetchone()
            )

    def update_model_test(self, model_id: str, model: Mapping[str, Any]) -> dict[str, Any] | None:
        """Replace a mutable draft and detach any stale training result."""

        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT testing_run_id, training_run_id FROM model_tests WHERE id = ?", (model_id,)
            ).fetchone()
            if row is None:
                return None
            if row["testing_run_id"]:
                raise PermissionError("样本外测试已经提交，模型配置已锁定")
            if row["training_run_id"]:
                training = connection.execute(
                    "SELECT status FROM runs WHERE id = ?", (row["training_run_id"],)
                ).fetchone()
                if training and training["status"] in {"queued", "running"}:
                    raise PermissionError("训练任务正在排队或执行，请等待完成后再修改模型")
            connection.execute(
                """
                UPDATE model_tests SET model_name = ?, terms_json = ?, expression = ?,
                    train_start = ?, train_end = ?, test_start = ?, test_end = ?,
                    horizon = ?, n_quantiles = ?, methods_json = ?,
                    training_method = ?, training_params_json = ?, fit_result_json = NULL,
                    training_run_id = NULL, updated_at = ?
                WHERE id = ?
                """,
                (
                    model["model_name"],
                    json.dumps(sanitize(model["terms"]), ensure_ascii=False),
                    model["expression"],
                    model["train_start"],
                    model["train_end"],
                    model["test_start"],
                    model["test_end"],
                    model["horizon"],
                    model["n_quantiles"],
                    json.dumps(model["methods"], ensure_ascii=False),
                    model["training_method"],
                    json.dumps(sanitize(model["training_params"]), ensure_ascii=False),
                    now_iso(),
                    model_id,
                ),
            )
        return self.get_model_test(model_id)

    def apply_model_training(
        self,
        *,
        model_id: str,
        run_id: str,
        terms: list[Mapping[str, Any]],
        expression: str,
        fit_result: Mapping[str, Any],
    ) -> bool:
        """Freeze one worker-produced fit into its still-current training session."""

        with self._write_lock, self.connection() as connection:
            connection.execute(
                "UPDATE runs SET expression = ? WHERE id = ? AND status = 'running'",
                (expression, run_id),
            )
            changed = connection.execute(
                """
                UPDATE model_tests SET terms_json = ?, expression = ?, fit_result_json = ?,
                    updated_at = ?
                WHERE id = ? AND training_run_id = ? AND testing_run_id IS NULL
                """,
                (
                    json.dumps(sanitize(terms), ensure_ascii=False),
                    expression,
                    json.dumps(sanitize(fit_result), ensure_ascii=False, allow_nan=False),
                    now_iso(),
                    model_id,
                    run_id,
                ),
            ).rowcount
            return bool(changed)

    def link_model_run(self, model_id: str, run_id: str, *, section: str) -> bool:
        if section not in {"training", "testing"}:
            raise ValueError(f"Unknown model-test section: {section}")
        column = "training_run_id" if section == "training" else "testing_run_id"
        with self._write_lock, self.connection() as connection:
            if section == "testing":
                changed = connection.execute(
                    """
                    UPDATE model_tests SET testing_run_id = ?, locked_at = ?, updated_at = ?
                    WHERE id = ? AND testing_run_id IS NULL
                    """,
                    (run_id, now_iso(), now_iso(), model_id),
                ).rowcount
            else:
                changed = connection.execute(
                    f"UPDATE model_tests SET {column} = ?, updated_at = ? WHERE id = ? AND testing_run_id IS NULL",
                    (run_id, now_iso(), model_id),
                ).rowcount
            return bool(changed)

    def _insert_runs(
        self, connection: sqlite3.Connection, runs: Iterable[Mapping[str, Any]]
    ) -> None:
        for run in runs:
            connection.execute(
                """
                INSERT INTO runs
                (id, job_id, factor_name, batch_id, expression, stage, status,
                 horizon, n_quantiles, methods_json, run_params_json, output_dir, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run["id"],
                    run["job_id"],
                    run["factor_name"],
                    run.get("batch_id"),
                    run["expression"],
                    run.get("stage"),
                    run.get("status", "queued"),
                    run["horizon"],
                    run["n_quantiles"],
                    json.dumps(run["methods"], ensure_ascii=False),
                    json.dumps(sanitize(run.get("run_params", {})), ensure_ascii=False),
                    run["output_dir"],
                    run["created_at"],
                ),
            )

    def add_runs(self, job_id: str, runs: Iterable[Mapping[str, Any]]) -> None:
        rows = list(runs)
        if not rows:
            return
        with self._write_lock, self.connection() as connection:
            self._insert_runs(connection, rows)
            connection.execute(
                """
                UPDATE jobs SET total_runs = total_runs + ?, status = 'running',
                    finished_at = NULL WHERE id = ?
                """,
                (len(rows), job_id),
            )
            self._refresh_job(connection, job_id)

    def get_job(self, job_id: str, *, include_runs: bool = True) -> dict[str, Any] | None:
        with self.connection() as connection:
            job = self._decode(
                connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            )
            if job and include_runs:
                rows = connection.execute(
                    "SELECT * FROM runs WHERE job_id = ? ORDER BY created_at, rowid",
                    (job_id,),
                ).fetchall()
                job["runs"] = [self._decode(row) for row in rows]
            return job

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._decode(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """Delete a terminal job and its run records, but keep disk artifacts untouched."""

        with self._write_lock, self.connection() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if job is None:
                return False
            if job["status"] in {"queued", "running", "cancelling"}:
                raise PermissionError("任务仍在排队或执行中，请先取消或停止任务")
            model_reference = connection.execute(
                """
                SELECT 1
                FROM model_tests AS model
                JOIN runs AS run
                  ON run.id = model.training_run_id OR run.id = model.testing_run_id
                WHERE run.job_id = ?
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if model_reference is not None:
                raise PermissionError("任务仍被模型会话引用，请先删除对应的模型会话")
            connection.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return True

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            return self._decode(
                connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            )

    def list_runs(
        self,
        *,
        status: str | None = None,
        factor_name: str | None = None,
        job_id: str | None = None,
        limit: int = 300,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("status", status),
            ("factor_name", factor_name),
            ("job_id", job_id),
        ):
            if value:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?",
                values,
            ).fetchall()
            return [self._decode(row) for row in rows]

    def latest_run(self, batch_id: str, factor_name: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM runs
                WHERE batch_id = ? AND factor_name = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (batch_id, factor_name),
            ).fetchone()
            return self._decode(row)

    def tags_for_factors(
        self,
        factors: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], list[str]]:
        """Return tags for several registry identities in one read."""
        identities = list(dict.fromkeys(factors))
        result = {identity: [] for identity in identities}
        if not identities:
            return result
        with self.connection() as connection:
            # Two bind parameters per identity. Chunk below SQLite's common
            # 999-parameter limit so large generated registries remain usable.
            for start in range(0, len(identities), 400):
                chunk = identities[start : start + 400]
                clauses = " OR ".join(
                    "(batch_id = ? AND factor_name = ?)" for _ in chunk
                )
                values = [value for identity in chunk for value in identity]
                rows = connection.execute(
                    f"""
                    SELECT batch_id, factor_name, tag FROM factor_tags
                    WHERE {clauses}
                    ORDER BY tag COLLATE NOCASE
                    """,
                    values,
                ).fetchall()
                for row in rows:
                    result[(row["batch_id"], row["factor_name"])].append(row["tag"])
        return result

    def replace_factor_tags(
        self,
        batch_id: str,
        factor_name: str,
        tags: Iterable[str],
    ) -> list[str]:
        """Replace one factor's user-managed tag set atomically."""
        values = list(tags)
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "DELETE FROM factor_tags WHERE batch_id = ? AND factor_name = ?",
                (batch_id, factor_name),
            )
            connection.executemany(
                """
                INSERT INTO factor_tags (batch_id, factor_name, tag, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [(batch_id, factor_name, tag, now_iso()) for tag in values],
            )
        return values

    def delete_factor_tags(self, batch_id: str, factor_name: str) -> None:
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "DELETE FROM factor_tags WHERE batch_id = ? AND factor_name = ?",
                (batch_id, factor_name),
            )

    def claim_next_run(self) -> dict[str, Any] | None:
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                """
                SELECT r.* FROM runs r JOIN jobs j ON j.id = r.job_id
                WHERE r.status = 'queued' AND j.status IN ('queued', 'running')
                ORDER BY r.created_at, r.rowid LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            timestamp = now_iso()
            changed = connection.execute(
                """
                UPDATE runs SET status = 'running', started_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (timestamp, row["id"]),
            ).rowcount
            if not changed:
                return None
            connection.execute(
                """
                UPDATE jobs SET status = 'running', started_at = COALESCE(started_at, ?)
                WHERE id = ?
                """,
                (timestamp, row["job_id"]),
            )
            return self._decode(
                connection.execute("SELECT * FROM runs WHERE id = ?", (row["id"],)).fetchone()
            )

    def complete_run(
        self,
        run_id: str,
        result: Mapping[str, Any],
        *,
        gate_outcome: str | None = None,
        gate_value: Any = None,
        gate_explanation: str | None = None,
    ) -> None:
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT job_id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return
            connection.execute(
                """
                UPDATE runs SET status = 'succeeded', result_json = ?, error = NULL,
                    gate_outcome = ?, gate_value = ?, gate_explanation = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(sanitize(result), ensure_ascii=False, allow_nan=False),
                    gate_outcome,
                    json.dumps(sanitize(gate_value), ensure_ascii=False),
                    gate_explanation,
                    now_iso(),
                    run_id,
                ),
            )
            self._refresh_job(connection, row["job_id"])

    def fail_run(self, run_id: str, error: str) -> None:
        self._finish_run(run_id, "failed", error=error)

    def cancel_run(self, run_id: str, reason: str = "强制取消") -> None:
        self._finish_run(run_id, "cancelled", error=reason)

    def _finish_run(self, run_id: str, status: str, *, error: str | None = None) -> None:
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT job_id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return
            connection.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, now_iso(), run_id),
            )
            self._refresh_job(connection, row["job_id"])

    def mark_skipped_runs(self, job_id: str, factor_name: str, reason: str) -> None:
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                UPDATE runs SET status = 'skipped', error = ?, finished_at = ?
                WHERE job_id = ? AND factor_name = ? AND status = 'queued'
                """,
                (reason, now_iso(), job_id, factor_name),
            )
            self._refresh_job(connection, job_id)

    def cancel_job(self, job_id: str) -> bool:
        with self._write_lock, self.connection() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not job:
                return False
            timestamp = now_iso()
            connection.execute(
                """
                UPDATE runs SET status = 'cancelled', finished_at = ?
                WHERE job_id = ? AND status = 'queued'
                """,
                (timestamp, job_id),
            )
            connection.execute(
                "UPDATE jobs SET status = 'cancelling' WHERE id = ? AND status IN ('queued','running')",
                (job_id,),
            )
            self._refresh_job(connection, job_id)
            return True

    def _refresh_job(self, connection: sqlite3.Connection, job_id: str) -> None:
        counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM runs WHERE job_id = ? GROUP BY status",
                (job_id,),
            ).fetchall()
        }
        total = sum(counts.values())
        finished = sum(
            counts.get(status, 0)
            for status in ("succeeded", "failed", "cancelled", "skipped")
        )
        active = counts.get("queued", 0) + counts.get("running", 0)
        if active:
            status = "running" if counts.get("running", 0) else "queued"
            finished_at = None
        elif counts.get("failed", 0):
            status, finished_at = "failed", now_iso()
        elif counts.get("cancelled", 0):
            status, finished_at = "cancelled", now_iso()
        else:
            status, finished_at = "succeeded", now_iso()
        connection.execute(
            """
            UPDATE jobs SET status = ?, total_runs = ?, finished_runs = ?, finished_at = ?
            WHERE id = ?
            """,
            (status, total, finished, finished_at, job_id),
        )

    def list_templates(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [
                self._decode(row)
                for row in connection.execute(
                    "SELECT * FROM pipeline_templates ORDER BY is_builtin DESC, id"
                ).fetchall()
            ]

    def get_template(self, template_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            return self._decode(
                connection.execute(
                    "SELECT * FROM pipeline_templates WHERE id = ?", (template_id,)
                ).fetchone()
            )

    def create_template(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        with self._write_lock, self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pipeline_templates
                (name, kind, methods_json, params_json, is_builtin, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    payload["name"],
                    payload["kind"],
                    json.dumps(payload.get("methods", []), ensure_ascii=False),
                    json.dumps(payload.get("params", {}), ensure_ascii=False),
                    timestamp,
                    timestamp,
                ),
            )
            template_id = cursor.lastrowid
        return self.get_template(template_id)  # type: ignore[return-value]

    def update_template(self, template_id: int, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT is_builtin FROM pipeline_templates WHERE id = ?", (template_id,)
            ).fetchone()
            if row is None:
                return None
            if row["is_builtin"]:
                raise PermissionError("Built-in templates are read-only")
            connection.execute(
                """
                UPDATE pipeline_templates SET name = ?, kind = ?, methods_json = ?,
                    params_json = ?, updated_at = ? WHERE id = ?
                """,
                (
                    payload["name"],
                    payload["kind"],
                    json.dumps(payload.get("methods", []), ensure_ascii=False),
                    json.dumps(payload.get("params", {}), ensure_ascii=False),
                    now_iso(),
                    template_id,
                ),
            )
        return self.get_template(template_id)

    def delete_template(self, template_id: int) -> bool:
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT is_builtin FROM pipeline_templates WHERE id = ?", (template_id,)
            ).fetchone()
            if row is None:
                return False
            if row["is_builtin"]:
                raise PermissionError("Built-in templates are read-only")
            connection.execute("DELETE FROM pipeline_templates WHERE id = ?", (template_id,))
            return True
