#!/usr/bin/env python3
"""FastAPI catalog, job submission, and custom-registry contract checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SERVER_DIR.parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def write_minimal_market_data(data_dir: Path) -> None:
    """Keep catalog tests independent from the user's local market-data files."""
    rng = np.random.default_rng(8)
    index = pd.date_range("2024-01-02", periods=125, freq="B")
    columns = [f"{code:06d}" for code in range(1, 9)]
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(size=(len(index), len(columns))), axis=0),
        index=index,
        columns=columns,
    )
    data_dir.mkdir()
    close.to_parquet(data_dir / "close_df.pq")
    (close * 1.001).to_parquet(data_dir / "open_df.pq")
    volume = pd.DataFrame(
        rng.lognormal(mean=12, sigma=0.7, size=close.shape),
        index=index,
        columns=columns,
    )
    volume.to_parquet(data_dir / "volume_df.pq")


def write_alpha_registry(registry_dir: Path) -> None:
    """Create the read-only catalog fixture without relying on a live batch file."""

    factors = [
        {
            "number": number,
            "factor_name": f"alpha{number:03d}",
            "entered_at": "2026-01-01T00:00:00+08:00",
            "implementation_set": "exact",
            "expression": "rank_cs(c)",
            "paper_expression": "",
            "required_symbols": ["c"],
            "uses_proxy": False,
            "proxy_description": "",
        }
        for number in range(1, 84)
    ]
    payload = {
        "schema_version": 1,
        "batch_id": "alpha101_runnable_factors",
        "batch_name": "Alpha101 API test fixture",
        "batch_version": 1,
        "generated_at": "2026-01-01T00:00:00+08:00",
        "source": {"type": "test", "name": "Synthetic Alpha101 catalog"},
        "formula_language": "engine expression",
        "factor_count": len(factors),
        "factors": factors,
    }
    (registry_dir / "alpha101_runnable_factors.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    with TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        data_dir = temporary_root / "data"
        write_minimal_market_data(data_dir)
        os.environ["FACTOR_WEBAPP_DISABLE_WORKER"] = "1"
        os.environ["FACTOR_WEBAPP_STATE_DIR"] = str(temporary_root / "state")

        from fastapi.testclient import TestClient

        from app.config import Settings
        from app.db import Database
        from app.main import app
        from app.services.funnel_service import default_funnel_stages, first_funnel_run
        from app.services.registry_service import RegistryService
        from app.worker.supervisor import WorkerSupervisor

        api_registry_dir = temporary_root / "api-registry"
        api_registry_dir.mkdir()
        write_alpha_registry(api_registry_dir)
        with TestClient(app) as client:
            # The app lifespan creates a registry from the checkout by default.
            # Replace it with a temporary one so this contract test never reads
            # or writes a user's test/formal factor library.
            app.state.registry = RegistryService(
                Settings(
                    project_root=PROJECT_ROOT,
                    data_dir=data_dir,
                    registry_dir=api_registry_dir,
                    state_dir=temporary_root / "state",
                    frontend_dist=PROJECT_ROOT / "webapp" / "frontend" / "dist",
                ),
                app.state.db,
            )
            assert client.get("/api/health").json() == {"status": "ok"}
            timeline = client.get("/api/market-data/timeline")
            assert timeline.status_code == 200
            assert timeline.json()["start_day"] == "2024-01-02"
            assert timeline.json()["end_day"] == "2024-06-24"
            assert timeline.json()["count"] == 125
            assert len(timeline.json()["trading_days"]) == 125
            methods = client.get("/api/methods")
            assert methods.status_code == 200 and len(methods.json()) >= 12
            assert [item["name"] for item in methods.json() if item["is_default"]] == [
                "rank_ic",
                "rank_icir",
                "quantile_returns",
                "quantile_cumulative",
                "quantile_plot",
            ]
            horizon_decay_method = next(
                item for item in methods.json() if item["name"] == "ic_horizon_decay"
            )
            assert horizon_decay_method["requires"] == []
            assert horizon_decay_method["required_data_symbols"] == ["o"]
            industry_method = next(
                item for item in methods.json() if item["name"] == "industry_neutralize"
            )
            assert industry_method["required_data_symbols"] == ["industry"]
            assert not industry_method["is_default"]
            joint_neutralization_method = next(
                item
                for item in methods.json()
                if item["name"] == "industry_market_cap_neutralize"
            )
            assert joint_neutralization_method["required_data_symbols"] == [
                "cap",
                "industry",
            ]
            assert not joint_neutralization_method["is_default"]
            training_methods = client.get("/api/models/training-methods")
            assert training_methods.status_code == 200
            training_catalog = {item["name"]: item for item in training_methods.json()}
            assert {"manual_weights", "rank_ridge"} <= set(training_catalog)
            assert training_catalog["manual_weights"]["term_weight_editable"]
            assert training_catalog["rank_ridge"]["requires_fitting"]
            assert training_catalog["rank_ridge"]["parameters"][0]["name"] == "ridge_alpha"
            operators = client.get("/api/operators/causality")
            assert operators.status_code == 200
            operator_rows = operators.json()
            assert any(
                row["name"] == "ts_mean" and row["passed"]
                for row in operator_rows
            )
            assert any(
                row["name"] == "np.log" and row["passed"]
                for row in operator_rows
            )
            stages = client.get("/api/funnel-stages")
            assert stages.status_code == 200
            assert [stage["name"] for stage in stages.json()] == [
                "stage1_validity",
                "stage2_ic",
                "stage2b_neutral",
                "stage3_portfolio",
            ]
            assert stages.json()[-1]["methods"][0] == "tradability_filter"
            factors = client.get("/api/test-factors")
            assert factors.status_code == 200
            assert sum(
                factor["batch_id"] == "alpha101_runnable_factors"
                for factor in factors.json()
            ) == 83
            assert {
                factor["project"]
                for factor in factors.json()
                if factor["batch_id"] == "alpha101_runnable_factors"
            } == {"Alpha101"}
            alpha_factor = next(
                factor
                for factor in factors.json()
                if factor["batch_id"] == "alpha101_runnable_factors"
            )
            reclassified_alpha = client.put(
                f"/api/factors/{alpha_factor['batch_id']}/{alpha_factor['factor_name']}/project",
                json={"project": "华夏191"},
            )
            assert reclassified_alpha.status_code == 200
            assert reclassified_alpha.json()["project"] == "华夏191"
            assert (
                client.get(
                    f"/api/factors/{alpha_factor['batch_id']}/{alpha_factor['factor_name']}"
                ).json()["project"]
                == "华夏191"
            )
            library_factors = client.get("/api/factors")
            assert library_factors.status_code == 200
            assert library_factors.json() == []
            correlation = client.get("/api/factor-correlation")
            assert correlation.status_code == 200
            assert correlation.json()["factor_names"] == []
            assert correlation.json()["matrix"] == []
            candidate = client.post(
                "/api/test-factors",
                json={
                    "factor_name": "api_correlation_factor",
                    "expression": "rank_cs(c)",
                    "project": "华夏191",
                },
            )
            assert candidate.status_code == 201
            assert candidate.json()["project"] == "华夏191"
            admitted = client.post(
                "/api/test-factors/webapp_test_factors/api_correlation_factor/submit"
            )
            assert admitted.status_code == 201
            assert admitted.json()["project"] == "华夏191"
            model_payload = {
                "model_name": "API 线性模型",
                "terms": [
                    {
                        "batch_id": admitted.json()["batch_id"],
                        "factor_name": admitted.json()["factor_name"],
                        "weight": 1.0,
                    }
                ],
                "train_start": "2024-01-02",
                "train_end": "2024-01-31",
                "test_start": "2024-02-01",
                "test_end": "2024-02-23",
                "horizon": 1,
                "n_quantiles": 5,
                "methods": ["rank_icir"],
                "training_method": "manual_weights",
            }
            model = client.post("/api/models", json=model_payload)
            assert model.status_code == 201
            assert model.json()["methods"] == ["rank_ic", "rank_icir"]
            assert model.json()["training_method"] == "manual_weights"
            assert model.json()["training_params"] == {}
            assert "rank_cs" in model.json()["expression"]
            model_id = model.json()["id"]
            assert client.get("/api/models").json()[0]["id"] == model_id
            training = client.post(f"/api/models/{model_id}/train")
            assert training.status_code == 201
            training_run = training.json()["training_run"]
            assert training_run["run_params"]["signal_start"] == "2024-01-02"
            assert training_run["run_params"]["signal_end"] == "2024-01-31"
            assert client.post(f"/api/models/{model_id}/test").status_code == 409
            app.state.db.complete_run(training_run["id"], {"ic_mean": 0.01})
            testing = client.post(f"/api/models/{model_id}/test")
            assert testing.status_code == 201
            assert testing.json()["locked"]
            assert testing.json()["testing_run"]["run_params"]["signal_start"] == "2024-02-01"
            assert client.put(f"/api/models/{model_id}", json=model_payload).status_code == 409
            training_job_id = training_run["job_id"]
            referenced_job_delete = client.delete(f"/api/jobs/{training_job_id}")
            assert referenced_job_delete.status_code == 409
            assert "模型会话" in referenced_job_delete.json()["detail"]
            invalid_model_payload = {**model_payload, "train_start": "2024-01-06"}
            invalid_model = client.post("/api/models", json=invalid_model_payload)
            assert invalid_model.status_code == 422
            assert "交易日范围" in invalid_model.json()["detail"]
            assert client.delete(f"/api/models/{model_id}").status_code == 204
            assert client.get(f"/api/models/{model_id}").status_code == 404
            assert client.delete(f"/api/models/{model_id}").status_code == 404
            assert client.delete(f"/api/jobs/{training_job_id}").status_code == 204
            assert client.get(f"/api/jobs/{training_job_id}").status_code == 404
            assert client.get(f"/api/runs/{training_run['id']}").status_code == 404

            default_payload = dict(model_payload)
            default_payload["model_name"] = "API 默认原始幅度模型"
            default_payload.pop("training_method")
            default_payload.pop("training_params", None)
            default_model_response = client.post("/api/models", json=default_payload)
            assert default_model_response.status_code == 201, default_model_response.text
            default_model = default_model_response.json()
            assert default_model["training_method"] == "winsorized_zscore_ridge"
            assert default_model["training_params"] == {
                "ridge_alpha": 1e-6,
                "winsor_lower_quantile": 0.01,
                "winsor_upper_quantile": 0.99,
            }
            assert client.delete(f"/api/models/{default_model['id']}").status_code == 204

            ridge_payload = {
                **model_payload,
                "model_name": "API Ridge 模型",
                "training_method": "winsorized_zscore_ridge",
                "training_params": {
                    "ridge_alpha": 0.001,
                    "winsor_lower_quantile": 0.01,
                    "winsor_upper_quantile": 0.99,
                },
            }
            ridge_model = client.post("/api/models", json=ridge_payload)
            assert ridge_model.status_code == 201
            assert ridge_model.json()["training_method"] == "winsorized_zscore_ridge"
            assert ridge_model.json()["training_params"]["ridge_alpha"] == 0.001
            assert not ridge_model.json()["can_run_testing"]
            invalid_training = client.post(
                "/api/models",
                json={**ridge_payload, "training_params": {"unknown": 1}},
            )
            assert invalid_training.status_code == 422
            assert "does not accept" in invalid_training.json()["detail"]
            assert client.delete(f"/api/models/{ridge_model.json()['id']}").status_code == 204
            correlation = client.get("/api/factor-correlation")
            assert correlation.status_code == 200
            assert correlation.json()["factor_names"] == ["api_correlation_factor"]
            assert correlation.json()["matrix"] == [[1.0]]
            second_candidate = client.post(
                "/api/test-factors",
                json={
                    "factor_name": "api_volume_correlation_factor",
                    "expression": "rank_cs(vol)",
                },
            )
            assert second_candidate.status_code == 201
            second_admitted = client.post(
                "/api/test-factors/webapp_test_factors/api_volume_correlation_factor/submit"
            )
            assert second_admitted.status_code == 201
            pair = client.get(
                "/api/factor-correlation/pair",
                params={
                    "factor_a": "api_correlation_factor",
                    "factor_b": "api_volume_correlation_factor",
                },
            )
            assert pair.status_code == 200
            pair_payload = pair.json()
            assert pair_payload["window_size"] == 60
            assert pair_payload["step_size"] == 60
            assert len(pair_payload["windows"]) == 2
            assert pair_payload["windows"][0]["start_day"] == "2024-01-02"
            assert pair_payload["windows"][-1]["end_day"] == "2024-06-17"
            assert pair_payload["max_abs_correlation"] == max(
                abs(row["correlation"])
                for row in pair_payload["windows"]
                if row["correlation"] is not None
            )
            assert pair_payload["violation_window_count"] == sum(
                abs(row["correlation"]) > pair_payload["threshold"]
                for row in pair_payload["windows"]
                if row["correlation"] is not None
            )
            same_factor_pair = client.get(
                "/api/factor-correlation/pair",
                params={
                    "factor_a": "api_correlation_factor",
                    "factor_b": "api_correlation_factor",
                },
            )
            assert same_factor_pair.status_code == 422
            duplicate = client.post(
                "/api/test-factors",
                json={
                    "factor_name": "api_duplicate_correlation_factor",
                    "expression": "rank_cs(c)",
                },
            )
            assert duplicate.status_code == 201
            rejected = client.post(
                "/api/test-factors/webapp_test_factors/api_duplicate_correlation_factor/submit"
            )
            assert rejected.status_code == 422
            assert "不超过 0.75" in rejected.json()["detail"]
            assert client.get("/api/factor-correlation").json()["factor_names"] == [
                "api_correlation_factor",
                "api_volume_correlation_factor",
            ]
            removed = client.delete(
                "/api/factors/webapp_factor_library/api_volume_correlation_factor"
            )
            assert removed.status_code == 204
            assert [
                row["factor_name"] for row in client.get("/api/factors").json()
            ] == ["api_correlation_factor"]
            source_after_withdrawal = client.get(
                "/api/test-factors"
            ).json()
            source_row = next(
                row
                for row in source_after_withdrawal
                if row["factor_name"] == "api_volume_correlation_factor"
            )
            assert not source_row["submitted"]
            rebuilt_matrix = client.get("/api/factor-correlation")
            assert rebuilt_matrix.status_code == 200
            assert rebuilt_matrix.json()["factor_names"] == ["api_correlation_factor"]
            assert client.delete(
                "/api/factors/webapp_factor_library/api_volume_correlation_factor"
            ).status_code == 404
            invalid = client.post(
                "/api/expressions/validate",
                json={"expression": "__import__('os').system('whoami')"},
            )
            assert invalid.status_code == 422
            resolved = client.post(
                "/api/methods/resolve", json={"names": ["rolling_sharpe"]}
            ).json()
            assert resolved["methods"] == ["quantile_returns", "rolling_sharpe"]
            cycle_resolved = client.post(
                "/api/methods/resolve", json={"names": ["cycle_context"]}
            ).json()
            assert cycle_resolved["methods"] == [
                "quantile_returns",
                "quantile_cumulative",
                "quantile_plot",
                "cycle_context",
            ]
            net_resolved = client.post(
                "/api/methods/resolve",
                json={"names": ["quantile_net_returns", "quantile_cumulative"]},
            ).json()
            assert net_resolved["methods"] == [
                "quantile_net_returns",
                "quantile_cumulative",
            ]
            methods = client.get("/api/methods").json()
            net_method = next(
                method
                for method in methods
                if method["name"] == "quantile_net_returns"
            )
            assert "quantile_returns" in net_method["provides"]
            cycle_method = next(
                method for method in methods if method["name"] == "cycle_context"
            )
            assert cycle_method["requires"] == ["quantile_plot"]
            expression_validation = client.post(
                "/api/expressions/validate",
                json={"expression": "where(c > o, np.log(ts_mean(c, 5)), c)"},
            ).json()
            assert expression_validation["operators"] == [
                "np.log",
                "ts_mean",
                "where",
            ]
            factor = factors.json()[0]
            tagged = client.put(
                f"/api/factors/{factor['batch_id']}/{factor['factor_name']}/tags",
                json={"tags": ["固定组合", "候选池"]},
            )
            assert tagged.status_code == 200
            assert tagged.json()["tags"] == ["候选池", "固定组合"]
            tag_catalog = client.get("/api/factor-tags?library=test")
            assert tag_catalog.status_code == 200
            assert {item["tag"] for item in tag_catalog.json()} >= {"固定组合", "候选池"}
            tag_job = client.post(
                "/api/jobs",
                json={
                    "kind": "evaluate",
                    "tags": ["固定组合", "候选池"],
                    "tag_match": "all",
                    "library": "test",
                    "methods": ["rank_ic", "rank_icir"],
                },
            )
            assert tag_job.status_code == 201
            assert tag_job.json()["total_runs"] == 1
            assert tag_job.json()["params"]["tag_selection"] == {
                "tags": ["固定组合", "候选池"],
                "match": "all",
                "library": "test",
            }
            created = client.post(
                "/api/jobs",
                json={
                    "kind": "evaluate",
                    "factors": [
                        {
                            "factor_name": factor["factor_name"],
                            "batch_id": factor["batch_id"],
                        }
                    ],
                    "methods": ["rank_ic", "rank_icir"],
                    "signal_start": "2024-01-10",
                    "signal_end": "2024-03-15",
                },
            )
            assert created.status_code == 201
            assert created.json()["runs"][0]["status"] == "queued"
            assert created.json()["params"]["signal_start"] == "2024-01-10"
            assert created.json()["params"]["signal_end"] == "2024-03-15"
            assert created.json()["runs"][0]["run_params"] == {
                "signal_start": "2024-01-10",
                "signal_end": "2024-03-15",
            }
            missing_window_end = client.post(
                "/api/jobs",
                json={
                    "kind": "evaluate",
                    "factors": [
                        {
                            "factor_name": factor["factor_name"],
                            "batch_id": factor["batch_id"],
                        }
                    ],
                    "methods": ["rank_ic", "rank_icir"],
                    "signal_start": "2024-01-10",
                },
            )
            assert missing_window_end.status_code == 422
            created_job_id = created.json()["id"]
            created_run_id = created.json()["runs"][0]["id"]
            active_job_delete = client.delete(f"/api/jobs/{created_job_id}")
            assert active_job_delete.status_code == 409
            assert "先取消或停止" in active_job_delete.json()["detail"]
            assert app.state.db.cancel_job(created_job_id)
            assert client.delete(f"/api/jobs/{created_job_id}").status_code == 204
            assert client.get(f"/api/jobs/{created_job_id}").status_code == 404
            assert client.get(f"/api/runs/{created_run_id}").status_code == 404
            assert client.delete(f"/api/jobs/{created_job_id}").status_code == 404
            template_rows = client.get("/api/templates").json()
            assert not {
                "快速IC筛查",
                "未来数据校验",
                "完整组合评估",
            } & {template["name"] for template in template_rows}
            default_template = next(
                template
                for template in template_rows
                if template["kind"] == "methods"
            )
            templated_evaluation = client.post(
                "/api/jobs",
                json={
                    "template_id": default_template["id"],
                    "factors": [
                        {
                            "factor_name": factor["factor_name"],
                            "batch_id": factor["batch_id"],
                        }
                    ],
                },
            )
            assert templated_evaluation.status_code == 201
            assert templated_evaluation.json()["kind"] == "evaluate"
            assert templated_evaluation.json()["runs"][0]["methods"] == default_template[
                "methods"
            ]
            editable_stages = stages.json()
            editable_stages[0]["methods"].append("rank_ic")
            funnel_template = client.post(
                "/api/templates",
                json={
                    "name": "可编辑漏斗模板",
                    "kind": "funnel",
                    "methods": ["rank_ic"],
                    "params": {
                        "horizon": 5,
                        "n_quantiles": 8,
                        "significance_level": 0.1,
                        "stages": editable_stages,
                    },
                },
            )
            assert funnel_template.status_code == 201
            saved_template = funnel_template.json()
            assert saved_template["methods"] == []
            assert saved_template["params"]["horizon"] == 5
            assert saved_template["params"]["n_quantiles"] == 8
            assert saved_template["params"]["significance_level"] == 0.1
            assert saved_template["params"]["stages"][0]["methods"] == [
                "future_data_perturbation",
                "rank_ic",
            ]
            updated_template = client.put(
                f"/api/templates/{saved_template['id']}",
                json={
                    "name": "已修改漏斗模板",
                    "kind": "funnel",
                    "params": {
                        "horizon": 3,
                        "n_quantiles": 6,
                        "significance_level": 0.01,
                        "stages": editable_stages,
                    },
                },
            )
            assert updated_template.status_code == 200
            assert updated_template.json()["name"] == "已修改漏斗模板"
            assert updated_template.json()["params"]["horizon"] == 3
            templated_job = client.post(
                "/api/jobs",
                json={
                    "template_id": saved_template["id"],
                    "factors": [
                        {
                            "factor_name": factor["factor_name"],
                            "batch_id": factor["batch_id"],
                        }
                    ],
                    "horizon": 3,
                    "n_quantiles": 6,
                    "significance_level": 0.01,
                },
            )
            assert templated_job.status_code == 201
            assert templated_job.json()["kind"] == "funnel"
            assert templated_job.json()["runs"][0]["methods"] == [
                "future_data_perturbation",
                "rank_ic",
            ]
            assert templated_job.json()["params"]["funnel_stages"][0]["methods"] == [
                "future_data_perturbation",
                "rank_ic",
            ]

            compare_output = temporary_root / "compare-run"
            (compare_output / "details").mkdir(parents=True)
            (compare_output / "details" / "ic.csv").write_text(
                "day,ic\n2026-01-01,1\n2026-01-02,1\n",
                encoding="utf-8-sig",
            )
            (compare_output / "details" / "ic__2.csv").write_text(
                "day,ic\n2026-01-01,2\n2026-01-02,3\n",
                encoding="utf-8-sig",
            )
            (compare_output / "details" / "cumulative_returns.csv").write_text(
                "day,G1,G10\n2026-01-01,0.01,0.03\n2026-01-02,0.02,0.05\n",
                encoding="utf-8-sig",
            )
            app.state.db.create_job(
                {
                    "id": "compare-job",
                    "kind": "evaluate",
                    "title": "compare",
                    "params": {},
                    "created_at": "2026-07-05T00:00:00+08:00",
                },
                [
                    {
                        "id": "compare-run",
                        "job_id": "compare-job",
                        "factor_name": "compare_factor",
                        "batch_id": "synthetic",
                        "expression": "rank_cs(c)",
                        "horizon": 1,
                        "n_quantiles": 10,
                        "methods": ["rank_ic", "rank_ic"],
                        "output_dir": str(compare_output),
                        "created_at": "2026-07-05T00:00:00+08:00",
                    }
                ],
            )
            app.state.db.complete_run(
                "compare-run",
                {
                    "evaluation_details": {
                        "ic": "details/ic.csv",
                        "ic__2": "details/ic__2.csv",
                        "cumulative_returns": "details/cumulative_returns.csv",
                    }
                },
            )
            comparison = client.post(
                "/api/compare", json={"run_ids": ["compare-run"]}
            )
            assert comparison.status_code == 200
            cumulative_ic = comparison.json()["curves"]["cumulative_ic"][0]["data"]
            assert cumulative_ic == [["2026-01-01", 2.0], ["2026-01-02", 5.0]]
            top_quantile = comparison.json()["curves"]["top_quantile"][0]["data"]
            assert top_quantile == [["2026-01-01", 0.03], ["2026-01-02", 0.05]]

        registry_dir = temporary_root / "service-registry"
        registry_dir.mkdir()
        write_alpha_registry(registry_dir)
        service = RegistryService(
            Settings(
                project_root=PROJECT_ROOT,
                data_dir=data_dir,
                registry_dir=registry_dir,
                state_dir=temporary_root / "custom-state",
                frontend_dist=PROJECT_ROOT / "webapp" / "frontend" / "dist",
            )
        )
        custom = service.create(
            {
                "factor_name": "web_test_factor",
                "expression": "rank_cs(delta(c, 5))",
                "project": "华夏191",
                "tags": ["固定组合"],
            }
        )
        assert custom["number"] == 1
        assert custom["implementation_set"] == "webapp_test"
        assert custom["library_scope"] == "test"
        assert custom["project"] == "华夏191"
        assert custom["tags"] == ["固定组合"]
        assert service.find("webapp_test_factors", "web_test_factor")
        submitted = service.submit("webapp_test_factors", "web_test_factor")
        assert submitted and submitted["batch_id"] == "webapp_factor_library"
        assert submitted["project"] == "华夏191"
        reassigned = service.set_project(
            "webapp_factor_library", "web_test_factor", "正式项目"
        )
        assert reassigned and reassigned["project"] == "正式项目"
        assert submitted["tags"] == ["固定组合"]
        assert service.factors("factor")[0]["factor_name"] == "web_test_factor"
        matrix = service.correlation_matrix()
        assert matrix["factor_names"] == ["web_test_factor"]
        assert matrix["matrix"] == [[1.0]]
        duplicate = service.create(
            {
                "factor_name": "duplicate_web_test_factor",
                "expression": "rank_cs(delta(c, 5))",
            }
        )
        try:
            service.submit(duplicate["batch_id"], duplicate["factor_name"])
        except ValueError as exc:
            assert "不超过 0.75" in str(exc)
        else:
            raise AssertionError("a fully correlated submitted factor must be rejected")
        assert service.correlation_matrix()["factor_names"] == ["web_test_factor"]
        assert service.delete("webapp_test_factors", "web_test_factor")
        assert service.delete("webapp_test_factors", "duplicate_web_test_factor")
        assert not (registry_dir / "webapp_test_factors.json").exists()

        funnel_settings = Settings(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            registry_dir=registry_dir,
            state_dir=temporary_root / "funnel-state",
            frontend_dist=PROJECT_ROOT / "webapp" / "frontend" / "dist",
        )
        funnel_settings.runs_dir.mkdir(parents=True)
        funnel_db = Database(funnel_settings.db_path)
        funnel_db.initialize()
        custom_funnel_stages = default_funnel_stages()
        custom_funnel_stages[1]["methods"].append("quantile_returns")
        funnel_run = first_funnel_run(
            job_id="funnel-job",
            factor={
                "factor_name": "funnel_factor",
                "batch_id": "synthetic",
                "expression": "rank_cs(c)",
            },
            horizon=1,
            n_quantiles=10,
            runs_dir=funnel_settings.runs_dir,
            stages=custom_funnel_stages,
            run_params={
                "signal_start": "2024-01-10",
                "signal_end": "2024-03-15",
            },
        )
        funnel_db.create_job(
            {
                "id": "funnel-job",
                "kind": "funnel",
                "title": "funnel",
                "params": {
                    "significance_level": 0.05,
                    "funnel_stages": custom_funnel_stages,
                },
                "created_at": "2026-07-05T00:00:00+08:00",
            },
            [funnel_run],
        )
        assert funnel_db.claim_next_run()["id"] == funnel_run["id"]
        WorkerSupervisor(funnel_db, funnel_settings)._handle_success(
            funnel_run["id"], {"future_perturbation_changed_values": 1}
        )
        funnel_job = funnel_db.get_job("funnel-job")
        assert funnel_job and len(funnel_job["runs"]) == 4
        assert funnel_job["runs"][0]["gate_outcome"] == "eliminated"
        assert [run["status"] for run in funnel_job["runs"][1:]] == [
            "skipped",
            "skipped",
            "skipped",
        ]
        assert funnel_job["runs"][1]["methods"][-1] == "quantile_returns"
        assert all(
            run["run_params"]
            == {"signal_start": "2024-01-10", "signal_end": "2024-03-15"}
            for run in funnel_job["runs"]
        )

    print("webapp API contract passed")


if __name__ == "__main__":
    main()
