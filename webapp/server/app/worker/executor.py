"""Long-lived child process that preloads market matrices once."""

from __future__ import annotations

import os
import json
import signal
import traceback
from pathlib import Path
from queue import Empty
from typing import Any


def _load_all(data_dir: str) -> dict[str, Any]:
    import pandas as pd

    from engine import DEFAULT_FILES, normalize_market_data_frame

    root = Path(data_dir)
    data = {
        symbol: pd.read_parquet(root / filename).sort_index()
        for symbol, filename in DEFAULT_FILES.items()
    }
    close = data["c"]
    for symbol, frame in list(data.items()):
        if symbol != "c":
            data[symbol] = normalize_market_data_frame(symbol, frame, close)
    return data


def worker_main(command_queue: Any, event_queue: Any, data_dir: str) -> None:
    # Uvicorn/terminal SIGINT belongs to the API parent. The supervisor sends
    # the explicit shutdown sentinel and owns forced termination.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    try:
        preloaded = _load_all(data_dir)
        event_queue.put({"type": "ready"})
    except BaseException:
        event_queue.put({"type": "startup_failed", "error": traceback.format_exc()})
        return

    from engine import evaluate_factor_expression
    from evaluators.base import REGISTERED_EVALUATION_METHODS
    from model_training import ModelTerm, ModelTrainingContext, run_model_training

    while True:
        try:
            command = command_queue.get(timeout=1)
        except Empty:
            continue
        if command is None:
            return
        run_id = command["id"]
        event_queue.put({"type": "started", "run_id": run_id})
        try:
            run_params = command.get("run_params") or {}
            expression = command["expression"]
            training_event = None
            training_request = run_params.get("model_training")
            if training_request:
                terms = tuple(
                    ModelTerm(
                        batch_id=str(term["batch_id"]),
                        factor_name=str(term["factor_name"]),
                        expression=str(term["expression"]),
                        weight=float(term["weight"]),
                    )
                    for term in training_request["terms"]
                )
                context = ModelTrainingContext(
                    terms=terms,
                    market_data=preloaded,
                    signal_start=str(run_params["signal_start"]),
                    signal_end=str(run_params["signal_end"]),
                    horizon=int(command["horizon"]),
                    parameters=dict(training_request.get("parameters") or {}),
                )
                fitted = run_model_training(training_request.get("method"), context)
                expression = fitted.expression
                fit_payload = fitted.as_dict(context, str(training_request["method"]))
                training_event = {
                    "model_id": run_params["model_test_id"],
                    "expression": expression,
                    "result": fit_payload,
                }
            result = evaluate_factor_expression(
                factor_name=command["factor_name"],
                expression=expression,
                data_dir=data_dir,
                output_dir=command["output_dir"],
                horizon=command["horizon"],
                n_quantiles=command["n_quantiles"],
                preloaded_data=preloaded,
                # Execute the stored pipeline exactly as ordered — it may
                # legitimately repeat a method (IC before/after neutralization).
                evaluation_methods=[
                    REGISTERED_EVALUATION_METHODS[name] for name in command["methods"]
                ],
                signal_start=run_params.get("signal_start"),
                signal_end=run_params.get("signal_end"),
            )
            if training_event:
                artifact_path = Path(command["output_dir"]) / "model_training.json"
                artifact_path.write_text(
                    json.dumps(training_event["result"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result["metrics"]["model_training_method"] = training_event["result"]["method"]
                result["metrics"]["model_training_result"] = training_event["result"]
                result["metrics"]["model_training_artifact"] = artifact_path.name
            event_queue.put(
                {
                    "type": "succeeded",
                    "run_id": run_id,
                    "metrics": result["metrics"],
                    "model_training": training_event,
                }
            )
        except BaseException:
            event_queue.put(
                {"type": "failed", "run_id": run_id, "error": traceback.format_exc()}
            )
