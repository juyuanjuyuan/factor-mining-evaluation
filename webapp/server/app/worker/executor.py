"""Long-lived child process with mtime-aware market-data preloading."""

from __future__ import annotations

import os
import json
import signal
import traceback
from pathlib import Path
from queue import Empty
from typing import Any


def _data_signature(data_dir: str) -> tuple[tuple[str, int, int], ...]:
    """Return a cheap signature for every configured market-data input."""

    from engine import DEFAULT_FILES

    root = Path(data_dir)
    signature = []
    for filename in sorted(set(DEFAULT_FILES.values())):
        path = root / filename
        if path.is_file():
            stat = path.stat()
            signature.append((filename, stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((filename, -1, -1))
    return tuple(signature)


def _load_all(data_dir: str) -> dict[str, Any]:
    import pandas as pd

    from engine import (
        DEFAULT_FILES,
        EVALUATOR_ONLY_DATA_SYMBOLS,
        normalize_market_data_frame,
    )

    root = Path(data_dir)
    data = {}
    for symbol, filename in DEFAULT_FILES.items():
        if symbol in EVALUATOR_ONLY_DATA_SYMBOLS:
            continue
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required market-data input: {path}")
        data[symbol] = pd.read_parquet(path).sort_index()
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
        data_signature = _data_signature(data_dir)
        event_queue.put({"type": "ready"})
    except BaseException:
        event_queue.put({"type": "startup_failed", "error": traceback.format_exc()})
        return

    import pandas as pd

    from engine import (
        DEFAULT_FILES,
        EVALUATOR_ONLY_DATA_SYMBOLS,
        evaluate_factor_expression,
        normalize_market_data_frame,
    )
    from evaluators.base import (
        REGISTERED_EVALUATION_METHODS,
        evaluation_required_data_symbols,
    )
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
            current_signature = _data_signature(data_dir)
            if current_signature != data_signature:
                # Reload every core matrix together so a changed close axis
                # cannot leave other cached inputs on stale axes. Evaluator-only
                # inputs are loaded lazily again below when the run needs them.
                preloaded = _load_all(data_dir)
                data_signature = current_signature
            run_params = command.get("run_params") or {}
            expression = command["expression"]
            selected_methods = [
                REGISTERED_EVALUATION_METHODS[name] for name in command["methods"]
            ]
            optional_symbols = (
                evaluation_required_data_symbols(selected_methods)
                & EVALUATOR_ONLY_DATA_SYMBOLS
            )
            for symbol in sorted(optional_symbols - set(preloaded)):
                path = Path(data_dir) / DEFAULT_FILES[symbol]
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {symbol!r} evaluator input required by this pipeline: {path}"
                    )
                frame = pd.read_parquet(path).sort_index()
                preloaded[symbol] = normalize_market_data_frame(
                    symbol,
                    frame,
                    preloaded["c"],
                )
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
                decay=run_params.get("decay", 1),
                preloaded_data=preloaded,
                # Execute the stored pipeline exactly as ordered — it may
                # legitimately repeat a method (IC before/after neutralization).
                evaluation_methods=selected_methods,
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
