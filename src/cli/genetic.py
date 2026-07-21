#!/usr/bin/env python3
"""Continuous genetic-programming factor discovery CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genetic_mining import EvolutionConfig, GeneticMiningRunner, MiningCampaignConfig
from paths import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT


def _comma_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "gp_factor_mining",
    )
    parser.add_argument(
        "--library-file",
        type=Path,
        default=PROJECT_ROOT / "factor_registry" / "webapp_factor_library.json",
    )
    parser.add_argument(
        "--correlation-state-dir",
        type=Path,
        default=OUTPUT_DIR / "webapp",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument(
        "--preprocess-mode",
        choices=("paper_local", "market_cap", "none"),
        default="paper_local",
    )
    parser.add_argument("--minimum-ic-days", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--hall-of-fame", type=int, default=100)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--init-depth-min", type=int, default=1)
    parser.add_argument("--init-depth-max", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=20)
    parser.add_argument("--parsimony-coefficient", type=float, default=0.0001)
    parser.add_argument("--p-crossover", type=float, default=0.40)
    parser.add_argument("--p-subtree-mutation", type=float, default=0.01)
    parser.add_argument("--p-hoist-mutation", type=float, default=0.0)
    parser.add_argument("--p-point-mutation", type=float, default=0.01)
    parser.add_argument("--p-point-replace", type=float, default=0.40)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=127)
    parser.add_argument("--candidate-correlation-threshold", type=float, default=0.90)
    parser.add_argument("--elite-size", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--windows", type=_comma_ints, default=(2, 3, 5, 10, 20, 40, 60))
    parser.add_argument(
        "--terminal",
        action="append",
        dest="terminals",
        help="repeat to replace the paper terminal set; engine expressions are accepted",
    )
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--minimum-ic-mean", type=float, default=0.0)
    parser.add_argument("--minimum-sharpe-60-median", type=float, default=1.0)
    parser.add_argument("--minimum-annualized-return", type=float, default=0.30)
    parser.add_argument("--correlation-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20190610)
    parser.add_argument("--project", default="遗传规划")
    parser.add_argument("--no-admit", action="store_true")
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=60.0)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> MiningCampaignConfig:
    defaults = EvolutionConfig()
    evolution = EvolutionConfig(
        generations=args.generations,
        population_size=args.population_size,
        hall_of_fame=args.hall_of_fame,
        n_components=args.components,
        init_depth_min=args.init_depth_min,
        init_depth_max=args.init_depth_max,
        tournament_size=args.tournament_size,
        parsimony_coefficient=args.parsimony_coefficient,
        p_crossover=args.p_crossover,
        p_subtree_mutation=args.p_subtree_mutation,
        p_hoist_mutation=args.p_hoist_mutation,
        p_point_mutation=args.p_point_mutation,
        p_point_replace=args.p_point_replace,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        candidate_correlation_threshold=args.candidate_correlation_threshold,
        elite_size=args.elite_size,
        n_jobs=args.n_jobs,
        terminals=tuple(args.terminals) if args.terminals else defaults.terminals,
        windows=tuple(args.windows),
        exponents=defaults.exponents,
        constant_range=defaults.constant_range,
    )
    return MiningCampaignConfig(
        campaign=args.campaign,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        library_file=args.library_file,
        correlation_state_dir=args.correlation_state_dir,
        evolution=evolution,
        horizon=args.horizon,
        n_quantiles=args.quantiles,
        preprocess_mode=args.preprocess_mode,
        minimum_ic_days=args.minimum_ic_days,
        significance_level=args.significance_level,
        minimum_ic_mean=args.minimum_ic_mean,
        minimum_rolling_sharpe_60_median=args.minimum_sharpe_60_median,
        minimum_annualized_return=args.minimum_annualized_return,
        correlation_threshold=args.correlation_threshold,
        seed=args.seed,
        admit=not args.no_admit,
        project=args.project,
    )


def main() -> int:
    args = build_parser().parse_args()
    config = build_config(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_fingerprint": config.fingerprint,
                    "config": config.as_dict(),
                    "mode": "continuous" if args.forever else "single_cycle",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    runner = GeneticMiningRunner(config)
    runner.run(
        forever=args.forever,
        pause_seconds=args.pause_seconds,
        max_cycles=args.max_cycles,
        stop_on_error=args.stop_on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
