"""
Sensitivity analysis for ma_smf and prob_coverage hyperparameters:
  1. Diversification intensity λ (theta)
  2. Soft-binning bandwidth σ  (sigma_geo / sigma_time)
  3. Geographic grid resolution (grid_size)

For each axis, we sweep one parameter while holding the others fixed at their
best-found values from the grid search.

Usage (run from the msdpp project root):
    # VG (time task)
    uv run python examples/sensitivity_analysis.py --dataset vg

    # I1M (geo task)
    uv run python examples/sensitivity_analysis.py --dataset i1m

    # All PP tasks
    uv run python examples/sensitivity_analysis.py --dataset pp

    # All at once
    uv run python examples/sensitivity_analysis.py --dataset all

Results are saved to results/sensitivity/<dataset>/<method>_<axis>.json
Each JSON has the form:
  {
    "axis": "theta" | "sigma_geo" | "sigma_time" | "grid_size",
    "fixed": { ... other params ... },
    "sweep": [
      {"param_value": 0.1, "increase": {...metrics...}, "decrease": {...metrics...}},
      ...
    ]
  }
"""
import argparse
import json
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from msdpp import registry
from msdpp.task import BaseTask, DivDir, TaskResult
from msdpp.schema.evaluator import EvalIndices

import msdpp.div_method  # noqa: F401  registers all methods

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sensitivity")

# ------------------------------------------------------------------
# Dataset → best-known fixed params (from Table 1/2 val-best hyperparameters)
#
# CHANGED: was a single "best_*" per dataset, applied to BOTH directions.
# That's wrong: PP_geo_hour_decrease and PP_geo_hour_increase have DIFFERENT
# val-best σ. Sweeping at a per-DATASET fixed σ characterizes a regime that
# is at neither direction's actual operating point.
#
# Now split into best_*_decrease and best_*_increase per method; the sweep
# runner picks the one matching the direction argument at call time.
# See results/sensitivity_v2/SUMMARY.md for the corrected σ-dependent-cliff
# findings that this restructure produced.
# ------------------------------------------------------------------
DATASET_META = {
    "PP_geo": {
        "data_is_pp": True,
        "retrieval_metric": "r10",
        # MASCOT decrease: θ=0.3 σ_geo=10 σ_time=0.5 → R@10=0.8105 (Table 1)
        # MASCOT increase: θ=0.8 σ_geo=15 σ_time=0.5 → R@10=0.9109 (Table 2)
        "best_ma_smf_decrease":  {"theta": 0.3, "sigma_geo": 10.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_ma_smf_increase":  {"theta": 0.8, "sigma_geo": 15.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_decrease": {"theta": 0.1, "sigma_geo": 10.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_increase": {"theta": 0.1, "sigma_geo": 10.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
    },
    "PP_hour": {
        "data_is_pp": True,
        "retrieval_metric": "r10",
        # MASCOT decrease: θ=0.4 σ_geo=1 σ_time=0.5 → R@10=0.9059
        # MASCOT increase: θ=0.9 σ_geo=1 σ_time=1.5 → R@10=0.8921
        "best_ma_smf_decrease":  {"theta": 0.4, "sigma_geo": 1.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_ma_smf_increase":  {"theta": 0.9, "sigma_geo": 1.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_decrease": {"theta": 0.1, "sigma_geo": 1.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_increase": {"theta": 0.1, "sigma_geo": 1.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
    },
    "PP_geo_hour": {
        "data_is_pp": True,
        "retrieval_metric": "r10",
        # MASCOT decrease: θ=0.1 σ_geo=15 σ_time=3.0 → R@10=0.9410 (Table 1 flagship)
        # MASCOT increase: θ=0.8 σ_geo=15 σ_time=1.5 → R@10=0.8356
        "best_ma_smf_decrease":  {"theta": 0.1, "sigma_geo": 15.0, "sigma_time": 3.0, "grid_size": 20, "num_bins": 24},
        "best_ma_smf_increase":  {"theta": 0.8, "sigma_geo": 15.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_decrease": {"theta": 0.5, "sigma_geo": 10.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_increase": {"theta": 0.4, "sigma_geo": 10.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
    },
    "VG_hour": {
        "data_is_pp": True,
        "retrieval_metric": "r10",
        "best_ma_smf_decrease":  {"theta": 0.4, "sigma_geo": 1.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_ma_smf_increase":  {"theta": 0.7, "sigma_geo": 1.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_decrease": {"theta": 0.4, "sigma_geo": 1.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_increase": {"theta": 0.4, "sigma_geo": 1.0, "sigma_time": 1.5, "grid_size": 20, "num_bins": 24},
    },
    "I1M_geo": {
        "data_is_pp": False,
        "retrieval_metric": "map_",
        "best_ma_smf_decrease":  {"theta": 0.7, "sigma_geo": 5.0,  "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_ma_smf_increase":  {"theta": 0.7, "sigma_geo": 10.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_decrease": {"theta": 0.2, "sigma_geo": 10.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
        "best_prob_coverage_increase": {"theta": 0.2, "sigma_geo": 10.0, "sigma_time": 0.5, "grid_size": 20, "num_bins": 24},
    },
    "Flickr30k_geo_hour": {
        "data_is_pp": False,
        "retrieval_metric": "map_",
        "best_ma_smf":       {"theta": 0.5, "sigma_geo": 10.0, "sigma_time": 1.5,  "grid_size": 20, "num_bins": 24},
        "best_prob_coverage": {"theta": 0.4, "sigma_geo": 10.0, "sigma_time": 1.5,  "grid_size": 20, "num_bins": 24},
    },
}

# ------------------------------------------------------------------
# Sweep ranges for each axis
# ------------------------------------------------------------------
SWEEPS = {
    "theta":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "sigma_geo": [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    "sigma_time":[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "grid_size": [5, 10, 15, 20, 25, 30, 40, 50],
}

# ------------------------------------------------------------------
# Dataset → which axes are relevant
# (geo tasks sweep sigma_geo & grid_size; time tasks sweep sigma_time & num_bins)
# ------------------------------------------------------------------
DATASET_AXES = {
    "PP_geo":     ["theta", "sigma_geo", "grid_size"],
    "PP_hour":    ["theta", "sigma_time"],
    "PP_geo_hour":["theta", "sigma_geo", "sigma_time", "grid_size"],
    "VG_hour":    ["theta", "sigma_time"],
    "I1M_geo":    ["theta", "sigma_geo", "grid_size"],
    "Flickr30k_geo_hour": ["theta", "sigma_geo", "sigma_time", "grid_size"],
}

DATASET_GROUPS = {
    "pp":  ["PP_geo", "PP_hour", "PP_geo_hour"],
    "vg":  ["VG_hour"],
    "i1m": ["I1M_geo"],
    "flickr30k": ["Flickr30k_geo_hour"],
    "all": ["PP_geo", "PP_hour", "PP_geo_hour", "VG_hour", "I1M_geo", "Flickr30k_geo_hour"],
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def metrics_from_indices(indices: EvalIndices, retrieval_metric: str) -> dict:
    ret = getattr(indices, retrieval_metric).item()
    return {
        retrieval_metric: round(ret, 4),
        "ext_vendi":  round(indices.ext_vendi.item(), 4),
        "img_vendi":  round(indices.img_vendi.item(), 4),
        "mean_vendi": round(indices.mean_vendi.item(), 4),
        "map_":       round(indices.map_.item(), 4),
        "r10":        round(indices.r10.item(), 4),
    }


def run_one(task: BaseTask, div_method_name: str, params: dict, direction: DivDir) -> EvalIndices:
    div_cls = registry.get_div_method(div_method_name)
    div_instance = div_cls(**params)
    result: TaskResult = task.run(div_instance, direction, force=True)
    return result.eval_indices


def load_task(dataset_name: str, split: str, hf_home: Path, overall_cfg: dict,
              grid_size: int = 20, num_bins: int = 24, n_thread: int = 8) -> BaseTask:
    pkl_path = hf_home / "tasks" / f"{dataset_name}_{split}.pkl"
    dataset = torch.load(pkl_path, weights_only=False)
    model_name = overall_cfg["model_name"]
    model_params = overall_cfg["model_params"]
    model = registry.get_model(model_name)(**model_params)
    return BaseTask(
        dataset=dataset,
        model=model,
        subset_k=overall_cfg["subset_k"],
        top_k=overall_cfg["top_k"],
        n_thread=n_thread,
        grid_size=grid_size,
        num_bins=num_bins,
    )


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
def run_sensitivity(dataset_name: str, method: str, axis: str,
                    task: BaseTask, best_params: dict,
                    retrieval_metric: str, out_dir: Path,
                    hf_home: Path, overall_cfg: dict, n_thread: int = 8,
                    best_params_decrease: dict | None = None) -> None:
    sweep_values = SWEEPS[axis]
    # grid_size / num_bins belong to BaseTask, not the div method
    task_level_axes = {"grid_size", "num_bins"}
    # Use direction-specific params for decrease if provided, otherwise fall back to best_params
    dec_params = best_params_decrease if best_params_decrease is not None else best_params
    records = []

    for val in tqdm(sweep_values, desc=f"{dataset_name}/{method}/{axis}"):
        row: dict = {"param_value": val}
        for direction, dir_key, params in [
            (DivDir.INCREASE, "increase", best_params),
            (DivDir.DECREASE, "decrease", dec_params),
        ]:
            if axis in task_level_axes:
                sweep_task = load_task(
                    dataset_name, "test", hf_home, overall_cfg,
                    grid_size=val if axis == "grid_size" else params.get("grid_size", 20),
                    num_bins=val  if axis == "num_bins"  else params.get("num_bins", 24),
                    n_thread=n_thread,
                )
                div_params = {k: v for k, v in params.items() if k not in task_level_axes}
            else:
                sweep_task = task
                div_params = {**{k: v for k, v in params.items() if k not in task_level_axes},
                              axis: val}
            indices = run_one(sweep_task, method, div_params, direction)
            row[dir_key] = metrics_from_indices(indices, retrieval_metric)

        records.append(row)
        logger.info(
            "%s | %s | %s=%.3g | INC %s=%.4f | DEC %s=%.4f",
            dataset_name, method, axis, val,
            retrieval_metric, row["increase"][retrieval_metric],
            retrieval_metric, row["decrease"][retrieval_metric],
        )

    fixed_params = {k: v for k, v in best_params.items() if k != axis}
    result = {
        "dataset": dataset_name,
        "method": method,
        "axis": axis,
        "fixed": fixed_params,
        "sweep": records,
    }

    out_path = out_dir / f"{method}_{axis}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved → %s", out_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASET_GROUPS.keys()))
    parser.add_argument("--method", default="both",
                        choices=["ma_smf", "prob_coverage", "both"])
    parser.add_argument("--config_dir", default="examples/configs")
    parser.add_argument("--result_dir", default="results/sensitivity_v2",
                        help="Output directory. Default is sensitivity_v2 (the "
                             "per-direction sweeps that supersede the old "
                             "results/sensitivity/ single-per-dataset σ sweeps).")
    parser.add_argument("--n_thread", type=int, default=8,
                        help="Threads for parallelising across queries (default: 8). "
                             "Max useful = min(n_queries, n_cpu_cores). "
                             "GPU is shared so >16 rarely helps.")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    hf_home = Path(os.environ.get("HF_HOME", "./"))

    with (config_dir / "overall.json").open() as f:
        overall_cfg = json.load(f)

    methods = ["ma_smf", "prob_coverage"] if args.method == "both" else [args.method]
    dataset_names = DATASET_GROUPS[args.dataset]

    for dataset_name in dataset_names:
        logger.info("=== Dataset: %s ===", dataset_name)
        meta = DATASET_META[dataset_name]
        axes = DATASET_AXES[dataset_name]

        out_dir = Path(args.result_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load test task once per dataset (shared across methods/axes that don't change grid_size)
        test_task = load_task(dataset_name, "test", hf_home, overall_cfg, n_thread=args.n_thread)

        for method in methods:
            # DATASET_META was restructured to per-direction hyperparameters
            # (best_{method}_increase / best_{method}_decrease). Older versions
            # of this script also accepted a shared "best_{method}" fallback.
            inc_key = f"best_{method}_increase"
            dec_key = f"best_{method}_decrease"
            legacy_key = f"best_{method}"
            best_params = (meta[inc_key] if inc_key in meta else meta[legacy_key]).copy()
            best_params_decrease = (meta[dec_key] if dec_key in meta else meta.get(legacy_key, {})).copy()

            for axis in axes:
                # Skip geo-only axes for time-only datasets and vice-versa
                if axis in ("sigma_geo", "grid_size") and "geo" not in dataset_name.lower():
                    continue
                if axis == "sigma_time" and "hour" not in dataset_name.lower():
                    continue

                out_path = out_dir / f"{method}_{axis}.json"
                if out_path.exists():
                    logger.info("Skipping (already done): %s", out_path)
                    continue

                run_sensitivity(
                    dataset_name=dataset_name,
                    method=method,
                    axis=axis,
                    task=test_task,
                    best_params=best_params,
                    retrieval_metric=meta["retrieval_metric"],
                    out_dir=out_dir,
                    hf_home=hf_home,
                    overall_cfg=overall_cfg,
                    n_thread=args.n_thread,
                    best_params_decrease=best_params_decrease,
                )

        logger.info("=== Done: %s ===", dataset_name)


if __name__ == "__main__":
    main()
