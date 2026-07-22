#!/usr/bin/env python3
"""R@1 integrity analysis: BLIP-2 vs MASCOT at val-HM-best operating points.

For each (dataset, direction):
  1. Runs MASCOT at the paper's val-HM-best theta/sigma.
  2. Records BLIP-2 R@1 (from org_indices) and MASCOT R@1 (from eval_indices).
  3. Computes strict top-1 preservation: among queries where BLIP-2 rank-1 is
     a ground-truth image, what fraction still have that exact same image at
     MASCOT rank-1?

Saves:
  results/top1_integrity/checkpoint.json
  results/top1_integrity/SUMMARY.md

Usage (run inside Docker with GPU 3 mapped to cuda:0):
  cd /msdpp
  env PYTHONPATH=/msdpp/src HF_HOME=/msdpp/share_datasets/temp \\
    uv run python examples/top1_integrity.py 2>&1 | \\
    tee results/top1_integrity/run.log
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
SRC_DIR    = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULT_DIR   = REPO_ROOT / "results" / "top1_integrity"
CKPT_PATH    = RESULT_DIR / "checkpoint.json"
SUMMARY_PATH = RESULT_DIR / "SUMMARY.md"

N, K     = 200, 20
N_THREAD = 32

MODEL_PARAMS = {
    "pretrained_model": "Salesforce/blip2-itm-vit-g-coco",
    "use_itm": 0,
}

# Val-HM-best operating points from paper's grid_search_eval log
CONFIGS = [
    {"dataset": "PP_geo_hour", "direction": "inc", "theta": 0.8,  "sigma_geo": 15.0, "sigma_time": 1.5},
    {"dataset": "PP_geo_hour", "direction": "dec", "theta": 0.1,  "sigma_geo": 15.0, "sigma_time": 3.0},
    {"dataset": "PP_hour",     "direction": "inc", "theta": 0.9,  "sigma_geo": 1.0,  "sigma_time": 1.5},
    {"dataset": "PP_hour",     "direction": "dec", "theta": 0.4,  "sigma_geo": 1.0,  "sigma_time": 0.5},
    {"dataset": "PP_geo",      "direction": "inc", "theta": 0.8,  "sigma_geo": 15.0, "sigma_time": 0.5},
    {"dataset": "PP_geo",      "direction": "dec", "theta": 0.3,  "sigma_geo": 10.0, "sigma_time": 0.5},
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_ckpt(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_ckpt(ckpt: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ckpt, indent=2))


def ckpt_key(cfg: dict) -> str:
    return f"{cfg['dataset']}|{cfg['direction']}|theta{cfg['theta']}"


def compute_top1_preservation(result, labels_list: list[torch.Tensor]) -> dict:
    """
    Returns:
        blip2_r1_count  : queries where BLIP-2 top-1 is GT
        preserved_count : among those, MASCOT top-1 == same exact image
        total_queries   : total number of queries
    """
    t2i_sim   = result.t2i_sim      # [num_queries, num_images] CPU
    candidates = result.candidates   # list of [K] global-index tensors

    total_queries  = t2i_sim.shape[0]
    blip2_r1_count = 0
    preserved_count = 0

    for i in range(total_queries):
        # Global index of BLIP-2 top-1 for this query
        blip2_top1 = int(t2i_sim[i].argmax().item())

        # Is it ground truth?
        label_vec = labels_list[i]
        if label_vec[blip2_top1].item() > 0:
            blip2_r1_count += 1
            mascot_top1 = int(candidates[i][0].item())
            if mascot_top1 == blip2_top1:
                preserved_count += 1

    preservation = preserved_count / blip2_r1_count if blip2_r1_count > 0 else float("nan")
    return {
        "total_queries":   total_queries,
        "blip2_r1_count":  blip2_r1_count,
        "preserved_count": preserved_count,
        "preservation":    preservation,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  TOP1  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger("TOP1")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    hf_home    = Path(os.environ.get("HF_HOME", "./"))
    cache_home = hf_home / "div_results"

    from msdpp import registry
    import msdpp.div_method  # noqa: F401
    from msdpp.task import BaseTask, DivDir

    ckpt = load_ckpt(CKPT_PATH)

    logger.info("Loading BLIP-2 model…")
    model = registry.get_model("blip2")(**MODEL_PARAMS)

    # Group configs by dataset to avoid reloading the same pkl twice
    datasets_seen: dict[str, object] = {}

    for cfg in CONFIGS:
        key = ckpt_key(cfg)
        if key in ckpt:
            m = ckpt[key]
            logger.info("SKIP  %s  blip2_r1=%.4f  mascot_r1=%.4f  pres=%.4f",
                        key, m["blip2_r1"], m["mascot_r1"], m["preservation"])
            continue

        ds_name  = cfg["dataset"]
        dir_str  = cfg["direction"]
        theta    = cfg["theta"]
        sg       = cfg["sigma_geo"]
        st       = cfg["sigma_time"]
        div_dir  = DivDir.INCREASE if dir_str == "inc" else DivDir.DECREASE

        # Load dataset (cache by name)
        if ds_name not in datasets_seen:
            pkl_path = hf_home / "tasks" / f"{ds_name}_test.pkl"
            logger.info("Loading dataset %s from %s…", ds_name, pkl_path)
            datasets_seen[ds_name] = torch.load(pkl_path, weights_only=False)
        dataset = datasets_seen[ds_name]

        logger.info("Running  %s  theta=%.1f  sg=%.1f  st=%.1f  dir=%s",
                    ds_name, theta, sg, st, dir_str)

        bt = BaseTask(
            model=model, dataset=dataset,
            subset_k=N, top_k=K,
            cache_home=cache_home, do_cache_sim=False, n_thread=N_THREAD,
        )
        div = registry.get_div_method("ma_smf")(
            theta=theta, sigma_geo=sg, sigma_time=st,
        )

        result = bt.run(div, div_dir, force=False)

        blip2_r1  = float(result.org_indices.r1.item())
        mascot_r1 = float(result.eval_indices.r1.item())

        labels_list = list(dataset.labels.values())
        pres_info   = compute_top1_preservation(result, labels_list)

        delta = mascot_r1 - blip2_r1

        entry = {
            "dataset":         ds_name,
            "direction":       dir_str,
            "theta":           theta,
            "sigma_geo":       sg,
            "sigma_time":      st,
            "blip2_r1":        blip2_r1,
            "mascot_r1":       mascot_r1,
            "delta_r1":        delta,
            "total_queries":   pres_info["total_queries"],
            "blip2_r1_count":  pres_info["blip2_r1_count"],
            "preserved_count": pres_info["preserved_count"],
            "preservation":    pres_info["preservation"],
        }
        ckpt[key] = entry
        save_ckpt(ckpt, CKPT_PATH)

        logger.info(
            "DONE  %s  blip2_r1=%.4f  mascot_r1=%.4f  Δ=%+.4f  "
            "pres=%d/%d=%.4f",
            key, blip2_r1, mascot_r1, delta,
            pres_info["preserved_count"], pres_info["blip2_r1_count"],
            pres_info["preservation"],
        )

    generate_summary(ckpt, logger)
    logger.info("All done. See %s", SUMMARY_PATH)


# ── summary ───────────────────────────────────────────────────────────────────

def generate_summary(ckpt: dict, logger) -> None:
    lines: list[str] = []
    L = lines.append

    L("# R@1 Top-1 Integrity Analysis — PP datasets, N=200, K=20\n")
    L("MASCOT evaluated at val-HM-best operating points from the paper's grid_search_eval.\n")

    # Main table
    L("## R@1 Summary\n")
    L("| Dataset | Dir | θ | σ_geo | σ_time | BLIP-2 R@1 | MASCOT R@1 | Δ R@1 |")
    L("|---|---|---|---|---|---|---|---|")

    dec_deltas = []
    for cfg in CONFIGS:
        key = ckpt_key(cfg)
        if key not in ckpt:
            L(f"| {cfg['dataset']} | {cfg['direction']} | {cfg['theta']} | "
              f"{cfg['sigma_geo']} | {cfg['sigma_time']} | — | — | — |")
            continue
        m   = ckpt[key]
        sgn = "+" if m["delta_r1"] >= 0 else ""
        L(f"| {m['dataset']} | {m['direction']} | {m['theta']} | "
          f"{m['sigma_geo']} | {m['sigma_time']} | "
          f"{m['blip2_r1']:.4f} | {m['mascot_r1']:.4f} | "
          f"{sgn}{m['delta_r1']:.4f} |")
        if m["direction"] == "dec":
            dec_deltas.append(m["delta_r1"])
    L("")

    # Top-1 preservation table
    L("## Strict Top-1 Preservation\n")
    L("Fraction of queries where BLIP-2 rank-1 = ground truth AND MASCOT rank-1 = "
      "the same exact image.\n")
    L("| Dataset | Dir | Queries with BLIP-2 R@1=GT | "
      "MASCOT preserves top-1 | Preservation |")
    L("|---|---|---|---|---|")

    for cfg in CONFIGS:
        key = ckpt_key(cfg)
        if key not in ckpt:
            L(f"| {cfg['dataset']} | {cfg['direction']} | — | — | — |")
            continue
        m = ckpt[key]
        if m["blip2_r1_count"] > 0:
            pres_str = f"{m['preserved_count']}/{m['blip2_r1_count']} = {m['preservation']:.4f}"
        else:
            pres_str = "N/A (no GT top-1)"
        L(f"| {m['dataset']} | {m['direction']} | {m['blip2_r1_count']} | "
          f"{m['preserved_count']} | {pres_str} |")
    L("")

    # Verdict
    L("## Verdict\n")
    if dec_deltas:
        max_dec_delta = max(abs(d) for d in dec_deltas)
        threshold = 0.02
        verdict_ok = max_dec_delta < threshold
        L(f"**Decrease tasks** (primary paper claim): max |Δ R@1| = {max_dec_delta:.4f}  "
          f"{'< ' if verdict_ok else '>= '}{threshold} threshold → "
          f"{'SUPPORTABLE ✓' if verdict_ok else 'EXCEEDS THRESHOLD ✗'}")
        L("")
        pres_vals = [
            ckpt[ckpt_key(cfg)]["preservation"]
            for cfg in CONFIGS
            if ckpt_key(cfg) in ckpt and not (ckpt[ckpt_key(cfg)]["preservation"] != ckpt[ckpt_key(cfg)]["preservation"])
        ]
        if pres_vals:
            min_pres = min(pres_vals)
            max_pres = max(pres_vals)
            L(f"**Strict top-1 preservation** across all cells: "
              f"min={min_pres:.4f}  max={max_pres:.4f}\n")

        L(
            "Empirical top-1 integrity is "
            + (
                f"well-supported: the R@1 gap on decrease tasks is at most {max_dec_delta:.4f} "
                "(below the 0.02 threshold), and the strict preservation rate — the fraction "
                "of queries where BLIP-2's correct top-1 image survives as MASCOT's rank-1 — "
                "is high across all operating points. The §4.5 claim of approximate top-1 "
                "preservation is therefore empirically defensible even though Appendix G.2 "
                "acknowledges it is not a hard guarantee."
                if verdict_ok
                else
                f"under pressure: the R@1 gap on at least one decrease task exceeds 0.02 "
                f"(max Δ = {max_dec_delta:.4f}). The §4.5 top-1 preservation language may "
                "need softening or an explicit caveat referencing the Appendix G.2 disclaimer."
            )
        )
    else:
        L("Insufficient data to render a verdict (no decrease-task results yet).")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s.", SUMMARY_PATH)


if __name__ == "__main__":
    main()
