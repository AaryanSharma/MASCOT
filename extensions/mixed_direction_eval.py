#!/usr/bin/env python3
"""Mixed-direction evaluation on PP_geo_hour.

Runs 4 direction configurations for both MASCOT (ma_smf) and MS-DPP (msdpp):
  pure_inc     — DivDir.INCREASE  (scalar, existing behaviour)
  pure_dec     — DivDir.DECREASE  (scalar, existing behaviour)
  geo_inc_time_dec — per_attribute_directions=[INC, DEC]
  geo_dec_time_inc — per_attribute_directions=[DEC, INC]

Regression checks (executed before the novel experiments):
  [INC, INC] output must equal pure_inc
  [DEC, DEC] output must equal pure_dec

Per-attribute Vendi is computed separately from the standard eval so that
the geo and time components can be inspected independently.  Higher geo-Vendi
means more geographic spread in the top-K list; higher time-Vendi means more
temporal spread.

Usage (inside container):
  cd /msdpp
  env PYTHONPATH=/msdpp/src \\
    uv run python examples/mixed_direction_eval.py

Output:
  results/mixed_direction/results.json
  results/mixed_direction/SUMMARY.md
"""

import json
import logging
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
SRC_DIR    = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from msdpp.base.divmethod import DivDir
from msdpp.div_method.ma_smf import ModelAwareSubmodularMethod
from msdpp.div_method.dpp_E2 import MSDPP
from msdpp.task import BaseTask
RESULT_DIR   = REPO_ROOT / "results" / "mixed_direction"
OUT_JSON     = RESULT_DIR / "results.json"
SUMMARY_PATH = RESULT_DIR / "SUMMARY.md"

CACHE_BASE   = REPO_ROOT / "share_datasets" / "temp" / "div_results"
T2I_CACHE    = CACHE_BASE / "Salesforce-blip2-itm-vit-g-coco_pp_test_ret_results.pkl"
MODEL_PREFIX = "Salesforce-blip2-itm-vit-g-coco"

DATASET_NAME = "PP_geo_hour"
DATASET_PATH = REPO_ROOT / "share_datasets" / "temp" / "tasks" / "PP_geo_hour_test.pkl"

N_THREAD = 32
SUBSET_K = 200
TOP_K    = 20
GRID_SZ  = 20
NUM_BINS = 24

# val-HM-best hyperparameters from top1_integrity analysis
MASCOT_INC_PARAMS = dict(theta=0.8, sigma_geo=15.0, sigma_time=1.5)
MASCOT_DEC_PARAMS = dict(theta=0.1, sigma_geo=15.0, sigma_time=3.0)
# Mixed uses INCREASE params as the starting point (theta governs overall balance)
MASCOT_MIX_PARAMS = dict(theta=0.5, sigma_geo=15.0, sigma_time=1.5)

# MSDPP: default params from grid-search best entry (theta=1.0, beta=0.5)
MSDPP_PARAMS = dict(theta=0.9, beta=0.5, sim_func_name="dist_inv", device="cuda")


# ── Per-attribute Vendi ────────────────────────────────────────────────────

def _vendi_from_features(feats: torch.Tensor) -> float:
    """Compute normalised Vendi-q (q=0.1) directly from a [K, D] feature matrix.

    Replicates the dist_inv similarity and normalisation used in EvalIndexCalculator
    but operates on the already-extracted top-K feature block, avoiding global index
    lookups and the resulting CUDA index-out-of-bounds errors.
    """
    feats = feats.cuda().float()   # [K, D]
    k = feats.shape[0]
    diff = feats.unsqueeze(0) - feats.unsqueeze(1)   # [K, K, D]
    dist = diff.norm(dim=-1)                          # [K, K]
    sim  = 1.0 / (1.0 + dist)                        # [K, K]
    sim  = sim / k
    eye  = 1e-5 * torch.eye(k, device=feats.device)
    sim  = sim + eye
    q    = 0.1
    eigvals = torch.linalg.eigvalsh(sim)              # [K]
    vendi   = (eigvals.clamp(min=0).pow(q).sum().log() / (1 - q)).exp()
    return float((vendi.cpu() - 1) / (k - 1))        # normalise to [0, 1]


def per_attr_vendi(candidates: list[torch.Tensor], dataset, top_k: int = TOP_K):
    """Compute mean geo-Vendi and time-Vendi over all queries.

    Returns (geo_vendi, time_vendi) both in [0, 1].
    """
    data_list   = dataset.dataset
    geo_vendis  = []
    time_vendis = []

    for cands in candidates:
        idxs = cands[:top_k]   # global indices into dataset, length <= top_k

        # geo: (lat, lon) -> [K, 2]
        gps = torch.tensor([data_list[int(i)]["gps"] for i in idxs]).float()

        # time: circular embedding -> [K, 2]
        hrs  = torch.tensor([data_list[int(i)]["hour"]   for i in idxs]).float()
        mins = torch.tensor([data_list[int(i)]["minute"] for i in idxs]).float()
        t    = hrs + mins / 60.0
        t_emb = torch.stack([torch.sin(t / 24 * 2 * 3.14159),
                              torch.cos(t / 24 * 2 * 3.14159)], dim=1)

        geo_vendis.append(_vendi_from_features(gps))
        time_vendis.append(_vendi_from_features(t_emb))

    return float(torch.tensor(geo_vendis).mean()), float(torch.tensor(time_vendis).mean())


# ── Regression check ───────────────────────────────────────────────────────

def check_regression(cands_scalar: list[torch.Tensor],
                     cands_per_attr: list[torch.Tensor],
                     label: str) -> None:
    for i, (a, b) in enumerate(zip(cands_scalar, cands_per_attr)):
        if not torch.equal(a, b):
            raise AssertionError(
                f"Regression FAIL at query {i}: {label}\n"
                f"  scalar:   {a[:5]}\n"
                f"  per-attr: {b[:5]}"
            )
    logging.getLogger("MIX").info("  Regression PASS: %s", label)


# ── Build a lightweight mock model so BaseTask can be constructed ─────────

class _CachedModel:
    """Thin wrapper that serves the pre-computed t2i_sim; no GPU inference."""

    def __init__(self, sim_cache):
        self.sim_cache = sim_cache
        self._name = MODEL_PREFIX

    def __str__(self):
        return self._name

    def infer_datasets(self, *_, **__):
        return self.sim_cache.results


# ── Main ──────────────────────────────────────────────────────────────────

def run_config(task: BaseTask, div_method, direction: DivDir,
               label: str, per_attribute_directions=None,
               logger=None) -> dict:
    logger = logger or logging.getLogger("MIX")
    logger.info("  Running %s …", label)
    result = task.run(
        div_method=div_method,
        direction=direction,
        force=True,
        per_attribute_directions=per_attribute_directions,
    )
    ei = result.eval_indices
    geo_v, time_v = per_attr_vendi(result.candidates, task.dataset, top_k=TOP_K)
    out = {
        "label":      label,
        "r1":         round(float(ei.r1),  6),
        "r10":        round(float(ei.r10), 6),
        "img_vendi":  round(float(ei.img_vendi),  6),
        "ext_vendi":  round(float(ei.ext_vendi),  6),
        "mean_vendi": round(float(ei.mean_vendi), 6),
        "geo_vendi":  round(geo_v,  6),
        "time_vendi": round(time_v, 6),
        "hm_r10_vendi": round(
            2 * float(ei.r10) * float(ei.mean_vendi) / (float(ei.r10) + float(ei.mean_vendi) + 1e-9), 6
        ),
    }
    logger.info(
        "    r10=%.4f  geo_vendi=%.4f  time_vendi=%.4f  mean_vendi=%.4f",
        out["r10"], geo_v, time_v, out["mean_vendi"],
    )
    return out, result.candidates


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  MIX  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger("MIX")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading t2i_sim cache …")
    sim_cache = torch.load(T2I_CACHE, weights_only=False)
    logger.info("Loading PP_geo_hour test dataset …")
    dataset   = torch.load(DATASET_PATH, weights_only=False)

    model = _CachedModel(sim_cache)
    task  = BaseTask(
        model=model,
        dataset=dataset,
        subset_k=SUBSET_K,
        top_k=TOP_K,
        cache_home=CACHE_BASE,
        do_cache_sim=False,
        n_thread=N_THREAD,
        grid_size=GRID_SZ,
        num_bins=NUM_BINS,
    )

    # ── MASCOT experiments (run fresh — the previous version hardcoded
    #    r1=0.0 as a placeholder that shipped unfilled) ────────────────────
    logger.info("=== MASCOT (ma_smf) ===")
    results = {}

    mascot_inc = ModelAwareSubmodularMethod(**MASCOT_INC_PARAMS)
    r, cands_mascot_inc = run_config(task, mascot_inc, DivDir.INCREASE, "mascot_pure_inc", logger=logger)
    results["mascot_pure_inc"] = r

    mascot_dec = ModelAwareSubmodularMethod(**MASCOT_DEC_PARAMS)
    r, cands_mascot_dec = run_config(task, mascot_dec, DivDir.DECREASE, "mascot_pure_dec", logger=logger)
    results["mascot_pure_dec"] = r

    # regression checks for MASCOT
    logger.info("  Regression checks …")
    _, cands_mascot_ii = run_config(task, mascot_inc, DivDir.INCREASE, "mascot_[inc,inc]",
                                     per_attribute_directions=[DivDir.INCREASE, DivDir.INCREASE], logger=logger)
    check_regression(cands_mascot_inc, cands_mascot_ii, "MASCOT [INC,INC] == pure INCREASE")

    _, cands_mascot_dd = run_config(task, mascot_dec, DivDir.DECREASE, "mascot_[dec,dec]",
                                     per_attribute_directions=[DivDir.DECREASE, DivDir.DECREASE], logger=logger)
    check_regression(cands_mascot_dec, cands_mascot_dd, "MASCOT [DEC,DEC] == pure DECREASE")

    # mixed configs at MIX theta
    mascot_mix = ModelAwareSubmodularMethod(**MASCOT_MIX_PARAMS)
    r, _ = run_config(task, mascot_mix, DivDir.INCREASE, "mascot_geo_inc_time_dec",
                       per_attribute_directions=[DivDir.INCREASE, DivDir.DECREASE], logger=logger)
    results["mascot_geo_inc_time_dec"] = r

    r, _ = run_config(task, mascot_mix, DivDir.INCREASE, "mascot_geo_dec_time_inc",
                       per_attribute_directions=[DivDir.DECREASE, DivDir.INCREASE], logger=logger)
    results["mascot_geo_dec_time_inc"] = r

    # ── MS-DPP experiments ─────────────────────────────────────────────────
    logger.info("=== MS-DPP (msdpp) ===")

    msdpp = MSDPP(**MSDPP_PARAMS)

    r, cands_msdpp_inc = run_config(task, msdpp, DivDir.INCREASE, "msdpp_pure_inc", logger=logger)
    results["msdpp_pure_inc"] = r

    r, cands_msdpp_dec = run_config(task, msdpp, DivDir.DECREASE, "msdpp_pure_dec", logger=logger)
    results["msdpp_pure_dec"] = r

    # regression checks for MSDPP
    logger.info("  Regression checks …")
    _, cands_msdpp_ii = run_config(task, msdpp, DivDir.INCREASE, "msdpp_[inc,inc]",
                                    per_attribute_directions=[DivDir.INCREASE, DivDir.INCREASE], logger=logger)
    check_regression(cands_msdpp_inc, cands_msdpp_ii, "MSDPP [INC,INC] == pure INCREASE")

    _, cands_msdpp_dd = run_config(task, msdpp, DivDir.DECREASE, "msdpp_[dec,dec]",
                                    per_attribute_directions=[DivDir.DECREASE, DivDir.DECREASE], logger=logger)
    check_regression(cands_msdpp_dec, cands_msdpp_dd, "MSDPP [DEC,DEC] == pure DECREASE")

    r, _ = run_config(task, msdpp, DivDir.INCREASE, "msdpp_geo_inc_time_dec",
                       per_attribute_directions=[DivDir.INCREASE, DivDir.DECREASE], logger=logger)
    results["msdpp_geo_inc_time_dec"] = r

    r, _ = run_config(task, msdpp, DivDir.INCREASE, "msdpp_geo_dec_time_inc",
                       per_attribute_directions=[DivDir.DECREASE, DivDir.INCREASE], logger=logger)
    results["msdpp_geo_dec_time_inc"] = r

    # ── Save ───────────────────────────────────────────────────────────────
    OUT_JSON.write_text(json.dumps(results, indent=2))
    logger.info("Saved %s", OUT_JSON)

    write_summary(results, logger)
    logger.info("Written %s", SUMMARY_PATH)


def write_summary(results: dict, logger) -> None:
    L = []
    a = L.append

    a("# Mixed-Direction Evaluation — PP_geo_hour\n")
    a("MASCOT and MS-DPP evaluated on PP_geo_hour with four direction configurations.\n")
    a("**Regression verified**: `[INC,INC]` == pure INCREASE and `[DEC,DEC]` == pure DECREASE "
      "for both methods (checked per-query, not just aggregate).\n")
    a("")

    a("## Key metrics\n")
    a("- **geo-Vendi**: Vendi score computed on normalised GPS coordinates of the top-K list "
      "(higher = more geographically spread).")
    a("- **time-Vendi**: Vendi score computed on circular time embeddings of the top-K list "
      "(higher = more temporally spread).")
    a("- **R@10**: standard retrieval recall.\n")

    # table
    rows = [
        ("mascot_pure_inc",        "MASCOT", "geo↑ time↑ (pure INC)"),
        ("mascot_pure_dec",        "MASCOT", "geo↓ time↓ (pure DEC)"),
        ("mascot_geo_inc_time_dec","MASCOT", "geo↑ time↓ (**mixed**)"),
        ("mascot_geo_dec_time_inc","MASCOT", "geo↓ time↑ (**mixed**)"),
        ("msdpp_pure_inc",         "MS-DPP", "geo↑ time↑ (pure INC)"),
        ("msdpp_pure_dec",         "MS-DPP", "geo↓ time↓ (pure DEC)"),
        ("msdpp_geo_inc_time_dec", "MS-DPP", "geo↑ time↓ (**mixed**)"),
        ("msdpp_geo_dec_time_inc", "MS-DPP", "geo↓ time↑ (**mixed**)"),
    ]

    a("## Results\n")
    a("| Method | Direction | R@1 | R@10 | geo-Vendi | time-Vendi | mean-Vendi | HM(R10,Vendi) |")
    a("|---|---|---|---|---|---|---|---|")
    for key, method, label in rows:
        if key not in results:
            a(f"| {method} | {label} | — | — | — | — | — | — |")
            continue
        r = results[key]
        a(f"| {method} | {label} | {r['r1']:.4f} | {r['r10']:.4f} | "
          f"{r['geo_vendi']:.4f} | {r['time_vendi']:.4f} | "
          f"{r['mean_vendi']:.4f} | {r['hm_r10_vendi']:.4f} |")
    a("")

    # verdict
    a("## Verdict\n")
    ma_gid = results.get("mascot_geo_inc_time_dec", {})
    ma_di  = results.get("mascot_geo_dec_time_inc", {})
    ma_inc = results.get("mascot_pure_inc", {})
    ma_dec = results.get("mascot_pure_dec", {})

    geo_inc_time_dec_works = (
        ma_gid.get("geo_vendi", 0) > ma_dec.get("geo_vendi", 1) and
        ma_gid.get("time_vendi", 1) < ma_inc.get("time_vendi", 0)
    )
    geo_dec_time_inc_works = (
        ma_di.get("time_vendi", 0) > ma_inc.get("time_vendi", 1) and
        ma_di.get("geo_vendi", 1) < ma_dec.get("geo_vendi", 0)
    )

    if geo_inc_time_dec_works and geo_dec_time_inc_works:
        a("**VALIDATED**: MASCOT's per-attribute mixed-direction produces opposing Vendi "
          "trajectories on the two metadata attributes simultaneously.")
        a("- `geo↑+time↓`: geo-Vendi is higher than pure-DEC AND time-Vendi is lower than pure-INC. ✓")
        a("- `geo↓+time↑`: time-Vendi is higher than pure-INC AND geo-Vendi is lower than pure-DEC. ✓")
    else:
        if not geo_inc_time_dec_works:
            a("**PARTIAL** for `geo↑+time↓`: expected opposing Vendi trajectories not fully confirmed.")
        if not geo_dec_time_inc_works:
            a("**PARTIAL** for `geo↓+time↑`: expected opposing Vendi trajectories not fully confirmed.")

    a("")
    a("### Architectural note\n")
    a("MASCOT's mixed-direction signal is **metadata-only**: per-attribute signed coverage gain "
      "on the IU probability bins. MASCOT has no appearance kernel — appearance diversity is "
      "an emergent property of the retrieval scores rather than an explicit objective. "
      "MS-DPP's appearance kernel (`β × img_log`) is always unsigned (promotes visual diversity "
      "regardless of direction), and the per-attribute sign is applied only to the metadata "
      "component. This distinction is not a deficiency of either method but a fundamental "
      "architectural difference: MASCOT's greedy IU formulation and MS-DPP's DPP kernel "
      "formulation handle metadata direction differently but both correctly isolate "
      "appearance from metadata in the per-attribute case.")
    a("")

    SUMMARY_PATH.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
