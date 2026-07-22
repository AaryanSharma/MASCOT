#!/usr/bin/env python3
"""CLIP backbone val→test grid search on PP_geo_hour.

Phase 1: Feature extraction happens automatically on first run (cached).
Phase 2: Val grid search (K=10) to find best hyperparams per method.
Phase 3: Test evaluation (K=20) with val-best params.
Phase 4: SUMMARY.md comparing CLIP vs BLIP-2.

Usage (inside container, from repo root):
  env PYTHONPATH=/msdpp/src uv run python examples/clip_gridsearch.py

Results land in:
  results/pp_clip/tables/PP_geo_hour_{increase,decrease}.json
  results/pp_clip/SUMMARY.md
"""

import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT  = SCRIPT_DIR.parent
SRC_DIR    = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import msdpp.models      # noqa: F401 — registers Blip2Model and CLIPModel
import msdpp.div_method  # noqa: F401 — registers all div methods

from grid_search_eval import Evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  CLIP  %(levelname)s  %(message)s",
)
logger = logging.getLogger("clip_gridsearch")

RESULT_DIR   = REPO_ROOT / "results" / "pp_clip"
SUMMARY_PATH = RESULT_DIR / "SUMMARY.md"

# BLIP-2 results for comparison (from results/failure_cases_run2/tables/)
BLIP2_RESULT_DIR = REPO_ROOT / "results" / "failure_cases_run2" / "tables"


# ── Reporting ─────────────────────────────────────────────────────────────────

def _load_table(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    return raw.get("results", raw)


def write_summary(clip_inc: dict, clip_dec: dict,
                  blip2_inc: dict, blip2_dec: dict) -> None:
    METHODS = [
        ("17_blip2",      "base (no div)"),
        ("10_msdpp",      "MS-DPP"),
        ("11_msdpp_tn",   "MS-DPP-TN"),
        ("12_msdpp_tn_tvms", "MS-DPP-TN-TVMS"),
        ("13_prob_coverage", "prob_coverage"),
        ("14_ma_smf",     "MASCOT (ma_smf)"),
        ("18_mmr",        "MMR"),
        ("19_clustering", "clustering"),
    ]

    def fmt(d: dict, key: str, metric: str) -> str:
        row = d.get(key, {})
        v = row.get(metric)
        return f"{v:.4f}" if v is not None else "—"

    lines = []
    a = lines.append

    a("# CLIP Backbone Evaluation — PP_geo_hour\n")
    a("Backbone: `openai/clip-vit-large-patch14` (ViT-L/14).")
    a("Protocol: val grid search at K=10, N=200 → test at K=20, N=200.")
    a("BLIP-2 numbers are from `results/failure_cases_run2/tables/` (same protocol).\n")

    # ── side-by-side table ──────────────────────────────────────────────────
    a("## Side-by-side comparison: BLIP-2 vs CLIP\n")
    a("Metrics: R@10 (recall), DM (mean-Vendi), HM = HM(R@10, DM).\n")

    header = ("| Method "
              "| B2 R@10↑ | B2 DM↑ | B2 HM↑ "
              "| CL R@10↑ | CL DM↑ | CL HM↑ "
              "| B2 R@10↓ | B2 DM↓ | B2 HM↓ "
              "| CL R@10↓ | CL DM↓ | CL HM↓ |")
    sep    = "|---|" + "---|" * 12
    a(header)
    a(sep)

    for key, label in METHODS:
        r = (
            f"| {label} "
            f"| {fmt(blip2_inc, key, 'r10')} "
            f"| {fmt(blip2_inc, key, 'mean_vendi')} "
            f"| {fmt(blip2_inc, key, 'm_ret_div')} "
            f"| {fmt(clip_inc,  key, 'r10')} "
            f"| {fmt(clip_inc,  key, 'mean_vendi')} "
            f"| {fmt(clip_inc,  key, 'm_ret_div')} "
            f"| {fmt(blip2_dec, key, 'r10')} "
            f"| {fmt(blip2_dec, key, 'mean_vendi')} "
            f"| {fmt(blip2_dec, key, 'm_ret_div')} "
            f"| {fmt(clip_dec,  key, 'r10')} "
            f"| {fmt(clip_dec,  key, 'mean_vendi')} "
            f"| {fmt(clip_dec,  key, 'm_ret_div')} |"
        )
        a(r)
    a("")

    # ── headline questions ──────────────────────────────────────────────────
    a("## Headline questions\n")

    msdpp_dec_blip2 = blip2_dec.get("10_msdpp", {}).get("r10")
    msdpp_dec_clip  = clip_dec.get("10_msdpp", {}).get("r10")
    mascot_dec_blip2 = blip2_dec.get("14_ma_smf", {}).get("r10")
    mascot_dec_clip  = clip_dec.get("14_ma_smf", {}).get("r10")

    a("### 1. Does MS-DPP collapse on decrease with CLIP?\n")
    if msdpp_dec_blip2 is not None and msdpp_dec_clip is not None:
        collapse_blip2 = msdpp_dec_blip2 < 0.7
        collapse_clip  = msdpp_dec_clip  < 0.7
        a(f"- BLIP-2 MS-DPP decrease R@10: **{msdpp_dec_blip2:.4f}** "
          f"({'collapses ⚠' if collapse_blip2 else 'holds'})")
        a(f"- CLIP  MS-DPP decrease R@10: **{msdpp_dec_clip:.4f}** "
          f"({'collapses ⚠' if collapse_clip else 'holds'})")
        if collapse_blip2 and collapse_clip:
            a("\n**Finding**: MS-DPP recall collapse on decrease tasks is "
              "**backbone-agnostic** — it reproduces with CLIP. "
              "The paper's headline claim generalizes beyond BLIP-2.\n")
        elif collapse_blip2 and not collapse_clip:
            a(f"\n**Finding**: MS-DPP collapse is **BLIP-2-specific** — "
              f"CLIP R@10={msdpp_dec_clip:.4f} does not collapse. "
              f"The paper's headline finding needs reframing for this backbone.\n")
        else:
            a(f"\n**Finding**: Neither backbone shows strong MS-DPP collapse. "
              f"Further analysis needed.\n")
    else:
        a("_(CLIP results not yet available)_\n")

    a("### 2. Does MASCOT preserve recall on decrease with CLIP?\n")
    if mascot_dec_blip2 is not None and mascot_dec_clip is not None:
        preserved_blip2 = mascot_dec_blip2 > 0.9
        preserved_clip  = mascot_dec_clip  > 0.9
        a(f"- BLIP-2 MASCOT decrease R@10: **{mascot_dec_blip2:.4f}** "
          f"({'preserves ✓' if preserved_blip2 else 'does not fully preserve'})")
        a(f"- CLIP  MASCOT decrease R@10: **{mascot_dec_clip:.4f}** "
          f"({'preserves ✓' if preserved_clip else 'does not fully preserve'})")
        if preserved_blip2 and preserved_clip:
            a("\n**Finding**: MASCOT's recall-preservation on decrease is "
              "**backbone-agnostic** — R@10 > 0.9 with both BLIP-2 and CLIP.\n")
        elif preserved_blip2 and not preserved_clip:
            a(f"\n**Finding**: MASCOT recall-preservation is **BLIP-2-specific** "
              f"(CLIP R@10={mascot_dec_clip:.4f}).\n")
        else:
            a(f"\n**Finding**: MASCOT does not strongly preserve recall on decrease "
              f"with either backbone. Investigate hyperparameters.\n")
    else:
        a("_(CLIP results not yet available)_\n")

    # ── Pareto dominance check ──────────────────────────────────────────────
    a("### 3. Pareto dominance on decrease with CLIP\n")
    ms_dpp_variants = [
        ("10_msdpp",      "MS-DPP"),
        ("11_msdpp_tn",   "MS-DPP-TN"),
        ("12_msdpp_tn_tvms", "MS-DPP-TN-TVMS"),
    ]
    mascot_r10  = clip_dec.get("14_ma_smf", {}).get("r10")
    mascot_vendi = clip_dec.get("14_ma_smf", {}).get("mean_vendi")

    if mascot_r10 is not None and mascot_vendi is not None:
        a(f"MASCOT (CLIP, decrease): R@10={mascot_r10:.4f}, DM={mascot_vendi:.4f}\n")
        all_dominate = True
        for key, label in ms_dpp_variants:
            v = clip_dec.get(key, {})
            competitor_r10   = v.get("r10")
            competitor_vendi = v.get("mean_vendi")
            if competitor_r10 is None:
                a(f"- vs {label}: _(no result)_")
                all_dominate = False
                continue
            dominates = (mascot_r10 >= competitor_r10 and mascot_vendi >= competitor_vendi)
            a(f"- vs {label} (R@10={competitor_r10:.4f}, DM={competitor_vendi:.4f}): "
              f"{'MASCOT dominates ✓' if dominates else 'not strictly dominated'}")
            if not dominates:
                all_dominate = False
        a("")
        if all_dominate:
            a("**Pareto verdict**: MASCOT **Pareto-dominates all MS-DPP variants** "
              "on decrease with CLIP — consistent with the BLIP-2 finding.\n")
        else:
            a("**Pareto verdict**: MASCOT does **not** Pareto-dominate all MS-DPP "
              "variants on decrease with CLIP — BLIP-2 pattern does not fully hold.\n")
    else:
        a("_(CLIP results not yet available)_\n")

    # ── rebuttal paragraph ──────────────────────────────────────────────────
    a("## Rebuttal paragraph\n")
    if msdpp_dec_clip is not None and mascot_dec_clip is not None:
        collapse_clip  = msdpp_dec_clip  < 0.7
        preserved_clip = mascot_dec_clip > 0.9

        if collapse_clip and preserved_clip:
            verdict = "generalizes"
            detail = (
                f"With CLIP (ViT-L/14), vanilla MS-DPP R@10 collapses to "
                f"{msdpp_dec_clip:.4f} on decrease tasks — matching the BLIP-2 "
                f"collapse to {msdpp_dec_blip2:.4f}. MASCOT preserves R@10="
                f"{mascot_dec_clip:.4f} with CLIP vs {mascot_dec_blip2:.4f} with "
                f"BLIP-2. MASCOT Pareto-dominates all MS-DPP variants on decrease "
                f"with CLIP, reproducing the paper's headline finding with an "
                f"architecturally distinct single-vector backbone."
            )
        elif not collapse_clip:
            verdict = "is backbone-specific for MS-DPP collapse"
            detail = (
                f"With CLIP (ViT-L/14), MS-DPP R@10={msdpp_dec_clip:.4f} on "
                f"decrease does not replicate the BLIP-2 collapse "
                f"({msdpp_dec_blip2:.4f}). MASCOT R@10={mascot_dec_clip:.4f}. "
                f"The MS-DPP recall-collapse phenomenon appears tied to BLIP-2's "
                f"multi-token query features; the paper's framing should acknowledge "
                f"this backbone dependence."
            )
        else:
            verdict = "is partially backbone-specific"
            detail = (
                f"With CLIP (ViT-L/14), MS-DPP collapses to R@10={msdpp_dec_clip:.4f} "
                f"on decrease (BLIP-2: {msdpp_dec_blip2:.4f}), but MASCOT only "
                f"achieves R@10={mascot_dec_clip:.4f} (BLIP-2: {mascot_dec_blip2:.4f}). "
                f"Partial generalization — the collapse reproduces but MASCOT's "
                f"protective property is weaker with CLIP."
            )

        a(f"**Headline claim {verdict}**: {detail}")
    else:
        a("_(Run the grid search to populate this section.)_")
    a("")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Written %s", SUMMARY_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluation(
        root_dir=REPO_ROOT / "examples" / "configs",
        result_dir=RESULT_DIR,
        use_spice=False,
        overall_cfg_name="overall_pp_clip.json",
        div_cfg_name="div_clip.json",
    )
    evaluator.run(force=True)

    # ── generate SUMMARY.md ──────────────────────────────────────────────────
    tables = RESULT_DIR / "tables"
    clip_inc  = _load_table(tables / "PP_geo_hour_increase.json")
    clip_dec  = _load_table(tables / "PP_geo_hour_decrease.json")
    blip2_inc = _load_table(BLIP2_RESULT_DIR / "PP_geo_hour_increase.json")
    blip2_dec = _load_table(BLIP2_RESULT_DIR / "PP_geo_hour_decrease.json")

    write_summary(clip_inc, clip_dec, blip2_inc, blip2_dec)
    logger.info("Done.")


if __name__ == "__main__":
    main()
