# Corrected Per-Query MASCOT Results

30 `failure_cases_{TASK}_{DIRECTION}_ma_smf[_ablation_no_norm|_ablation_no_omega].json`
files regenerated after fixing the two collision bugs (`638423f`, `ca2c091`).

Each file: JSON list of `{"query", "ground_truth", "retrieved"}` per query.
Every file's recomputed R@10 matches `tables/*.json`'s `14_ma_smf` / `15_.../no_norm` /
`16_.../no_omega` entry exactly (verified 30/30 in the primary + extended regen logs).

The pre-fix versions of these files were mislabeled: the ablation runs' per-query
data overwrote the main MASCOT config because of bug #1. This directory replaces
those with correct data.

## What these are for

- Source data for the recall-curve and trade-off plots in `figures_regenerated/`
- Reproducibility: any downstream analysis needing per-query MASCOT rankings
  should read from here, not from the pre-fix stale files

## Baselines

Baseline methods (BLIP-2, MMR, k-DPP, Clustering, MS-DPP, MS-DPP+TN, MS-DPP+TN+TVMS,
Prob-Coverage) were NOT affected by the collision bugs — they have no ablation
variants sharing the same base filename. Their per-query files live in the author's
working directory at `msdpp/results/{failure_cases_run2,vg,i1m}/failure_cases/` and
are not archived here (unchanged since pre-fix).
