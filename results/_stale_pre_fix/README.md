# Stale Files — Quarantined Pre-Fix

This directory preserves failure_cases files that predate the fixes for bug #1
(commit `638423f`) and bug #2 (commit `ca2c091`).

## Why kept, not deleted

Reviewers and future collaborators may want to inspect the exact inputs that
produced the plots in the original submission. Physical preservation removes
any doubt about what was on disk before the fix, while the corrected files
now live in `results/failure_cases_run2/failure_cases/` (baselines,
unaffected) and `results/figures_regenerated/` (regenerated MASCOT plots).

## What is here

Each `failure_cases_*_ma_smf.json` file in this directory contained MS-DPP
+TN+TVMS-family Uniform Binning data mislabeled as MASCOT, because the
per-query save path used only `div_method` (not `div_method + suffix`).
All three MASCOT-family ablations (main, no_norm, no_omega) collided at the
same filename; whichever ran last wins.

For every stale file here, cross-checking `r10` against
`results/failure_cases_run2/tables/<TASK>.json`:
- Table's `14_ma_smf.r10` ≠ file's fresh r10 (Table says MASCOT, file has UB)
- Table's `16_ma_smf_ablation_no_omega.r10` == file's fresh r10 (confirms UB content)

## What is NOT here

The aggregated `tables/*.json` values are *not* corrupted by this bug: they
were computed from `test_result.eval_indices` on each fresh grid-search run
(`force=True`), not loaded from cache. All numeric values in the paper's
Tables 1, 2, 9, 10, 11 remain trustworthy.

Baseline `failure_cases_*_{blip2,msdpp,mmr,k-DPP,clustering,prob_coverage,...}.json`
were also uncorrupted (no ablation → no collision) and are not archived here.
