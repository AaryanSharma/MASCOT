# Rebuttal & Post-Submission Materials

This directory contains materials produced during the **ACM MM 2026 rebuttal period** for MASCOT (Submission #9126).

## Contents

- **`response.md`** — Point-by-point response to reviewers b46S, jKYi, Kr7y, hR87, nxXq
- **`camera_ready_todos.md`** — Concrete revisions promised in the rebuttal, tracking status for camera-ready

## Rebuttal-Cited Experiments (with reproducible code)

All rebuttal claims are backed by code and result artifacts in the main tree:

| Rebuttal claim | Script | Result |
|---|---|---|
| Metadata generality — visual clusters extend MASCOT beyond geo/time | [`extensions/evaluate_visual_clusters.py`](../extensions/evaluate_visual_clusters.py) | [`results/visual_clusters/final_vc_test.log`](../results/visual_clusters/final_vc_test.log) |
| Vectorized implementation — 2.99 ms end-to-end at N=200, K=20 | [`efficiency/benchmark_full_pipeline.py`](../efficiency/benchmark_full_pipeline.py) | [`results/runtime/runtime_full.json`](../results/runtime/runtime_full.json) |
| `search_vec()` and `search()` produce bit-identical outputs (float32) | [`efficiency/test_vec_correctness.py`](../efficiency/test_vec_correctness.py) | (test asserts pass) |
| Fixed-recall analysis — Max DM at R@10≥{0.95, 0.90, 0.85} | [`extensions/build_fixed_recall_tables.py`](../extensions/build_fixed_recall_tables.py) | [`results/fixed_recall/fixed_recall_diversity.md`](../results/fixed_recall/fixed_recall_diversity.md) |

## Post-Submission Extensions (not headline in the paper but shipped for reproducibility)

| Extension | Script | Result |
|---|---|---|
| CLIP backbone — MS-DPP collapse reproduces with CLIP; MASCOT preserves | [`examples/clip_gridsearch.py`](../examples/clip_gridsearch.py) | [`results/pp_clip/SUMMARY.md`](../results/pp_clip/SUMMARY.md) |
| Mixed-direction diversification (per-attribute d) | [`extensions/mixed_direction_eval.py`](../extensions/mixed_direction_eval.py) | [`results/mixed_direction/SUMMARY.md`](../results/mixed_direction/SUMMARY.md) |
| Principled displacement — validates Appendix G.2 top-1 caveat | [`prs/principled_displacement.py`](../prs/principled_displacement.py) | [`results/top1_integrity/SUMMARY.md`](../results/top1_integrity/SUMMARY.md) |
