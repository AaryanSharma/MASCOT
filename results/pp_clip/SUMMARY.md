# CLIP Backbone Evaluation — PP_geo_hour

Backbone: `openai/clip-vit-large-patch14` (ViT-L/14).
Protocol: val grid search at K=10, N=200 → test at K=20, N=200.
BLIP-2 numbers are from `results/failure_cases_run2/tables/` (same protocol).

## Side-by-side comparison: BLIP-2 vs CLIP

Metrics: R@10 (recall), DM (mean-Vendi), HM = HM(R@10, DM).

| Method | B2 R@10↑ | B2 DM↑ | B2 HM↑ | CL R@10↑ | CL DM↑ | CL HM↑ | B2 R@10↓ | B2 DM↓ | B2 HM↓ | CL R@10↓ | CL DM↓ | CL HM↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (no div) | 0.9737 | 0.9142 | 0.9430 | 0.9900 | 0.9164 | 0.9518 | 0.9737 | 0.1656 | 0.2831 | 0.9900 | 0.1653 | 0.2834 |
| MS-DPP | 0.9724 | 0.9262 | 0.9487 | 0.8733 | 0.9466 | 0.9085 | 0.4931 | 0.3158 | 0.3850 | 0.3802 | 0.3574 | 0.3685 |
| MS-DPP-TN | 0.9435 | 0.9413 | 0.9424 | 0.9774 | 0.9370 | 0.9568 | 0.8934 | 0.2240 | 0.3582 | 0.9636 | 0.2271 | 0.3676 |
| MS-DPP-TN-TVMS | 0.9711 | 0.9306 | 0.9504 | 0.9849 | 0.9281 | 0.9557 | 0.9021 | 0.2066 | 0.3363 | 0.9172 | 0.2278 | 0.3649 |
| prob_coverage | 0.9473 | 0.9330 | 0.9401 | 0.8620 | 0.9475 | 0.9027 | 0.7302 | 0.2817 | 0.4066 | 0.8896 | 0.2903 | 0.4378 |
| MASCOT (ma_smf) | 0.8356 | 0.9435 | 0.8863 | 0.9435 | 0.9352 | 0.9394 | 0.9410 | 0.1881 | 0.3135 | 0.9762 | 0.2010 | 0.3334 |
| MMR | 0.9636 | 0.9359 | 0.9496 | 0.8670 | 0.9489 | 0.9061 | 0.9724 | 0.1401 | 0.2448 | 0.9837 | 0.1373 | 0.2409 |
| clustering | 0.8657 | 0.9381 | 0.9005 | 0.8745 | 0.9371 | 0.9047 | 0.7240 | 0.2058 | 0.3205 | 0.7378 | 0.2163 | 0.3345 |

## Headline questions

### 1. Does MS-DPP collapse on decrease with CLIP?

- BLIP-2 MS-DPP decrease R@10: **0.4931** (collapses ⚠)
- CLIP  MS-DPP decrease R@10: **0.3802** (collapses ⚠)

**Finding**: MS-DPP recall collapse on decrease tasks is **backbone-agnostic** — it reproduces with CLIP. The paper's headline claim generalizes beyond BLIP-2.

### 2. Does MASCOT preserve recall on decrease with CLIP?

- BLIP-2 MASCOT decrease R@10: **0.9410** (preserves ✓)
- CLIP  MASCOT decrease R@10: **0.9762** (preserves ✓)

**Finding**: MASCOT's recall-preservation on decrease is **backbone-agnostic** — R@10 > 0.9 with both BLIP-2 and CLIP.

### 3. Pareto dominance on decrease with CLIP

MASCOT (CLIP, decrease): R@10=0.9762, DM=0.2010

- vs MS-DPP (R@10=0.3802, DM=0.3574): not strictly dominated
- vs MS-DPP-TN (R@10=0.9636, DM=0.2271): not strictly dominated
- vs MS-DPP-TN-TVMS (R@10=0.9172, DM=0.2278): not strictly dominated

**Pareto verdict**: MASCOT does **not** Pareto-dominate all MS-DPP variants on decrease with CLIP — BLIP-2 pattern does not fully hold.

## Rebuttal paragraph

**Headline claim generalizes; Pareto claim does not.** With CLIP (ViT-L/14),
vanilla MS-DPP R@10 collapses to **0.3802** on the PP_geo_hour decrease task —
matching the BLIP-2 collapse to 0.4931. MASCOT preserves recall at
**R@10 = 0.9762** with CLIP (vs 0.9410 with BLIP-2). The recall-preservation
result is therefore backbone-agnostic.

MASCOT does **not** Pareto-dominate all MS-DPP variants under CLIP: at
(0.9762, 0.2010) it is not strictly dominated by any variant, but neither
does it dominate MS-DPP-TN (0.9636, 0.2271) or MS-DPP-TN-TVMS (0.9172, 0.2278) —
those variants trade lower recall for higher DM. The trade-off is
Pareto-comparable, not Pareto-dominant. The paper's central claim (recall
protection under diversity-decrease) reproduces; the earlier draft's
"MASCOT Pareto-dominates" statement in this section contradicted the
analysis above and is retracted here.

