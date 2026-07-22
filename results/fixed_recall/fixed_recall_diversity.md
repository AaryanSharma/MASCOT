# Fixed-Recall / Fixed-Diversity Comparison — PixelProse

> **Source:**
> - **PP_geo_hour**: sensitivity sweeps (`results/sensitivity/PP_geo_hour/`) + pareto checkpoints
>   (`results/pareto/checkpoint*.json`), all at σ_geo=10, σ_time=1.5; plus Table 1 val-best point
>   (θ=0.10, σ_geo=15, σ_time=3.0) manually included. MS-DPP-TN-TVMS from pareto frontier.
> - **PP_geo / PP_hour**: full θ × σ_geo × σ_time grid from pkl cache
>   (`index_Salesforce-blip2-itm-vit-g-coco_pp_{geo,hour}_test_*`), 55/29 ma_smf and 54/28
>   prob_coverage operating points per direction.
>
> All values on PixelProse test split (BLIP-2). DM = `mean_vendi` (direction-aware: for decrease,
> `ext_vendi` flipped to `1 − ext_vendi` before harmonic mean). `—` = threshold not achievable.

---

## Dataset: PP_geo_hour *(headline)*

### Decrease Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.1784 | 0.1881 | 0.1881 |
| Both-Ablated (prob\_cov) | 0.1656 | 0.1656 | 0.2860 |
| BLIP-2 (baseline) | 0.1656 | 0.1656 | 0.1656 |
| MS-DPP | 0.3158 | 0.3158 | 0.3158 |
| MS-DPP-TN-TVMS | 0.1902 | 0.2066 | 0.2189 |

*MASCOT at R@10 ≥ 0.95: θ=0.10, σ_geo=10, σ_time=1.5 (r10=0.9686, DM=0.1784).
At R@10 ≥ 0.90: **θ=0.10, σ_geo=15, σ_time=3.0 [Table 1]** (r10=0.9410, DM=0.1881) — +13.6% vs baseline.
(Sensitivity-sweep best at same floor: θ=0.15, σ_geo=10, σ_time=1.5, DM=0.1872; Table 1 point surpasses it by +0.0009.)
Both-Ablated cannot improve DM above baseline while maintaining R@10 ≥ 0.90 (θ=0.1 drops to r10=0.8557).
MS-DPP: r10=0.4931, DM=0.3158 (single operating point) — cannot sustain R@10 ≥ 0.90; design limitation.
MS-DPP-TN-TVMS at R@10 ≥ 0.85: config 0.4\_0.8 (r10=0.8846, DM=0.2189).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.15 | DM ≥ 0.20 | DM ≥ 0.25 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9762 | 0.1443 | — |
| Both-Ablated (prob\_cov) | 0.9737 | 0.8557 | 0.8557 |
| BLIP-2 (baseline) | 0.9737 | — | — |
| MS-DPP | 0.9737 | 0.4931 | 0.4931 |
| MS-DPP-TN-TVMS | 0.9674 | 0.9021 | — |

*MASCOT at DM ≥ 0.20: θ=0.25 (r10=0.1443) — retrieval collapses before DM=0.20 is reached.
Both-Ablated at DM ≥ 0.20: θ=0.10 (r10=0.8557, DM=0.2860); maintains recall well above 0.85.
MS-DPP at DM ≥ 0.20: r10=0.4931 (single operating point) — recall collapses below Both-Ablated and MS-DPP-TN-TVMS.
MS-DPP-TN-TVMS at DM ≥ 0.20: config 0.5\_0.8 (r10=0.9021, DM=0.2066) — strongest at this floor.*

---

### Increase Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9299 | 0.9397 | 0.9419 |
| Both-Ablated (prob\_cov) | 0.9142 | 0.9142 | 0.9482 |
| BLIP-2 (baseline) | 0.9142 | 0.9142 | 0.9142 |
| MS-DPP | 0.9262 | 0.9262 | 0.9262 |
| MS-DPP-TN-TVMS | 0.9306 | 0.9306 | 0.9306 |

*MASCOT at R@10 ≥ 0.95: θ=0.40 (r10=0.9624, DM=0.9299). At R@10 ≥ 0.90: θ=0.70 (r10=0.9046, DM=0.9397).
Both-Ablated cannot improve DM above baseline while maintaining R@10 ≥ 0.90 (θ=0.1 drops to r10=0.8645).
MS-DPP: r10=0.9724, DM=0.9262 (single operating point) — maintains high recall at +1.3% DM above baseline.
MS-DPP-TN-TVMS: all configs maintain R@10 ≥ 0.96; best DM = 0.9306 (config 0.4\_0.8).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.93 | DM ≥ 0.94 | DM ≥ 0.95 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9486 | 0.8557 | — |
| Both-Ablated (prob\_cov) | 0.8645 | 0.8645 | 0.7114 |
| BLIP-2 (baseline) | — | — | — |
| MS-DPP-TN-TVMS | 0.9711 | — | — |

*MASCOT at DM ≥ 0.93: θ=0.50 (r10=0.9486, DM=0.9336). At DM ≥ 0.94: θ=0.80 (r10=0.8557, DM=0.9419).
MS-DPP-TN-TVMS reaches DM=0.9306 max; cannot cross 0.94.
BLIP-2 baseline DM=0.9142 — cannot reach any threshold without diversification.*

---

## Dataset: PP_geo *(secondary)*

> **Source for PP_geo/PP_hour:** Full θ × σ_geo × σ_time grid from pkl index cache
> (`share_datasets/temp/div_results/rerank/index_Salesforce-blip2-itm-vit-g-coco_pp_geo_test_*`).
> 55 ma_smf / 54 prob_coverage operating points per direction. Baseline from sensitivity sweep (θ=0).

### Decrease Direction

> Baseline DM for PP_geo decrease = 0.3414 (θ=0). Table B uses tighter thresholds (≥0.35, ≥0.40, ≥0.45)
> since the 0.15/0.20/0.25 floors are all trivially met by the baseline.

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.3894 | 0.3894 | 0.3894 |
| Both-Ablated (prob\_cov) | 0.3476 | 0.3511 | 0.5084 |
| BLIP-2 (baseline) | 0.3414 | 0.3414 | 0.3414 |

*MASCOT at R@10 ≥ 0.95: θ=0.20, σ_geo=10, σ_time=0.5 (r10=0.9586, DM=0.3894) — both r10 and DM improve over baseline.
Both-Ablated at R@10 ≥ 0.95: θ=0.30, σ_geo=1, σ_time=0.5 (r10=0.9548, DM=0.3476) — now improves above baseline (was stuck at 0.3414 in sensitivity sweep).
At R@10 ≥ 0.90: θ=0.60, σ_geo=1, σ_time=0.5 (r10=0.9172, DM=0.3511). At R@10 ≥ 0.85: θ=0.20, σ_geo=10, σ_time=0.5 (r10=0.8670, DM=0.5084).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.35 | DM ≥ 0.40 | DM ≥ 0.45 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9737 | 0.8105 | — |
| Both-Ablated (prob\_cov) | 0.9172 | 0.8770 | 0.8770 |
| BLIP-2 (baseline) | — | — | — |

*MASCOT at DM ≥ 0.35: θ=0.10, σ_geo=10, σ_time=0.5 (r10=0.9737, DM=0.3614) — maintains full baseline recall.
At DM ≥ 0.40: θ=0.30, σ_geo=10, σ_time=0.5 (r10=0.8105, DM=0.4136) — recall cost.
Both-Ablated at DM ≥ 0.40: θ=0.10, σ_geo=10, σ_time=0.5 (r10=0.8770, DM=0.4997) — better recall than MASCOT at this floor.*

---

### Increase Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9338 | 0.9397 | 0.9419 |
| Both-Ablated (prob\_cov) | 0.9142 | 0.9304 | 0.9482 |
| BLIP-2 (baseline) | 0.8435 | 0.8435 | 0.8435 |

*MASCOT at R@10 ≥ 0.95: θ=0.50, σ_geo=10, σ_time=1 (r10=0.9561, DM=0.9338) — substantially higher than sensitivity-only estimate of 0.8888.
At R@10 ≥ 0.90: θ=0.70, σ_geo=10, σ_time=1.5 (r10=0.9046, DM=0.9397). At R@10 ≥ 0.85: θ=0.80, σ_geo=10, σ_time=1.5 (r10=0.8557, DM=0.9419).
Both-Ablated at R@10 ≥ 0.95: stuck at θ=0 baseline (DM=0.9142). At R@10 ≥ 0.90: θ=0.10, σ_geo=10, σ_time=0.5 (r10=0.9009, DM=0.9304).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.93 | DM ≥ 0.94 | DM ≥ 0.95 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9561 | 0.8871 | — |
| Both-Ablated (prob\_cov) | 0.9009 | 0.8996 | 0.7892 |
| BLIP-2 (baseline) | — | — | — |

*MASCOT at DM ≥ 0.93: θ=0.50, σ_geo=10, σ_time=1 (r10=0.9561) — full grid reveals this operating point (missed by sensitivity sweep).
At DM ≥ 0.94: θ=0.50, σ_geo=10, σ_time=1.5 (r10=0.8871).
Both-Ablated max DM = 0.9502; can reach DM ≥ 0.95 but recall collapses (r10=0.7892).*

---

## Dataset: PP_hour *(secondary)*

### Decrease Direction

> Baseline DM for PP_hour decrease = 0.3006. Table B uses tighter thresholds (≥0.35, ≥0.40, ≥0.45)
> since the 0.15/0.20/0.25 floors are trivially met by the baseline.

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.3086 | 0.3570 | 0.3570 |
| Both-Ablated (prob\_cov) | 0.3006 | 0.4235 | 0.4780 |
| BLIP-2 (baseline) | 0.3006 | 0.3006 | 0.3006 |

*MASCOT at R@10 ≥ 0.95: θ=0.10, σ_geo=15, σ_time=1.5 (r10=0.9724, DM=0.3086).
At R@10 ≥ 0.90: θ=0.40, σ_geo=1, σ_time=0.5 (r10=0.9059, DM=0.3570) — full grid reveals σ_geo=1 operating point (sensitivity sweep missed it, giving only 0.3248).
Both-Ablated at R@10 ≥ 0.90: θ=0.10, σ_geo=1, σ_time=0.5 (r10=0.9021, DM=0.4235) — was stuck at baseline 0.3006 in sensitivity sweep.
At R@10 ≥ 0.85: θ=0.40, σ_geo=1, σ_time=1.5 (r10=0.8519, DM=0.4780).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.35 | DM ≥ 0.40 | DM ≥ 0.45 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9059 | — | — |
| Both-Ablated (prob\_cov) | 0.9021 | 0.9021 | — |
| BLIP-2 (baseline) | — | — | — |

*MASCOT at DM ≥ 0.35: θ=0.40, σ_geo=1, σ_time=0.5 (r10=0.9059, DM=0.3570) — retrieval still viable.
Both-Ablated at DM ≥ 0.40: same operating point (r10=0.9021, DM=0.4235). MASCOT cannot reach DM=0.40.*

---

### Increase Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.8866 | 0.8910 | 0.8982 |
| Both-Ablated (prob\_cov) | 0.9065 | 0.9157 | 0.9204 |
| BLIP-2 (baseline) | 0.8706 | 0.8706 | 0.8706 |

*PP_hour increase: Both-Ablated outperforms MASCOT at every recall floor — consistent with the time-only setting where prob_coverage's uniform bin weighting aligns well with the 24-bin IU matrix.
Both-Ablated at R@10 ≥ 0.95: θ=0.10, σ_geo=1, σ_time=0.5 (r10=0.9511, DM=0.9065) — significantly higher than sensitivity-only estimate of 0.8886.
At R@10 ≥ 0.90: θ=0.20, σ_geo=1, σ_time=0.5 (r10=0.9084, DM=0.9157). At R@10 ≥ 0.85: θ=0.50, σ_geo=1, σ_time=0.5 (r10=0.8645, DM=0.9204).
MASCOT at R@10 ≥ 0.95: θ=0.70, σ_geo=15, σ_time=1.5 (r10=0.9523, DM=0.8866).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.88 | DM ≥ 0.90 | DM ≥ 0.92 |
|---|---|---|---|
| MASCOT (ma\_smf) | 0.9699 | 0.1330 | — |
| Both-Ablated (prob\_cov) | 0.9598 | 0.9511 | 0.8645 |
| BLIP-2 (baseline) | — | — | — |

*MASCOT at DM ≥ 0.88: θ=0.50, σ_geo=1, σ_time=0.5 (r10=0.9699, DM=0.8855) — strong recall at moderate diversity.
At DM ≥ 0.90: retrieval collapses to r10=0.1330 (θ=1.0). MASCOT max DM=0.9148 is achievable only at extreme θ.
Both-Ablated at DM ≥ 0.90: θ=0.10, σ_geo=1, σ_time=0.5 (r10=0.9511) — maintains useful recall where MASCOT fails.
Both-Ablated max DM = 0.9221.*

---

## Summary for Rebuttal (PP_geo_hour, headline task)

### Decrease direction — key numbers for reviewer nxXq

At **R@10 ≥ 0.95** (near-lossless retrieval):
- MASCOT achieves DM = **0.1784** vs BLIP-2 baseline 0.1656 (+7.7% relative)
- MS-DPP-TN-TVMS achieves DM = **0.1902** (+14.7% relative vs baseline)
- Both-Ablated stuck at baseline DM = 0.1656 (no diversity gain at this recall floor)

At **R@10 ≥ 0.90** (1% recall relaxation):
- MASCOT [Table 1]: DM = **0.1881** (+13.6%) — θ=0.10, σ_geo=15, σ_time=3.0
- MS-DPP-TN-TVMS: DM = **0.2066** (+24.7%)
- Both-Ablated: still stuck at 0.1656

At **DM ≥ 0.20** (diversity floor):
- MASCOT: best R@10 = **0.1443** — retrieval collapses
- Both-Ablated: best R@10 = **0.8557** — maintains useful recall
- MS-DPP-TN-TVMS: best R@10 = **0.9021** — best recall-diversity trade-off at this floor

### Increase direction — key numbers for reviewer nxXq

At **R@10 ≥ 0.95**:
- MASCOT: DM = **0.9299** vs BLIP-2 0.9142 (+1.7%)
- MS-DPP-TN-TVMS: DM = **0.9306** (+1.8%)
- Both-Ablated: stuck at baseline 0.9142

At **DM ≥ 0.93**:
- MASCOT: R@10 = **0.9486** — maintains near-full recall at high diversity
- MS-DPP-TN-TVMS: R@10 = **0.9711** — best at this floor
- Both-Ablated: R@10 = **0.8645** — 8.5 points below MASCOT
- BLIP-2: cannot reach DM = 0.93 at all

---

## Key correction vs previous version

Previous table: MASCOT at R@10 ≥ 0.90 (PP_geo_hour decrease): DM = 0.1872 (θ=0.15, σ_geo=10, σ_time=1.5)

Updated table: DM = **0.1881** (θ=0.10, σ_geo=15, σ_time=3.0 [Table 1])

Same R@10=0.9410, but the paper's val-best operating point achieves higher DM than the sensitivity-sweep best
(σ_geo=10 fixed). The Table 1 selection was correct; DM values are sigma-independent at evaluation time
(ext_vendi uses raw GPS/time features via dist_inv, not the smoothed IU probabilities).
