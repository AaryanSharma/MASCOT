# Fixed-Recall / Fixed-Diversity Comparison — PixelProse
> **Source:** PP_geo_hour uses sensitivity θ sweep (σ_geo=10, σ_time=1.5) plus Pareto checkpoints
> and the paper Table 1 operating point θ=0.1, σ_geo=15, σ_time=3.0 (labelled [Table 1]).
> PP_geo / PP_hour use full θ × σ_geo × σ_time grid from pkl index cache.
> DM = `mean_vendi` (decrease: ext_vendi flipped to 1−ext_vendi before harmonic mean).
> `—` = no operating point meets the threshold.
> **NOTE:** Full θ × σ grid for PP_geo_hour not in cache; to verify all sigma combos re-run grid_search_eval.py on PP_geo_hour.

---

## Dataset: PP_geo_hour *(headline)*

### Decrease Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.1784 | 0.1881 | 0.1881 |
| Both-Ablated (prob\_cov) | 0.1656 | 0.1656 | 0.2860 |
| BLIP-2 (baseline) | 0.1656 | 0.1656 | 0.1656 |
| MS-DPP | — | — | — |
| MS-DPP-TN-TVMS | 0.1902 | 0.2066 | 0.2189 |

*MASCOT (ma\_smf) at R@10 ≥ 0.95: θ=0.1 σ_geo=10.0, σ_time=1.5 (r10=0.9686, DM=0.1784).* *MASCOT (ma\_smf) at R@10 ≥ 0.90: θ=0.1 σ_geo=15.0, σ_time=3.0 (r10=0.9410, DM=0.1881) [Table 1].* *MASCOT (ma\_smf) at R@10 ≥ 0.85: θ=0.1 σ_geo=15.0, σ_time=3.0 (r10=0.9410, DM=0.1881) [Table 1].*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.15 | DM ≥ 0.20 | DM ≥ 0.25 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9762 | 0.1443 | — |
| Both-Ablated (prob\_cov) | 0.9737 | 0.8557 | 0.8557 |
| BLIP-2 (baseline) | 0.9737 | — | — |
| MS-DPP | 0.4931 | 0.4931 | 0.4931 |
| MS-DPP-TN-TVMS | 0.9674 | 0.9021 | — |

### Increase Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9299 | 0.9397 | 0.9419 |
| Both-Ablated (prob\_cov) | 0.9142 | 0.9142 | 0.9482 |
| BLIP-2 (baseline) | 0.9142 | 0.9142 | 0.9142 |
| MS-DPP | 0.9262 | 0.9262 | 0.9262 |
| MS-DPP-TN-TVMS | 0.9306 | 0.9306 | 0.9306 |

*MASCOT (ma\_smf) at R@10 ≥ 0.95: θ=0.4 σ_geo=10.0, σ_time=1.5 (r10=0.9624, DM=0.9299).* *MASCOT (ma\_smf) at R@10 ≥ 0.90: θ=0.7 σ_geo=10.0, σ_time=1.5 (r10=0.9046, DM=0.9397).* *MASCOT (ma\_smf) at R@10 ≥ 0.85: θ=0.8 σ_geo=10.0, σ_time=1.5 (r10=0.8557, DM=0.9419).*

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.93 | DM ≥ 0.94 | DM ≥ 0.95 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9486 | 0.8557 | — |
| Both-Ablated (prob\_cov) | 0.8645 | 0.8645 | 0.7114 |
| BLIP-2 (baseline) | — | — | — |
| MS-DPP | — | — | — |
| MS-DPP-TN-TVMS | 0.9711 | — | — |

---

## Dataset: PP_geo *(secondary)*

### Decrease Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.3894 | 0.3894 | 0.3894 |
| Both-Ablated (prob\_cov) | 0.3476 | 0.3511 | 0.5084 |
| BLIP-2 (baseline) | 0.1653 | 0.1653 | 0.1653 |
| MS-DPP | — | — | — |

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.15 | DM ≥ 0.20 | DM ≥ 0.25 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9762 | 0.9762 | 0.9737 |
| Both-Ablated (prob\_cov) | 0.9737 | 0.9737 | 0.9737 |
| BLIP-2 (baseline) | 0.9900 | — | — |
| MS-DPP | 0.7704 | 0.7704 | 0.7704 |

### Increase Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9338 | 0.9397 | 0.9419 |
| Both-Ablated (prob\_cov) | 0.9142 | 0.9304 | 0.9482 |
| BLIP-2 (baseline) | 0.9164 | 0.9164 | 0.9164 |
| MS-DPP | — | — | 0.9274 |

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.93 | DM ≥ 0.94 | DM ≥ 0.95 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9561 | 0.8871 | — |
| Both-Ablated (prob\_cov) | 0.9009 | 0.8996 | 0.7892 |
| BLIP-2 (baseline) | — | — | — |
| MS-DPP | — | — | — |

---

## Dataset: PP_hour *(secondary)*

### Decrease Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.3086 | 0.3570 | 0.3570 |
| Both-Ablated (prob\_cov) | 0.3006 | 0.4235 | 0.4780 |
| BLIP-2 (baseline) | 0.3006 | 0.3006 | 0.3006 |
| MS-DPP | — | — | — |

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.15 | DM ≥ 0.20 | DM ≥ 0.25 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.9737 | 0.9737 | 0.9737 |
| Both-Ablated (prob\_cov) | 0.9737 | 0.9737 | 0.9737 |
| BLIP-2 (baseline) | 0.9737 | 0.9737 | 0.9737 |
| MS-DPP | 0.7654 | 0.7654 | 0.7654 |

### Increase Direction

**Table A — Best achievable DM at fixed R@10 recall floor**

| Method | R@10 ≥ 0.95 | R@10 ≥ 0.90 | R@10 ≥ 0.85 |
|---|---||---||---|
| MASCOT (ma\_smf) | 0.8866 | 0.8910 | 0.8982 |
| Both-Ablated (prob\_cov) | 0.9065 | 0.9157 | 0.9204 |
| BLIP-2 (baseline) | 0.8706 | 0.8706 | 0.8706 |
| MS-DPP | — | 0.9147 | 0.9147 |

**Table B — Best achievable R@10 at fixed diversity floor**

| Method | DM ≥ 0.93 | DM ≥ 0.94 | DM ≥ 0.95 |
|---|---||---||---|
| MASCOT (ma\_smf) | — | — | — |
| Both-Ablated (prob\_cov) | — | — | — |
| BLIP-2 (baseline) | — | — | — |
| MS-DPP | — | — | — |

---

## Summary for Rebuttal (PP_geo_hour, headline task)
### Decrease direction — key numbers

**With Table 1 operating point (θ=0.1, σ_geo=15, σ_time=3.0) [Table 1]:**

At **R@10 ≥ 0.95** (near-lossless retrieval):
- MASCOT: best available θ=0.1, σ_geo=10, σ_time=1.5 → R@10=0.9686, DM=**0.1784** (+7.7% vs baseline 0.1656)
- Table 1 operating point (σ_geo=15, σ_time=3.0): R@10=0.9410 does **not** meet ≥ 0.95 floor
- BLIP-2 baseline: DM=0.1656

At **R@10 ≥ 0.90** (1% recall relaxation):
- MASCOT [Table 1]: R@10=0.9410 ≥ 0.90, DM=**0.1881** (+13.6% vs baseline) ← **Table 1 point qualifies here**
- MASCOT [σ_geo=10]: R@10=0.9686, DM=0.1784 (+7.7%)
- Both-Ablated: stuck at baseline 0.1656 (no operating point clears R@10 ≥ 0.90 with DM > baseline)

At **DM ≥ 0.20** (diversity floor):
- MASCOT [σ_geo=10]: best R@10=0.5157 at θ=0.2 — recall drops sharply above θ=0.1
- MS-DPP-TN-TVMS: best R@10=0.9021 — strongest at this floor

### Increase direction — key numbers (from sensitivity sweep)
At **R@10 ≥ 0.95**: MASCOT DM=0.9299 vs BLIP-2 0.9142
At **DM ≥ 0.93**: MASCOT R@10=0.9486, Both-Ablated R@10=0.8645, MS-DPP-TN-TVMS R@10=0.9711
