# Appendix — Per-Task Hyperparameter Table

Val-best hyperparameters for each (task, direction, method) triple.
Every row rebuilt programmatically from `tables/*.json` after the
bug #1 + #2 fixes. R@10 / MAP / DM cells come directly from the
`14_ma_smf` / `15_ma_smf_ablation_no_norm` / `16_ma_smf_ablation_no_omega`
entries in `results/{failure_cases_run2, vg, i1m}/tables/<TASK>.json`.
Hyperparameters (θ, σ_geo, σ_time) reverse-matched from
`all_metrices.log` / `verified_old.log` / `vg_metrices.log` /
`i1m_metrices*.log` grid-search best-param records: for PP by (R@10, DM),
for VG and I1M by (MAP, DM) since those tasks record MAP as Retrieval Index.

Units: **σ_geo in degrees**, **σ_time in hours**.

The paper uses two distinct geographic representations at two stages —
easy to conflate, so worth spelling out:

- **Soft-binning kernel** `p(u, i)` = `get_geo_iu_probs` in
  `src/msdpp/data.py` operates on `(lat, lon)` pairs in degrees against
  a `linspace(-90, 90) × linspace(-180, 180)` grid. Distance is Euclidean
  over degrees. **σ_geo is in degrees.**
- **Diversity metric (DM / Vendi score)** `calc_vendi_score` in
  `src/msdpp/evalindex/eval_index.py` operates on the `ext` field of
  each dataset, which for PP_geo is `F.normalize([x, y, z], 2, -1)` —
  unit 3D vectors on the sphere (`x = cos(lat)cos(lon)`,
  `y = cos(lat)sin(lon)`, `z = sin(lat)`). This is what the paper's §5.1
  refers to when it describes "distance over unit 3D vectors": the
  similarity kernel used inside the Vendi computation, not the σ_geo
  bandwidth of the soft-binning kernel.

**DM column below is `mean_vendi` (= `div_index`) from tables/\*.json,
which is directly comparable to the paper's DM column in Tables 1, 2,
9 and 10.** The decrease-direction transform (Appendix G.3) is applied
inside `calc_vendi_score` before the value is stored, so `mean_vendi`
already has higher-is-better semantics in both directions. Verification:
for every MASCOT row below, HM(R@10, DM) reproduces the paper's
reported harmonic mean exactly (PP_geo dec: 0.5478; PP_geo_hour dec:
0.3135; PP_hour dec: 0.5121; PP_geo inc: 0.9084; PP_geo_hour inc:
0.8863; PP_hour inc: 0.8951).

PP and VG tasks report R@10; **I1M reports MAP** (Table 10).

| Task | Dir | Method | λ (θ) | σ_geo | σ_time | R@10 | MAP | DM |
|---|---|---|---|---|---|---|---|---|
| PP_geo | decrease | MASCOT | 0.3 | 10.0 | 0.5 | 0.8105 | 0.6820 | 0.4136 |
| PP_geo | decrease | Uniform Binning | 0.5 | 10.0 | 0.5 | 0.8695 | 0.8224 | 0.5060 |
| PP_geo | decrease | w/o Normalization | 0.5 | 5.0 | 0.5 | 0.5947 | 0.4743 | 0.4352 |
| PP_geo | increase | MASCOT | 0.8 | 15.0 | 0.5 | 0.9109 | 0.8053 | 0.9059 |
| PP_geo | increase | Uniform Binning | 0.5 | 10.0 | 0.5 | 0.8821 | 0.8221 | 0.9343 |
| PP_geo | increase | w/o Normalization | 0.9 | 1.0 | 0.5 | 0.9398 | 0.2257 | 0.8488 |
| PP_hour | decrease | MASCOT | 0.4 | 1.0 | 0.5 | 0.9059 | 0.7518 | 0.3570 |
| PP_hour | decrease | Uniform Binning | 0.7 | 1.0 | 0.5 | 0.8381 | 0.8080 | 0.4915 |
| PP_hour | decrease | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.2974 | 0.2410 | 0.4215 |
| PP_hour | increase | MASCOT | 0.9 | 1.0 | 1.5 | 0.8921 | 0.7701 | 0.8982 |
| PP_hour | increase | Uniform Binning | 0.7 | 1.0 | 0.5 | 0.8946 | 0.8227 | 0.9189 |
| PP_hour | increase | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.9072 | 0.7761 | 0.9139 |
| PP_geo_hour | decrease | MASCOT | 0.1 | 15.0 | 3.0 | 0.9410 | 0.7883 | 0.1881 |
| PP_geo_hour | decrease | Uniform Binning | 0.7 | 10.0 | 0.5 | 0.8118 | 0.7747 | 0.2870 |
| PP_geo_hour | decrease | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.2873 | 0.2331 | 0.2401 |
| PP_geo_hour | increase | MASCOT | 0.8 | 15.0 | 1.5 | 0.8356 | 0.7892 | 0.9435 |
| PP_geo_hour | increase | Uniform Binning | 0.5 | 1.0 | 0.5 | 0.9072 | 0.7392 | 0.9381 |
| PP_geo_hour | increase | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.8871 | 0.7355 | 0.9381 |
| VG_hour | decrease | MASCOT | 0.4 | 1.0 | 0.5 | 0.9333 | 0.3823 | 0.3840 |
| VG_hour | decrease | Uniform Binning | 0.9 | 1.0 | 0.5 | 0.8667 | 0.4102 | 0.4948 |
| VG_hour | decrease | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.8667 | 0.3522 | 0.4677 |
| VG_hour | increase | MASCOT | 0.7 | 1.0 | 1.5 | 0.8667 | 0.3754 | 0.8881 |
| VG_hour | increase | Uniform Binning | 0.7 | 1.0 | 0.5 | 0.9333 | 0.4040 | 0.9219 |
| VG_hour | increase | w/o Normalization | 0.3 | 1.0 | 0.5 | 0.9333 | 0.3929 | 0.9085 |
| I1M_geo | decrease | MASCOT | 0.7 | 5.0 | 0.5 | 0.9674 | 0.6949 | 0.3871 |
| I1M_geo | decrease | Uniform Binning | 0.7 | 10.0 | 0.5 | 0.9783 | 0.6781 | 0.4520 |
| I1M_geo | decrease | w/o Normalization | 0.1 | 10.0 | 0.5 | 0.9565 | 0.6521 | 0.3977 |
| I1M_geo | increase | MASCOT | 0.7 | 10.0 | 0.5 | 0.9674 | 0.7160 | 0.8790 |
| I1M_geo | increase | Uniform Binning | 0.1 | 10.0 | 0.5 | 0.9674 | 0.7333 | 0.8685 |
| I1M_geo | increase | w/o Normalization | 0.1 | 10.0 | 0.5 | 0.9783 | 0.7178 | 0.8830 |

## Notes

- The previous hand-populated version of this table mixed up several DM values from the wrong column (e.g. VG_hour decrease MASCOT showed 0.8881, which is the VG_hour increase DM; PP_hour increase Uniform Binning showed 0.8796, which is MS-DPP+TN+TVMS's DM). This rebuild reads each cell directly from `tables/*.json` to eliminate that class of transcription error.
- **MASCOT** = full method (`ablation_no_norm=False, ablation_no_omega=False`).
- **Uniform Binning** = `ablation_no_omega=True` (blind coverage; Ω(u)=1).
- **w/o Normalization** = `ablation_no_norm=True` (raw VLM cosine, no local min-max).
- **Prob-Coverage** (in main tables) = both ablations enabled simultaneously; reported as its own baseline row (`13_prob_coverage` in tables).
