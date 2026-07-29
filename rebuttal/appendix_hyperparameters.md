# Appendix — Per-Task Hyperparameter Table

Val-best hyperparameters for each (task, direction, method) triple.
Every row **rebuilt programmatically** from `tables/*.json` after the
bug #1+#2 fixes. R@10 / MAP / DM cells come directly from the
`14_ma_smf` / `15_ma_smf_ablation_no_norm` / `16_ma_smf_ablation_no_omega`
entries in `results/{failure_cases_run2, vg, i1m}/tables/<TASK>.json`.
Hyperparameters (θ, σ_geo, σ_time) reverse-matched from
`all_metrices.log` / `verified_old.log` / `vg_metrices.log` /
`i1m_metrices*.log` grid-search best-param records: for PP by (R@10, DM),
for VG and I1M by (MAP, DM) since those tasks record MAP as Retrieval Index.

Units: **σ_geo in degrees** (Euclidean distance in lat/lon degree-space;
see `src/msdpp/data.py::get_geo_iu_probs`). **σ_time in hours**.
For decrease rows, DM is raw `mean_vendi`; the paper's main tables use
`1 − DM` so higher is better in both directions (Appendix G.3).
PP and VG tasks report R@10; **I1M reports MAP** (Table 10).

| Task | Dir | Method | λ (θ) | σ_geo | σ_time | R@10 | MAP | mean_vendi | 1−DM (dec) |
|---|---|---|---|---|---|---|---|---|---|
| PP_geo | decrease | MASCOT | 0.3 | 10.0 | 0.5 | 0.8105 | 0.6820 | 0.4136 | 0.5864 |
| PP_geo | decrease | Uniform Binning | 0.5 | 10.0 | 0.5 | 0.8695 | 0.8224 | 0.5060 | 0.4940 |
| PP_geo | decrease | w/o Normalization | 0.5 | 5.0 | 0.5 | 0.5947 | 0.4743 | 0.4352 | 0.5648 |
| PP_geo | increase | MASCOT | 0.8 | 15.0 | 0.5 | 0.9109 | 0.8053 | 0.9059 | — |
| PP_geo | increase | Uniform Binning | 0.5 | 10.0 | 0.5 | 0.8821 | 0.8221 | 0.9343 | — |
| PP_geo | increase | w/o Normalization | 0.9 | 1.0 | 0.5 | 0.9398 | 0.2257 | 0.8488 | — |
| PP_hour | decrease | MASCOT | 0.4 | 1.0 | 0.5 | 0.9059 | 0.7518 | 0.3570 | 0.6430 |
| PP_hour | decrease | Uniform Binning | 0.7 | 1.0 | 0.5 | 0.8381 | 0.8080 | 0.4915 | 0.5085 |
| PP_hour | decrease | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.2974 | 0.2410 | 0.4215 | 0.5785 |
| PP_hour | increase | MASCOT | 0.9 | 1.0 | 1.5 | 0.8921 | 0.7701 | 0.8982 | — |
| PP_hour | increase | Uniform Binning | 0.7 | 1.0 | 0.5 | 0.8946 | 0.8227 | 0.9189 | — |
| PP_hour | increase | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.9072 | 0.7761 | 0.9139 | — |
| PP_geo_hour | decrease | MASCOT | 0.1 | 15.0 | 3.0 | 0.9410 | 0.7883 | 0.1881 | 0.8119 |
| PP_geo_hour | decrease | Uniform Binning | 0.7 | 10.0 | 0.5 | 0.8118 | 0.7747 | 0.2870 | 0.7130 |
| PP_geo_hour | decrease | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.2873 | 0.2331 | 0.2401 | 0.7599 |
| PP_geo_hour | increase | MASCOT | 0.8 | 15.0 | 1.5 | 0.8356 | 0.7892 | 0.9435 | — |
| PP_geo_hour | increase | Uniform Binning | 0.5 | 1.0 | 0.5 | 0.9072 | 0.7392 | 0.9381 | — |
| PP_geo_hour | increase | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.8871 | 0.7355 | 0.9381 | — |
| VG_hour | decrease | MASCOT | 0.4 | 1.0 | 0.5 | 0.9333 | 0.3823 | 0.3840 | 0.6160 |
| VG_hour | decrease | Uniform Binning | 0.9 | 1.0 | 0.5 | 0.8667 | 0.4102 | 0.4948 | 0.5052 |
| VG_hour | decrease | w/o Normalization | 0.5 | 1.0 | 0.5 | 0.8667 | 0.3522 | 0.4677 | 0.5323 |
| VG_hour | increase | MASCOT | 0.7 | 1.0 | 1.5 | 0.8667 | 0.3754 | 0.8881 | — |
| VG_hour | increase | Uniform Binning | 0.7 | 1.0 | 0.5 | 0.9333 | 0.4040 | 0.9219 | — |
| VG_hour | increase | w/o Normalization | 0.3 | 1.0 | 0.5 | 0.9333 | 0.3929 | 0.9085 | — |
| I1M_geo | decrease | MASCOT | 0.7 | 5.0 | 0.5 | 0.9674 | 0.6949 | 0.3871 | 0.6129 |
| I1M_geo | decrease | Uniform Binning | 0.7 | 10.0 | 0.5 | 0.9783 | 0.6781 | 0.4520 | 0.5480 |
| I1M_geo | decrease | w/o Normalization | 0.1 | 10.0 | 0.5 | 0.9565 | 0.6521 | 0.3977 | 0.6023 |
| I1M_geo | increase | MASCOT | 0.7 | 10.0 | 0.5 | 0.9674 | 0.7160 | 0.8790 | — |
| I1M_geo | increase | Uniform Binning | 0.1 | 10.0 | 0.5 | 0.9674 | 0.7333 | 0.8685 | — |
| I1M_geo | increase | w/o Normalization | 0.1 | 10.0 | 0.5 | 0.9783 | 0.7178 | 0.8830 | — |

## Notes

- The previous version of this table was populated by hand and mixed up several DM values from the wrong column (e.g. VG_hour decrease MASCOT showed 0.8881 which is the VG_hour increase DM; PP_hour increase Uniform Binning showed 0.8796 which is MS-DPP+TN+TVMS's DM in that column). This rebuild reads each cell directly from `tables/*.json` to eliminate that class of transcription error.
- **`?`** in the hyperparameter columns means the log-based reverse match could not resolve those params uniquely from the recorded grid-search runs (either the log entry is missing or its (metric, DM) pair rounds to something different than the table's canonical value). The R@10/MAP/DM values themselves come from `tables/*.json` and are canonical.
- **MASCOT** = full method (`ablation_no_norm=False, ablation_no_omega=False`).
- **Uniform Binning** = `ablation_no_omega=True` (blind coverage; Ω(u)=1).
- **w/o Normalization** = `ablation_no_norm=True` (raw VLM cosine, no local min-max).
- **Prob-Coverage** (in main tables) = both ablations enabled simultaneously; reported as its own baseline row (13_prob_coverage in tables).
