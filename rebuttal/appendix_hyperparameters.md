# Appendix D — Per-Task Hyperparameter Table

Val-best hyperparameters for each (task, direction, method) triple from the
grid search reported in Tables 1, 2, 9, 10, 11. Extracted from `all_metrices.log`
and `verified_old.log`, then cross-verified: for every row a fresh run at these
hyperparameters (using the bug-fixed pipeline) reproduces the published R@10.

*(This table was missing from the submission — Appendix D only stated the values
were "tuned on the validation set via grid search".)*

| Task | Dir | Method | λ (θ) | σ_geo | σ_time | grid | R@10 | DM | HM |
|---|---|---|---|---|---|---|---|---|---|
| PP_geo | dec | MASCOT | 0.3 | 10.0 | 0.5 | 20 | 0.8105 | 0.4136 | 0.5478 |
| PP_geo | dec | Uniform Binning | 0.5 | 10.0 | 0.5 | 20 | 0.8695 | 0.5060 | 0.6397 |
| PP_geo | dec | w/o Normalization | 0.5 | 5.0 | 0.5 | 20 | 0.5947 | 0.4352 | 0.5026 |
| PP_geo | inc | MASCOT | 0.8 | 15.0 | 0.5 | 20 | 0.9109 | 0.9059 | 0.9084 |
| PP_geo | inc | Uniform Binning | 0.5 | 10.0 | 0.5 | 20 | 0.8821 | 0.9343 | 0.9074 |
| PP_geo | inc | w/o Normalization | 0.9 | 1.0 | 0.5 | 20 | 0.9398 | 0.8488 | 0.8920 |
| PP_hour | dec | MASCOT | 0.4 | 1.0 | 0.5 | — | 0.9059 | 0.3570 | 0.5121 |
| PP_hour | dec | Uniform Binning | 0.7 | 1.0 | 0.5 | — | 0.8381 | 0.4915 | 0.6194 |
| PP_hour | dec | w/o Normalization | 0.5 | 1.0 | 0.5 | — | 0.2974 | 0.4215 | 0.3487 |
| PP_hour | inc | MASCOT | 0.9 | 1.0 | 1.5 | — | 0.8921 | 0.8982 | 0.8951 |
| PP_hour | inc | Uniform Binning | 0.7 | 1.0 | 0.5 | — | 0.8946 | 0.8796 | 0.8870 |
| PP_hour | inc | w/o Normalization | 0.5 | 1.0 | 0.5 | — | 0.9072 | 0.9139 | 0.9105 |
| PP_geo_hour | dec | MASCOT | 0.1 | 15.0 | 3.0 | 20 | 0.9410 | 0.1881 | 0.3135 |
| PP_geo_hour | dec | Uniform Binning | 0.7 | 10.0 | 0.5 | 20 | 0.8118 | 0.2870 | 0.4240 |
| PP_geo_hour | dec | w/o Normalization | 0.5 | 1.0 | 0.5 | 20 | 0.2873 | 0.2401 | 0.2616 |
| PP_geo_hour | inc | MASCOT | 0.8 | 15.0 | 1.5 | 20 | 0.8356 | 0.9435 | 0.8863 |
| PP_geo_hour | inc | Uniform Binning | 0.5 | 1.0 | 0.5 | 20 | 0.9072 | 0.9381 | 0.9224 |
| PP_geo_hour | inc | w/o Normalization | 0.5 | 1.0 | 0.5 | 20 | 0.8871 | 0.9381 | 0.9119 |
| VG_hour | dec | MASCOT | 0.4 | 1.0 | 0.5 | — | 0.9333 | 0.8881 | 0.9101 |
| VG_hour | dec | Uniform Binning | 0.9 | 1.0 | 0.5 | — | 0.8667 | 0.9219 | 0.8934 |
| VG_hour | dec | w/o Normalization | 0.5 | 1.0 | 0.5 | — | 0.8667 | 0.9085 | 0.8871 |
| VG_hour | inc | MASCOT | 0.7 | 1.0 | 1.5 | — | 0.8667 | 0.8840 | 0.8752 |
| VG_hour | inc | Uniform Binning | 0.7 | 1.0 | 0.5 | — | 0.9333 | 0.9219 | 0.9276 |
| VG_hour | inc | w/o Normalization | 0.3 | 1.0 | 0.5 | — | 0.9333 | 0.9085 | 0.9207 |
| I1M_geo | dec | MASCOT | 0.7 | 5.0 | 0.5 | 20 | 0.9674 | 0.8790 | 0.9211 |
| I1M_geo | dec | Uniform Binning | 0.7 | 10.0 | 0.5 | 20 | 0.9783 | 0.8685 | 0.9202 |
| I1M_geo | dec | w/o Normalization | 0.1 | 10.0 | 0.5 | 20 | 0.9565 | 0.8830 | 0.9183 |
| I1M_geo | inc | MASCOT | 0.7 | 10.0 | 0.5 | 20 | 0.9674 | 0.8790 | 0.9211 |
| I1M_geo | inc | Uniform Binning | 0.1 | 10.0 | 0.5 | 20 | 0.9674 | 0.8685 | 0.9153 |
| I1M_geo | inc | w/o Normalization | 0.1 | 10.0 | 0.5 | 20 | 0.9783 | 0.8830 | 0.9282 |

**Notes:**
- σ_geo is in km; σ_time in hours; grid is the number of geographic bins per axis (grid×grid total).
- "MASCOT" = full method (ablation_no_norm=False, ablation_no_omega=False).
- "Uniform Binning" = MASCOT with ablation_no_omega=True (blind coverage; Ω(u)=1).
- "w/o Normalization" = MASCOT with ablation_no_norm=True (raw VLM cosine, no local min-max).
- "Prob-Coverage" (in main tables) = both ablations enabled simultaneously (a separate baseline row).
