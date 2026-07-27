# PixelProse (PP) — Exact R@1 Values

Source: analysis_vg_i1m.py / failure_cases_run2/analysis.py stdout (Part 2, per-query R@K).

| Method | PP_GEO_DECREASE | PP_GEO_HOUR_DECREASE | PP_GEO_HOUR_INCREASE | PP_GEO_INCREASE | PP_HOUR_DECREASE | PP_HOUR_INCREASE |
|---|---|---|---|---|---|---|
| BLIP-2 | 0.7905 | 0.7905 | 0.7905 | 0.7905 | 0.7905 | 0.7905 |
| MMR | 0.7905 | 0.7905 | 0.7905 | 0.7905 | 0.7905 | 0.7905 |
| k-DPP | 0.0000 | 0.7792 | 0.7880 | 0.7892 | 0.7854 | 0.7880 |
| Clustering | 0.4191 | 0.4141 | 0.4304 | 0.4191 | 0.4053 | 0.4391 |
| MS-DPP | 0.4668 | 0.2346 | 0.7829 | 0.7252 | 0.4040 | 0.7654 |
| MS-DPP + TN | 0.4605 | 0.5859 | 0.7528 | 0.6412 | 0.5709 | 0.7026 |
| MS-DPP+TN+TVMS | 0.4316 | 0.5609 | 0.7390 | 0.6688 | 0.5872 | 0.7666 |
| Prob-Coverage | 0.7867 | 0.6750 | 0.7654 | 0.7842 | 0.7553 | 0.7629 |
| MASCOT (Ours) | 0.7905 | 0.7528 | 0.6211 | 0.7792 | 0.7854 | 0.7691 |
