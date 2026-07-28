# lambda (theta) Sensitivity Sweep -- at each (task, direction)'s Table 1 val-best sigma

Regenerated after `fix: task.py:124` and using per-direction sigma values
extracted from Table 1/2 val-best, not the stale hardcoded DATASET_META which
fixed sigma per DATASET (not per direction) and diverged from Table 1's actual optima.

Each row: R@10 as theta sweeps at Table 1's own sigma. Val-best theta is bolded.

| Task | Dir | sigma_geo | sigma_time | val-best theta | R@10 at theta=0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PP_geo | decrease | 10.0 | 0.5 | 0.3 | 0.9737 | 0.9737 | 0.9586 | **0.8105** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PP_geo | increase | 15.0 | 0.5 | 0.8 | 0.9737 | 0.9737 | 0.9686 | 0.9649 | 0.9511 | 0.9348 | 0.8607 | 0.3752 |
| PP_hour | decrease | 1.0 | 0.5 | 0.4 | 0.9737 | 0.9724 | 0.9737 | 0.9598 | 0.5307 | 0.0063 | 0.0000 | 0.0000 |
| PP_hour | increase | 1.0 | 1.5 | 0.9 | 0.9737 | 0.9749 | 0.9737 | 0.9686 | 0.9649 | 0.9523 | **0.8921** | 0.1330 |
| PP_geo_hour | decrease | 15.0 | 3.0 | 0.1 | 0.9737 | **0.9410** | 0.0477 | 0.0088 | 0.0013 | 0.0000 | 0.0000 | 0.0000 |
| PP_geo_hour | increase | 15.0 | 1.5 | 0.8 | 0.9737 | 0.9737 | 0.9674 | 0.9573 | 0.9222 | 0.8770 | 0.7955 | 0.6236 |

## Key observations

**PP_geo_hour decrease** (paper flagship, sigma_geo=15, sigma_time=3.0):
  - Cliff between theta=0.1 (R@10=0.9410) and theta=0.2 (R@10=0.0477) -- 20x drop
  - Table 1 operating point theta=0.1 sits AT THE CLIFF EDGE

**PP_geo decrease** (sigma_geo=10, sigma_time=0.5):
  - Cliff between theta=0.3 (0.8105) and theta=0.5 (0.0000)

**PP_hour decrease** (sigma_geo=1, sigma_time=0.5):
  - Cliff between theta=0.4 (0.9059) and theta=0.7 (0.0063)

**Comparison to Appendix D Table 5 (as published):**
  - Table 5 sensitivity for PP_hour used sigma_time=1.5, Table 1's PP_hour_dec uses sigma_time=0.5
  - Table 5 for PP_geo_hour used (sigma_geo=10, sigma_time=1.5); Table 1 uses (15, 3.0)
  - Only PP_geo's sensitivity was at Table 1's actual optimum

**Camera-ready implication for section 8.5:**
  - The 'practical operating range' is NOT a fixed lambda in [0.3, 0.5] -- it is sigma-dependent
  - At sigma_time=3.0 the operable lambda for PP_geo_hour_dec collapses down to 0.1
  - Correct characterization: MASCOT achieves its Table 1 results by operating at the edge of stability, one grid step above collapse
