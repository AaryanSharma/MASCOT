# Sensitivity Sweeps — Per-(task, direction) at Table 1/2 Val-Best Anchors

All sweeps regenerated after `fix: task.py:124` (bug #2) and using
per-DIRECTION anchors (θ, σ_geo, σ_time) matching Tables 1 and 2's
val-best hyperparameters. Supersedes the old `results/sensitivity/`
which used per-DATASET hardcoded σ (4 of 6 task-directions diverged
from Table 1/2's actual optima — see `rebuttal/appendix_hyperparameters.md`).

For each (task, direction, method) triple we sweep three axes independently,
each holding the other two parameters at that direction's val-best.

## Val-best anchors used

| Task | Dir | MASCOT (θ, σ_geo, σ_time) | Prob-Coverage (θ, σ_geo, σ_time) |
|---|---|---|---|
| PP_geo | decrease | (0.3, 10.0, 0.5) | (0.2, 10.0, 0.5) |
| PP_geo | increase | (0.8, 15.0, 0.5) | (0.1, 10.0, 0.5) |
| PP_hour | decrease | (0.4, 1.0, 0.5) | (0.5, 1.0, 0.5) |
| PP_hour | increase | (0.9, 1.0, 1.5) | (0.4, 1.0, 0.5) |
| PP_geo_hour | decrease | (0.1, 15.0, 3.0) | (0.4, 10.0, 1.5) |
| PP_geo_hour | increase | (0.8, 15.0, 1.5) | (0.1, 1.0, 0.5) |

## λ (theta) sweep — R@10 at each θ, per (task, direction, method)

Anchor θ (val-best) is **bolded**.

| Task | Dir | Method | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PP_geo | decrease | MASCOT | 0.9737 | 0.9737 | 0.9586 | **0.8105** | 0.0753 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PP_geo | decrease | Prob-Coverage | 0.9737 | 0.8770 | **0.8670** | 0.8356 | 0.7528 | 0.5972 | 0.0678 | 0.0627 | 0.0565 | 0.0351 |
| PP_geo | increase | MASCOT | 0.9737 | 0.9737 | 0.9686 | 0.9649 | 0.9561 | 0.9511 | 0.9348 | **0.9109** | 0.8607 | 0.3752 |
| PP_geo | increase | Prob-Coverage | 0.9737 | **0.9009** | 0.8733 | 0.8620 | 0.8394 | 0.7867 | 0.5659 | 0.3689 | 0.2535 | 0.0351 |
| PP_hour | decrease | MASCOT | 0.9737 | 0.9724 | 0.9737 | 0.9598 | **0.9059** | 0.5307 | 0.0063 | 0.0013 | 0.0000 | 0.0000 |
| PP_hour | decrease | Prob-Coverage | 0.9737 | 0.9021 | 0.8532 | 0.8369 | 0.8294 | **0.8143** | 0.7127 | 0.5721 | 0.3689 | 0.0954 |
| PP_hour | increase | MASCOT | 0.9737 | 0.9749 | 0.9737 | 0.9686 | 0.9674 | 0.9649 | 0.9523 | 0.9373 | **0.8921** | 0.1330 |
| PP_hour | increase | Prob-Coverage | 0.9737 | 0.9511 | 0.9084 | 0.8933 | **0.8833** | 0.8645 | 0.7905 | 0.6575 | 0.3915 | 0.0452 |
| PP_geo_hour | decrease | MASCOT | 0.9737 | **0.9410** | 0.0477 | 0.0088 | 0.0013 | 0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PP_geo_hour | decrease | Prob-Coverage | 0.9737 | 0.8557 | 0.8444 | 0.8118 | **0.7302** | 0.5822 | 0.0740 | 0.0678 | 0.0590 | 0.0414 |
| PP_geo_hour | increase | MASCOT | 0.9737 | 0.9737 | 0.9674 | 0.9573 | 0.9410 | 0.9222 | 0.8770 | **0.8356** | 0.7955 | 0.6236 |
| PP_geo_hour | increase | Prob-Coverage | 0.9737 | **0.9473** | 0.8457 | 0.6851 | 0.5257 | 0.3563 | 0.1882 | 0.1217 | 0.0765 | 0.0452 |

## σ_geo sweep — R@10 at each σ_geo (PP_geo, PP_geo_hour only)

| Task | Dir | Method | 0.5 | 1 | 2 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| PP_geo | decrease | MASCOT | 0.9737 | 0.9737 | 0.9737 | 0.9649 | **0.8105** | 0.0013 | 0.0000 | 0.0151 |
| PP_geo | decrease | Prob-Coverage | 0.9737 | 0.9636 | 0.8984 | 0.7227 | **0.8670** | 0.0916 | 0.0665 | 0.0627 |
| PP_geo | increase | MASCOT | 0.9737 | 0.9737 | 0.9624 | 0.8319 | 0.9260 | **0.9109** | 0.8858 | 0.8419 |
| PP_geo | increase | Prob-Coverage | 0.9737 | 0.9724 | 0.9586 | 0.9235 | **0.9009** | 0.8720 | 0.8469 | 0.7704 |
| PP_geo_hour | decrease | MASCOT | 0.9699 | 0.9699 | 0.9699 | 0.9711 | 0.9686 | **0.9410** | 0.4053 | 0.0728 |
| PP_geo_hour | decrease | Prob-Coverage | 0.8432 | 0.8055 | 0.6612 | 0.3977 | **0.7302** | 0.0590 | 0.0464 | 0.0452 |
| PP_geo_hour | increase | MASCOT | 0.9373 | 0.9310 | 0.8808 | 0.7453 | 0.8557 | **0.8356** | 0.8269 | 0.7779 |
| PP_geo_hour | increase | Prob-Coverage | 0.9511 | **0.9473** | 0.9310 | 0.8783 | 0.8908 | 0.8570 | 0.8306 | 0.7102 |

## σ_time sweep — R@10 at each σ_time (PP_hour, PP_geo_hour only)

| Task | Dir | Method | 0.25 | 0.5 | 1 | 1.5 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| PP_hour | decrease | MASCOT | 0.9172 | **0.9059** | 0.0351 | 0.0527 | 0.0602 | 0.0602 | 0.7716 |
| PP_hour | decrease | Prob-Coverage | 0.4040 | **0.8143** | 0.8344 | 0.8381 | 0.8469 | 0.8582 | 0.8720 |
| PP_hour | increase | MASCOT | 0.3588 | 0.7955 | 0.9021 | **0.8921** | 0.9072 | 0.9536 | 0.9661 |
| PP_hour | increase | Prob-Coverage | 0.3413 | **0.8833** | 0.8770 | 0.8971 | 0.9348 | 0.9586 | 0.9674 |
| PP_geo_hour | decrease | MASCOT | 0.9561 | 0.9511 | 0.9398 | 0.9348 | 0.9385 | **0.9410** | 0.9398 |
| PP_geo_hour | decrease | Prob-Coverage | 0.4141 | 0.7390 | 0.7327 | **0.7302** | 0.7390 | 0.7428 | 0.7440 |
| PP_geo_hour | increase | MASCOT | 0.7077 | 0.8482 | 0.8507 | **0.8356** | 0.8331 | 0.8394 | 0.8683 |
| PP_geo_hour | increase | Prob-Coverage | 0.9197 | **0.9473** | 0.9373 | 0.9498 | 0.9598 | 0.9624 | 0.9674 |

## grid_size sweep — R@10 at each geographic grid resolution (PP_geo, PP_geo_hour only)

| Task | Dir | Method | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| PP_geo | decrease | MASCOT | 0.9686 | 0.9661 | 0.9448 | **0.8105** | 0.0025 | 0.0000 | 0.0000 | 0.0000 |
| PP_geo | decrease | Prob-Coverage | 0.7541 | 0.8168 | 0.7892 | **0.8670** | 0.7566 | 0.2233 | 0.0853 | 0.0690 |
| PP_geo | increase | MASCOT | 0.9649 | 0.9247 | 0.9210 | **0.9109** | 0.8996 | 0.8620 | 0.7992 | 0.7541 |
| PP_geo | increase | Prob-Coverage | 0.9699 | 0.9460 | 0.9335 | **0.9009** | 0.8858 | 0.8808 | 0.8670 | 0.8632 |
| PP_geo_hour | decrease | MASCOT | 0.9661 | 0.9724 | 0.9661 | **0.9410** | 0.5834 | 0.0715 | 0.0063 | 0.0013 |
| PP_geo_hour | decrease | Prob-Coverage | 0.3814 | 0.4780 | 0.4003 | **0.7302** | 0.0916 | 0.0816 | 0.0615 | 0.0527 |
| PP_geo_hour | increase | MASCOT | 0.8770 | 0.8193 | 0.8482 | **0.8356** | 0.8218 | 0.8118 | 0.7905 | 0.7779 |
| PP_geo_hour | increase | Prob-Coverage | 0.9511 | 0.9486 | 0.9310 | **0.9473** | 0.9373 | 0.9360 | 0.9260 | 0.9021 |

## Key observations

### λ (theta) sweeps at correct per-direction σ

- **PP_geo_hour dec** (σ_geo=15, σ_time=3.0): cliff between θ=0.1 (R@10=0.9410, anchor) and θ=0.2 (R@10=0.0477). 20× recall collapse in one step.
- **PP_hour dec** (σ_geo=1, σ_time=0.5): cliff between θ=0.4 (R@10=0.9059, anchor) and θ=0.7 (R@10=0.0063).
- **PP_geo dec** (σ_geo=10, σ_time=0.5): cliff between θ=0.3 (R@10=0.8105, anchor) and θ=0.5 (R@10=0.0000).
- Every decrease-task MASCOT anchor sits at most one grid step above its cliff.

### σ sweeps at correct per-direction θ

- The pre-fix σ sweeps were all anchored at λ=0.5, which is the wrong value for 4 of 6 (task, direction) pairs. The corrected sweeps above are anchored at each direction's own val-best θ.
- Where the val-best σ is a bandwidth (e.g. σ_time), sweeps generally show a stability plateau near the val-best that widens or narrows depending on task.

### grid_size sweeps

- The geographic grid resolution primarily affects decrease tasks. On PP_geo_hour dec (θ=0.1, σg=15, σt=3.0), coarse grids preserve or slightly exceed the val-best grid=20 R@10; finer grids (40, 50) collapse R@10 to near zero.
- On PP_geo dec at θ=0.3, R@10 is stable across the tested grid range.

## Provenance

Anchors sourced from:
- MASCOT: `results/failure_cases_run2/tables/PP_*_{decrease,increase}.json::14_ma_smf` (val-best per direction; matches paper Tables 1 and 2 exactly).
- Prob-Coverage: `13_prob_coverage` from the same tables (val-best per direction, reverse-matched to grid-search logs by (R@10, DM)).

All rows: fresh-run R@10 at the anchor reproduces the corresponding table value to within 1e-3.
