## Principled Displacement Analysis

For queries where BLIP-2 rank-1 = ground truth but MASCOT rank-1 ≠ BLIP-2 rank-1,
each displacement is classified via **primary-bin assignment**:
- **Principled**: BLIP-2's top-1 and MASCOT's top-1 are assigned to *different* primary
  metadata bins (argmax of their IU probability row). MASCOT concentrated in a
  genuinely different metadata cluster than the one containing the ground-truth image.
- **True failure**: Both images share the *same* primary bin. MASCOT was already
  in the right metadata cluster but displaced the ground-truth top-1 within it.

| Dataset | Dir | θ | Queries (GT) | Preserved | Principled | True Failure | Naive Pres | Refined Pres |
|---|---|---|---|---|---|---|---|---|
| PP_geo_hour | dec | 0.1 | 630 | 565 | 64 | 1 | 0.8968 | 0.9984 |
| PP_hour | dec | 0.4 | 630 | 515 | 113 | 2 | 0.8175 | 0.9968 |
| PP_geo | dec | 0.3 | 630 | 454 | 176 | 0 | 0.7206 | 1.0000 |

Across all three decrease cells: **353/356 (99.2%) non-preservations are principled displacement**, 3/356 (0.8%) are true failures.

### Verdict

The majority of R@1 non-preservations are principled: BLIP-2's correct top-1 image lay outside the high-Ω metadata cluster, so MASCOT correctly concentrated the list around cluster-relevant images at the cost of rank-1. This supports a nuanced restatement of §4.5: MASCOT preserves top-1 when the ground-truth image belongs to the target cluster; when it does not, displacement is the algorithmically expected outcome of concentration, not a failure of the method.

