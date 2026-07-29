# Rebuttal: MASCOT — Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval

**Submission #9126 · ACM MM 2026**

We thank all reviewers. The reviews identify the same central contribution: MASCOT addresses a failure mode of manifold-repulsion methods under diversity-decrease constraints while preserving early-rank semantic relevance. We clarify the main concerns below.

## Factual Incorrectness and Appendix Availability (b46S-W2/W4, jKYi-W3, Kr7y-W1/W5)

Reviewer b46S states that "no such appendix exists" (Appendix A). Reviewer jKYi raises the absence of sensitivity and granularity analyses (Appendix D), while Reviewer Kr7y raises the absence of runtime analysis, complexity discussion (Appendix G.4), and evaluation beyond a single dataset (Appendix E,F). These are incorrect. These topics are already covered in the supplementary material, including sensitivity studies (λ, σ_geo, σ_time, grid resolution), additional datasets (VG, I1M, SkyScript), and runtime analysis. We ask the reviewers to re-examine the submission package.

## Metadata Generality and Practical Value (b46S-W1)

The optimizer operates on a generic probability matrix and is not restricted to geo-temporal metadata. Categorical attributes (semantic tags, object categories, user annotations) reduce directly to one-hot or multi-hot set coverage. Continuous attributes (geo, time) require Gaussian soft-binning to preserve neighborhood structure and are the harder case we deliberately target. Unstructured metadata (free-text annotations or learned embeddings) requires only an upstream clustering step to induce bins; the coverage objective itself is unchanged.

To verify this empirically, we induced **10 visual clusters** from BLIP-2 image embeddings on PixelProse and used cluster assignments as metadata. (The initial rebuttal draft stated 50 clusters, which matches the codebase defaults in `create_pp_visualcluster.py` and `run_visual_cluster_verification.sh`; the actual PP_visualcluster_val/test.pkl on disk was generated with num_clusters=10, verified by loading the pickled `RetrievalDataset` object. The DM numbers below are from the 10-cluster run and can be reproduced by `python final_vc_test.py`.) Without modifying the objective, **MASCOT improved DM from 0.815 to 0.869 at R@10=0.98 (increase) and from 0.420 to 0.689 at R@10=0.96 (decrease)**, demonstrating extension beyond geo-temporal metadata. *(See [`extensions/evaluate_visual_clusters.py`](../extensions/evaluate_visual_clusters.py) and [`results/visual_clusters/final_vc_test.log`](../results/visual_clusters/final_vc_test.log) — reproduces exactly.)*

The practical value of diversification is quantified through PRS, which measures alignment between requested and achieved retrieval behavior. MASCOT achieves PRS=1.0 on 4/6 task-direction pairs while maintaining high retrieval quality.

## Runtime and Memory Usage (hR87-W3, Kr7y-W1, nxXq-W4)

The originally reported 1266 ms/query reflected implementation overhead rather than algorithmic cost. After GPU caching and vectorization, the same objective achieves **2.99 ms end-to-end latency** at N=200, K=20, |U|=424 (matching MASCOT and MS-DPP settings), with outputs verified identical to the original sequential implementation in float32. Additional GPU memory remains below 2 MB per query. We will update Table 12 and release the optimized implementation. *(See [`efficiency/benchmark_full_pipeline.py`](../efficiency/benchmark_full_pipeline.py) and [`results/runtime/runtime_full.json`](../results/runtime/runtime_full.json). Numerical equivalence verified in [`efficiency/test_vec_correctness.py`](../efficiency/test_vec_correctness.py).)*

## Sensitivity, Dataset Scope, and Fixed-Recall Analysis (jKYi-W3, Kr7y-W4/W5, hR87-W1/W2, nxXq-W1/W2)

Our claim is targeted: MASCOT preserves early-rank semantic relevance under diversity-decrease constraints, where manifold-repulsion methods can collapse retrieval quality. On PixelProse, MASCOT improves R@10 from 0.4931 to 0.9410 on PP_geo_hour decrease.

MASCOT achieves its strongest results on PixelProse and the best overall result on out-of-domain SkyScript. We acknowledge that MS-DPP achieves higher HM on VG_hour and I1M_geo decrease tasks. VG_hour contains only 15 query types and 683 images, while I1M_geo relies on IP-derived coordinates rather than EXIF metadata. We will discuss these results explicitly in the camera-ready and frame MASCOT as a targeted solution for early-rank recall preservation under diversity-decrease constraints rather than a universally dominant diversification method.

Per reviewer nxXq's request, we provide a fixed-recall analysis on PP_geo_hour decrease:

| Method | Max DM @ R@10≥0.95 | Max DM @ R@10≥0.90 |
|---|---|---|
| **MASCOT** | **0.1784** | **0.1881** |
| Prob-Coverage | 0.1656 | 0.1656 |

At matched recall floors, MASCOT achieves stronger concentration. *(See [`extensions/build_fixed_recall_tables.py`](../extensions/build_fixed_recall_tables.py) and [`results/fixed_recall/fixed_recall_diversity.md`](../results/fixed_recall/fixed_recall_diversity.md).)*

## Theoretical Clarification (jKYi-W1/W2, b46S-W1)

**Increase mode:** F⁺(S) = (1−θ)R(S) + θ cov(S), where R is modular and cov is monotone submodular. Therefore F⁺ is monotone submodular and greedy selection under a cardinality constraint attains the classical (1−1/e) approximation guarantee (Nemhauser et al., 1978).

**Decrease mode:** F⁻(S) = (1−θ)R(S) − θ cov(S). Since cov is monotone submodular, −cov is supermodular and F⁻ becomes a difference-of-submodular objective (Iyer and Bilmes, 2012). The classical (1−1/e) guarantee therefore does not apply. We treat greedy selection as a principled heuristic and validate it empirically through Recall@K, PRS, and the early-rank failure-mode analysis.

## Camera-Ready Revisions

See [`camera_ready_todos.md`](camera_ready_todos.md) for the tracked checklist of revisions promised above.

| Concern | Revision |
|---|---|
| b46S-W3 | Remove "truly meets user intent" and reframe claims around early-rank semantic relevance |
| Kr7y-W2 | Rename to "Zero-Penalty Initialization" throughout |
| Kr7y-W4, jKYi-W3 | Promote sensitivity studies from Appendix D into the main paper |
| Kr7y-W5, hR87-W1 | Explicit discussion of VG_hour, I1M_geo, and SkyScript results |
| Kr7y-W6 | Expand relation to coverage-based retrieval and submodular optimization methods |

## Summary for the Area Chair

MASCOT identifies a specific failure mode of manifold-based diversification under diversity-decrease constraints and addresses it through metadata-aware coverage optimization. The central PixelProse result remains unchanged: under PP_geo_hour decrease, MS-DPP reduces R@10 to 0.4931 whereas MASCOT maintains 0.9410 while satisfying the requested concentration objective. Three reviewers recommend acceptance. The weak-reject and borderline review contains factual errors regarding the submitted appendix, and all remaining concerns regarding theory, runtime, sensitivity, trade-offs, and dataset scope are addressed above and will be incorporated into the camera-ready version.
