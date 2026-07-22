# Camera-Ready Revision Checklist

Tracks concrete edits promised in the rebuttal. Each item lists the source concern, the required change, and where the supporting material lives.

## Content revisions

- [ ] **b46S-W3** — Remove the phrase "truly meets user intent" from Section 1. Reframe claims around *early-rank semantic relevance under diversity-decrease constraints*, not universal superiority.
- [ ] **Kr7y-W2** — Rename "Zero-Shot Integrity" → **"Zero-Penalty Initialization"** throughout (Section 4.5, Appendix G.2, algorithm captions).

## Table & figure revisions

- [ ] **Table 12 (Appendix G.4)** — **Delete Table 12** and replace with a short paragraph citing the vectorized numbers from `results/runtime/runtime_full.json` (2.99 ms end-to-end at N=200, K=20). Add one sentence: *"The vectorized `search_vec()` is numerically identical to the original sequential `search()` in float32 (verified in `efficiency/test_vec_correctness.py`); accuracy results in this paper use the original implementation, while runtime is measured on `search_vec()`."*
- [ ] **Kr7y-W4, jKYi-W3** — Promote sensitivity studies from Appendix D into a subsection of the main paper (Section 7 or a new Section 7.6). Reference the full sweeps in the appendix.
- [ ] **Kr7y-W5, hR87-W1** — Add a paragraph in Section 7 explicitly discussing VG_hour, I1M_geo, and SkyScript results. State clearly: MS-DPP wins on VG_hour and I1M_geo decrease; MASCOT wins on SkyScript and on PixelProse. Attribute the gap to VG_hour's tiny query set (15 types, 683 images) and I1M's IP-derived coordinates.

## New content to add

- [ ] **b46S-W1 (metadata generality)** — Add a brief paragraph in Section 7 or Appendix H referencing the visual-cluster extension result (DM: 0.815→0.869 increase, 0.420→0.689 decrease). Point to `extensions/evaluate_visual_clusters.py` in the code release.
- [ ] **nxXq (fixed-recall)** — Add the fixed-recall table (Max DM at R@10 ≥ 0.95 / 0.90 / 0.85) from `results/fixed_recall/fixed_recall_diversity.md` into Appendix D or a new subsection. Reference the script.
- [ ] **Kr7y-W6** — Expand related-work paragraph on coverage-based retrieval and submodular optimization (Yang et al. 2024, Xu et al. 2014, Iyer & Bilmes 2012).

## Theoretical clarification

- [ ] **jKYi-W1/W2** — Explicit statement of approximation guarantees in Section 5.1:
  - Increase mode: monotone submodular → (1 − 1/e) guarantee (Nemhauser et al. 1978)
  - Decrease mode: difference-of-submodular; greedy is a principled heuristic; empirical validation via R@K, PRS, early-rank collapse.

## De-anonymization (post-acceptance)

- [ ] Update author list on title page: Aaryan Sharma, Virendra Singh, Ganesh Ramakrishnan
- [ ] Add affiliation: Department of Electrical Engineering, Indian Institute of Technology Bombay
- [ ] Update Appendix H code link: `anonymous.4open.science/r/MASCOT-8D32/` → `github.com/AaryanSharma/MASCOT`
- [ ] Update BibTeX in README/citation with real DOI once assigned
