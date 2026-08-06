<div align="center">

# MASCOT: Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval

[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-blue.svg)](https://2026.acmmm.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org)
[![Built on MS-DPP](https://img.shields.io/badge/Built%20on-MS--DPP%20codebase-lightgrey.svg)](https://arxiv.org/abs/2507.06654)

**ACM MM 2026** | Rio de Janeiro, Brazil

**Aaryan Sharma¹ · Vishak Prasad C² · Virendra Singh¹ · Ganesh Ramakrishnan²**

¹Department of Electrical Engineering · ²Department of Computer Science and Engineering
Indian Institute of Technology Bombay

</div>

---

## What is MASCOT?

**MASCOT** (Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval) is a novel post-hoc re-ranking framework that addresses a critical failure mode of current state-of-the-art retrieval diversification systems.

While manifold-based methods (MS-DPP) excel at *increasing* diversity via spatial repulsion, they catastrophically collapse when asked to *decrease* diversity (e.g., "retrieve only images from a specific time window or region"). MASCOT solves this by reframing diversity control as a **probabilistic resource allocation** problem over discrete metadata bins, using the principle of **diminishing marginal returns** instead of continuous geometric repulsion.

### Key Results

| Setting | Metric | MS-DPP | MASCOT |
|---|---|---|---|
| PP_geo_hour (Decrease) | R@10 | 0.4931 | **0.9410** |
| PP_hour (Decrease) | R@10 | 0.7654 | **0.9059** |
| PP_geo (Decrease) | R@10 | 0.7704 | **0.8105** |
| Average (Decrease) | R@10 | 67.63% | **88.58%** |

MASCOT preserves early-rank recall in the decrease regime — MS-DPP's manifold repulsion collapses R@10 from >0.97 to 0.49 on the composite PP_geo_hour decrease task, whereas MASCOT maintains 0.94. On the harmonic mean of R@10 and Diversity Index, the coverage ablations (Uniform Binning, Prob-Coverage) reach comparable or higher scores than full MASCOT on these decrease tasks; MASCOT's contribution is the recall protection under diversity-decrease constraints, not universal HM dominance.

---

## Core Contributions

1. **Manifold Vulnerability Identification**: We show that MS-DPP collapses below 50% R@1 during diversity-decrease tasks due to its continuous spatial repulsion being structurally incompatible with tight-cluster constraints.

2. **MASCOT Framework**: A submodular coverage formulation with:
   - **Soft Information Units (IUs)**: Gaussian-kernel soft-binning of discrete metadata (24 temporal bins; geographic grid) to prevent hard boundary penalties.
   - **Normalized Semantic Relevance (R-hat)**: Local min-max normalization of VLM cosine scores to compete against cumulative coverage sums.
   - **Query-Driven Bin Importance (Ω)**: Dynamic bin weights equal to peak relevance within each bin — prevents wasting retrieval budget on empty or irrelevant bins.

3. **Recall-Preserving Diversity Decrease**: MASCOT preserves early-rank recall (R@10) under diversity-decrease constraints where manifold repulsion (MS-DPP) collapses it. This is MASCOT's targeted contribution; ablations of MASCOT (Uniform Binning, Prob-Coverage) achieve comparable or higher harmonic mean on these tasks — recall protection is the differentiator, not universal HM dominance.

---

## MASCOT Objective

```
f(S) = (1-λ) Σ_{i∈S} R̂(i,q) + d·λ Σ_{u∈U} Ω(u,q) · (1 - Π_{i∈S}(1 - p(u,i)))
```

- `λ ∈ [0,1]`: user-defined diversification intensity
- `d ∈ {-1, +1}`: task direction (decrease / increase)
- `p(u, i)`: Gaussian soft-bin probability of image `i` in bin `u`
- `Ω(u, q)`: query-driven bin importance (peak relevance in bin)
- `R̂(i, q)`: locally normalized semantic relevance

For `d = -1` (decrease), coverage becomes a **penalty**: selecting images in already-covered bins incurs near-zero penalty, defaulting selection to semantic relevance — this is the zero-penalty saturation property.

---

## Project Structure (built on MS-DPP codebase)

```
msdpp/
├── src/msdpp/
│   ├── div_method/
│   │   ├── ma_smf.py           ← MASCOT implementation (ModelAwareSubmodularMethod)
│   │   ├── prob_coverage.py    ← Baseline: naive probabilistic coverage
│   │   ├── dpp.py              ← k-DPP baseline
│   │   ├── dpp_E2.py           ← MS-DPP / MS-DPP+TN / MS-DPP+TN+TVMS
│   │   └── standard_baselines.py  ← MMR, Clustering, BLIP-2 (org)
│   ├── data.py                 ← Soft-binning: get_time_iu_probs, get_geo_iu_probs
│   ├── task.py                 ← BaseTask: retrieval pipeline + bin probability computation
│   ├── models/blip2.py         ← BLIP-2 dense retriever
│   ├── evalindex/eval_index.py ← R@K, MAP, MRR, Vendi Score (Div Index)
│   └── schema/                 ← Data structures (TaskResult, RetrievalDataset, etc.)
├── data/
│   ├── pixelprose_preprocess.py   ← PixelProse download + EXIF filter + GPS/time extraction
│   ├── vg_preprocess.py           ← Visual Genome (VG_hour) preprocessing
│   ├── i1m_preprocess.py          ← Incidents1M (I1M_geo) preprocessing + IP geolocation
│   ├── skyscript_preprocess.py    ← SkyScript remote sensing preprocessing
│   ├── cluster_util.py            ← K-means visual clustering utility
│   ├── create_pp_visualcluster.py ← Build PP_visualcluster dataset (BLIP-2 features → clusters)
│   └── add_visual_clusters.py     ← Add cluster IDs to an existing dataset
├── examples/
│   ├── grid_search_eval.py        ← Main evaluation script (grid search over λ, σ)
│   ├── sensitivity_analysis.py    ← Hyperparameter sensitivity tables
│   ├── clip_gridsearch.py         ← CLIP backbone generalization (post-submission)
│   └── configs/
│       ├── overall.json           ← Dataset/model config (PP variants)
│       ├── overall_pp_clip.json   ← CLIP backbone config
│       ├── div_selected.json      ← Method + hyperparameter grid configs
│       ├── div_clip.json          ← CLIP div-method configs
│       └── tables/                ← Per-task result JSON files
├── extensions/                    ← Post-submission extensions (rebuttal + camera-ready)
│   ├── evaluate_visual_clusters.py    ← Visual cluster metadata (BLIP-2 features → 10 IUs)
│   ├── mixed_direction_eval.py        ← Per-attribute direction (geo↑+time↓, etc.)
│   ├── build_fixed_recall_tables.py   ← Max DM at fixed R@10 floors
│   └── verify_fixed_recall.py         ← Cache verification for fixed_recall
├── efficiency/                    ← Vectorized runtime + correctness
│   ├── benchmark_full_pipeline.py     ← End-to-end 2.99 ms benchmark (rebuttal)
│   └── test_vec_correctness.py        ← Proves search() == search_vec() in float32
├── prs/                           ← Preference Reflection Score & top-1 integrity analysis
│   ├── principled_displacement.py     ← Cluster-aware displacement classification (99.2% principled)
│   ├── threshold_robustness.py        ← Threshold-invariance check for the above
│   └── top1_integrity.py              ← R@1 preservation sweeps
├── results/                       ← Result artifacts referenced from the paper & rebuttal
│   ├── fixed_recall/                  ← fixed_recall_diversity.md
│   ├── runtime/                       ← runtime_full.json (2.99 ms)
│   ├── mixed_direction/               ← SUMMARY.md + JSONs
│   ├── top1_integrity/                ← SUMMARY.md + principled_displacement.json
│   ├── visual_clusters/               ← final_vc_test.log (10-cluster run: DM 0.815→0.869 inc, 0.420→0.689 dec)
│   ├── pp_clip/                       ← CLIP backbone SUMMARY.md + tables
│   ├── sensitivity_v2/                ← Per-direction θ sweeps at Table 1 val-best σ (supersedes results/sensitivity/)
│   ├── figures_regenerated/           ← Recall@K + trade-off plots after bug #1+#2 fixes
│   ├── failure_cases_corrected/       ← Per-query MASCOT/UB/no_norm rankings (30 files, verified against Table 1/2/9/10/11)
│   ├── _stale_pre_fix/                ← Documentation of the mislabeled failure_cases files from the pre-fix pipeline
│   └── analysis_vg_i1m.py             ← VG and I1M result analysis
├── rebuttal/                      ← ACM MM 2026 rebuttal materials
│   ├── README.md
│   ├── response.md                    ← Point-by-point response
│   ├── camera_ready_todos.md          ← Tracked revisions for camera-ready
│   └── appendix_hyperparameters.md    ← Per-(task, direction, method) hyperparameter table for Appendix D
├── scripts/                       ← Reproducibility scripts (regen failure_cases + sensitivity sweeps)
└── pyproject.toml                 ← Dependencies (uv)
```

---

## Installation

### Prerequisites
- CUDA-compatible GPU (required for BLIP-2 inference)
- Python 3.10–3.11

### Option 1: Docker (Recommended)

```bash
# 1. Configure mount paths
vim Makefile

# 2. Launch container
make up
```

### Option 2: uv

```bash
uv sync
source .venv/bin/activate
```

### Dependencies (key packages)

- `torch >= 2.5.1`, `torchvision >= 0.20.1`
- `transformers >= 4.25.0` (BLIP-2)
- `datasets >= 3.0.2` (HuggingFace datasets)
- `scipy`, `scikit-learn`, `numpy`
- `pydantic >= 2.10.3`

---

## Data Preparation

MASCOT is evaluated on four datasets. All preprocessing scripts live in `data/`.

### PixelProse (PP) — Primary Benchmark

```bash
# Download, EXIF-filter, extract GPS + shooting time, compute embeddings
cd data/
python pixelprose_preprocess.py
```

- Source: HuggingFace `tomg-group-umd/pixelprose`
- Filter: retain only images with valid `GPSLatitude/GPSLongitude` AND `DateTimeOriginal` EXIF
- Result: ~996 images (996 of 1,799 EXIF-filtered survive URL attrition)
- Supports: `PP_geo`, `PP_hour`, `PP_geo_hour` tasks

> **Note**: The paper reports 25,151 images following MS-DPP [48]. Due to subsequent PixelProse HuggingFace updates and URL decay, the current reproducible count is 996. All comparisons remain valid as the same 996 images are used for all methods.

### Visual Genome (VG) — Temporal Task

```bash
python vg_preprocess.py
```

- Scans HTTP EXIF headers (first 64KB) to find `DateTimeOriginal`
- Yields ~683 images with valid shooting-time metadata
- Supports: `VG_hour`

### Incidents1M (I1M) — Geographic Task

```bash
python i1m_preprocess.py
```

- Uses `multi_label_val.json`; approximates GPS via domain→IP→geolocation
- Yields ~26,517 images with valid GPS
- Supports: `I1M_geo`

### SkyScript — Remote Sensing Geographic Task

```bash
python skyscript_preprocess.py
```

- Uses `SkyScript_test_30K` split; GPS from bounding box center
- Yields ~15,650 images
- Supports: `SkyScript_geo`

### Dataset Splits

All datasets use a **20% val / 80% test** split. Hyperparameters are tuned on val and held fixed at test time.

---

## Running Evaluation

### Full Grid Search (PP datasets)

```bash
cd examples/
python grid_search_eval.py
```

Config is controlled by:
- `configs/overall.json`: dataset name, BLIP-2 model variant, `subset_k=200`, `top_k=20`
- `configs/div_selected.json`: method groups + hyperparameter grids

### Key config parameters (`div_selected.json`)

| Group | Method | Key Params |
|---|---|---|
| `model_aware` | `ma_smf` (MASCOT) | `theta`, `sigma_geo`, `sigma_time` |
| `summarization` | `prob_coverage` | `theta`, `sigma_geo`, `sigma_time` |
| `dpps` | `msdpp`, `msdpp_tn`, `msdpp_tn_tvms` | `beta`, `theta` |
| `baseline_*` | `blip2`, `mmr`, `clustering` | varies |
| `ablations_no_norm` | `ma_smf` | `ablation_no_norm=true` |
| `ablations_no_omega` | `ma_smf` | `ablation_no_omega=true` |

### Sensitivity Analysis

```bash
python sensitivity_analysis.py
```

Reproduces Tables 5–8 in the paper (sensitivity to λ, σ_geo, σ_time, grid resolution).

### VG and I1M Results

```bash
python results/analysis_vg_i1m.py
```

---

## MASCOT Implementation Details

### Registration

MASCOT is registered as `"ma_smf"` in the method registry:

```python
from msdpp import registry

method = registry.build_div_method(
    "ma_smf",
    theta=0.5,          # λ: diversity intensity
    sigma_geo=10.0,     # geographic Gaussian bandwidth (degrees)
    sigma_time=1.5,     # temporal Gaussian bandwidth (hours)
    ablation_no_norm=False,
    ablation_no_omega=False,
)
```

### Soft-Binning (from `src/msdpp/data.py`)

```python
from msdpp.data import get_time_iu_probs, get_geo_iu_probs

# Temporal: [N, 24] — soft probabilities over 24 hourly bins
time_probs = get_time_iu_probs(hours, minutes, num_bins=24, sigma=1.5)

# Geographic: [N, grid_size²] — soft probabilities over geographic grid
geo_probs = get_geo_iu_probs(gps_xyz, grid_size=20, sigma=10.0)
```

Both use Gaussian kernels: `p(u, i) = exp(-dist² / 2σ²)`, with probabilities below 1% zeroed for sparsity.

### Ablations

| Ablation | Flag | Effect |
|---|---|---|
| w/o Normalization | `ablation_no_norm=True` | Use raw VLM cosine scores (not normalized) |
| Uniform Binning | `ablation_no_omega=True` | Set all Ω(u,q) = 1.0 (blind coverage) |
| Prob-Coverage (both ablated) | both `True` | Naive coverage without normalization or query-driven weights |

---

## Hyperparameters

Val-best MASCOT hyperparameters per (task, direction). σ_geo is in **degrees**
(distance is Euclidean in lat/lon degree-space; see `get_geo_iu_probs` in
`src/msdpp/data.py`). σ_time is in hours.

| Task | Direction | λ (θ) | σ_geo | σ_time | grid | R@10 |
|---|---|---|---|---|---|---|
| PP_geo | decrease | 0.3 | 10.0 | 0.5 | 20 | 0.8105 |
| PP_geo | increase | 0.8 | 15.0 | 0.5 | 20 | 0.9109 |
| PP_hour | decrease | 0.4 | 1.0 | 0.5 | — | 0.9059 |
| PP_hour | increase | 0.9 | 1.0 | 1.5 | — | 0.8921 |
| PP_geo_hour | decrease | **0.1** | 15.0 | 3.0 | 20 | 0.9410 |
| PP_geo_hour | increase | 0.8 | 15.0 | 1.5 | 20 | 0.8356 |

Full per-method (MASCOT / Uniform Binning / w/o Normalization) hyperparameter
table for all datasets is in [`rebuttal/appendix_hyperparameters.md`](rebuttal/appendix_hyperparameters.md).

**Sensitivity to λ is σ-dependent, not fixed.** At each task's val-best σ,
the corrected θ sweeps in [`results/sensitivity_v2/SUMMARY.md`](results/sensitivity_v2/SUMMARY.md)
show every decrease-task operating point sits one grid step above a hard
recall cliff (e.g. PP_geo_hour_dec at θ=0.1 gives R@10=0.9410; at θ=0.2 it
collapses to 0.0477). The earlier claim of a wide stable λ range was based
on stale sensitivity sweeps at σ values that did not match the Table 1
val-best; see [`results/sensitivity_v2/SUMMARY.md`](results/sensitivity_v2/SUMMARY.md)
for the reconciled per-direction findings.

---

## Metrics

| Category | Metric | Description |
|---|---|---|
| Semantic Retention | **R@10** | Recall of ground-truth image in top-10 |
| Semantic Retention | MAP, MRR, R@1, R@5 | Standard retrieval metrics |
| Attribute Diversity | **Div Index** | Normalized Vendi Score = exp(von Neumann entropy of similarity matrix) |
| Combined | **HM (Overall Score)** | Harmonic Mean of R@10 and Div Index |
| Controllability | **PRS** | Preference Reflection Score — monotonicity of Div Index as λ sweeps 0→1 |

In **increase** tasks, higher Div Index = better. In **decrease** tasks, the paper reports the transformed metric `1 − DM` so that higher is always better in both directions (see Appendix G.3). The `mean_vendi` / `div_index` field in `tables/*.json` is already transformed to higher-is-better in `EvalIndexCalculator.calc()` (`src/msdpp/evalindex/eval_index.py:162-164` applies `1 − ext_vendi` when `direction == DivDir.DECREASE` before harmonic-mean combination); no downstream conversion needed.

---

## Reproducing Paper Tables

### Table 1 (Decrease Tasks, PP datasets)

```bash
cd examples/
python grid_search_eval.py --dataset pp  # uses configs/overall.json + div_selected.json
# Results written to results/pp/tables/PP_geo_decrease.json, etc.
# (The paper's cited PP tables live at results/failure_cases_run2/tables/, produced
#  by re-running with --result_dir results/failure_cases_run2 after bug #1/#2 fixes.)
```

### Table 2 (Increase Tasks, PP datasets)

Same script; direction is controlled per-task in the evaluation config.

### Table 3 (Preference Reflection Score)

```bash
python time_prs.py   # from examples/
```

### Tables 9–10 (VG, I1M)

```bash
python results/analysis_vg_i1m.py
```

### Figure 1 (Recall@K curves)

```bash
# PP datasets — recall curves + trade-off scatter
cd results/failure_cases_run2/
python analysis.py

# VG_hour + I1M_geo — recall curves + trade-off scatter
cd ../
python analysis_vg_i1m.py
```

Curves are written under `figures/` and `figures_vg_i1m/` respectively.
`visualize_dataset.py` was a scratch script, not the figure generator.

---

## Post-Submission Experiments (Rebuttal & Camera-Ready)

The rebuttal cites four experiments not in the original submission. All scripts and result artifacts are included and reproduce the cited numbers exactly. See [`rebuttal/`](rebuttal/) for the full response and camera-ready checklist.

### 1. Visual cluster metadata extension (rebuttal §"Metadata Generality")

Verifies that MASCOT extends beyond geo/time metadata to learned image-embedding clusters.

```bash
# Preprocess (one-time): induce 10 visual clusters from BLIP-2 embeddings
python data/create_pp_visualcluster.py --base_dataset_path ./tasks/PP_base --output_path ./tasks --num_clusters 10

# Evaluate MASCOT + MS-DPP + Prob-Coverage on the cluster metadata
python extensions/evaluate_visual_clusters.py
# → results/visual_clusters/final_vc_test.log
# Rebuttal numbers: MASCOT DM 0.815→0.869 (increase), 0.420→0.689 (decrease)
```

### 2. Vectorized runtime (rebuttal §"Runtime and Memory Usage")

The original submission reported 1266 ms/query. After GPU caching and vectorization, the same objective runs in 2.99 ms/query at N=200, K=20.

```bash
python efficiency/benchmark_full_pipeline.py
# → results/runtime/runtime_full.json

# Verify search() and search_vec() produce bit-identical outputs (float32):
python efficiency/test_vec_correctness.py
```

### 3. Fixed-recall analysis (rebuttal §"Sensitivity, Dataset Scope, and Fixed-Recall Analysis")

Max DM achievable while holding R@10 above a floor (0.95 / 0.90 / 0.85).

```bash
python extensions/build_fixed_recall_tables.py
# → results/fixed_recall/fixed_recall_diversity.md
```

### 4. CLIP backbone generalization

Reproduces the MS-DPP decrease-collapse and MASCOT recall-preservation results with CLIP ViT-L/14 instead of BLIP-2.

```bash
python examples/clip_gridsearch.py
# → results/pp_clip/SUMMARY.md
```

### Note on runtime vs. accuracy

`search_vec()` is numerically identical to the original `search()` in float32 (verified in `efficiency/test_vec_correctness.py`). Accuracy results reported in the paper use `search()`; the 2.99 ms number is measured on `search_vec()`.

---

## Adding New Methods

MASCOT integrates into the MS-DPP plugin registry:

```python
from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod, DivDir
import torch

@registry.register_div_method("your_method")
class YourMethod(BaseDiversificationMethod):
    def __init__(self, theta: float = 0.5, **kwargs):
        self.theta = theta

    def search(self, info: torch.Tensor, direction: DivDir,
               t2i_sim: torch.Tensor, top_k: int = 20, **kwargs) -> torch.Tensor:
        # info: [N, num_bins] — soft bin probability matrix
        # t2i_sim: [N] — raw VLM relevance scores
        # direction: DivDir.INCREASE or DivDir.DECREASE
        ...
        return torch.tensor(selected_indices)

    @property
    def get_params(self) -> dict:
        return {"theta": self.theta}

    @property
    def get_name(self) -> str:
        return "your_method"
```

---

## Citation

If you use MASCOT in your research, please cite:

```bibtex
@inproceedings{sharma2026mascot,
  title={{MASCOT: Model-Aware Submodular Coverage for Composite-Attribute Text-to-Image Retrieval}},
  author={Sharma, Aaryan and Prasad C, Vishak and Singh, Virendra and Ramakrishnan, Ganesh},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia (ACM MM '26)},
  year={2026}
}
```

Also cite the MS-DPP baseline and codebase this work builds on:

```bibtex
@inproceedings{sogi2025msdpp,
  author={Sogi, Naoya and Shibata, Takashi and Terao, Makoto and Suganuma, Masanori and Okatani, Takayuki},
  title={{MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text-to-Image Retrieval}},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025},
}
```

---

## License

See the [LICENSE](LICENSE) file for details.
