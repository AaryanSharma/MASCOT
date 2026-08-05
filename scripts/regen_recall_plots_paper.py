"""
Regenerate the 10 recall plots at paper-insertion size.

figsize=(6, 5); legend fontsize=8 outside axes; title/label/ticks scaled down.
ma_smf variants (MASCOT, Uniform Binning, w/o Normalization) load from
results/failure_cases_corrected/ (bug #1 fix — file with plain _ma_smf.json
now contains the true MASCOT variant). All other methods load from their
respective directories in the msdpp working tree (unaffected by bug #1
because their filenames are unambiguous).

Overwrites the previous wide versions at:
  /tmp/MASCOT_work/results/figures_regenerated/pp/plot_recall_*.png
  /tmp/MASCOT_work/results/figures_regenerated/vg_i1m/plot_recall_*.png
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

MSDPP    = Path("/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp/results")
CORR     = Path("/tmp/MASCOT_work/results/failure_cases_corrected")
OUT_PP   = Path("/tmp/MASCOT_work/results/figures_regenerated/pp")
OUT_VGI  = Path("/tmp/MASCOT_work/results/figures_regenerated/vg_i1m")
OUT_PP.mkdir(parents=True, exist_ok=True)
OUT_VGI.mkdir(parents=True, exist_ok=True)

STYLE_MAP = {
    "MASCOT":            {"color": "#003f5c", "marker": "*", "ls": "-",  "lw": 2.5, "z": 10},
    "w/o Normalization": {"color": "#4292c6", "marker": "X", "ls": "--", "lw": 1.8, "z": 9},
    "Uniform Binning":   {"color": "#9ecae1", "marker": "P", "ls": "--", "lw": 1.8, "z": 9},
    "MS-DPP":            {"color": "#d73027", "marker": "s", "ls": "-",  "lw": 1.8, "z": 8},
    "MS-DPP + TN":       {"color": "#fc8d59", "marker": "^", "ls": "-.", "lw": 1.5, "z": 7},
    "MS-DPP+TN+TVMS":    {"color": "#fee090", "marker": "D", "ls": "-.", "lw": 1.5, "z": 7},
    "Prob-Coverage":     {"color": "#756bb1", "marker": "o", "ls": "-",  "lw": 1.5, "z": 6},
    "k-DPP":             {"color": "#8c564b", "marker": "v", "ls": ":",  "lw": 1.5, "z": 5},
    "MMR":               {"color": "#e377c2", "marker": "d", "ls": ":",  "lw": 1.5, "z": 4},
    "Clustering":        {"color": "#bcbd22", "marker": "h", "ls": ":",  "lw": 1.5, "z": 4},
    "BLIP-2":            {"color": "#525252", "marker": "x", "ls": "--", "lw": 1.5, "z": 3},
}
DETAILED_METHOD_MAP = {
    "msdpp_tn_tvms": "MS-DPP+TN+TVMS",
    "msdpp_tn":      "MS-DPP + TN",
    "msdpp":         "MS-DPP",
    "prob_coverage": "Prob-Coverage",
    "dpp_sim_average": "k-DPP",
    "ma_smf_ablation_no_norm":  "w/o Normalization",
    "ma_smf_ablation_no_omega": "Uniform Binning",
    "ma_smf":        "MASCOT",
    "blip2":         "BLIP-2",
    "mmr":           "MMR",
    "clustering":    "Clustering",
}
RANKS = [1, 3, 5, 10, 20]

def recall_at_k(cases, ranks):
    total = len(cases)
    if total == 0: return {k: 0.0 for k in ranks}
    hits = {k: 0 for k in ranks}
    for c in cases:
        gt = set(c.get("ground_truth", []))
        retr = c.get("top_retrieved", c.get("retrieved", []))
        for k in ranks:
            if any(x in gt for x in retr[:k]):
                hits[k] += 1
    return {k: hits[k] / total for k in ranks}

def method_from_stem(stem: str):
    parts = stem.split("_")
    if "increase" in parts:   idx = parts.index("increase")
    elif "decrease" in parts: idx = parts.index("decrease")
    else: return None, None
    dataset = "_".join(parts[2:idx+1])
    method_key = "_".join(parts[idx+1:])
    for k in sorted(DETAILED_METHOD_MAP.keys(), key=len, reverse=True):
        if k in method_key:
            return dataset, DETAILED_METHOD_MAP[k]
    return dataset, None

def collect_recall(base_dir: Path, corrected_dir: Path, dataset_filter=None):
    """dataset_filter: e.g. lambda ds: ds.startswith('PP_')"""
    results = defaultdict(dict)
    # Non-ma_smf from base_dir
    for f in base_dir.glob("failure_cases_*.json"):
        ds, method = method_from_stem(f.stem)
        if not method or method in ("MASCOT", "Uniform Binning", "w/o Normalization"):
            continue
        if dataset_filter and not dataset_filter(ds):
            continue
        with open(f) as fh: cases = json.load(fh)
        results[ds][method] = recall_at_k(cases, RANKS)
    # ma_smf variants from corrected_dir
    for f in corrected_dir.glob("failure_cases_*_ma_smf*.json"):
        ds, method = method_from_stem(f.stem)
        if not method or method not in ("MASCOT", "Uniform Binning", "w/o Normalization"):
            continue
        if dataset_filter and not dataset_filter(ds):
            continue
        with open(f) as fh: cases = json.load(fh)
        results[ds][method] = recall_at_k(cases, RANKS)
    return results

def plot_recall(dataset, methods, out_path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    order = sorted(methods.keys(), key=lambda m: STYLE_MAP.get(m, {"z": 0})["z"])
    for m in order:
        style = STYLE_MAP.get(m)
        if not style: continue
        vals = [methods[m][k] for k in RANKS]
        ax.plot(
            RANKS, vals, label=m,
            color=style["color"], marker=style["marker"], linestyle=style["ls"],
            linewidth=style["lw"], markersize=6,
            markeredgecolor="white", markeredgewidth=0.7, zorder=style["z"],
        )
    ax.set_xlabel("Rank Cutoff (k)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Recall @ k",       fontsize=9, fontweight="bold")
    ax.set_title(f"Early-Rank Retrieval Comparison: {dataset.upper()}",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(RANKS)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=True, shadow=False, borderpad=0.4,
              handlelength=1.6, handletextpad=0.5, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

# ---------- PP ----------
pp_results = collect_recall(
    base_dir      = MSDPP / "failure_cases_run2" / "failure_cases",
    corrected_dir = CORR,
    dataset_filter = lambda ds: ds.startswith("PP_"),
)
print(f"\nPP datasets: {sorted(pp_results.keys())}")
for ds, methods in sorted(pp_results.items()):
    plot_recall(ds, methods, OUT_PP / f"plot_recall_{ds}.png")

# ---------- VG / I1M ----------
for sub, prefix in [("vg", "VG_"), ("i1m", "I1M_")]:
    base = MSDPP / sub / "failure_cases"
    res  = collect_recall(base_dir=base, corrected_dir=CORR,
                          dataset_filter=lambda ds, p=prefix: ds.startswith(p))
    print(f"\n{sub.upper()} datasets: {sorted(res.keys())}")
    for ds, methods in sorted(res.items()):
        plot_recall(ds, methods, OUT_VGI / f"plot_recall_{ds}.png")

print("\ndone")
