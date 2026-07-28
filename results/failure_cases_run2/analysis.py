import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# --- FOLDER PATHS ---
TABLE_DIR = Path("tables")
RESULTS_DIR = Path("failure_cases") 
SAVE_DIR = Path("figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# MASTER STYLING DICTIONARY
# Groups related methods by color family and line style
# ==========================================
STYLE_MAP = {
    # OURS (Blues) - Thick lines, prominent markers
    "MASCOT":      {"color": "#003f5c", "marker": "*", "ls": "-",  "lw": 3.5, "s": 350, "z": 10},
    "w/o Normalization":  {"color": "#4292c6", "marker": "X", "ls": "--", "lw": 2.5, "s": 200, "z": 9},
    "Uniform Binning": {"color": "#9ecae1", "marker": "P", "ls": "--", "lw": 2.5, "s": 200, "z": 9},
    
    # MS-DPP FAMILY (Reds/Oranges)
    "MS-DPP":             {"color": "#d73027", "marker": "s", "ls": "-",  "lw": 2.5, "s": 150, "z": 8},
    "MS-DPP + TN":        {"color": "#fc8d59", "marker": "^", "ls": "-.", "lw": 2.0, "s": 150, "z": 7},
    "MS-DPP+TN+TVMS":     {"color": "#fee090", "marker": "D", "ls": "-.", "lw": 2.0, "s": 150, "z": 7},
    
    # OTHER BASELINES (Muted Colors)
    "Prob-Coverage":      {"color": "#756bb1", "marker": "o", "ls": "-",  "lw": 2.0, "s": 150, "z": 6},
    "k-DPP":              {"color": "#8c564b", "marker": "v", "ls": ":",  "lw": 2.0, "s": 150, "z": 5},
    "MMR":                {"color": "#e377c2", "marker": "d", "ls": ":",  "lw": 2.0, "s": 150, "z": 4},
    "Clustering":         {"color": "#bcbd22", "marker": "h", "ls": ":",  "lw": 2.0, "s": 150, "z": 4},
    
    # ANCHORS (Grays/Blacks)
    "BLIP-2":             {"color": "#525252", "marker": "x", "ls": "--", "lw": 2.0, "s": 150, "z": 3},
    "Original":           {"color": "#d9d9d9", "marker": "+", "ls": "-",  "lw": 1.5, "s": 200, "z": 1}
}

# ==========================================
# PART 1: PARETO FRONT (TRADE-OFF) PLOTTING
# ==========================================

METHOD_KEYS = {
    "09_dpp_sim_average": "k-DPP",
    "10_msdpp": "MS-DPP",
    "11_msdpp_tn": "MS-DPP + TN",
    "12_msdpp_tn_tvms": "MS-DPP+TN+TVMS", # Unified naming
    "13_prob_coverage": "Prob-Coverage",
    "14_ma_smf": "MASCOT",
    "15_ma_smf_ablation_no_norm": "w/o Normalization",
    "16_ma_smf_ablation_no_omega": "Uniform Binning",
    "17_blip2": "BLIP-2",
    "18_mmr": "MMR",
    "19_clustering": "Clustering"
}

if not TABLE_DIR.exists():
    print(f"Error: Could not find directory {TABLE_DIR.absolute()}")
else:
    # Set a clean Seaborn style for better background grids
    sns.set_theme(style="whitegrid")
    
    for table_path in TABLE_DIR.glob("PP_*.json"):
        with open(table_path) as f:
            table_json = json.load(f)

        table = table_json.get("results", {})
        if not table: continue

        dataset_name = table_path.stem
        fig, ax = plt.subplots(figsize=(10, 7))

        for key, label in METHOD_KEYS.items():
            if key not in table: continue
            
            entry = table[key]
            style = STYLE_MAP[label]
            alpha = 1.0 if "Ours" in label else 0.85

            ax.scatter(entry["div_index"], entry["ret_index"], label=label, 
                       color=style["color"], marker=style["marker"], 
                       s=style["s"], edgecolors="black", linewidths=1.5,
                       alpha=alpha, zorder=style["z"])

        if "00_org" in table:
            org = table["00_org"]
            style = STYLE_MAP["Original"]
            ax.scatter(org["div_index"], org["ret_index"], label="Original", 
                       color=style["color"], marker=style["marker"], 
                       s=style["s"], edgecolors="black", zorder=style["z"])

        ax.set_xlabel("Diversity Index (Mean Vendi)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Retrieval Index (R@10)", fontsize=13, fontweight='bold')
        
        direction = "Increase Diversity" if "increase" in dataset_name else "Decrease Diversity (Redundancy)"
        clean_name = dataset_name.replace("_increase", "").replace("_decrease", "").upper()
        ax.set_title(f"Retrieval–Diversity Trade-off: {clean_name}\n({direction})", fontsize=15, fontweight="bold")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True, shadow=True)
        plt.tight_layout()

        save_path = SAVE_DIR / f"plot_tradeoff_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Trade-off Plot: {save_path}")
        plt.close()

# ==========================================
# PART 2: EARLY-RANK RECALL (R@K) PLOTTING
# ==========================================

DETAILED_METHOD_MAP = {
    "msdpp_tn_tvms": "MS-DPP+TN+TVMS",
    "msdpp_tn": "MS-DPP + TN",
    "msdpp": "MS-DPP",
    "prob_coverage": "Prob-Coverage",
    "dpp_sim_average": "k-DPP",
    "ma_smf_ablation_no_norm": "w/o Normalization",
    "ma_smf_ablation_no_omega": "Uniform Binning",
    "ma_smf": "MASCOT",  
    "blip2": "BLIP-2",
    "mmr": "MMR",
    "clustering": "Clustering",
    "org": "Original"
}

RANKS = [1, 3, 5, 10, 20]

def compute_recall_at_k(cases, ranks):
    total = len(cases)
    if total == 0: return {k: 0.0 for k in ranks}
    hits = {k: 0 for k in ranks}
    for case in cases:
        gt = set(case.get("ground_truth", []))
        retrieved = case.get("top_retrieved", case.get("retrieved", []))
        for k in ranks:
            if any(img in gt for img in retrieved[:k]):
                hits[k] += 1
    return {k: hits[k] / total for k in ranks}

results = defaultdict(dict)

if not RESULTS_DIR.exists():
    print(f"Directory not found: {RESULTS_DIR.absolute()}")
else:
    for file in RESULTS_DIR.glob("failure_cases_*.json"):
        with open(file) as f: cases = json.load(f)

        parts = file.stem.split('_')
        if 'increase' in parts: idx = parts.index('increase')
        elif 'decrease' in parts: idx = parts.index('decrease')
        else: continue
        
        dataset = "_".join(parts[2:idx+1])
        method_key = "_".join(parts[idx+1:])

        method = None
        sorted_keys = sorted(DETAILED_METHOD_MAP.keys(), key=len, reverse=True)
        for k in sorted_keys:
            if k in method_key:
                method = DETAILED_METHOD_MAP[k]
                break
        
        if method: results[dataset][method] = compute_recall_at_k(cases, RANKS)

print("\n=== EARLY-RANK RECALL TABLES ===")
for dataset, methods in results.items():
    print(f"\nDataset: {dataset.upper()}")
    df = pd.DataFrame(methods).T
    if df.empty: continue
    df = df[RANKS]
    df.columns = [f"R@{k}" for k in RANKS]
    df = df.sort_index()
    print(df.to_string(float_format="%.4f"))
    print("-" * 60)

for dataset, methods in results.items():
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid")
    
    # Sort methods so 'Ours' plots last (on top of the others visually)
    sorted_methods = sorted(methods.keys(), key=lambda x: STYLE_MAP[x]['z'])

    for method in sorted_methods:
        recall = methods[method]
        vals = [recall[k] for k in RANKS]
        style = STYLE_MAP[method]
        
        # Add a subtle shadow/edge to markers for extra pop
        plt.plot(RANKS, vals, 
                 label=method, 
                 color=style["color"], 
                 marker=style["marker"], 
                 linestyle=style["ls"], 
                 linewidth=style["lw"], 
                 markersize=10 if "Ours" in method else 8,
                 markeredgecolor="white",
                 markeredgewidth=1.2,
                 zorder=style["z"])

    plt.xlabel("Rank Cutoff (k)", fontsize=13, fontweight='bold')
    plt.ylabel("Recall @ k", fontsize=13, fontweight='bold')
    plt.title(f"Early-Rank Retrieval Comparison: {dataset.upper()}", fontsize=15, fontweight='bold')
    plt.xticks(RANKS, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Force y-axis to be slightly taller than the max value so legends don't overlap lines
    plt.ylim(bottom=0.0, top=1.05) 
    
    # Clean up the legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()
    
    save_path = SAVE_DIR / f"plot_recall_{dataset}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved Recall Plot: {save_path}")
    plt.close()