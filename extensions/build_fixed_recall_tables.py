"""
Build fixed-recall / fixed-diversity tables from all available operating points.

Data sources:
  PP_geo_hour: sensitivity θ sweep (σ_geo=10, σ_time=1.5) + pareto checkpoints
               (finer θ grid for ma_smf decrease) + pareto data for msdpp_tn_tvms.
               NOTE: Table 1 operating point θ=0.1, σ_geo=15, σ_time=3.0 is added
               manually from paper Table 1 numbers (R@10=0.9410, DM=0.1881).
  PP_geo:      Full θ × σ_geo × σ_time grid from pkl index files (via grid_data.json).
  PP_hour:     Full θ × σ_geo × σ_time grid from pkl index files (via grid_data.json).

Outputs:
  examples/results/fixed_recall_diversity.md  (overwrites previous version)
"""
import json
from pathlib import Path

GRID_DATA = Path("examples/results/grid_data.json")
PARETO_CKPT_DEC = Path("results/pareto/checkpoint.json")
PARETO_CKPT_INC = Path("results/pareto/checkpoint_increase.json")
SENS_DIR = Path("results/sensitivity")

# ─── load data ────────────────────────────────────────────────────────────────

with GRID_DATA.open() as f:
    grid = json.load(f)

with PARETO_CKPT_DEC.open() as f:
    ckpt_dec = json.load(f)      # ma_smf|theta0.05 → {r10, mean_vendi, ...}

with PARETO_CKPT_INC.open() as f:
    ckpt_inc = json.load(f)      # ma_smf|theta0.3 → {r10, mean_vendi, ...}

# Also load prob_coverage pareto checkpoints
PARETO_CKPT_PROB_DEC = Path("results/pareto/checkpoint.json")  # includes prob_coverage keys

# ─── helper: fixed-recall / fixed-diversity analysis ─────────────────────────

def best_dm_at_recall_floor(points, recall_floor):
    """Best DM achievable among points with r10 >= recall_floor."""
    eligible = [p for p in points if p["r10"] is not None and p["r10"] >= recall_floor]
    if not eligible:
        return None, None
    best = max(eligible, key=lambda p: p["mean_vendi"])
    return best["mean_vendi"], best

def best_r10_at_diversity_floor(points, dm_floor):
    """Best R@10 achievable among points with mean_vendi >= dm_floor."""
    eligible = [p for p in points if p["mean_vendi"] is not None and p["mean_vendi"] >= dm_floor]
    if not eligible:
        return None, None
    best = max(eligible, key=lambda p: p["r10"])
    return best["r10"], best

def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"

# ─── build PP_geo_hour operating points ───────────────────────────────────────

def build_pp_geo_hour_points(direction: str):
    """
    Merge all available PP_geo_hour operating points for ma_smf and prob_coverage.
    """
    dir_key = direction  # "increase" or "decrease"
    data = grid["PP_geo_hour"][dir_key]

    # 1. Sensitivity θ sweep (σ_geo=10, σ_time=1.5)
    ma_pts   = list(data.get("ma_smf", []))
    prob_pts = list(data.get("prob_coverage", []))

    # 2. Pareto checkpoints for ma_smf (finer θ grid for decrease, σ_geo=10, σ_time=1.5)
    ckpt = ckpt_dec if direction == "decrease" else ckpt_inc
    for key, val in ckpt.items():
        if key.startswith("ma_smf|"):
            try:
                theta = float(key.split("theta")[1])
            except Exception:
                continue
            # Avoid duplicating θ values already in sensitivity sweep
            if not any(abs(p["theta"] - theta) < 0.001 for p in ma_pts):
                ma_pts.append({
                    "theta": theta, "sigma_geo": 10.0, "sigma_time": 1.5,
                    "r10": val["r10"], "mean_vendi": val["mean_vendi"],
                })
        elif key.startswith("prob_coverage|"):
            try:
                theta = float(key.split("theta")[1])
            except Exception:
                continue
            if not any(abs(p["theta"] - theta) < 0.001 for p in prob_pts):
                prob_pts.append({
                    "theta": theta, "sigma_geo": 10.0, "sigma_time": 1.5,
                    "r10": val["r10"], "mean_vendi": val["mean_vendi"],
                })

    # 3. Table 1 operating point for ma_smf (θ=0.1, σ_geo=15, σ_time=3.0)
    # These are from the paper Table 1 / val-best grid search, not in any cached file.
    if direction == "decrease":
        table1_pt = {"theta": 0.1, "sigma_geo": 15.0, "sigma_time": 3.0,
                     "r10": 0.9410, "mean_vendi": 0.1881, "_source": "Table1"}
        # Add only if not already present (it won't be since σ_geo differs)
        ma_pts.append(table1_pt)
        # Also add best increase operating point from Table 1: θ=0.4, σ_geo=10, σ_time=1.5
    else:
        # Increase: Table 1 says θ=0.4, σ_geo=10, σ_time=1.5; that's already in sensitivity sweep
        pass

    # 4. msdpp_tn_tvms from pareto data
    tvms_pts = data.get("msdpp_tn_tvms", [])

    # 5. Blip2 baseline (θ=0)
    blip2 = data.get("blip2") or {"r10": 0.9737, "mean_vendi": 0.1656 if direction == "decrease" else 0.9142}

    return ma_pts, prob_pts, tvms_pts, blip2


def build_secondary_points(dataset: str, direction: str):
    """For PP_geo and PP_hour — use full grid from pkl cache."""
    data = grid[dataset][direction]
    ma_pts   = data.get("ma_smf", [])
    prob_pts = data.get("prob_coverage", [])
    blip2    = data.get("blip2")
    return ma_pts, prob_pts, blip2


# ─── table formatting helpers ─────────────────────────────────────────────────

def table_A(ma_pts, prob_pts, tvms_pts, blip2, recall_floors, direction):
    """Best achievable DM at fixed R@10 recall floor."""
    lines = []
    lines.append("| Method | " + " | ".join(f"R@10 ≥ {f:.2f}" for f in recall_floors) + " |")
    lines.append("|---|" + "|".join("---|" for _ in recall_floors))

    rows = [
        ("MASCOT (ma\\_smf)",     ma_pts),
        ("Both-Ablated (prob\\_cov)", prob_pts),
    ]
    if blip2:
        rows.append(("BLIP-2 (baseline)", [blip2]))
    if tvms_pts:
        rows.append(("MS-DPP-TN-TVMS", tvms_pts))

    notes = []
    for method_name, pts in rows:
        cells = []
        for floor in recall_floors:
            dm, best = best_dm_at_recall_floor(pts, floor)
            cells.append(fmt(dm))
            if best and method_name == "MASCOT (ma\\_smf)" and dm is not None:
                src = best.get("_source", "")
                sigma_note = f"σ_geo={best.get('sigma_geo')}, σ_time={best.get('sigma_time')}" if best.get("sigma_geo") else ""
                notes.append(f"*{method_name} at R@10 ≥ {floor:.2f}: θ={best.get('theta', best.get('config'))} {sigma_note} (r10={best['r10']:.4f}, DM={dm:.4f}){' [Table 1]' if src=='Table1' else ''}.*")
        lines.append(f"| {method_name} | " + " | ".join(cells) + " |")

    return "\n".join(lines), notes


def table_B(ma_pts, prob_pts, tvms_pts, blip2, dm_floors, direction):
    """Best achievable R@10 at fixed diversity floor."""
    lines = []
    lines.append("| Method | " + " | ".join(f"DM ≥ {f:.2f}" for f in dm_floors) + " |")
    lines.append("|---|" + "|".join("---|" for _ in dm_floors))

    rows = [
        ("MASCOT (ma\\_smf)",     ma_pts),
        ("Both-Ablated (prob\\_cov)", prob_pts),
        ("BLIP-2 (baseline)", [blip2] if blip2 else []),
    ]
    if tvms_pts:
        rows.append(("MS-DPP-TN-TVMS", tvms_pts))

    notes = []
    for method_name, pts in rows:
        cells = []
        for floor in dm_floors:
            r10, best = best_r10_at_diversity_floor(pts, floor)
            cells.append(fmt(r10))
        lines.append(f"| {method_name} | " + " | ".join(cells) + " |")

    return "\n".join(lines), notes


# ─── generate markdown ────────────────────────────────────────────────────────

sections = []
sections.append("# Fixed-Recall / Fixed-Diversity Comparison — PixelProse")
sections.append("""
> **Source:** PP_geo_hour uses sensitivity θ sweep (σ_geo=10, σ_time=1.5) plus Pareto checkpoints
> and the paper Table 1 operating point θ=0.1, σ_geo=15, σ_time=3.0 (labelled [Table 1]).
> PP_geo / PP_hour use full θ × σ_geo × σ_time grid from pkl index cache.
> DM = `mean_vendi` (decrease: ext_vendi flipped to 1−ext_vendi before harmonic mean).
> `—` = no operating point meets the threshold.
> **NOTE:** Full θ × σ grid for PP_geo_hour not in cache; to verify all sigma combos re-run grid_search_eval.py on PP_geo_hour.
""".strip())

# ─── PP_geo_hour ──────────────────────────────────────────────────────────────
sections.append("\n---\n\n## Dataset: PP_geo_hour *(headline)*")

for direction in ("decrease", "increase"):
    dir_label = direction.capitalize()
    sections.append(f"\n### {dir_label} Direction")

    ma_pts, prob_pts, tvms_pts, blip2 = build_pp_geo_hour_points(direction)

    if direction == "decrease":
        recall_floors = [0.95, 0.90, 0.85]
        dm_floors_B   = [0.15, 0.20, 0.25]
    else:
        recall_floors = [0.95, 0.90, 0.85]
        dm_floors_B   = [0.93, 0.94, 0.95]

    tA, notesA = table_A(ma_pts, prob_pts, tvms_pts, blip2, recall_floors, direction)
    sections.append("\n**Table A — Best achievable DM at fixed R@10 recall floor**\n")
    sections.append(tA)
    if notesA:
        sections.append("\n" + " ".join(notesA))

    tB, notesB = table_B(ma_pts, prob_pts, tvms_pts, blip2, dm_floors_B, direction)
    sections.append("\n**Table B — Best achievable R@10 at fixed diversity floor**\n")
    sections.append(tB)

# ─── PP_geo ───────────────────────────────────────────────────────────────────
sections.append("\n---\n\n## Dataset: PP_geo *(secondary)*")

for direction in ("decrease", "increase"):
    dir_label = direction.capitalize()
    sections.append(f"\n### {dir_label} Direction")

    ma_pts, prob_pts, blip2 = build_secondary_points("PP_geo", direction)
    tvms_pts = []  # not available for PP_geo

    if direction == "decrease":
        recall_floors = [0.95, 0.90, 0.85]
        dm_floors_B   = [0.15, 0.20, 0.25]
    else:
        recall_floors = [0.95, 0.90, 0.85]
        dm_floors_B   = [0.93, 0.94, 0.95]

    tA, notesA = table_A(ma_pts, prob_pts, tvms_pts, blip2, recall_floors, direction)
    sections.append("\n**Table A — Best achievable DM at fixed R@10 recall floor**\n")
    sections.append(tA)

    tB, notesB = table_B(ma_pts, prob_pts, tvms_pts, blip2, dm_floors_B, direction)
    sections.append("\n**Table B — Best achievable R@10 at fixed diversity floor**\n")
    sections.append(tB)

# ─── PP_hour ──────────────────────────────────────────────────────────────────
sections.append("\n---\n\n## Dataset: PP_hour *(secondary)*")

for direction in ("decrease", "increase"):
    dir_label = direction.capitalize()
    sections.append(f"\n### {dir_label} Direction")

    ma_pts, prob_pts, blip2 = build_secondary_points("PP_hour", direction)
    tvms_pts = []

    if direction == "decrease":
        recall_floors = [0.95, 0.90, 0.85]
        dm_floors_B   = [0.15, 0.20, 0.25]
    else:
        recall_floors = [0.95, 0.90, 0.85]
        dm_floors_B   = [0.93, 0.94, 0.95]

    tA, notesA = table_A(ma_pts, prob_pts, tvms_pts, blip2, recall_floors, direction)
    sections.append("\n**Table A — Best achievable DM at fixed R@10 recall floor**\n")
    sections.append(tA)

    tB, notesB = table_B(ma_pts, prob_pts, tvms_pts, blip2, dm_floors_B, direction)
    sections.append("\n**Table B — Best achievable R@10 at fixed diversity floor**\n")
    sections.append(tB)

# ─── PP_geo_hour summary ──────────────────────────────────────────────────────
sections.append("\n---\n\n## Summary for Rebuttal (PP_geo_hour, headline task)")

# Manually compile key numbers using Table 1 operating point for MASCOT decrease
sections.append("""
### Decrease direction — key numbers

**With Table 1 operating point (θ=0.1, σ_geo=15, σ_time=3.0) [Table 1]:**

At **R@10 ≥ 0.95** (near-lossless retrieval):
- MASCOT: best available θ=0.1, σ_geo=10, σ_time=1.5 → R@10=0.9686, DM=**0.1784** (+7.7% vs baseline 0.1656)
- Table 1 operating point (σ_geo=15, σ_time=3.0): R@10=0.9410 does **not** meet ≥ 0.95 floor
- BLIP-2 baseline: DM=0.1656

At **R@10 ≥ 0.90** (1% recall relaxation):
- MASCOT [Table 1]: R@10=0.9410 ≥ 0.90, DM=**0.1881** (+13.6% vs baseline) ← **Table 1 point qualifies here**
- MASCOT [σ_geo=10]: R@10=0.9686, DM=0.1784 (+7.7%)
- Both-Ablated: stuck at baseline 0.1656 (no operating point clears R@10 ≥ 0.90 with DM > baseline)

At **DM ≥ 0.20** (diversity floor):
- MASCOT [σ_geo=10]: best R@10=0.5157 at θ=0.2 — recall drops sharply above θ=0.1
- MS-DPP-TN-TVMS: best R@10=0.9021 — strongest at this floor

### Increase direction — key numbers (from sensitivity sweep)
""".strip())

ma_inc, prob_inc, tvms_inc, blip2_inc = build_pp_geo_hour_points("increase")
dm_95, _ = best_dm_at_recall_floor(ma_inc, 0.95)
r10_93, best93 = best_r10_at_diversity_floor(ma_inc, 0.93)
r10_93_prob, _ = best_r10_at_diversity_floor(prob_inc, 0.93)
r10_93_tvms, _ = best_r10_at_diversity_floor(tvms_inc, 0.93)
dm_95_base, _ = best_dm_at_recall_floor(blip2_inc and [blip2_inc] or [], 0.95)

inc_lines = []
inc_lines.append(f"At **R@10 ≥ 0.95**: MASCOT DM={fmt(dm_95)} vs BLIP-2 {fmt(dm_95_base)}")
inc_lines.append(f"At **DM ≥ 0.93**: MASCOT R@10={fmt(r10_93)}, Both-Ablated R@10={fmt(r10_93_prob)}, MS-DPP-TN-TVMS R@10={fmt(r10_93_tvms)}")
sections.append("\n".join(inc_lines))

# ─── write file ───────────────────────────────────────────────────────────────
out_path = Path("examples/results/fixed_recall_diversity.md")
out_path.write_text("\n".join(sections) + "\n")
print(f"Written to {out_path}")
