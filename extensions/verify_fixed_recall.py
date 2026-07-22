"""
Comprehensive verification of fixed_recall_diversity.md against pkl cache.
Scans all BLIP-2 ma_smf index pkl files for PP_geo and PP_hour test splits,
computes r10 and mean_vendi for each operating point, then checks every
table cell in the MD against the actual best achievable values.

Only includes files with no_norm_False_no_omega_False (main method, not ablations),
and excludes old ambiguous short-format names (theta_0.X.pkl / theta_0.X_no_norm_False_no_omega_False.pkl
without sigma) — those are treated as sigma_geo=15.0, sigma_time=1.5.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

import os
_HF = Path(os.environ.get("HF_HOME", "share_datasets/temp"))
RERANK_DIR = Path(os.environ.get("MASCOT_RERANK_DIR", str(_HF / "div_results" / "rerank")))
BACKUP_DIR = Path(os.environ.get("MASCOT_RERANK_BACKUP_DIR", str(_HF.parent / "temp_backup" / "div_results" / "rerank")))

# Regex for new-format files: theta_X_sigma_geo_Y_sigma_time_Z_no_norm_False_no_omega_False
RE_FULL = re.compile(
    r"index_Salesforce-blip2-itm-vit-g-coco_pp_(geo|hour)_test_(increase|decrease)"
    r"_ma_smf_theta_([0-9.]+)_sigma_geo_([0-9.]+)_sigma_time_([0-9.]+)"
    r"_no_norm_False_no_omega_False\.pkl$"
)
# Regex for old-format without sigma: theta_X_no_norm_False_no_omega_False
RE_OLD_NOSIGMA = re.compile(
    r"index_Salesforce-blip2-itm-vit-g-coco_pp_(geo|hour)_test_(increase|decrease)"
    r"_ma_smf_theta_([0-9.]+)_no_norm_False_no_omega_False\.pkl$"
)


def load_file(path: Path):
    d = torch.load(path, weights_only=False)
    ei = d["eval"]
    r10 = float(ei.r10)
    dm  = float(ei.mean_vendi)
    return r10, dm


def collect_points(dataset: str) -> dict[str, list[tuple]]:
    """Returns {'increase': [(r10, dm, theta, sg, st, path), ...], 'decrease': [...]}"""
    results = {"increase": [], "decrease": []}

    for dirp in [RERANK_DIR, BACKUP_DIR]:
        if not dirp.exists():
            continue
        for f in dirp.glob(f"index_Salesforce-blip2-itm-vit-g-coco_pp_{dataset}_test_*.pkl"):
            fname = f.name
            m = RE_FULL.match(fname)
            if m:
                ds, direction, theta, sg, st = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5))
            else:
                m2 = RE_OLD_NOSIGMA.match(fname)
                if m2:
                    ds, direction, theta = m2.group(1), m2.group(2), float(m2.group(3))
                    sg, st = 15.0, 1.5  # defaults at time of creation
                else:
                    continue  # skip ablations and other formats
            try:
                r10, dm = load_file(f)
            except Exception as e:
                print(f"  WARN: could not load {f.name}: {e}")
                continue
            results[direction].append((r10, dm, theta, sg, st, f.name))

    return results


def best_dm_at_r10_floor(points, r10_floor: float):
    """Best (DM, theta, sg, st, r10) among points with r10 >= r10_floor."""
    valid = [(r10, dm, th, sg, st, name) for r10, dm, th, sg, st, name in points if r10 >= r10_floor]
    if not valid:
        return None
    return max(valid, key=lambda x: x[1])


def best_r10_at_dm_floor(points, dm_floor: float):
    """Best (r10, dm, theta, sg, st) among points with dm >= dm_floor."""
    valid = [(r10, dm, th, sg, st, name) for r10, dm, th, sg, st, name in points if dm >= dm_floor]
    if not valid:
        return None
    return max(valid, key=lambda x: x[0])


def fmt(val):
    if val is None:
        return "—"
    r10, dm, th, sg, st, _ = val
    return f"DM={dm:.4f} r10={r10:.4f} θ={th} σg={sg} σt={st}"


def check(label: str, md_val, actual_val, key: str):
    """Compare MD value to actual. key='dm' or 'r10'."""
    if actual_val is None:
        actual = None
    else:
        actual = actual_val[1] if key == "dm" else actual_val[0]

    if actual is None:
        if md_val != "—":
            print(f"  ❌ {label}: MD={md_val}, actual=— (not achievable)")
        else:
            print(f"  ✅ {label}: both —")
        return

    match = (abs(float(md_val) - actual) < 0.0005) if md_val != "—" else False
    sym = "✅" if match else "❌"
    detail = fmt(actual_val)
    print(f"  {sym} {label}: MD={md_val}, actual={actual:.4f}  [{detail}]")


# ─── PP_GEO ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("PP_GEO")
print("=" * 70)
geo = collect_points("geo")
print(f"  Points: increase={len(geo['increase'])}, decrease={len(geo['decrease'])}")

print("\n--- PP_geo DECREASE ---")
dec = geo["decrease"]
print("  Table A (best DM at r10 floor):")
check("R@10≥0.95", "0.3894", best_dm_at_r10_floor(dec, 0.95), "dm")
check("R@10≥0.90", "0.3894", best_dm_at_r10_floor(dec, 0.90), "dm")
check("R@10≥0.85", "0.3894", best_dm_at_r10_floor(dec, 0.85), "dm")
print("  Table B (best r10 at DM floor):")
check("DM≥0.35", "0.9737", best_r10_at_dm_floor(dec, 0.35), "r10")
check("DM≥0.40", "0.8105", best_r10_at_dm_floor(dec, 0.40), "r10")
check("DM≥0.45", "—",      best_r10_at_dm_floor(dec, 0.45), "r10")

# Also print the DM≥0.40 best point for the footnote check
best_0_40 = best_r10_at_dm_floor(dec, 0.40)
if best_0_40:
    r10, dm, th, sg, st, name = best_0_40
    print(f"  [DM≥0.40 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")

# DM≥0.35 footnote: MD says θ=0.10, σg=10, σt=0.5
best_0_35 = best_r10_at_dm_floor(dec, 0.35)
if best_0_35:
    r10, dm, th, sg, st, name = best_0_35
    print(f"  [DM≥0.35 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")

print("\n--- PP_geo INCREASE ---")
inc = geo["increase"]
print("  Table A (best DM at r10 floor):")
check("R@10≥0.95", "0.9338", best_dm_at_r10_floor(inc, 0.95), "dm")
check("R@10≥0.90", "0.9397", best_dm_at_r10_floor(inc, 0.90), "dm")
check("R@10≥0.85", "0.9419", best_dm_at_r10_floor(inc, 0.85), "dm")
print("  Table B (best r10 at DM floor):")
check("DM≥0.93", "0.9561", best_r10_at_dm_floor(inc, 0.93), "r10")
check("DM≥0.94", "0.8871", best_r10_at_dm_floor(inc, 0.94), "r10")
check("DM≥0.95", "—",      best_r10_at_dm_floor(inc, 0.95), "r10")

# Footnote details
b90 = best_dm_at_r10_floor(inc, 0.90)
if b90:
    r10, dm, th, sg, st, name = b90
    print(f"  [R@10≥0.90 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")
b94 = best_r10_at_dm_floor(inc, 0.94)
if b94:
    r10, dm, th, sg, st, name = b94
    print(f"  [DM≥0.94 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")


# ─── PP_HOUR ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PP_HOUR")
print("=" * 70)
hour = collect_points("hour")
print(f"  Points: increase={len(hour['increase'])}, decrease={len(hour['decrease'])}")

print("\n--- PP_hour DECREASE ---")
dec_h = hour["decrease"]
print("  Table A (best DM at r10 floor):")
check("R@10≥0.95", "0.3086", best_dm_at_r10_floor(dec_h, 0.95), "dm")
check("R@10≥0.90", "0.3570", best_dm_at_r10_floor(dec_h, 0.90), "dm")
check("R@10≥0.85", "0.3570", best_dm_at_r10_floor(dec_h, 0.85), "dm")
print("  Table B (best r10 at DM floor):")
check("DM≥0.35", "0.9059", best_r10_at_dm_floor(dec_h, 0.35), "r10")
check("DM≥0.40", "—",      best_r10_at_dm_floor(dec_h, 0.40), "r10")
check("DM≥0.45", "—",      best_r10_at_dm_floor(dec_h, 0.45), "r10")

# Footnote: R@10≥0.95 says θ=0.10, σg=15, σt=1.5
b95 = best_dm_at_r10_floor(dec_h, 0.95)
if b95:
    r10, dm, th, sg, st, name = b95
    print(f"  [R@10≥0.95 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")
# DM≥0.40 — print best even if not achievable
b40 = best_r10_at_dm_floor(dec_h, 0.40)
print(f"  [DM≥0.40 best found: {fmt(b40)}]")

print("\n--- PP_hour INCREASE ---")
inc_h = hour["increase"]
print("  Table A (best DM at r10 floor):")
check("R@10≥0.95", "0.8866", best_dm_at_r10_floor(inc_h, 0.95), "dm")
check("R@10≥0.90", "0.8910", best_dm_at_r10_floor(inc_h, 0.90), "dm")
check("R@10≥0.85", "0.8982", best_dm_at_r10_floor(inc_h, 0.85), "dm")
print("  Table B (best r10 at DM floor):")
check("DM≥0.88", "0.9699", best_r10_at_dm_floor(inc_h, 0.88), "r10")
check("DM≥0.90", "0.1330", best_r10_at_dm_floor(inc_h, 0.90), "r10")
check("DM≥0.92", "—",      best_r10_at_dm_floor(inc_h, 0.92), "r10")

# Footnotes
b95_h = best_dm_at_r10_floor(inc_h, 0.95)
if b95_h:
    r10, dm, th, sg, st, name = b95_h
    print(f"  [R@10≥0.95 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")
b90_h = best_dm_at_r10_floor(inc_h, 0.90)
if b90_h:
    r10, dm, th, sg, st, name = b90_h
    print(f"  [R@10≥0.90 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")
b85_h = best_dm_at_r10_floor(inc_h, 0.85)
if b85_h:
    r10, dm, th, sg, st, name = b85_h
    print(f"  [R@10≥0.85 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")
b90dm = best_r10_at_dm_floor(inc_h, 0.90)
print(f"  [DM≥0.90 best: {fmt(b90dm)}]")
b88dm = best_r10_at_dm_floor(inc_h, 0.88)
if b88dm:
    r10, dm, th, sg, st, name = b88dm
    print(f"  [DM≥0.88 footnote: θ={th}, σg={sg}, σt={st}  r10={r10:.4f}, DM={dm:.4f}]")

print("\nDone.")
