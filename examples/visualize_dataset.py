"""
Dataset sanity-check visualisation.

Shows a grid of sample images with their caption queries and shooting
hours, plus a histogram of the hour distribution across the full split.

Usage (run from the msdpp project root):
    # Flickr30K
    uv run python examples/visualize_dataset.py --dataset Flickr30K_hour

    # VG
    uv run python examples/visualize_dataset.py --dataset VG_hour

    # I1M
    uv run python examples/visualize_dataset.py --dataset I1M_geo

    # Custom split / n_samples
    uv run python examples/visualize_dataset.py --dataset Flickr30K_hour --split val --n 16

Output is saved to results/vis/<dataset>_<split>.png  (and shown if a
display is available).
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe; switch to "TkAgg" if you have a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from msdpp.schema import RetrievalDataset

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Dataset sanity-check visualisation")
parser.add_argument("--dataset", type=str, default="Flickr30K_hour",
                    help="Dataset base name, e.g. Flickr30K_hour / VG_hour / I1M_geo")
parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                    help="Which split to inspect (default: val)")
parser.add_argument("--n", type=int, default=12,
                    help="Number of sample images to show in the grid (default: 12)")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for sample selection")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Output directory (default: results/vis/)")
args = parser.parse_args()

HF_HOME = Path(os.environ.get("HF_HOME", "./"))
pkl_path = HF_HOME / "tasks" / f"{args.dataset}_{args.split}.pkl"

if not pkl_path.exists():
    print(f"ERROR: pkl not found at {pkl_path}")
    print("Run the preprocessing script first.")
    sys.exit(1)

print(f"Loading {pkl_path} ...")
ret_ds: RetrievalDataset = torch.load(pkl_path, weights_only=False)
ds = ret_ds.dataset
n_total = len(ds)
print(f"Dataset: {ret_ds.name}  |  {n_total} images  |  {len(ret_ds.retrieval_words)} queries")

# ------------------------------------------------------------------
# Detect metadata type: temporal (hour) or geographic (gps)
# ------------------------------------------------------------------
sample_row = ds[0]
has_hour = "hour" in sample_row
has_gps  = "gps"  in sample_row

ext = ret_ds.ext_data  # shape (N, 2) for time, (N, 3) for geo xyz

# ------------------------------------------------------------------
# Build per-image metadata label
# ------------------------------------------------------------------
def meta_label(i: int) -> str:
    row = ds[i]
    parts = []
    if has_hour:
        h, m = row["hour"] or 12, row.get("minute") or 0
        parts.append(f"{h:02d}:{m:02d}")
    if has_gps:
        gps = row["gps"]
        if gps:
            lat, lon = gps[0], gps[1]
            parts.append(f"({lat:.1f},{lon:.1f})")
    return " ".join(parts)


def meta_values() -> np.ndarray:
    """Return a 1-D array of the primary diversity attribute for histogram."""
    if has_hour:
        return np.array([(ds[i]["hour"] or 12) + (ds[i].get("minute") or 0) / 60.0 for i in range(n_total)])
    if has_gps:
        return np.array([ds[i]["gps"][1] for i in range(n_total)])  # longitude as proxy
    return np.zeros(n_total)


# ------------------------------------------------------------------
# Inverse-query map: image_idx → caption (for single-GT datasets)
# ------------------------------------------------------------------
idx_to_caption: dict[int, str] = {}
for q, label_vec in ret_ds.labels.items():
    pos = torch.where(label_vec == 1)[0].tolist()
    for idx in pos:
        if idx not in idx_to_caption:
            idx_to_caption[idx] = q


def caption_for(i: int) -> str:
    cap = idx_to_caption.get(i, "")
    # Wrap at 45 chars
    if len(cap) > 45:
        words = cap.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 44:
                lines.append(" ".join(line[:-1]))
                line = [w]
        lines.append(" ".join(line))
        cap = "\n".join(lines[:3])  # max 3 lines
    return cap


# ------------------------------------------------------------------
# Sample n images
# ------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
n_show = min(args.n, n_total)
indices = rng.choice(n_total, size=n_show, replace=False)
indices = sorted(indices.tolist())

# ------------------------------------------------------------------
# Figure layout:  top panel = image grid, bottom panel = histogram
# ------------------------------------------------------------------
cols = min(4, n_show)
rows = (n_show + cols - 1) // cols

fig = plt.figure(figsize=(cols * 3.5, rows * 3.8 + 3.5))
gs = gridspec.GridSpec(
    rows + 1, cols,
    figure=fig,
    hspace=0.55,
    wspace=0.25,
    height_ratios=[3.5] * rows + [3.0],
)

# --- Image grid ---
for plot_idx, ds_idx in enumerate(indices):
    r, c = divmod(plot_idx, cols)
    ax = fig.add_subplot(gs[r, c])

    # Load image
    raw_img = ds[ds_idx]["image"]
    if isinstance(raw_img, dict) and "bytes" in raw_img:
        import io
        pil_img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
    elif isinstance(raw_img, Image.Image):
        pil_img = raw_img.convert("RGB")
    else:
        pil_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

    ax.imshow(pil_img)
    ax.axis("off")

    meta = meta_label(ds_idx)
    caption = caption_for(ds_idx)

    title = f"[{meta}]\n{caption}" if caption else f"[{meta}]"
    ax.set_title(title, fontsize=6.5, pad=3, wrap=True,
                 verticalalignment="top")

# --- Histogram (bottom row, spanning all columns) ---
ax_hist = fig.add_subplot(gs[rows, :])
vals = meta_values()

if has_hour:
    bins = np.arange(0, 25) - 0.5          # 24 integer hour bins
    ax_hist.hist(vals, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax_hist.set_xlabel("Shooting hour (local time)", fontsize=9)
    ax_hist.set_xticks(range(0, 24))
    ax_hist.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=6.5, rotation=45)
    ax_hist.set_title(f"Hour distribution — {n_total} images", fontsize=10)

    # Annotate basic stats
    mean_h = float(np.mean(vals))
    std_h  = float(np.std(vals))
    unique_h = len(np.unique(vals.astype(int)))
    ax_hist.axvline(mean_h, color="crimson", linestyle="--", linewidth=1.2, label=f"mean={mean_h:.1f}h")
    ax_hist.legend(fontsize=8)
    info = f"unique hours: {unique_h}/24   std: {std_h:.2f}h"
    ax_hist.text(0.99, 0.97, info, transform=ax_hist.transAxes,
                 fontsize=8, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))

elif has_gps:
    ax_hist.hist(vals, bins=40, color="#55A868", edgecolor="white", linewidth=0.6)
    ax_hist.set_xlabel("Longitude", fontsize=9)
    ax_hist.set_title(f"Longitude distribution — {n_total} images", fontsize=10)

ax_hist.set_ylabel("Count", fontsize=9)
ax_hist.grid(axis="y", alpha=0.3)

# --- Figure title ---
fig.suptitle(
    f"{ret_ds.name}  ·  {args.split} split  ·  {n_total} images  ·  {len(ret_ds.retrieval_words)} queries",
    fontsize=11, y=1.0,
)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
out_dir = Path(args.out_dir) if args.out_dir else Path("results/vis")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{args.dataset}_{args.split}.png"

fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Saved → {out_path}")

# Print quick stats to terminal
print(f"\n--- Quick stats ({args.split}) ---")
print(f"  Total images : {n_total}")
print(f"  Total queries: {len(ret_ds.retrieval_words)}")
if has_hour:
    hour_ints = vals.astype(int)
    unique_hrs, counts = np.unique(hour_ints, return_counts=True)
    # Recover actual HH:MM from dataset for exact min/max display
    times_hm = [(ds[i]["hour"] or 12, ds[i].get("minute") or 0) for i in range(n_total)]
    min_t = min(times_hm); max_t = max(times_hm)
    common_h = unique_hrs[np.argmax(counts)]
    common_m_vals = [m for h, m in times_hm if h == common_h]
    common_m = int(np.mean(common_m_vals)) if common_m_vals else 0
    print(f"  Hours present: {len(unique_hrs)}/24  {unique_hrs.tolist()}")
    print(f"  Most common  : {common_h:02d}:{common_m:02d} ({counts.max()} images)")
    print(f"  Mean/Std     : {np.mean(vals):.1f}h / {np.std(vals):.2f}h")
    print(f"  Min/Max      : {min_t[0]:02d}:{min_t[1]:02d} / {max_t[0]:02d}:{max_t[1]:02d}")
if has_gps:
    lats = np.array([ds[i]["gps"][0] for i in range(n_total)])
    lons = np.array([ds[i]["gps"][1] for i in range(n_total)])
    print(f"  Lat range    : [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"  Lon range    : [{lons.min():.2f}, {lons.max():.2f}]")

# Sample captions
print(f"\n--- {min(5, len(ret_ds.retrieval_words))} sample queries ---")
for q in ret_ds.retrieval_words[:5]:
    n_pos = int(ret_ds.labels[q].sum().item())
    print(f"  [{n_pos} gt] {q[:80]}")
