"""
Generate qualitative figure like MS-DPP paper.
Layout per row: [img1 | img2 | img3 | polar-hist (hour tasks) | geo-heatmap (geo tasks)]
For geo_hour: both polar-hist AND geo-heatmap columns are shown.
Captions: horizontal (a)/(b)/… labels below each row.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter
from datasets import load_from_disk
from PIL import Image

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp/share_datasets/temp/self_datasets/pixelprose_filtered/val"
FAILURE_DIR    = Path("/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp/results/failure_cases_run2/failure_cases")
CORRECTED_DIR  = Path("/tmp/MASCOT_work/results/failure_cases_corrected")
SAVE_DIR       = Path("/tmp/MASCOT_work/results/figures_regenerated/qualitative")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─── LOAD DATASET METADATA ────────────────────────────────────────────────────
print("Loading dataset metadata …")
ds = load_from_disk(DATASET_PATH)
uid_to_meta = {e["uid"]: {"gps": e["gps"], "hour": e["hour"]} for e in ds}
print(f"  Loaded {len(uid_to_meta)} entries")

# ─── WORLD OUTLINE (geopandas + geodatasets if available, else grid lines) ────
try:
    import geopandas as gpd
    import geodatasets
    _world_gdf = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    HAS_GEOPANDAS = True
    print("  geopandas world outline loaded")
except Exception:
    _world_gdf = None
    HAS_GEOPANDAS = False
    print("  geopandas unavailable — using grid fallback")


# ─── PATH / IMAGE HELPERS ─────────────────────────────────────────────────────
def resolve_image_path(raw_path: str) -> str | None:
    if Path(raw_path).exists():
        return raw_path
    candidate = raw_path.replace("/msdpp/", "/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp/")
    if Path(candidate).exists():
        return candidate
    return None


def load_image(raw_path: str):
    resolved = resolve_image_path(raw_path)
    if resolved is None:
        return None
    try:
        return Image.open(resolved).convert("RGB")
    except Exception:
        return None


def uid_from_path(img_path: str) -> str:
    return Path(img_path).stem.split("_")[0]


def get_meta(raw_path: str) -> dict:
    return uid_to_meta.get(uid_from_path(raw_path), {"gps": None, "hour": None})


# ─── POLAR HISTOGRAM ─────────────────────────────────────────────────────────

def draw_polar_histogram(ax, hours: list, title: str = "", highlight: list = None):
    """24-bin polar clock histogram.  highlight = hours of top-3 images."""
    bins = np.zeros(24)
    for h in hours:
        if h is not None:
            bins[int(h) % 24] += 1

    theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    width = 2 * np.pi / 24

    colors = ["#aec6e8"] * 24
    if highlight:
        for h in highlight:
            if h is not None:
                colors[int(h) % 24] = "#d73027"

    ax.bar(theta, bins, width=width * 0.85, bottom=0.0,
           color=colors, edgecolor="white", linewidth=0.5, align="edge")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=5)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=7, pad=3, fontweight="bold")
    ax.spines["polar"].set_visible(False)


# ─── GEO KDE HEATMAP ─────────────────────────────────────────────────────────

# Pre-build a world outline path list for fast reuse
_world_polys = None  # list of (lons, lats) arrays


def _get_world_polys():
    global _world_polys
    if _world_polys is not None:
        return _world_polys
    _world_polys = []
    if HAS_GEOPANDAS and _world_gdf is not None:
        for geom in _world_gdf.geometry:
            if geom is None:
                continue
            geom_type = geom.geom_type
            polys = list(geom.geoms) if "Multi" in geom_type else [geom]
            for poly in polys:
                try:
                    xs, ys = poly.exterior.xy
                    _world_polys.append((np.array(xs), np.array(ys)))
                except Exception:
                    pass
    return _world_polys


def draw_geo_heatmap(ax, coords: list, title: str = "", highlight: list = None):
    """
    Gaussian-KDE density heatmap on a lat/lon grid.
    coords : list of [lat, lon] or None
    highlight : list of [lat, lon] for top-3 stars
    """
    # ── Build density grid ──────────────────────────────────────────
    LON_BINS, LAT_BINS = 360, 180
    valid = [(c[1], c[0]) for c in coords if c is not None]  # (lon, lat)

    grid = np.zeros((LAT_BINS, LON_BINS), dtype=float)
    if valid:
        lons = np.array([v[0] for v in valid])
        lats = np.array([v[1] for v in valid])
        # Digitise into grid
        lon_idx = np.clip(((lons + 180) / 360 * LON_BINS).astype(int), 0, LON_BINS - 1)
        lat_idx = np.clip(((lats + 90)  / 180 * LAT_BINS).astype(int), 0, LAT_BINS - 1)
        for li, lni in zip(lat_idx, lon_idx):
            grid[li, lni] += 1
        grid = gaussian_filter(grid, sigma=5)

    # ── Background ocean colour ──────────────────────────────────────
    ax.set_facecolor("#d4eaf7")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")

    # ── Heatmap ─────────────────────────────────────────────────────
    if grid.max() > 0:
        grid_norm = grid / grid.max()
        # Custom colormap: transparent at 0, opaque orange-red at max
        cmap_colors = [
            (0.83, 0.92, 0.97, 0.0),   # transparent ocean
            (0.98, 0.85, 0.37, 0.55),   # light yellow
            (0.98, 0.55, 0.15, 0.80),   # orange
            (0.84, 0.10, 0.11, 0.95),   # deep red
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("geo_heat", cmap_colors)
        ax.imshow(
            grid_norm,
            origin="lower",
            extent=[-180, 180, -90, 90],
            aspect="auto",
            cmap=cmap,
            vmin=0, vmax=1,
            zorder=2,
            interpolation="bilinear",
        )

    # ── Land / coastlines ───────────────────────────────────────────
    polys = _get_world_polys()
    if polys:
        for xs, ys in polys:
            ax.fill(xs, ys, color="#e8e4d9", zorder=3, linewidth=0)
            ax.plot(xs, ys, color="#9a9a9a", linewidth=0.35, zorder=4)
    else:
        # Minimal grid fallback
        ax.set_facecolor("#d4eaf7")
        for lat in range(-60, 90, 30):
            ax.axhline(lat, color="#b8d4e8", linewidth=0.3, zorder=1)
        for lon in range(-120, 180, 60):
            ax.axvline(lon, color="#b8d4e8", linewidth=0.3, zorder=1)

    # ── Re-draw heatmap on top of land ──────────────────────────────
    if grid.max() > 0:
        cmap2_colors = [
            (0, 0, 0, 0.0),
            (0.98, 0.85, 0.37, 0.40),
            (0.98, 0.55, 0.15, 0.70),
            (0.84, 0.10, 0.11, 0.90),
        ]
        cmap2 = mcolors.LinearSegmentedColormap.from_list("geo_heat2", cmap2_colors)
        ax.imshow(
            grid_norm,
            origin="lower",
            extent=[-180, 180, -90, 90],
            aspect="auto",
            cmap=cmap2,
            vmin=0, vmax=1,
            zorder=5,
            interpolation="bilinear",
        )

    # ── Top-3 stars ─────────────────────────────────────────────────
    if highlight:
        h3 = [c for c in highlight if c is not None]
        if h3:
            ax.scatter(
                [c[1] for c in h3], [c[0] for c in h3],
                s=80, color="#ffff00", edgecolors="#222", linewidths=0.6,
                marker="*", zorder=8, alpha=1.0,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#888")
    ax.set_title(title, fontsize=7, pad=3, fontweight="bold")


# ─── QUERY SELECTION ──────────────────────────────────────────────────────────

def pick_good_query(cases_ms, cases_our, task_type: str, n_top=3, seed=3):
    """Shuffle and pick the first index where all top-n images load + have metadata."""
    rng = random.Random(seed)
    idxs = list(range(min(len(cases_ms), len(cases_our))))
    rng.shuffle(idxs)

    for i in idxs:
        paths = (cases_ms[i]["retrieved"][:n_top] +
                 cases_our[i]["retrieved"][:n_top])
        if not all(resolve_image_path(p) is not None for p in paths):
            continue
        metas = [get_meta(p) for p in paths]
        if "hour" in task_type and any(m["hour"] is None for m in metas):
            continue
        if "geo" in task_type and any(m["gps"] is None for m in metas):
            continue
        return i

    # Fallback: first index where images exist
    for i in range(min(len(cases_ms), len(cases_our))):
        paths = (cases_ms[i]["retrieved"][:n_top] +
                 cases_our[i]["retrieved"][:n_top])
        if all(resolve_image_path(p) is not None for p in paths):
            return i
    return 0


# ─── FIGURE GENERATION ───────────────────────────────────────────────────────

def _pool_field(cases_list: list, field: str) -> list:
    vals = []
    for c in (cases_list or []):
        for p in c.get("retrieved", []):
            vals.append(get_meta(p).get(field))
    return vals


def show_image(ax, raw_path: str, rank: int):
    img = load_image(raw_path)
    ax.axis("off")
    if img:
        ax.imshow(img)
    else:
        ax.set_facecolor("#ddd")
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#888")
    # Rank label above
    ax.text(0.5, 1.01, f"Rank {rank}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333")


def make_qualitative_figure(
    query: str,
    row_specs: list,     # list of dicts: {cases, all_cases, caption}
    query_idx: int,
    task_type: str,      # "hour" | "geo" | "geo_hour"
    save_path: Path,
    n_top: int = 3,
):
    """
    row_specs items:
      cases      – list of failure-case dicts (for this method/direction)
      all_cases  – full pool list for background distribution
      caption    – caption string e.g. "(a) MS-DPP — Div(Geo, Hour) ↑"

    Columns per row:
      hour:     img×3 | polar_hist
      geo:      img×3 | geo_heatmap
      geo_hour: img×3 | polar_hist | geo_heatmap
    """
    has_hour = "hour" in task_type
    has_geo  = "geo"  in task_type
    n_meta   = has_hour + has_geo           # 1 or 2
    n_cols   = n_top + n_meta
    n_rows   = len(row_specs)

    # Figure sizing: images are square-ish, meta panels slightly smaller
    img_w  = 2.9
    meta_w = 2.5
    fig_w  = img_w * n_top + meta_w * n_meta + 0.6
    fig_h  = 3.2 * n_rows + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)

    # Short query for title
    short_q = query.strip()
    if len(short_q) > 110:
        short_q = short_q[:107] + "…"
    fig.suptitle(f'Text query: "{short_q}"', fontsize=9, y=0.995,
                 fontweight="bold", ha="center")

    # Column widths: image cols have equal width, meta cols are slightly smaller
    col_widths = [img_w] * n_top + [meta_w] * n_meta
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        width_ratios=col_widths,
        hspace=0.55,
        wspace=0.06,
        left=0.02, right=0.99,
        top=0.955, bottom=0.04,
    )

    for row_i, spec in enumerate(row_specs):
        cases     = spec["cases"]
        all_cases = spec.get("all_cases") or cases
        caption   = spec["caption"]

        entry     = cases[query_idx]
        retrieved = entry["retrieved"][:n_top]
        top_metas = [get_meta(p) for p in retrieved]

        # ── Image columns ────────────────────────────────────────────
        center_ax = None
        for col_i, raw_path in enumerate(retrieved):
            ax = fig.add_subplot(gs[row_i, col_i])
            show_image(ax, raw_path, col_i + 1)
            if col_i == n_top // 2:
                center_ax = ax

        # ── Metadata columns ─────────────────────────────────────────
        meta_col = n_top  # first metadata column index

        if has_hour:
            ax_hour = fig.add_subplot(gs[row_i, meta_col], projection="polar")
            all_hours   = _pool_field(all_cases, "hour")
            top_hours   = [m["hour"] for m in top_metas]
            draw_polar_histogram(ax_hour, all_hours,
                                 title="Hour dist.", highlight=top_hours)
            meta_col += 1

        if has_geo:
            ax_geo = fig.add_subplot(gs[row_i, meta_col])
            all_gps   = _pool_field(all_cases, "gps")
            top_gps   = [m["gps"] for m in top_metas]
            draw_geo_heatmap(ax_geo, all_gps,
                             title="Location dist.", highlight=top_gps)

        # ── Caption below row (centered on middle image axes) ─────────
        if center_ax is not None:
            center_ax.text(
                0.5, -0.09, caption,
                transform=center_ax.transAxes,
                ha="center", va="top",
                fontsize=8.5, fontweight="bold", color="#111",
            )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ─── CAPTION HELPERS ─────────────────────────────────────────────────────────

_LABEL_LETTERS = "abcdefghijklmnop"

def make_caption(letter: str, method: str, task_str: str, direction: str) -> str:
    arrow = "↑" if direction == "inc" else "↓"
    return f"({letter}) {method} — Div({task_str}) {arrow}"


TASK_LABEL = {
    "hour":     "Time",
    "geo":      "Geo",
    "geo_hour": "Geo, Time",
}


# ─── MAIN ────────────────────────────────────────────────────────────────────

def load_cases(pattern: str, base: Path = FAILURE_DIR) -> list:
    p = base / pattern
    if not p.exists():
        print(f"    WARNING: {p} not found")
        return []
    with open(p) as f:
        return json.load(f)


CONFIGS = [
    # (task_type, inc_suffix, dec_suffix, figure_name)
    ("hour",     "PP_hour_increase",     "PP_hour_decrease",     "qualitative_hour"),
    ("geo",      "PP_geo_increase",      "PP_geo_decrease",      "qualitative_geo"),
    ("geo_hour", "PP_geo_hour_increase", "PP_geo_hour_decrease", "qualitative_geo_hour"),
]

for task_type, inc_sfx, dec_sfx, fig_name in CONFIGS:
    print(f"\n── Generating: {fig_name} ──")

    cases_inc_ms  = load_cases(f"failure_cases_{inc_sfx}_msdpp_tn_tvms.json")
    cases_dec_ms  = load_cases(f"failure_cases_{dec_sfx}_msdpp_tn_tvms.json")
    # ma_smf loads from corrected dir (bug #1 fix: filename now includes ablation suffix,
    # so plain _ma_smf.json is the true MASCOT variant with ablation_no_omega=False)
    cases_inc_our = load_cases(f"failure_cases_{inc_sfx}_ma_smf.json", base=CORRECTED_DIR)
    cases_dec_our = load_cases(f"failure_cases_{dec_sfx}_ma_smf.json", base=CORRECTED_DIR)

    if not all([cases_inc_ms, cases_dec_ms, cases_inc_our, cases_dec_our]):
        print("  Missing data — skipping.")
        continue

    n = min(len(cases_inc_ms), len(cases_dec_ms),
            len(cases_inc_our), len(cases_dec_our))
    cases_inc_ms  = cases_inc_ms[:n]
    cases_dec_ms  = cases_dec_ms[:n]
    cases_inc_our = cases_inc_our[:n]
    cases_dec_our = cases_dec_our[:n]

    # Pick a good query for both methods simultaneously
    qi_ms  = pick_good_query(cases_inc_ms,  cases_dec_ms,  task_type, seed=21)
    qi_our = pick_good_query(cases_inc_our, cases_dec_our, task_type, seed=21)
    # Use the same query index
    qi = qi_ms
    print(f"  Query index: {qi}")
    print(f"  Query: {cases_inc_ms[qi]['query'][:80]}…")

    def _bn(p): return Path(p).name
    # Also load buggy versions (pre-fix) for BEFORE/AFTER comparison at the same qi
    buggy_inc = load_cases(f"failure_cases_{inc_sfx}_ma_smf.json", base=FAILURE_DIR)
    buggy_dec = load_cases(f"failure_cases_{dec_sfx}_ma_smf.json", base=FAILURE_DIR)
    print(f"  ROW c ({fig_name} — MASCOT increase) top-3:")
    print(f"    BEFORE (buggy _ma_smf.json = Uniform Binning):")
    for i, p in enumerate(buggy_inc[qi]['retrieved'][:3]):
        print(f"      {i+1}. {_bn(p)}")
    print(f"    AFTER (corrected _ma_smf.json = true MASCOT):")
    for i, p in enumerate(cases_inc_our[qi]['retrieved'][:3]):
        print(f"      {i+1}. {_bn(p)}")
    print(f"  ROW d ({fig_name} — MASCOT decrease) top-3:")
    print(f"    BEFORE:")
    for i, p in enumerate(buggy_dec[qi]['retrieved'][:3]):
        print(f"      {i+1}. {_bn(p)}")
    print(f"    AFTER:")
    for i, p in enumerate(cases_dec_our[qi]['retrieved'][:3]):
        print(f"      {i+1}. {_bn(p)}")

    task_str = TASK_LABEL[task_type]
    row_specs = [
        {
            "cases":     cases_inc_ms,
            "all_cases": cases_inc_ms,
            "caption":   make_caption("a", "MS-DPP", task_str, "inc"),
        },
        {
            "cases":     cases_dec_ms,
            "all_cases": cases_dec_ms,
            "caption":   make_caption("b", "MS-DPP", task_str, "dec"),
        },
        {
            "cases":     cases_inc_our,
            "all_cases": cases_inc_our,
            "caption":   make_caption("c", "MASCOT", task_str, "inc"),
        },
        {
            "cases":     cases_dec_our,
            "all_cases": cases_dec_our,
            "caption":   make_caption("d", "MASCOT", task_str, "dec"),
        },
    ]

    make_qualitative_figure(
        query=cases_inc_ms[qi]["query"],
        row_specs=row_specs,
        query_idx=qi,
        task_type=task_type,
        save_path=SAVE_DIR / f"{fig_name}.png",
        n_top=3,
    )

print("\nDone.")
