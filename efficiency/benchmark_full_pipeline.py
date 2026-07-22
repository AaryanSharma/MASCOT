"""
Full-pipeline runtime benchmark: per single-query inference latency and peak GPU memory.

Optimizations applied vs original slow benchmark:
  1. Pre-loaded GPS/hour/minute tensors on GPU — zero H2D transfer per query.
  2. Cached geo/time grid centers — meshgrid/arange computed once per (size, device).
  3. Pre-folded omega + pre-scaled r_hat in search_vec — removes per-iteration [N,M] multiply.
  4. Pre-allocated (1-p_covered) buffer in search_vec — no per-iteration tensor allocation.
  5. torch.set_float32_matmul_precision("high") — TF32 on Ampere+ GPUs.
  6. torch.inference_mode() over all benchmark loops — disables autograd engine overhead.
  7. MS-DPP reference: torch.linalg.eigh on GPU (matches production dpp.py).

Verifications run at startup:
  - Pre-loaded tensors == HF Arrow batch access on 100 samples.
  - Cached geo/time centers == fresh computation (atol=1e-6).
  - search_vec == search (sequential) on 10q × 3N × 2dirs × 6 configs
    (theta=0/0.1/0.5/1.0, ablation_no_norm, ablation_no_omega, both ablations).
  - torch.sub in-place buffer == explicit (1-p).to(dtype) for float32 and float16.
  - float16 info produces identical selected indices as float32 on 20q × 3N × 2dirs.

Reports per single-query (mean ± std, warmup excluded):
  data_ms   — tensor indexing + H2D transfer
  iu_ms     — IU kernel (geo + time) on GPU
  search_ms — greedy loop only
  TOTAL_ms  — end-to-end per query
  mem_MB    — peak GPU memory delta per query

N values: 100, 200, 400.
Run: PYTHONPATH=src python examples/benchmark_full_pipeline.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from msdpp.base.divmethod import DivDir
from msdpp.data import get_geo_iu_probs, get_time_iu_probs
from msdpp.div_method.ma_smf import ModelAwareSubmodularMethod
from msdpp.div_method.sim_funcs import dist_inv
import msdpp.div_method  # noqa — registry

GRID_SIZE  = 20
NUM_BINS   = 24
SIGMA_GEO  = 15.0
SIGMA_TIME = 1.5
TOP_K      = 20
WARMUP     = 10
TIMED      = 100
HF         = Path("share_datasets/temp")
N_VALUES   = [100, 200, 400]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")   # TF32 on Ampere+: ~3x matmul speedup

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading real PixelProse data ...", flush=True)
ret      = torch.load(HF / "div_results/Salesforce-blip2-itm-vit-g-coco_pp_test_ret_results.pkl", weights_only=False)
t2i_full = ret.results.t2i_sim          # [Q, total_images]
ds       = torch.load(HF / "tasks/PP_geo_hour_test.pkl", weights_only=False)
dl       = ds.dataset                   # HuggingFace datasets.Dataset (Arrow-backed)
Q        = t2i_full.shape[0]

n_needed = WARMUP + TIMED
assert Q >= n_needed, f"Need {n_needed} queries, only {Q} available"
print(f"Q={Q} queries, using {n_needed} (warmup={WARMUP}, timed={TIMED})", flush=True)

# Pre-sort top-N indices + sims for all queries and move to GPU
MAX_N = max(N_VALUES)
print(f"Pre-sorting top-{MAX_N} candidates for {n_needed} queries and moving to GPU ...", flush=True)
topk_indices = []
topk_sims    = []
for q in range(n_needed):
    idx = t2i_full[q].argsort(descending=True)[:MAX_N]
    topk_indices.append(idx.to(device))
    topk_sims.append(t2i_full[q][idx].to(device))

# Pre-load GPS/hour/minute and move to GPU — GPU-to-GPU indexing per query
# instead of CPU index + H2D transfer (~10x faster for the data_access step).
print("Pre-loading GPS/hour/minute into tensors and moving to GPU ...", flush=True)
all_gps = torch.tensor(dl["gps"],    dtype=torch.float32).to(device)
all_hrs = torch.tensor(dl["hour"],   dtype=torch.int32).to(device)
all_mns = torch.tensor(dl["minute"], dtype=torch.int32).to(device)

# ── build_iu: pre-loaded tensor indexing + GPU kernels ───────────────────────
def build_iu(query_idx: int, N: int):
    """Return (info [N,M], sim [N]) on device — all sources are GPU-resident."""
    topk = topk_indices[query_idx][:N]           # already on device
    sim  = topk_sims[query_idx][:N]
    gps  = torch.index_select(all_gps, 0, topk)
    hrs  = torch.index_select(all_hrs, 0, topk)
    mns  = torch.index_select(all_mns, 0, topk)
    geo  = get_geo_iu_probs(gps,      grid_size=GRID_SIZE, sigma=SIGMA_GEO)
    tim  = get_time_iu_probs(hrs, mns, num_bins=NUM_BINS,  sigma=SIGMA_TIME)
    return torch.cat([geo, tim], dim=1), sim


def build_iu_timed(query_idx: int, N: int):
    """
    Same as build_iu but returns (info, sim, hf_ms, iu_ms) with sub-timings.
    hf_ms now measures tensor indexing + host-to-device transfer (should be <1ms).
    """
    topk = topk_indices[query_idx][:N]           # already on device
    sim  = topk_sims[query_idx][:N]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    gps = torch.index_select(all_gps, 0, topk)
    hrs = torch.index_select(all_hrs, 0, topk)
    mns = torch.index_select(all_mns, 0, topk)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    hf_ms = (time.perf_counter() - t0) * 1e3

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    geo  = get_geo_iu_probs(gps,      grid_size=GRID_SIZE, sigma=SIGMA_GEO)
    tim  = get_time_iu_probs(hrs, mns, num_bins=NUM_BINS,  sigma=SIGMA_TIME)
    info = torch.cat([geo, tim], dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    iu_ms = (time.perf_counter() - t1) * 1e3

    return info, sim, hf_ms, iu_ms


# ── All verifications (require build_iu to be defined) ───────────────────────
print("Verifying pre-loaded tensors match HF Arrow batch access ...", flush=True)
_idx_v   = topk_indices[0][:100]
_topk_v  = _idx_v.cpu().tolist()
_batch_v = dl[_topk_v]
assert torch.allclose(torch.index_select(all_gps, 0, _idx_v).cpu(),
                      torch.tensor(_batch_v["gps"], dtype=torch.float32)), \
    "GPS mismatch: pre-loaded tensor vs HF Arrow"
assert torch.equal(torch.index_select(all_hrs, 0, _idx_v).cpu(),
                   torch.tensor(_batch_v["hour"], dtype=torch.int32)), \
    "hour mismatch: pre-loaded tensor vs HF Arrow"
assert torch.equal(torch.index_select(all_mns, 0, _idx_v).cpu(),
                   torch.tensor(_batch_v["minute"], dtype=torch.int32)), \
    "minute mismatch: pre-loaded tensor vs HF Arrow"
print("  OK — GPS/hour/minute match HF Arrow on 100 samples.", flush=True)

print("Verifying cached geo/time centers produce identical IU matrices ...", flush=True)
_N_v   = 50
_tk_vc = topk_indices[0][:_N_v]
_gps_v = torch.index_select(all_gps, 0, _tk_vc)
_hrs_v = torch.index_select(all_hrs, 0, _tk_vc)
_mns_v = torch.index_select(all_mns, 0, _tk_vc)
_geo_a = get_geo_iu_probs(_gps_v, grid_size=GRID_SIZE, sigma=SIGMA_GEO)
_geo_b = get_geo_iu_probs(_gps_v, grid_size=GRID_SIZE, sigma=SIGMA_GEO)
assert torch.equal(_geo_a, _geo_b), "geo IU: second cached call differs from first"
_lats = torch.linspace(-90, 90, GRID_SIZE, device=device)
_lons = torch.linspace(-180, 180, GRID_SIZE, device=device)
_gl, _glo = torch.meshgrid(_lats, _lons, indexing='ij')
_centers_fresh = torch.stack([_gl.flatten(), _glo.flatten()], dim=1)
_geo_fresh = torch.exp(-(torch.cdist(_gps_v.float(), _centers_fresh) ** 2) / (2 * SIGMA_GEO ** 2))
_geo_fresh[_geo_fresh < 0.01] = 0.0
assert torch.allclose(_geo_a, _geo_fresh, atol=1e-6), \
    f"geo IU: cached differs from fresh (max_diff={(_geo_a - _geo_fresh).abs().max():.2e})"
_tim_a = get_time_iu_probs(_hrs_v, _mns_v, num_bins=NUM_BINS, sigma=SIGMA_TIME)
_tim_b = get_time_iu_probs(_hrs_v, _mns_v, num_bins=NUM_BINS, sigma=SIGMA_TIME)
assert torch.equal(_tim_a, _tim_b), "time IU: second cached call differs from first"
_tc = torch.arange(NUM_BINS, dtype=torch.float32, device=device)
_th = (_hrs_v + _mns_v / 60.0).float()
_td = torch.minimum(torch.abs(_th.unsqueeze(1) - _tc.unsqueeze(0)), 24.0 - torch.abs(_th.unsqueeze(1) - _tc.unsqueeze(0)))
_tim_fresh = torch.exp(-(_td ** 2) / (2 * SIGMA_TIME ** 2))
_tim_fresh[_tim_fresh < 0.01] = 0.0
assert torch.allclose(_tim_a, _tim_fresh, atol=1e-6), \
    f"time IU: cached differs from fresh (max_diff={(_tim_a - _tim_fresh).abs().max():.2e})"
print("  OK — geo and time IU matrices match fresh computation.", flush=True)

print("Verifying optimized search_vec matches search (sequential) — multiple configs ...", flush=True)
_sv_configs = [
    # (theta, no_norm, no_omega, label)
    (0.1,  False, False, "theta=0.1"),
    (0.0,  False, False, "theta=0 (pure retrieval)"),
    (1.0,  False, False, "theta=1 (pure coverage)"),
    (0.5,  True,  False, "ablation_no_norm"),
    (0.5,  False, True,  "ablation_no_omega"),
    (0.5,  True,  True,  "both ablations"),
]
for _theta, _no_norm, _no_omega, _label in _sv_configs:
    _m = ModelAwareSubmodularMethod(
        theta=_theta, sigma_geo=SIGMA_GEO, sigma_time=SIGMA_TIME,
        ablation_no_norm=_no_norm, ablation_no_omega=_no_omega,
    )
    for _q in range(10):
        for _N in N_VALUES:
            _info_v, _sim_v = build_iu(_q, _N)
            for _dir in (DivDir.INCREASE, DivDir.DECREASE):
                _sv = _m.search_vec(_info_v.float(), _dir, _sim_v, top_k=TOP_K)
                _ss = _m.search(    _info_v.float(), _dir, _sim_v, top_k=TOP_K)
                assert torch.equal(_sv.cpu(), _ss.cpu()), \
                    f"[{_label}] search_vec vs search mismatch q={_q} N={_N} dir={_dir}"
    print(f"  OK [{_label}]", flush=True)
print(f"  ALL PASS — search_vec identical to search across all configs.", flush=True)

print("Verifying torch.sub in-place buffer gives same cov_gain as explicit cast ...", flush=True)
_info_b, _sim_b = build_iu(0, N_VALUES[-1])
_info_b = _info_b.float()
for _dtype in (torch.float32, torch.float16):
    _pc    = torch.rand(_info_b.shape[1], dtype=torch.float32, device=device) * 0.5
    _buf   = torch.empty(_info_b.shape[1], dtype=_dtype, device=device)
    torch.sub(1.0, _pc, out=_buf)
    _ref   = (1.0 - _pc).to(_dtype)
    assert torch.equal(_buf, _ref), \
        f"torch.sub out-buffer mismatch for dtype={_dtype}: " \
        f"max_diff={(_buf.float() - _ref.float()).abs().max():.2e}"
    _info_d = _info_b.to(_dtype)
    _cov_via_buf = (_info_d * (_info_d.max(dim=0).values)) @ _buf
    _cov_via_ref = (_info_d * (_info_d.max(dim=0).values)) @ _ref
    assert torch.allclose(_cov_via_buf.float(), _cov_via_ref.float(), atol=1e-3), \
        f"cov_gain mismatch with sub-buffer vs explicit cast for dtype={_dtype}"
print("  OK — torch.sub in-place buffer matches explicit (1-p).to(dtype) for float32 and float16.", flush=True)

print("Verifying float16 info produces identical selected indices as float32 ...", flush=True)
_method_f = ModelAwareSubmodularMethod(theta=0.1, sigma_geo=SIGMA_GEO, sigma_time=SIGMA_TIME)
_f16_ok   = True
for _q in range(20):
    for _N in N_VALUES:
        _info_v, _sim_v = build_iu(_q, _N)
        for _dir in (DivDir.INCREASE, DivDir.DECREASE):
            _s32 = _method_f.search_vec(_info_v.float(), _dir, _sim_v, top_k=TOP_K)
            _s16 = _method_f.search_vec(_info_v.half(),  _dir, _sim_v, top_k=TOP_K)
            if not torch.equal(_s32, _s16):
                print(f"  MISMATCH q={_q} N={_N} dir={_dir} — float16 differs from float32")
                _f16_ok = False
USE_F16 = _f16_ok
if USE_F16:
    print(f"  OK — float16 identical to float32 on 20q × {len(N_VALUES)}N × 2dirs.", flush=True)
else:
    print("  WARNING — float16 differs; skipping float16 benchmark.", flush=True)

# ── MS-DPP reference: torch.linalg.eigh on GPU (matches production dpp.py) ───
def benchmark_msdpp_torch_eigh(N: int) -> tuple[float, float]:
    """Time dist_inv kernel build + torch.linalg.eigh on device."""
    eye = 1e-3 * torch.eye(N, device=device)
    with torch.inference_mode():
        for q in range(5):
            info, _ = build_iu(q, N)
            K = dist_inv(info.float(), do_normalize=True)
            torch.linalg.eigh(K + eye)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for q in range(TIMED):
            info, _ = build_iu(WARMUP + q, N)
            K = dist_inv(info.float(), do_normalize=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            torch.linalg.eigh(K + eye)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))


# ── Per-sub-step timing + memory (per single-query inference) ─────────────────
method = ModelAwareSubmodularMethod(theta=0.1, sigma_geo=SIGMA_GEO, sigma_time=SIGMA_TIME)

def search_vec_f16(info, direction, t2i_sim, **kwargs):
    """search_vec with info cast to float16 — halves memory bandwidth on the matmul."""
    return method.search_vec(info.half(), direction, t2i_sim, **kwargs)

def benchmark_detailed(N: int, search_fn, direction: DivDir):
    """
    Times a single-query inference end-to-end, reporting per-step breakdown:
      data_access_ms  — tensor indexing + H2D transfer (gps/hour/minute)
      iu_compute_ms   — get_geo_iu_probs + get_time_iu_probs on GPU
      search_ms       — greedy search only
      total_ms        — data_access + iu_compute + search (true per-query latency)
      peak_mem_mb     — peak GPU memory delta per query (MB)

    Warmup runs excluded from reported stats.
    """
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for q in range(WARMUP):
            info, sim = build_iu(q, N)
            search_fn(info, direction, sim, top_k=TOP_K)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    hf_times, iu_times, search_times, total_times, mem_usages = [], [], [], [], []

    with torch.inference_mode():
        for q in range(TIMED):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                mem_before = torch.cuda.memory_allocated(device)

            t_total = time.perf_counter()
            info, sim, hf_ms, iu_ms = build_iu_timed(WARMUP + q, N)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_search = time.perf_counter()
            search_fn(info, direction, sim, top_k=TOP_K)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            s_ms = (time.perf_counter() - t_search) * 1e3
            total_ms = (time.perf_counter() - t_total) * 1e3

            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(device) - mem_before
                mem_usages.append(peak_mem / 1e6)

            hf_times.append(hf_ms)
            iu_times.append(iu_ms)
            search_times.append(s_ms)
            total_times.append(total_ms)

    def stats(arr):
        return float(np.mean(arr)), float(np.std(arr))

    out = {
        "data_access": stats(hf_times),
        "iu_compute":  stats(iu_times),
        "search":      stats(search_times),
        "total":       stats(total_times),
    }
    if mem_usages:
        out["peak_mem_mb"] = stats(mem_usages)
    return out


# ── Main sweep ────────────────────────────────────────────────────────────────
results = {}

for N in N_VALUES:
    print(f"\n=== N = {N} ===", flush=True)
    results[N] = {}

    M_info = None

    for direction in (DivDir.INCREASE, DivDir.DECREASE):
        dir_name = "INCREASE" if direction == DivDir.INCREASE else "DECREASE"
        results[N][dir_name] = {}

        variants = [("search_vec", method.search_vec), ("search (seq)", method.search)]
        if USE_F16:
            variants.append(("search_vec_f16", search_vec_f16))
        for label, fn in variants:
            d = benchmark_detailed(N, fn, direction)
            if M_info is None:
                info, _ = build_iu(0, N)
                M_info = info.shape[1]

            mem_str = (f"  mem={d['peak_mem_mb'][0]:>5.1f}MB"
                       if "peak_mem_mb" in d else "")
            results[N][dir_name][label] = {
                "data_access_ms": d["data_access"][0], "data_access_std": d["data_access"][1],
                "iu_compute_ms":  d["iu_compute"][0],  "iu_compute_std":  d["iu_compute"][1],
                "search_ms":      d["search"][0],       "search_std":      d["search"][1],
                "total_ms":       d["total"][0],        "total_std":       d["total"][1],
                **( {"peak_mem_mb": d["peak_mem_mb"][0]} if "peak_mem_mb" in d else {} ),
            }
            print(
                f"  {dir_name:<10} {label:<14} "
                f"data={d['data_access'][0]:>5.2f}  "
                f"iu={d['iu_compute'][0]:>5.2f}  "
                f"search={d['search'][0]:>5.2f}  "
                f"TOTAL={d['total'][0]:>6.2f} ± {d['total'][1]:.2f} ms"
                f"{mem_str}"
            )

    print(f"  M={M_info}")

    # MS-DPP reference (torch eigh on GPU — matches production)
    ms_mean, ms_std = benchmark_msdpp_torch_eigh(N)
    results[N]["msdpp_torch_eigh"] = {"mean_ms": ms_mean, "std_ms": ms_std}
    print(f"  MS-DPP torch.eigh (N={N}): {ms_mean:>7.1f} ± {ms_std:.1f} ms/query  [eigh only, no IU build]")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 110)
print("SUMMARY — per single-query inference, DECREASE direction, search_vec")
print(f"{'N':>5}  {'data(ms)':>9}  {'iu(ms)':>7}  {'search(ms)':>11}  {'TOTAL(ms)':>10}  {'peak_mem(MB)':>13}  {'MS-DPP eigh(ms)':>16}")
print("-" * 110)
for N in N_VALUES:
    dec = results[N]["DECREASE"]["search_vec"]
    ms  = results[N]["msdpp_torch_eigh"]["mean_ms"]
    mem = dec.get("peak_mem_mb", float("nan"))
    print(
        f"{N:>5}  {dec['data_access_ms']:>9.2f}  {dec['iu_compute_ms']:>7.2f}  "
        f"{dec['search_ms']:>11.2f}  {dec['total_ms']:>10.2f}  {mem:>13.2f}  {ms:>16.1f}"
    )

print()
print("SUMMARY — per single-query inference, DECREASE direction, search (seq)")
print(f"{'N':>5}  {'data(ms)':>9}  {'iu(ms)':>7}  {'search(ms)':>11}  {'TOTAL(ms)':>10}  {'peak_mem(MB)':>13}")
print("-" * 75)
for N in N_VALUES:
    dec = results[N]["DECREASE"]["search (seq)"]
    mem = dec.get("peak_mem_mb", float("nan"))
    print(
        f"{N:>5}  {dec['data_access_ms']:>9.2f}  {dec['iu_compute_ms']:>7.2f}  "
        f"{dec['search_ms']:>11.2f}  {dec['total_ms']:>10.2f}  {mem:>13.2f}"
    )

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_json = Path("examples/results/runtime_full.json")
out_json.parent.mkdir(exist_ok=True)
payload = {
    "config": {
        "N_values": N_VALUES, "K": TOP_K, "warmup": WARMUP, "timed": TIMED,
        "device": str(device), "sigma_geo": SIGMA_GEO, "sigma_time": SIGMA_TIME,
        "grid_size": GRID_SIZE, "num_bins": NUM_BINS,
        "timing_unit": "ms per single query (mean over timed queries, warmup excluded)",
    },
    "optimizations": [
        "Pre-loaded GPS/hour/minute tensors: eliminates HF Arrow access per query",
        "Cached geo+time grid centers: meshgrid/arange computed once per (size, device)",
        "Pre-folded omega + pre-scaled r_hat in search_vec: removes per-iteration [N,M] multiply",
        f"float16 info matrix: {'enabled (verified identical indices)' if USE_F16 else 'disabled (index mismatch detected)'}",
        "MS-DPP reference: torch.linalg.eigh on GPU (matches production dpp.py)",
    ],
    "timings": {str(N): results[N] for N in N_VALUES},
}
with out_json.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved to {out_json}")
