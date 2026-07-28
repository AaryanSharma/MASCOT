#!/usr/bin/env python3
"""
Re-run λ (theta) sensitivity sweep at each task-direction's Table 1 val-best
σ_geo/σ_time. Fixes Appendix D's stale hardcoded DATASET_META, which fixed
σ per dataset (not per direction) and diverged from Table 1's optima.

Output: /tmp/plot_gen_v2/sensitivity_v2/{DATASET}/{DIR}/ma_smf_theta.json
"""
import json, logging, sys, shutil
from pathlib import Path
import torch

FIXED = Path("/tmp/MASCOT_work")
MSDPP = Path("/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp")
# Fixed source
shutil.copy(FIXED / "src/msdpp/task.py",              MSDPP / "src/msdpp/task.py")
shutil.copy(FIXED / "src/msdpp/div_method/ma_smf.py", MSDPP / "src/msdpp/div_method/ma_smf.py")
shutil.copy(FIXED / "src/msdpp/data.py",              MSDPP / "src/msdpp/data.py")
shutil.copy(FIXED / "src/msdpp/schema/retrieval_dataset.py", MSDPP / "src/msdpp/schema/retrieval_dataset.py")
sys.path.insert(0, str(MSDPP / "src"))

from msdpp import registry
from msdpp.base.divmethod import DivDir
from msdpp.models.blip2 import Blip2Model
from msdpp.task import BaseTask
import msdpp.div_method  # noqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("sens")

# Table 1/2 val-best σ per (task, direction). One sweep per row.
# thetas: 8 points including 0.0 (baseline)
THETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
import os
ALL_SWEEPS = [
    ("PP_geo",      DivDir.DECREASE, 10.0, 0.5),
    ("PP_geo",      DivDir.INCREASE, 15.0, 0.5),
    ("PP_hour",     DivDir.DECREASE,  1.0, 0.5),
    ("PP_hour",     DivDir.INCREASE,  1.0, 1.5),
    ("PP_geo_hour", DivDir.DECREASE, 15.0, 3.0),
    ("PP_geo_hour", DivDir.INCREASE, 15.0, 1.5),
]
# Shard via env var SHARD_IDX (0-indexed) out of SHARD_N (default 1)
_shard = int(os.environ.get("SHARD_IDX", "0"))
_nshards = int(os.environ.get("SHARD_N", "1"))
SWEEPS = [s for i, s in enumerate(ALL_SWEEPS) if i % _nshards == _shard]

HF = MSDPP / "share_datasets" / "temp"
OUT_ROOT = Path("/tmp/plot_gen_v2/sensitivity_v2")

log.info("Loading BLIP-2...")
model = Blip2Model(pretrained_model="Salesforce/blip2-itm-vit-g-coco", use_itm=False)
log.info("BLIP-2 loaded.")

dataset_cache = {}

for (ds_name, direction, sg, st) in SWEEPS:
    dir_str = "decrease" if direction == DivDir.DECREASE else "increase"
    out_dir = OUT_ROOT / ds_name / dir_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ma_smf_theta.json"
    log.info(f"=== SWEEP {ds_name}/{dir_str}  (fixed σg={sg}, σt={st}) ===")

    if ds_name not in dataset_cache:
        p = HF / "tasks" / f"{ds_name}_test.pkl"
        dataset_cache[ds_name] = torch.load(p, weights_only=False)
    dataset = dataset_cache[ds_name]
    task = BaseTask(model=model, dataset=dataset, subset_k=200, top_k=20, n_thread=4)

    sweep_data = []
    for theta in THETAS:
        div_instance = registry.get_div_method("ma_smf")(
            theta=theta, sigma_geo=sg, sigma_time=st,
            ablation_no_norm=False, ablation_no_omega=False,
        )
        result = task.run(div_instance, direction, force=True)
        ev = result.eval_indices
        sweep_data.append({
            "theta": theta,
            "r1":   ev.r1.item(),
            "r10":  ev.r10.item(),
            "img_vendi":  ev.img_vendi.item(),
            "ext_vendi":  ev.ext_vendi.item(),
            "mean_vendi": ev.mean_vendi.item(),
            "map_": ev.map_.item(),
        })
        log.info(f"  θ={theta}: r1={ev.r1.item():.4f} r10={ev.r10.item():.4f} DM={ev.ext_vendi.item():.4f}")

    with open(out_path, "w") as f:
        json.dump({
            "dataset":   ds_name,
            "direction": dir_str,
            "method":    "ma_smf",
            "axis":      "theta",
            "fixed":     {"sigma_geo": sg, "sigma_time": st, "grid_size": 20, "num_bins": 24},
            "sweep":     sweep_data,
        }, f, indent=2)
    log.info(f"  wrote {out_path}")
