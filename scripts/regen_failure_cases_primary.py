#!/usr/bin/env python3
"""
Regenerate correct MASCOT and Uniform Binning failure_cases_*.json files
for every task, using the FIXED pipeline (bugs #1 and #2 patched).

Uses Table 1/2 val-best hyperparameters (extracted from verified_old.log /
all_metrices.log). Writes to /tmp/plot_gen_v2/failure_cases/.

Run from the msdpp project root with:
    PYTHONPATH=./src ./.venv/bin/python /tmp/plot_gen_v2/regen_failure_cases.py
"""
import json
import logging
import sys
from pathlib import Path

import torch

# Preserve the msdpp/ working paths but replace ma_smf.py + task.py + retrieval_dataset.py
# with the FIXED versions from the paper repo before importing.
FIXED = Path("/tmp/MASCOT_work")
MSDPP = Path("/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp")

# Copy fixed files into msdpp/src BEFORE importing so we run the fix, not the bug
import shutil
shutil.copy(FIXED / "src/msdpp/task.py",              MSDPP / "src/msdpp/task.py")
shutil.copy(FIXED / "src/msdpp/div_method/ma_smf.py", MSDPP / "src/msdpp/div_method/ma_smf.py")
shutil.copy(FIXED / "src/msdpp/data.py",              MSDPP / "src/msdpp/data.py")
shutil.copy(FIXED / "src/msdpp/schema/retrieval_dataset.py", MSDPP / "src/msdpp/schema/retrieval_dataset.py")

sys.path.insert(0, str(MSDPP / "src"))

from msdpp import registry
from msdpp.base.divmethod import DivDir
from msdpp.models.blip2 import Blip2Model
from msdpp.task import BaseTask
import msdpp.div_method  # noqa: F401 registers all methods

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("regen")

# Table 1/2 val-best hyperparameters, from verified_old.log & all_metrices.log
# format: (dataset_name, direction, method_key, hyperparams_dict)
CONFIGS = [
    # PP_geo_hour
    ("PP_geo_hour", DivDir.DECREASE, "ma_smf",                     dict(theta=0.1, sigma_geo=15.0, sigma_time=3.0, ablation_no_norm=False, ablation_no_omega=False)),
    ("PP_geo_hour", DivDir.DECREASE, "ma_smf_ablation_no_omega",   dict(theta=0.7, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("PP_geo_hour", DivDir.INCREASE, "ma_smf",                     dict(theta=0.8, sigma_geo=15.0, sigma_time=1.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("PP_geo_hour", DivDir.INCREASE, "ma_smf_ablation_no_omega",   dict(theta=0.5, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    # PP_geo
    ("PP_geo",      DivDir.DECREASE, "ma_smf",                     dict(theta=0.3, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("PP_geo",      DivDir.DECREASE, "ma_smf_ablation_no_omega",   dict(theta=0.5, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("PP_geo",      DivDir.INCREASE, "ma_smf",                     dict(theta=0.8, sigma_geo=15.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("PP_geo",      DivDir.INCREASE, "ma_smf_ablation_no_omega",   dict(theta=0.5, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    # PP_hour
    ("PP_hour",     DivDir.DECREASE, "ma_smf",                     dict(theta=0.4, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("PP_hour",     DivDir.DECREASE, "ma_smf_ablation_no_omega",   dict(theta=0.7, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("PP_hour",     DivDir.INCREASE, "ma_smf",                     dict(theta=0.9, sigma_geo=1.0,  sigma_time=1.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("PP_hour",     DivDir.INCREASE, "ma_smf_ablation_no_omega",   dict(theta=0.7, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
]

HF = MSDPP / "share_datasets" / "temp"
OUT = Path("/tmp/plot_gen_v2/failure_cases")
OUT.mkdir(parents=True, exist_ok=True)

log.info("Loading BLIP-2...")
model = Blip2Model(pretrained_model="Salesforce/blip2-itm-vit-g-coco", use_itm=False)
log.info("BLIP-2 loaded.")

# Cache datasets to avoid re-loading
dataset_cache = {}

# Also emit sanity check on r10 per config
sanity = []

for (ds_name, direction, key, params) in CONFIGS:
    dir_str = "decrease" if direction == DivDir.DECREASE else "increase"
    log.info(f"=== {ds_name} / {dir_str} / {key} @ {params} ===")

    # Load dataset (cached)
    if ds_name not in dataset_cache:
        p = HF / "tasks" / f"{ds_name}_test.pkl"
        dataset_cache[ds_name] = torch.load(p, weights_only=False)
    dataset = dataset_cache[ds_name]

    task = BaseTask(model=model, dataset=dataset, subset_k=200, top_k=20, n_thread=4)
    task.verify_iu_cache(n_samples=3)

    div_instance = registry.get_div_method("ma_smf")(**params)

    # FORCE=True so we always compute fresh, don't hit the (now-fixed) cache
    result = task.run(div_instance, direction, force=True)

    # Extract per-query retrieved image paths
    n = len(dataset.retrieval_words)
    fc = []
    for i, q in enumerate(dataset.retrieval_words):
        gt_indices = torch.where(dataset.labels[q] == 1)[0].tolist()
        gt_imgs = [getattr(dataset.dataset[j]["image"], "filename", str(j)) for j in gt_indices]
        ret_imgs = [getattr(dataset.dataset[j]["image"], "filename", str(j)) for j in result.candidates[i].tolist()]
        fc.append({"query": q, "ground_truth": gt_imgs, "retrieved": ret_imgs})

    out_path = OUT / f"failure_cases_{ds_name}_{dir_str}_{key}.json"
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)

    # Sanity: recompute r10 from what we just wrote, compare to eval_indices
    ev = result.eval_indices
    r10_fc = sum(1 for c in fc if any(r in set(c["ground_truth"]) for r in c["retrieved"][:10])) / n
    r1_fc = sum(1 for c in fc if any(r in set(c["ground_truth"]) for r in c["retrieved"][:1])) / n
    log.info(f"  wrote {out_path.name}: n={n}, r1={r1_fc:.4f}, r10={r10_fc:.4f}  (eval r10={ev.r10.item():.4f}, r1={ev.r1.item():.4f})")
    sanity.append((ds_name, dir_str, key, params, r1_fc, r10_fc, ev.r10.item()))

log.info("\n=== SANITY TABLE ===")
for s in sanity:
    log.info(f"  {s[0]:12s} {s[1]:9s} {s[2]:30s}  r1={s[4]:.4f}  r10={s[5]:.4f}  (eval={s[6]:.4f})")
