#!/usr/bin/env python3
"""
Extended regen:
  - Fixes PP_geo_hour_dec UB (correct hyperparams: theta=0.7, sigma_geo=10, sigma_time=0.5)
  - Adds `ma_smf_ablation_no_norm` for all 6 PP tasks
  - Adds MASCOT + UB + no_norm for VG_hour and I1M_geo (both directions)
  = 19 configs total.

Skip-if-exists so it's safe to re-run.
"""
import json, logging, sys, shutil
from pathlib import Path
import torch

FIXED = Path("/tmp/MASCOT_work")
MSDPP = Path("/data1/aaryan_shivansh/ddp_aaryan_shiv/msdpp")
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
log = logging.getLogger("regen_ext")

# All targets. R@10 in comments = expected from tables/*.json.
CONFIGS = [
    # ── FIX for PP_geo_hour dec UB (r10=0.8118 expected) ─────────────
    ("PP_geo_hour", DivDir.DECREASE, "ma_smf_ablation_no_omega",   dict(theta=0.7, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    # ── PP no_norm rows (map/DM from Table 1/2) ──────────────────────
    ("PP_geo_hour", DivDir.DECREASE, "ma_smf_ablation_no_norm",  dict(theta=0.5, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),  # r10=0.2873
    ("PP_geo_hour", DivDir.INCREASE, "ma_smf_ablation_no_norm",  dict(theta=0.5, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),  # r10=0.8871
    ("PP_geo",      DivDir.DECREASE, "ma_smf_ablation_no_norm",  dict(theta=0.5, sigma_geo=5.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),  # r10=0.5947
    ("PP_geo",      DivDir.INCREASE, "ma_smf_ablation_no_norm",  dict(theta=0.9, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),  # r10=0.9398
    ("PP_hour",     DivDir.DECREASE, "ma_smf_ablation_no_norm",  dict(theta=0.5, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),
    ("PP_hour",     DivDir.INCREASE, "ma_smf_ablation_no_norm",  dict(theta=0.5, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),
    # ── VG_hour: MASCOT + UB + no_norm ──────────────────────────────
    # target map: MASCOT dec=0.3823, MASCOT inc=0.3754, UB dec=0.4102, UB inc=0.4040, no_norm dec=0.3522, no_norm inc=0.3929
    ("VG_hour",     DivDir.DECREASE, "ma_smf",                    dict(theta=0.4, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("VG_hour",     DivDir.INCREASE, "ma_smf",                    dict(theta=0.7, sigma_geo=1.0,  sigma_time=1.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("VG_hour",     DivDir.DECREASE, "ma_smf_ablation_no_omega",  dict(theta=0.9, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("VG_hour",     DivDir.INCREASE, "ma_smf_ablation_no_omega",  dict(theta=0.7, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("VG_hour",     DivDir.DECREASE, "ma_smf_ablation_no_norm",   dict(theta=0.5, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),
    ("VG_hour",     DivDir.INCREASE, "ma_smf_ablation_no_norm",   dict(theta=0.3, sigma_geo=1.0,  sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),
    # ── I1M_geo: MASCOT + UB + no_norm ─────────────────────────────
    # target map: MASCOT dec=0.6949, MASCOT inc=0.7160, UB dec=0.6781, UB inc=0.7333, no_norm dec=0.6521, no_norm inc=0.7178
    ("I1M_geo",     DivDir.DECREASE, "ma_smf",                    dict(theta=0.7, sigma_geo=5.0,  sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("I1M_geo",     DivDir.INCREASE, "ma_smf",                    dict(theta=0.7, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=False)),
    ("I1M_geo",     DivDir.DECREASE, "ma_smf_ablation_no_omega",  dict(theta=0.7, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("I1M_geo",     DivDir.INCREASE, "ma_smf_ablation_no_omega",  dict(theta=0.1, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=False, ablation_no_omega=True)),
    ("I1M_geo",     DivDir.DECREASE, "ma_smf_ablation_no_norm",   dict(theta=0.1, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),
    ("I1M_geo",     DivDir.INCREASE, "ma_smf_ablation_no_norm",   dict(theta=0.1, sigma_geo=10.0, sigma_time=0.5, ablation_no_norm=True,  ablation_no_omega=False)),
]

HF = MSDPP / "share_datasets" / "temp"
OUT = Path("/tmp/plot_gen_v2/failure_cases")
OUT.mkdir(parents=True, exist_ok=True)

log.info("Loading BLIP-2...")
model = Blip2Model(pretrained_model="Salesforce/blip2-itm-vit-g-coco", use_itm=False)
log.info("BLIP-2 loaded.")

dataset_cache = {}
sanity = []

for (ds_name, direction, key, params) in CONFIGS:
    dir_str = "decrease" if direction == DivDir.DECREASE else "increase"
    out_path = OUT / f"failure_cases_{ds_name}_{dir_str}_{key}.json"
    if out_path.exists():
        log.info(f"SKIP (exists) {out_path.name}")
        continue
    log.info(f"=== {ds_name} / {dir_str} / {key} @ {params} ===")

    if ds_name not in dataset_cache:
        p = HF / "tasks" / f"{ds_name}_test.pkl"
        dataset_cache[ds_name] = torch.load(p, weights_only=False)
    dataset = dataset_cache[ds_name]

    task = BaseTask(model=model, dataset=dataset, subset_k=200, top_k=20, n_thread=4)
    task.verify_iu_cache(n_samples=3)

    div_instance = registry.get_div_method("ma_smf")(**params)
    result = task.run(div_instance, direction, force=True)

    n = len(dataset.retrieval_words)
    fc = []
    for i, q in enumerate(dataset.retrieval_words):
        gt_indices = torch.where(dataset.labels[q] == 1)[0].tolist()
        gt_imgs = [getattr(dataset.dataset[j]["image"], "filename", str(j)) for j in gt_indices]
        ret_imgs = [getattr(dataset.dataset[j]["image"], "filename", str(j)) for j in result.candidates[i].tolist()]
        fc.append({"query": q, "ground_truth": gt_imgs, "retrieved": ret_imgs})
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)

    ev = result.eval_indices
    r10_fc = sum(1 for c in fc if any(r in set(c["ground_truth"]) for r in c["retrieved"][:10])) / n
    r1_fc  = sum(1 for c in fc if any(r in set(c["ground_truth"]) for r in c["retrieved"][:1]))  / n
    match = "OK" if abs(r10_fc - ev.r10.item()) < 1e-3 else "MISMATCH"
    log.info(f"  {out_path.name}: n={n}, r1={r1_fc:.4f}, r10={r10_fc:.4f}  (eval r10={ev.r10.item():.4f}, r1={ev.r1.item():.4f})  [{match}]")
    sanity.append((ds_name, dir_str, key, r1_fc, r10_fc, ev.r10.item(), ev.r1.item(), ev.map_.item() if hasattr(ev, 'map_') else None))

log.info("\n=== SANITY TABLE (extended) ===")
for s in sanity:
    log.info(f"  {s[0]:12s} {s[1]:9s} {s[2]:30s}  r1={s[3]:.4f}  r10={s[4]:.4f}  map={s[7] if s[7] is not None else '—'}  (eval r10={s[5]:.4f})")
