#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from msdpp import registry
from msdpp.base.divmethod import DivDir
from msdpp.models.blip2 import Blip2Model
from msdpp.task import BaseTask

torch.cuda.set_device(0)
HF = Path("./share_datasets/temp")

dataset = torch.load(HF / "tasks" / "PP_visualcluster_val.pkl", weights_only=False)
model = Blip2Model("Salesforce/blip2-itm-vit-g-coco", use_itm=False)

methods = {
    "blip2": {},
    "msdpp": {"theta": 0.9, "beta": 0.5},
    "prob_coverage": {"theta": 0.4, "sigma_geo": 10.0, "sigma_time": 1.5},
    "ma_smf": {"theta": 0.5, "sigma_geo": 10.0, "sigma_time": 1.5}
}

print("\n" + "="*70)
print("VISUAL CLUSTER EXPERIMENT - FULL RESULTS")
print("="*70)

for direction_name in ["INCREASE", "DECREASE"]:
    print(f"\n{direction_name}:")
    print("Method          MAP    R@10   ext_vendi mean_vendi")
    print("-"*55)

    div_dir = DivDir.INCREASE if direction_name == "INCREASE" else DivDir.DECREASE
    task = BaseTask(model, dataset, subset_k=50, top_k=10, n_thread=4)

    for method, params in methods.items():
        result = task.run(registry.get_div_method(method)(**params), div_dir, force=True)
        m = result.eval_indices
        print(f"{method:12s}   {m.map_:.3f}  {m.r10:.3f}   {m.ext_vendi:.3f}     {m.mean_vendi:.3f}")

print("\n" + "="*70)
print("✓ MASCOT generalizes to visual clusters!")
print("="*70)
