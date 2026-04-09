import os
import time
import numpy as np
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm

from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod
from msdpp.task import BaseTask, DivDir

# Setup simple logging
logging.basicConfig(level=logging.WARNING)

# 1. LOAD THE TASK
def load_task(dataset_name="PP_geo_hour"):
    print(f"\n[ SYSTEM ] Loading Dataset: {dataset_name}...")
    hf_home = Path(os.environ.get("HF_HOME", "./examples/configs"))
    
    config_path = Path("examples/configs/overall.json")
    with config_path.open() as f:
        overall_cfg = json.load(f)
        
    model_name = overall_cfg["model_name"]
    model_params = overall_cfg["model_params"]
    model = registry.get_model(model_name)(**model_params)
    
    test_dataset_path = Path(os.environ.get("HF_HOME", "./")) / "tasks" / f"{dataset_name}_test.pkl"
    test_dataset = torch.load(test_dataset_path, weights_only=False)
    
    task = BaseTask(
        dataset=test_dataset,
        model=model,
        subset_k=overall_cfg["subset_k"],
        top_k=overall_cfg["top_k"],
        n_thread=1,
    )
    return task

# 2. PREFERENCE REFLECTION SCORE (PRS) EXPERIMENT
def run_prs(task: BaseTask, methods: dict, attribute_type="ext"):
    weights = np.linspace(0.0, 1.0, 11)
    
    # We will store results for both directions
    final_prs = {m: {"INC": 0.0, "DEC": 0.0} for m in methods.keys()}
    
    directions = [DivDir.INCREASE, DivDir.DECREASE]
    
    for direction in directions:
        dir_name = "INC" if direction == DivDir.INCREASE else "DEC"
        print(f"\n--- Running Task: {dir_name.upper()} ---")
        
        for method_name, base_params in methods.items():
            raw_div_scores = []
            
            for w in tqdm(weights, desc=f"{method_name.upper()[:15]:<15}"):
                params = base_params.copy()
                
                # --- THE MAPPING FIX ---
                # w = 0.0 means "Pure Relevance"
                # w = 1.0 means "Pure Constraint (Diversity/Redundancy)"
                
                if "dpp" in method_name.lower():
                    # For MS-DPP, theta=1.0 is Pure Relevance. theta=0.0 is Pure Constraint.
                    # So we INVERT the weight: theta = 1.0 - w
                    safe_theta = 1.0 - w
                    if safe_theta >= 1.0:
                        safe_theta = 0.99  # Prevent ZeroDivisionError in MS-DPP
                    params["theta"] = float(safe_theta)
                else:
                    # For MASCOT/Prob-Coverage/MMR, theta=0.0 is Pure Relevance. theta=1.0 is Pure Constraint.
                    params["theta"] = float(w)
                    
                div_cls = registry.get_div_method(method_name)
                div_instance = div_cls(**params)
                
                # Run the task
                result = task.run(div_instance, direction, force=True)
                
                # Get the diversity score
                if attribute_type == "ext":
                    div_score = result.eval_indices.ext_vendi.item()
                else:
                    div_score = result.eval_indices.img_vendi.item()
                    
                raw_div_scores.append(div_score)
                
            # Normalize the scores (Min-Max Scaling)
            div_arr = np.array(raw_div_scores)
            min_div, max_div = div_arr.min(), div_arr.max()
            
            if max_div == min_div:
                norm_div = np.zeros_like(div_arr)
            else:
                norm_div = (div_arr - min_div) / (max_div - min_div)
                
            # Calculate the PRS summation
            prs = 0.0
            for j in range(1, len(weights)):
                numerator = norm_div[j] - norm_div[j-1]
                denominator = weights[j] - weights[j-1]
                prs += (numerator / denominator)
                
            # Average the PRS score (matches the theoretical bounds of -10 to +10)
            avg_prs = prs / (len(weights) - 1)
            final_prs[method_name][dir_name] = avg_prs
            print(f" -> PRS ({dir_name}): {avg_prs:.4f}")

    return final_prs

if __name__ == "__main__":
    
    # 1. Define models and base parameters
    methods_to_test = {
        "mmr": {"theta": 0.2},
        "dpp_sim_average": {"beta": 0.4},
        "msdpp": {"beta": 0.4},
        "msdpp_tn": {"beta": 0.5},
        "msdpp_tn_tvms": {"beta": 0.4},
        "prob_coverage": {"sigma_geo": 1.0, "sigma_time": 0.5},
        "ma_smf": {"sigma_geo": 15.0, "sigma_time": 1.5, "ablation_no_norm": False, "ablation_no_omega": False},
    }
    
    datasets = ["PP_geo_hour", "PP_hour", "PP_geo"]
    master_results = {}
    
    # 2. Loop through all datasets
    for dataset in datasets:
        print("\n" + "="*80)
        print(f"STARTING EXPERIMENT ON DATASET: {dataset.upper()}")
        print("="*80)
        
        task = load_task(dataset)
        dataset_prs = run_prs(task, methods_to_test, attribute_type="ext")
        master_results[dataset] = dataset_prs
        
    # 3. Print Final Formatted Master Table
    print("\n\n" + "="*80)
    print("FINAL MASTER PRS SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<20} | {'PP_GEO_HOUR (INC/DEC)':<25} | {'PP_HOUR (INC/DEC)':<25} | {'PP_GEO (INC/DEC)':<25}")
    print("-" * 80)
    
    for m in methods_to_test.keys():
        geo_hr_inc = master_results["PP_geo_hour"][m]["INC"]
        geo_hr_dec = master_results["PP_geo_hour"][m]["DEC"]
        
        hr_inc = master_results["PP_hour"][m]["INC"]
        hr_dec = master_results["PP_hour"][m]["DEC"]
        
        geo_inc = master_results["PP_geo"][m]["INC"]
        geo_dec = master_results["PP_geo"][m]["DEC"]
        
        # Format strings carefully (e.g. 10.0000 / 9.9876)
        geo_hr_str = f"{geo_hr_inc:.4f} / {geo_hr_dec:.4f}"
        hr_str = f"{hr_inc:.4f} / {hr_dec:.4f}"
        geo_str = f"{geo_inc:.4f} / {geo_dec:.4f}"
        
        print(f"{m.upper():<20} | {geo_hr_str:<25} | {hr_str:<25} | {geo_str:<25}")