import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from pydantic import BaseModel, StrictFloat
from tqdm import tqdm

from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod
from msdpp.schema.evaluator import EvalIndices
from msdpp.task import BaseTask, DivDir, TaskResult

import msdpp.div_method 

if TYPE_CHECKING:
    from msdpp.schema import RetrievalDataset


VAL_SPICE_PATH = "/msdpp/share_datasets/SPICE_ver2/pp_spice_val.pt"
TEST_SPICE_PATH = "/msdpp/share_datasets/SPICE_ver2/pp_spice_test.pt"

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("=== MSDPP Grid Search ===")

DIV_METHOD_IDS = {
    "org": "00",
    "dpp_sim_average": "09",
    "msdpp": "10",
    "msdpp_tn": "11",
    "msdpp_tn_tvms": "12",
    "prob_coverage": "13",
    "ma_smf": "14",
    # --- New Ablations ---
    "ma_smf_ablation_no_norm": "15",
    "ma_smf_ablation_no_omega": "16",
    # --- New Baselines ---
    "blip2": "17",
    "mmr": "18",
    "clustering": "19",
}

class TableEntry(BaseModel):
    ret_index: StrictFloat
    div_index: StrictFloat
    m_ret_div: StrictFloat
    m_ret_div_ratio: StrictFloat
    val_score: StrictFloat
    # --- ADD THE NEW METRICS BELOW ---
    mrr: float = 0.0
    map_: float = 0.0
    r1: float = 0.0
    r5: float = 0.0
    r10: float = 0.0
    img_ilad: float = 0.0
    img_ilmd: float = 0.0
    ext_ilad: float = 0.0
    ext_ilmd: float = 0.0
    img_vendi: float = 0.0
    ext_vendi: float = 0.0
    mean_vendi: float = 0.0

class ResultTable(BaseModel):
    results: dict[str, TableEntry]

    def to_dict(self) -> dict:
        result_dict = {}

        for key, entry in self.results.items():
            result_dict[key] = entry.model_dump()

        return result_dict


class Evaluation:
    def __init__(
        self,
        root_dir: Path,
        result_dir: Path = Path("results"),
        use_spice: bool = False,
        overall_cfg_name: str = "overall.json",
        div_cfg_name: str = "div.json",
    ) -> None:
        self.root_dir = root_dir
        self.use_spice = use_spice
        self.result_dir = result_dir # <--- UPDATED: Store the result directory
        # table

        self.result_dir.mkdir(parents=True, exist_ok=True)

        if use_spice:
            self.table_path = self.result_dir / "tables_spice"
        else:
            self.table_path = self.result_dir / "tables"
        self.table_path.mkdir(parents=True, exist_ok=True)

        # hf home
        hf_home_str = os.environ.get("HF_HOME", "./")

        self.hf_home = Path(hf_home_str)

        # params
        with (root_dir / overall_cfg_name).open() as f:
            overall_cfg = json.load(f)

        self.dataset_names = overall_cfg["data_name"]
        self.subset_k = overall_cfg["subset_k"]
        self.top_k = overall_cfg["top_k"]

        model_name = overall_cfg["model_name"]
        model_params = overall_cfg["model_params"]

        self.model = registry.get_model(model_name)(**model_params)

        with (root_dir / div_cfg_name).open() as f:
            self.div_cfg = json.load(f)

    @staticmethod
    def improve_score(result: TaskResult, data_is_pp: bool) -> float:
        # Calculate the geometric mean of the evaluation indices
        org_ret_index = (
            result.org_indices.r10 if data_is_pp else result.org_indices.map_
        )
        org_div_index = result.org_indices.mean_vendi

        eval_ret_index = (
            result.eval_indices.r10 if data_is_pp else result.eval_indices.map_
        )
        eval_div_index = result.eval_indices.mean_vendi

        ret_ratio = eval_ret_index / org_ret_index
        div_ratio = eval_div_index / org_div_index

        return float((2 / (1 / ret_ratio + 1 / div_ratio)).item())

    def _eval(
        self,
        task: BaseTask,
        div_cls: type[BaseDiversificationMethod],
        params: dict,
        direction: DivDir,
        data_is_pp: bool,
        force: bool,
        spice_mat_path: str,
        score_name: str,
    ) -> tuple[float, dict]:
        # Evaluate the diversification method with the given parameters
        # and return the geometric mean of the evaluation indices
        div_instance = div_cls(**params)
        result: TaskResult = task.run(
            div_instance, direction, force=force, spice_mat_path=spice_mat_path
        )
        geo_mean_ret = self.improve_score(result, data_is_pp)

        geo_mean_ret = next(iter(result.eval_indices.harmonic_means.values()))[
            score_name
        ].item()

        return geo_mean_ret, params

    def grid_search(
        self,
        div_method: str,
        div_params: dict,
        direction: DivDir,
        task: BaseTask,
        data_is_pp: bool,
        force: bool = True,
    ) -> tuple[dict, float]:
        # Perform a grid search over the parameters of the diversification method
        # and return the best parameters and the best score

        score_name = "m_r10_vendi" if data_is_pp else "m_map_vendi"
        param_prods = [
            dict(zip(div_params.keys(), values, strict=False))
            for values in product(*div_params.values())
        ]
        div_cls = registry.get_div_method(div_method)

        best_params = {}
        best_score = -1000

        spice_mat_path = ""
        if data_is_pp and self.use_spice:
            spice_mat_path = VAL_SPICE_PATH

        self._eval(
            task,
            div_cls,
            param_prods[0],
            direction,
            data_is_pp,
            force,
            spice_mat_path,
            score_name,
        )

        with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as executor:
            futures = [
                executor.submit(
                    self._eval,
                    task,
                    div_cls,
                    params,
                    direction,
                    data_is_pp,
                    force,
                    spice_mat_path,
                    score_name,
                )
                for params in param_prods
            ]

            for future in tqdm(futures, total=len(futures), desc=f"{div_method}"):
                geo_mean_ret, params = future.result()

                if geo_mean_ret > best_score:
                    best_score = geo_mean_ret
                    best_params = params

        logger.info("----------------------------------")
        msg_bparam = f"{div_method}: Best params:" + ", ".join(
            f"{k}={v}" for k, v in best_params.items()
        )
        msg_bscore = f"{div_method}: Val best score: {best_score:.4f}"
        logger.info(msg_bparam)
        logger.info(msg_bscore)

        return best_params, best_score

    def gen_table_entry(
        self,
        indices: EvalIndices,
        improvement_score: float,
        data_is_pp: bool,
        best_score: float,
    ) -> TableEntry:
        # Generate a table entry from the evaluation indices and scores
        # This includes the retrieval index, diversity index, and the mean retrieval-diversity ratio
        if data_is_pp:
            score_name = "m_ncs_vendi" if self.use_spice else "m_r10_vendi"
            ret_index = indices.ncs.item() if self.use_spice else indices.r10.item()
        else:
            score_name = "m_map_vendi"
            ret_index = indices.map_.item()

        hms = next(iter(indices.harmonic_means.values()))

        m_ret_div = hms[score_name].item()

        return TableEntry(
            ret_index=ret_index,
            div_index=indices.mean_vendi.item(),
            m_ret_div=m_ret_div,
            m_ret_div_ratio=improvement_score,
            val_score=best_score,
            mrr=indices.mrr.item(),
            map_=indices.map_.item(),
            r1=indices.r1.item(),
            r5=indices.r5.item(),
            r10=indices.r10.item(),
            img_ilad=indices.img_ilad.mean().item(),
            img_ilmd=indices.img_ilmd.mean().item(),
            ext_ilad=indices.ext_ilad.mean().item(),
            ext_ilmd=indices.ext_ilmd.mean().item(),
            img_vendi=indices.img_vendi.item(),
            ext_vendi=indices.ext_vendi.item(),
            mean_vendi=indices.mean_vendi.item()
        )

    def run_eval(
        self,
        div_method: str,
        div_params: dict,
        suffix: str,
        direction: DivDir,
        task: BaseTask,
        test_task: BaseTask,
        data_is_pp: bool,
        force: bool = True,
    ) -> tuple[str, TableEntry, EvalIndices]:
        # Run the evaluation (grid-seaerch and eval on the test split)
        # for a specific diversification method
        # and return the table key, table entry, and original indices

        # grid search
        best_params, best_score = self.grid_search(
            div_method, div_params, direction, task, data_is_pp, force
        )

        spice_mat_path = ""
        if data_is_pp and self.use_spice:
            spice_mat_path = TEST_SPICE_PATH

        # test
        div_cls = registry.get_div_method(div_method)
        div_instance = div_cls(**best_params)
        test_result: TaskResult = test_task.run(
            div_instance, direction, force=force, spice_mat_path=spice_mat_path
        )
        score = self.improve_score(test_result, data_is_pp)

        results_dir = self.result_dir / "failure_cases"
        results_dir.mkdir(parents=True, exist_ok=True)     

        failure_cases = []
        for i, query in enumerate(test_task.dataset.retrieval_words):
            ground_truth_indices = torch.where(test_task.dataset.labels[query] == 1)[0].tolist()
            ground_truth_images = [getattr(test_task.dataset.dataset[j]["image"], "filename", str(j)) for j in ground_truth_indices]
            retrieved_images = [getattr(test_task.dataset.dataset[j]["image"], "filename", str(j)) for j in test_result.candidates[i].tolist()]
            
            failure_cases.append(
                {
                    "query": query,
                    "ground_truth": ground_truth_images,
                    "retrieved": retrieved_images,
                }
            )
        
        # MODIFICATION HERE: Include dataset name and direction in the filename
        direction_str = "increase" if direction == DivDir.INCREASE else "decrease"
        dataset_name = test_task.dataset.name.replace('_test', '') # Get the base dataset name
        
        failure_cases_dir = self.result_dir / "failure_cases"
        failure_cases_dir.mkdir(parents=True, exist_ok=True)
        
        with open(failure_cases_dir / f"failure_cases_{dataset_name}_{direction_str}_{div_method}.json", "w") as f:
            json.dump(failure_cases, f, indent=4)      

        # register result
        table_entry = self.gen_table_entry(
            test_result.eval_indices, score, data_is_pp, best_score
        )

        # create table key
        div_id = DIV_METHOD_IDS[div_method + suffix]
        table_key = f"{div_id}_{div_method}{suffix}"

        # log the result
        indices = {
            "Overall": table_entry.m_ret_div,
            "Retrieval Index": table_entry.ret_index,
            "Div Index": table_entry.div_index,
        }
        indices_str = [f"{k}: {x:.4f}" for k, x in indices.items()]
        result_msg = "Test result:" + " ".join(indices_str)
        logger.info(result_msg)
        return table_key, table_entry, test_result.org_indices

    def dataset_run(
        self, dataset_name: str, direction: DivDir, force: bool = True
    ) -> pd.DataFrame:
        # Run the evaluation for a specific dataset and direction for all diversification methods
        # This includes loading the dataset, running the grid search for each
        # diversification method, evaluating on the test split,
        # and saving the results to a table
        data_is_pp = "PP_" in dataset_name or "SkyScript" in dataset_name or "VG_" in dataset_name

        val_dataset_path = self.hf_home / "tasks" / f"{dataset_name}_val.pkl"
        test_dataset_path = self.hf_home / "tasks" / f"{dataset_name}_test.pkl"

        # load data
        val_dataset: RetrievalDataset = torch.load(
            val_dataset_path, weights_only=False
        )
        test_dataset: RetrievalDataset = torch.load(
            test_dataset_path, weights_only=False
        )

        # task
        task = BaseTask(
            dataset=val_dataset,
            model=self.model,
            subset_k=self.subset_k,
            top_k=self.top_k // 2,
            n_thread=32,
        )
        test_task = BaseTask(
            dataset=test_dataset,
            model=self.model,
            subset_k=self.subset_k,
            top_k=self.top_k,
            n_thread=32,
        )

        direction_str = "increase" if direction == DivDir.INCREASE else "decrease"

        table_path = self.table_path / f"{dataset_name}_{direction_str}.json"

        if table_path.exists():
            with table_path.open() as f:
                table_json = json.load(f)
            table = ResultTable.model_validate_json(json.dumps(table_json))
        else:
            table = ResultTable(results={})

        first = True

        for base_params in self.div_cfg.values():
            div_params = base_params["div_params"]
            div_methods = base_params["div_method"]
            suffix = base_params["name_suffix"]

            for div_method in div_methods:
                table_key, table_entry, org_indices = self.run_eval(
                    div_method,
                    div_params,
                    suffix,
                    direction,
                    task,
                    test_task,
                    data_is_pp,
                    force=force,
                )
                table.results[table_key] = table_entry

                # register original result
                if first:
                    first = False
                    table_entry = self.gen_table_entry(
                        org_indices, 1.0, data_is_pp, 1.0
                    )
                    table.results["00_org"] = table_entry

        table_json = json.loads(table.model_dump_json())
        with table_path.open("w") as f:
            json.dump(table_json, f)

        return pd.DataFrame(table.to_dict()).sort_index().T

    def run(self, force: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
        # Run the evaluation for all datasets and directions
        # This includes running the dataset_run for each dataset and direction
        # and returning a dictionary of dataframes for each dataset and direction

        data_frames = {}
        eval_df = None
        for dataset_name in self.dataset_names:
            msg = f"Running evaluation for dataset: {dataset_name}\n"
            data_is_pp = "PP_" in dataset_name or "SkyScript" in dataset_name or "VG_" in dataset_name

            if data_is_pp:
                if self.use_spice:
                    msg += ":: USE NCS as the retrieval metric\n"
                else:
                    msg += ":: Use R@10 as the retrieval metric\n"
            else:
                msg += ":: Use MAP as the retrieval metric\n"

            logger.info(msg)

            eval_df = self.dataset_run(dataset_name, DivDir.INCREASE, force=force)
            data_frames[dataset_name] = {"increase": eval_df}
            logger.info("##### Increase done")

            eval_df = self.dataset_run(dataset_name, DivDir.DECREASE, force=force)
            data_frames[dataset_name] = {"decrease": eval_df}
            logger.info("##### Decrease done")

            logger.info("##### Done")
        return data_frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MSDPP Grid Search Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pp",
        choices=["pp", "vg", "i1m", "flickr30k", "skyscript_geo", "skyscript_hour", "skyscript_geo_hour", "all"],
        help="Dataset to evaluate: pp, vg, i1m, flickr30k, skyscript_geo, skyscript_hour, skyscript_geo_hour, all",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/<dataset>)",
    )
    parser.add_argument(
        "--div_cfg",
        type=str,
        default=None,
        help="Override div config filename (e.g. div_decrease.json). Default: per-dataset config.",
    )
    args = parser.parse_args()

    DATASET_CONFIGS = {
        "pp": {
            "overall_cfg": "overall.json",
            "div_cfg": "div_selected.json",
            "result_dir": "results/pp",
            "use_spice": True,
        },
        "vg": {
            "overall_cfg": "overall_vg.json",
            "div_cfg": "div_selected_vg.json",
            "result_dir": "results/vg",
            "use_spice": False,
        },
        "i1m": {
            "overall_cfg": "overall_i1m.json",
            "div_cfg": "div_selected_i1m.json",
            "result_dir": "results/i1m",
            "use_spice": False,
        },
        "flickr30k": {
            "overall_cfg": "overall_flickr30k.json",
            "div_cfg": "div_selected_flickr30k.json",
            "result_dir": "results/flickr30k",
            "use_spice": False,
        },
        "skyscript_geo": {
            "overall_cfg": "overall_skyscript_geo.json",
            "div_cfg": "div_selected_skyscript_geo.json",
            "result_dir": "results/skyscript_geo",
            "use_spice": False,
        },
        "skyscript_hour": {
            "overall_cfg": "overall_skyscript_hour.json",
            "div_cfg": "div_selected_skyscript_hour.json",
            "result_dir": "results/skyscript_hour",
            "use_spice": False,
        },
        "skyscript_geo_hour": {
            "overall_cfg": "overall_skyscript_geo_hour.json",
            "div_cfg": "div_selected_skyscript_geo_hour.json",
            "result_dir": "results/skyscript_geo_hour",
            "use_spice": False,
        },
    }

    datasets_to_run = (
        ["pp", "vg", "i1m", "skyscript_geo", "skyscript_hour", "skyscript_geo_hour"]
        if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets_to_run:
        cfg = DATASET_CONFIGS[ds]
        result_dir = Path(args.result_dir) if args.result_dir else Path(cfg["result_dir"])
        logger.info(f"=== Running evaluation for: {ds.upper()} ===")
        evaluator = Evaluation(
            root_dir=Path("examples/configs"),
            result_dir=result_dir,
            use_spice=cfg["use_spice"],
            overall_cfg_name=cfg["overall_cfg"],
            div_cfg_name=args.div_cfg if args.div_cfg else cfg["div_cfg"],
        )
        evaluator.run()
