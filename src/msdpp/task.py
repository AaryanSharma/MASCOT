import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from msdpp.base.divmethod import (
    BaseDiversificationMethod,
    DivDir,
)
from msdpp.base.model import BaseModel, RetrievalResults
from msdpp.evalindex.eval_index import EvalIndexCalculator
from msdpp.schema import (
    CacheResult,
    RetrievalDataset,
    TaskResult,
)
# Binning logic to handle "Uncertain Attributes" from Paper 2
from msdpp.data import get_time_iu_probs, get_geo_iu_probs 

CACHE_HOME = Path(os.environ.get("HF_HOME", "./")) / "div_results"

class BaseTask:
    def __init__(
        self,
        model: BaseModel,
        dataset: RetrievalDataset,
        subset_k: int = 200,
        top_k: int = 50,
        cache_home: Path = CACHE_HOME,
        do_cache_sim: bool = True,
        n_thread: int = 1,
        grid_size: int = 20,
        num_bins: int = 24,
    ) -> None:
        self.model = model
        self.subset_k = subset_k
        self.top_k = top_k
        self.cache_home = cache_home
        self.do_cache_sim = do_cache_sim
        self.n_thread = n_thread
        self.grid_size = grid_size
        self.num_bins = num_bins

        self.eval_calculator = EvalIndexCalculator(top_k)

        # load cached retrieval results if available
        self.retrieval_results = self.load_cache_sim(
            str(self.model), dataset.name, dataset.retrieval_words
        )
        self.dataset = dataset

        if not cache_home.exists():
            cache_home.mkdir(exist_ok=True, parents=True)

    def _cache_sim_path(self, model_name: str, data_name: str) -> Path:
        _data_name = data_name.split("_")[0].lower()
        if data_name.endswith("val"): _data_name += "_val"
        if data_name.endswith("test"): _data_name += "_test"
        return self.cache_home / f"{model_name!s}_{_data_name}_ret_results.pkl"

    def load_cache_sim(self, model_name: str, data_name: str, retrieval_words: list[str]) -> RetrievalResults | None:
        result_path = self._cache_sim_path(model_name, data_name)
        if not Path(result_path).exists(): return None
        cached_sim: CacheResult = torch.load(result_path, weights_only=False)
        if not all(i in cached_sim.retrieval_words for i in retrieval_words): return None
        idxs = [cached_sim.retrieval_words.index(word) for word in retrieval_words]
        cached_sim.results.t2i_sim = cached_sim.results.t2i_sim[idxs].contiguous()
        cached_sim.results.text_feats = cached_sim.results.text_feats[idxs].contiguous()
        return cached_sim.results

    def cache_sim(self, model_name: str, data_name: str, results: RetrievalResults, retrieval_words: list[str]) -> None:
        t2i_sim_path = self._cache_sim_path(model_name, data_name)
        results.to("cpu")
        torch.save(CacheResult(results=results, retrieval_words=retrieval_words), t2i_sim_path)

    def _cache_rerank_path(self, model_name: str, data_name: str, div_method: BaseDiversificationMethod, direction: DivDir) -> tuple[Path, Path]:
        params = div_method.get_params
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        data_name_split = data_name.split("_")
        _data_name = "_".join([data_name_split[0].lower(), data_name_split[1].lower()]) if "pp_" in data_name.lower() else data_name.split("_")[0].lower()
        if data_name.endswith("val"): _data_name += "_val"
        if data_name.endswith("test"): _data_name += "_test"
        rerank_dir = self.cache_home / "rerank"
        rerank_dir.mkdir(exist_ok=True)
        direction_name = "increase" if direction == DivDir.INCREASE else "decrease"
        return (
            rerank_dir / f"{model_name!s}_{_data_name}_{direction_name}_{div_method.get_name}_{param_str}.pkl",
            rerank_dir / f"index_{model_name!s}_{_data_name}_{direction_name}_{div_method.get_name}_{param_str}.pkl"
        )

    def load_cache_rerank(self, model_name: str, data_name: str, div_method: BaseDiversificationMethod, direction: DivDir) -> list[torch.Tensor] | None:
        result_path, _ = self._cache_rerank_path(model_name, data_name, div_method, direction)
        return torch.load(result_path, weights_only=False) if Path(result_path).exists() else None

    def cache_rerank(self, result: TaskResult, div_method: BaseDiversificationMethod, direction: DivDir) -> None:
        result_path, index_path = self._cache_rerank_path(str(self.model), self.dataset.name, div_method, direction)
        torch.save(result.candidates, result_path)
        torch.save({"org": result.org_indices, "eval": result.eval_indices}, index_path)

    def _retrieve(self) -> RetrievalResults:
        retrieval_results = self.retrieval_results
        if retrieval_results is None:
            retrieval_results = self.model.infer_datasets(self.dataset.dataset, texts=self.dataset.retrieval_words)
            if self.do_cache_sim:
                self.cache_sim(str(self.model), self.dataset.name, retrieval_results, self.dataset.retrieval_words)
                self.retrieval_results = retrieval_results
        return retrieval_results

    def _run_trial(
        self, i, div_method, direction, t2i_sim, image_feats, ext_data, subset_k, top_k
    ) -> tuple[int, torch.Tensor]:
        # Identify the hardware device (CPU or GPU)
        device = image_feats.device 
        
        topk_idx = t2i_sim[i].argsort(descending=True).cpu()[:subset_k]
        topk_t2i_sim = t2i_sim[i][topk_idx]

        method_name = div_method.get_name

        # INTEGRATION: Probabilistic Coverage (Paper 2)
        if method_name in ["prob_coverage", "ma_smf"]:
            data_list = self.dataset.dataset
            data_name_lower = self.dataset.name.lower()
            iu_matrices = []

            s_geo = getattr(div_method, 'sigma_geo')
            s_time = getattr(div_method, 'sigma_time')

            # 1. Check for Geographic Data
            if "geo" in data_name_lower:
                raw_gps = torch.tensor([data_list[int(idx)]["gps"] for idx in topk_idx])
                iu_matrices.append(get_geo_iu_probs(raw_gps, grid_size=self.grid_size, sigma=s_geo).to(device))

            # 2. Check for Time Data
            if "hour" in data_name_lower:
                raw_hours = torch.tensor([data_list[int(idx)]["hour"] for idx in topk_idx])
                raw_mins = torch.tensor([data_list[int(idx)]["minute"] for idx in topk_idx])
                iu_matrices.append(get_time_iu_probs(raw_hours, raw_mins, num_bins=self.num_bins, sigma=s_time).to(device))
            # if "hour" in self.dataset.name.lower():
            #     raw_hours = torch.tensor([data_list[int(idx)]["hour"] for idx in topk_idx])
            #     raw_mins = torch.tensor([data_list[int(idx)]["minute"] for idx in topk_idx])
            #     # Convert to IU probs and move to GPU
            #     iu_matrix = get_time_iu_probs(raw_hours, raw_mins).to(device)
            # else:
            #     raw_gps = torch.tensor([data_list[int(idx)]["gps"] for idx in topk_idx])
            #     # Convert to IU probs and move to GPU
            #     iu_matrix = get_geo_iu_probs(raw_gps).to(device)

            iu_matrix = torch.cat(iu_matrices, dim=1)
            # Paper 2: Greedy Probabilistic Coverage Optimization (on GPU)
            candidates = div_method.search(
                info=iu_matrix, 
                direction=direction,
                t2i_sim=topk_t2i_sim.to(device), # Ensure sim is on GPU
                top_k=top_k,
            )
        else:
            # Paper 1: MS-DPP Manifold Representation
            topk_ext_data = [e[topk_idx] for e in ext_data] if isinstance(ext_data, list) else ext_data[topk_idx]
            candidates = div_method.diversify(
                topk_t2i_sim, topk_ext_data, direction, top_k, image_feats[topk_idx]
            )

        return i, torch.stack([topk_idx[j] for j in candidates])

    def _run(self, div_method, direction, t2i_sim, image_feats, ext_data, retrieval_words, subset_k, top_k) -> list[torch.Tensor]:
        all_candidates = []
        with torch.inference_mode():
            for i in range(len(retrieval_words)):
                _, indices = self._run_trial(i, div_method, direction, t2i_sim, image_feats, ext_data, subset_k, top_k)
                all_candidates.append(indices)
        return all_candidates

    def _run_parallel(self, div_method, direction, t2i_sim, image_feats, ext_data, retrieval_words, subset_k, top_k) -> list[torch.Tensor]:
        with torch.inference_mode(), ThreadPoolExecutor(self.n_thread) as executor:
            unsorted = list(executor.map(lambda i: self._run_trial(i, div_method, direction, t2i_sim, image_feats, ext_data, subset_k, top_k), range(len(retrieval_words))))
            return [x[1] for x in sorted(unsorted, key=lambda x: x[0])]

    def run(self, div_method: BaseDiversificationMethod, direction: DivDir, force: bool = False, spice_mat_path: str = "") -> TaskResult:
        dataset, retrieval_results = self.dataset, self._retrieve()
        t2i_sim = retrieval_results.t2i_sim.to("cpu")
        labels_list = list(dataset.labels.values())

        org_indices = self.eval_calculator.run(t2i_sim, labels_list, retrieval_results.image_feats, dataset.ext_data, direction, spice_mat_path)

        all_candidates = []
        cached_result = None if force else self.load_cache_rerank(str(self.model), dataset.name, div_method, direction)
        if cached_result is not None:
            all_candidates = cached_result
        else:
            run_func = self._run if self.n_thread <= 1 else self._run_parallel
            all_candidates = run_func(div_method, direction, t2i_sim, retrieval_results.image_feats, dataset.ext_data, dataset.retrieval_words, self.subset_k, self.top_k)

        # Build final rankings
        best_idxs = torch.zeros((len(dataset.retrieval_words), t2i_sim.shape[1]), dtype=torch.long)
        for i, b_idx in enumerate(all_candidates):
            rem = torch.tensor(list(set(range(t2i_sim.shape[1])) - set(b_idx.tolist())))
            best_idxs[i] = torch.cat([b_idx, rem])

        mod_sim = torch.zeros_like(t2i_sim)
        for i in range(best_idxs.shape[0]):
            for j in range(self.top_k): mod_sim[i, best_idxs[i, j]] = self.top_k - j

        eval_indices = self.eval_calculator.run(mod_sim, labels_list, retrieval_results.image_feats, dataset.ext_data, direction, spice_mat_path)
        task_result = TaskResult(org_indices=org_indices, eval_indices=eval_indices, candidates=all_candidates, t2i_sim=t2i_sim)

        if self.do_cache_sim and cached_result is None:
            self.cache_rerank(task_result, div_method, direction)
        return task_result