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
from msdpp.data import get_time_iu_probs, get_geo_iu_probs, get_coco_iu_probs, get_cluster_iu_probs

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
        self.coco_cat_to_bin = getattr(dataset, 'coco_cat_to_bin', {})

        if not cache_home.exists():
            cache_home.mkdir(exist_ok=True, parents=True)

        # Pre-load IU fields from HF Arrow into CPU tensors once.
        # On first _run_trial call the tensors are moved to the compute device and
        # cached there, so subsequent queries pay zero H2D transfer cost.
        self._iu_cache: dict[str, torch.Tensor | list] = {}
        dl = self.dataset.dataset
        dn = self.dataset.name.lower()
        if dl is not None:
            if "geo" in dn:
                self._iu_cache["gps"] = torch.tensor(dl["gps"], dtype=torch.float32)
            if "hour" in dn:
                self._iu_cache["hour"]   = torch.tensor(dl["hour"],   dtype=torch.int32)
                self._iu_cache["minute"] = torch.tensor(dl["minute"], dtype=torch.int32)
            if "coco" in dn:
                self._iu_cache["category_ids"] = dl["category_ids"]  # list[list[int]]
            if "cluster" in dn or "visualcluster" in dn:
                self._iu_cache["cluster_id"] = torch.tensor(dl["cluster_id"], dtype=torch.int32)
                self._iu_cache["num_clusters"] = getattr(self.dataset, 'num_clusters', 50)

    def verify_iu_cache(self, n_samples: int = 5) -> None:
        """Assert pre-loaded tensors match HF Arrow row-by-row on n_samples indices."""
        dl = self.dataset.dataset
        dn = self.dataset.name.lower()
        indices = list(range(min(n_samples, len(dl))))
        if "geo" in dn:
            expected = torch.tensor([dl[i]["gps"] for i in indices], dtype=torch.float32)
            assert torch.allclose(self._iu_cache["gps"][indices], expected), \
                "GPS cache mismatch"
        if "hour" in dn:
            exp_h = torch.tensor([dl[i]["hour"]   for i in indices], dtype=torch.int32)
            exp_m = torch.tensor([dl[i]["minute"] for i in indices], dtype=torch.int32)
            assert torch.equal(self._iu_cache["hour"][indices],   exp_h), "hour cache mismatch"
            assert torch.equal(self._iu_cache["minute"][indices], exp_m), "minute cache mismatch"
        if "coco" in dn:
            for i in indices:
                assert self._iu_cache["category_ids"][i] == dl[i]["category_ids"], \
                    f"category_ids mismatch at index {i}"
        if "cluster" in dn or "visualcluster" in dn:
            exp_c = torch.tensor([dl[i]["cluster_id"] for i in indices], dtype=torch.int32)
            assert torch.equal(self._iu_cache["cluster_id"][indices], exp_c), \
                "cluster_id cache mismatch"

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

    def _cache_rerank_path(self, model_name: str, data_name: str, div_method: BaseDiversificationMethod, direction: DivDir,
                           per_attribute_directions: list[DivDir] | None = None) -> tuple[Path, Path]:
        params = div_method.get_params
        param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
        # Use the full dataset name (minus the trailing _val/_test split marker) so
        # composite tasks like PP_geo_hour do not collide with their single-attribute
        # siblings PP_geo / PP_hour in the rerank cache. The previous version kept
        # only the first two underscore-separated parts, which mapped both
        # PP_geo_hour_test and PP_geo_test to `pp_geo_test`, causing whichever task
        # ran last at a given (theta, sigma_geo, sigma_time) to overwrite the other.
        split_marker = "_val" if data_name.endswith("val") else ("_test" if data_name.endswith("test") else "")
        stem = data_name[:-len(split_marker)] if split_marker else data_name
        _data_name = stem.lower() + split_marker
        rerank_dir = self.cache_home / "rerank"
        rerank_dir.mkdir(exist_ok=True)
        if per_attribute_directions is not None:
            # Build an unambiguous direction tag for mixed-direction runs so they
            # never collide with single-direction cache files.
            attr_tags = []
            dn = data_name.lower()
            if "geo" in dn:
                attr_tags.append("geo_" + ("inc" if per_attribute_directions[0] == DivDir.INCREASE else "dec"))
            if "hour" in dn:
                attr_tags.append("time_" + ("inc" if per_attribute_directions[-1] == DivDir.INCREASE else "dec"))
            direction_name = "_".join(attr_tags) if attr_tags else "mixed"
        else:
            direction_name = "increase" if direction == DivDir.INCREASE else "decrease"
        return (
            rerank_dir / f"{model_name!s}_{_data_name}_{direction_name}_{div_method.get_name}_{param_str}.pkl",
            rerank_dir / f"index_{model_name!s}_{_data_name}_{direction_name}_{div_method.get_name}_{param_str}.pkl"
        )

    def load_cache_rerank(self, model_name: str, data_name: str, div_method: BaseDiversificationMethod, direction: DivDir,
                           per_attribute_directions: list[DivDir] | None = None) -> list[torch.Tensor] | None:
        result_path, _ = self._cache_rerank_path(model_name, data_name, div_method, direction, per_attribute_directions)
        return torch.load(result_path, weights_only=False) if Path(result_path).exists() else None

    def cache_rerank(self, result: TaskResult, div_method: BaseDiversificationMethod, direction: DivDir,
                     per_attribute_directions: list[DivDir] | None = None) -> None:
        result_path, index_path = self._cache_rerank_path(str(self.model), self.dataset.name, div_method, direction, per_attribute_directions)
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
        self, i, div_method, direction, t2i_sim, image_feats, ext_data, subset_k, top_k,
        per_attribute_directions=None,
    ) -> tuple[int, torch.Tensor]:
        # Identify the hardware device (CPU or GPU)
        device = image_feats.device

        topk_idx = t2i_sim[i].argsort(descending=True).cpu()[:subset_k]
        topk_t2i_sim = t2i_sim[i][topk_idx]

        method_name = div_method.get_name

        # INTEGRATION: Probabilistic Coverage (Paper 2)
        if method_name in ["prob_coverage", "ma_smf"]:
            data_name_lower = self.dataset.name.lower()
            iu_matrices = []
            attr_boundaries = [0]   # tracks column split-points for per-attribute direction

            s_geo = getattr(div_method, 'sigma_geo')
            s_time = getattr(div_method, 'sigma_time')
            idx_long = topk_idx.long()

            # Lazily promote CPU cache tensors to compute device on first use.
            for key in ("gps", "hour", "minute"):
                if key in self._iu_cache and isinstance(self._iu_cache[key], torch.Tensor) \
                        and self._iu_cache[key].device != device:
                    self._iu_cache[key] = self._iu_cache[key].to(device)

            # index_select on GPU-resident tensors: zero H2D transfer per query
            idx_dev = idx_long.to(device)

            # 1. Check for Geographic Data
            if "geo" in data_name_lower:
                geo_mat = get_geo_iu_probs(
                    torch.index_select(self._iu_cache["gps"], 0, idx_dev),
                    grid_size=self.grid_size, sigma=s_geo,
                )
                iu_matrices.append(geo_mat)
                attr_boundaries.append(attr_boundaries[-1] + geo_mat.shape[1])

            # 2. Check for Time Data
            if "hour" in data_name_lower:
                time_mat = get_time_iu_probs(
                    torch.index_select(self._iu_cache["hour"],   0, idx_dev),
                    torch.index_select(self._iu_cache["minute"], 0, idx_dev),
                    num_bins=self.num_bins, sigma=s_time,
                )
                iu_matrices.append(time_mat)
                attr_boundaries.append(attr_boundaries[-1] + time_mat.shape[1])

            # 3. Check for COCO category data
            if "coco" in data_name_lower:
                cat_ids_list = [self._iu_cache["category_ids"][int(idx)] for idx in topk_idx]
                cat_mat = get_coco_iu_probs(cat_ids_list, self.coco_cat_to_bin).to(device)
                iu_matrices.append(cat_mat)
                attr_boundaries.append(attr_boundaries[-1] + cat_mat.shape[1])

            # 4. Check for Visual Cluster data
            if "cluster" in data_name_lower or "visualcluster" in data_name_lower:
                cluster_mat = get_cluster_iu_probs(
                    torch.index_select(self._iu_cache["cluster_id"], 0, idx_dev),
                    self._iu_cache["num_clusters"],
                ).to(device)
                iu_matrices.append(cluster_mat)
                attr_boundaries.append(attr_boundaries[-1] + cluster_mat.shape[1])

            iu_matrix = torch.cat(iu_matrices, dim=1)
            candidates = div_method.search(
                info=iu_matrix,
                direction=direction,
                t2i_sim=topk_t2i_sim.to(device),
                top_k=top_k,
                attr_boundaries=attr_boundaries if per_attribute_directions is not None else None,
                per_attribute_directions=per_attribute_directions,
            )
        else:
            # Paper 1: MS-DPP Manifold Representation
            topk_ext_data = [e[topk_idx] for e in ext_data] if isinstance(ext_data, list) else ext_data[topk_idx]
            candidates = div_method.diversify(
                topk_t2i_sim, topk_ext_data, direction, top_k, image_feats[topk_idx],
                per_attribute_directions=per_attribute_directions,
            )

        return i, torch.stack([topk_idx[j] for j in candidates])

    def _run(self, div_method, direction, t2i_sim, image_feats, ext_data, retrieval_words, subset_k, top_k,
             per_attribute_directions=None) -> list[torch.Tensor]:
        all_candidates = []
        with torch.inference_mode():
            for i in range(len(retrieval_words)):
                _, indices = self._run_trial(i, div_method, direction, t2i_sim, image_feats, ext_data, subset_k, top_k,
                                             per_attribute_directions=per_attribute_directions)
                all_candidates.append(indices)
        return all_candidates

    def _run_parallel(self, div_method, direction, t2i_sim, image_feats, ext_data, retrieval_words, subset_k, top_k,
                      per_attribute_directions=None) -> list[torch.Tensor]:
        with torch.inference_mode(), ThreadPoolExecutor(self.n_thread) as executor:
            unsorted = list(executor.map(
                lambda i: self._run_trial(i, div_method, direction, t2i_sim, image_feats, ext_data, subset_k, top_k,
                                          per_attribute_directions=per_attribute_directions),
                range(len(retrieval_words)),
            ))
            return [x[1] for x in sorted(unsorted, key=lambda x: x[0])]

    def run(self, div_method: BaseDiversificationMethod, direction: DivDir, force: bool = False,
            spice_mat_path: str = "",
            per_attribute_directions: list[DivDir] | None = None) -> TaskResult:
        dataset, retrieval_results = self.dataset, self._retrieve()
        t2i_sim = retrieval_results.t2i_sim.to("cpu")
        labels_list = list(dataset.labels.values())

        org_indices = self.eval_calculator.run(t2i_sim, labels_list, retrieval_results.image_feats, dataset.ext_data, direction, spice_mat_path)

        all_candidates = []
        cached_result = None if force else self.load_cache_rerank(str(self.model), dataset.name, div_method, direction, per_attribute_directions)
        if cached_result is not None:
            all_candidates = cached_result
        else:
            run_func = self._run if self.n_thread <= 1 else self._run_parallel
            all_candidates = run_func(div_method, direction, t2i_sim, retrieval_results.image_feats, dataset.ext_data,
                                      dataset.retrieval_words, self.subset_k, self.top_k,
                                      per_attribute_directions=per_attribute_directions)

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
            self.cache_rerank(task_result, div_method, direction, per_attribute_directions)
        return task_result