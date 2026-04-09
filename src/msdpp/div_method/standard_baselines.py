import torch
import numpy as np
from sklearn.cluster import KMeans
from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod, DivDir

@registry.register_div_method("blip2")
class BLIP2Baseline(BaseDiversificationMethod):
    def __init__(self, **kwargs):
        pass

    def search(self, *args, **kwargs): pass

    def diversify(self, t2i_sim, ext_data, direction, top_k, image_feats, **kwargs):
        return torch.arange(top_k)

    @property
    def get_params(self): return {}
    @property
    def get_name(self): return "blip2"


@registry.register_div_method("mmr")
class MMRBaseline(BaseDiversificationMethod):
    def __init__(self, theta: float = 0.5, **kwargs):
        self.theta = theta 

    def search(self, *args, **kwargs): pass

    def diversify(self, t2i_sim, ext_data, direction, top_k, image_feats, **kwargs):
        N = len(t2i_sim)
        selected = []
        candidates = list(range(N))
        
        # 1. Process Image Features
        if image_feats.dim() == 3: # shape: [N, 32, D]
            image_feats = image_feats.mean(dim=1)
        # L2 Normalize for stable inner product (Cosine Similarity)
        image_feats = torch.nn.functional.normalize(image_feats, p=2, dim=-1)
        
        # 2. Process Attribute Features (Appendix C.2: Multiply by s_i and w_i)
        s_i = -1.0 if direction == DivDir.DECREASE else 1.0
        
        if isinstance(ext_data, list):
            w_i = 1.0 / len(ext_data) # Uniform weighting for multiple attributes
            ext_list = [torch.nn.functional.normalize(e.float(), p=2, dim=-1) * (s_i * w_i) for e in ext_data]
            ext_cat = torch.cat(ext_list, dim=-1)
        else:
            w_i = 1.0
            ext_cat = torch.nn.functional.normalize(ext_data.float(), p=2, dim=-1) * (s_i * w_i)

        # 3. Concatenate Visuals + Attributes
        concat_feats = torch.cat([image_feats, ext_cat], dim=-1)

        # 4. Unified Similarity Matrix (Dot Product)
        # Because we multiplied attributes by s_i, the dot product natively handles 
        # the Decrease task penalty (Vis_sim - Attr_sim)
        sim_matrix = torch.matmul(concat_feats, concat_feats.T)
        
        # Min-Max normalize similarity matrix to [0, 1] to balance properly with theta
        sim_min, sim_max = sim_matrix.min(), sim_matrix.max()
        sim_matrix = (sim_matrix - sim_min) / (sim_max - sim_min + 1e-8)

        # 5. Standard MMR Loop
        for _ in range(top_k):
            if not selected:
                best_idx = candidates[torch.argmax(t2i_sim[candidates]).item()]
            else:
                best_score = -float('inf')
                best_idx = -1
                for c in candidates:
                    # Penalty is the maximum similarity to ANY single already-selected item
                    max_sim = torch.max(sim_matrix[c, selected])
                    
                    # The unified similarity matrix handles the direction internally
                    score = (1 - self.theta) * t2i_sim[c] - self.theta * max_sim
                    
                    if score > best_score:
                        best_score = score
                        best_idx = c
                        
            selected.append(best_idx)
            candidates.remove(best_idx)
            
        return torch.tensor(selected)

    @property
    def get_params(self): return {"theta": self.theta}
    @property
    def get_name(self): return "mmr"


@registry.register_div_method("clustering")
class ClusteringBaseline(BaseDiversificationMethod):
    def __init__(self, num_clusters: int = 40, **kwargs):
        self.num_clusters = num_clusters

    def search(self, *args, **kwargs): pass

    def diversify(self, t2i_sim, ext_data, direction, top_k, image_feats, **kwargs):
        # 1. Process Image Features
        if image_feats.dim() == 3:
            image_feats = image_feats.mean(dim=1)
        image_feats = torch.nn.functional.normalize(image_feats, p=2, dim=-1)
            
        # 2. Process Attribute Features (Appendix C.2: Multiply by s_i and w_i)
        s_i = -1.0 if direction == DivDir.DECREASE else 1.0
        
        if isinstance(ext_data, list):
            w_i = 1.0 / len(ext_data)
            ext_list = [torch.nn.functional.normalize(e.float(), p=2, dim=-1) * (s_i * w_i) for e in ext_data]
            ext_cat = torch.cat(ext_list, dim=-1)
        else:
            w_i = 1.0
            ext_cat = torch.nn.functional.normalize(ext_data.float(), p=2, dim=-1) * (s_i * w_i)

        # 3. Concatenate Visuals + Attributes 
        concat_feats = torch.cat([image_feats, ext_cat], dim=-1).cpu().numpy()
        
        # 4. K-Means Clustering on Unified Feature Space
        n_clusters = min(self.num_clusters, len(t2i_sim))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(concat_feats)
        labels = kmeans.labels_
        
        # 5. Rank clusters based on average relevance
        cluster_scores = {}
        t2i_sim_np = t2i_sim.cpu().numpy()
        for i in range(n_clusters):
            mask = (labels == i)
            if mask.any():
                cluster_scores[i] = t2i_sim_np[mask].mean()
                
        ranked_clusters = sorted(cluster_scores.keys(), key=lambda x: cluster_scores[x], reverse=True)
        
        # 6. Selection Logic
        selected = []
        if direction == DivDir.DECREASE:
            # Suck up all images from the highest-ranked clusters to force redundancy
            for cluster_id in ranked_clusters:
                images_in_cluster = np.where(labels == cluster_id)[0]
                images_in_cluster = sorted(images_in_cluster, key=lambda x: t2i_sim_np[x], reverse=True)
                for img in images_in_cluster:
                    selected.append(img)
                    if len(selected) == top_k:
                        return torch.tensor(selected)
        else:
            # Round robin across clusters to force diversity
            while len(selected) < top_k:
                for cluster_id in ranked_clusters:
                    images_in_cluster = np.where(labels == cluster_id)[0]
                    valid_imgs = [img for img in images_in_cluster if img not in selected]
                    if valid_imgs:
                        best_img = max(valid_imgs, key=lambda x: t2i_sim_np[x])
                        selected.append(best_img)
                        if len(selected) == top_k:
                            return torch.tensor(selected)
                            
        return torch.tensor(selected)

    @property
    def get_params(self): return {"num_clusters": self.num_clusters}
    @property
    def get_name(self): return "clustering"