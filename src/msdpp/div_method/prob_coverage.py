# src/msdpp/div_method/prob_coverage.py
import torch
from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod, DivDir

@registry.register_div_method("prob_coverage")
class ProbabilisticCoverageMethod(BaseDiversificationMethod):
    def __init__(self, theta: float = 0.8, sigma_geo: float = 15.0, sigma_time: float = 1.5, **kwargs):
        self.theta = theta
        self.sigma_geo = sigma_geo
        self.sigma_time = sigma_time

    def search(self, info, direction, t2i_sim, top_k=20, **kwargs):
        device = info.device # Ensure we use the GPU if the data is there
        selected = []
        remaining = list(range(len(t2i_sim)))
        p_u = torch.zeros(info.shape[1]).to(device) 

        for _ in range(top_k):
            rel_scores = t2i_sim[remaining]
            # This matmul is much faster on CUDA
            cov_gain = torch.matmul(info[remaining], (1 - p_u))

            if direction == DivDir.DECREASE:
                # Penalize new coverage to force images into tight clusters
                marginal_gains = (1 - self.theta) * rel_scores - self.theta * cov_gain
            else:
                # Reward new coverage to force spatial/temporal diversity
                marginal_gains = (1 - self.theta) * rel_scores + self.theta * cov_gain

            # marginal_gains = (1 - self.theta) * rel_scores + self.theta * cov_gain
            
            best_idx = torch.argmax(marginal_gains)
            best_global = remaining.pop(best_idx)
            selected.append(best_global)
            
            p_u = 1 - (1 - p_u) * (1 - info[best_global])
            
        return torch.tensor(selected)
    @property
    def get_params(self):
        return {
            "theta": self.theta,
            "sigma_geo": self.sigma_geo,
            "sigma_time": self.sigma_time,
        }
    @property
    def get_name(self):
        return "prob_coverage"