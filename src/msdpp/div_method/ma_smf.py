import torch
from msdpp import registry
from msdpp.base.divmethod import BaseDiversificationMethod, DivDir

@registry.register_div_method("ma_smf")
class ModelAwareSubmodularMethod(BaseDiversificationMethod):
    def __init__(
            self,
            theta: float = 0.5,
            sigma_geo: float = 15.0,
            sigma_time: float = 1.5,
            ablation_no_norm: bool = False,
            ablation_no_omega: bool = False,
            **kwargs
        ):        # theta acts as the trade-off parameter 'lambda' from the formulation
        self.theta = theta
        self.sigma_geo = sigma_geo
        self.sigma_time = sigma_time
        self.ablation_no_norm = ablation_no_norm
        self.ablation_no_omega = ablation_no_omega

    def search(self, info, direction, t2i_sim, top_k=20, **kwargs):
        """
        Args:
            info (torch.Tensor): [N, num_bins] Probability matrix P(u, i)
            direction: (Unused in greedy loop, controlled via theta trade-off)
            t2i_sim (torch.Tensor): [N] Raw retrieval scores (logits or dot products)
            top_k (int): Number of items to select
        """
        device = info.device
        
        # --- 1. Normalize Relevance Scores (\hat{R}) ---
        if self.ablation_no_norm:
            # ABLATION: Use raw VLM scores directly
            r_hat = t2i_sim
        else:
            # NORMAL: Scale locally over candidate set V to [0, 1]
            rel_min = t2i_sim.min()
            rel_max = t2i_sim.max()
            r_hat = (t2i_sim - rel_min) / (rel_max - rel_min + 1e-8)
        
        # --- 2. Compute Query-Driven Bin Importance (\Omega) ---
        weighted_potential = info * r_hat.unsqueeze(1) 
        omega, _ = torch.max(weighted_potential, dim=0) 
        
        if self.ablation_no_omega:
            # ABLATION: Force all bins to have equal weight of 1.0 (blind coverage)
            omega = torch.ones_like(omega).to(device)
        
        # --- 3. Greedy Optimization Loop ---
        selected = []
        remaining = list(range(len(t2i_sim)))
        n_bins = info.shape[1]
        
        p_covered = torch.zeros(n_bins).to(device)
        
        for _ in range(top_k):
            current_r_hat = r_hat[remaining]
            current_info = info[remaining]
            
            bin_weights = omega * (1 - p_covered)
            cov_gain = torch.matmul(current_info, bin_weights)

            if direction == DivDir.DECREASE:
                # Penalize new coverage to force images into tight clusters
                marginal_gains = (1 - self.theta) * current_r_hat - self.theta * cov_gain
            else:
                # Reward new coverage to force spatial/temporal diversity (Equation 1)
                marginal_gains = (1 - self.theta) * current_r_hat + self.theta * cov_gain    


            # marginal_gains = (1 - self.theta) * current_r_hat + self.theta * cov_gain
            
            best_local_idx = torch.argmax(marginal_gains)
            best_global_idx = remaining.pop(best_local_idx)
            selected.append(best_global_idx)
            
            p_covered = 1 - (1 - p_covered) * (1 - info[best_global_idx])
            
        return torch.tensor(selected)

    @property
    def get_params(self):
        # Adding ablations here ensures the cache manager creates unique files for them!
        return {
            "theta": self.theta,
            "sigma_geo": self.sigma_geo,
            "sigma_time": self.sigma_time,
            "no_norm": self.ablation_no_norm,
            "no_omega": self.ablation_no_omega,
        }

    @property
    def get_name(self):
        return "ma_smf"



# import torch
# from msdpp import registry
# from msdpp.base.divmethod import BaseDiversificationMethod

# @registry.register_div_method("ma_smf")
# class ModelAwareSubmodularMethod(BaseDiversificationMethod):
#     def __init__(self, theta: float = 0.5, **kwargs):
#         # theta acts as the trade-off parameter 'lambda' from the formulation
#         self.theta = theta

#     def search(self, info, direction, t2i_sim, top_k=20, **kwargs):
#         """
#         Args:
#             info (torch.Tensor): [N, num_bins] Probability matrix P(u, i)
#             direction: (Unused in greedy loop, controlled via theta trade-off)
#             t2i_sim (torch.Tensor): [N] Raw retrieval scores (logits or dot products)
#             top_k (int): Number of items to select
#         """
#         device = info.device
        
#         # --- 1. Normalize Relevance Scores (\hat{R}) ---
#         # We normalize locally over the candidate set V to [0, 1]
#         # This ensures compatibility with the coverage term
#         rel_min = t2i_sim.min()
#         rel_max = t2i_sim.max()
#         # Add epsilon to prevent division by zero
#         r_hat = (t2i_sim - rel_min) / (rel_max - rel_min + 1e-8)
        
#         # --- 2. Compute Query-Driven Bin Importance (\Omega) ---
#         # Omega(u, q) = max_{j in V} ( P(u, j) * \hat{R}(j, q) )
#         # This identifies bins that contain at least one highly relevant image
        
#         # shape: [N, bins] = [N, bins] * [N, 1]
#         weighted_potential = info * r_hat.unsqueeze(1) 
        
#         # shape: [bins]
#         omega, _ = torch.max(weighted_potential, dim=0) 
        
#         # --- 3. Greedy Optimization Loop ---
#         selected = []
#         remaining = list(range(len(t2i_sim)))
#         n_bins = info.shape[1]
        
#         # Track cumulative coverage (Probability of Union)
#         p_covered = torch.zeros(n_bins).to(device)
        
#         for _ in range(top_k):
#             # Get data for remaining candidates
#             current_r_hat = r_hat[remaining] # [N_rem]
#             current_info = info[remaining]   # [N_rem, bins]
            
#             # Marginal Gain Calculation
#             # Gain = Omega(u) * (1 - P_covered(u)) * P(u, i)
#             # The term (1 - P_covered) is the "space left" in the bin
#             # Omega weights the value of that space
            
#             # Vectorized: [bins]
#             bin_weights = omega * (1 - p_covered)
            
#             # Matmul: [N_rem, bins] @ [bins] -> [N_rem]
#             cov_gain = torch.matmul(current_info, bin_weights)
            
#             # Total Marginal Gain
#             marginal_gains = (1 - self.theta) * current_r_hat + self.theta * cov_gain
            
#             # Select best
#             best_local_idx = torch.argmax(marginal_gains)
#             best_global_idx = remaining.pop(best_local_idx)
#             selected.append(best_global_idx)
            
#             # Update Coverage State
#             # P_new = 1 - (1 - P_old)(1 - P_selected)
#             p_covered = 1 - (1 - p_covered) * (1 - info[best_global_idx])
            
#         return torch.tensor(selected)

#     @property
#     def get_params(self):
#         return {"theta": self.theta}

#     @property
#     def get_name(self):
#         return "ma_smf"