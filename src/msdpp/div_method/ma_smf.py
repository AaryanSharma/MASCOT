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

    def search(self, info, direction, t2i_sim, top_k=20,
               attr_boundaries=None, per_attribute_directions=None, **kwargs):
        """
        Args:
            info (torch.Tensor): [N, num_bins] Probability matrix P(u, i)
            direction (DivDir): Global direction; ignored when per_attribute_directions given.
            t2i_sim (torch.Tensor): [N] Raw retrieval scores (logits or dot products)
            top_k (int): Number of items to select
            attr_boundaries (list[int] | None): Bin-axis split points, e.g. [0, 400, 424].
                Required when per_attribute_directions is not None.
            per_attribute_directions (list[DivDir] | None): Per-attribute direction list,
                one entry per attribute slice defined by attr_boundaries.  When provided,
                the coverage gain is computed per slice and signed independently, enabling
                mixed-direction diversification (e.g. geo↑ + time↓).  When None the
                original scalar-direction logic is used unchanged (backward-compatible).
        """
        device = info.device

        # --- 1. Normalize Relevance Scores (\hat{R}) ---
        if self.ablation_no_norm:
            r_hat = t2i_sim
        else:
            rel_min = t2i_sim.min()
            rel_max = t2i_sim.max()
            r_hat = (t2i_sim - rel_min) / (rel_max - rel_min + 1e-8)

        # --- 2. Compute Query-Driven Bin Importance (\Omega) ---
        weighted_potential = info * r_hat.unsqueeze(1)
        omega, _ = torch.max(weighted_potential, dim=0)

        if self.ablation_no_omega:
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

            if per_attribute_directions is not None:
                # Per-attribute signed coverage gain (mixed-direction).
                # Each attribute slice [start:end] gets its own sign so that
                # e.g. geo bins reward coverage while time bins penalise it.
                total_cov_gain = torch.zeros(len(remaining), device=device)
                for (start, end), d in zip(
                    zip(attr_boundaries[:-1], attr_boundaries[1:]),
                    per_attribute_directions,
                ):
                    attr_gain = torch.matmul(current_info[:, start:end], bin_weights[start:end])
                    sign = -1 if d == DivDir.DECREASE else 1
                    total_cov_gain += sign * attr_gain
                marginal_gains = (1 - self.theta) * current_r_hat + self.theta * total_cov_gain
            elif direction == DivDir.DECREASE:
                cov_gain = torch.matmul(current_info, bin_weights)
                marginal_gains = (1 - self.theta) * current_r_hat - self.theta * cov_gain
            else:
                cov_gain = torch.matmul(current_info, bin_weights)
                marginal_gains = (1 - self.theta) * current_r_hat + self.theta * cov_gain

            best_local_idx = torch.argmax(marginal_gains)
            best_global_idx = remaining.pop(best_local_idx)
            selected.append(best_global_idx)

            p_covered = 1 - (1 - p_covered) * (1 - info[best_global_idx])

        return torch.tensor(selected)

    def search_vec(self, info, direction, t2i_sim, top_k=20, **kwargs):
        """Vectorized variant of search(). Eliminates Python list pop and
        list-to-LongTensor overhead by using a bool mask over all N rows.
        One argmax CPU sync per iteration is unavoidable (needed for p_covered
        update). Handles only the scalar-direction case; per_attribute_directions
        is not supported (falls back to search() if needed).
        Results are identical to search() for the same inputs.
        """
        device = info.device
        N = info.shape[0]

        # 1. Normalize R_hat (identical to search())
        if self.ablation_no_norm:
            r_hat = t2i_sim
        else:
            r_hat = (t2i_sim - t2i_sim.min()) / \
                    (t2i_sim.max() - t2i_sim.min() + 1e-8)

        # 2. Compute Omega (identical to search())
        omega, _ = torch.max(info * r_hat.unsqueeze(1), dim=0)
        if self.ablation_no_omega:
            omega = torch.ones_like(omega)

        # 3. Vectorized greedy loop
        # - Pre-fold omega and pre-scale r_hat: removes per-iteration [N,M]/[N] multiplies.
        # - Preserve info.dtype so float16 callers keep half-precision for the hot matmul.
        # - Keep info_f32 for p_covered update: avoids .float() cast on every iteration.
        # - Pre-allocate one_minus_pc buffer: avoids per-iteration (1-p_covered) allocation.
        compute_dtype = info.dtype
        sign       = -1.0 if direction == DivDir.DECREASE else 1.0
        info_omega = (info * omega).to(compute_dtype)    # [N, M]
        info_f32   = info.float()                        # [N, M] float32 for p_covered update
        rel_r      = ((1 - self.theta) * r_hat).to(compute_dtype)   # [N]
        s_theta    = sign * self.theta

        available     = torch.ones(N, dtype=torch.bool, device=device)
        p_covered     = torch.zeros(info.shape[1], dtype=torch.float32, device=device)
        one_minus_pc  = torch.empty(info.shape[1], dtype=compute_dtype, device=device)
        selected      = []

        for _ in range(top_k):
            # sub into pre-allocated buffer; p_covered is float32, out is compute_dtype —
            # PyTorch casts the result to out.dtype before writing, so precision is preserved.
            torch.sub(1.0, p_covered, out=one_minus_pc)
            cov_gain = info_omega @ one_minus_pc                     # [N]
            gains    = rel_r + s_theta * cov_gain
            gains    = gains.masked_fill(~available, float('-inf'))

            best_idx = int(gains.argmax())                           # 1 CPU sync
            selected.append(best_idx)
            available[best_idx] = False
            p_covered = 1 - (1 - p_covered) * (1 - info_f32[best_idx])

        return torch.tensor(selected, device=device)

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