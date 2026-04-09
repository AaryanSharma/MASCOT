
# src/msdpp/data.py
import torch


def datetime_embeds(hours: torch.Tensor, minutes: torch.Tensor) -> torch.Tensor:
    time_embeds = hours * 60 + minutes

    max_embeds = 23 * 60 + 59

    return torch.stack(
        [
            torch.sin(time_embeds / max_embeds * 2 * torch.pi),
            torch.cos(time_embeds / max_embeds * 2 * torch.pi),
        ],
        1,
    )

# def get_time_iu_probs(hours: torch.Tensor, minutes: torch.Tensor, num_bins: int = 24) -> torch.Tensor:
#     """Converts raw time into probabilistic Information Units."""
#     time_in_hours = hours + (minutes / 60.0)
#     probs = torch.zeros((len(hours), num_bins))
#     for i, t in enumerate(time_in_hours):
#         bin_idx = int(t % num_bins)
#         # Soft-binning to handle uncertainty
#         probs[i, bin_idx] = 0.8
#         probs[i, (bin_idx + 1) % num_bins] = 0.1
#         probs[i, (bin_idx - 1) % num_bins] = 0.1
#     return probs

# def get_geo_iu_probs(gps_coords: torch.Tensor, grid_size: int = 10) -> torch.Tensor:
#     """Converts GPS [lat, lon] into a grid of probabilistic Information Units."""
#     # Assuming gps_coords is a tensor of shape (N, 2)
#     lats = gps_coords[:, 0]
#     lons = gps_coords[:, 1]
    
#     lat_min, lat_max = -90.0, 90.0
#     lon_min, lon_max = -180.0, 180.0
    
#     lat_bins = ((lats - lat_min) / (lat_max - lat_min) * (grid_size - 1)).clamp(0, grid_size - 1).long()
#     lon_bins = ((lons - lon_min) / (lon_max - lon_min) * (grid_size - 1)).clamp(0, grid_size - 1).long()
    
#     num_bins = grid_size * grid_size
#     probs = torch.zeros((len(lats), num_bins))
#     for i in range(len(lats)):
#         bin_idx = lat_bins[i] * grid_size + lon_bins[i]
#         probs[i, bin_idx] = 1.0 # Can be smoothed for neighborhood coverage
#     return probs

import torch

def get_time_iu_probs(hours: torch.Tensor, minutes: torch.Tensor, num_bins: int = 24, sigma: float = 1.5) -> torch.Tensor:
    time_in_hours = hours + (minutes / 60.0) 
    centers = torch.arange(num_bins).float().to(hours.device)
    
    diff = torch.abs(time_in_hours.unsqueeze(1) - centers.unsqueeze(0))
    dist = torch.minimum(diff, 24.0 - diff)
    
    probs = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    
    # Any probability less than 1% (0.01) becomes exactly 0.0
    probs[probs < 0.01] = 0.0 
    
    return probs

def get_geo_iu_probs(gps_coords: torch.Tensor, grid_size: int = 20, sigma: float = 15.0) -> torch.Tensor:
    lats = torch.linspace(-90, 90, grid_size) 
    lons = torch.linspace(-180, 180, grid_size)
    grid_lats, grid_lons = torch.meshgrid(lats, lons, indexing='ij')
    
    centers = torch.stack([grid_lats.flatten(), grid_lons.flatten()], dim=1).to(gps_coords.device)
    
    dist = torch.cdist(gps_coords.float(), centers.float())
    probs = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    
    # Any probability less than 1% (0.01) becomes exactly 0.0
    # This turns the dense matrix back into a highly efficient sparse matrix!
    probs[probs < 0.01] = 0.0 
    
    return probs