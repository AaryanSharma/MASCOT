import torch
import torch.nn.functional as F


_GEO_CENTERS:  dict[tuple, torch.Tensor] = {}
_TIME_CENTERS: dict[tuple, torch.Tensor] = {}


def get_geo_iu_probs(gps_coords: torch.Tensor, grid_size: int = 20, sigma: float = 15.0) -> torch.Tensor:
    key = (grid_size, str(gps_coords.device))
    if key not in _GEO_CENTERS:
        lats = torch.linspace(-90, 90, grid_size, device=gps_coords.device)
        lons = torch.linspace(-180, 180, grid_size, device=gps_coords.device)
        grid_lats, grid_lons = torch.meshgrid(lats, lons, indexing='ij')
        _GEO_CENTERS[key] = torch.stack([grid_lats.flatten(), grid_lons.flatten()], dim=1)
    dist  = torch.cdist(gps_coords.float(), _GEO_CENTERS[key])
    probs = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    probs[probs < 0.01] = 0.0
    return probs


def get_time_iu_probs(hours: torch.Tensor, minutes: torch.Tensor, num_bins: int = 24, sigma: float = 1.5) -> torch.Tensor:
    key = (num_bins, str(hours.device))
    if key not in _TIME_CENTERS:
        _TIME_CENTERS[key] = torch.arange(num_bins, dtype=torch.float32, device=hours.device)
    time_in_hours = (hours + minutes / 60.0).float()
    diff  = torch.abs(time_in_hours.unsqueeze(1) - _TIME_CENTERS[key].unsqueeze(0))
    dist  = torch.minimum(diff, 24.0 - diff)
    probs = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    probs[probs < 0.01] = 0.0
    return probs


def get_coco_iu_probs(
    category_ids_list: list,
    coco_cat_to_bin: dict,
    num_categories: int = 80,
) -> torch.Tensor:
    """
    Multi-hot [N, 80] IU matrix for COCO object categories.
    No smoothing needed — exact set-cover assignment for discrete attributes.
    """
    N = len(category_ids_list)
    probs = torch.zeros(N, num_categories)
    for i, cat_ids in enumerate(category_ids_list):
        for cat_id in cat_ids:
            if cat_id in coco_cat_to_bin:
                probs[i, coco_cat_to_bin[cat_id]] = 1.0
    return probs


def get_cluster_iu_probs(cluster_ids: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    One-hot [N, num_clusters] IU matrix for visual cluster membership.
    Discrete categorical attribute — no smoothing, no bandwidth parameter.
    """
    return F.one_hot(cluster_ids.long(), num_classes=num_clusters).float()
