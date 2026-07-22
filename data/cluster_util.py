"""Visual clustering utility for generating cluster assignments from image embeddings."""

import torch
from sklearn.cluster import KMeans


def generate_visual_clusters(
    image_feats: torch.Tensor,
    num_clusters: int = 50,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate visual clusters from image feature embeddings using k-means.

    Args:
        image_feats: Image feature embeddings of shape [N, D] (e.g., BLIP-2 features)
        num_clusters: Number of clusters (default: 50)
        random_state: Random seed for reproducibility
        n_init: Number of initializations for k-means

    Returns:
        cluster_ids: Tensor of shape [N] with cluster assignments (0 to num_clusters-1)
        centroids: Tensor of shape [num_clusters, D] with cluster centroids
    """
    if isinstance(image_feats, torch.Tensor):
        feats_np = image_feats.cpu().numpy()
    else:
        feats_np = image_feats

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    cluster_ids_np = kmeans.fit_predict(feats_np)
    cluster_ids = torch.from_numpy(cluster_ids_np).long()
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()

    return cluster_ids, centroids
