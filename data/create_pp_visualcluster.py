"""
Script to create PP_visualcluster dataset with visual cluster diversity attribute.

This script:
1. Loads or creates an existing PP dataset (e.g., PP_geo, PP_hour)
2. Generates visual clusters from image embeddings
3. Creates a new variant (PP_visualcluster) with cluster_id field
4. Saves val/test splits as RetrievalDataset objects

Usage:
    python create_pp_visualcluster.py \
        --base_dataset_path /path/to/pp_dataset \
        --output_path /path/to/output \
        --num_clusters 50
"""

import argparse
import logging
from pathlib import Path

import datasets
import torch
from datasets import load_from_disk

from msdpp.schema import RetrievalDataset
from cluster_util import generate_visual_clusters
from msdpp.models.blip2 import Blip2Model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_visualcluster_dataset(
    base_dataset_path: Path,
    output_dir: Path,
    dataset_name: str = "PP_visualcluster",
    num_clusters: int = 50,
    batch_size: int = 64,
    model_name: str = "Salesforce/blip2-itm-vit-g-coco",
    retrieval_words: list[str] = None,
    labels: dict = None,
) -> None:
    """
    Create a visual cluster variant of an existing dataset.
    Complete pipeline: features → clusters → RetrievalDataset splits.

    Args:
        base_dataset_path: Path to base HF Dataset
        output_dir: Directory to save output dataset
        dataset_name: Name prefix for output (e.g., "PP_visualcluster")
        num_clusters: Number of visual clusters
        batch_size: Batch size for feature extraction
        model_name: Model for feature extraction
        retrieval_words: Optional list of retrieval queries
        labels: Optional dict of retrieval labels
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base dataset
    logger.info(f"Loading base dataset from {base_dataset_path}")
    base_dataset = load_from_disk(str(base_dataset_path))

    logger.info(f"Creating {dataset_name} with {num_clusters} visual clusters")
    logger.info(f"Base dataset size: {len(base_dataset)}")

    # Extract image features
    logger.info("Extracting image features using BLIP-2...")
    model = Blip2Model(pretrained_model=model_name, use_itm=False)

    def extract_features(batch):
        images = batch["image"]
        with torch.inference_mode():
            feats = model.extract_features(images)
        return {"image_feats": feats}

    dataset_with_feats = base_dataset.map(
        extract_features,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        desc="Extracting image features",
    )

    # Collect all features for clustering
    logger.info("Collecting features for clustering...")
    all_feats = torch.stack([f for f in dataset_with_feats["image_feats"]])
    logger.info(f"Feature shape: {all_feats.shape}")

    # Generate clusters
    logger.info(f"Generating {num_clusters} visual clusters...")
    cluster_ids, centroids = generate_visual_clusters(
        all_feats,
        num_clusters=num_clusters,
        random_state=42,
        n_init=10,
    )

    logger.info(f"Cluster distribution: {torch.bincount(cluster_ids).tolist()}")

    # Add cluster_id to dataset
    logger.info("Adding cluster_id field to dataset...")

    def add_cluster_id(example, idx):
        example["cluster_id"] = cluster_ids[idx].item()
        return example

    final_dataset = dataset_with_feats.map(
        add_cluster_id,
        with_indices=True,
        desc="Adding cluster IDs",
    )

    # Remove image features (no longer needed)
    final_dataset = final_dataset.remove_columns(["image_feats"])

    # Save full dataset with clusters
    logger.info(f"Saving full dataset to {output_dir / 'full'}")
    full_dir = output_dir / "full"
    final_dataset.save_to_disk(str(full_dir))

    # Save centroids
    centroids_path = output_dir / f"{dataset_name}_centroids.pt"
    torch.save(centroids, centroids_path)
    logger.info(f"Saved centroids to {centroids_path}")

    # Create val/test splits with RetrievalDataset if labels provided
    if labels is not None and retrieval_words is not None:
        logger.info("Creating val/test RetrievalDataset splits...")

        torch.random.set_rng_state(torch.Generator().manual_seed(42).get_state())
        target_ids = torch.randperm(len(final_dataset)).tolist()
        n_val = int(len(final_dataset) * 0.2)

        train_idx = target_ids[:n_val]
        test_idx = target_ids[n_val:]

        val_dataset_hf = final_dataset.select(train_idx)
        test_dataset_hf = final_dataset.select(test_idx)

        # Convert cluster_ids to one-hot vectors
        import torch.nn.functional as F
        train_cluster_ids = cluster_ids[train_idx]
        test_cluster_ids = cluster_ids[test_idx]

        train_ext_data = F.one_hot(train_cluster_ids.long(), num_classes=num_clusters).float()
        test_ext_data = F.one_hot(test_cluster_ids.long(), num_classes=num_clusters).float()

        # Split labels according to train/test indices
        val_labels = {k: v[train_idx] for k, v in labels.items()}
        test_labels = {k: v[test_idx] for k, v in labels.items()}

        # Create RetrievalDataset objects
        val_dataset_ret = RetrievalDataset(
            f"{dataset_name}_val",
            val_dataset_hf,
            retrieval_words,
            val_labels,
            train_ext_data,
        )
        val_dataset_ret.num_clusters = num_clusters

        test_dataset_ret = RetrievalDataset(
            f"{dataset_name}_test",
            test_dataset_hf,
            retrieval_words,
            test_labels,
            test_ext_data,
        )
        test_dataset_ret.num_clusters = num_clusters

        # Save RetrievalDataset splits
        output_dir.mkdir(parents=True, exist_ok=True)
        val_path = output_dir / f"{dataset_name}_val.pkl"
        test_path = output_dir / f"{dataset_name}_test.pkl"

        logger.info(f"Saving val RetrievalDataset to {val_path}")
        with val_path.open("wb") as f:
            torch.save(val_dataset_ret, f)

        logger.info(f"Saving test RetrievalDataset to {test_path}")
        with test_path.open("wb") as f:
            torch.save(test_dataset_ret, f)

        logger.info("RetrievalDataset splits created successfully!")
        logger.info(f"  Val ext_data shape: {train_ext_data.shape}")
        logger.info(f"  Test ext_data shape: {test_ext_data.shape}")
    else:
        logger.warning("No labels/retrieval_words provided - skipping RetrievalDataset creation")
        logger.warning("To create RetrievalDataset splits, pass labels and retrieval_words to this function")

    logger.info("Done!")


def create_retrieval_splits_with_clusters(
    dataset_path: Path,
    output_path: Path,
    dataset_name: str,
    val_captions: list[str],
    test_captions: list[str],
    val_labels: dict,
    test_labels: dict,
    ext_data: torch.Tensor,
    num_clusters: int,
    target_ids: list[int],
    n_val: int,
) -> None:
    """
    Create val/test RetrievalDataset splits from a base dataset with clusters.
    """
    dataset = load_from_disk(str(dataset_path))

    train_idx = target_ids[:n_val]
    test_idx = target_ids[n_val:]

    val_dataset = dataset.select(train_idx)
    test_dataset = dataset.select(test_idx)

    if isinstance(ext_data, torch.Tensor):
        train_ext_data = ext_data[train_idx]
        test_ext_data = ext_data[test_idx]
    else:
        train_ext_data = [e[train_idx] for e in ext_data]
        test_ext_data = [e[test_idx] for e in ext_data]

    # Create RetrievalDataset objects
    val_dataset_ret = RetrievalDataset(
        f"{dataset_name}_val",
        val_dataset,
        val_captions,
        val_labels,
        train_ext_data,
    )
    val_dataset_ret.num_clusters = num_clusters

    test_dataset_ret = RetrievalDataset(
        f"{dataset_name}_test",
        test_dataset,
        test_captions,
        test_labels,
        test_ext_data,
    )
    test_dataset_ret.num_clusters = num_clusters

    # Save splits
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val_path = output_path.parent / f"{dataset_name}_val.pkl"
    test_path = output_path.parent / f"{dataset_name}_test.pkl"

    logger.info(f"Saving val split to {val_path}")
    with val_path.open("wb") as f:
        torch.save(val_dataset_ret, f)

    logger.info(f"Saving test split to {test_path}")
    with test_path.open("wb") as f:
        torch.save(test_dataset_ret, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visual cluster dataset variant"
    )
    parser.add_argument(
        "--base_dataset_path",
        type=str,
        required=True,
        help="Path to base dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PP_visualcluster",
        help="Dataset name",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=50,
        help="Number of visual clusters",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-itm-vit-g-coco",
        help="Model for feature extraction",
    )

    args = parser.parse_args()

    create_visualcluster_dataset(
        base_dataset_path=Path(args.base_dataset_path),
        output_dir=Path(args.output_path),
        dataset_name=args.dataset_name,
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
        model_name=args.model,
    )
