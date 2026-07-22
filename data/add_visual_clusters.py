"""
Script to add visual clusters to existing datasets.
Usage: python add_visual_clusters.py --dataset_name PP --output_name PP_visualcluster
"""

import argparse
import logging
from pathlib import Path
import pickle

import torch
from datasets import load_from_disk

from msdpp.data.cluster_util import generate_visual_clusters
from msdpp.models import BLIP2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_visual_clusters_to_dataset(
    dataset_path: Path,
    output_path: Path,
    model: str = "Salesforce/blip2-itm-vit-g-coco",
    num_clusters: int = 50,
    batch_size: int = 64,
    num_proc: int = 8,
) -> None:
    """
    Add visual cluster assignments to a dataset.

    Args:
        dataset_path: Path to the base dataset (HF Dataset)
        output_path: Path to save the dataset with clusters
        model: Model name for feature extraction
        num_clusters: Number of visual clusters
        batch_size: Batch size for feature extraction
        num_proc: Number of processes for dataset mapping
    """
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))

    # Initialize model for feature extraction
    logger.info(f"Initializing model: {model}")
    blip2_model = BLIP2(pretrained_model=model, use_itm=0)

    # Extract image features
    logger.info("Extracting image features...")

    def extract_features(batch):
        images = batch["image"]
        with torch.inference_mode():
            feats = blip2_model.extract_features(images)
        return {"image_feats": feats}

    dataset = dataset.map(
        extract_features,
        batched=True,
        batch_size=batch_size,
        num_proc=1,  # Use single process for CUDA
        desc="Extracting image features",
    )

    # Collect all features
    logger.info("Collecting features for clustering...")
    all_feats = torch.stack([f for f in dataset["image_feats"]])

    # Generate visual clusters
    logger.info(f"Generating {num_clusters} visual clusters...")
    cluster_ids, centroids = generate_visual_clusters(
        all_feats, num_clusters=num_clusters
    )

    # Add cluster_id to dataset
    logger.info("Adding cluster assignments to dataset...")

    def add_cluster_id(example, idx):
        example["cluster_id"] = cluster_ids[idx].item()
        return example

    dataset = dataset.map(
        add_cluster_id,
        with_indices=True,
        num_proc=num_proc,
        desc="Adding cluster IDs",
    )

    # Remove image_feats (no longer needed, and saves space)
    dataset = dataset.remove_columns(["image_feats"])

    # Save dataset
    logger.info(f"Saving dataset with clusters to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    # Save centroids for reference
    centroids_path = output_path.parent / f"{output_path.name}_centroids.pt"
    torch.save(centroids, centroids_path)
    logger.info(f"Saved centroids to {centroids_path}")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add visual clusters to an existing dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output dataset with clusters",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-itm-vit-g-coco",
        help="Model for feature extraction",
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

    args = parser.parse_args()

    add_visual_clusters_to_dataset(
        Path(args.dataset_path),
        Path(args.output_path),
        model=args.model,
        num_clusters=args.num_clusters,
        batch_size=args.batch_size,
    )
