"""
Utility class for Visual Genome dataset preprocessing.
VG is used for tasks involving shooting time (VG_hour).
Text queries: 15 dominant object names.
"""
import contextlib
import time
import warnings
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from PIL.Image import UnidentifiedImageError
from ssl import SSLError

import datasets

from msdpp.schema import RetrievalDataset

warnings.filterwarnings("ignore")

VG_OBJECT_QUERIES = [
    "person", "building", "table", "vehicle", "animal",
    "tree", "sky", "road", "chair", "light",
    "car", "desk", "bird", "apple", "dog",
]


class VGPreprocess:
    def __init__(self, img_save_root: Path) -> None:
        self.img_save_root = img_save_root

    # ------------------------------------------------------------------
    # Image download helpers
    # ------------------------------------------------------------------
    def fix_orientation(self, img: Image.Image) -> Image.Image:
        f = {
            0: lambda img: img,
            1: lambda img: img,
            2: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            3: lambda img: img.transpose(Image.Transpose.ROTATE_180),
            4: lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            5: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                Image.Transpose.ROTATE_90
            ),
            6: lambda img: img.transpose(Image.Transpose.ROTATE_270),
            7: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(
                Image.Transpose.ROTATE_270
            ),
            8: lambda img: img.transpose(Image.Transpose.ROTATE_90),
        }
        if not hasattr(img, "_getexif"):
            return img
        exif = img._getexif()
        if exif is None:
            return img
        orientation = exif.get(0x112, 1)
        return f[orientation](img)

    def dl_img(self, url: str) -> Image.Image | None:
        max_size = 1024
        try:
            response = requests.get(url, stream=True, verify=False, timeout=15)
            org_img = Image.open(response.raw)
            rotated_img = org_img
            with contextlib.suppress(KeyError):
                rotated_img = self.fix_orientation(org_img)
            img = rotated_img
            w, h = rotated_img.size
            if w > max_size or h > max_size:
                if w > h:
                    img = rotated_img.resize((max_size, int(h / w * max_size)))
                else:
                    img = rotated_img.resize((int(w / h * max_size), max_size))
            img = img.convert("RGB")
        except (UnidentifiedImageError, SSLError, requests.exceptions.SSLError,
                requests.exceptions.ProxyError, requests.exceptions.ConnectionError,
                requests.exceptions.TooManyRedirects, Exception):
            img = None
        return img

    def dl_and_save_img(self, data: dict) -> dict:
        img_root = self.img_save_root
        image_id = data["image_id"]
        url = data["url"]
        img_path = img_root / f"{image_id}.jpg"
        if img_path.exists():
            data["local_path"] = str(img_path)
            return data
        img = None
        for _ in range(3):
            img = self.dl_img(url)
            if img is not None:
                break
            time.sleep(3)
        if img is not None:
            try:
                img.save(img_path, "JPEG")
                data["local_path"] = str(img_path)
            except OSError:
                data["local_path"] = "-1"
        else:
            data["local_path"] = "-1"
        return data

    # ------------------------------------------------------------------
    # EXIF time extraction
    # ------------------------------------------------------------------
    def extract_exif_time(self, img_path: str) -> dict | None:
        """Extract hour and minute from image EXIF datetime."""
        try:
            img = Image.open(img_path)
            exif_data = img._getexif()
            if exif_data is None:
                return None
            # Tag 36867 = DateTimeOriginal, 306 = DateTime
            datetime_str = exif_data.get(36867) or exif_data.get(306)
            if not datetime_str:
                return None
            # Format: "YYYY:MM:DD HH:MM:SS"
            parts = datetime_str.strip().split(" ")
            if len(parts) < 2:
                return None
            time_parts = parts[1].split(":")
            if len(time_parts) < 2:
                return None
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            return {"hour": hour, "minute": minute}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Label generation from VG object annotations
    # ------------------------------------------------------------------
    def build_object_labels(
        self, dataset: datasets.Dataset
    ) -> tuple[list[str], dict]:
        """
        Build multi-positive labels for 15 VG object categories.
        Uses exact name matching against VG object annotations.
        Returns query list and labels dict {query: binary_tensor}.
        """
        n = len(dataset)
        labels: dict[str, torch.Tensor] = {}
        for query in VG_OBJECT_QUERIES:
            labels[query] = torch.zeros(n, dtype=torch.int)

        for i in range(n):
            objects = dataset[i].get("objects", [])
            obj_names_lower = set()
            for obj in objects:
                name = obj.get("name", "").lower().strip()
                obj_names_lower.add(name)

            for query in VG_OBJECT_QUERIES:
                if query in obj_names_lower:
                    labels[query][i] = 1

        # Remove queries with no positive images
        valid_queries = [q for q in VG_OBJECT_QUERIES if labels[q].sum() > 0]
        labels = {q: labels[q] for q in valid_queries}
        return valid_queries, labels

    # ------------------------------------------------------------------
    # Dataset filtering
    # ------------------------------------------------------------------
    def _add_image(self, data: dict) -> dict:
        img_path = self.img_save_root / data["image_path"]
        img = Image.open(img_path)
        data["image"] = img
        return data

    def filter_and_process(
        self, raw_dataset: datasets.Dataset
    ) -> datasets.Dataset:
        """
        Download images, extract EXIF time, filter to those with valid time.
        Returns dataset with 'hour', 'minute', 'image' (HF Image) fields.
        """
        processed = []
        for i in range(len(raw_dataset)):
            item = dict(raw_dataset[i])
            item = self.dl_and_save_img(item)
            if item["local_path"] == "-1":
                continue
            time_info = self.extract_exif_time(item["local_path"])
            if time_info is None:
                continue
            item["hour"] = time_info["hour"]
            item["minute"] = time_info["minute"]
            # Store filename relative to img_save_root for later loading
            item["image_path"] = Path(item["local_path"]).name
            processed.append(item)

        base_ds = datasets.Dataset.from_list(processed)
        # Load PIL Images and cast to HuggingFace Image type
        return (
            base_ds
            .map(self._add_image)
            .cast_column("image", datasets.Image())
        )

    # ------------------------------------------------------------------
    # Val/test split and save
    # ------------------------------------------------------------------
    def split_and_save(
        self,
        dataset: datasets.Dataset,
        save_root: Path,
        n_val_ratio: float = 0.2,
    ) -> None:
        """Split dataset into val/test and save as pkl files."""
        from msdpp.data import datetime_embeds

        n = len(dataset)
        torch.random.set_rng_state(torch.Generator().manual_seed(42).get_state())
        perm = torch.randperm(n).tolist()
        n_val = int(n * n_val_ratio)

        val_indices = perm[:n_val]
        test_indices = perm[n_val:]

        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)

        # Build time embeddings for ext_data
        hours_val = torch.tensor([val_dataset[i]["hour"] for i in range(len(val_dataset))], dtype=torch.float32)
        mins_val = torch.tensor([val_dataset[i]["minute"] for i in range(len(val_dataset))], dtype=torch.float32)
        val_ext = datetime_embeds(hours_val, mins_val)

        hours_test = torch.tensor([test_dataset[i]["hour"] for i in range(len(test_dataset))], dtype=torch.float32)
        mins_test = torch.tensor([test_dataset[i]["minute"] for i in range(len(test_dataset))], dtype=torch.float32)
        test_ext = datetime_embeds(hours_test, mins_test)

        # Build labels on val_dataset and test_dataset
        print("Building val labels...")
        val_queries, val_labels = self.build_object_labels(val_dataset)
        print("Building test labels...")
        test_queries, test_labels = self.build_object_labels(test_dataset)

        # Use intersection of valid queries
        common_queries = [q for q in val_queries if q in test_labels]
        val_labels = {q: val_labels[q] for q in common_queries}
        test_labels = {q: test_labels[q] for q in common_queries}

        print(f"Valid queries: {common_queries}")
        print(f"Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

        val_ret = RetrievalDataset(
            name="VG_hour_val",
            dataset=val_dataset,
            retrieval_words=common_queries,
            labels=val_labels,
            ext_data=val_ext,
        )
        test_ret = RetrievalDataset(
            name="VG_hour_test",
            dataset=test_dataset,
            retrieval_words=common_queries,
            labels=test_labels,
            ext_data=test_ext,
        )

        val_path = save_root / "VG_hour_val.pkl"
        test_path = save_root / "VG_hour_test.pkl"

        with val_path.open("wb") as f:
            torch.save(val_ret, f)
        with test_path.open("wb") as f:
            torch.save(test_ret, f)

        print(f"Saved: {val_path}")
        print(f"Saved: {test_path}")