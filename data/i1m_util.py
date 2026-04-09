"""
Utility class for Incidents 1M (I1M) dataset preprocessing.
I1M is used for tasks involving shooting location (I1M_geo).
Text queries: disaster types and shooting location types.
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


def _lat_lon_to_xyz(lat: float, lon: float) -> list[float]:
    """Convert lat/lon degrees to unit 3D xyz vector."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = float(np.cos(lat_r) * np.cos(lon_r))
    y = float(np.cos(lat_r) * np.sin(lon_r))
    z = float(np.sin(lat_r))
    return [x, y, z]


class I1MPreprocess:
    def __init__(self, img_save_root: Path) -> None:
        self.img_save_root = img_save_root

    # ------------------------------------------------------------------
    # Image download helpers
    # ------------------------------------------------------------------
    def fix_orientation(self, img: Image.Image) -> Image.Image:
        f = {
            0: lambda img: img, 1: lambda img: img,
            2: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            3: lambda img: img.transpose(Image.Transpose.ROTATE_180),
            4: lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            5: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.ROTATE_90),
            6: lambda img: img.transpose(Image.Transpose.ROTATE_270),
            7: lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.ROTATE_270),
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
            if w < 32 or h < 32:
                return None # Discard tracking pixels and broken 1x1 icons
            if w > max_size or h > max_size:
                if w > h:
                    img = rotated_img.resize((max_size, int(h / w * max_size)))
                else:
                    img = rotated_img.resize((int(w / h * max_size), max_size))
            img = img.convert("RGB")
        except Exception:
            img = None
        return img

    def dl_and_save_img(self, data: dict) -> dict:
        img_root = self.img_save_root
        img_id = str(data.get("id", data.get("image_id", data.get("url", "unknown"))).split("/")[-1].split(".")[0])
        url = data.get("url", data.get("image_url", ""))
        img_path = img_root / f"{img_id}.jpg"
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
    # GPS validation and conversion
    # ------------------------------------------------------------------
    def is_valid_gps(self, lat: float, lon: float) -> bool:
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (lat == 0.0 and lon == 0.0)

    def add_gps_noise(self, lat: float, lon: float, noise_scale: float = 1e-4) -> tuple[float, float]:
        """Add small uniform noise to avoid exact duplicate locations."""
        lat_noise = (torch.rand(1).item() - 0.5) * 2 * noise_scale
        lon_noise = (torch.rand(1).item() - 0.5) * 2 * noise_scale
        return lat + lat_noise, lon + lon_noise

    # ------------------------------------------------------------------
    # Dataset filtering
    # ------------------------------------------------------------------
    def _add_image(self, data: dict) -> dict:
        img_path = self.img_save_root / data["image_path"]
        img = Image.open(img_path)
        data["image"] = img
        return data

    def filter_and_process(
        self,
        raw_dataset: datasets.Dataset,
        lat_field: str = "latitude",
        lon_field: str = "longitude",
        url_field: str = "url",
    ) -> datasets.Dataset:
        """
        Filter images to those with valid GPS, add gps and gps_xyz fields.
        Deduplicates near-identical GPS coordinates with small noise.
        Returns dataset with 'gps', 'gps_xyz', 'image' (HF Image) fields.
        """
        processed = []
        seen_gps: set[tuple] = set()
        has_hf_images = (
            "image" in (raw_dataset.column_names if hasattr(raw_dataset, "column_names") else [])
            and raw_dataset[0].get("image") is not None
        )

        for i in range(len(raw_dataset)):
            item = dict(raw_dataset[i])

            # Extract GPS
            lat = item.get(lat_field)
            lon = item.get(lon_field)
            if lat is None or lon is None:
                continue
            try:
                lat, lon = float(lat), float(lon)
            except (TypeError, ValueError):
                continue

            if not self.is_valid_gps(lat, lon):
                continue

            # Add small noise to exact duplicate locations (following I1M paper methodology)
            gps_key = (round(lat, 5), round(lon, 5))
            if gps_key in seen_gps:
                lat, lon = self.add_gps_noise(lat, lon)
            seen_gps.add(gps_key)

            item["gps"] = [lat, lon]
            item["gps_xyz"] = _lat_lon_to_xyz(lat, lon)

            if has_hf_images:
                # Image already in dataset (will be handled as HF Image type)
                processed.append(item)
            else:
                # Need to download image
                item = self.dl_and_save_img(item)
                if item.get("local_path", "-1") == "-1":
                    continue
                item["image_path"] = Path(item["local_path"]).name
                processed.append(item)

        base_ds = datasets.Dataset.from_list(processed)

        if has_hf_images:
            # Cast existing image column to proper HF Image type if not already
            if not isinstance(base_ds.features.get("image"), datasets.Image):
                base_ds = base_ds.cast_column("image", datasets.Image())
            return base_ds
        else:
            # Load PIL Images from disk and cast
            return (
                base_ds
                .map(self._add_image)
                .cast_column("image", datasets.Image())
            )

    # ------------------------------------------------------------------
    # Label generation from incident/disaster type annotations
    # ------------------------------------------------------------------
    def build_incident_labels(
        self,
        dataset: datasets.Dataset,
        incident_type_field: str = "incident_type",
        place_field: str = "place",
    ) -> tuple[list[str], dict]:
        """
        Build multi-positive labels from incident types and place types.
        Returns query list and labels dict {query: binary_tensor}.
        """
        n = len(dataset)
        type_counts: dict[str, int] = {}

        # Collect all incident types and place types
        for i in range(n):
            item = dataset[i]
            inc_type = item.get(incident_type_field)
            if inc_type and isinstance(inc_type, str) and inc_type.strip():
                type_counts[inc_type.strip()] = type_counts.get(inc_type.strip(), 0) + 1
            place = item.get(place_field)
            if place and isinstance(place, str) and place.strip():
                key = f"place:{place.strip()}"
                type_counts[key] = type_counts.get(key, 0) + 1

        # Use types with at least 2 images as queries
        valid_types = sorted([t for t, c in type_counts.items() if c >= 2])

        if not valid_types:
            raise ValueError("No valid incident/place types found. Check field names.")

        labels: dict[str, torch.Tensor] = {}
        for t in valid_types:
            labels[t] = torch.zeros(n, dtype=torch.int)

        for i in range(n):
            item = dataset[i]
            inc_type = item.get(incident_type_field)
            if inc_type and isinstance(inc_type, str) and inc_type.strip() in labels:
                labels[inc_type.strip()][i] = 1
            place = item.get(place_field)
            if place and isinstance(place, str):
                key = f"place:{place.strip()}"
                if key in labels:
                    labels[key][i] = 1

        return valid_types, labels

    # ------------------------------------------------------------------
    # Val/test split and save
    # ------------------------------------------------------------------
    def split_and_save(
        self,
        dataset: datasets.Dataset,
        save_root: Path,
        n_val_ratio: float = 0.2,
        incident_type_field: str = "incident_type",
        place_field: str = "place",
    ) -> None:
        """Split dataset into val/test and save as pkl files."""
        n = len(dataset)
        torch.random.set_rng_state(torch.Generator().manual_seed(42).get_state())
        perm = torch.randperm(n).tolist()
        n_val = int(n * n_val_ratio)

        val_indices = perm[:n_val]
        test_indices = perm[n_val:]

        val_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)

        # Build geo xyz ext_data
        val_xyz = torch.tensor(
            [val_dataset[i]["gps_xyz"] for i in range(len(val_dataset))],
            dtype=torch.float32
        )
        test_xyz = torch.tensor(
            [test_dataset[i]["gps_xyz"] for i in range(len(test_dataset))],
            dtype=torch.float32
        )
        # Normalize to unit sphere
        val_ext = torch.nn.functional.normalize(val_xyz, dim=-1)
        test_ext = torch.nn.functional.normalize(test_xyz, dim=-1)

        # Build labels
        print("Building val labels...")
        val_queries, val_labels = self.build_incident_labels(
            val_dataset, incident_type_field, place_field
        )
        print("Building test labels...")
        test_queries, test_labels = self.build_incident_labels(
            test_dataset, incident_type_field, place_field
        )

        # Use intersection of valid queries
        common_queries = [q for q in val_queries if q in test_labels]
        val_labels = {q: val_labels[q] for q in common_queries}
        test_labels = {q: test_labels[q] for q in common_queries}

        print(f"Valid queries ({len(common_queries)}): {common_queries[:10]}...")
        print(f"Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

        val_ret = RetrievalDataset(
            name="I1M_geo_val",
            dataset=val_dataset,
            retrieval_words=common_queries,
            labels=val_labels,
            ext_data=val_ext,
        )
        test_ret = RetrievalDataset(
            name="I1M_geo_test",
            dataset=test_dataset,
            retrieval_words=common_queries,
            labels=test_labels,
            ext_data=test_ext,
        )

        val_path = save_root / "I1M_geo_val.pkl"
        test_path = save_root / "I1M_geo_test.pkl"

        with val_path.open("wb") as f:
            torch.save(val_ret, f)
        with test_path.open("wb") as f:
            torch.save(test_ret, f)

        print(f"Saved: {val_path}")
        print(f"Saved: {test_path}")
