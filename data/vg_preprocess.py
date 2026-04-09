"""
Visual Genome (VG) dataset preprocessing for shooting time tasks.

Uses VG annotation JSON files directly (lightweight, ~25MB total) instead of
the full HuggingFace dataset (9.73GB).

Steps:
1. Download image_data.json.zip + objects.json.zip from VG servers
2. For each image, download and check for EXIF DateTimeOriginal
3. Build dataset of ~649 images with valid EXIF time
4. Text queries: 15 dominant object names from VG annotations
5. Saves VG_hour_val.pkl and VG_hour_test.pkl
"""
import io
import json
import os
import warnings
import zipfile
from pathlib import Path

import requests
import torch
from PIL import Image
from tqdm import tqdm

from vg_util import VGPreprocess

warnings.filterwarnings("ignore")

HF_HOME = Path(os.environ.get("HF_HOME", "./"))
save_task_data_root = HF_HOME / "tasks"
img_save_root = HF_HOME / "images" / "vg"

save_task_data_root.mkdir(parents=True, exist_ok=True)
img_save_root.mkdir(parents=True, exist_ok=True)

preprocessor = VGPreprocess(img_save_root=img_save_root)

# ------------------------------------------------------------------
# Step 1: Download VG metadata JSONs (lightweight)
# ------------------------------------------------------------------
VG_BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"

# Candidate URLs for each file (tried in order)
IMAGE_DATA_URLS = [
    f"{VG_BASE}/image_data.json.zip",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/image_data.json.zip",
]
OBJECTS_URLS = [
    f"{VG_BASE}/objects.json.zip",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/objects.json.zip",
]


def download_and_extract_json(urls: list[str], filename: str, cache_dir: Path) -> list | dict:
    cache_path = cache_dir / filename
    if cache_path.exists():
        print(f"Loading cached {filename}...")
        with cache_path.open() as f:
            return json.load(f)
    last_err = None
    for url in urls:
        try:
            print(f"Downloading {url} ...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            content = response.content
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                json_name = next(n for n in zf.namelist() if n.endswith(".json"))
                data = json.loads(zf.read(json_name))
            with cache_path.open("w") as f:
                json.dump(data, f)
            print(f"Saved to {cache_path}")
            return data
        except Exception as e:
            print(f"  Failed ({url}): {e}")
            last_err = e
    raise RuntimeError(f"Could not download {filename}. Last error: {last_err}")


cache_dir = HF_HOME / "vg_metadata"
cache_dir.mkdir(parents=True, exist_ok=True)

print("Fetching VG image_data.json (~17 MB)...")
image_data = download_and_extract_json(IMAGE_DATA_URLS, "image_data.json", cache_dir)
print(f"Total VG images in metadata: {len(image_data)}")

print("Fetching VG objects.json (~349 MB, one-time download)...")
objects_data = download_and_extract_json(OBJECTS_URLS, "objects.json", cache_dir)
# Build image_id → objects lookup
print("Building image_id → objects index...")
img_objects: dict[int, list] = {}
for entry in objects_data:
    img_objects[entry["image_id"]] = entry.get("objects", [])

# ------------------------------------------------------------------
# Step 2: Scan all images for EXIF time using parallel range requests
#
# Key optimisation: JPEG EXIF is always in the first ~64 KB of the file.
# A Range request avoids downloading the full image (~500 KB–2 MB each).
# With N_WORKERS parallel workers this scans 108 K images in ~10–20 min.
# Progress is saved to a cache file so interrupted runs can resume.
# ------------------------------------------------------------------
import concurrent.futures

N_WORKERS = 64          # parallel HTTP workers
EXIF_RANGE_BYTES = 65535  # first 64 KB is always enough for JPEG EXIF

SCAN_CACHE = cache_dir / "exif_scan_cache.json"


def check_exif_fast(meta: dict) -> dict | None:
    """
    Fetch only the first 64 KB of the image and extract EXIF datetime.
    Returns a dict with hour/minute on success, None otherwise.
    """
    img_id = meta["image_id"]
    url = meta.get("url", "")
    if not url:
        return None

    # If already downloaded to disk, use the cached file
    img_path = img_save_root / f"{img_id}.jpg"
    if img_path.exists():
        result = preprocessor.extract_exif_time(str(img_path))
        if result:
            return {"image_id": img_id, "url": url, **result}
        return None

    try:
        resp = requests.get(
            url,
            headers={"Range": f"bytes=0-{EXIF_RANGE_BYTES}"},
            timeout=10,
            verify=False,
        )
        if resp.status_code not in (200, 206):
            return None

        img_bytes = io.BytesIO(resp.content)
        img = Image.open(img_bytes)

        # Try to get EXIF from partial data
        exif_data = getattr(img, "_getexif", lambda: None)()
        if not exif_data:
            return None

        datetime_str = exif_data.get(36867) or exif_data.get(306)  # DateTimeOriginal / DateTime
        if not datetime_str:
            return None

        parts = str(datetime_str).strip().split(" ")
        if len(parts) < 2:
            return None
        time_parts = parts[1].split(":")
        if len(time_parts) < 2:
            return None
        hour, minute = int(time_parts[0]), int(time_parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None

        return {"image_id": img_id, "url": url, "hour": hour, "minute": minute}
    except Exception:
        return None


# Load previous scan progress if available
already_scanned: dict[int, dict | None] = {}
if SCAN_CACHE.exists():
    with SCAN_CACHE.open() as f:
        raw_cache = json.load(f)
    # Restore: keys stored as strings in JSON
    already_scanned = {int(k): v for k, v in raw_cache.items()}
    print(f"Resuming scan: {len(already_scanned)} images already checked, "
          f"{sum(v is not None for v in already_scanned.values())} with EXIF.")

todo = [m for m in image_data if m["image_id"] not in already_scanned]
print(f"\nScanning {len(todo)} remaining images for EXIF time "
      f"({N_WORKERS} workers, range requests)...")

SAVE_INTERVAL = 2000  # save progress every N items


def _scan_batch(batch: list[dict]) -> list[tuple[int, dict | None]]:
    results = []
    for meta in batch:
        results.append((meta["image_id"], check_exif_fast(meta)))
    return results


with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(check_exif_fast, meta): meta["image_id"] for meta in todo}
    n_done = 0
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Scanning images",
    ):
        img_id = futures[future]
        result = future.result()
        already_scanned[img_id] = result
        n_done += 1
        if n_done % SAVE_INTERVAL == 0:
            with SCAN_CACHE.open("w") as f:
                json.dump({str(k): v for k, v in already_scanned.items()}, f)

# Final save of scan cache
with SCAN_CACHE.open("w") as f:
    json.dump({str(k): v for k, v in already_scanned.items()}, f)

exif_hits = {img_id: info for img_id, info in already_scanned.items() if info is not None}
print(f"\nFound {len(exif_hits)} VG images with EXIF time data (expected ~649)")

# Now download full images only for the hits
from PIL import Image as PILImage

processed = []
print(f"Downloading {len(exif_hits)} full images with EXIF time...")
for img_id, time_info in tqdm(exif_hits.items(), desc="Downloading EXIF images"):
    url = time_info["url"]
    img_path = img_save_root / f"{img_id}.jpg"

    if not img_path.exists():
        img = preprocessor.dl_img(url)
        if img is None:
            continue
        try:
            img.save(img_path, "JPEG")
        except OSError:
            continue

    processed.append({
        "image_id": img_id,
        "url": url,
        "hour": time_info["hour"],
        "minute": time_info["minute"],
        "image_path": f"{img_id}.jpg",
        "objects": img_objects.get(img_id, []),
    })

print(f"\nFound {len(processed)} VG images with EXIF time data (expected ~649)")

if len(processed) == 0:
    raise RuntimeError(
        "No VG images with EXIF time data found.\n"
        "Check that image URLs are accessible and images have EXIF DateTimeOriginal."
    )

# ------------------------------------------------------------------
# Step 3: Build HuggingFace dataset and save task pkl files
# ------------------------------------------------------------------
import datasets

print("Building HuggingFace dataset from processed images...")
base_ds = datasets.Dataset.from_list(processed)
full_dataset = (
    base_ds
    .map(preprocessor._add_image, desc="Loading images")
    .cast_column("image", datasets.Image())
)

print("Splitting and saving task datasets...")
preprocessor.split_and_save(
    dataset=full_dataset,
    save_root=save_task_data_root,
    n_val_ratio=0.2,
)

print("VG preprocessing complete.")
