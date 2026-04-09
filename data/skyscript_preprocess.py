"""
SkyScript dataset preprocessing for geo and/or temporal diversity tasks.

Source: SkyScript test set (30K CLIP-filtered pairs) from S3.
  Images:   remote sensing (satellite/aerial) from Google Earth Engine
  Captions: auto-generated from OpenStreetMap tags (focus object)
  GPS:      center of image bounding box (from meta .pickle file)
  Time:     acquisition time (year, month, day, hour, minute) from GEE

Downloads from S3 on first run; everything is cached locally under HF_HOME.

Steps:
  1. Download SkyScript_test_30K CSV from S3
  2. For each row: download meta .pickle (GPS + time) + image .jpg
  3. Filter to images with valid GPS and/or valid time
  4. Deduplicate captions (single-GT: 1 caption → 1 image)
  5. Build HuggingFace dataset, 20/80 val/test split
  6. Save SkyScript_geo_val/test.pkl, SkyScript_hour_val/test.pkl,
     SkyScript_geo_hour_val/test.pkl

Outputs (under $HF_HOME/tasks/):
  SkyScript_geo_val.pkl / SkyScript_geo_test.pkl
  SkyScript_hour_val.pkl / SkyScript_hour_test.pkl
  SkyScript_geo_hour_val.pkl / SkyScript_geo_hour_test.pkl
"""
import argparse
import csv
import io
import json
import os
import pickle
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from skyscript_util import (
    CSV_URLS,
    IMAGES_ZIP_URL,
    META_ZIP_URL,
    RemoteZipReader,
    download_bytes,
    download_image_from_bytes,
    gps_from_meta,
    lat_lon_to_xyz,
    meta_path_from_img_path,
    s3_zip_url,
    save_skyscript_splits,
    time_from_meta,
    zip_num_from_filepath,
)

# ------------------------------------------------------------------
# CLI args
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SkyScript preprocessing")
parser.add_argument("--limit",   type=int, default=20_000,
                    help="Max images to keep after filtering (default: 20000)")
parser.add_argument("--seed",    type=int, default=42,
                    help="Random seed for row shuffling before limit (default: 42)")
parser.add_argument("--workers", type=int, default=128,
                    help="Parallel download workers (default: 128)")
args = parser.parse_args()

LIMIT       = args.limit
SEED        = args.seed
N_WORKERS   = args.workers
SAVE_INTERVAL = 500

print(f"Config: limit={LIMIT}  seed={SEED}  workers={N_WORKERS}")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
HF_HOME = Path(os.environ.get("HF_HOME", "./"))
save_task_data_root = HF_HOME / "tasks"
img_save_root       = HF_HOME / "images" / "skyscript"
meta_cache_dir      = HF_HOME / "skyscript_metadata"
csv_cache_dir       = HF_HOME / "skyscript_csv"

for d in [save_task_data_root, img_save_root, meta_cache_dir, csv_cache_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Step 1: Download CSV
# ------------------------------------------------------------------
CSV_CACHE = csv_cache_dir / "SkyScript_test_30K.csv"

if not CSV_CACHE.exists():
    print(f"Downloading SkyScript test CSV...")
    data = download_bytes(CSV_URLS["test_30k"], timeout=60)
    if data is None:
        raise RuntimeError(f"Failed to download CSV from {CSV_URLS['test_30k']}")
    CSV_CACHE.write_bytes(data)
    print(f"  Saved to {CSV_CACHE}")
else:
    print(f"Using cached CSV: {CSV_CACHE}")

with CSV_CACHE.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

print(f"CSV rows: {len(csv_rows)}  |  columns: {list(csv_rows[0].keys())}")

# Shuffle with fixed seed and cap at LIMIT rows so we only download
# what we need (reproducible subset of the 30K test set)
random.seed(SEED)
random.shuffle(csv_rows)
csv_rows = csv_rows[:LIMIT]
print(f"Using {len(csv_rows)} rows (limit={LIMIT}, seed={SEED})")

# ------------------------------------------------------------------
# Step 2: Download meta files (GPS + time) via RemoteZipReader in parallel
# ------------------------------------------------------------------
META_STATUS_FILE = meta_cache_dir / "meta_status.json"
meta_status: dict[str, dict | None] = {}

if META_STATUS_FILE.exists():
    with META_STATUS_FILE.open() as f:
        meta_status = json.load(f)
    n_hits = sum(1 for v in meta_status.values() if v is not None)
    print(f"Meta cache: {len(meta_status)} entries, {n_hits} with valid GPS/time")

# Build list of uncached filepaths
todo_rows = [r for r in csv_rows if r["filepath"] not in meta_status]
print(f"Downloading meta for {len(todo_rows)} images ({N_WORKERS} workers)...")

# Group rows by zip number for efficient loading
zip_groups = {}
for row in todo_rows:
    n = zip_num_from_filepath(row["filepath"])
    if n not in zip_groups:
        zip_groups[n] = []
    zip_groups[n].append(row)

print(f"Meta zips: {sorted(zip_groups.keys())}")

# Load each meta zip and read files in parallel
for zip_n in sorted(zip_groups.keys()):
    rows_in_zip = zip_groups[zip_n]
    meta_zip_url = META_ZIP_URL.format(n=zip_n)
    print(f"\nLoading meta{zip_n}.zip ({len(rows_in_zip)} rows)...")

    reader = RemoteZipReader(meta_zip_url)
    try:
        reader.load_index()
    except Exception as e:
        print(f"  Failed to load index: {e}")
        continue

    def fetch_meta_from_zip(row: dict) -> tuple[str, dict | None]:
        filepath = row["filepath"]
        meta_rel = meta_path_from_img_path(filepath)

        # Read pickled meta from zip
        try:
            meta_bytes = reader.read(meta_rel)
            if meta_bytes is None:
                return filepath, None
            meta = pickle.loads(meta_bytes)
        except Exception:
            return filepath, None

        gps = gps_from_meta(meta)
        time = time_from_meta(meta)
        if gps is None and time is None:
            return filepath, None

        result: dict = {}
        if gps is not None:
            result["lat"] = gps[0]
            result["lon"] = gps[1]
            result["gps_xyz"] = lat_lon_to_xyz(gps[0], gps[1])
        if time is not None:
            result["hour"] = time["hour"]
            result["minute"] = time["minute"]
        return filepath, result

    n_saved = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(fetch_meta_from_zip, r): r["filepath"] for r in rows_in_zip
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Meta{zip_n}"
        ):
            fp, result = future.result()
            meta_status[fp] = result
            n_saved += 1
            if n_saved % SAVE_INTERVAL == 0:
                with META_STATUS_FILE.open("w") as f:
                    json.dump(meta_status, f)

    with META_STATUS_FILE.open("w") as f:
        json.dump(meta_status, f)

meta_hits = sum(1 for v in meta_status.values() if v is not None)
print(f"Meta hits: {meta_hits} / {len(meta_status)}")

# ------------------------------------------------------------------
# Step 3: Download images in parallel via RemoteZipReader
# ------------------------------------------------------------------
IMG_STATUS_FILE = meta_cache_dir / "img_status.json"
img_status: dict[str, bool] = {}

if IMG_STATUS_FILE.exists():
    with IMG_STATUS_FILE.open() as f:
        img_status = json.load(f)

rows_with_meta = [r for r in csv_rows if meta_status.get(r["filepath"]) is not None]
rows_need_img = [r for r in rows_with_meta if r["filepath"] not in img_status]
print(f"\nDownloading images for {len(rows_need_img)} rows ({N_WORKERS} workers)...")

# Group rows by zip number
img_zip_groups = {}
for row in rows_need_img:
    n = zip_num_from_filepath(row["filepath"])
    if n not in img_zip_groups:
        img_zip_groups[n] = []
    img_zip_groups[n].append(row)

print(f"Image zips: {sorted(img_zip_groups.keys())}")

# Load each image zip and read files in parallel
for zip_n in sorted(img_zip_groups.keys()):
    rows_in_zip = img_zip_groups[zip_n]
    img_zip_url = IMAGES_ZIP_URL.format(n=zip_n)
    print(f"\nLoading images{zip_n}.zip ({len(rows_in_zip)} rows)...")

    reader = RemoteZipReader(img_zip_url)
    try:
        reader.load_index()
    except Exception as e:
        print(f"  Failed to load index: {e}")
        continue

    def fetch_image_from_zip(row: dict) -> tuple[str, bool]:
        filepath = row["filepath"]
        stem = Path(filepath).stem
        save_path = img_save_root / f"{stem}.jpg"

        if save_path.exists():
            return filepath, True

        try:
            img_bytes = reader.read(filepath)
            if img_bytes is None:
                return filepath, False
            img = download_image_from_bytes(img_bytes)
            if img is None:
                return filepath, False
            img.save(save_path, "JPEG")
            return filepath, True
        except Exception:
            return filepath, False

    n_saved = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(fetch_image_from_zip, r): r["filepath"]
            for r in rows_in_zip
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Images{zip_n}"
        ):
            fp, ok = future.result()
            img_status[fp] = ok
            n_saved += 1
            if n_saved % SAVE_INTERVAL == 0:
                with IMG_STATUS_FILE.open("w") as f:
                    json.dump(img_status, f)

    with IMG_STATUS_FILE.open("w") as f:
        json.dump(img_status, f)

img_hits = sum(1 for v in img_status.values() if v)
print(f"Image hits: {img_hits} / {len(img_status)}")

# ------------------------------------------------------------------
# Step 4: Build filtered records (valid meta + downloaded image)
# ------------------------------------------------------------------
print("\nBuilding filtered records...")

# Use title_multi_objects: more unique (includes surrounding objects)
# "title" has ~4.8K unique captions per 20K rows; "title_multi_objects" has ~15.7K
caption_col = "title_multi_objects" if "title_multi_objects" in csv_rows[0] else "title"
print(f"Using caption column: '{caption_col}'")

seen_captions: set[str] = set()
records = []

for row in csv_rows:
    fp = row["filepath"]
    meta = meta_status.get(fp)
    if meta is None:
        continue
    if not img_status.get(fp, False):
        continue

    caption = row.get(caption_col, "").strip()
    if not caption or caption in seen_captions:
        continue
    seen_captions.add(caption)

    stem     = Path(fp).stem
    img_path = img_save_root / f"{stem}.jpg"
    if not img_path.exists():
        continue

    record = {
        "img_id":  stem,
        "caption": caption,
        "img_path": str(img_path),
    }
    if "lat" in meta:
        record["lat"]     = meta["lat"]
        record["lon"]     = meta["lon"]
        record["gps"]     = [meta["lat"], meta["lon"]]
        record["gps_xyz"] = meta["gps_xyz"]
    if "hour" in meta:
        record["hour"]   = meta["hour"]
        record["minute"] = meta["minute"]

    records.append(record)

print(f"Final records: {len(records)}")
has_geo  = any("gps_xyz" in r for r in records)
has_time = any("hour" in r for r in records)
print(f"  has_geo={has_geo}  has_time={has_time}")

if len(records) < 100:
    raise RuntimeError("Too few valid records. Check S3 connectivity and meta downloads.")

# ------------------------------------------------------------------
# Step 5: Build HuggingFace dataset
# ------------------------------------------------------------------
import datasets as hf_datasets

print("Building HuggingFace dataset...")


def load_image(data: dict) -> dict:
    data["image"] = Image.open(data["img_path"]).convert("RGB")
    return data


base_ds = hf_datasets.Dataset.from_list(records)
full_dataset = (
    base_ds
    .map(load_image, desc="Loading images", num_proc=4)
    .cast_column("image", hf_datasets.Image())
)

# ------------------------------------------------------------------
# Step 6: 20/80 val/test split
# ------------------------------------------------------------------
n = len(full_dataset)
rng  = torch.Generator().manual_seed(42)
perm = torch.randperm(n, generator=rng).tolist()
n_val = int(n * 0.2)

val_dataset  = full_dataset.select(perm[:n_val])
test_dataset = full_dataset.select(perm[n_val:])
print(f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ------------------------------------------------------------------
# Step 7: Save all pkl variants
# ------------------------------------------------------------------
save_skyscript_splits(val_dataset, test_dataset, save_task_data_root)
print("SkyScript preprocessing complete.")
