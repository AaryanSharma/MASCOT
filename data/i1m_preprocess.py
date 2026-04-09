"""
Incidents 1M (I1M) dataset preprocessing for shooting location tasks.

Source: multi_label_val.json from Weber et al. 2022 PAMI paper.
Filter: images with at least one positive incident label → ~48,666 images.

Location (GPS): approximated from the image URL's domain, following Section 6.2
of Weber et al. 2022:
  1. Extract domain from URL
  2. DNS lookup → IP address
  3. Query geolocation API with IP → (lat, lon)
  4. Cache by domain (shared server → shared GPS)
  5. Add small uniform noise to exact duplicates

Text queries: 43 incident types + 49 place types (those with ≥2 positives in split).
Labels: binary per incident/place type from the multi_label annotations.

Saves: I1M_geo_val.pkl, I1M_geo_test.pkl
"""
import json
import os
import socket
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from msdpp.schema import RetrievalDataset

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
HF_HOME = Path(os.environ.get("HF_HOME", "./"))
I1M_JSON_DIR = HF_HOME / "i1m"
save_task_data_root = HF_HOME / "tasks"
img_save_root = HF_HOME / "images" / "i1m"
cache_dir = HF_HOME / "i1m_metadata"

save_task_data_root.mkdir(parents=True, exist_ok=True)
img_save_root.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Step 1: Load multi_label_val.json, filter to positive incident images
# ------------------------------------------------------------------
print("Loading multi_label_val.json ...")
with (I1M_JSON_DIR / "multi_label_val.json").open() as f:
    ml_val = json.load(f)

print(f"Total entries: {len(ml_val)}")

# Keep entries with at least one positive incident label
positive_entries = {
    k: v for k, v in ml_val.items()
    if any(lbl == 1 for lbl in v.get("incidents", {}).values())
}
print(f"Entries with positive incident label: {len(positive_entries)}")

# ------------------------------------------------------------------
# Step 2: Geolocate each image URL via domain → DNS → IP → lat/lon
# ------------------------------------------------------------------
GEO_CACHE_PATH = cache_dir / "domain_gps_cache.json"

geo_cache: dict[str, list | None] = {}
if GEO_CACHE_PATH.exists():
    with GEO_CACHE_PATH.open() as f:
        geo_cache = json.load(f)
    cached_valid = sum(1 for v in geo_cache.values() if v is not None)
    print(f"Loaded domain GPS cache: {len(geo_cache)} domains, {cached_valid} with valid GPS")

GEO_APIS = [
    "https://geolocation-db.com/json/{ip}",
    "http://ip-api.com/json/{ip}?fields=lat,lon,status",
]

def query_gps_from_ip(ip: str) -> tuple[float, float] | None:
    for api_template in GEO_APIS:
        try:
            resp = requests.get(api_template.format(ip=ip), timeout=5)
            data = resp.json()
            lat = data.get("latitude") or data.get("lat")
            lon = data.get("longitude") or data.get("lon")
            if lat and lon and str(lat) not in ("Not found", "None", "") and str(lon) not in ("Not found", "None", ""):
                lat, lon = float(lat), float(lon)
                if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
                    return lat, lon
        except Exception:
            continue
    return None


def lat_lon_to_xyz(lat: float, lon: float) -> list[float]:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return [
        float(np.cos(lat_r) * np.cos(lon_r)),
        float(np.cos(lat_r) * np.sin(lon_r)),
        float(np.sin(lat_r)),
    ]


SAVE_INTERVAL = 500
GEO_WORKERS = 64

print("\nGeolocating image URLs by domain ...")

# Collect unique uncached domains
domain_to_keys: dict[str, list[str]] = {}
for key, entry in positive_entries.items():
    url = entry.get("url", "")
    if not url:
        continue
    try:
        domain = urllib.parse.urlparse(url).netloc
    except Exception:
        continue
    if domain:
        domain_to_keys.setdefault(domain, []).append(key)

uncached_domains = [d for d in domain_to_keys if d not in geo_cache]
print(f"Total unique domains: {len(domain_to_keys)}, uncached: {len(uncached_domains)}")

# Resolve uncached domains in parallel
geo_cache_lock = Lock()
n_resolved = 0

def resolve_domain(domain: str) -> None:
    result = None
    try:
        ip = socket.gethostbyname(domain)
        result = query_gps_from_ip(ip)
    except Exception:
        pass
    with geo_cache_lock:
        global n_resolved
        geo_cache[domain] = list(result) if result is not None else None
        n_resolved += 1
        if n_resolved % SAVE_INTERVAL == 0:
            with GEO_CACHE_PATH.open("w") as f:
                json.dump(geo_cache, f)

if uncached_domains:
    with ThreadPoolExecutor(max_workers=GEO_WORKERS) as executor:
        futures = {executor.submit(resolve_domain, d): d for d in uncached_domains}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="DNS+Geo lookup"):
            pass

with GEO_CACHE_PATH.open("w") as f:
    json.dump(geo_cache, f)

# Build geolocated list
geolocated = []
for key, entry in positive_entries.items():
    url = entry.get("url", "")
    if not url:
        continue
    try:
        domain = urllib.parse.urlparse(url).netloc
    except Exception:
        continue
    if not domain:
        continue
    v = geo_cache.get(domain)
    if v is None:
        continue
    geolocated.append({
        "key": key,
        "url": url,
        "lat": float(v[0]),
        "lon": float(v[1]),
        "incidents": entry.get("incidents", {}),
        "places": entry.get("places", {}),
    })

print(f"Successfully geolocated: {len(geolocated)}")

# ------------------------------------------------------------------
# Step 3: Add small uniform noise to exact-duplicate GPS locations
# ------------------------------------------------------------------
NOISE_SCALE = 1e-3
seen_coords: dict[tuple, int] = {}

for item in geolocated:
    coord_key = (round(item["lat"], 4), round(item["lon"], 4))
    count = seen_coords.get(coord_key, 0)
    if count > 0:
        item["lat"] += (np.random.rand() - 0.5) * 2 * NOISE_SCALE * count
        item["lon"] += (np.random.rand() - 0.5) * 2 * NOISE_SCALE * count
    seen_coords[coord_key] = count + 1

# ------------------------------------------------------------------
# Step 4: Download images (parallel)
# ------------------------------------------------------------------
def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    if save_path.exists():
        return True
    try:
        resp = requests.get(url, stream=True, timeout=timeout, verify=False)
        if resp.status_code != 200:
            return False
        img = Image.open(resp.raw).convert("RGB")
        w, h = img.size
        if w < 32 or h < 32:          # ← add this filter
            return False
        max_size = 1024
        if w > max_size or h > max_size:
            if w > h:
                img = img.resize((max_size, int(h / w * max_size)))
            else:
                img = img.resize((int(w / h * max_size), max_size))
        img.save(save_path, "JPEG")
        return True
    except Exception:
        return False


print("\nDownloading images ...")
IMG_DL_CACHE = cache_dir / "img_dl_status.json"
img_status: dict[str, bool] = {}
if IMG_DL_CACHE.exists():
    with IMG_DL_CACHE.open() as f:
        img_status = json.load(f)

img_status_lock = Lock()
DL_WORKERS = 64

def process_item(item: dict) -> dict | None:
    key_safe = item["key"].replace("/", "_").replace(" ", "_")
    if key_safe.endswith('.jpg'):
        key_safe = key_safe[:-4]

    img_path = img_save_root / f"{key_safe}.jpg"

    with img_status_lock:
        cached = img_status.get(key_safe)
    if cached is not None:
        ok = cached
    else:
        ok = download_image(item["url"], img_path)
        with img_status_lock:
            img_status[key_safe] = ok
            if len(img_status) % SAVE_INTERVAL == 0:
                with IMG_DL_CACHE.open("w") as f:
                    json.dump(img_status, f)

    if ok:
        return {
            **item,
            "image_path": str(img_path),
            "gps": [item["lat"], item["lon"]],
            "gps_xyz": lat_lon_to_xyz(item["lat"], item["lon"]),
        }
    return None

downloaded = []
with ThreadPoolExecutor(max_workers=DL_WORKERS) as executor:
    futures = {executor.submit(process_item, item): item for item in geolocated}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        result = future.result()
        if result is not None:
            downloaded.append(result)

with IMG_DL_CACHE.open("w") as f:
    json.dump(img_status, f)

print(f"Successfully downloaded: {len(downloaded)} images")

if len(downloaded) == 0:
    raise RuntimeError("No I1M images downloaded. Check URL accessibility.")

# ------------------------------------------------------------------
# Step 5: Build HuggingFace dataset
# ------------------------------------------------------------------
import datasets as hf_datasets

def load_pil(data: dict) -> dict:
    data["image"] = Image.open(data["image_path"])
    return data

records = []
for item in downloaded:
    records.append({
        "image_path": item["image_path"],
        "gps": item["gps"],
        "gps_xyz": item["gps_xyz"],
        "incidents": item["incidents"],   # dict {type: 0/1}
        "places": item["places"],         # dict {type: 0/1}
    })

print("Building HuggingFace dataset ...")
# import os

# --- NEW SAFETY FILTER ---
# Only keep items where the image actually exists on the hard drive
# clean_processed = []
# for item in records:
#     # Handle whichever key your script uses for the file path
#     path_to_check = item.get("image_path", item.get("local_path", ""))
#     if os.path.exists(path_to_check):
#         clean_processed.append(item)

# print(f"Filtered out {len(records) - len(clean_processed)} missing/bad images.")
# -------------------------

# base_ds = hf_datasets.Dataset.from_list(clean_processed)
base_ds = hf_datasets.Dataset.from_list(records)
full_dataset = (
    base_ds
    .map(load_pil, desc="Loading images")
    .cast_column("image", hf_datasets.Image())
)

# ------------------------------------------------------------------
# Step 6: Split into val/test (20/80) and build labels
# ------------------------------------------------------------------
n = len(full_dataset)
rng = torch.Generator().manual_seed(42)
perm = torch.randperm(n, generator=rng).tolist()
n_val = int(n * 0.2)

val_dataset = full_dataset.select(perm[:n_val])
test_dataset = full_dataset.select(perm[n_val:])

print(f"Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")


def build_labels(
    dataset: hf_datasets.Dataset,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """
    Build multi-positive labels from explicit incidents + places annotations.
    Queries with <2 positives are dropped.
    """
    n = len(dataset)
    type_positives: dict[str, list[int]] = {}

    for i in range(n):
        row = dataset[i]
        for inc_type, lbl in (row.get("incidents") or {}).items():
            if lbl == 1:
                type_positives.setdefault(inc_type, []).append(i)
        for place_type, lbl in (row.get("places") or {}).items():
            if lbl == 1:
                type_positives.setdefault(f"place:{place_type}", []).append(i)

    labels: dict[str, torch.Tensor] = {}
    for q, pos_idxs in type_positives.items():
        if len(pos_idxs) >= 2:
            t = torch.zeros(n, dtype=torch.int)
            t[pos_idxs] = 1
            labels[q] = t

    queries = sorted(labels.keys())
    return queries, labels


print("Building val labels ...")
val_queries, val_labels = build_labels(val_dataset)
print("Building test labels ...")
test_queries, test_labels = build_labels(test_dataset)

common_queries = [q for q in val_queries if q in test_labels]
val_labels = {q: val_labels[q] for q in common_queries}
test_labels = {q: test_labels[q] for q in common_queries}

print(f"Queries ({len(common_queries)}): {common_queries[:10]} ...")

# Build geo xyz ext_data tensors
val_xyz = torch.tensor(
    [val_dataset[i]["gps_xyz"] for i in range(len(val_dataset))], dtype=torch.float32
)
test_xyz = torch.tensor(
    [test_dataset[i]["gps_xyz"] for i in range(len(test_dataset))], dtype=torch.float32
)
val_ext = torch.nn.functional.normalize(val_xyz, dim=-1)
test_ext = torch.nn.functional.normalize(test_xyz, dim=-1)

# ------------------------------------------------------------------
# Step 7: Save pkl files
# ------------------------------------------------------------------
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

val_path = save_task_data_root / "I1M_geo_val.pkl"
test_path = save_task_data_root / "I1M_geo_test.pkl"

with val_path.open("wb") as f:
    torch.save(val_ret, f)
with test_path.open("wb") as f:
    torch.save(test_ret, f)

print(f"\nSaved: {val_path}")
print(f"Saved: {test_path}")
print("I1M preprocessing complete.")
