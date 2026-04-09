"""
Utility functions for SkyScript dataset preprocessing.

SkyScript: 5.2M remote sensing image-text pairs with GPS bounding boxes
and image acquisition timestamps from Google Earth Engine.

Meta file fields (per image .pickle):
  bbox:            (lon1, lat1, lon2, lat2) — image bounding box  time:            int (year only), 3-tuple (year, month, day), or 5-tuple (year, month, day, hour, minute)
  center_tags:     {key: value} — OSM tags of focus object
  surrounding_tags: [{key: value}] — OSM tags of surrounding objects

CSV fields (per pair):
  filepath:              relative path, e.g. "images2/a198234555_CH_19.jpg"
  title:                 caption for focus object
  title_multi_objects:   caption including surrounding objects
"""
import io
import pickle
import struct
import warnings
import zlib
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

import datasets

warnings.filterwarnings("ignore")

S3_BASE = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript"
CSV_URLS = {
    "test_30k": f"{S3_BASE}/dataframe/SkyScript_test_30K_filtered_by_CLIP_openai.csv",
    "val_5k":   f"{S3_BASE}/dataframe/SkyScript_val_5K_filtered_by_CLIP_openai.csv",
}

# Zip URL template:  meta2.zip … meta7.zip  /  images2.zip … images7.zip
META_ZIP_URL   = f"{S3_BASE}/meta{{n}}.zip"
IMAGES_ZIP_URL = f"{S3_BASE}/images{{n}}.zip"


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------

def meta_path_from_img_path(filepath: str) -> str:
    """'images2/foo.jpg'  →  'meta2/foo.pickle'"""
    return filepath.replace("images", "meta", 1).replace(".jpg", ".pickle")


def zip_num_from_filepath(filepath: str) -> int:
    """'images3/foo.jpg'  →  3"""
    folder = filepath.split("/")[0]           # e.g. 'images3'
    return int("".join(c for c in folder if c.isdigit()))


def s3_zip_url(filepath: str, kind: str = "images") -> str:
    """Return the S3 zip URL for a given filepath.
    kind = 'images' or 'meta'
    """
    n = zip_num_from_filepath(filepath)
    template = IMAGES_ZIP_URL if kind == "images" else META_ZIP_URL
    return template.format(n=n)


# ------------------------------------------------------------------
# RemoteZipReader — extracts individual files from a remote ZIP
# using HTTP Range requests (no full download needed).
# Supports Zip64 for archives >4 GB.
# ------------------------------------------------------------------

class RemoteZipReader:
    """
    Read specific files from a remote ZIP archive via HTTP Range requests.

    Usage:
        reader = RemoteZipReader(url)
        reader.load_index()          # 3 HTTP requests, call once
        data = reader.read(filename) # 2 HTTP requests per file (thread-safe)
    """

    _EOCD_SIG   = b"PK\x05\x06"
    _EOCD64_LOC = b"PK\x06\x07"
    _EOCD64_SIG = b"PK\x06\x06"
    _CD_SIG     = b"PK\x01\x02"
    _LF_SIG     = b"PK\x03\x04"

    def __init__(self, url: str) -> None:
        self.url = url
        self._file_index: dict[str, dict] | None = None
        self._size: int | None = None

    # ---- private helpers ----

    def _get_range(self, start: int, end: int) -> bytes:
        r = requests.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=30,
            verify=False,
        )
        if r.status_code not in (200, 206):
            raise RuntimeError(
                f"Range {start}-{end} → HTTP {r.status_code}  url={self.url}"
            )
        return r.content

    @property
    def size(self) -> int:
        if self._size is None:
            r = requests.head(self.url, timeout=10, verify=False)
            self._size = int(r.headers["Content-Length"])
        return self._size

    # ---- public API ----

    def load_index(self) -> None:
        """Fetch and parse the Central Directory (3 HTTP requests)."""
        sz = self.size

        # 1. Fetch last 65 KB to locate EOCD
        tail_start = max(0, sz - 65536)
        tail = self._get_range(tail_start, sz - 1)

        pos = tail.rfind(self._EOCD_SIG)
        if pos == -1:
            raise RuntimeError(f"EOCD not found in {self.url}")

        eocd = tail[pos:]
        cd_size   = struct.unpack_from("<I", eocd, 12)[0]
        cd_offset = struct.unpack_from("<I", eocd, 16)[0]

        # 2. If Zip64, resolve actual CD location from Zip64 EOCD
        if cd_size == 0xFFFF_FFFF or cd_offset == 0xFFFF_FFFF:
            loc_pos = tail.rfind(self._EOCD64_LOC)
            if loc_pos == -1:
                raise RuntimeError(f"Zip64 EOCD locator not found in {self.url}")
            eocd64_off = struct.unpack_from("<Q", tail, loc_pos + 8)[0]
            eocd64 = self._get_range(eocd64_off, eocd64_off + 55)
            if eocd64[:4] != self._EOCD64_SIG:
                raise RuntimeError(f"Zip64 EOCD signature mismatch in {self.url}")
            cd_size   = struct.unpack_from("<Q", eocd64, 40)[0]
            cd_offset = struct.unpack_from("<Q", eocd64, 48)[0]

        # 3. Fetch and parse Central Directory
        cd = self._get_range(cd_offset, cd_offset + cd_size - 1)
        index: dict[str, dict] = {}
        p = 0
        while p < len(cd):
            if cd[p : p + 4] != self._CD_SIG:
                break
            compress_type = struct.unpack_from("<H", cd, p + 10)[0]
            comp_size     = struct.unpack_from("<I", cd, p + 20)[0]
            uncomp_size   = struct.unpack_from("<I", cd, p + 24)[0]
            fname_len     = struct.unpack_from("<H", cd, p + 28)[0]
            extra_len     = struct.unpack_from("<H", cd, p + 30)[0]
            comment_len   = struct.unpack_from("<H", cd, p + 32)[0]
            lhdr_off      = struct.unpack_from("<I", cd, p + 42)[0]

            fname = cd[p + 46 : p + 46 + fname_len].decode("utf-8", errors="replace")

            # Resolve Zip64 extended fields if needed
            if comp_size == 0xFFFF_FFFF or uncomp_size == 0xFFFF_FFFF or lhdr_off == 0xFFFF_FFFF:
                extra = cd[p + 46 + fname_len : p + 46 + fname_len + extra_len]
                ep = 0
                while ep + 4 <= len(extra):
                    hid   = struct.unpack_from("<H", extra, ep)[0]
                    dsize = struct.unpack_from("<H", extra, ep + 2)[0]
                    if hid == 0x0001:
                        vp = ep + 4
                        if uncomp_size == 0xFFFF_FFFF and vp + 8 <= ep + 4 + dsize:
                            uncomp_size = struct.unpack_from("<Q", extra, vp)[0]; vp += 8
                        if comp_size == 0xFFFF_FFFF and vp + 8 <= ep + 4 + dsize:
                            comp_size = struct.unpack_from("<Q", extra, vp)[0]; vp += 8
                        if lhdr_off == 0xFFFF_FFFF and vp + 8 <= ep + 4 + dsize:
                            lhdr_off = struct.unpack_from("<Q", extra, vp)[0]
                        break
                    ep += 4 + dsize

            index[fname] = {
                "lhdr_off":     lhdr_off,
                "comp_size":    comp_size,
                "compress_type": compress_type,
            }
            p += 46 + fname_len + extra_len + comment_len

        self._file_index = index

    def read(self, filename: str) -> bytes | None:
        """
        Extract one file by name (2 HTTP requests, thread-safe).
        Returns raw bytes or None if not found.
        """
        if self._file_index is None:
            raise RuntimeError("Call load_index() before read()")
        info = self._file_index.get(filename)
        if info is None:
            return None

        # Read local file header to find actual data start
        lhdr = self._get_range(info["lhdr_off"], info["lhdr_off"] + 29)
        if lhdr[:4] != self._LF_SIG:
            return None
        fname_len = struct.unpack_from("<H", lhdr, 26)[0]
        extra_len = struct.unpack_from("<H", lhdr, 28)[0]
        data_off  = info["lhdr_off"] + 30 + fname_len + extra_len

        comp = self._get_range(data_off, data_off + info["comp_size"] - 1)
        if info["compress_type"] == 0:   # stored (no compression)
            return comp
        if info["compress_type"] == 8:   # deflated
            return zlib.decompress(comp, -15)
        raise RuntimeError(f"Unsupported compress type {info['compress_type']}")

    def __contains__(self, filename: str) -> bool:
        if self._file_index is None:
            raise RuntimeError("Call load_index() first")
        return filename in self._file_index


# ------------------------------------------------------------------
# Download helpers (non-zip, used for CSV)
# ------------------------------------------------------------------

def download_bytes(url: str, timeout: int = 30) -> bytes | None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://github.com/wangzhecheng/SkyScript",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout, verify=False)
        if r.status_code == 200:
            return r.content
        else:
            print(f"Download failed: {url} -> status {r.status_code}")
    except Exception as e:
        print(f"Exception for {url}: {e}")
    return None

def download_image_from_bytes(data: bytes, max_size: int = 1024) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if w < 32 or h < 32:
            return None
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    except Exception:
        return None


# ------------------------------------------------------------------
# GPS / time extraction from meta pickle
# ------------------------------------------------------------------

def gps_from_meta(meta: dict) -> tuple[float, float] | None:
    # Try "bbox" (new format) or "box" (old format)
    bbox = meta.get("bbox") or meta.get("box")
    if not bbox or len(bbox) != 4:
        return None
    lon1, lat1, lon2, lat2 = bbox
    lat = (lat1 + lat2) / 2.0
    lon = (lon1 + lon2) / 2.0
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if lat == 0.0 and lon == 0.0:
        return None
    return float(lat), float(lon)


def time_from_meta(meta: dict) -> dict | None:
    t = meta.get("time")
    if t is None:
        return None

    # Handle different time formats:
    # - Integer (year only): 2018
    # - 3-tuple (year, month, day)
    # - 5-tuple (year, month, day, hour, minute)

    try:
        if isinstance(t, int):
            # Year only: use 12:00 noon as default hour
            return {"hour": 12, "minute": 0}
        elif isinstance(t, (list, tuple)):
            if len(t) == 3:
                # Date only: use 12:00 noon
                return {"hour": 12, "minute": 0}
            elif len(t) >= 5:
                # Full timestamp
                hour, minute = int(t[3]), int(t[4])
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    return None
                return {"hour": hour, "minute": minute}
        return None
    except Exception:
        return None


def lat_lon_to_xyz(lat: float, lon: float) -> list[float]:
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    return [
        float(np.cos(lat_r) * np.cos(lon_r)),
        float(np.cos(lat_r) * np.sin(lon_r)),
        float(np.sin(lat_r)),
    ]


# ------------------------------------------------------------------
# Label builder — single ground truth per caption
# ------------------------------------------------------------------

def build_single_gt_labels(
    captions: list[str], n: int
) -> tuple[list[str], dict[str, torch.Tensor]]:
    labels: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for i, cap in enumerate(captions):
        cap = cap.strip()
        if not cap or cap in seen:
            continue
        seen.add(cap)
        t = torch.zeros(n, dtype=torch.int)
        t[i] = 1
        labels[cap] = t
    return list(labels.keys()), labels


# ------------------------------------------------------------------
# Save all three task variants from one dataset
# ------------------------------------------------------------------

def save_skyscript_splits(
    val_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    save_root: Path,
) -> None:
    # Import here to avoid dependency issues during preprocessing
    from msdpp.schema import RetrievalDataset
    from msdpp.data import datetime_embeds

    for split_name, ds in [("val", val_dataset), ("test", test_dataset)]:
        captions = [ds[i]["caption"] for i in range(len(ds))]
        queries, labels = build_single_gt_labels(captions, len(ds))

        has_geo  = "gps_xyz" in ds.column_names
        has_time = "hour"    in ds.column_names

        geo_ext = time_ext = None

        if has_geo:
            xyz = torch.tensor(
                [ds[i]["gps_xyz"] or [0.0, 0.0, 1.0] for i in range(len(ds))], dtype=torch.float32
            )
            geo_ext = torch.nn.functional.normalize(xyz, dim=-1)
            _save_pkl(save_root, f"SkyScript_geo_{split_name}", ds, queries, labels, geo_ext)

        if has_time:
            hours = torch.tensor([ds[i]["hour"]   or 12 for i in range(len(ds))], dtype=torch.float32)
            mins  = torch.tensor([ds[i]["minute"] or  0 for i in range(len(ds))], dtype=torch.float32)
            time_ext = datetime_embeds(hours, mins)
            _save_pkl(save_root, f"SkyScript_hour_{split_name}", ds, queries, labels, time_ext)

        if has_geo and has_time:
            _save_pkl(save_root, f"SkyScript_geo_hour_{split_name}", ds, queries, labels,
                      [geo_ext, time_ext])


def _save_pkl(
    save_root: Path,
    name: str,
    dataset: datasets.Dataset,
    queries: list[str],
    labels: dict,
    ext_data,
) -> None:
    from msdpp.schema import RetrievalDataset

    ret = RetrievalDataset(
        name=name,
        dataset=dataset,
        retrieval_words=queries,
        labels=labels,
        ext_data=ext_data,
    )
    path = save_root / f"{name}.pkl"
    with path.open("wb") as f:
        torch.save(ret, f)
    print(f"Saved: {path}")
