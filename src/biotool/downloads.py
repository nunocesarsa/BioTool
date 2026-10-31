"""Fetching model files on first use. Everything is cached in the per-user models folder."""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .paths import models_dir

# BirdNET v2.4 is distributed inside the `birdnetlib` wheel on PyPI. We pull only the model
# files out of the wheel instead of installing birdnetlib and its dependencies.
BIRDNETLIB_VERSION = "0.18.1"
BIRDNET_FILES = {
    "model": "birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite",
    "labels": "birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Labels.txt",
    "licence": "birdnetlib/models/analyzer/LICENSE",
    "meta_model": "birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite",
}

KAGGLE_HANDLES = {
    "perch_v2": "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu",
    "yamnet": "google/yamnet/tensorFlow2/yamnet",
}


def _report(progress, msg: str) -> None:
    if progress:
        progress(msg)


def _download(url: str, dest: Path, progress=None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -10
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = int(100 * done / total)
                if pct >= last + 10:
                    _report(progress, f"  downloading {dest.name}: {pct}%")
                    last = pct
    tmp.replace(dest)


def birdnet_files(progress=None) -> dict[str, Path]:
    """Return local paths of the BirdNET v2.4 model/labels, downloading once if needed."""
    target = models_dir() / "birdnet_v2.4"
    out = {k: target / Path(v).name for k, v in BIRDNET_FILES.items()}
    if all(p.exists() for p in out.values()):
        return out

    # 1) Local override: a folder containing the two files (e.g. copied from BirdNET-Analyzer)
    local = os.environ.get("BIOTOOL_BIRDNET_DIR")
    if local and Path(local).exists():
        return {k: Path(local) / Path(v).name for k, v in BIRDNET_FILES.items()}

    target.mkdir(parents=True, exist_ok=True)
    _report(progress, "BirdNET v2.4 model not found locally - downloading (~60 MB, once)")
    meta_url = f"https://pypi.org/pypi/birdnetlib/{BIRDNETLIB_VERSION}/json"
    with urllib.request.urlopen(meta_url, timeout=30) as r:
        meta = json.load(r)
    wheel = next(u for u in meta["urls"] if u["packagetype"] == "bdist_wheel")
    whl_path = target / wheel["filename"]
    _download(wheel["url"], whl_path, progress)
    with zipfile.ZipFile(whl_path) as z:
        for key, member in BIRDNET_FILES.items():
            with z.open(member) as src, open(out[key], "wb") as dst:
                shutil.copyfileobj(src, dst)
    whl_path.unlink(missing_ok=True)
    _report(progress, "BirdNET model ready")
    return out


def kaggle_model(name: str, progress=None) -> Path:
    """Download (once) a public Kaggle model and return its folder.

    Set BIOTOOL_<NAME>_DIR (e.g. BIOTOOL_PERCH_V2_DIR) to use an already-downloaded copy.
    """
    env = os.environ.get(f"BIOTOOL_{name.upper()}_DIR")
    if env:
        return Path(env)
    target = models_dir() / name
    marker = target / ".path"
    if marker.exists():
        p = Path(marker.read_text().strip())
        if p.exists():
            return p
    _report(progress, f"{name}: downloading from Kaggle (once)")
    os.environ.setdefault("KAGGLEHUB_CACHE", str(models_dir() / "kagglehub"))
    import sys

    if sys.stdout is None or sys.stderr is None:  # no console (pythonw): tqdm would crash
        sys.stdout = sys.stdout or open(os.devnull, "w")
        sys.stderr = sys.stderr or open(os.devnull, "w")
    import kagglehub

    path = Path(kagglehub.model_download(KAGGLE_HANDLES[name]))
    target.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(path))
    _report(progress, f"{name}: ready")
    return path
