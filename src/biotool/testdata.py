"""Download the test data used by Evaluate.bat / HighpassTest.bat into test_data/.

  * BirdNET example soundscape (2 min, from the BirdNET-Analyzer repository)
  * Powdermill dawn-chorus sample (Chronister et al. 2021, Zenodo 4656848): 4 five-minute WAVs +
    all annotations + README. Only the 4 WAVs are read out of the 1.4 GB zip (HTTP range requests).
  * ESC-50 sample (Piczak 2015): 2 clips for each of the 50 classes + labels.csv

Run:  uv run biotool get-test-data   (or GetTestData.bat)
"""

from __future__ import annotations

import csv
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZENODO = "https://zenodo.org/records/4656848/files/{}?download=1"
POWDERMILL_SEGMENTS = [  # the four richest annotated segments (7-17 species each)
    "Recording_2/Recording_2_Segment_07.wav",
    "Recording_2/Recording_2_Segment_10.wav",
    "Recording_4/Recording_4_Segment_01.wav",
    "Recording_4/Recording_4_Segment_16.wav",
]
ESC_RAW = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/{}"
BIRDNET_EXAMPLE = ("https://raw.githubusercontent.com/birdnet-team/BirdNET-Analyzer/main/"
                   "birdnet_analyzer/example/soundscape.wav")
ESC_GROUPS = {0: ("Animals", "biophony"), 1: ("Natural soundscapes & water", "geophony"),
              2: ("Human, non-speech", "anthropophony"), 3: ("Interior/domestic", "anthropophony"),
              4: ("Exterior/urban noises", "anthropophony")}
UA = {"User-Agent": "biotool-testdata/1.0"}


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _download(url: str, dest: Path, progress=print) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -10
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and (pct := int(100 * done / total)) >= last + 10:
                progress(f"    {dest.name}: {pct}%")
                last = pct
    tmp.replace(dest)


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise OSError("server does not support range requests")
            self.url = r.geturl()  # follow redirects once
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _extract(zf: zipfile.ZipFile, wanted: dict[str, Path], progress) -> None:
    names = {n.lower(): n for n in zf.namelist()}
    for member, dest in wanted.items():
        real = names.get(member.lower())
        if real is None:
            progress(f"    not found in archive: {member}")
            continue
        progress(f"    extracting {Path(real).name} ({zf.getinfo(real).file_size / 1e6:.0f} MB)")
        with zf.open(real) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)


def get_powdermill(root: Path, progress=print) -> None:
    pm = root / "Powdermill"
    audio, ann = pm / "sample_20min", pm / "annotations"
    audio.mkdir(parents=True, exist_ok=True)
    progress("Powdermill (Zenodo 4656848, cite Chronister et al. 2021, Ecology):")
    if not (pm / "README.txt").exists():
        (pm / "README.txt").write_bytes(_get(ZENODO.format("README.txt")))
    if not ann.exists():
        progress("    annotations")
        with zipfile.ZipFile(io.BytesIO(_get(ZENODO.format("annotation_Files.zip")))) as zf:
            zf.extractall(ann)
    wanted = {m: audio / Path(m).name for m in POWDERMILL_SEGMENTS if not (audio / Path(m).name).exists()}
    if not wanted:
        progress("    audio already present")
        return
    url = ZENODO.format("wav_Files.zip")
    try:
        with zipfile.ZipFile(HttpRangeFile(url)) as zf:
            _extract(zf, wanted, progress)
    except Exception as e:  # no range support: fall back to the full archive
        progress(f"    partial download not possible ({e}); downloading the full 1.4 GB archive")
        full = pm / "wav_Files.zip"
        if not full.exists():
            _download(url, full, progress)
        with zipfile.ZipFile(full) as zf:
            _extract(zf, wanted, progress)


def get_esc50(root: Path, per_class: int = 2, progress=print) -> None:
    esc = root / "ESC-50_sample"
    (esc / "audio").mkdir(parents=True, exist_ok=True)
    progress(f"ESC-50 sample ({per_class} clips x 50 classes; CC BY-NC, cite Piczak 2015):")
    meta = list(csv.DictReader(io.StringIO(_get(ESC_RAW.format("meta/esc50.csv")).decode())))
    picked, rows = {}, []
    for r in meta:
        t = int(r["target"])
        if picked.get(t, 0) >= per_class:
            continue
        picked[t] = picked.get(t, 0) + 1
        dest = esc / "audio" / r["filename"]
        if not dest.exists():
            dest.write_bytes(_get(ESC_RAW.format(f"audio/{r['filename']}")))
        major, group = ESC_GROUPS[t // 10]
        rows.append({**r, "major_category": major, "expected_group": group})
    with open(esc / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (esc / "README.txt").write_text(
        f"Sample of ESC-50 (Piczak 2015): {per_class} clips x 50 classes, 5 s each, 44.1 kHz.\n"
        "Licence: CC BY-NC (ESC-10 subset CC BY). Cite: K. J. Piczak. ESC: Dataset for Environmental "
        "Sound Classification. ACM Multimedia 2015.\n"
        "labels.csv: category, ESC-50 major category and the BioTool sound group we expect.\n", encoding="utf-8")
    progress(f"    {len(rows)} clips")


def get_birdnet_example(root: Path, progress=print) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "soundscape.wav"
    if not dest.exists():
        progress("BirdNET example soundscape")
        _download(BIRDNET_EXAMPLE, dest, progress)


def main(root: str | Path = "test_data", progress=print) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for step in (get_birdnet_example, get_esc50, get_powdermill):
        try:
            step(root, progress=progress)
        except Exception as e:
            progress(f"  FAILED: {e}")
    progress(f"Test data in {root.resolve()}")
