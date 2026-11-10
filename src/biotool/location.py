"""Location / season species filter using BirdNET's meta-model.

Given latitude, longitude and week of the year, BirdNET's metadata model predicts how likely
each of its ~6,500 species is to occur there. Species below a threshold (default 0.03, as in
BirdNET-Analyzer) are treated as out of range, and their BirdNET / Perch detections are removed.

Only species BirdNET knows are judged. Non-species classes (engine, human...) and
species unknown to BirdNET (e.g. some Perch-only taxa such as many insects/frogs) are kept.
"""

from __future__ import annotations

import math
import re
from calendar import isleap
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .downloads import birdnet_files
from .models.base import split_birdnet_label
from .models.birdnet import NON_SPECIES_GROUPS, _interpreter

FILTERED_MODELS = ("birdnet", "perch_v2", "dfs", "birdcode")

# 20230415_063000 (AudioMoth), SITE_20230415_063000 (Song Meter), 2023-04-15T06-30-00 ...
_DATE_RE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?!\d)")
# AudioMoth WAV comment: "Recorded at 06:30:00 15/04/2023 (UTC) ..."
_AUDIOMOTH_RE = re.compile(r"Recorded at \d{2}:\d{2}:\d{2} (\d{2})/(\d{2})/(\d{4})")


@dataclass
class LocationSettings:
    enabled: bool = False
    lat: float = 0.0
    lon: float = 0.0
    date_mode: str = "file"     # "file" (from file name / AudioMoth metadata), "fixed", "year"
    fixed_date: str = ""        # YYYY-MM-DD when date_mode == "fixed"
    min_occurrence: float = 0.03

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LocationSettings":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def week48(d: date) -> int:
    """BirdNET's 48-week calendar (4 'weeks' per month)."""
    doy = d.timetuple().tm_yday
    return max(1, min(48, math.ceil(doy / (366 if isleap(d.year) else 365) * 48)))


def date_from_file(path: str | Path) -> date | None:
    """Recording date from the file name, else from an AudioMoth WAV comment."""
    m = _DATE_RE.search(Path(path).stem)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    try:
        import soundfile as sf

        extra = sf.info(str(path)).extra_info or ""
        m = _AUDIOMOTH_RE.search(extra)
        if m:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except Exception:
        pass
    return None


class LocationFilter:
    def __init__(self, settings: LocationSettings, threads: int = 0):
        self.s = settings
        self.threads = threads
        self._cache: dict[int, set[str]] = {}
        self._interp = None

    def load(self, progress=None) -> None:
        files = birdnet_files(progress)
        labels = [l.strip() for l in open(files["labels"], encoding="utf-8") if l.strip()]
        self._sci = [split_birdnet_label(l)[0] for l in labels]
        self.known = {s.lower() for s in self._sci if s not in NON_SPECIES_GROUPS}
        self._interp = _interpreter(str(files["meta_model"]), self.threads, embeddings=False)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]["index"]
        self._out = self._interp.get_output_details()[0]["index"]

    def week_for(self, path: str | Path) -> tuple[int, str]:
        """(week48 or -1 for all year, human-readable source)."""
        if self.s.date_mode == "fixed" and self.s.fixed_date:
            d = datetime.strptime(self.s.fixed_date, "%Y-%m-%d").date()
            return week48(d), f"fixed date {d}"
        if self.s.date_mode == "file":
            d = date_from_file(path)
            if d:
                return week48(d), f"date {d} from file"
            return -1, "no date found in file name/metadata - using all-year species list"
        return -1, "all year"

    def allowed(self, week: int) -> set[str]:
        if week not in self._cache:
            x = np.array([[self.s.lat, self.s.lon, float(week)]], dtype=np.float32)
            self._interp.set_tensor(self._in, x)
            self._interp.invoke()
            p = self._interp.get_tensor(self._out)[0]
            self._cache[week] = {self._sci[j].lower() for j in np.nonzero(p >= self.s.min_occurrence)[0]}
        return self._cache[week]

    def apply(self, det: pd.DataFrame, path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Split detections into (kept, removed) and return a note on the date used."""
        week, how = self.week_for(path)
        if det.empty:
            return det, det.iloc[0:0], how
        ok = self.allowed(week)
        judged = det["model"].isin(FILTERED_MODELS) & det["label_key"].isin(self.known)
        out = judged & ~det["label_key"].isin(ok)
        return det[~out].reset_index(drop=True), det[out].reset_index(drop=True), how

    def species_list(self, week: int) -> list[str]:
        return sorted(self.allowed(week))
