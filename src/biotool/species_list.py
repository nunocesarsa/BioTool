"""Custom species list: keep only the listed species, or remove them (BirdNET / Perch species).

The list is a text file with one species per line. Accepted forms (case-insensitive):
    Hylocichla mustelina                 scientific name
    Wood Thrush                          English name (as BirdNET uses it)
    Hylocichla mustelina_Wood Thrush     BirdNET species-list format
Empty lines and lines starting with # are ignored. Sound-event classes (YAMNet, BirdNET's
'Engine' etc.) are never affected.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SPECIES_MODELS = ("birdnet", "perch_v2")


@dataclass
class SpeciesListSettings:
    enabled: bool = False
    mode: str = "include"  # "include" = keep only these, "exclude" = remove these
    path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SpeciesListSettings":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def read_names(path: str | Path) -> set[str]:
    names = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for part in line.split("_", 1):  # "Sci_Common" -> both
            part = " ".join(part.split()).lower()
            if part:
                names.add(part)
    return names


def file_hash(path: str | Path) -> str:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
    except OSError:
        return "missing"


class SpeciesList:
    def __init__(self, settings: SpeciesListSettings):
        self.s = settings
        self.names = read_names(settings.path)
        if not self.names:
            raise ValueError(f"Species list {settings.path} contains no names")

    def apply(self, det: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if det.empty:
            return det, det.iloc[0:0]
        is_species = det["model"].isin(SPECIES_MODELS) & (det["scientific_name"].astype(str) != "") & \
            ~det["label_key"].astype(str).str.contains(":")
        listed = (det["label_key"].astype(str).str.lower().isin(self.names)
                  | det["scientific_name"].astype(str).str.lower().isin(self.names)
                  | det["common_name"].astype(str).str.lower().isin(self.names))
        drop = is_species & (~listed if self.s.mode == "include" else listed)
        return det[~drop].reset_index(drop=True), det[drop].reset_index(drop=True)
