"""Common interface every model adapter implements.

Each adapter turns audio into rows of the shared detection schema:

    file | start_s | end_s | model | label | label_key | scientific_name | common_name | group | score

`label_key` is the harmonised key used to compare models (lower-case scientific
name for species, the class name for sound-event classes).  `group` is one of
biophony / geophony / anthropophony / other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DETECTION_COLUMNS = [
    "file", "start_s", "end_s", "model", "label", "label_key",
    "scientific_name", "common_name", "group", "score",
]

GROUPS = ("biophony", "geophony", "anthropophony", "other")


@dataclass
class ModelOutput:
    """Result of running one model over one block of audio."""

    detections: list[dict] = field(default_factory=list)
    emb_times: list[float] = field(default_factory=list)  # window start times (s)
    embeddings: list[np.ndarray] = field(default_factory=list)  # one vector per window

    def extend(self, other: "ModelOutput") -> None:
        self.detections.extend(other.detections)
        self.emb_times.extend(other.emb_times)
        self.embeddings.extend(other.embeddings)


class ModelAdapter:
    """Base class. Subclasses set the class attributes and implement load() + process()."""

    key: str = ""             # short id used in configs and outputs
    name: str = ""            # display name
    samplerate: int = 48000   # sample rate the model expects
    window_s: float = 3.0     # analysis window length
    default_threshold: float = 0.3
    fmax_hz: int = 24000      # highest frequency the model actually uses
    max_per_window: int = 10  # safety cap: keep at most this many labels per window
    produces_embeddings: bool = True
    # Extra settings shown under 'Advanced' in the app. Each: key, label, kind ('float'|'int'|'text'),
    # default, (min, max, step) for numbers, help text. Values arrive in self.options / max_per_window.
    advanced: list = [
        dict(key="max_per_window", label="Max labels per window", kind="int", default=10,
             range=(1, 50, 1), help="Keep at most this many labels per window (highest scores first)."),
    ]
    licence_note: str = ""
    description: str = ""

    def __init__(self, threshold: float | None = None, threads: int = 0, **options):
        self.threshold = self.default_threshold if threshold is None else float(threshold)
        self.threads = threads
        self.max_per_window = int(options.pop("max_per_window", self.max_per_window))
        q = options.pop("quiet_dbfs", None)
        self.quiet_dbfs = None if q in (None, "", 0) else float(q)
        self.options = options
        self._loaded = False

    # -- lifecycle -------------------------------------------------------------
    def hop_s(self) -> float | None:
        """Step between window starts (for joining windows into sound intervals); None = not windowed."""
        return self.window_s - float(self.options.get("overlap_s", 0.0) or 0.0)

    def window_len_s(self) -> float:
        return self.window_s

    def ensure_loaded(self, progress=None) -> None:
        if not self._loaded:
            self.load(progress=progress)
            self._loaded = True

    def load(self, progress=None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def process(self, audio: np.ndarray, offset_s: float, file: str) -> ModelOutput:  # pragma: no cover
        """`audio` is mono float32 at self.samplerate; `offset_s` is its start time in the file."""
        raise NotImplementedError

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """(usable, reason). Optional models override this (e.g. when extra packages are missing)."""
        return True, ""

    def loud_indices(self, frames: np.ndarray) -> np.ndarray:
        """Indices of frames to analyse: all, or only those louder than quiet_dbfs (RMS, dB full scale)."""
        if self.quiet_dbfs is None or len(frames) == 0:
            return np.arange(len(frames))
        f = frames - frames.mean(axis=1, keepdims=True)
        db = 20 * np.log10(np.sqrt((f.astype(np.float64) ** 2).mean(axis=1)) + 1e-12)
        return np.nonzero(db >= self.quiet_dbfs)[0]

    def config(self) -> dict:
        return {"key": self.key, "threshold": self.threshold, **self.options}


def sigmoid(x: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-sensitivity * np.clip(x, -30, 30)))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cap_per_window(scores: np.ndarray, threshold: float, k: int) -> np.ndarray:
    """Indices of classes >= threshold, best first, at most k."""
    idx = np.nonzero(scores >= threshold)[0]
    if len(idx) > k:
        idx = idx[np.argsort(scores[idx])[::-1][:k]]
    return idx


def split_birdnet_label(label: str) -> tuple[str, str]:
    """'Turdus merula_Eurasian Blackbird' -> ('Turdus merula', 'Eurasian Blackbird')."""
    if "_" in label:
        sci, common = label.split("_", 1)
        return sci.strip(), common.strip()
    return label.strip(), label.strip()
