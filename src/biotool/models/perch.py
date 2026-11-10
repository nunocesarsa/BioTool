"""Google Perch v2 - multi-taxa classifier (~15k classes: birds, amphibians, mammals, insects)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from ..audio import frame_signal
from ..downloads import kaggle_model
from ..paths import RESOURCES
from .base import ModelAdapter, ModelOutput, cap_per_window, softmax


def _normalize(frames: np.ndarray, target_peak: float = 0.25) -> np.ndarray:
    """Same per-window normalisation as Google's perch-hoplite loader."""
    f = frames - frames.mean(axis=-1, keepdims=True)
    peak = np.abs(f).max(axis=-1, keepdims=True)
    f = np.divide(f, peak, out=np.zeros_like(f), where=peak > 0)
    return (f * target_peak).astype(np.float32)


def _read_class_list(path: Path) -> list[str]:
    """perch class-list CSV: first row = namespace, then one class per row (first column)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r]
    return [r[0].strip() for r in rows[1:]]


class PerchV2(ModelAdapter):
    key = "perch_v2"
    name = "Perch v2 (Google)"
    samplerate = 32000
    window_s = 5.0
    default_threshold = 0.03  # best F1 on Powdermill (softmax scores are small)
    fmax_hz = 16000
    licence_note = "Perch v2: Apache 2.0."
    description = "Multi-taxa classifier (~15k classes: birds, frogs, mammals, insects). Strongest general model."

    advanced = [
        dict(key="overlap_s", label="Window overlap (s)", kind="float", default=0.0, range=(0.0, 4.0, 0.5),
             help="Overlap between 5 s windows. 2.5 = a window every 2.5 s: calls split across\n"
                  "window edges are caught, about twice as slow."),
    ] + ModelAdapter.advanced

    def load(self, progress=None) -> None:
        import tensorflow as tf

        if self.threads:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(self.threads)
            except RuntimeError:
                pass
        path = kaggle_model("perch_v2", progress)
        if progress:
            progress("Perch v2: loading model (can take ~30 s)")
        self._model = tf.saved_model.load(str(path))
        self._fn = self._model.signatures["serving_default"]
        csvs = sorted((path / "assets").glob("*.csv"))
        label_csv = next((c for c in csvs if c.stem == "labels"), csvs[0] if csvs else None)
        if label_csv is None:
            raise RuntimeError(f"No class list found in {path / 'assets'}")
        self.labels = _read_class_list(label_csv)
        self._tf = tf
        # Species ("Genus species") are biophony; Perch's ~200 non-species classes come from
        # FSD50K, which uses AudioSet names - map those to YAMNet's groups and keys.
        with open(RESOURCES / "yamnet_groups.csv", newline="", encoding="utf-8") as f:
            audioset = {r["display_name"].lower(): r["group"] for r in csv.DictReader(f)}
        binomial = re.compile(r"^[A-Z][a-z]+ [a-z]+( [a-z]+)?$")
        self._group, self._key = [], []
        for lab in self.labels:
            if binomial.match(lab):
                self._group.append("biophony"); self._key.append(lab.lower())
            else:
                n = lab.replace("_", " ").lower()
                self._group.append(audioset.get(n, "other")); self._key.append(f"audioset:{n}")

    def _run(self, frames: np.ndarray):
        out = self._fn(inputs=self._tf.constant(frames))
        logits = out["label"].numpy()
        emb = out["embedding"].numpy() if "embedding" in out else None
        return logits, emb

    def hop_s(self) -> float:
        return max(0.5, self.window_s - float(self.options.get("overlap_s", 0.0) or 0.0))

    def process(self, audio, offset_s, file):
        out = ModelOutput()
        win = int(self.window_s * self.samplerate)
        hop_s = self.hop_s()
        hop = int(hop_s * self.samplerate)
        frames = frame_signal(audio, win, hop)
        idx = self.loud_indices(frames)  # before normalisation (which removes level information)
        if len(idx) == 0:
            return out
        frames = _normalize(frames[idx])
        for b in range(0, len(frames), 16):
            chunk = frames[b : b + 16]
            ii = idx[b : b + 16]
            logits, emb = self._run(chunk)
            # Perch 2.0's classifier is trained with softmax cross-entropy (scores sum to 1 per
            # window; k overlapping species each get ~1/k). Sigmoid would mark thousands of
            # classes as ~1.0.
            probs = softmax(logits)
            for i, fi in enumerate(ii):
                start = offset_s + fi * hop_s
                for j in cap_per_window(probs[i], self.threshold, self.max_per_window):
                    lab = self.labels[j] if j < len(self.labels) else f"class_{j}"
                    sp = j < len(self.labels) and self._group[j] == "biophony" and not self._key[j].startswith("audioset:")
                    out.detections.append(dict(
                        file=file, start_s=round(start, 3), end_s=round(start + self.window_s, 3),
                        model=self.key, label=lab, label_key=self._key[j] if j < len(self._key) else lab.lower(),
                        scientific_name=lab if sp else "", common_name="" if sp else lab,
                        group=self._group[j] if j < len(self._group) else "other",
                        score=round(float(probs[i, j]), 4),
                    ))
                if emb is not None:
                    out.emb_times.append(round(start, 3))
                    out.embeddings.append(np.asarray(emb[i]).reshape(-1).astype(np.float16))
        return out
