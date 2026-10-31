"""YAMNet (Google) - general sound-event tagger, 521 AudioSet classes.

Used here mainly for anthropophony (speech, vehicles, aircraft, machinery...) and
geophony (wind, rain, thunder, water), plus coarse animal classes.
YAMNet works on 0.96 s frames every 0.48 s; we aggregate frames into fixed windows
(default 3 s) by taking the max score, so its output lines up with BirdNET.
"""

from __future__ import annotations

import csv

import numpy as np

from ..downloads import kaggle_model
from ..paths import RESOURCES
from .base import ModelAdapter, ModelOutput, cap_per_window

FRAME_HOP_S = 0.48
FRAME_LEN_S = 0.96


class YAMNet(ModelAdapter):
    key = "yamnet"
    name = "YAMNet (AudioSet)"
    samplerate = 16000
    window_s = 3.0
    default_threshold = 0.2
    fmax_hz = 8000
    licence_note = "YAMNet: Apache 2.0."
    description = "General sound tagger (521 classes): speech, vehicles, aircraft, wind, rain, insects, frogs..."

    # Very generic classes that add noise rather than information
    DEFAULT_IGNORE = {"Silence", "Outside, rural or natural", "Outside, urban or manmade",
                      "Inside, small room", "Inside, large room or hall", "Animal", "Wild animals",
                      "Environmental noise", "Noise", "Static", "White noise", "Pink noise", "Hum"}

    advanced = [
        dict(key="group_s", label="Grouping window (s)", kind="float", default=3.0, range=(0.96, 10.0, 0.5),
             help="YAMNet scores ~1 s frames every 0.48 s; frames are grouped into windows of this length.\n"
                  "Shorter = finer timing (more rows); 3 s lines up with BirdNET."),
        dict(key="group_method", label="Grouping method", kind="choice", default="max", choices=["max", "mean"],
             help="max = report a sound if any frame in the window has it (sensitive);\n"
                  "mean = average over the window (steadier, fewer false alarms in noise)."),
    ] + ModelAdapter.advanced + [
        dict(key="ignore", label="Ignore classes", kind="text",
             default="Silence; Outside, rural or natural; Outside, urban or manmade; Inside, small room; "
                     "Inside, large room or hall; Animal; Wild animals; Environmental noise; Noise; Static; "
                     "White noise; Pink noise; Hum",
             help="AudioSet classes to drop, separated by ';' (exact names, see yamnet_groups.csv).\n"
                  "E.g. add 'Turkey; Gobble' if those fire on birdsong at your sites."),
    ]

    def load(self, progress=None) -> None:
        import tensorflow as tf

        path = kaggle_model("yamnet", progress)
        self._model = tf.saved_model.load(str(path))
        self._tf = tf
        with open(RESOURCES / "yamnet_groups.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.labels = [r["display_name"] for r in rows]
        self.groups = [r["group"] for r in rows]
        ign = self.options.get("ignore", self.DEFAULT_IGNORE)
        if isinstance(ign, str):
            ign = [x.strip() for x in ign.split(";") if x.strip()]
        ignore = set(ign)
        unknown = ignore - set(self.labels)
        if unknown and progress:
            progress(f"YAMNet: ignore list has unknown class names: {', '.join(sorted(unknown))}")
        self._keep = np.array([l not in ignore for l in self.labels])

    def process(self, audio, offset_s, file):
        out = ModelOutput()
        if len(audio) < int(FRAME_LEN_S * self.samplerate):
            audio = np.pad(audio, (0, int(FRAME_LEN_S * self.samplerate) - len(audio)))
        scores, emb, _ = self._model(self._tf.constant(audio, dtype=self._tf.float32))
        scores, emb = scores.numpy(), emb.numpy()
        n = scores.shape[0]
        gs = float(self.options.get("group_s", self.window_s))
        use_mean = str(self.options.get("group_method", "max")) == "mean"
        frame_t = np.arange(n) * FRAME_HOP_S
        win_idx = (frame_t // gs + 1e-9).astype(int)
        block_end = len(audio) / self.samplerate
        for w in np.unique(win_idx):
            m = win_idx == w
            s = scores[m].mean(axis=0) if use_mean else scores[m].max(axis=0)
            start = offset_s + w * gs
            end = offset_s + min((w + 1) * gs, block_end)
            hits = cap_per_window(np.where(self._keep, s, 0.0), self.threshold, self.max_per_window)
            for j in hits:
                lab = self.labels[j]
                out.detections.append(dict(
                    file=file, start_s=round(start, 3), end_s=round(end, 3), model=self.key,
                    label=lab, label_key=f"audioset:{lab.lower()}", scientific_name="",
                    common_name=lab, group=self.groups[j], score=round(float(s[j]), 4),
                ))
            out.emb_times.append(round(start, 3))
            out.embeddings.append(emb[m].mean(axis=0).astype(np.float16))
        return out
