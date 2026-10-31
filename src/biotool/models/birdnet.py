"""BirdNET v2.4 (Cornell / Chemnitz) - ~6,500 classes, mostly birds plus a few non-bird classes."""

from __future__ import annotations

import numpy as np

from ..audio import frame_signal
from ..downloads import birdnet_files
from .base import ModelAdapter, ModelOutput, cap_per_window, sigmoid, split_birdnet_label

NON_SPECIES_GROUPS = {
    "Dog": "biophony",
    "Engine": "anthropophony",
    "Fireworks": "anthropophony",
    "Gun": "anthropophony",
    "Human non-vocal": "anthropophony",
    "Human vocal": "anthropophony",
    "Human whistle": "anthropophony",
    "Power tools": "anthropophony",
    "Siren": "anthropophony",
    "Environmental": "geophony",
    "Noise": "other",
}


def _interpreter(model_path: str, threads: int, embeddings: bool):
    """TFLite interpreter.

    Embeddings are the input of the final layer. The default XNNPACK delegate hides that
    tensor, so when embeddings are wanted we run the builtin kernels instead (~10 % slower).
    (experimental_preserve_all_tensors would also work but uses GBs of RAM.)
    """
    try:  # LiteRT is the successor of tf.lite and lighter where available
        from ai_edge_litert import interpreter as lite  # type: ignore
        Interpreter, Resolver = lite.Interpreter, lite.OpResolverType
    except ImportError:
        import warnings
        import tensorflow as tf

        warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
        Interpreter, Resolver = tf.lite.Interpreter, tf.lite.experimental.OpResolverType
    kw = dict(model_path=model_path, num_threads=threads or None)
    if embeddings:
        kw["experimental_op_resolver_type"] = Resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    return Interpreter(**kw)


class BirdNET(ModelAdapter):
    key = "birdnet"
    name = "BirdNET v2.4"
    samplerate = 48000
    window_s = 3.0
    default_threshold = 0.25
    fmax_hz = 15000
    advanced = [
        dict(key="overlap_s", label="Window overlap (s)", kind="float", default=0.0, range=(0.0, 2.5, 0.5),
             help="Overlap between 3 s windows. 1.5 = a window every 1.5 s: fewer calls split across\n"
                  "window edges, about twice as slow."),
        dict(key="sensitivity", label="Sensitivity", kind="float", default=1.0, range=(0.5, 1.5, 0.05),
             help="BirdNET's sensitivity: >1 raises scores (more detections), <1 lowers them."),
    ] + ModelAdapter.advanced
    licence_note = "BirdNET model: CC BY-NC-SA 4.0 (non-commercial use)."
    description = "Global bird classifier (~6,000 species) plus a few insects, dog, human, engine, siren, gun..."

    def load(self, progress=None) -> None:
        files = birdnet_files(progress)
        self.labels = [l.strip() for l in open(files["labels"], encoding="utf-8") if l.strip()]
        want_emb = bool(self.options.get("embeddings", True))
        self._interp = _interpreter(str(files["model"]), self.threads, want_emb)
        self._in = self._interp.get_input_details()[0]["index"]
        self._out = self._interp.get_output_details()[0]["index"]
        self._emb = next(
            (t["index"] for t in self._interp.get_tensor_details() if "GLOBAL_AVG_POOL" in t["name"]), None
        ) if want_emb else None
        self._batch = None
        parsed = [split_birdnet_label(l) for l in self.labels]
        self._sci = [p[0] for p in parsed]
        self._common = [p[1] for p in parsed]
        self._group = [NON_SPECIES_GROUPS.get(s, "biophony") for s in self._sci]
        self._key = [
            s.lower() if s not in NON_SPECIES_GROUPS else f"birdnet:{s.lower()}" for s in self._sci
        ]

    def _run(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self._batch != len(frames):
            self._interp.resize_tensor_input(self._in, [len(frames), frames.shape[1]])
            self._interp.allocate_tensors()
            self._batch = len(frames)
        self._interp.set_tensor(self._in, frames)
        self._interp.invoke()
        logits = self._interp.get_tensor(self._out)
        emb = self._interp.get_tensor(self._emb).copy() if self._emb is not None else None
        return logits, emb

    def process(self, audio, offset_s, file):
        out = ModelOutput()
        win = int(self.window_s * self.samplerate)
        overlap = float(self.options.get("overlap_s", 0.0))
        hop = int((self.window_s - overlap) * self.samplerate)
        frames = frame_signal(audio, win, hop)
        idx = self.loud_indices(frames)
        if len(idx) == 0:
            return out
        for b in range(0, len(idx), 32):
            ii = idx[b : b + 32]
            chunk = frames[ii]
            logits, emb = self._run(chunk)
            # Same convention as BirdNET-Analyzer: user sensitivity 0.5-1.5, higher = more detections,
            # applied as a flatter sigmoid (effective slope 2 - sensitivity).
            sens = min(1.5, max(0.5, 2.0 - float(self.options.get("sensitivity", 1.0))))
            probs = sigmoid(logits, sens)
            for i, fi in enumerate(ii):
                start = offset_s + fi * hop / self.samplerate
                for j in cap_per_window(probs[i], self.threshold, self.max_per_window):
                    out.detections.append(dict(
                        file=file, start_s=round(start, 3), end_s=round(start + self.window_s, 3),
                        model=self.key, label=self.labels[j], label_key=self._key[j],
                        scientific_name=self._sci[j], common_name=self._common[j],
                        group=self._group[j], score=round(float(probs[i, j]), 4),
                    ))
                if emb is not None:
                    out.emb_times.append(round(start, 3))
                    out.embeddings.append(emb[i].astype(np.float16))
        return out
