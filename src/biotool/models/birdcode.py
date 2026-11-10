"""BirdCODE (Earth Species Project) - zero-shot bird sound event detection.

BirdCODE marks *when* each bird vocalisation starts and ends (frame-level scores, several frames
per second) for thousands of species, instead of giving one score per fixed window.

    Code:   https://github.com/earthspecies/sound-event-detection (MIT)
    Model:  https://huggingface.co/EarthSpeciesProject/sed-birdcode
    Paper:  bioRxiv 10.64898/2026.07.31.742086

BioTool reuses the original code for everything that affects results: the model, the
overlapping-window inference (`FrameDetector.run`), resampling (librosa kaiser_best) and the
post-processing (threshold -> events -> merge same-species events -> cross-class NMS).

Two practical differences from the original command-line tool:
  * The model is built without first downloading the BEATs backbone from a Google Cloud bucket.
    That download is only an initialisation that the BirdCODE checkpoint then overwrites; skipping
    it lets BirdCODE run where Google Cloud Storage is blocked. Loading checks that the checkpoint
    covers every model weight, so nothing is left at a random value.
  * In a run, BirdCODE gets the whole recording at once (like the original), not BioTool's 60 s
    blocks. To keep memory low on long files, frame scores are kept only for species that reach the
    threshold somewhere; events only contain frames above the threshold and their score is the mean
    over those frames, so the result is the same as the original's.

Needs the optional dependencies:  InstallBirdCODE.bat  /  uv sync --extra birdcode
The model (a few hundred MB) is downloaded from Hugging Face on first use into the models folder; copying
that folder to another computer works without internet.
"""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import models_dir
from .base import ModelAdapter, ModelOutput

HF_REPO = "EarthSpeciesProject/sed-birdcode"
FOLDER = "birdcode"
NEEDED = ("config.json", "checkpoint/best_model.pt", "checkpoint/labels.txt")
_COLS = {"Begin Time (s)": "start", "End Time (s)": "end", "Species": "label", "Score": "score"}


def available() -> tuple[bool, str]:
    """Are the optional dependencies installed?"""
    missing = [m for m in ("torch", "librosa", "huggingface_hub", "sound_event_detection")
               if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"not installed ({', '.join(missing)} missing) - run InstallBirdCODE.bat"
    return True, ""


def model_folder() -> Path:
    return models_dir() / FOLDER


def ensure_files(progress=None) -> Path:
    """Download the checkpoint from Hugging Face once; afterwards it works offline."""
    d = model_folder()
    if all((d / f).exists() for f in NEEDED):
        return d
    if progress:
        progress("BirdCODE: downloading the model from Hugging Face (first use, a few hundred MB) ...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=HF_REPO, local_dir=str(d))
    except Exception as e:
        raise RuntimeError(
            f"Could not download BirdCODE from huggingface.co ({e}).\n"
            f"If this network blocks Hugging Face, copy the folder\n  {d}\n"
            "from a computer where BirdCODE already ran (it holds config.json and checkpoint/).") from e
    missing = [f for f in NEEDED if not (d / f).exists()]
    if missing:
        raise RuntimeError(f"BirdCODE download is incomplete, missing {missing} in {d}")
    return d


def build_detector(folder: Path):
    """FrameDetector from a local checkpoint folder, without fetching backbone weights."""
    import avex
    import torch
    import yaml
    import sound_event_detection.models.frame_detector as fd
    from sound_event_detection.utils.io_utils import load_labels

    cfg = yaml.safe_load((folder / "config.json").read_text(encoding="utf-8"))
    with redirect_stdout(io.StringIO()):
        labels = load_labels(str(folder / "checkpoint" / "labels.txt"))
    original = fd.load_model

    def no_download(name, *a, checkpoint_path=None, **k):  # random init; checkpoint overwrites it
        spec = avex.get_model_spec(name)
        if spec is not None and getattr(spec, "pretrained", False):
            avex.register_model(name, spec.model_copy(update={"pretrained": False}))
        return original(name, *a, checkpoint_path=None, **k)

    fd.load_model = no_download
    try:
        with redirect_stdout(io.StringIO()):
            model = fd.create_detector_from_config(cfg, labels=labels)
    finally:
        fd.load_model = original
    model.pooling_temperature = float(cfg.get("pooling_temperature", 1.0))
    state = torch.load(str(folder / "checkpoint" / "best_model.pt"), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"BirdCODE checkpoint does not cover {len(res.missing_keys)} model weights "
                           f"(e.g. {res.missing_keys[:3]}); refusing to run with random weights.")
    return model.eval()


class BirdCODE(ModelAdapter):
    key = "birdcode"
    name = "BirdCODE (ESP)"
    samplerate = 32000
    native_sr = True          # resamples itself with librosa kaiser_best, like the original code
    window_s = 5.0
    default_threshold = 0.5
    fmax_hz = 16000
    produces_embeddings = False
    whole_file = True         # the pipeline hands over the full recording via add_block / finish_file
    licence_note = "BirdCODE: code MIT (Earth Species Project); model from Hugging Face. Optional - needs PyTorch."
    description = ("Zero-shot detector for thousands of bird species: marks the start and end of each "
                   "call or song (not fixed windows). Optional, needs PyTorch.")
    advanced = [
        dict(key="overlap", label="Window overlap (fraction)", kind="float", default=0.5,
             range=(0.0, 0.9, 0.1),
             help="Overlap of the model's 5 s windows; only the centre frames of each window are kept\n"
                  "(BirdCODE default 0.5; 0 = no overlap, faster)."),
        dict(key="merge_gap_s", label="Merge gap (s)", kind="float", default=1.0, range=(0.0, 10.0, 0.5),
             help="Join detections of the same species separated by less than this (0 = off)."),
        dict(key="nms_iou", label="Overlap suppression (IoU)", kind="float", default=0.8,
             range=(0.05, 1.0, 0.05),
             help="When events of different species overlap at least this much, keep only the\n"
                  "highest-scoring one (BirdCODE default 0.8; 1.0 = practically off)."),
    ]

    def __init__(self, threshold=None, threads=0, **options):
        options.pop("embeddings", None)
        super().__init__(threshold, threads, **options)
        self.max_per_window = 10_000  # events, not windows: no per-window cap

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        return available()

    def hop_s(self) -> None:  # reports events with their own start/end, not windows
        return None

    def pp_config(self) -> dict:
        cfg: dict = {}
        gap = float(self.options.get("merge_gap_s", 1.0))
        if gap > 0:
            cfg["merge_max_gap"] = gap
        cfg["nms"] = {"iou_threshold": float(self.options.get("nms_iou", 0.8))}
        return cfg

    def load(self, progress=None) -> None:
        ok, msg = available()
        if not ok:
            raise RuntimeError(f"BirdCODE is {msg}")
        import torch

        if self.threads:
            torch.set_num_threads(self.threads)
        folder = ensure_files(progress)
        if progress:
            progress("BirdCODE: loading model ...")
        from sound_event_detection.inference.lsi_postprocess_cli import (
            postprocess_and_convert_detector_output_to_selection_table as to_table)
        from sound_event_detection.utils.postprocessing import postprocess_selection_table

        self._model = build_detector(folder)
        self._to_table, self._pp = to_table, postprocess_selection_table
        self._torch = torch
        self.samplerate = int(self._model.sample_rate)
        self.input_sr = self.samplerate

    def table(self, audio: np.ndarray, sr: int) -> pd.DataFrame:
        """Original BirdCODE selection table (Begin/End Time, Species, Score) for one stretch of audio."""
        x = self._resampled(audio, sr)
        if len(x) < self.samplerate // 10:  # shorter than 0.1 s: nothing to detect
            return pd.DataFrame(columns=list(_COLS))
        ov = float(self.options.get("overlap", 0.5))
        with self._torch.no_grad():
            preds = self._model.run(x[None, :], overlap=ov if ov > 0 else None)
        tsv = self._to_table(preds, self.threshold, self.pp_config())
        return pd.read_csv(io.StringIO(tsv), sep="\t")

    def _rows(self, t: pd.DataFrame, offset_s: float, file: str) -> list[dict]:
        rows = []
        for r in t.itertuples(index=False):
            sci = str(r[2])
            rows.append(dict(file=file, start_s=round(offset_s + float(r[0]), 3),
                             end_s=round(offset_s + float(r[1]), 3), model=self.key, label=sci,
                             label_key=sci.lower(), scientific_name=sci, common_name="",
                             group="biophony", score=round(float(r[3]), 4)))
        return rows

    # -- whole-file run (pipeline) ------------------------------------------------
    def add_block(self, audio: np.ndarray, sr: int, offset_s: float) -> None:
        if offset_s == 0 or not getattr(self, "_blocks", None):  # new file (or after a stopped run)
            self._blocks, self._block_sr = [], sr
        self._blocks.append(np.asarray(audio, dtype=np.float32))

    def finish_file(self, file: str) -> list[dict]:
        blocks, sr = getattr(self, "_blocks", None) or [], getattr(self, "_block_sr", self.samplerate)
        self._blocks = []
        if not blocks:
            return []
        audio = np.concatenate(blocks)
        del blocks
        t = self.table_long(audio, sr)
        return self._rows(t[list(_COLS)], 0.0, file) if len(t) else []

    def _resampled(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        x = np.asarray(audio, dtype=np.float32)
        if sr != self.samplerate:
            x = librosa.resample(y=x, orig_sr=sr, target_sr=self.samplerate, scale=True, res_type="kaiser_best")
        return np.ascontiguousarray(x, dtype=np.float32)

    def table_long(self, audio: np.ndarray, sr: int, windows_per_chunk: int = 128) -> pd.DataFrame:
        """Same result as table(), with memory independent of the recording length.

        Mirrors FrameDetector.run / _run_windowed_inference (same windows, margins, batches of 32,
        end padding), but keeps frame scores only for classes that reach the threshold.
        """
        from esp_research.protocols.detector import DetectorOutput

        x = self._resampled(audio, sr)
        if len(x) < self.samplerate // 10:
            return pd.DataFrame(columns=list(_COLS))
        mdl, torch = self._model, self._torch
        win = int(mdl.window_duration * mdl.sample_rate)
        fpw = int(mdl.window_duration * mdl.frame_rate)
        ov = float(self.options.get("overlap", 0.5))
        if ov > 0:
            margin = max(2, min(int(round(ov * fpw)), fpw - 1))
            left, right = margin // 2, margin - margin // 2
        else:
            left = right = 0
        keep = fpw - left - right
        hop = win if left + right == 0 else int(round(win * keep / fpw))
        n_win = 1 if len(x) <= win else 1 + int(np.ceil((len(x) - win) / hop))
        n_pad = max(0, (n_win - 1) * hop + win - len(x))
        if n_pad:
            x = np.pad(x, (0, n_pad))
        thr = self.threshold
        hits_t, hits_c, hits_p = [], [], []
        frame0 = 0
        batch = 32
        chunk = max(batch, windows_per_chunk // batch * batch)
        for c0 in range(0, n_win, chunk):
            idx = range(c0, min(n_win, c0 + chunk))
            probs = []
            for b0 in range(idx.start, idx.stop, batch):
                wins = np.stack([x[i * hop:i * hop + win] for i in range(b0, min(idx.stop, b0 + batch))])
                with torch.no_grad():
                    probs.append(torch.sigmoid(mdl.forward(torch.from_numpy(wins).float())).cpu().numpy())
            probs = np.concatenate(probs)  # (windows, fpw, classes)
            for k, i in enumerate(idx):
                a = 0 if (i == 0 or left + right == 0) else left
                b = fpw if (i == n_win - 1 or left + right == 0) else fpw - right
                p = probs[k, a:b]
                t, c = np.nonzero(p >= thr)
                if len(t):
                    hits_t.append(t + frame0)
                    hits_c.append(c)
                    hits_p.append(p[t, c])
                frame0 += b - a
        n_frames = frame0
        if n_pad:
            n_pad_frames = int(np.ceil(n_pad / mdl.sample_rate * mdl.frame_rate))
            if 0 < n_pad_frames < n_frames:
                n_frames -= n_pad_frames
        if not hits_t:
            return pd.DataFrame(columns=list(_COLS))
        t, c, p = np.concatenate(hits_t), np.concatenate(hits_c), np.concatenate(hits_p)
        ok = t < n_frames
        t, c, p = t[ok], c[ok], p[ok]
        active = np.unique(c)
        dense = np.zeros((n_frames, len(active)), dtype=np.float32)  # below-threshold frames never matter
        dense[t, np.searchsorted(active, c)] = p
        labels = [mdl.labels[j] for j in active]
        preds = DetectorOutput(predictions=dense[None], frame_rate=mdl.frame_rate, class_names=labels)
        tsv = self._to_table(preds, thr, self.pp_config())
        return pd.read_csv(io.StringIO(tsv), sep="\t")

    def process(self, audio, offset_s, file):
        out = ModelOutput()
        sr = int(getattr(self, "input_sr", self.samplerate))
        t = self.table(audio, sr)
        out.detections = self._rows(t[list(_COLS)], offset_s, file) if len(t) else []
        return out
