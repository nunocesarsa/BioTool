"""Background workers so the window stays responsive while models run."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ..audio import info, iter_blocks
from ..pipeline import Engine, RunSettings, run_folder

_ENGINE_CACHE: dict[str, Engine] = {}


def get_engine(settings: RunSettings, progress) -> Engine:
    """Reuse loaded models between runs when the model settings are unchanged."""
    key = f"{settings.fingerprint()}|{settings.threads}"
    eng = _ENGINE_CACHE.get(key)
    if eng is None:
        _ENGINE_CACHE.clear()
        eng = Engine(settings, progress)
        eng.load()
        _ENGINE_CACHE[key] = eng
    eng.progress = progress
    eng.settings = settings
    return eng


def compute_spectrogram(path: str | Path, max_cols: int = 12000, nfft: int = 512):
    """dB spectrogram [freq, time] for display; long files are max-pooled in time."""
    inf = info(path)
    hop = nfft
    cols = []
    for block, sr, _ in iter_blocks(path, 60.0):
        n = len(block) // hop
        if n == 0:
            continue
        frames = block[: n * hop].reshape(n, hop) * np.hanning(hop)[None, :].astype(np.float32)
        cols.append(np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32))
    if not cols:
        return np.zeros((nfft // 2 + 1, 1), np.float32), inf
    S = np.concatenate(cols, axis=0)  # [time, freq]
    if S.shape[0] > max_cols:
        f = int(np.ceil(S.shape[0] / max_cols))
        pad = (-S.shape[0]) % f
        S = np.pad(S, ((0, pad), (0, 0))).reshape(-1, f, S.shape[1]).max(axis=1)
    S = 20 * np.log10(S + 1e-6)
    return S.T, inf


class _Worker(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop = False

    def stop(self):
        self._stop = True

    def should_stop(self):
        return self._stop


class FileWorker(_Worker):
    spectrogram = Signal(object, object)

    def __init__(self, path: str, settings: RunSettings, run_models: bool = True, nfft: int = 512):
        super().__init__()
        self.path, self.settings, self.run_models, self.nfft = path, settings, run_models, nfft

    def run(self):
        try:
            self.log.emit(f"Reading {Path(self.path).name} ...")
            S, inf = compute_spectrogram(self.path, nfft=self.nfft)
            self.spectrogram.emit(S, inf)
            if not self.run_models:
                self.finished.emit(None)
                return
            eng = get_engine(self.settings, self.log.emit)
            self.log.emit("Running models ...")
            res = eng.analyze_file(self.path, should_stop=self.should_stop)
            self.log.emit(f"Done: {len(res.detections)} detections in {res.seconds:.1f} s "
                          f"({res.duration_s / max(res.seconds, 1e-6):.0f}x real time)")
            for w in res.warnings:
                self.log.emit(f"Note: {w}")
            self.finished.emit(res)
        except InterruptedError:
            self.log.emit("Stopped.")
            self.finished.emit(None)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class SelectionWorker(_Worker):
    """Run the models on one selected span (Analyse selection)."""

    def __init__(self, path, settings: RunSettings, start_s, end_s, band_hz=None):
        super().__init__()
        self.args = (path, settings, start_s, end_s, band_hz)

    def run(self):
        try:
            from ..selection import analyse_selection

            path, settings, s, e, band = self.args
            eng = get_engine(settings, self.log.emit)
            self.log.emit(f"Analysing {s:.1f}-{e:.1f} s{' (band-filtered)' if band else ''} ...")
            df, note = analyse_selection(eng, path, s, e, band_hz=band)
            self.finished.emit((df, note))
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class FolderWorker(_Worker):
    def __init__(self, input_dir: str, output_dir: str, settings: RunSettings, recursive: bool):
        super().__init__()
        self.input_dir, self.output_dir = input_dir, output_dir
        self.settings, self.recursive = settings, recursive

    def run(self):
        try:
            eng = get_engine(self.settings, self.log.emit)
            summary = run_folder(self.input_dir, self.output_dir, self.settings, self.recursive,
                                 progress=self.log.emit,
                                 file_progress=lambda i, n, f: self.progress.emit(i, n, f),
                                 should_stop=self.should_stop, engine=eng)
            self.finished.emit(summary)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def start(worker: _Worker, owner) -> QThread:
    th = QThread(owner)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    worker.finished.connect(th.quit)
    worker.failed.connect(th.quit)
    th.finished.connect(worker.deleteLater)
    th.start()
    return th
