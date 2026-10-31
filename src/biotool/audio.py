"""Audio reading, block streaming and resampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}


@dataclass
class AudioInfo:
    path: Path
    samplerate: int
    channels: int
    duration_s: float


def info(path: str | Path) -> AudioInfo:
    i = sf.info(str(path))
    return AudioInfo(Path(path), int(i.samplerate), int(i.channels), float(i.frames) / i.samplerate)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def iter_blocks(path: str | Path, block_s: float = 60.0) -> Iterator[tuple[np.ndarray, int, float]]:
    """Yield (mono_float32_block, samplerate, offset_seconds) without loading the whole file."""
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        block = int(round(block_s * sr))
        offset = 0
        while True:
            data = f.read(block, dtype="float32", always_2d=False)
            if data is None or len(data) == 0:
                break
            yield to_mono(data), sr, offset / sr
            offset += len(data)


def load(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono(data), int(sr)


class HighPass:
    """Butterworth high-pass that keeps its state between blocks, so a long file filtered in
    60 s blocks gives the same result as filtering it in one go (no clicks at block edges)."""

    def __init__(self, cutoff_hz: float, samplerate: int, order: int = 4):
        from scipy.signal import butter, sosfilt_zi

        self.cutoff = float(cutoff_hz)
        self.sr = samplerate
        self.sos = butter(order, self.cutoff, btype="highpass", fs=samplerate, output="sos")
        self.zi = None
        self._zi0 = sosfilt_zi(self.sos)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt

        if self.zi is None:
            self.zi = self._zi0 * (x[0] if len(x) else 0.0)
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


def find_audio_files(folder: str | Path, recursive: bool = True) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def frame_signal(x: np.ndarray, frame_len: int, hop: int | None = None, pad: bool = True) -> np.ndarray:
    """Split a 1-D signal into frames [n, frame_len]; the last partial frame is zero-padded."""
    hop = hop or frame_len
    L = len(x)
    if L == 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    if pad:
        n = max(1, int(np.ceil(max(L - frame_len, 0) / hop)) + 1)
        # drop a final frame that would be mostly padding (<25 % real audio)
        if n > 1:
            real = L - (n - 1) * hop
            if real < 0.25 * frame_len:
                n -= 1
        need = (n - 1) * hop + frame_len
        if need > L:
            x = np.pad(x, (0, need - L))
    else:
        if L < frame_len:
            return np.zeros((0, frame_len), dtype=np.float32)
        n = 1 + (L - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return x[idx].astype(np.float32)
