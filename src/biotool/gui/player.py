"""Play (or export) an exact time range of an audio file, optionally band-pass filtered and slowed.

Everything here works on a copy of the samples: original files are never modified.
Playback uses `sounddevice` (PortAudio, bundled in the Windows/macOS wheels).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import resample, to_mono


def read_segment(path: str, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(max(0.0, start_s) * sr))
        x = to_mono(f.read(max(1, int((end_s - start_s) * sr)), dtype="float32", always_2d=False))
    return x, sr


def band_filter(x: np.ndarray, sr: int, low_hz: float | None, high_hz: float | None) -> np.ndarray:
    """Zero-phase Butterworth band-pass (or high/low-pass if one edge is open)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2
    lo = low_hz if low_hz and low_hz > 20 else None
    hi = high_hz if high_hz and high_hz < 0.98 * nyq else None
    if lo and hi and hi <= lo * 1.05:
        hi = None
    if lo and hi:
        sos = butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    elif lo:
        sos = butter(6, lo, btype="highpass", fs=sr, output="sos")
    elif hi:
        sos = butter(6, hi, btype="lowpass", fs=sr, output="sos")
    else:
        return x
    if len(x) < 64:
        return x
    return sosfiltfilt(sos, x).astype(np.float32)


def reduce_hiss(x: np.ndarray, sr: int, noise_ref: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Spectral gating for listening: learn the background spectrum from the quietest moments of
    `noise_ref` (the selection plus a few seconds around it) and attenuate time-frequency cells that
    are not clearly above it. strength 0..1. Listening/export only - never used for the models."""
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    if strength <= 0 or len(x) < 2048:
        return x
    nper = 1024 if sr >= 32000 else 512
    _, _, R = stft(noise_ref, sr, nperseg=nper, noverlap=nper * 3 // 4)
    noise = np.percentile(np.abs(R), 20, axis=1, keepdims=True)  # background per frequency
    _, _, Z = stft(x, sr, nperseg=nper, noverlap=nper * 3 // 4)
    mag = np.abs(Z) + 1e-12
    k = 1.0 + 2.5 * strength                      # how far above the background a cell must be
    floor = 10 ** (-(6 + 24 * strength) / 20)     # max attenuation: -6 dB (light) .. -30 dB (strong)
    gain = np.clip((mag - k * noise) / mag, 0.0, 1.0)
    gain = uniform_filter(gain, size=(3, 5))      # smooth over frequency x time: fewer artefacts
    gain = np.maximum(gain, floor)
    _, y = istft(Z * gain, sr, nperseg=nper, noverlap=nper * 3 // 4)
    return y[: len(x)].astype(np.float32)


def prepare(path: str, start_s: float, end_s: float, band=None, hiss: float = 0.0):
    """Read a segment and apply the listening filters (band-pass, hiss reduction)."""
    x, sr = read_segment(path, start_s, end_s)
    if hiss > 0:
        ctx, _ = read_segment(path, max(0.0, start_s - 5), end_s + 5)
        if band:
            ctx = band_filter(ctx, sr, *band)
    if band:
        x = band_filter(x, sr, *band)
    if hiss > 0:
        x = reduce_hiss(x, sr, ctx, hiss)
    return x, sr


def normalise(x: np.ndarray, max_gain: float = 30.0) -> np.ndarray:
    peak = float(np.abs(x).max()) if len(x) else 0.0
    return x * min(0.9 / peak, max_gain) if peak > 0 else x


def export_clip(path: str, start_s: float, end_s: float, out_path: str | Path,
                band: tuple[float, float] | None = None, speed: float = 1.0, hiss: float = 0.0) -> Path:
    """Write a WAV copy of the segment (optionally band-passed / hiss-reduced / time-expanded)."""
    x, sr = prepare(path, start_s, end_s, band, hiss)
    out_sr = int(round(sr * speed)) if speed != 1.0 else sr  # time expansion: same samples, lower rate
    out_path = Path(out_path)
    sf.write(str(out_path), np.clip(x, -1, 1), out_sr, subtype="PCM_16")
    return out_path


class SegmentPlayer:
    def __init__(self, parent=None):
        self._until = 0.0

    @property
    def playing(self) -> bool:
        return time.time() < self._until

    def stop(self):
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        self._until = 0.0

    def play(self, path: str, start_s: float, end_s: float, gain_to_peak: bool = True,
             speed: float = 1.0, band: tuple[float, float] | None = None, hiss: float = 0.0) -> str | None:
        """Start playback in the background; returns an error message or None.

        speed < 1 slows playback down (time expansion: pitch drops by the same factor, as with
        bat detectors). band = (low_hz, high_hz) plays only that frequency band.
        """
        try:
            import sounddevice as sd
        except Exception as e:  # missing PortAudio etc.
            return f"Audio playback is not available ({e})."
        self.stop()
        x, sr = prepare(path, start_s, end_s, band, hiss)
        if len(x) == 0:
            return "Selection is empty."
        if gain_to_peak:  # field recordings are often quiet: normalise the selection
            x = normalise(x)
        try:
            out_rate = int(sd.query_devices(kind="output")["default_samplerate"]) or 48000
        except Exception:
            out_rate = 48000
        # Slowing down = treating the samples as if recorded at sr*speed, then converting to the
        # device rate. At speed 1 we try the file's own rate first (no resampling).
        play_x, play_sr = x, sr
        if speed != 1.0:
            play_x, play_sr = resample(x, int(round(sr * speed)), out_rate), out_rate
        try:
            sd.play(play_x, play_sr)
        except Exception:
            try:  # some devices only accept their native rate
                play_x = resample(play_x, play_sr, out_rate)
                play_sr = out_rate
                sd.play(play_x, play_sr)
            except Exception as e:
                return f"Could not play audio: {e}"
        self._until = time.time() + len(play_x) / play_sr
        return None
