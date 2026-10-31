"""Acoustic indices (soundscape summaries) via scikit-maad. Computed per block (default 60 s)."""

from __future__ import annotations

import warnings

import numpy as np

INDEX_COLUMNS = ["file", "start_s", "end_s", "ACI", "ADI", "AEI", "BI", "NDSI", "H", "Hf", "Ht", "rms_dbfs"]

INDEX_HELP = {
    "ACI": "Acoustic Complexity: frame-to-frame intensity change (birdsong high, steady noise low)",
    "ADI": "Acoustic Diversity: Shannon diversity of energy across 1 kHz bands",
    "AEI": "Acoustic Evenness: Gini of energy across bands (low = even spread)",
    "BI": "Bioacoustic Index: energy in the 2-11 kHz band above its minimum",
    "NDSI": "Normalised Difference Soundscape Index: (bio - anthro)/(bio + anthro), -1..1",
    "H": "Total acoustic entropy (Hf x Ht): low for silence or pure tones",
    "Hf": "Spectral entropy",
    "Ht": "Temporal entropy",
    "rms_dbfs": "Overall level in dB full scale",
}


class AcousticIndices:
    key = "indices"
    name = "Acoustic indices (scikit-maad)"
    description = "Soundscape summaries per minute: ACI, ADI, AEI, BI, NDSI, entropy, level."

    advanced = [
        dict(key="bio_low_hz", label="Biophony band from (Hz)", kind="int", default=2000, range=(100, 20000, 100),
             help="Lower edge of the 'animal sound' band used by BI and NDSI."),
        dict(key="bio_high_hz", label="Biophony band to (Hz)", kind="int", default=11000, range=(500, 24000, 500),
             help="Upper edge of the 'animal sound' band used by BI and NDSI."),
        dict(key="anthro_low_hz", label="Anthrophony band from (Hz)", kind="int", default=1000, range=(0, 10000, 100),
             help="Lower edge of the 'human noise' band used by NDSI (traffic, engines)."),
        dict(key="anthro_high_hz", label="Anthrophony band to (Hz)", kind="int", default=2000, range=(100, 10000, 100),
             help="Upper edge of the 'human noise' band used by NDSI."),
        dict(key="db_threshold", label="ADI/AEI threshold (dB)", kind="float", default=-50.0, range=(-90.0, -10.0, 5.0),
             help="Frequency bands quieter than this don't count as 'occupied' for ADI and AEI."),
    ]

    def __init__(self, **options):
        self.options = options

    def ensure_loaded(self, progress=None) -> None:
        from maad import features, sound  # noqa: F401 - import check

    def config(self) -> dict:
        return {"key": self.key, **self.options}

    def process(self, x: np.ndarray, sr: int, offset_s: float, file: str) -> dict | None:
        from maad import features, sound

        if len(x) < sr:  # skip tails shorter than 1 s
            return None
        fmax = min(sr // 2, 20000)
        o = self.options
        bio = (float(o.get("bio_low_hz", 2000)), min(float(o.get("bio_high_hz", 11000)), fmax))
        anth = (float(o.get("anthro_low_hz", 1000)), min(float(o.get("anthro_high_hz", 2000)), fmax))
        dbt = float(o.get("db_threshold", -50))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Sxx, tn, fn, _ = sound.spectrogram(x, sr, nperseg=1024, noverlap=0, mode="amplitude")
            Sp = Sxx ** 2
            aci = features.acoustic_complexity_index(Sxx)[2]
            adi = features.acoustic_diversity_index(Sxx, fn, fmin=0, fmax=fmax, bin_step=1000, dB_threshold=dbt)
            aei = features.acoustic_eveness_index(Sxx, fn, fmin=0, fmax=fmax, bin_step=1000, dB_threshold=dbt)
            bi = features.bioacoustics_index(Sxx, fn, flim=bio)
            ndsi = features.soundscape_index(Sp, fn, flim_bioPh=bio, flim_antroPh=anth)[0]
            hf = features.frequency_entropy(Sp)[0]
            ht = features.temporal_entropy(x)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        return dict(
            file=file, start_s=round(offset_s, 3), end_s=round(offset_s + len(x) / sr, 3),
            ACI=float(aci), ADI=float(adi), AEI=float(aei), BI=float(bi), NDSI=float(ndsi),
            H=float(hf * ht), Hf=float(hf), Ht=float(ht),
            rms_dbfs=round(20 * np.log10(rms + 1e-12), 2),
        )
