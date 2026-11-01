"""Fast unit tests: no models, no network."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from biotool import merge
from biotool.audio import HighPass, frame_signal, resample
from biotool.evaluate import _prf, norm, powdermill_codes
from biotool.location import date_from_file, week48
from biotool.models.base import DETECTION_COLUMNS, cap_per_window, sigmoid, softmax, split_birdnet_label


def det(rows):
    base = dict(file="f.wav", label_key="", scientific_name="", common_name="", group="biophony")
    return pd.DataFrame([{**base, **r} for r in rows], columns=DETECTION_COLUMNS)


# ---------------------------------------------------------------- audio
def test_frame_signal_pads_and_drops_tiny_tail():
    x = np.ones(10_000, dtype=np.float32)
    assert frame_signal(x, 3000).shape == (4, 3000)  # 10000 = 3x3000 + 1000: 33 % tail kept, padded
    assert frame_signal(np.ones(9_500, np.float32), 3000).shape == (3, 3000)  # 500 tail (<25 %) dropped
    assert frame_signal(np.ones(1000, np.float32), 3000).shape == (1, 3000)


def test_resample_length():
    x = np.random.default_rng(0).standard_normal(44_100).astype(np.float32)
    assert len(resample(x, 44_100, 48_000)) == 48_000


def test_highpass_blockwise_equals_single_pass_and_removes_low_tone():
    sr = 16_000
    t = np.arange(sr * 10) / sr
    x = (np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    a = HighPass(500, sr)
    blocks = np.concatenate([a(x[i:i + 3 * sr]) for i in range(0, len(x), 3 * sr)])
    whole = HighPass(500, sr)(x)
    assert np.abs(blocks - whole).max() < 1e-4
    spec = np.abs(np.fft.rfft(whole[sr:2 * sr])) / (sr / 2)
    assert spec[60] < 0.01 and spec[2000] > 0.95


# ---------------------------------------------------------------- scores
def test_softmax_and_cap():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(p.sum(), 1.0)
    s = np.array([0.1, 0.9, 0.5, 0.05])
    assert list(cap_per_window(s, 0.08, 2)) == [1, 2]
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_split_birdnet_label():
    assert split_birdnet_label("Turdus merula_Eurasian Blackbird") == ("Turdus merula", "Eurasian Blackbird")
    assert split_birdnet_label("Engine_Engine") == ("Engine", "Engine")


# ---------------------------------------------------------------- merge
def test_species_agreement_across_window_lengths():
    d = det([
        dict(start_s=0, end_s=3, model="birdnet", label_key="turdus merula", scientific_name="Turdus merula", score=0.8),
        dict(start_s=0, end_s=5, model="perch_v2", label_key="turdus merula", scientific_name="Turdus merula", score=0.3),
    ])
    a = merge.species_agreement(d, 3.0)
    first = a[a["win_start_s"] == 0].iloc[0]
    assert first["n_models"] == 2 and first["models"] == "birdnet+perch_v2" and first["max_score"] == 0.8
    # the 5 s Perch window also covers 3-6 s
    assert (a["win_start_s"] == 3).any()


def test_raven_table_columns():
    d = det([dict(start_s=1, end_s=4, model="yamnet", label="Rain", common_name="Rain", group="geophony", score=0.4)])
    r = merge.raven_table(d, 32000)
    assert list(r.columns[:7]) == ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
                                   "Low Freq (Hz)", "High Freq (Hz)"]
    assert r["High Freq (Hz)"].iloc[0] == 16000


# ---------------------------------------------------------------- location / evaluation helpers
def test_week48_and_dates_from_file_names():
    assert week48(date(2023, 1, 1)) == 1
    assert week48(date(2023, 12, 31)) == 48
    assert date_from_file("20230415_063000.WAV") == date(2023, 4, 15)
    assert date_from_file("SITE01_20230415_063000.wav") == date(2023, 4, 15)
    assert date_from_file("Recording_2_Segment_07.wav") is None


def test_prf_and_synonyms(tmp_path):
    assert _prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert norm("Picoides pubescens") == "dryobates pubescens"
    readme = tmp_path / "README.txt"
    readme.write_text("---Alpha Codes---\nCommon Name\t\t\tAlpha Code\tLatin Name\n"
                      "Wood thrush\t\t\tWOTH\t\tHylocichla mustelina\n", encoding="utf-8")
    assert powdermill_codes(readme) == {"WOTH": "hylocichla mustelina"}


# ---------------------------------------------------------------- explore (clustering)
def test_cluster_separates_obvious_groups():
    from biotool import explore

    rng = np.random.default_rng(1)
    centres = rng.standard_normal((3, 64)) * 5
    X = np.concatenate([c + rng.standard_normal((20, 64)) * 0.3 for c in centres]).astype(np.float32)
    meta = pd.DataFrame({"stem": "f", "source_file": "", "start_s": np.arange(60) * 3.0,
                         "end_s": np.arange(60) * 3.0 + 3, "labels": "", "identified": False,
                         "any_detection": False})
    w, s = explore.cluster(meta, X, "all", min_cluster_size=5)
    assert (s["cluster"] >= 0).sum() == 3
    # each true group lands in one cluster
    for g in range(3):
        assert w["cluster"].iloc[g * 20:(g + 1) * 20].nunique() == 1


# ---------------------------------------------------------------- listening filters
def test_band_filter_and_hiss_reduction():
    from biotool.gui.player import band_filter, reduce_hiss

    sr = 32_000
    t = np.arange(sr * 4) / sr
    rng = np.random.default_rng(2)
    tone = np.sin(2 * np.pi * 3000 * t) * (t % 1 < 0.3)  # 0.3 s bursts every second
    x = (0.3 * tone + 0.05 * rng.standard_normal(len(t)) + 0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    y = band_filter(x, sr, 2000, 5000)
    spec = np.abs(np.fft.rfft(y)) / (len(y) / 2)
    assert spec[int(200 * 4)] < 0.01  # 200 Hz removed
    z = reduce_hiss(y, sr, y, strength=0.8)
    quiet = (t % 1 > 0.5)
    assert np.sqrt(np.mean(z[quiet] ** 2)) < 0.5 * np.sqrt(np.mean(y[quiet] ** 2))  # gaps get quieter


# ---------------------------------------------------------------- species list / quiet windows
def test_species_list_include_exclude(tmp_path):
    from biotool.species_list import SpeciesList, SpeciesListSettings, read_names

    f = tmp_path / "list.txt"
    f.write_text("# comment\nWood Thrush\nPoecile atricapillus_Black-capped Chickadee\n\n", encoding="utf-8")
    assert {"wood thrush", "poecile atricapillus", "black-capped chickadee"} <= read_names(f)
    d = det([
        dict(start_s=0, end_s=3, model="birdnet", label_key="hylocichla mustelina",
             scientific_name="Hylocichla mustelina", common_name="Wood Thrush", score=0.9),
        dict(start_s=0, end_s=3, model="birdnet", label_key="turdus merula",
             scientific_name="Turdus merula", common_name="Eurasian Blackbird", score=0.8),
        dict(start_s=0, end_s=3, model="yamnet", label_key="audioset:rain", common_name="Rain",
             group="geophony", score=0.5),
    ])
    keep, rem = SpeciesList(SpeciesListSettings(True, "include", str(f))).apply(d)
    assert set(keep["label_key"]) == {"hylocichla mustelina", "audioset:rain"}  # sound events untouched
    keep, rem = SpeciesList(SpeciesListSettings(True, "exclude", str(f))).apply(d)
    assert set(rem["label_key"]) == {"hylocichla mustelina"}


def test_loud_indices_skip_quiet_frames():
    from biotool.models.base import ModelAdapter

    m = ModelAdapter(quiet_dbfs=-40)
    frames = np.stack([np.full(100, 1e-4), np.sin(np.linspace(0, 50, 100)) * 0.1]).astype(np.float32)
    assert list(m.loud_indices(frames)) == [1]
    assert list(ModelAdapter().loud_indices(frames)) == [0, 1]


# ---------------------------------------------------------------- optional DeepForestSound adapter
def test_deepforestsound_registered_with_paper_thresholds():
    from biotool.models import DETECTORS
    from biotool.models.dfs import SPECIES, DeepForestSound

    assert "dfs" in DETECTORS
    ok, why = DeepForestSound.availability()
    assert isinstance(ok, bool) and (ok or "InstallDeepForestSound" in why)
    assert len(SPECIES) == 12 and dict((s[0], s[3]) for s in SPECIES)["pan_troglodytes"] == 0.99
    m = DeepForestSound()
    assert m._thresholds()[8] == 0.99  # per-species thresholds by default
    m.options["thresholds"] = "min score above"
    assert (m._thresholds() == m.threshold).all()
