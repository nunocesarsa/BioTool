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


def test_birdcode_registered_with_original_defaults():
    from biotool.models import DETECTORS
    from biotool.models.birdcode import BirdCODE

    assert DETECTORS["birdcode"] is BirdCODE
    ok, msg = BirdCODE.availability()
    assert ok or "InstallBirdCODE.bat" in msg
    m = BirdCODE(embeddings=True)  # pipeline passes embeddings=...; BirdCODE has none
    assert m.threshold == 0.5 and "embeddings" not in m.options
    assert m.pp_config() == {"merge_max_gap": 1.0, "nms": {"iou_threshold": 0.8}}
    assert BirdCODE(merge_gap_s=0).pp_config() == {"nms": {"iou_threshold": 0.8}}


def test_birdcode_gets_whole_recording_not_blocks():
    from biotool.models.birdcode import BirdCODE

    m = BirdCODE()
    assert m.whole_file
    m.add_block(np.ones(10, np.float32), 48000, 0.0)
    m.add_block(np.ones(5, np.float32), 48000, 60.0)
    assert sum(len(b) for b in m._blocks) == 15
    m.add_block(np.zeros(3, np.float32), 44100, 0.0)  # next file: earlier audio is dropped
    assert sum(len(b) for b in m._blocks) == 3 and m._block_sr == 44100


def _win(file, model, sci, starts, w, score=0.8):
    return [dict(file=file, start_s=s, end_s=s + w, model=model, label=sci, label_key=sci.lower(),
                 scientific_name=sci, common_name="", group="biophony", score=score) for s in starts]


def test_sound_intervals_join_consecutive_windows():
    from biotool.events import build_events

    det = pd.DataFrame(_win("a.wav", "birdnet", "Turdus merula", [0, 3, 6, 15], 3.0))
    ev = build_events(det, {"birdnet": 3.0})
    got = [(r.start_s, r.end_s, r.outer_start_s, r.outer_end_s, r.n_windows) for r in ev.itertuples()]
    # centre of first -> centre of last window; a lone window is one window step long
    assert got == [(1.5, 7.5, 0.0, 9.0, 3), (15.0, 18.0, 15.0, 18.0, 1)]


def test_sound_intervals_overlap_and_block_edges():
    from biotool.events import build_events

    # 3 s windows every 0.5 s; windows restart at the 60 s block edge (57 -> 60)
    det = pd.DataFrame(_win("a.wav", "birdnet", "Turdus merula", [56.0, 56.5, 57.0, 60.0, 60.5, 64.0], 3.0))
    ev = build_events(det, {"birdnet": 0.5}, block_s=60.0)
    assert [(r.start_s, r.end_s, r.n_windows) for r in ev.itertuples()] == [(57.5, 62.0, 5), (65.25, 65.75, 1)]


def test_sound_intervals_use_birdcode_times_for_birds():
    from biotool.events import build_events

    det = pd.DataFrame(_win("a.wav", "birdnet", "Turdus merula", [0, 3], 3.0)
                       + _win("a.wav", "birdcode", "Turdus merula", [], 0)
                       + [dict(file="a.wav", start_s=2.2, end_s=2.9, model="birdcode", label="Turdus merula",
                               label_key="turdus merula", scientific_name="Turdus merula", common_name="",
                               group="biophony", score=0.7),
                          dict(file="a.wav", start_s=4.1, end_s=4.6, model="birdcode", label="Turdus merula",
                               label_key="turdus merula", scientific_name="Turdus merula", common_name="",
                               group="biophony", score=0.6)])
    ev = build_events(det, {"birdnet": 3.0})
    bn = ev[ev.model == "birdnet"].iloc[0]
    assert (bn.start_s, bn.end_s, bn.method, bn.n_calls) == (2.2, 4.6, "birdcode", 2)
    assert (ev.model == "birdcode").sum() == 2
    assert build_events(det, {"birdnet": 3.0}, use_birdcode=False).query("model == 'birdnet'").method.iloc[0] == "windows"


def test_spectrogram_blobs_find_a_tone_and_only_trim_intervals():
    from biotool.events import build_events, find_blobs, refine_with_blobs

    sr = 22050
    t = np.arange(10 * sr) / sr
    rng = np.random.default_rng(0)
    x = 0.01 * rng.standard_normal(len(t))
    on = (t >= 4.0) & (t < 5.0)
    ramp = np.clip(np.minimum(t[on] - 4.0, 5.0 - t[on]) / 0.02, 0, 1)  # 20 ms fades, no clicks
    x[on] += 0.05 * ramp * np.sin(2 * np.pi * 3000 * t[on])  # 1 s tone at 3 kHz
    assert len(find_blobs((0.01 * rng.standard_normal(len(t))).astype(np.float32), sr)) == 0  # hiss only
    blobs = find_blobs(x.astype(np.float32), sr)
    k = blobs.loc[blobs["energy"].idxmax()]
    assert abs(k.start_s - 4.0) < 0.1 and abs(k.end_s - 5.0) < 0.1
    assert k.low_hz < 3000 < k.high_hz and k.high_hz - k.low_hz < 1500

    det = pd.DataFrame(_win("a.wav", "birdnet", "Turdus merula", [3.0], 3.0))  # window 3-6 s
    ev = refine_with_blobs(build_events(det, {"birdnet": 3.0}), blobs).iloc[0]
    assert ev.method == "windows+spectrogram"
    assert 3.0 <= ev.start_s and ev.end_s <= 6.0 and abs(ev.start_s - 4.0) < 0.1 and abs(ev.end_s - 5.0) < 0.1
    assert ev.low_hz < 3000 < ev.high_hz


def test_single_run_is_saved_and_read_back(tmp_path):
    from biotool.pipeline import FileResult, RunSettings, read_file_result, read_run, save_single_run

    det = pd.DataFrame(_win("rec.wav", "birdnet", "Turdus merula", [0.0, 3.0], 3.0))
    ev = pd.DataFrame([dict(file="rec.wav", start_s=1.5, end_s=4.5, duration_s=3.0, model="birdnet",
                            label="Turdus merula", label_key="turdus merula", scientific_name="Turdus merula",
                            common_name="", group="biophony", max_score=0.8, n_windows=2, outer_start_s=0,
                            outer_end_s=6, method="windows", n_calls=0, low_hz=None, high_hz=None)])
    from biotool.models.indices import INDEX_COLUMNS

    res = FileResult(tmp_path / "rec.wav", 60.0, 48000, det, pd.DataFrame(columns=INDEX_COLUMNS), {}, 1.0,
                     [], None, "", ev)
    s = RunSettings(models={"birdnet": {"threshold": 0.4}}, events=True, grid_s=5.0)
    a = save_single_run(res, tmp_path / "runs", s)
    b = save_single_run(res, tmp_path / "runs", s)  # same second: a new folder, nothing overwritten
    assert a != b and a.parent == b.parent
    assert (a / "detections_all.csv").exists() and (a / "sound_intervals.csv").exists()
    settings, files = read_run(a)
    assert settings.models == {"birdnet": {"threshold": 0.4}} and settings.events and settings.grid_s == 5.0
    assert [f["stem"] for f in files] == ["rec"] and files[0]["samplerate"] == 48000
    r = read_file_result(a, files[0])
    assert len(r.detections) == 2 and r.detections["score"].dtype.kind == "f"
    assert len(r.events) == 1 and r.events.iloc[0].end_s == 4.5 and r.duration_s == 60.0
