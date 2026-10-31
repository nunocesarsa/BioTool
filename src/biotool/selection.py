"""'Analyse selection': run the loaded models on one time span (optionally one frequency band)
and summarise what each model thinks is there. Nothing is written to disk.

The span is analysed with finely overlapping windows (default: a new window every 0.5 s) so a
call is never cut by a window edge. Spans shorter than a model's window are widened around
their centre to the model's window length (models only work on the length they were trained on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .audio import info, resample
from .pipeline import Engine, fill_common_names

COLUMNS = ["model", "name", "common_name", "group", "max_score", "above_min_score", "windows_found",
           "best_at_s", "note"]


def _read(path, a, b):
    from .gui.player import read_segment  # same reader as playback
    return read_segment(path, a, b)


def analyse_selection(engine: Engine, path: str, start_s: float, end_s: float,
                      band_hz: tuple[float, float] | None = None, hop_s: float = 0.5,
                      top_k: int = 8) -> tuple[pd.DataFrame, str]:
    inf = info(path)
    rows, notes = [], []
    for key, m in engine.detectors.items():
        win = m.window_s
        span = max(end_s - start_s, win)
        a = max(0.0, (start_s + end_s) / 2 - span / 2)
        b = min(inf.duration_s, a + span)
        a = max(0.0, b - span)
        x, sr = _read(path, a, b)
        if band_hz:
            from .gui.player import band_filter
            x = band_filter(x, sr, *band_hz)
        x = resample(x, sr, m.samplerate)
        if span > end_s - start_s + 1e-6:
            notes.append(f"{m.name}: widened to {a:.1f}-{b:.1f} s (its {win:g} s window)")

        # temporary settings: low threshold, fine hop, no quiet-skipping
        saved = (m.threshold, m.max_per_window, m.quiet_dbfs, dict(m.options))
        try:
            m.threshold, m.max_per_window, m.quiet_dbfs = 0.005, max(20, top_k * 3), None
            if key == "yamnet":
                m.options["group_s"] = max(0.96, b - a)
            else:
                m.options["overlap_s"] = max(0.0, win - hop_s)
            out = m.process(x, a, "selection")
        finally:
            m.threshold, m.max_per_window, m.quiet_dbfs, m.options = saved
        det = pd.DataFrame(out.detections)
        if det.empty:
            continue
        det = fill_common_names(det)
        n_windows = max(1, det["start_s"].nunique())
        min_score = saved[0]
        if engine.location is not None:
            _, removed, _ = engine.location.apply(det, path)
            out_of_range = set(removed["label_key"])
        else:
            out_of_range = set()
        if engine.species_list is not None:
            _, rem2 = engine.species_list.apply(det)
            off_list = set(rem2["label_key"])
        else:
            off_list = set()
        for lk, d in det.groupby("label_key"):
            best = d.loc[d["score"].idxmax()]
            name = best["scientific_name"] or best["common_name"] or best["label"]
            note = ", ".join(n for n, flag in (("out of range (location filter)", lk in out_of_range),
                                              ("filtered by species list", lk in off_list)) if flag)
            rows.append(dict(model=key, name=name, common_name=best["common_name"], group=best["group"],
                             max_score=round(float(d["score"].max()), 4),
                             above_min_score="yes" if d["score"].max() >= min_score else "no",
                             windows_found=f"{int((d['score'] >= min_score).sum())}/{n_windows}",
                             best_at_s=round(float(best["start_s"]), 2), note=note))
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df = (df.sort_values(["model", "max_score"], ascending=[True, False])
                .groupby("model", sort=False).head(top_k).reset_index(drop=True))
    return df, "; ".join(notes)
