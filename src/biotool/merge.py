"""Combine detections from several models into comparable tables.

Different models use different window lengths (BirdNET 3 s, Perch 5 s, YAMNet frames),
so detections are projected onto a common time grid (default 3 s). Within each grid
window we then count how many models report the same label (species agreement) and
which models report each sound group (biophony / geophony / anthropophony).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models.base import DETECTION_COLUMNS


def _explode_to_grid(det: pd.DataFrame, grid_s: float) -> pd.DataFrame:
    first = np.floor(det["start_s"].to_numpy() / grid_s).astype(int)
    last = np.maximum(first, np.ceil(det["end_s"].to_numpy() / grid_s - 1e-6).astype(int) - 1)
    reps = last - first + 1
    out = det.loc[det.index.repeat(reps)].copy()
    out["window"] = np.concatenate([np.arange(a, b + 1) for a, b in zip(first, last)]) if len(det) else []
    out["win_start_s"] = out["window"] * grid_s
    out["win_end_s"] = out["win_start_s"] + grid_s
    return out


def _display_name(df: pd.DataFrame) -> pd.Series:
    name = df["scientific_name"].where(df["scientific_name"].astype(bool), df["common_name"])
    return name.fillna(df["label"])


def species_agreement(det: pd.DataFrame, grid_s: float = 3.0) -> pd.DataFrame:
    """One row per (file, grid window, label): which models detected it and the best score."""
    cols = ["file", "win_start_s", "win_end_s", "label_key", "name", "common_name", "group",
            "n_models", "models", "max_score"]
    if det.empty:
        return pd.DataFrame(columns=cols)
    g = _explode_to_grid(det, grid_s)
    g["name"] = _display_name(g)
    agg = (
        g.groupby(["file", "win_start_s", "win_end_s", "label_key"], sort=True)
        .agg(name=("name", "first"),
             common_name=("common_name", lambda s: next((v for v in s if v), "")),
             group=("group", "first"),
             models=("model", lambda s: "+".join(sorted(set(s)))),
             n_models=("model", "nunique"),
             max_score=("score", "max"))
        .reset_index()
    )
    return agg[cols].sort_values(["file", "win_start_s", "n_models", "max_score"],
                                 ascending=[True, True, False, False]).reset_index(drop=True)


def groups_per_window(det: pd.DataFrame, grid_s: float = 3.0) -> pd.DataFrame:
    """One row per (file, grid window, group) with the models that saw that group and top labels."""
    cols = ["file", "win_start_s", "win_end_s", "group", "n_models", "models", "top_labels"]
    if det.empty:
        return pd.DataFrame(columns=cols)
    g = _explode_to_grid(det, grid_s)
    g["name"] = _display_name(g)

    def top(df):
        best = df.sort_values("score", ascending=False).drop_duplicates("label_key").head(3)
        return "; ".join(f"{n} ({s:.2f})" for n, s in zip(best["name"], best["score"]))

    rows = []
    for (f, ws, we, grp), df in g.groupby(["file", "win_start_s", "win_end_s", "group"], sort=True):
        rows.append(dict(file=f, win_start_s=ws, win_end_s=we, group=grp,
                         n_models=df["model"].nunique(), models="+".join(sorted(set(df["model"]))),
                         top_labels=top(df)))
    return pd.DataFrame(rows, columns=cols)


def species_summary(det: pd.DataFrame) -> pd.DataFrame:
    """One row per (file, label): per-model number of windows and max score."""
    if det.empty:
        return pd.DataFrame(columns=["file", "label_key", "name", "common_name", "group", "n_models", "models"])
    d = det.copy()
    d["name"] = _display_name(d)
    base = (d.groupby(["file", "label_key"])
            .agg(name=("name", "first"),
                 common_name=("common_name", lambda s: next((v for v in s if v), "")),
                 group=("group", "first"),
                 n_models=("model", "nunique"),
                 models=("model", lambda s: "+".join(sorted(set(s)))),
                 first_s=("start_s", "min"),
                 max_score=("score", "max"))
            .reset_index())
    per = d.pivot_table(index=["file", "label_key"], columns="model", values="score",
                        aggfunc=["count", "max"])
    per.columns = [f"{m}_{'n' if a == 'count' else 'max'}" for a, m in per.columns]
    out = base.merge(per.reset_index(), on=["file", "label_key"], how="left")
    return out.sort_values(["file", "n_models", "max_score"], ascending=[True, False, False]).reset_index(drop=True)


def raven_table(det: pd.DataFrame, samplerate: int | None = None) -> pd.DataFrame:
    """Raven Pro selection table (tab-separated when written)."""
    high = (samplerate or 48000) / 2
    d = det.sort_values(["start_s", "model"]).reset_index(drop=True)
    name = _display_name(d) if len(d) else pd.Series(dtype=str)
    return pd.DataFrame({
        "Selection": np.arange(1, len(d) + 1),
        "View": "Spectrogram 1",
        "Channel": 1,
        "Begin Time (s)": d["start_s"],
        "End Time (s)": d["end_s"],
        "Low Freq (Hz)": 0,
        "High Freq (Hz)": high,
        "Label": name,
        "Common Name": d["common_name"],
        "Group": d["group"],
        "Model": d["model"],
        "Score": d["score"],
    })


def empty_detections() -> pd.DataFrame:
    return pd.DataFrame(columns=DETECTION_COLUMNS)
