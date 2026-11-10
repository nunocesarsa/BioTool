"""Sound intervals: turn window detections into events with a start and end time.

Window models (BirdNET, Perch, YAMNet, DeepForestSound) only say that a sound is somewhere in a
window. Consecutive windows of one model with the same label are joined into one interval:

    start_s / end_s    centre of the first detecting window -> centre of the last one
                       (at least one window step long). On the Powdermill annotations (BirdNET,
                       4 files) the start and end were off by about 1 s (median) and the centre by
                       about 0.5 s; calls shorter than a window come out roughly one window long.
    outer_start_s /    first window start -> last window end: the sound is (almost always)
    outer_end_s        inside these bounds.

When BirdCODE ran in the same run, a bird interval that overlaps BirdCODE events of the same
species takes BirdCODE's frame-level times instead (method "birdcode"). BirdCODE's own events are
included as they are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["file", "start_s", "end_s", "duration_s", "model", "label", "label_key",
                 "scientific_name", "common_name", "group", "max_score", "n_windows",
                 "outer_start_s", "outer_end_s", "method", "n_calls", "low_hz", "high_hz"]
_TOL = 1e-3


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _runs(starts: np.ndarray, ends: np.ndarray, hop: float, block_s: float) -> list[np.ndarray]:
    """Indices of consecutive windows (next start one step later, or the first window of the next
    reading block touching the previous window)."""
    order = np.argsort(starts, kind="stable")
    runs, cur = [], [order[0]]
    for i in order[1:]:
        prev = cur[-1]
        ds = starts[i] - starts[prev]
        at_block = block_s > 0 and abs(starts[i] / block_s - round(starts[i] / block_s)) * block_s < _TOL
        if ds <= hop + _TOL or (at_block and starts[i] <= ends[prev] + _TOL):
            cur.append(i)
        else:
            runs.append(np.array(cur))
            cur = [i]
    runs.append(np.array(cur))
    return runs


def window_events(det: pd.DataFrame, hops: dict[str, float], block_s: float = 60.0) -> pd.DataFrame:
    """Join consecutive windows of each (file, model, label) into intervals."""
    rows = []
    for (file, model, key), g in det[det["model"].isin(hops)].groupby(["file", "model", "label_key"], sort=False):
        hop = float(hops[model])
        g = g.drop_duplicates("start_s")
        st, en, sc = g["start_s"].to_numpy(float), g["end_s"].to_numpy(float), g["score"].to_numpy(float)
        first = g.iloc[0]
        for r in _runs(st, en, hop, block_s):
            r = r[np.argsort(st[r])]
            c0 = (st[r[0]] + en[r[0]]) / 2
            c1 = (st[r[-1]] + en[r[-1]]) / 2
            if c1 - c0 < hop:
                m = (c0 + c1) / 2
                c0, c1 = m - hop / 2, m + hop / 2
            o0, o1 = st[r].min(), en[r].max()
            c0, c1 = max(c0, o0), min(c1, o1)
            rows.append(dict(file=file, start_s=round(c0, 3), end_s=round(c1, 3), duration_s=round(c1 - c0, 3),
                             model=model, label=first["label"], label_key=key,
                             scientific_name=first["scientific_name"], common_name=first["common_name"],
                             group=first["group"], max_score=round(float(sc[r].max()), 4), n_windows=len(r),
                             outer_start_s=round(o0, 3), outer_end_s=round(o1, 3), method="windows", n_calls=0))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS) if rows else empty_events()


def birdcode_events(det: pd.DataFrame) -> pd.DataFrame:
    b = det[det["model"] == "birdcode"]
    if not len(b):
        return empty_events()
    return pd.DataFrame({
        "file": b["file"], "start_s": b["start_s"], "end_s": b["end_s"],
        "duration_s": (b["end_s"] - b["start_s"]).round(3), "model": "birdcode", "label": b["label"],
        "label_key": b["label_key"], "scientific_name": b["scientific_name"], "common_name": b["common_name"],
        "group": b["group"], "max_score": b["score"], "n_windows": 0, "outer_start_s": b["start_s"],
        "outer_end_s": b["end_s"], "method": "birdcode", "n_calls": 1,
    }, columns=EVENT_COLUMNS)


def refine_with_birdcode(ev: pd.DataFrame, bc: pd.DataFrame) -> pd.DataFrame:
    """Bird intervals of other models take the times of overlapping BirdCODE events of the same species."""
    if not len(ev) or not len(bc):
        return ev
    ev = ev.copy()
    sci_ev = ev["scientific_name"].astype(str).str.strip().str.lower()
    sci_bc = bc["scientific_name"].astype(str).str.strip().str.lower()
    for i in ev.index[(ev["model"] != "birdcode") & (sci_ev != "")]:
        m = bc[(bc["file"] == ev.at[i, "file"]) & (sci_bc == sci_ev[i])
               & (bc["end_s"] > ev.at[i, "outer_start_s"]) & (bc["start_s"] < ev.at[i, "outer_end_s"])]
        if len(m):
            s, e = float(m["start_s"].min()), float(m["end_s"].max())
            ev.loc[i, ["start_s", "end_s", "duration_s", "method", "n_calls"]] = [
                round(s, 3), round(e, 3), round(e - s, 3), "birdcode", len(m)]
    return ev


def build_events(det: pd.DataFrame, hops: dict[str, float], block_s: float = 60.0,
                 use_birdcode: bool = True) -> pd.DataFrame:
    if not len(det):
        return empty_events()
    ev = window_events(det, hops, block_s)
    bc = birdcode_events(det)
    if use_birdcode and len(bc):
        ev = refine_with_birdcode(ev, bc)
    out = pd.concat([x for x in (ev, bc) if len(x)], ignore_index=True) if (len(ev) or len(bc)) else empty_events()
    return out.sort_values(["file", "start_s", "model"]).reset_index(drop=True)


# ------------------------------------------------------------------ spectrogram intensity refinement
BLOB_COLUMNS = ["start_s", "end_s", "low_hz", "high_hz", "energy"]
MAX_MASK_FRACTION = 0.3


def find_blobs(audio: np.ndarray, sr: int, offset_s: float = 0.0, max_len_s: float = 10.0) -> pd.DataFrame:
    """Sounds as time-frequency blobs: scikit-maad background removal, double threshold and connected
    components, with the parameters of scikit-maad's own example (fallback for steady hiss below).
    'energy' = summed dB above the background."""
    from scipy import ndimage
    from maad import rois, sound, util

    x = np.asarray(audio, dtype=np.float64)
    if len(x) < 2048:
        return pd.DataFrame(columns=BLOB_COLUMNS)
    Sxx, tn, fn, _ = sound.spectrogram(x, sr, nperseg=1024, noverlap=512)
    Sdb = util.power2dB(Sxx) + 96
    Sdb, _, _ = sound.remove_background(Sdb)
    sm = sound.smooth(Sdb, std=1.2)
    mask = rois.create_mask(sm, mode_bin="relative", bin_std=2, bin_per=0.25)
    if mask.mean() > MAX_MASK_FRACTION:
        # remove_background can leave steady hiss well above zero (~13 dB for white noise) and the mask
        # then covers most of the spectrogram (Powdermill blocks: 5-11 %). Fall back to also removing
        # the remaining per-frequency median.
        Sdb = Sdb - np.median(Sdb, axis=1, keepdims=True)
        Sdb[Sdb < 0] = 0
        sm = sound.smooth(Sdb, std=1.2)
        mask = rois.create_mask(sm, mode_bin="relative", bin_std=2, bin_per=0.25)
    Sdb = sm
    lab, R = rois.select_rois(mask, min_roi=50, max_roi=None)
    if not len(R):
        return pd.DataFrame(columns=BLOB_COLUMNS)
    energy = ndimage.sum(np.clip(Sdb, 0, None), lab, R["labelID"].to_numpy())
    R = util.format_features(R, tn, fn)
    out = pd.DataFrame({"start_s": R["min_t"].to_numpy() + offset_s, "end_s": R["max_t"].to_numpy() + offset_s,
                        "low_hz": R["min_f"].to_numpy(), "high_hz": R["max_f"].to_numpy(), "energy": energy})
    return out[(out.end_s - out.start_s) <= max_len_s].reset_index(drop=True)


def refine_with_blobs(ev: pd.DataFrame, blobs: pd.DataFrame, band_margin_hz: float = 500.0) -> pd.DataFrame:
    """Add a frequency band to window intervals and trim their edges with spectrogram blobs.

    The blobs inside an interval's outer bounds are grouped by frequency around the loudest one
    (blobs overlapping its band +- band_margin_hz). Their band becomes low_hz/high_hz. Their time span
    may only shorten the interval, never lengthen it: on the Powdermill annotations, letting blobs
    set the times freely was no better than windows, trimming was slightly better.
    """
    ev = ev.copy()
    for c in ("low_hz", "high_hz"):
        if c not in ev:
            ev[c] = np.nan
    if not len(ev) or not len(blobs):
        return ev
    bc = (blobs["start_s"] + blobs["end_s"]).to_numpy() / 2
    for i in ev.index[ev["method"] == "windows"]:
        inside = (blobs["file"].to_numpy() == ev.at[i, "file"]) if "file" in blobs else np.ones(len(blobs), bool)
        inside &= (bc > ev.at[i, "outer_start_s"]) & (bc < ev.at[i, "outer_end_s"])
        if not inside.any():
            continue
        b = blobs[inside]
        k = b.iloc[int(np.argmax(b["energy"].to_numpy()))]
        g = b[(b["high_hz"] >= k["low_hz"] - band_margin_hz) & (b["low_hz"] <= k["high_hz"] + band_margin_hz)]
        s = max(float(ev.at[i, "start_s"]), float(g["start_s"].min()))
        e = min(float(ev.at[i, "end_s"]), float(g["end_s"].max()))
        ev.at[i, "low_hz"], ev.at[i, "high_hz"] = round(float(g["low_hz"].min())), round(float(g["high_hz"].max()))
        if e > s:
            ev.loc[i, ["start_s", "end_s", "duration_s", "method"]] = [round(s, 3), round(e, 3), round(e - s, 3),
                                                                       "windows+spectrogram"]
    return ev
