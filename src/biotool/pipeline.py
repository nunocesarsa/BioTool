"""Run selected models over one file or a whole folder and write merged outputs."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__, merge
from .audio import HighPass, find_audio_files, info, iter_blocks, resample
from .location import LocationFilter, LocationSettings
from .species_list import SpeciesList, SpeciesListSettings, file_hash
from .models import DETECTORS, AcousticIndices
from .events import BLOB_COLUMNS, build_events, empty_events, find_blobs, refine_with_blobs
from .models.base import DETECTION_COLUMNS, ModelOutput
from .models.indices import INDEX_COLUMNS

Progress = Callable[[str], None]


@dataclass
class RunSettings:
    models: dict[str, dict] = field(default_factory=lambda: {"birdnet": {}, "indices": {}})
    # e.g. {"birdnet": {"threshold": 0.3}, "perch_v2": {"threshold": 0.5}, "yamnet": {}, "indices": {}}
    save_embeddings: bool = True
    save_raven: bool = True
    grid_s: float = 3.0
    block_s: float = 60.0
    threads: int = 0  # 0 = let the libraries decide
    location: dict = field(default_factory=dict)  # LocationSettings as dict (empty = off)
    highpass_hz: float = 0.0  # >0: high-pass filter the audio the models see (indices use raw audio)
    species_list: dict = field(default_factory=dict)  # SpeciesListSettings as dict (empty = off)
    quiet_dbfs: float | None = None  # skip BirdNET/Perch windows quieter than this (RMS dBFS); None = off
    events: bool = False  # also write sound intervals (consecutive windows joined; BirdCODE times for birds)
    events_spectrogram: bool = False  # + frequency band and trimmed edges from spectrogram intensity blobs

    def fingerprint(self) -> str:
        loc = self.location if self.location.get("enabled") else {}
        hp = {"hp": self.highpass_hz} if self.highpass_hz else {}
        extra = {}
        if self.species_list.get("enabled"):
            extra["sl"] = {**self.species_list, "hash": file_hash(self.species_list.get("path", ""))}
        if self.quiet_dbfs is not None:
            extra["quiet"] = self.quiet_dbfs
        if self.events:
            extra["events"] = 2 if self.events_spectrogram else 1
        d = json.dumps({"models": self.models, "grid": self.grid_s, "block": self.block_s,
                        "emb": self.save_embeddings, "loc": loc, "v": __version__, **hp, **extra},
                       sort_keys=True)
        return hashlib.sha1(d.encode()).hexdigest()[:12]


@dataclass
class FileResult:
    path: Path
    duration_s: float
    samplerate: int
    detections: pd.DataFrame
    indices: pd.DataFrame
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]]  # model -> (times, vectors)
    seconds: float
    warnings: list[str] = field(default_factory=list)
    removed: pd.DataFrame | None = None  # detections dropped by the location filter / species list
    location_note: str = ""
    events: pd.DataFrame | None = None  # sound intervals (RunSettings.events)


_COMMON: dict[str, str] | None = None


def fill_common_names(det: pd.DataFrame) -> pd.DataFrame:
    """Models like Perch only give scientific names; borrow English names from BirdNET's list."""
    global _COMMON
    if det.empty or not (det["common_name"] == "").any():
        return det
    if _COMMON is None:
        _COMMON = {}
        from .models.base import split_birdnet_label
        from .paths import models_dir

        f = models_dir() / "birdnet_v2.4" / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                sci, com = split_birdnet_label(line)
                if sci != com:
                    _COMMON[sci.lower()] = com
    if not _COMMON:
        return det
    miss = det["common_name"] == ""
    det.loc[miss, "common_name"] = det.loc[miss, "label_key"].map(_COMMON).fillna("")
    return det


class Engine:
    """Holds loaded models so they are loaded once and reused across files."""

    def __init__(self, settings: RunSettings, progress: Progress | None = None):
        self.settings = settings
        self.progress = progress or (lambda m: None)
        self.detectors = {}
        self.indices = None
        loc = LocationSettings.from_dict(settings.location)
        self.location = LocationFilter(loc, settings.threads) if loc.enabled else None
        sl = SpeciesListSettings.from_dict(settings.species_list)
        self.species_list = SpeciesList(sl) if sl.enabled and sl.path else None
        if settings.threads and any(k in settings.models for k in ("perch_v2", "yamnet")):
            try:
                import tensorflow as tf
                tf.config.threading.set_intra_op_parallelism_threads(settings.threads)
                tf.config.threading.set_inter_op_parallelism_threads(max(1, settings.threads // 2))
            except Exception:
                pass
        for key, opts in settings.models.items():
            if key == AcousticIndices.key:
                self.indices = AcousticIndices(**opts)
            elif key in DETECTORS:
                kw = {"embeddings": settings.save_embeddings, "quiet_dbfs": settings.quiet_dbfs, **opts}
                self.detectors[key] = DETECTORS[key](threads=settings.threads, **kw)
            else:
                raise ValueError(f"Unknown model '{key}'. Available: {list(DETECTORS) + ['indices']}")

    def load(self) -> None:
        for m in list(self.detectors.values()) + ([self.indices] if self.indices else []):
            self.progress(f"Loading {m.name} ...")
            m.ensure_loaded(progress=self.progress)
        if self.location:
            s = self.location.s
            self.progress(f"Loading location filter ({s.lat:.4f}, {s.lon:.4f}) ...")
            self.location.load(self.progress)

    def analyze_file(self, path: str | Path, label: str | None = None,
                     should_stop: Callable[[], bool] | None = None) -> FileResult:
        path = Path(path)
        label = label or path.name
        t0 = time.time()
        inf = info(path)
        warn = []
        for m in self.detectors.values():
            if inf.samplerate / 2 < m.fmax_hz:  # only warn when the model loses part of its band
                warn.append(f"{m.name} uses sounds up to {m.fmax_hz} Hz but this file only contains "
                            f"up to {inf.samplerate // 2} Hz (sample rate {inf.samplerate} Hz)")
        outs = {k: ModelOutput() for k in self.detectors}
        idx_rows = []
        blob_parts = []
        want_blobs = self.settings.events and self.settings.events_spectrogram and bool(self.detectors)
        hp = None
        if self.settings.highpass_hz and self.settings.highpass_hz < inf.samplerate / 2:
            hp = HighPass(self.settings.highpass_hz, inf.samplerate)
        for block, sr, offset in iter_blocks(path, self.settings.block_s):
            if should_stop and should_stop():
                raise InterruptedError("stopped")
            mblock = hp(block) if hp else block
            if want_blobs:
                blob_parts.append(find_blobs(block, sr, offset))
            for key, m in self.detectors.items():
                if getattr(m, "whole_file", False):  # e.g. BirdCODE: analyses the whole recording at once
                    m.add_block(mblock, sr, offset)
                elif getattr(m, "native_sr", False):  # model resamples itself (e.g. DeepForestSound)
                    m.input_sr = sr
                    outs[key].extend(m.process(mblock, offset, label))
                else:
                    outs[key].extend(m.process(resample(mblock, sr, m.samplerate), offset, label))
            if self.indices:
                row = self.indices.process(block, sr, offset, label)
                if row:
                    idx_rows.append(row)
        for key, m in self.detectors.items():
            if getattr(m, "whole_file", False):
                self.progress(f"{m.name}: analysing the whole recording ...")
                outs[key].detections = m.finish_file(label)
        dets = [d for o in outs.values() for d in o.detections]
        det_df = pd.DataFrame(dets, columns=DETECTION_COLUMNS) if dets else merge.empty_detections()
        det_df = fill_common_names(det_df)
        removed_parts, note = [], ""
        if self.location:
            det_df, rem, note = self.location.apply(det_df, path)
            if len(rem):
                names = ", ".join(sorted(set(rem["scientific_name"]))[:8])
                warn.append(f"location filter ({note}): removed {len(rem)} detections of "
                            f"{rem['label_key'].nunique()} out-of-range species ({names}"
                            f"{', ...' if rem['label_key'].nunique() > 8 else ''})")
                removed_parts.append(rem.assign(removed_by="location filter"))
        if self.species_list:
            det_df, rem = self.species_list.apply(det_df)
            if len(rem):
                verb = "not on" if self.species_list.s.mode == "include" else "on"
                warn.append(f"species list: removed {len(rem)} detections of {rem['label_key'].nunique()} "
                            f"species {verb} the list")
                removed_parts.append(rem.assign(removed_by=f"species list ({self.species_list.s.mode})"))
        removed = pd.concat(removed_parts, ignore_index=True) if removed_parts else None
        if self.location is None and self.species_list is None:
            removed = None
        elif removed is None:
            removed = pd.DataFrame(columns=DETECTION_COLUMNS + ["removed_by"])
        det_df = det_df.sort_values(["start_s", "model", "score"], ascending=[True, True, False]).reset_index(drop=True)
        emb = {k: (np.array(o.emb_times, dtype=np.float32), np.stack(o.embeddings))
               for k, o in outs.items() if o.embeddings}
        events = None
        if self.settings.events:
            hops = {k: m.hop_s() for k, m in self.detectors.items() if m.hop_s()}
            events = build_events(det_df, hops, self.settings.block_s)
            if want_blobs:
                blobs = pd.concat([b for b in blob_parts if len(b)] or [pd.DataFrame(columns=BLOB_COLUMNS)],
                                  ignore_index=True)
                events = refine_with_blobs(events, blobs)
        return FileResult(path, inf.duration_s, inf.samplerate, det_df,
                          pd.DataFrame(idx_rows, columns=INDEX_COLUMNS), emb, time.time() - t0, warn,
                          removed, note, events)


# ---------------------------------------------------------------------------- outputs

def _safe_stem(rel: Path) -> str:
    return "__".join(rel.with_suffix("").parts)


def write_file_outputs(res: FileResult, out_dir: Path, stem: str, settings: RunSettings) -> None:
    pf = out_dir / "per_file"
    pf.mkdir(parents=True, exist_ok=True)
    res.detections.to_csv(pf / f"{stem}.detections.csv", index=False)
    res.indices.to_csv(pf / f"{stem}.indices.csv", index=False)
    if res.removed is not None:
        res.removed.to_csv(pf / f"{stem}.removed.csv", index=False)
    if res.events is not None:
        res.events.to_csv(pf / f"{stem}.intervals.csv", index=False)
    if settings.save_embeddings and res.embeddings:
        ed = out_dir / "embeddings"
        ed.mkdir(exist_ok=True)
        for model, (times, vecs) in res.embeddings.items():
            np.savez_compressed(ed / f"{stem}__{model}.npz", start_s=times, embeddings=vecs,
                                source_file=str(res.path))
    if settings.save_raven:
        rd = out_dir / "raven"
        rd.mkdir(exist_ok=True)
        merge.raven_table(res.detections, res.samplerate).to_csv(
            rd / f"{stem}.selections.txt", sep="\t", index=False)
        if res.events is not None:
            merge.raven_table(res.events.rename(columns={"max_score": "score"}), res.samplerate).to_csv(
                rd / f"{stem}.intervals.selections.txt", sep="\t", index=False)
    (pf / f"{stem}.done.json").write_text(json.dumps({
        "file": str(res.path), "fingerprint": settings.fingerprint(), "duration_s": res.duration_s,
        "samplerate": res.samplerate, "seconds": round(res.seconds, 2), "warnings": res.warnings,
        "location_note": res.location_note,
        "finished": datetime.now().isoformat(timespec="seconds")}))


def combine_outputs(out_dir: Path, grid_s: float = 3.0) -> dict[str, Path]:
    """Merge all per-file tables into run-level tables."""
    out_dir = Path(out_dir)
    pf = out_dir / "per_file"
    dets = [pd.read_csv(p, keep_default_na=False) for p in sorted(pf.glob("*.detections.csv"))]
    dets = [d for d in dets if len(d)]
    det = pd.concat(dets, ignore_index=True) if dets else merge.empty_detections()
    for c in ("start_s", "end_s", "score"):
        det[c] = pd.to_numeric(det[c])
    idx = [pd.read_csv(p) for p in sorted(pf.glob("*.indices.csv"))]
    idx = [d for d in idx if len(d)]
    paths = {
        "detections": out_dir / "detections_all.csv",
        "agreement": out_dir / "species_agreement.csv",
        "groups": out_dir / "groups_per_window.csv",
        "summary": out_dir / "species_summary.csv",
        "indices": out_dir / "indices_all.csv",
    }
    rem = [pd.read_csv(p, keep_default_na=False) for p in sorted(pf.glob("*.removed.csv"))]
    rem = [d for d in rem if len(d)]
    evs = [pd.read_csv(p, keep_default_na=False) for p in sorted(pf.glob("*.intervals.csv"))]
    if evs:
        paths["events"] = out_dir / "sound_intervals.csv"
        ev = pd.concat([d for d in evs if len(d)] or [empty_events()], ignore_index=True)
        ev.to_csv(paths["events"], index=False)
    if rem:
        paths["removed"] = out_dir / "removed_detections.csv"
        pd.concat(rem, ignore_index=True).to_csv(paths["removed"], index=False)
    det.to_csv(paths["detections"], index=False)
    merge.species_agreement(det, grid_s).to_csv(paths["agreement"], index=False)
    merge.groups_per_window(det, grid_s).to_csv(paths["groups"], index=False)
    merge.species_summary(det).to_csv(paths["summary"], index=False)
    (pd.concat(idx, ignore_index=True) if idx else pd.DataFrame(columns=INDEX_COLUMNS)).to_csv(
        paths["indices"], index=False)
    return paths


# ----------------------------------------------------------------------------- saved runs
def settings_from_dict(d: dict) -> RunSettings:
    """RunSettings from run_config.json (unknown keys from other versions are ignored)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(RunSettings)}
    return RunSettings(**{k: v for k, v in (d or {}).items() if k in names})


def save_single_run(res: FileResult, root: str | Path, settings: RunSettings) -> Path:
    """Save one file's results as a run folder <root>/<date_time>_<file name> (never overwrites)."""
    root = Path(root)
    stem = Path(res.path).stem
    base = root / f"{datetime.now():%Y-%m-%d_%H%M%S}_{stem}"
    out, n = base, 1
    while out.exists():
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True)
    write_file_outputs(res, out, stem, settings)
    combine_outputs(out, settings.grid_s)
    (out / "run_config.json").write_text(json.dumps({
        "kind": "single", "input": str(res.path), "settings": settings.__dict__,
        "fingerprint": settings.fingerprint(), "n_files": 1, "biotool_version": __version__,
        "started": datetime.now().isoformat(timespec="seconds")}, indent=2, default=str))
    return out


def read_run(run_dir: str | Path) -> tuple[RunSettings | None, list[dict]]:
    """Settings and finished files of a saved run (single file or batch).

    Each file: {"stem", "file" (audio path when the run was made), "duration_s", "samplerate"}."""
    run_dir = Path(run_dir)
    settings = None
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        try:
            settings = settings_from_dict(json.loads(cfg.read_text()).get("settings"))
        except Exception:
            settings = None
    files = []
    for m in sorted((run_dir / "per_file").glob("*.done.json")):
        try:
            d = json.loads(m.read_text())
        except Exception:
            continue
        files.append({"stem": m.name[: -len(".done.json")], "file": d.get("file", ""),
                      "duration_s": d.get("duration_s"), "samplerate": d.get("samplerate")})
    return settings, files


def read_file_result(run_dir: str | Path, entry: dict, audio_path: str | Path | None = None) -> FileResult:
    """One file's saved tables as a FileResult (embeddings are not loaded)."""
    pf = Path(run_dir) / "per_file"
    stem = entry["stem"]

    def table(name, empty):
        p = pf / f"{stem}.{name}.csv"
        if not p.exists():
            return empty
        d = pd.read_csv(p, keep_default_na=False)
        for c in ("start_s", "end_s", "score", "duration_s", "max_score", "outer_start_s", "outer_end_s",
                  "low_hz", "high_hz"):
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d

    det = table("detections", merge.empty_detections())
    idx = table("indices", pd.DataFrame(columns=INDEX_COLUMNS))
    removed = table("removed", None)
    events = table("intervals", None)
    return FileResult(Path(audio_path or entry.get("file", "")), float(entry.get("duration_s") or 0),
                      int(entry.get("samplerate") or 0), det, idx, {}, 0.0, [], removed, "", events)


def run_folder(input_dir: str | Path, output_dir: str | Path, settings: RunSettings,
               recursive: bool = True, progress: Progress | None = None,
               file_progress: Callable[[int, int, str], None] | None = None,
               should_stop: Callable[[], bool] | None = None,
               engine: Engine | None = None) -> dict:
    """Process every audio file in a folder. Already-finished files (same settings) are skipped."""
    progress = progress or print
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = find_audio_files(input_dir, recursive)
    (output_dir / "run_config.json").write_text(json.dumps({
        "input": str(input_dir), "settings": settings.__dict__, "fingerprint": settings.fingerprint(),
        "n_files": len(files), "biotool_version": __version__,
        "started": datetime.now().isoformat(timespec="seconds")}, indent=2, default=str))
    progress(f"Found {len(files)} audio files in {input_dir}")
    if not files:
        return {"processed": 0, "skipped": 0, "failed": 0}

    engine = engine or Engine(settings, progress)
    engine.load()
    fp = settings.fingerprint()
    done = skipped = failed = 0
    errors = []
    t_start = time.time()
    for i, f in enumerate(files, 1):
        if should_stop and should_stop():
            progress("Stopped by user.")
            break
        rel = f.relative_to(input_dir)
        stem = _safe_stem(rel)
        marker = output_dir / "per_file" / f"{stem}.done.json"
        if marker.exists():
            try:
                if json.loads(marker.read_text()).get("fingerprint") == fp:
                    skipped += 1
                    if file_progress:
                        file_progress(i, len(files), f"{rel} (already done)")
                    continue
            except Exception:
                pass
        if file_progress:
            file_progress(i - 1, len(files), str(rel))
        try:
            res = engine.analyze_file(f, label=str(rel), should_stop=should_stop)
            write_file_outputs(res, output_dir, stem, settings)
            done += 1
            rate = res.duration_s / max(res.seconds, 1e-6)
            progress(f"[{i}/{len(files)}] {rel}: {len(res.detections)} detections, "
                     f"{res.duration_s:.0f} s audio in {res.seconds:.1f} s ({rate:.0f}x real time)")
            for w in res.warnings:
                progress(f"    note: {w}")
        except InterruptedError:
            progress("Stopped by user.")
            break
        except Exception as e:  # keep going on bad files
            failed += 1
            errors.append({"file": str(rel), "error": repr(e), "trace": traceback.format_exc()})
            progress(f"[{i}/{len(files)}] {rel}: FAILED - {e}")
        if file_progress:
            file_progress(i, len(files), str(rel))
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
    progress("Combining results ...")
    paths = combine_outputs(output_dir, settings.grid_s)
    progress(f"Done: {done} processed, {skipped} skipped (already done), {failed} failed "
             f"in {time.time() - t_start:.0f} s. Results in {output_dir}")
    return {"processed": done, "skipped": skipped, "failed": failed, "outputs": paths}
