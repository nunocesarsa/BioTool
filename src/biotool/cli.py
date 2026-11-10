"""Command line: biotool run INPUT OUTPUT --models birdnet,perch_v2,yamnet,indices"""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import ALL_KEYS
from .location import LocationSettings
from .pipeline import RunSettings, run_folder


def location_from_args(lat, lon, date_arg) -> dict:
    if lat is None or lon is None:
        return {}
    mode = date_arg if date_arg in ("file", "year") else "fixed"
    return LocationSettings(enabled=True, lat=lat, lon=lon, date_mode=mode,
                            fixed_date=date_arg if mode == "fixed" else "").to_dict()


def _add_evaluate(sub):
    e = sub.add_parser("evaluate", help="score models against annotated test data")
    e.add_argument("--powdermill", help="folder with Powdermill WAVs (e.g. test_data/Powdermill/sample_20min)")
    e.add_argument("--powdermill-annotations", help="Powdermill annotations folder (searched recursively)")
    e.add_argument("--powdermill-readme", help="Powdermill README.txt (species code table)")
    e.add_argument("--esc50", help="ESC-50 sample folder containing audio/ and labels.csv")
    e.add_argument("--output", required=True, help="folder for the evaluation report")
    e.add_argument("--models", default="birdnet,perch_v2,yamnet")
    e.add_argument("--threads", type=int, default=0)
    e.add_argument("--highpass", default="0",
                   help="comma-separated high-pass cut-offs (Hz) to compare, e.g. 0,300,500,1000")


def main(argv=None):
    p = argparse.ArgumentParser(prog="biotool", description="Multi-model bioacoustics processing")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a file or a folder")
    r.add_argument("input", help="audio file or folder")
    r.add_argument("output", help="output folder")
    r.add_argument("--models", default="birdnet,indices", help=f"comma list from {ALL_KEYS}")
    r.add_argument("--threshold", action="append", default=[], metavar="MODEL=VALUE",
                   help="per-model score threshold, e.g. --threshold birdnet=0.3")
    r.add_argument("--no-recursive", action="store_true")
    r.add_argument("--no-embeddings", action="store_true")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--species-list", help="text file with species names (one per line)")
    r.add_argument("--species-mode", choices=["include", "exclude"], default="include",
                   help="include = keep only listed species (default), exclude = remove them")
    r.add_argument("--grid", type=float, default=3.0, help="time grid (s) for the model-agreement tables")
    r.add_argument("--intervals", action="store_true",
                   help="also write sound intervals (start/end per sound; sound_intervals.csv)")
    r.add_argument("--intervals-spectrogram", action="store_true",
                   help="with --intervals: add each sound's frequency band and trim edges using spectrogram intensity")
    r.add_argument("--skip-quiet", type=float, default=None, metavar="DBFS",
                   help="skip BirdNET/Perch windows quieter than this RMS level, e.g. -70")
    r.add_argument("--highpass", type=float, default=0.0,
                   help="high-pass filter (Hz) applied to the audio the models see, e.g. 300; 0 = off")
    r.add_argument("--lat", type=float, help="site latitude: enables the location/season species filter")
    r.add_argument("--lon", type=float, help="site longitude")
    r.add_argument("--date", default="file",
                   help="for the location filter: 'file' (from file name/AudioMoth metadata, default), "
                        "'year' (all-year list) or a fixed date YYYY-MM-DD")
    _add_evaluate(sub)
    g = sub.add_parser("get-test-data", help="download the test data (BirdNET example, ESC-50 sample, Powdermill)")
    g.add_argument("--dest", default="test_data")
    a = p.parse_args(argv)
    if a.cmd == "get-test-data":
        from .testdata import main as testdata_main
        return testdata_main(a.dest)
    if a.cmd == "evaluate":
        from .evaluate import main as evaluate_main
        return evaluate_main(a)

    models = {m.strip(): {} for m in a.models.split(",") if m.strip()}
    for t in a.threshold:
        k, v = t.split("=")
        models.setdefault(k, {})["threshold"] = float(v)
    s = RunSettings(models=models, save_embeddings=not a.no_embeddings, threads=a.threads, events=a.intervals or a.intervals_spectrogram,
                    events_spectrogram=a.intervals_spectrogram,
                    location=location_from_args(a.lat, a.lon, a.date), highpass_hz=a.highpass,
                    grid_s=a.grid, quiet_dbfs=a.skip_quiet,
                    species_list={"enabled": True, "mode": a.species_mode, "path": a.species_list}
                    if a.species_list else {})
    inp = Path(a.input)
    if inp.is_file():  # single file: run on its folder but only that file
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / inp.name
            try:
                link.symlink_to(inp.resolve())
            except OSError:
                shutil.copy2(inp, link)
            run_folder(tmp, a.output, s)
    else:
        run_folder(inp, a.output, s, recursive=not a.no_recursive)


if __name__ == "__main__":
    main()
