"""Evaluate the models against annotated test data.

Powdermill (Chronister et al. 2021): expert-annotated dawn-chorus soundscapes.
  * window level   - for each model window, are the annotated species detected?
                     precision / recall / F1 over (window, species) pairs, swept over thresholds
  * combinations   - BirdNET OR / AND Perch on a common 3 s grid
  * file level     - per-file species list vs the annotated species list
  * location filter- every metric with and without the BirdNET range filter

ESC-50 sample: 5 s clips of 50 sound classes in 5 categories
  * sound-group accuracy (biophony / geophony / anthropophony) per model and per category

The models are run once at low thresholds (results cached in <output>/runs), so any threshold
can be evaluated afterwards without re-running.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import info
from .location import LocationFilter, LocationSettings
from .pipeline import RunSettings, run_folder

POWDERMILL_LATLON = (40.16, -79.27)  # Powdermill Nature Reserve, Rector, PA
# Older names in the Powdermill table -> names used by BirdNET / Perch (Clements/eBird)
SYNONYMS = {
    "picoides pubescens": "dryobates pubescens",
    "leuconotopicus villosus": "dryobates villosus",
    "picoides villosus": "dryobates villosus",
    "regulus calendula": "corthylio calendula",
}
LOW_THRESHOLDS = {"birdnet": 0.05, "perch_v2": 0.005, "yamnet": 0.05}
SWEEP = {
    "birdnet": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "perch_v2": [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5],
    "yamnet": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
}
DEFAULTS = {"birdnet": 0.25, "perch_v2": 0.03, "yamnet": 0.2}
NATIVE_WINDOW = {"birdnet": 3.0, "perch_v2": 5.0}
SPECIES_MODELS = ("birdnet", "perch_v2")


def norm(name: str) -> str:
    n = " ".join(str(name).lower().split())
    return SYNONYMS.get(n, n)


# ----------------------------------------------------------------------------- Powdermill data
def powdermill_codes(readme: str | Path) -> dict[str, str]:
    txt = Path(readme).read_text(encoding="utf-8", errors="replace")
    sec = txt.split("---Alpha Codes---")[-1]
    out = {}
    for line in sec.splitlines():
        m = re.match(r"\s*(.+?)\s*\t+\s*([A-Z]{4})\s*\t+\s*([A-Z][a-z]+ [a-z]+)", line)
        if m:
            out[m.group(2)] = norm(m.group(3))
    return out


def powdermill_dates(readme: str | Path) -> dict[str, str]:
    """{'Recording_2': '2018-05-08', ...} from the README methods section."""
    txt = Path(readme).read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r'"(Recording_\d+)" was collected on (\w+ \d+, \d{4})', txt):
        out[m.group(1)] = datetime.strptime(m.group(2), "%B %d, %Y").strftime("%Y-%m-%d")
    return out


def powdermill_annotations(folder: str | Path, codes: dict[str, str]) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(folder).rglob("*.selections.txt")):
        stem = f.name.split(".Table")[0]
        df = pd.read_csv(f, sep="\t")  # pandas handles the old-Mac \r line endings
        if df.empty:
            rows.append(pd.DataFrame(columns=["stem", "start_s", "end_s", "code", "species"]))
            continue
        rows.append(pd.DataFrame({
            "stem": stem, "start_s": df["Begin Time (s)"].astype(float), "end_s": df["End Time (s)"].astype(float),
            "code": df["Species"].astype(str).str.strip(),
        }))
    ann = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ann["species"] = ann["code"].map(codes)
    return ann


# ----------------------------------------------------------------------------- metrics
def _windows(duration: float, win: float) -> np.ndarray:
    return np.arange(0, duration - 0.25 * win + 1e-9, win)


def _gt_pairs(ann: pd.DataFrame, stem: str, starts: np.ndarray, win: float) -> set:
    """(window_start, species) pairs where an annotation overlaps the window enough."""
    a = ann[(ann["stem"] == stem) & ann["species"].notna()]
    pairs = set()
    for s, e, sp in zip(a["start_s"], a["end_s"], a["species"]):
        need = min(0.2, 0.5 * (e - s))
        for w in starts[(starts < e) & (starts + win > s)]:
            if min(e, w + win) - max(s, w) >= need:
                pairs.add((round(float(w), 3), sp))
    return pairs


def _pred_pairs(det: pd.DataFrame, stem: str, model: str, win: float, grid: bool) -> pd.DataFrame:
    """Max score per (window, species). grid=True maps detections onto a `win` grid."""
    d = det[(det["stem"] == stem) & (det["model"] == model)]
    if d.empty:
        return pd.DataFrame(columns=["w", "species", "score"])
    if not grid:
        out = pd.DataFrame({"w": d["start_s"].round(3), "species": d["species"], "score": d["score"]})
    else:
        first = np.floor(d["start_s"].to_numpy() / win + 1e-9).astype(int)
        last = np.maximum(first, np.ceil(d["end_s"].to_numpy() / win - 1e-6).astype(int) - 1)
        reps = last - first + 1
        out = pd.DataFrame({
            "w": np.concatenate([np.arange(a, b + 1) for a, b in zip(first, last)]) * win,
            "species": np.repeat(d["species"].to_numpy(), reps),
            "score": np.repeat(d["score"].to_numpy(), reps),
        })
        out["w"] = out["w"].round(3)
    return out.groupby(["w", "species"], as_index=False)["score"].max()


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p + r) and not (np.isnan(p) or np.isnan(r)) else 0.0
    return p, r, f


def sweep_model(det, ann, durations, model, variant) -> pd.DataFrame:
    win = NATIVE_WINDOW[model]
    rows, preds, gts = [], [], []
    for stem, dur in durations.items():
        starts = _windows(dur, win)
        gt = _gt_pairs(ann, stem, starts, win)
        pp = _pred_pairs(det, stem, model, win, grid=False)
        pp["hit"] = [(w, s) in gt for w, s in zip(pp["w"], pp["species"])]
        preds.append(pp)
        gts.append(len(gt))
    pp = pd.concat(preds, ignore_index=True)
    n_gt = sum(gts)
    for t in SWEEP[model]:
        sel = pp[pp["score"] >= t]
        tp = int(sel["hit"].sum())
        fp = len(sel) - tp
        p, r, f = _prf(tp, fp, n_gt - tp)
        rows.append(dict(model=model, variant=variant, threshold=t, tp=tp, fp=fp, fn=n_gt - tp,
                         precision=p, recall=r, f1=f))
    return pd.DataFrame(rows)


def combos(det, ann, durations, thr: dict, variant: str) -> pd.DataFrame:
    """BirdNET / Perch alone, OR, AND - all on a 3 s grid, at the given thresholds."""
    win = 3.0
    tot = {k: [0, 0, 0] for k in ["birdnet", "perch_v2", "birdnet OR perch_v2", "birdnet AND perch_v2"]}
    for stem, dur in durations.items():
        gt = _gt_pairs(ann, stem, _windows(dur, win), win)
        sets = {}
        for m in SPECIES_MODELS:
            pp = _pred_pairs(det, stem, m, win, grid=True)
            pp = pp[pp["score"] >= thr[m]]
            sets[m] = set(zip(pp["w"], pp["species"]))
        sets["birdnet OR perch_v2"] = sets["birdnet"] | sets["perch_v2"]
        sets["birdnet AND perch_v2"] = sets["birdnet"] & sets["perch_v2"]
        for k, s in sets.items():
            tp = len(s & gt)
            tot[k][0] += tp
            tot[k][1] += len(s) - tp
            tot[k][2] += len(gt) - tp
    rows = []
    for k, (tp, fp, fn) in tot.items():
        p, r, f = _prf(tp, fp, fn)
        rows.append(dict(method=k, variant=variant, tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f))
    return pd.DataFrame(rows)


def file_level(det, ann, durations, thr: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, detail = [], []
    methods = {"birdnet": ["birdnet"], "perch_v2": ["perch_v2"], "birdnet OR perch_v2": list(SPECIES_MODELS)}
    for name, ms in methods.items():
        tp = fp = fn = 0
        for stem in durations:
            truth = set(ann.loc[(ann["stem"] == stem) & ann["species"].notna(), "species"])
            d = det[(det["stem"] == stem) & det["model"].isin(ms)]
            d = d[d["score"] >= d["model"].map(thr)]
            found = set(d["species"])
            tp += len(found & truth); fp += len(found - truth); fn += len(truth - found)
            detail.append(dict(variant=variant, method=name, file=stem,
                               correct=", ".join(sorted(found & truth)),
                               missed=", ".join(sorted(truth - found)),
                               not_annotated=", ".join(sorted(found - truth))))
        p, r, f = _prf(tp, fp, fn)
        rows.append(dict(method=name, variant=variant, species_correct=tp, false_species=fp,
                         missed_species=fn, precision=p, recall=r, f1=f))
    return pd.DataFrame(rows), pd.DataFrame(detail)


# ----------------------------------------------------------------------------- runners
def _run(audio_dir: Path, out: Path, models: list[str], threads: int, progress=print,
         highpass: float = 0.0) -> pd.DataFrame:
    s = RunSettings(models={m: {"threshold": LOW_THRESHOLDS[m], "max_per_window": 30} for m in models},
                    save_embeddings=False, save_raven=False, threads=threads, highpass_hz=highpass)
    if highpass:
        out = out.with_name(f"{out.name}_hp{int(highpass)}")
        progress(f"--- high-pass {int(highpass)} Hz ---")
    run_folder(audio_dir, out, s, recursive=True, progress=progress)
    det = pd.read_csv(out / "detections_all.csv", keep_default_na=False)
    for c in ("start_s", "end_s", "score"):
        det[c] = pd.to_numeric(det[c])
    det["stem"] = det["file"].map(lambda f: Path(f).stem)
    det["species"] = det["label_key"].map(norm)
    return det


def evaluate_powdermill(audio_dir, ann_dir, readme, out: Path, models, threads=0, progress=print,
                        highpass: float = 0.0):
    codes = powdermill_codes(readme)
    dates = powdermill_dates(readme)
    ann = powdermill_annotations(ann_dir, codes)
    files = sorted(p for p in Path(audio_dir).rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3"))
    durations = {p.stem: info(p).duration_s for p in files}
    ann = ann[ann["stem"].isin(durations)]
    progress(f"Powdermill: {len(files)} files, {len(ann)} annotations, "
             f"{ann['species'].nunique()} species")
    det = _run(Path(audio_dir), out / "runs" / "powdermill", models, threads, progress, highpass)
    det = det[det["model"].isin(SPECIES_MODELS)]

    # location filter variant
    lf = LocationFilter(LocationSettings(enabled=True, lat=POWDERMILL_LATLON[0], lon=POWDERMILL_LATLON[1],
                                         date_mode="fixed"), threads)
    lf.load(progress)
    keep = []
    for stem, d in det.groupby("stem"):
        rec = "_".join(stem.split("_")[:2])
        lf.s.fixed_date = dates.get(rec, "")
        lf.s.date_mode = "fixed" if lf.s.fixed_date else "year"
        k, _, _ = lf.apply(d.drop(columns=["stem", "species"]).assign(_i=d.index), stem)
        keep.extend(k["_i"].tolist())
    det_loc = det.loc[keep]

    sw, cb, fl, fd = [], [], [], []
    best = {}
    for variant, d in (("no filter", det), ("location filter", det_loc)):
        for m in [m for m in SPECIES_MODELS if m in models]:
            s = sweep_model(d, ann, durations, m, variant)
            sw.append(s)
            best[(m, variant)] = float(s.loc[s["f1"].idxmax(), "threshold"])
        if all(m in models for m in SPECIES_MODELS):
            for label, thr in (("default thresholds", DEFAULTS),
                               ("best-F1 thresholds", {m: best[(m, variant)] for m in SPECIES_MODELS})):
                c = combos(d, ann, durations, thr, variant)
                c.insert(1, "thresholds", label)
                cb.append(c)
        thr = {m: DEFAULTS[m] for m in SPECIES_MODELS}
        f, dd = file_level(d, ann, durations, thr, variant)
        fl.append(f)
        fd.append(dd)
    return dict(sweep=pd.concat(sw, ignore_index=True),
                combos=pd.concat(cb, ignore_index=True) if cb else pd.DataFrame(),
                files=pd.concat(fl, ignore_index=True), file_detail=pd.concat(fd, ignore_index=True),
                best=best, n_files=len(files), n_ann=len(ann), n_species=ann["species"].nunique())


def evaluate_esc50(folder, out: Path, models, threads=0, progress=print, highpass: float = 0.0):
    folder = Path(folder)
    labels = pd.read_csv(folder / "labels.csv")
    det = _run(folder / "audio", out / "runs" / "esc50", models, threads, progress, highpass)
    det["filename"] = det["file"].map(lambda f: Path(f).name)
    rows, per_clip = [], []
    for m in models:
        t = DEFAULTS[m]
        d = det[(det["model"] == m) & (det["score"] >= t) & (det["group"] != "other")]
        for _, r in labels.iterrows():
            x = d[d["filename"] == r["filename"]]
            if x.empty:
                pred, top = "(nothing)", ""
            else:
                g = x.groupby("group")["score"].max()
                pred = g.idxmax()
                b = x.sort_values("score", ascending=False).iloc[0]
                top = f"{b['common_name'] or b['label']} ({b['score']:.2f})"
            per_clip.append(dict(model=m, filename=r["filename"], category=r["category"],
                                 major_category=r["major_category"], expected=r["expected_group"],
                                 predicted=pred, correct=pred == r["expected_group"], top_label=top))
    pc = pd.DataFrame(per_clip)
    for (m, mc), d in pc.groupby(["model", "major_category"]):
        rows.append(dict(model=m, major_category=mc, clips=len(d), accuracy=d["correct"].mean(),
                         nothing_detected=(d["predicted"] == "(nothing)").mean()))
    for m, d in pc.groupby("model"):
        rows.append(dict(model=m, major_category="ALL", clips=len(d), accuracy=d["correct"].mean(),
                         nothing_detected=(d["predicted"] == "(nothing)").mean()))
    return dict(groups=pd.DataFrame(rows), clips=pc)


# ----------------------------------------------------------------------------- report
def _fmt(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format=lambda v: f"{v:.3f}", na_rep="", border=0, classes="t")


def _plot_pr(sweep: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"birdnet": "#1f77b4", "perch_v2": "#d62728"}
    for (m, v), d in sweep.groupby(["model", "variant"]):
        ax.plot(d["recall"], d["precision"], marker="o", ms=3, color=colors.get(m, "gray"),
                ls="-" if v == "no filter" else "--", label=f"{m} ({v})")
        b = d.loc[d["f1"].idxmax()]
        ax.annotate(f"{b['threshold']:g}", (b["recall"], b["precision"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points", color=colors.get(m, "gray"))
    ax.set_xlabel("Recall (share of annotated calls found)")
    ax.set_ylabel("Precision (share of detections that are correct)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_title("Window-level precision / recall over thresholds\n(label = best-F1 threshold)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def highpass_comparison(pms: dict, escs: dict) -> dict:
    """Summary tables across high-pass cut-offs (0 = no filter)."""
    out = {}
    if pms:
        rows = []
        for hp, pm in pms.items():
            sw = pm["sweep"][pm["sweep"]["variant"] == "no filter"]
            for m, d in sw.groupby("model"):
                b = d.loc[d["f1"].idxmax()]
                dfl = d[np.isclose(d["threshold"], DEFAULTS[m])]
                rows.append(dict(highpass_hz=hp, model=m, best_threshold=b["threshold"],
                                 best_precision=b["precision"], best_recall=b["recall"], best_f1=b["f1"],
                                 f1_at_default=float(dfl["f1"].iloc[0]) if len(dfl) else np.nan))
            f = pm["files"][(pm["files"]["variant"] == "no filter")]
            for _, r in f.iterrows():
                rows.append(dict(highpass_hz=hp, model=f"{r['method']} (species lists)",
                                 best_precision=r["precision"], best_recall=r["recall"], best_f1=r["f1"]))
        t = pd.DataFrame(rows).sort_values(["model", "highpass_hz"])
        base = t[t["highpass_hz"] == min(pms)].set_index("model")["best_f1"]
        t["f1_change_vs_no_filter"] = t["best_f1"] - t["model"].map(base)
        out["powdermill"] = t.reset_index(drop=True)
    if escs:
        rows = []
        for hp, esc in escs.items():
            g = esc["groups"].assign(highpass_hz=hp)
            rows.append(g)
        g = pd.concat(rows, ignore_index=True)
        out["esc50"] = g.pivot_table(index=["model", "major_category"], columns="highpass_hz",
                                     values="accuracy").reset_index()
        out["esc50"].columns = [c if isinstance(c, str) else f"accuracy @ {int(c)} Hz" for c in out["esc50"].columns]
    return out


def write_report(out: Path, pm: dict | None, esc: dict | None, comparison: dict | None = None) -> Path:
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;color:#222}"
           "table.t{border-collapse:collapse;font-size:13px;margin:8px 0 20px}"
           "table.t th,table.t td{padding:4px 8px;border-bottom:1px solid #ddd;text-align:left}"
           "table.t th{background:#f3f3f3}h2{margin-top:32px}p.n{color:#555;font-size:13px}")
    h = [f"<html><head><meta charset='utf-8'><title>BioTool evaluation</title><style>{css}</style></head><body>",
         f"<h1>BioTool evaluation</h1><p class=n>Generated {datetime.now():%Y-%m-%d %H:%M}</p>"]
    if comparison:
        h += ["<h2>High-pass filter comparison</h2><p class=n>The models were run on the same audio with a "
              "high-pass filter at each cut-off (4th-order Butterworth, applied only to what the models "
              "analyse). <b>f1_change_vs_no_filter</b> &gt; 0 means the filter helped. Detailed sections "
              "below are for the run without filter.</p>"]
        if "powdermill" in comparison:
            comparison["powdermill"].to_csv(out / "highpass_powdermill.csv", index=False)
            h += ["<h3>Powdermill (birds, window level and species lists)</h3>", _fmt(comparison["powdermill"])]
        if "esc50" in comparison:
            comparison["esc50"].to_csv(out / "highpass_esc50.csv", index=False)
            h += ["<h3>ESC-50 sound-group accuracy</h3><p class=n>Watch geophony (wind, rain, thunder) and "
                  "low-pitched anthropophony (engines): these live partly below the cut-off.</p>",
                  _fmt(comparison["esc50"])]
    if pm:
        pm["sweep"].to_csv(out / "powdermill_threshold_sweep.csv", index=False)
        pm["files"].to_csv(out / "powdermill_species_lists.csv", index=False)
        pm["file_detail"].to_csv(out / "powdermill_species_per_file.csv", index=False)
        if len(pm["combos"]):
            pm["combos"].to_csv(out / "powdermill_combinations.csv", index=False)
        _plot_pr(pm["sweep"], out / "powdermill_pr.png")
        best = pm["sweep"].loc[pm["sweep"].groupby(["model", "variant"])["f1"].idxmax()]
        dflt = pm["sweep"][pm["sweep"].apply(lambda r: np.isclose(r["threshold"], DEFAULTS[r["model"]]), axis=1)]
        h += [f"<h2>Powdermill bird soundscapes</h2><p class=n>{pm['n_files']} files, {pm['n_ann']} expert "
              f"annotations, {pm['n_species']} species. A detection is correct when the species was annotated in "
              "that model window (BirdNET 3 s, Perch 5 s). Location filter: BirdNET range model at "
              f"{POWDERMILL_LATLON} on each recording's date.</p>",
              "<h3>Window level: current default thresholds</h3>", _fmt(dflt),
              "<h3>Window level: best threshold (max F1)</h3>", _fmt(best),
              "<img src='powdermill_pr.png'>"]
        if len(pm["combos"]):
            h += ["<h3>Combining models (3 s grid)</h3><p class=n>OR = either model detects the species; "
                  "AND = both agree.</p>", _fmt(pm["combos"])]
        h += ["<h3>Species lists per file (default thresholds)</h3><p class=n>Does the tool produce the right "
              "species list for each 5-minute file? (Annotators skipped short chips, so some 'not annotated' "
              "species may still be present.)</p>", _fmt(pm["files"]),
              _fmt(pm["file_detail"])]
    if esc:
        esc["groups"].to_csv(out / "esc50_group_accuracy.csv", index=False)
        esc["clips"].to_csv(out / "esc50_per_clip.csv", index=False)
        h += ["<h2>ESC-50 sample: sound groups</h2><p class=n>Each 5 s clip should be assigned to its "
              "expected group (animals = biophony, natural soundscapes = geophony, human/domestic/urban = "
              "anthropophony). Predicted group = group with the highest score at default thresholds.</p>",
              _fmt(esc["groups"]),
              "<h3>Per clip</h3>", _fmt(esc["clips"])]
    h.append("</body></html>")
    p = out / "evaluation_report.html"
    p.write_text("\n".join(h), encoding="utf-8")
    return p


def main(a) -> None:
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    cutoffs = sorted({float(x) for x in str(getattr(a, "highpass", "0")).split(",") if x.strip()}) or [0.0]
    pms, escs = {}, {}
    if a.powdermill and not (a.powdermill_annotations and a.powdermill_readme):
        raise SystemExit("--powdermill needs --powdermill-annotations and --powdermill-readme")
    for hp in cutoffs:
        if a.powdermill:
            pms[hp] = evaluate_powdermill(a.powdermill, a.powdermill_annotations, a.powdermill_readme, out,
                                          [m for m in models if m in SPECIES_MODELS], a.threads, highpass=hp)
        if a.esc50:
            escs[hp] = evaluate_esc50(a.esc50, out, models, a.threads, highpass=hp)
    pm = pms.get(cutoffs[0])
    esc = escs.get(cutoffs[0])
    comparison = highpass_comparison(pms, escs) if len(cutoffs) > 1 else None
    rep = write_report(out, pm, esc, comparison)
    if comparison and "powdermill" in comparison:
        print("\nHigh-pass comparison (Powdermill, best F1 per model):")
        print(comparison["powdermill"][["highpass_hz", "model", "best_threshold", "best_f1",
                                        "f1_change_vs_no_filter"]].to_string(index=False))
    print(f"\nReport: {rep}")
    (out / "summary.json").write_text(json.dumps(
        {"best_thresholds": {f"{m} ({v})": t for (m, v), t in (pm or {}).get("best", {}).items()}}, indent=2))
