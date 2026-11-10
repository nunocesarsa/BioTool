"""Check that BioTool's BirdCODE gives the same events as the original BirdCODE code.

Uses the two demo recordings and expected selection tables shipped with
github.com/earthspecies/sound-event-detection (same commit as the installed package) and
runs them through BioTool's normal pipeline with BirdCODE's default settings.

    uv run python -m biotool.check_birdcode        (or TestBirdCODE.bat)
"""

from __future__ import annotations

import sys
import urllib.request

COMMIT = "617c6408fdd1b56c3873ef2d17006ffbaee61783"
RAW = "https://raw.githubusercontent.com/earthspecies/sound-event-detection/{}/tests/samples/demo/audio/{}"
# tests/samples/demo/output_expected/*.txt at COMMIT: (begin s, end s, species, score)
EXPECTED = {
    "20230730.wav": [(3.8157894736842106, 5.7894736842105265, "Corvus brachyrhynchos", 0.6134416)],
    "20260623.wav": [(0.2631578947368421, 1.5789473684210527, "Corvus corax", 0.8774136),
                     (3.4210526315789473, 5.2631578947368425, "Corvus corax", 0.8232688),
                     (8.026315789473685, 8.421052631578947, "Corvus corax", 0.86987114),
                     (3.68421052631579, 4.078947368421053, "Catharus ustulatus", 0.57417274)],
}
TIME_TOL = 0.06   # s; less than one model frame
SCORE_TOL = 0.02  # CPU vs GPU and library versions give tiny score differences


def main(argv=None) -> int:
    from .paths import app_data_dir
    from .pipeline import Engine, RunSettings

    folder = app_data_dir() / "birdcode_check"
    folder.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        f = folder / name
        if not f.exists():
            print(f"downloading demo audio {name} ...")
            urllib.request.urlretrieve(RAW.format(COMMIT, name), f)

    engine = Engine(RunSettings(models={"birdcode": {}}), progress=print)
    engine.load()
    ok = True
    lines = []
    for name, exp in EXPECTED.items():
        det = engine.analyze_file(folder / name).detections
        got = sorted((float(r.start_s), float(r.end_s), r.scientific_name, float(r.score))
                     for r in det.itertuples())
        exp = sorted((round(a, 3), round(b, 3), sp, sc) for a, b, sp, sc in exp)
        lines.append(f"\n{name}: expected {len(exp)} events, BioTool found {len(got)}")
        same = len(got) == len(exp)
        for e in exp:
            m = [g for g in got if g[2] == e[2] and abs(g[0] - e[0]) <= TIME_TOL and abs(g[1] - e[1]) <= TIME_TOL]
            if m:
                g = min(m, key=lambda g: abs(g[3] - e[3]))
                good = abs(g[3] - e[3]) <= SCORE_TOL
                same &= good
                lines.append(f"  {'OK  ' if good else 'DIFF'} {e[2]:24s} {e[0]:6.2f}-{e[1]:6.2f} s  "
                             f"score expected {e[3]:.4f}, BioTool {g[3]:.4f} ({g[0]:.2f}-{g[1]:.2f} s)")
            else:
                same = False
                lines.append(f"  MISS {e[2]:24s} {e[0]:6.2f}-{e[1]:6.2f} s  score {e[3]:.4f}")
        for g in got:
            if not any(g[2] == e[2] and abs(g[0] - e[0]) <= TIME_TOL for e in exp):
                lines.append(f"  EXTRA {g[2]:23s} {g[0]:6.2f}-{g[1]:6.2f} s  score {g[3]:.4f}")
        ok &= same
    import torch
    lines.append(f"\ntorch {torch.__version__}; tolerance {TIME_TOL} s / score {SCORE_TOL}")
    lines.append("RESULT: BirdCODE matches the original" if ok else "RESULT: differences found (see above)")
    text = "\n".join(lines)
    print(text)
    out = folder / "birdcode_check_report.txt"
    out.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"\nreport saved to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
