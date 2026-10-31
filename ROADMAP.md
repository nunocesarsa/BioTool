# BioTool roadmap

## Current state (v0.2.0, 30 Oct 2026)
- Models: BirdNET v2.4, Perch v2, YAMNet, acoustic indices (scikit-maad); embeddings saved per window.
- Desktop app: Single file, Batch (with resume), Explore sounds (embedding clusters, naming, export).
- Location/season filter (BirdNET range model); advanced per-model settings; settings remembered.
- `Evaluate.bat`: Powdermill + ESC-50 report. Best minimum scores found: BirdNET 0.25-0.3, Perch 0.03.

## Agreed items (31 Oct 2026)
1. **Review mode** - NEXT - step through detections (by species / model / score), listen, mark
   *correct / wrong / unsure*. Save verdicts next to the results (e.g. `reviews.csv`), show them in the
   tables. Verified data from your own sites feeds per-site threshold tuning and (later) custom evaluation.
2. **Listening and viewing aids** - DONE: colour map, background removal, brightness/contrast,
   FFT size, frequency presets/zoom, slowed playback, band-pass playback, hiss reduction,
   time x frequency box selection, clip export (copies only).
3. **High-pass pre-filter experiment** - DONE: `HighpassTest.bat`. BirdNET gets clearly worse with any
   high-pass (F1 0.78 -> 0.51 at 300 Hz), so no pre-filtering; kept as CLI option `--highpass`.
   Perch / YAMNet results: see `evaluation/highpass_powdermill.csv` after running the .bat.

4. **More settings + Analyse selection** - DONE: Perch overlap, YAMNet grouping window/method,
   species list (include/exclude), agreement grid, skip quiet windows, index bands; Analyse selection
   (Ctrl+E) shows each model's top labels for one selection/box; themes, shortcuts, drag & drop.

## Later
- **AnimalCLAP** as an optional experimental model (separate install, PyTorch): text search
  ("frog chorus") and suggested names for groups in Explore; evaluate before trusting for species.
  Ask the authors about a licence for the code and weights first.
- **Evaluation on custom datasets** (when real field data is available): dataset folder with
  `audio/`, Raven tables or `annotations.csv`, `labels.csv` mapping, `dataset.json`
  (complete / targets / presence annotations, site coordinates); per-species metrics; Evaluate tab.
- **One-click .exe** once features settle: first drop TensorFlow (Perch ONNX, BirdNET TFLite,
  YAMNet converted) to shrink the installer to ~300-400 MB; consider code signing.
