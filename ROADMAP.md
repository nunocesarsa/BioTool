# BioTool roadmap

## Current state (v0.5.0, 10 Nov 2026)
- Models: BirdNET v2.4, Perch v2, YAMNet, acoustic indices (scikit-maad); optional PyTorch models
  DeepForestSound (African forest species) and BirdCODE (zero-shot bird sound event detection), both
  checked against their authors' reference outputs / expected results (BirdCODE: run `TestBirdCODE.bat`).
- Sound intervals (start/end per sound) with optional frequency band from the spectrogram.
- Desktop app: Single file, Batch (with resume), Explore sounds, Advanced settings; every run is saved
  automatically and can be loaded again.
- Location/season filter, species list, settings remembered. `Evaluate.bat`: Powdermill + ESC-50 report.

## Next
- **Background sound file** - remove the confidently detected sounds (time x frequency boxes from the
  sound intervals) from the spectrogram, fill them with the background level, and write
  `background.wav`: ambience beds for training data, indices of the background alone, and a way to hear
  what the models missed.
- **Review mode** - step through detections (by species / model / score), listen, mark
  *correct / wrong / unsure*; save verdicts next to the results (`reviews.csv`) and show them in the
  tables. Verified data from your own sites feeds per-site threshold tuning and custom evaluation.
- **Import models / open models folder** - for networks that block the model downloads
  (storage.googleapis.com, huggingface.co): copy the models folder from another computer from the app.

## Done
- Listening and viewing aids (colour map, background removal, contrast, FFT size, frequency zoom,
  slowed / band-pass playback, hiss reduction, box selection, clip export).
- High-pass pre-filter experiment (`HighpassTest.bat`): BirdNET gets clearly worse with any high-pass
  (F1 0.78 -> 0.51 at 300 Hz), so no pre-filtering; kept as CLI option `--highpass`.
- More settings + Analyse selection (Ctrl+E), themes, shortcuts, drag & drop.
- DeepForestSound (0.4.0), BirdCODE, sound intervals, clearer interface with auto-saved runs (0.5.0).

## Later
- **AnimalCLAP** as an optional experimental model (separate install, PyTorch): text search
  ("frog chorus") and suggested names for groups in Explore; evaluate before trusting for species.
  Ask the authors about a licence for the code and weights first.
- **Evaluation on custom datasets** (when real field data is available): dataset folder with
  `audio/`, Raven tables or `annotations.csv`, `labels.csv` mapping, `dataset.json`
  (complete / targets / presence annotations, site coordinates); per-species metrics; Evaluate tab.
- **One-click .exe** once features settle: first drop TensorFlow (Perch ONNX, BirdNET TFLite,
  YAMNet converted) to shrink the installer to ~300-400 MB; consider code signing.
