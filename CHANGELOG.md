# Changelog

## 0.4.0 (1 Nov 2026)
- Optional model **DeepForestSound** (MNHN): 12 African tropical-forest species incl. forest elephant
  rumbles and chimpanzee. Installed with `InstallDeepForestSound.bat` (PyTorch); code and weights are
  downloaded from the original repository on first use (not redistributed). Validated against the
  authors' 7 test files: same 13 positive windows, all 504 scores within 0.004 of their reference
  output; per-species thresholds from the DFS paper by default.
- Every model that ran gets a lane under the spectrogram (also when it found nothing).
- Clearer empty view before a file is loaded; wider settings panel.

## 0.3.0 – first public release (31 Oct 2026)
- Models: BirdNET v2.4, Perch v2, YAMNet, acoustic indices (scikit-maad); embeddings saved per window.
- Desktop app: Single file, Batch (resumable), Explore sounds (embedding clusters, naming, export).
- Model agreement, sound groups (bio/geo/anthropophony), species summaries, Raven tables.
- Location/season filter (BirdNET range model); custom species list (include/exclude).
- Advanced settings per model (overlap, sensitivity, YAMNet grouping, max labels, index bands),
  agreement grid, skip quiet windows; settings remembered.
- Listening aids: colour maps, contrast, FFT size, frequency zoom, slowed playback, band-pass playback,
  hiss reduction, time × frequency box, clip export (copies only).
- Analyse selection: what each model thinks is in one selection/box.
- Evaluation against Powdermill + ESC-50 (`Evaluate.bat`), high-pass experiment (`HighpassTest.bat`),
  test-data downloader (`GetTestData.bat`).
- Themes (Classic, Light, Grey, Dark), keyboard shortcuts, drag & drop.
