# Changelog

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
