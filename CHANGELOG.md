# Changelog

## 0.5.0 (10 Nov 2026)
- Optional model **BirdCODE** (Earth Species Project): zero-shot sound event detection for 9,258 bird
  species, with a start and end time for each call or song. Installed with `InstallBirdCODE.bat`; the
  model downloads from Hugging Face on first use (or copy the models folder). Uses the original code
  pinned to a tested commit, with its default settings (threshold 0.5, window overlap 0.5, merge gap 1 s,
  NMS IoU 0.8, adjustable under *Advanced*). It analyses each recording as a whole, like the original,
  instead of BioTool's 60 s reading blocks.
  `TestBirdCODE.bat` checks the output against the expected results published with BirdCODE: on
  Windows (torch 2.11 CPU) all 5 events of the two demo recordings matched, with the same species,
  times and scores (to 4 decimals).
- **Clearer interface.** The left panel is reduced to three steps: models, extra outputs and the
  results folder. Model parameters, location/season filter, species list, CPU threads, embeddings and
  Raven tables moved to a new *Advanced settings* tab; the left panel lists any that differ from the
  defaults. The single-file toolbar shows the main actions as large buttons, the selection tools in a
  second row, and the view & listening options folded away. Every button and setting has a tooltip.
- **Runs are saved automatically** to `<results folder>/<date>_<time>_<file name>` (option on the
  left; default `Documents\BioTool results`). **Load results…** (Ctrl+L, *File → Recent results*)
  reopens a saved single-file or batch run with its recording, without running the models again.
  Batch output now defaults to `batch_<folder name>` in the results folder.
- **Sound intervals** (option, also `--intervals`): consecutive windows with the same label are joined
  into one interval per sound with start, end and duration; bird intervals take BirdCODE's times when
  BirdCODE ran too. New *Sound intervals* tab, lanes draw intervals, `sound_intervals.csv` and Raven
  interval tables. Checked on the Powdermill annotations (BirdNET): median centre error 0.5 s,
  start/end about 1 s.
- **+ frequency band from the spectrogram** (sub-option, `--intervals-spectrogram`): scikit-maad
  segmentation gives each interval a frequency band and trims its edges (Powdermill: band within
  ~0.7/1.9 kHz, start/end error 1.00 -> 0.93 s). Includes a fallback for steady hiss, which otherwise
  merged most of the spectrogram into one region.
- DeepForestSound now runs on current PyTorch / timm (torch 2.11, timm 1.0), so both optional models
  share one installation. Re-checked against the DFS reference output on the 2 reference recordings
  available for this release (504 scores): largest difference 0.0001, same decisions.
- torch, torchaudio and torchvision are pinned as a matched set (mismatched versions fail to load).
- Python 3.12 required (was 3.11-3.12; `setup.bat` always used 3.12).
- `setup.bat` and the optional installers keep every optional model that was installed before.

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
