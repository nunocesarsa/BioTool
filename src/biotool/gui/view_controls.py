"""Viewing and listening controls for the spectrogram (display/playback only - never touch the data)."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

FREQ_PRESETS = [
    ("Full range", None),
    ("Low 0-2 kHz (frogs, mammals, engines)", (0.0, 2.0)),
    ("Birds 1-10 kHz", (1.0, 10.0)),
    ("Songbirds 2-8 kHz", (2.0, 8.0)),
    ("High 4 kHz+ (insects)", (4.0, None)),
    ("Custom", "custom"),
]
HISS = [("Off", 0.0), ("Light", 0.3), ("Medium", 0.55), ("Strong", 0.8)]
SPEEDS = [("1x", 1.0), ("0.5x", 0.5), ("0.25x", 0.25), ("0.1x", 0.1)]
FFTS = [256, 512, 1024, 2048]
CMAPS = ["viridis", "inferno", "magma", "CET-L1"]  # CET-L1 = greyscale


class ViewControls(QWidget):
    display_changed = Signal()      # colours / contrast / background removal
    fft_changed = Signal(int)       # needs the spectrogram recomputed
    freq_changed = Signal(float, float)  # kHz range to show
    export_clicked = Signal()
    box_clicked = Signal(bool)      # True = add box, False = remove

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fmax_khz = 24.0
        self._block = False

        # --- view row
        self.cmap = QComboBox(); self.cmap.addItems(["viridis", "inferno", "magma", "grey"])
        self.denoise = QCheckBox("Remove background")
        self.denoise.setChecked(True)
        self.denoise.setToolTip("Subtract each frequency's median level (display only) so calls stand out")
        self.bright = QSlider(Qt.Horizontal); self.bright.setRange(-30, 30); self.bright.setValue(0)
        self.bright.setFixedWidth(90); self.bright.setToolTip("Brightness (dB). Double-click label to reset.")
        self.contrast = QSlider(Qt.Horizontal); self.contrast.setRange(25, 300); self.contrast.setValue(100)
        self.contrast.setFixedWidth(90); self.contrast.setToolTip("Contrast: lower = more contrast")
        self.fft = QComboBox()
        for n in FFTS:
            self.fft.addItem(str(n), n)
        self.fft.setCurrentIndex(1)
        self.fft.setToolTip("FFT size: larger = finer frequency detail (low/tonal sounds), smaller = finer timing")
        self.preset = QComboBox()
        for name, _ in FREQ_PRESETS:
            self.preset.addItem(name)
        self.fmin = QDoubleSpinBox(); self.fmax = QDoubleSpinBox()
        for w in (self.fmin, self.fmax):
            w.setDecimals(1); w.setSingleStep(0.5); w.setSuffix(" kHz"); w.setRange(0, 200)
        self.reset_btn = QPushButton("Reset view")

        r1 = QHBoxLayout()
        r1.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("<b>View</b>"), QLabel("Colours"), self.cmap, self.denoise, QLabel("Brightness"),
                  self.bright, QLabel("Contrast"), self.contrast, QLabel("FFT"), self.fft,
                  QLabel("Frequency"), self.preset, self.fmin, QLabel("-"), self.fmax, self.reset_btn):
            r1.addWidget(w)
        r1.addStretch(1)

        # --- listen row
        self.speed = QComboBox()
        for name, v in SPEEDS:
            self.speed.addItem(name, v)
        self.speed.setToolTip("Slow playback (time expansion): pitch drops by the same factor.\n"
                              "Useful for fast trills, insects and ultrasound.")
        self.band = QCheckBox("Only the visible frequency band")
        self.band.setToolTip("Play through a band-pass filter matching the frequency range shown above\n"
                             "(e.g. pick 'Songbirds 2-8 kHz' to hide wind and traffic rumble while listening).")
        self.hiss = QComboBox()
        for name, v in HISS:
            self.hiss.addItem(name, v)
        self.hiss.setToolTip("Reduce steady background hiss while listening (spectral gating).\n"
                             "Learns the background from the quietest moments around the selection.\n"
                             "Playback/export only - the models always analyse the original audio.")
        self.box_btn = QPushButton("Draw box")
        self.box_btn.setObjectName("accent")
        self.box_btn.setCheckable(True)
        self.box_btn.setToolTip("Put a time x frequency box on the spectrogram (drag/resize it).\n"
                                "Play and Export then use exactly the box: its time span and frequency band.")
        self.box_btn.toggled.connect(lambda on: (self.box_btn.setText("Remove box" if on else "Draw box"),
                                                 self.box_clicked.emit(on)))
        self.export_btn = QPushButton("Export clip...")
        self.export_btn.setToolTip("Save a WAV copy of the selection (with the band filter if ticked).\n"
                                   "The original file is not changed.")
        r2 = QHBoxLayout()
        r2.setContentsMargins(0, 0, 0, 0)
        note = QLabel("display and playback only - models and files are not affected")
        note.setObjectName("muted")
        for w in (QLabel("<b>Listen</b>"), QLabel("Speed"), self.speed, QLabel("Reduce hiss"), self.hiss,
                  self.band, self.box_btn, self.export_btn, note):
            r2.addWidget(w)
        r2.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(r1)
        lay.addLayout(r2)

        for sig in (self.cmap.currentIndexChanged, self.denoise.toggled, self.bright.valueChanged,
                    self.contrast.valueChanged):
            sig.connect(lambda *_: self.display_changed.emit())
        self.fft.currentIndexChanged.connect(lambda _: self.fft_changed.emit(self.fft.currentData()))
        self.preset.currentIndexChanged.connect(self._preset)
        self.fmin.editingFinished.connect(self._manual_freq)
        self.fmax.editingFinished.connect(self._manual_freq)
        self.reset_btn.clicked.connect(self.reset)
        self.export_btn.clicked.connect(self.export_clicked.emit)
        self._restore()

    # -- values
    def colormap_name(self) -> str:
        return {"grey": "CET-L1"}.get(self.cmap.currentText(), self.cmap.currentText())

    def freq_range(self) -> tuple[float, float]:
        return self.fmin.value(), self.fmax.value()

    def playback_speed(self) -> float:
        return float(self.speed.currentData())

    def hiss_strength(self) -> float:
        return float(self.hiss.currentData())

    def playback_band_hz(self) -> tuple[float, float] | None:
        if not self.band.isChecked():
            return None
        lo, hi = self.freq_range()
        if lo <= 0 and hi >= self.fmax_khz:
            return None
        return lo * 1000, hi * 1000

    def levels(self, base_lo: float, base_hi: float) -> tuple[float, float]:
        rng = (base_hi - base_lo) * self.contrast.value() / 100
        hi = base_hi - self.bright.value()
        return hi - rng, hi

    # -- frequency range
    def set_nyquist(self, fmax_khz: float):
        self.fmax_khz = fmax_khz
        self._block = True
        self.fmin.setRange(0, fmax_khz); self.fmax.setRange(0.1, fmax_khz)
        self._block = False
        self._preset(self.preset.currentIndex())

    def _apply(self, lo, hi, emit=True):
        hi = self.fmax_khz if hi is None else min(hi, self.fmax_khz)
        lo = max(0.0, min(lo, hi - 0.1))
        self._block = True
        self.fmin.setValue(lo); self.fmax.setValue(hi)
        self._block = False
        if emit:
            self.freq_changed.emit(lo, hi)

    def _preset(self, i):
        val = FREQ_PRESETS[i][1]
        if val == "custom":
            return
        lo, hi = (0.0, None) if val is None else val
        self._apply(lo, hi)
        self._save()

    def _manual_freq(self):
        if self._block:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(len(FREQ_PRESETS) - 1)  # Custom
        self.preset.blockSignals(False)
        self._apply(self.fmin.value(), self.fmax.value())

    def reset(self):
        self.bright.setValue(0); self.contrast.setValue(100); self.denoise.setChecked(True)
        self.cmap.setCurrentIndex(0); self.speed.setCurrentIndex(0); self.band.setChecked(False)
        self.hiss.setCurrentIndex(0); self.box_btn.setChecked(False)
        self.preset.setCurrentIndex(0)
        self._apply(0.0, None)
        if self.fft.currentIndex() != 1:
            self.fft.setCurrentIndex(1)

    # -- remember
    def _save(self):
        q = QSettings("BioTool", "BioTool")
        q.setValue("view/cmap", self.cmap.currentIndex()); q.setValue("view/denoise", self.denoise.isChecked())
        q.setValue("view/fft", self.fft.currentIndex()); q.setValue("view/preset", self.preset.currentIndex())

    def _restore(self):
        q = QSettings("BioTool", "BioTool")
        try:
            self.cmap.setCurrentIndex(int(q.value("view/cmap", 0)))
            self.denoise.setChecked(str(q.value("view/denoise", "true")).lower() == "true")
            self.fft.setCurrentIndex(int(q.value("view/fft", 1)))
            p = int(q.value("view/preset", 0))
            self.preset.setCurrentIndex(p if FREQ_PRESETS[p][1] != "custom" else 0)
        except (TypeError, ValueError, IndexError):
            pass
        for sig in (self.cmap.currentIndexChanged, self.denoise.toggled, self.fft.currentIndexChanged):
            sig.connect(lambda *_: self._save())
