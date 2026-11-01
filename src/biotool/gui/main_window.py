"""Main window: model panel on the left, Single-file and Batch tabs on the right."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QRectF, QSettings, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QToolButton,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import __version__, merge
from ..audio import AUDIO_EXTENSIONS
from ..models import DETECTORS, AcousticIndices
from ..models.indices import INDEX_HELP
from ..location import LocationSettings
from ..pipeline import RunSettings, combine_outputs, write_file_outputs
from . import theme, workers
from .selection_dialog import SelectionDialog
from .table import DataFrameView

GROUP_COLORS = {
    "biophony": (44, 160, 44),
    "geophony": (31, 119, 180),
    "anthropophony": (214, 39, 40),
    "other": (127, 127, 127),
}

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

SETTINGS_VERSION = 2
RETUNED_DEFAULTS = {"perch_v2"}  # defaults changed in v2 (Perch 0.1 -> 0.03 after evaluation)


# ----------------------------------------------------------------------------- model panel
class ModelPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        box = QGroupBox("Models")
        form = QVBoxLayout(box)
        self.rows = {}
        self.adv: dict[str, dict] = {}
        for key, cls in DETECTORS.items():
            cb = QCheckBox(cls.name)
            cb.setChecked(key in ("birdnet", "perch_v2", "yamnet"))  # optional models start unticked
            cb.setToolTip(f"{cls.description}\n{cls.licence_note}")
            th = QDoubleSpinBox()
            th.setDecimals(3)
            th.setRange(0.001, 0.99)
            th.setSingleStep(0.01 if cls.default_threshold < 0.1 else 0.05)
            th.setValue(cls.default_threshold)
            th.setToolTip("Minimum score to report a detection")
            r = QHBoxLayout()
            r.addWidget(cb, 1)
            r.addWidget(QLabel("min score"))
            r.addWidget(th)
            form.addLayout(r)
            ok, why = cls.availability()
            desc = QLabel(cls.description + ("" if ok else f"<br><b>{why}</b>"))
            desc.setWordWrap(True)
            desc.setObjectName("muted")
            desc.setStyleSheet("font-size: 9pt; margin-left: 20px;")
            form.addWidget(desc)
            if not ok:
                cb.setChecked(False)
                cb.setEnabled(False)
                th.setEnabled(False)
            if key == "dfs":
                th.setToolTip("Used only when Advanced > Thresholds = 'min score above'.\n"
                              "By default DFS uses the per-species thresholds from its paper.")
            self.rows[key] = (cb, th)
            self._add_advanced(form, key, cls)
        self.indices = QCheckBox(AcousticIndices.name)
        self.indices.setChecked(True)
        self.indices.setToolTip("\n".join(f"{k}: {v}" for k, v in INDEX_HELP.items()))
        form.addWidget(self.indices)
        d = QLabel(AcousticIndices.description)
        d.setWordWrap(True)
        d.setObjectName("muted")
        d.setStyleSheet("font-size: 9pt; margin-left: 20px;")
        form.addWidget(d)
        self._add_advanced(form, "indices", AcousticIndices)
        lay.addWidget(box)

        opt = QGroupBox("Options")
        f2 = QFormLayout(opt)
        self.embeddings = QCheckBox("Save embeddings")
        self.embeddings.setChecked(True)
        self.embeddings.setToolTip("Feature vectors per window - for clustering unknown sounds,\n"
                                   "search-by-example and training custom classifiers later.")
        self.raven = QCheckBox("Write Raven selection tables")
        self.raven.setChecked(True)
        self.threads = QSpinBox()
        self.threads.setRange(1, max(1, os.cpu_count() or 1))
        self.threads.setValue(max(1, (os.cpu_count() or 2) - 1))
        self.threads.setToolTip("CPU threads used by the models")
        self.grid = QDoubleSpinBox(); self.grid.setRange(1.0, 10.0); self.grid.setSingleStep(0.5)
        self.grid.setDecimals(1); self.grid.setValue(3.0); self.grid.setSuffix(" s")
        self.grid.setToolTip("Time grid used to compare models in the agreement / sound-group tables.\n"
                             "3 s lines up with BirdNET; use 5 s to line up with Perch.")
        self.quiet_on = QCheckBox("Skip quiet windows below")
        self.quiet = QDoubleSpinBox(); self.quiet.setRange(-100, -20); self.quiet.setDecimals(0)
        self.quiet.setSingleStep(5); self.quiet.setValue(-70); self.quiet.setSuffix(" dBFS")
        tip = ("BirdNET/Perch skip windows quieter than this (RMS level) - faster for long, mostly silent\n"
               "recordings. Check your files' level first (column rms_dbfs in the acoustic indices):\n"
               "quiet recorders sit around -45 to -60 dBFS, and faint distant calls may be skipped.")
        self.quiet_on.setToolTip(tip); self.quiet.setToolTip(tip)
        self.quiet.setEnabled(False)
        self.quiet_on.toggled.connect(self.quiet.setEnabled)
        qrow = QHBoxLayout(); qrow.addWidget(self.quiet_on); qrow.addWidget(self.quiet)
        f2.addRow(self.embeddings)
        f2.addRow(self.raven)
        f2.addRow("CPU threads", self.threads)
        f2.addRow("Agreement grid", self.grid)
        f2.addRow(qrow)
        lay.addWidget(opt)

        # custom species list
        self.sl_box = QGroupBox("Species list")
        self.sl_box.setCheckable(True)
        self.sl_box.setChecked(False)
        self.sl_box.setToolTip("Keep only (or remove) the species in a text file - BirdNET and Perch species.")
        f4 = QFormLayout(self.sl_box)
        self.sl_mode = QComboBox(); self.sl_mode.addItems(["Keep only these species", "Remove these species"])
        self.sl_path = QLineEdit(); self.sl_path.setPlaceholderText("species_list.txt")
        sl_browse = QPushButton("..."); sl_browse.setFixedWidth(32)
        sl_browse.clicked.connect(self._browse_species_list)
        prow = QHBoxLayout(); prow.addWidget(self.sl_path, 1); prow.addWidget(sl_browse)
        f4.addRow(self.sl_mode)
        f4.addRow("File", prow)
        sl_hint = QLabel("One species per line: scientific name, English name or BirdNET's "
                         "'Scientific_Common' format. Lines starting with # are ignored.")
        sl_hint.setWordWrap(True); sl_hint.setObjectName("muted"); sl_hint.setStyleSheet("font-size: 9pt;")
        f4.addRow(sl_hint)
        lay.addWidget(self.sl_box)

        # location / season filter
        self.loc_box = QGroupBox("Location / season filter")
        self.loc_box.setCheckable(True)
        self.loc_box.setChecked(False)
        self.loc_box.setToolTip("Drop BirdNET/Perch species that BirdNET's range model considers\n"
                                "unlikely at this site and time of year.")
        f3 = QFormLayout(self.loc_box)
        self.lat = QDoubleSpinBox(); self.lat.setRange(-90, 90); self.lat.setDecimals(5)
        self.lon = QDoubleSpinBox(); self.lon.setRange(-180, 180); self.lon.setDecimals(5)
        self.date_mode = QComboBox()
        self.date_mode.addItems(["Date from file name / AudioMoth", "Fixed date", "All year"])
        self.date_edit = QDateEdit(QDate.currentDate()); self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_mode.currentIndexChanged.connect(lambda i: self.date_edit.setEnabled(i == 1))
        self.date_edit.setEnabled(False)
        self.min_occ = QDoubleSpinBox(); self.min_occ.setRange(0.001, 0.5); self.min_occ.setDecimals(3)
        self.min_occ.setSingleStep(0.01); self.min_occ.setValue(0.03)
        self.min_occ.setToolTip("Species with a predicted occurrence below this are removed (BirdNET default 0.03)")
        f3.addRow("Latitude", self.lat)
        f3.addRow("Longitude", self.lon)
        f3.addRow("Date", self.date_mode)
        f3.addRow("", self.date_edit)
        f3.addRow("Min. occurrence", self.min_occ)
        hint = QLabel("Files without a date in their name use the all-year species list.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        hint.setStyleSheet("font-size: 9pt;")
        f3.addRow(hint)
        lay.addWidget(self.loc_box)
        reset = QPushButton("Reset model settings to defaults")
        reset.clicked.connect(self.reset_defaults)
        lay.addWidget(reset)
        lay.addStretch(1)
        self._restore()

    # -- advanced per-model settings
    def _add_advanced(self, form, key, cls):
        self.adv.setdefault(key, {})
        btn = QToolButton()
        btn.setText("Advanced")
        btn.setCheckable(True)
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setArrowType(Qt.RightArrow)
        btn.setStyleSheet("QToolButton { border: none; color: palette(link); margin-left: 16px; }")
        box = QWidget()
        f = QFormLayout(box)
        f.setContentsMargins(28, 0, 0, 6)
        for spec in cls.advanced:
            if spec["kind"] == "text":
                w = QLineEdit(str(spec["default"]))
                w.setCursorPosition(0)
            elif spec["kind"] == "choice":
                w = QComboBox()
                w.addItems(spec["choices"])
                w.setCurrentText(str(spec["default"]))
            elif spec["kind"] == "int":
                w = QSpinBox()
                lo, hi, st = spec["range"]
                w.setRange(int(lo), int(hi)); w.setSingleStep(int(st)); w.setValue(int(spec["default"]))
            else:
                w = QDoubleSpinBox()
                lo, hi, st = spec["range"]
                w.setDecimals(2); w.setRange(lo, hi); w.setSingleStep(st); w.setValue(float(spec["default"]))
            w.setToolTip(spec.get("help", ""))
            lab = QLabel(spec["label"])
            lab.setToolTip(spec.get("help", ""))
            f.addRow(lab, w)
            self.adv[key][spec["key"]] = (spec, w)
        box.setVisible(False)

        def toggle(on):
            box.setVisible(on)
            btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        btn.toggled.connect(toggle)
        form.addWidget(btn)
        form.addWidget(box)

    def _adv_values(self, key) -> dict:
        out = {}
        for k, (spec, w) in self.adv.get(key, {}).items():
            v = (w.text().strip() if spec["kind"] == "text" else
                 w.currentText() if spec["kind"] == "choice" else w.value())
            if spec["kind"] == "float":
                v = round(float(v), 3)
            out[k] = v
        return out

    def _adv_set(self, key, k, value):
        spec, w = self.adv[key][k]
        if spec["kind"] == "text":
            w.setText(str(value)); w.setCursorPosition(0)
        elif spec["kind"] == "choice":
            if str(value) in spec["choices"]:
                w.setCurrentText(str(value))
        elif spec["kind"] == "int":
            w.setValue(int(float(value)))
        else:
            w.setValue(float(value))

    def reset_defaults(self):
        for key, cls in DETECTORS.items():
            self.rows[key][1].setValue(cls.default_threshold)
        for key in self.adv:
            for k, (spec, _) in self.adv[key].items():
                self._adv_set(key, k, spec["default"])
        self.min_occ.setValue(0.03)
        self.grid.setValue(3.0)
        self.quiet_on.setChecked(False); self.quiet.setValue(-70)
        self._save()

    def _browse_species_list(self):
        p, _ = QFileDialog.getOpenFileName(self, "Species list", self.sl_path.text() or str(Path.home()),
                                           "Text files (*.txt *.csv);;All files (*)")
        if p:
            self.sl_path.setText(p)

    def settings(self) -> RunSettings:
        models = {k: {"threshold": round(th.value(), 3), **self._adv_values(k)}
                  for k, (cb, th) in self.rows.items() if cb.isChecked()}
        if self.indices.isChecked():
            models["indices"] = self._adv_values("indices")
        loc = {}
        if self.loc_box.isChecked():
            loc = LocationSettings(enabled=True, lat=round(self.lat.value(), 5), lon=round(self.lon.value(), 5),
                                   date_mode=["file", "fixed", "year"][self.date_mode.currentIndex()],
                                   fixed_date=self.date_edit.date().toString("yyyy-MM-dd")
                                   if self.date_mode.currentIndex() == 1 else "",
                                   min_occurrence=round(self.min_occ.value(), 4)).to_dict()
        sl = {}
        if self.sl_box.isChecked() and self.sl_path.text().strip():
            sl = {"enabled": True, "mode": ["include", "exclude"][self.sl_mode.currentIndex()],
                  "path": self.sl_path.text().strip()}
        self._save()
        return RunSettings(models=models, save_embeddings=self.embeddings.isChecked(),
                           save_raven=self.raven.isChecked(), threads=self.threads.value(), location=loc,
                           grid_s=round(self.grid.value(), 2), species_list=sl,
                           quiet_dbfs=round(self.quiet.value(), 1) if self.quiet_on.isChecked() else None)

    # -- remember settings between sessions
    def _save(self):
        q = QSettings("BioTool", "BioTool")
        q.setValue("settings_version", SETTINGS_VERSION)
        for k, (cb, th) in self.rows.items():
            q.setValue(f"models/{k}/on", cb.isChecked())
            q.setValue(f"models/{k}/threshold", th.value())
        for k in self.adv:
            for ak, v in self._adv_values(k).items():
                q.setValue(f"models/{k}/adv/{ak}", v)
        q.setValue("models/indices/on", self.indices.isChecked())
        q.setValue("opt/grid", self.grid.value())
        q.setValue("opt/quiet_on", self.quiet_on.isChecked())
        q.setValue("opt/quiet", self.quiet.value())
        q.setValue("sl/on", self.sl_box.isChecked())
        q.setValue("sl/mode", self.sl_mode.currentIndex())
        q.setValue("sl/path", self.sl_path.text())
        q.setValue("opt/embeddings", self.embeddings.isChecked())
        q.setValue("opt/raven", self.raven.isChecked())
        q.setValue("loc/on", self.loc_box.isChecked())
        q.setValue("loc/lat", self.lat.value())
        q.setValue("loc/lon", self.lon.value())
        q.setValue("loc/mode", self.date_mode.currentIndex())
        q.setValue("loc/date", self.date_edit.date().toString("yyyy-MM-dd"))
        q.setValue("loc/min_occ", self.min_occ.value())

    def _restore(self):
        q = QSettings("BioTool", "BioTool")
        b = lambda v, d: d if v is None else (v if isinstance(v, bool) else str(v).lower() == "true")
        f = lambda v, d: d if v is None else float(v)
        old = int(f(q.value("settings_version"), 0)) < SETTINGS_VERSION
        for k, (cb, th) in self.rows.items():
            cb.setChecked(b(q.value(f"models/{k}/on"), cb.isChecked()) and cb.isEnabled())
            if not (old and k in RETUNED_DEFAULTS):  # keep new defaults that replaced old ones
                th.setValue(f(q.value(f"models/{k}/threshold"), th.value()))
        for k in self.adv:
            for ak in self.adv[k]:
                v = q.value(f"models/{k}/adv/{ak}")
                if v is not None:
                    try:
                        self._adv_set(k, ak, v)
                    except (TypeError, ValueError):
                        pass
        self.indices.setChecked(b(q.value("models/indices/on"), True))
        self.embeddings.setChecked(b(q.value("opt/embeddings"), True))
        self.raven.setChecked(b(q.value("opt/raven"), True))
        self.loc_box.setChecked(b(q.value("loc/on"), False))
        self.lat.setValue(f(q.value("loc/lat"), 0.0))
        self.lon.setValue(f(q.value("loc/lon"), 0.0))
        self.date_mode.setCurrentIndex(int(f(q.value("loc/mode"), 0)))
        d = QDate.fromString(str(q.value("loc/date") or ""), "yyyy-MM-dd")
        if d.isValid():
            self.date_edit.setDate(d)
        self.min_occ.setValue(f(q.value("loc/min_occ"), 0.03))
        self.grid.setValue(f(q.value("opt/grid"), 3.0))
        self.quiet_on.setChecked(b(q.value("opt/quiet_on"), False))
        self.quiet.setValue(f(q.value("opt/quiet"), -70.0))
        self.sl_box.setChecked(b(q.value("sl/on"), False))
        self.sl_mode.setCurrentIndex(int(f(q.value("sl/mode"), 0)))
        self.sl_path.setText(str(q.value("sl/path") or ""))


# ----------------------------------------------------------------------------- single file
class SingleFileTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self.path: str | None = None
        self.result = None
        self.duration = 0.0
        self._thread = None
        self._worker = None

        bar = QHBoxLayout()
        self.open_btn = theme.role(QPushButton("Open audio file..."), "accent")
        self.open_btn.setToolTip("Open a recording (Ctrl+O). You can also drag a file onto the window.")
        self.run_btn = theme.role(QPushButton("Run models"), "primary")
        self.run_btn.setToolTip("Run the ticked models on this file (Ctrl+R)")
        self.run_btn.setEnabled(False)
        self.stop_btn = theme.role(QPushButton("Stop"), "danger")
        self.stop_btn.setEnabled(False)
        self.play_btn = theme.role(QPushButton("Play selection"), "accent")
        self.play_btn.setToolTip("Play / stop the selection (Ctrl+P)")
        self.play_btn.setEnabled(False)
        self.run_settings = None  # settings used for the current result
        self.save_btn = QPushButton("Save results...")
        self.save_btn.setEnabled(False)
        self.analyse_btn = theme.role(QPushButton("Analyse selection"), "primary")
        self.analyse_btn.setToolTip("Run the ticked models on just the selection or box and show what each\n"
                                    "model thinks is there (temporary - nothing is saved). Ctrl+E")
        self.analyse_btn.setEnabled(False)
        self.analyse_band = QCheckBox("filter to box band")
        self.analyse_band.setToolTip("Also band-pass the audio to the box's frequency range before analysis.\n"
                                     "Usually WORSE: the models were trained on unfiltered audio (on a Wood\n"
                                     "Thrush test BirdNET no longer recognised it after filtering to 2-4.5 kHz).")
        self.analyse_band.setEnabled(False)
        self._dialogs = []
        self.file_label = QLabel("No file loaded")
        for w in (self.open_btn, self.run_btn, self.stop_btn, self.play_btn, self.analyse_btn,
                  self.analyse_band, self.save_btn):
            bar.addWidget(w)
        bar.addWidget(self.file_label, 1)
        from .view_controls import ViewControls

        self.view = ViewControls()
        self.S_raw = None
        self.inf = None

        # spectrogram + lanes, sharing the time axis
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setLabel("left", "Frequency", units="kHz")
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_plot.getAxis("left").enableAutoSIPrefix(False)
        self.spec_plot.setXRange(0, 60, padding=0)
        self.spec_plot.setYRange(0, 24, padding=0)
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("viridis"))
        self.spec_plot.addItem(self.img)
        self.region = pg.LinearRegionItem([0, 3], movable=True, brush=(255, 255, 255, 40))
        self.region.setZValue(10)
        self.spec_plot.addItem(self.region)
        self.region.hide()

        self.lanes = pg.PlotWidget()
        self.lanes.setLabel("bottom", "Time", units="s")
        self.lanes.getAxis("bottom").enableAutoSIPrefix(False)
        self.lanes.getAxis("left").setTicks([[]])
        self.lanes.setYRange(-0.6, 0.6, padding=0)
        self.lanes.setXLink(self.spec_plot)
        self.lanes.setMouseEnabled(x=True, y=False)
        self.lanes.setMaximumHeight(170)
        self.lanes.showGrid(x=True, y=False, alpha=0.2)
        legend = "   ".join(f"<span style='color: rgb{c}'>&#9632; {g}</span>" for g, c in GROUP_COLORS.items())
        self.legend = QLabel(f"Detections per model (colour = sound group, opacity = score):  {legend}")

        self.details = QLabel("Click on the spectrogram or the detection lanes to see what every model reported there.")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet("padding: 4px; background: palette(alternate-base);")

        self.det_table = DataFrameView(on_row=self._row_clicked)
        self.agree_table = DataFrameView(on_row=self._row_clicked)
        self.idx_table = DataFrameView()
        self.res_tabs = QTabWidget()
        self.res_tabs.addTab(self.agree_table, "Model agreement (per 3 s window)")
        self.res_tabs.addTab(self.det_table, "All detections")
        self.res_tabs.addTab(self.idx_table, "Acoustic indices (per minute)")

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.spec_plot, 3)
        tl.addWidget(self.legend)
        tl.addWidget(self.lanes, 1)
        tl.addWidget(self.details)
        split = QSplitter(Qt.Vertical)
        split.addWidget(top)
        split.addWidget(self.res_tabs)
        split.setSizes([520, 300])

        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.view)
        lay.addWidget(split, 1)
        self.view.display_changed.connect(self._render)
        self.view.freq_changed.connect(lambda lo, hi: self.spec_plot.setYRange(lo, hi, padding=0))
        self.view.fft_changed.connect(lambda _: self.path and self._launch(run_models=False, keep=True))
        self.view.export_clicked.connect(self.export_clip)
        self.view.box_clicked.connect(self._toggle_box)
        self.box = None

        self.open_btn.clicked.connect(self.open_file)
        self.run_btn.clicked.connect(self.run_models)
        self.stop_btn.clicked.connect(self.stop)
        self.play_btn.clicked.connect(self.play_selection)
        self.save_btn.clicked.connect(self.save_results)
        self.analyse_btn.clicked.connect(self.analyse_selection)
        self.spec_plot.scene().sigMouseClicked.connect(lambda ev: self._plot_clicked(ev, self.spec_plot))
        self.lanes.scene().sigMouseClicked.connect(lambda ev: self._plot_clicked(ev, self.lanes))
        self._player = None
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._poll_playback)

    # -- actions
    def open_file(self, path: str | None = None):
        if not path:
            exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))
            path, _ = QFileDialog.getOpenFileName(self, "Open audio file", self.main.last_dir,
                                                  f"Audio ({exts});;All files (*)")
            if not path:
                return
        if self.main.busy_job:
            QMessageBox.information(self, "Busy", "Wait for the current job to finish (or press Stop).")
            return
        self.main.last_dir = str(Path(path).parent)
        if self._player is not None:
            self._player.stop()
        self.path = path
        self.result = None
        self.run_settings = None
        self.view.box_btn.setChecked(False)
        self.region.hide()
        self.details.setText("Click on the spectrogram or the detection lanes to see what every model reported there.")
        self.det_table.set_frame(pd.DataFrame())
        self.agree_table.set_frame(pd.DataFrame())
        self.idx_table.set_frame(pd.DataFrame())
        self.lanes.clear()
        self.file_label.setText(Path(path).name)
        self.save_btn.setEnabled(False)
        self._launch(run_models=False)

    def run_models(self):
        if self.path and self.S_raw is not None:
            self._launch(run_models=True)

    def _launch(self, run_models: bool, keep: bool = False):
        if self.main.busy_job:
            if not run_models and keep:
                self.main.log("FFT size will apply when the current job has finished - change it again then.")
            return
        settings = self.main.panel.settings()
        if run_models:
            if not settings.models:
                QMessageBox.information(self, "No models", "Tick at least one model on the left.")
                return
            if not self.main.check_location(settings):
                return
            self.run_settings = settings
        self._set_busy(True)
        self._worker = workers.FileWorker(self.path, settings, run_models, nfft=self.view.fft.currentData())
        self._keep_view = keep
        self._worker.log.connect(self.main.log)
        self._worker.spectrogram.connect(self._show_spectrogram)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._thread = workers.start(self._worker, self)

    def stop(self):
        if self._worker:
            self._worker.stop()

    def _set_busy(self, busy: bool):
        self.stop_btn.setEnabled(busy)
        self.main.set_busy(busy)

    def refresh_enabled(self, busy: bool):
        """Called by the main window whenever any job starts/ends."""
        self.open_btn.setEnabled(not busy)
        ready = not busy and self.path is not None and self.S_raw is not None
        self.run_btn.setEnabled(ready)
        self.analyse_btn.setEnabled(ready)
        loaded = self.S_raw is not None
        self.view.box_btn.setEnabled(loaded)
        self.view.export_btn.setEnabled(loaded)

    def _failed(self, msg: str):
        opening = self._worker is not None and not self._worker.run_models
        self._set_busy(False)
        self.main.log("ERROR: " + msg.splitlines()[0])
        if opening:  # the file could not be read: don't leave the previous spectrogram behind
            self.path, self.S_raw, self.inf = None, None, None
            self.img.clear()
            self.play_btn.setEnabled(False)
            self.file_label.setText("Could not open the file - see the log")
            self.main.refresh_enabled()
        QMessageBox.critical(self, "Error", msg[:3000])

    def _show_spectrogram(self, S, inf):
        self.duration = inf.duration_s
        fmax_khz = inf.samplerate / 2000
        keep = getattr(self, "_keep_view", False) and self.S_raw is not None
        xr = self.spec_plot.getPlotItem().vb.viewRange()[0]
        self.S_raw, self.inf = S, inf
        self._base = {}
        self._render()
        self.img.setRect(QRectF(0, 0, inf.duration_s, fmax_khz))
        self.spec_plot.setLimits(xMin=0, xMax=inf.duration_s, yMin=0, yMax=fmax_khz)
        self.view.set_nyquist(fmax_khz)
        if keep:
            self.spec_plot.setXRange(*xr, padding=0)
        else:
            self.spec_plot.setXRange(0, min(inf.duration_s, 60), padding=0)
        self.file_label.setText(f"{Path(self.path).name}  |  {inf.duration_s:.1f} s  |  "
                                f"{inf.samplerate} Hz  |  {inf.channels} ch")
        self.play_btn.setEnabled(True)
        self.main.refresh_enabled()

    def _render(self):
        """Redraw the spectrogram with the current view settings (no recomputation)."""
        if self.S_raw is None:
            return
        den = self.view.denoise.isChecked()
        S = self.S_raw - np.median(self.S_raw, axis=1, keepdims=True) if den else self.S_raw
        if den not in self._base:
            self._base[den] = tuple(np.percentile(S, [40, 99.8] if den else [20, 99.8]))
        self.img.setColorMap(pg.colormap.get(self.view.colormap_name()))
        self.img.setImage(S, autoLevels=False)
        self.img.setLevels(list(self.view.levels(*self._base[den])))
        if self.inf is not None:
            self.img.setRect(QRectF(0, 0, self.inf.duration_s, self.inf.samplerate / 2000))

    def _toggle_box(self, on: bool):
        self.analyse_band.setEnabled(on and self.S_raw is not None)
        if self.box is not None:
            self.spec_plot.removeItem(self.box)
            self.box = None
        if not on or self.S_raw is None:
            return
        s, e = self._selection()
        lo, hi = self.view.freq_range()
        h = hi - lo
        self.box = pg.RectROI([s, lo + 0.15 * h], [max(0.5, e - s), 0.6 * h],
                              pen=pg.mkPen("w", width=2), hoverPen=pg.mkPen("y", width=2),
                              maxBounds=QRectF(0, 0, self.duration, self.inf.samplerate / 2000))
        self.box.addScaleHandle([0, 0], [1, 1])
        self.box.setZValue(20)
        self.spec_plot.addItem(self.box)
        self.region.hide()
        self.main.log("Box: drag it, or resize from the corners. Play/Export now use the box "
                      "(time span and frequency band).")

    def _box_values(self):
        """(start_s, end_s, (low_hz, high_hz)) of the box, or None."""
        if self.box is None:
            return None
        (x, y), (w, h) = self.box.pos(), self.box.size()
        return max(0.0, x), min(self.duration, x + w), (max(0.0, y) * 1000, (y + h) * 1000)

    def _selection(self) -> tuple[float, float]:
        if self.region.isVisible():
            s, e = self.region.getRegion()
        else:  # no selection: what is visible, at most 30 s
            s, e = self.spec_plot.getPlotItem().vb.viewRange()[0]
            e = min(e, s + 30)
        return max(0.0, s), min(self.duration, e)

    def export_clip(self):
        if not self.path:
            return
        from .player import export_clip

        s, e = self._selection()
        band = self.view.playback_band_hz()
        bx = self._box_values()
        if bx:
            s, e, band = bx
        hiss = self.view.hiss_strength()
        tag = f"_bp{band[0] / 1000:.1f}-{band[1] / 1000:.1f}kHz" if band else ""
        tag += f"_hiss{self.view.hiss.currentText().lower()}" if hiss else ""
        default = Path(self.main.last_dir) / f"{Path(self.path).stem}_{s:.1f}-{e:.1f}s{tag}.wav"
        out, _ = QFileDialog.getSaveFileName(self, "Export clip (copy)", str(default), "WAV (*.wav)")
        if not out:
            return
        if Path(out).resolve() == Path(self.path).resolve():
            QMessageBox.warning(self, "Export", "Choose a different file name - the original is never overwritten.")
            return
        export_clip(self.path, s, e, out, band, hiss=hiss)
        self.main.last_dir = str(Path(out).parent)
        self.main.log(f"Exported {s:.1f}-{e:.1f} s ({tag.strip('_') or 'unfiltered'}) to {out}")

    def _done(self, res):
        self._set_busy(False)
        if res is None:
            return
        self.result = res
        self.save_btn.setEnabled(True)
        det = res.detections
        self.det_table.set_frame(det.drop(columns=["file"]))
        grid = self.run_settings.grid_s if self.run_settings else 3.0
        agree = merge.species_agreement(det, grid)
        self.res_tabs.setTabText(0, f"Model agreement (per {grid:g} s window)")
        self.agree_table.set_frame(agree.drop(columns=["file"]))
        self.idx_table.set_frame(res.indices.drop(columns=["file"]) if len(res.indices) else res.indices)
        self._draw_lanes(det)
        self.res_tabs.setCurrentIndex(0)

    def _draw_lanes(self, det: pd.DataFrame):
        self.lanes.clear()
        used = self.run_settings.models if self.run_settings else {}
        models = [k for k in used if k in DETECTORS]
        # every model that ran gets a lane, also when it found nothing (an empty lane is informative)
        self._lane_models = models
        ax = self.lanes.getAxis("left")
        ax.setTicks([[(i, DETECTORS[m].name.split(" (")[0]) for i, m in enumerate(models)]])
        self.lanes.setYRange(-0.6, len(models) - 0.4, padding=0)
        if det.empty:
            return
        best = (det.groupby(["model", "start_s", "end_s", "group"], as_index=False)["score"].max())
        for i, m in enumerate(models):
            d = best[best["model"] == m]
            if d.empty:
                continue
            brushes = [pg.mkBrush(*GROUP_COLORS.get(g, (127, 127, 127)), int(60 + 195 * s))
                       for g, s in zip(d["group"], d["score"])]
            # offset groups vertically inside the lane so overlapping groups stay visible
            off = d["group"].map({"biophony": 0.0, "geophony": -0.27, "anthropophony": 0.27}).fillna(-0.27)
            item = pg.BarGraphItem(x0=d["start_s"].to_numpy(), width=(d["end_s"] - d["start_s"]).to_numpy(),
                                   y=(i + off).to_numpy(), height=0.26, brushes=brushes, pen=None)
            self.lanes.addItem(item)

    def _plot_clicked(self, ev, plot):
        if self.S_raw is None or (self.result is None and plot is self.lanes):
            return
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        t = vb.mapSceneToView(ev.scenePos()).x()
        self.show_window(t)

    def show_window(self, t: float, model: str | None = None, start=None, end=None):
        if start is None:
            g = self.run_settings.grid_s if self.run_settings else 3.0
            start = np.floor(t / g) * g
            end = start + g
        if self.box is not None:  # move the box to this window, keep its frequency band
            (_, y), (_, h) = self.box.pos(), self.box.size()
            self.box.setPos([start, y], update=False)
            self.box.setSize([end - start, h])
        else:
            self.region.setRegion([start, end])
            self.region.show()
        if self.result is None:
            return
        det = self.result.detections
        hit = det[(det["start_s"] < end) & (det["end_s"] > start)]
        if hit.empty:
            self.details.setText(f"<b>{start:.1f}-{end:.1f} s</b>: no detections above threshold.")
            return
        parts = [f"<b>{start:.1f}-{end:.1f} s</b>"]
        for m, d in hit.groupby("model"):
            d = d.sort_values("score", ascending=False).drop_duplicates("label_key").head(5)
            items = ", ".join(
                f"<span style='color: rgb{GROUP_COLORS.get(g, (127, 127, 127))}'>"
                f"{(c if c else s)} {('(<i>' + s + '</i>) ') if (c and s and c != s) else ''}{sc:.2f}</span>"
                for s, c, g, sc in zip(d["scientific_name"], d["common_name"], d["group"], d["score"]))
            parts.append(f"<b>{DETECTORS[m].name}</b>: {items}")
        self.details.setText("<br>".join(parts))

    def _row_clicked(self, row: dict):
        s = row.get("start_s", row.get("win_start_s"))
        e = row.get("end_s", row.get("win_end_s"))
        if s is None:
            return
        s, e = float(s), float(e)
        cur = self.spec_plot.getPlotItem().vb.viewRange()[0]
        width = max(cur[1] - cur[0], 10)
        if not (cur[0] <= s and e <= cur[1]):
            c = (s + e) / 2
            self.spec_plot.setXRange(max(0, c - width / 2), c + width / 2, padding=0)
        self.show_window((s + e) / 2, start=s, end=e)

    def play_selection(self):
        if self._player is None:
            try:
                from .player import SegmentPlayer
            except ImportError as e:
                QMessageBox.information(self, "Playback", f"Audio playback is not available: {e}")
                return
            self._player = SegmentPlayer(self)
        if self._playing():
            self._player.stop()
            return
        s, e = self._selection()
        speed, band, hiss = self.view.playback_speed(), self.view.playback_band_hz(), self.view.hiss_strength()
        bx = self._box_values()
        if bx:
            s, e, band = bx
        try:
            err = self._player.play(self.path, s, e, speed=speed, band=band, hiss=hiss)
        except Exception as ex:  # report instead of crashing
            err = repr(ex)
        if err:
            self.main.log(f"Playback: {err}")
            QMessageBox.warning(self, "Playback", err)
            return
        extra = (f" at {speed:g}x" if speed != 1 else "") + (
            f", band {band[0] / 1000:.1f}-{band[1] / 1000:.1f} kHz" if band else "") + (
            f", hiss reduction {self.view.hiss.currentText().lower()}" if hiss else "")
        self.main.log(f"Playing {s:.1f}-{e:.1f} s{extra}")
        self.play_btn.setText("Stop playback")
        self._stop_timer.start(200)

    def _playing(self) -> bool:
        return self._player is not None and self._player.playing

    def _poll_playback(self):
        if self._playing():
            self._stop_timer.start(200)
        else:
            self.play_btn.setText("Play selection")

    def analyse_selection(self):
        if not self.path or self.S_raw is None or self.main.busy_job:
            return
        settings = self.main.panel.settings()
        if not [k for k in settings.models if k in DETECTORS]:
            QMessageBox.information(self, "Analyse selection", "Tick at least one model on the left.")
            return
        if not self.main.check_location(settings):
            return
        s, e = self._selection()
        band = None
        bx = self._box_values()
        if bx:
            s, e, b = bx
            band = b if self.analyse_band.isChecked() else None
        if e - s < 0.2:
            QMessageBox.information(self, "Analyse selection", "Select a stretch first (click the spectrogram, "
                                    "click a table row, or draw a box).")
            return
        self._sel_span = (s, e, band)
        self._set_busy(True)
        self._worker = workers.SelectionWorker(self.path, settings, s, e, band)
        self._worker.run_models = True  # failures are not 'could not open file'
        self._worker.log.connect(self.main.log)
        self._worker.finished.connect(self._selection_done)
        self._worker.failed.connect(self._failed)
        self._thread = workers.start(self._worker, self)

    def _selection_done(self, res):
        self._set_busy(False)
        df, note = res
        s, e, band = self._sel_span
        title = f"{Path(self.path).name}: {s:.1f}-{e:.1f} s"
        extra = f"band {band[0] / 1000:.1f}-{band[1] / 1000:.1f} kHz (filtered)" if band else "full frequency range"
        dlg = SelectionDialog(self, title, df, note, f"Analysed {extra}.")
        dlg.show()
        self._dialogs = [d for d in self._dialogs if d.isVisible()] + [dlg]
        self.main.log(f"Analysed {s:.1f}-{e:.1f} s: {len(df)} labels from {df['model'].nunique() if len(df) else 0} models")

    def save_results(self):
        if self.result is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose output folder", self.main.last_dir)
        if not out:
            return
        settings = self.run_settings or self.main.panel.settings()
        stem = Path(self.path).stem
        write_file_outputs(self.result, Path(out), stem, settings)
        combine_outputs(Path(out), settings.grid_s)
        self.main.log(f"Saved results to {out}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))


# ----------------------------------------------------------------------------- batch
class BatchTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self._worker = None
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()
        in_btn = QPushButton("Browse...")
        out_btn = QPushButton("Browse...")
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        form = QFormLayout()
        r1 = QHBoxLayout(); r1.addWidget(self.in_edit, 1); r1.addWidget(in_btn)
        r2 = QHBoxLayout(); r2.addWidget(self.out_edit, 1); r2.addWidget(out_btn)
        form.addRow("Recordings folder", r1)
        form.addRow("Output folder", r2)
        form.addRow("", self.recursive)

        self.start_btn = theme.role(QPushButton("Start batch run"), "primary")
        self.stop_btn = theme.role(QPushButton("Stop"), "danger")
        self.stop_btn.setEnabled(False)
        self.open_btn = QPushButton("Open output folder")
        btns = QHBoxLayout()
        for b in (self.start_btn, self.stop_btn, self.open_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        self.progress = QProgressBar()
        self.current = QLabel("")
        note = QLabel("Finished files are skipped when you restart with the same settings, "
                      "so a stopped or interrupted run can simply be started again.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        self.summary = DataFrameView()
        q = QSettings("BioTool", "BioTool")
        self.in_edit.setText(str(q.value("batch/input", "") or ""))
        self.out_edit.setText(str(q.value("batch/output", "") or ""))

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)
        lay.addWidget(note)
        lay.addWidget(self.progress)
        lay.addWidget(self.current)
        lay.addWidget(QLabel("Species / sound summary (after the run):"))
        lay.addWidget(self.summary, 1)

        in_btn.clicked.connect(lambda: self._pick(self.in_edit, True))
        out_btn.clicked.connect(lambda: self._pick(self.out_edit, False))
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(lambda: self._worker and self._worker.stop())
        self.open_btn.clicked.connect(self._open_out)

    def _pick(self, edit: QLineEdit, is_input: bool):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text() or self.main.last_dir)
        if d:
            edit.setText(d)
            self.main.last_dir = d
            if is_input and not self.out_edit.text():
                self.out_edit.setText(str(Path(d).parent / f"{Path(d).name}_biotool"))

    def _open_out(self):
        if self.out_edit.text() and Path(self.out_edit.text()).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_edit.text()))

    def refresh_enabled(self, busy: bool):
        self.start_btn.setEnabled(not busy)

    def start(self):
        if self.main.busy_job:
            return
        inp, out = self.in_edit.text().strip(), self.out_edit.text().strip()
        if not inp or not Path(inp).is_dir():
            QMessageBox.warning(self, "Batch", "Choose an existing recordings folder.")
            return
        if not out:
            QMessageBox.warning(self, "Batch", "Choose an output folder.")
            return
        settings = self.main.panel.settings()
        if not settings.models:
            QMessageBox.information(self, "No models", "Tick at least one model on the left.")
            return
        if not self.main.check_location(settings):
            return
        q = QSettings("BioTool", "BioTool")
        q.setValue("batch/input", inp); q.setValue("batch/output", out)
        self.stop_btn.setEnabled(True)
        self.main.set_busy(True)
        self.progress.setValue(0)
        self.summary.set_frame(pd.DataFrame())
        self.current.setText("Starting (loading models) ...")
        self._worker = workers.FolderWorker(inp, out, settings, self.recursive.isChecked())
        self._worker.log.connect(self.main.log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._failed)
        workers.start(self._worker, self)

    def _on_progress(self, i, n, f):
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(i)
        self.current.setText(f"{i}/{n}  {f}")

    def _done(self, summary):
        self.stop_btn.setEnabled(False)
        self.main.set_busy(False)
        p = Path(self.out_edit.text()) / "species_summary.csv"
        self.main.explore.set_folder(self.out_edit.text())
        if p.exists():
            self.summary.set_frame(pd.read_csv(p, keep_default_na=False))
        if summary:
            self.current.setText(f"Finished: {summary.get('processed', 0)} processed, "
                                 f"{summary.get('skipped', 0)} skipped, {summary.get('failed', 0)} failed")

    def _failed(self, msg):
        self.stop_btn.setEnabled(False)
        self.main.set_busy(False)
        self.main.log("ERROR: " + msg.splitlines()[0])
        QMessageBox.critical(self, "Error", msg[:3000])


# ----------------------------------------------------------------------------- window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"BioTool {__version__} - multi-model bioacoustics")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        q = QSettings("BioTool", "BioTool")
        self.last_dir = str(q.value("ui/last_dir", "") or Path.home())
        self.busy_job = False
        self.theme_name = str(q.value("ui/theme", theme.DEFAULT) or theme.DEFAULT)
        app = QApplication.instance()
        self.theme = theme.apply(app, self.theme_name) if app else theme.THEMES[theme.DEFAULT]

        self.panel = ModelPanel()
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(460)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { padding: 6px 14px; font-size: 10pt; }")
        self.single = SingleFileTab(self)
        self.batch = BatchTab(self)
        from .explore_tab import ExploreTab

        self.explore = ExploreTab(self)
        self.tabs.addTab(self.single, "Single file")
        self.tabs.addTab(self.batch, "Batch (folder)")
        self.tabs.addTab(self.explore, "Explore sounds")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMaximumHeight(130)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.tabs)
        right.addWidget(self.log_box)
        right.setSizes([760, 130])
        main = QSplitter(Qt.Horizontal)
        main.addWidget(scroll)
        main.addWidget(right)
        main.setSizes([385, 1015])
        self.setCentralWidget(main)

        self.busy = QProgressBar()
        self.busy.setMaximumWidth(160)
        self.busy.setRange(0, 0)
        self.busy.hide()
        self.statusBar().addPermanentWidget(self.busy)
        self.statusBar().showMessage("Ready. Models download automatically the first time they are used.")

        m = self.menuBar().addMenu("&File")
        a = QAction("Open audio file...", self)
        a.setShortcut(QKeySequence("Ctrl+O"))
        a.triggered.connect(lambda: (self.tabs.setCurrentIndex(0), self.single.open_file()))
        m.addAction(a)
        q_ = QAction("Quit", self)
        q_.setShortcut(QKeySequence("Ctrl+Q"))
        q_.triggered.connect(self.close)
        m.addAction(q_)
        v = self.menuBar().addMenu("&View")
        tm = v.addMenu("Theme")
        grp = QActionGroup(self)
        for name in theme.THEMES:
            act = QAction(name, self, checkable=True)
            act.setChecked(name == self.theme.name)
            act.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            grp.addAction(act)
            tm.addAction(act)
        h = self.menuBar().addMenu("&Help")
        keys = QAction("Keyboard shortcuts", self)
        keys.triggered.connect(lambda: QMessageBox.information(
            self, "Keyboard shortcuts",
            "Ctrl+O  open audio file\nCtrl+R  run models on the open file\nCtrl+P  play / stop selection\n"
            "Ctrl+E  analyse the selection / box\n"
            "Ctrl+Q  quit\n\nDrag an audio file onto the window to open it,\n"
            "or a folder to use it as the batch recordings folder."))
        h.addAction(keys)
        about = QAction("About models && licences", self)
        about.triggered.connect(self._about)
        h.addAction(about)

        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._shortcut_run)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self._shortcut_play)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=lambda: (self.tabs.setCurrentIndex(0),
                                                                  self.single.analyse_selection()))

        geo = q.value("ui/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        theme.style_plots(self, self.theme)
        self.refresh_enabled()

    # -- jobs: only one model / clustering job at a time (models are shared between tabs)
    def set_busy(self, busy: bool):
        # The left panel stays enabled (no flicker): every job takes a copy of the settings when it
        # starts, so changing them while it runs only affects the next run.
        self.busy_job = busy
        self.busy.setVisible(busy)
        self.refresh_enabled()

    def refresh_enabled(self):
        for tab in (self.single, self.batch, self.explore):
            tab.refresh_enabled(self.busy_job)

    def check_location(self, settings) -> bool:
        sl = settings.species_list
        if sl.get("enabled"):
            p = Path(sl.get("path", ""))
            if not p.is_file():
                QMessageBox.warning(self, "Species list", f"Species list file not found:\n{p}\n\n"
                                    "Choose the file on the left, or untick 'Species list'.")
                return False
            from ..species_list import read_names
            if not read_names(p):
                QMessageBox.warning(self, "Species list", f"No species names found in {p.name}.")
                return False
        loc = settings.location
        if loc.get("enabled") and abs(loc.get("lat", 0)) < 1e-6 and abs(loc.get("lon", 0)) < 1e-6:
            QMessageBox.warning(self, "Location filter",
                                "The location filter is on but latitude/longitude are 0, 0 (in the Atlantic "
                                "Ocean) - almost every species would be removed.\n\nEnter the site "
                                "coordinates on the left, or untick 'Location / season filter'.")
            return False
        return True

    def _shortcut_run(self):
        self.tabs.setCurrentIndex(0)
        self.single.run_models()

    def _shortcut_play(self):
        if self.tabs.currentIndex() == 2:
            self.explore._play_window()
        elif self.single.play_btn.isEnabled():
            self.single.play_selection()

    # -- theme
    def set_theme(self, name: str):
        app = QApplication.instance()
        self.theme = theme.apply(app, name)
        self.theme_name = name
        theme.style_plots(self, self.theme)
        QSettings("BioTool", "BioTool").setValue("ui/theme", name)

    # -- drag & drop
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if not urls:
            return
        p = Path(urls[0])
        if p.is_dir():
            self.tabs.setCurrentIndex(1)
            self.batch.in_edit.setText(str(p))
            if not self.batch.out_edit.text():
                self.batch.out_edit.setText(str(p.parent / f"{p.name}_biotool"))
        elif p.suffix.lower() in AUDIO_EXTENSIONS:
            self.tabs.setCurrentIndex(0)
            self.single.open_file(str(p))
        else:
            self.log(f"Not an audio file or folder: {p.name}")

    # -- closing
    def closeEvent(self, ev):
        if self.busy_job:
            r = QMessageBox.question(self, "Quit", "A job is still running. Stop it and quit?")
            if r != QMessageBox.Yes:
                ev.ignore()
                return
            for tab in (self.single, self.batch, self.explore):
                w = getattr(tab, "_worker", None)
                if w is not None and hasattr(w, "stop"):
                    w.stop()
            for th in self.findChildren(QThread):
                if th.isRunning():
                    th.quit()
                    th.wait(5000)
        q = QSettings("BioTool", "BioTool")
        q.setValue("ui/geometry", self.saveGeometry())
        q.setValue("ui/last_dir", self.last_dir)
        try:
            self.panel._save()
        except Exception:
            pass
        ev.accept()

    def log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.statusBar().showMessage(msg.splitlines()[0][:200])

    def _about(self):
        lines = [f"<b>{c.name}</b>: {c.description}<br><i>{c.licence_note}</i>" for c in DETECTORS.values()]
        lines.append(f"<b>{AcousticIndices.name}</b>: {AcousticIndices.description}")
        lines.append("<br>Indices:<br>" + "<br>".join(f"<b>{k}</b> - {v}" for k, v in INDEX_HELP.items()))
        QMessageBox.about(self, "Models and licences", "<br><br>".join(lines))
