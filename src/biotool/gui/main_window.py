"""Main window: model panel on the left, Single-file and Batch tabs on the right."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QRectF, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QInputDialog,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QToolButton,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import __version__, merge
from ..audio import AUDIO_EXTENSIONS
from ..models import DETECTORS, AcousticIndices
from ..models.indices import INDEX_HELP
from ..location import LocationSettings
from ..pipeline import RunSettings, read_file_result, read_run, save_single_run
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
def default_results_root() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "BioTool results"


def _muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName("muted")
    lab.setStyleSheet("font-size: 9pt;")
    return lab


class ModelPanel(QWidget):
    """Left side: models, outputs and where results go. The rarely changed settings live on
    `advanced_page` (shown as the Advanced settings tab); the panel summarises which of them differ
    from the defaults."""

    open_advanced = Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.rows = {}
        self.adv: dict[str, dict] = {}
        self._loading = True

        # ---- 1. models
        step1 = QLabel("1. Choose models"); step1.setObjectName("step")
        lay.addWidget(step1)
        box = QGroupBox()
        form = QVBoxLayout(box)
        for key, cls in DETECTORS.items():
            cb = QCheckBox(cls.name)
            cb.setChecked(key in ("birdnet", "perch_v2", "yamnet"))  # optional models start unticked
            ok, why = cls.availability()
            tip = f"{cls.description}\n{cls.licence_note}"
            if cls.advanced:
                tip += "\n\nMore parameters: Advanced settings tab."
            cb.setToolTip(tip)
            th = QDoubleSpinBox()
            th.setDecimals(3)
            th.setRange(0.001, 0.99)
            th.setSingleStep(0.01 if cls.default_threshold < 0.1 else 0.05)
            th.setValue(cls.default_threshold)
            th.setToolTip(f"Minimum score to report a detection (default {cls.default_threshold:g}).\n"
                          "Lower = more detections but more false ones.")
            if key == "dfs":
                th.setToolTip("Used only when Advanced settings > DeepForestSound > Thresholds = 'min score above'.\n"
                              "By default DFS uses the per-species thresholds from its paper.")
            r = QHBoxLayout()
            r.addWidget(cb, 1)
            lab = QLabel("min score"); lab.setToolTip(th.toolTip())
            r.addWidget(lab)
            r.addWidget(th)
            form.addLayout(r)
            if not ok:
                form.addWidget(_muted(f"<b>{why}</b>"))
                cb.setChecked(False)
                cb.setEnabled(False)
                th.setEnabled(False)
            self.rows[key] = (cb, th)
        self.indices = QCheckBox(AcousticIndices.name)
        self.indices.setChecked(True)
        self.indices.setToolTip(AcousticIndices.description + "\n\n"
                                + "\n".join(f"{k}: {v}" for k, v in INDEX_HELP.items()))
        form.addWidget(self.indices)
        lay.addWidget(box)

        # ---- 2. outputs
        step2 = QLabel("2. Extra outputs"); step2.setObjectName("step")
        lay.addWidget(step2)
        obox = QGroupBox()
        of = QVBoxLayout(obox)
        self.events = QCheckBox("Sound intervals (start/end per sound)")
        self.events.setToolTip(
            "Join consecutive windows with the same label into one interval per sound, so you can see\n"
            "how long it lasted. Window models give the position to about 0.5 s and the start/end to\n"
            "about 1 s (measured on the Powdermill annotations); sounds shorter than a window come out\n"
            "about one window long. If BirdCODE also ran, bird intervals take BirdCODE's precise times.\n"
            "Fast: computed from the detections, the models do not run again.\n"
            "Results: 'Sound intervals' tab, and the lanes under the spectrogram show intervals.")
        self.events_spec = QCheckBox("+ frequency band from the spectrogram")
        self.events_spec.setToolTip(
            "Find each sound's frequency band (low/high Hz) from the loud regions of the spectrogram\n"
            "(scikit-maad segmentation) and trim interval edges where the sound is clearly shorter.\n"
            "On the Powdermill annotations: band within ~0.7 kHz (low) / ~1.9 kHz (high); edges improved\n"
            "slightly (1.00 -> 0.93 s). Adds ~0.6 s per minute of audio. Overlapping sounds (choruses,\n"
            "rain, insects) can widen the band.")
        self.events_spec.setStyleSheet("margin-left: 18px;")
        self.events_spec.setEnabled(False)
        self.events.toggled.connect(self.events_spec.setEnabled)
        of.addWidget(self.events)
        of.addWidget(self.events_spec)
        lay.addWidget(obox)

        # ---- 3. results folder
        step3 = QLabel("3. Results are saved to"); step3.setObjectName("step")
        lay.addWidget(step3)
        rbox = QGroupBox()
        rf = QVBoxLayout(rbox)
        self.results_root = QLineEdit(str(default_results_root()))
        self.results_root.setToolTip("Every run of the models is saved here, in its own folder named\n"
                                     "<date>_<time>_<file name>. Batch runs go to the output folder you choose.")
        rb = QPushButton("Change..."); rb.setToolTip("Choose the folder where runs are saved")
        ro = QPushButton("Open"); ro.setToolTip("Open the results folder in the file explorer")
        rb.clicked.connect(self._browse_results_root)
        ro.clicked.connect(self._open_results_root)
        rr = QHBoxLayout(); rr.addWidget(self.results_root, 1); rr.addWidget(rb); rr.addWidget(ro)
        self.autosave = QCheckBox("Save every run automatically")
        self.autosave.setChecked(True)
        self.autosave.setToolTip("Write the tables (CSV, Raven, embeddings) of every model run to a new folder\n"
                                 "in the results folder. Load them later with 'Load results...'.")
        rf.addLayout(rr)
        rf.addWidget(self.autosave)
        lay.addWidget(rbox)

        # ---- advanced summary
        self.adv_summary = _muted("")
        self.adv_summary.setTextFormat(Qt.RichText)
        adv_btn = QPushButton("Advanced settings...")
        adv_btn.setToolTip("Model parameters, location / season filter, species list, CPU threads,\n"
                           "embeddings, Raven tables and other processing options")
        adv_btn.clicked.connect(self.open_advanced.emit)
        lay.addWidget(self.adv_summary)
        lay.addWidget(adv_btn)
        lay.addStretch(1)

        self.advanced_page = self._build_advanced_page()
        self._restore()
        self._loading = False
        self._watch(self)
        self._watch(self.advanced_page)
        self.update_summary()

    # -- advanced settings page
    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        intro = QLabel("These settings apply to every run (single file, batch and Analyse selection). "
                       "The defaults are what the model authors use or what BioTool's evaluation "
                       "found best; hover over a setting for what it does. Changed settings are listed "
                       "on the left, under the results folder.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        cols = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        lay.addLayout(cols)

        mp = QGroupBox("Model parameters")
        mpl = QVBoxLayout(mp)
        for key, cls in list(DETECTORS.items()) + [("indices", AcousticIndices)]:
            self.adv.setdefault(key, {})
            if not cls.advanced:
                continue
            g = QGroupBox(cls.name)
            g.setToolTip(cls.description)
            f = QFormLayout(g)
            self._add_advanced(f, key, cls)
            mpl.addWidget(g)
        mpl.addStretch(1)
        left.addWidget(mp)

        proc = QGroupBox("Processing")
        f2 = QFormLayout(proc)
        self.threads = QSpinBox()
        self.threads.setRange(1, max(1, os.cpu_count() or 1))
        self.threads.setValue(max(1, (os.cpu_count() or 2) - 1))
        self.threads.setToolTip("CPU threads used by the models (default: all but one)")
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
        f2.addRow("CPU threads", self.threads)
        f2.addRow("Agreement grid", self.grid)
        f2.addRow(qrow)
        right.addWidget(proc)

        outs = QGroupBox("Saved outputs")
        f5 = QVBoxLayout(outs)
        self.embeddings = QCheckBox("Save embeddings")
        self.embeddings.setChecked(True)
        self.embeddings.setToolTip("Feature vectors per window - needed for the Explore sounds tab\n"
                                   "(clustering unknown sounds) and for training custom classifiers later.")
        self.raven = QCheckBox("Write Raven selection tables")
        self.raven.setChecked(True)
        self.raven.setToolTip("Write a Raven Pro selection table per file (folder 'raven' in the run),\n"
                              "to check detections in Raven or Audacity.")
        f5.addWidget(self.embeddings)
        f5.addWidget(self.raven)
        right.addWidget(outs)

        # custom species list
        self.sl_box = QGroupBox("Species list")
        self.sl_box.setCheckable(True)
        self.sl_box.setChecked(False)
        self.sl_box.setToolTip("Keep only (or remove) the species in a text file - BirdNET, Perch, DFS and BirdCODE.")
        f4 = QFormLayout(self.sl_box)
        self.sl_mode = QComboBox(); self.sl_mode.addItems(["Keep only these species", "Remove these species"])
        self.sl_path = QLineEdit(); self.sl_path.setPlaceholderText("species_list.txt")
        self.sl_path.setToolTip("Text file with one species per line")
        sl_browse = QPushButton("..."); sl_browse.setFixedWidth(32)
        sl_browse.setToolTip("Choose the species list file")
        sl_browse.clicked.connect(self._browse_species_list)
        prow = QHBoxLayout(); prow.addWidget(self.sl_path, 1); prow.addWidget(sl_browse)
        f4.addRow(self.sl_mode)
        f4.addRow("File", prow)
        f4.addRow(_muted("One species per line: scientific name, English name or BirdNET's "
                         "'Scientific_Common' format. Lines starting with # are ignored."))
        right.addWidget(self.sl_box)

        # location / season filter
        self.loc_box = QGroupBox("Location / season filter")
        self.loc_box.setCheckable(True)
        self.loc_box.setChecked(False)
        self.loc_box.setToolTip("Drop species that BirdNET's range model considers unlikely at this site\n"
                                "and time of year (BirdNET, Perch, DFS, BirdCODE).")
        f3 = QFormLayout(self.loc_box)
        self.lat = QDoubleSpinBox(); self.lat.setRange(-90, 90); self.lat.setDecimals(5)
        self.lon = QDoubleSpinBox(); self.lon.setRange(-180, 180); self.lon.setDecimals(5)
        self.lat.setToolTip("Latitude of the recording site (decimal degrees)")
        self.lon.setToolTip("Longitude of the recording site (decimal degrees)")
        self.date_mode = QComboBox()
        self.date_mode.addItems(["Date from file name / AudioMoth", "Fixed date", "All year"])
        self.date_mode.setToolTip("Which week of the year to use for the species list")
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
        f3.addRow(_muted("Files without a date in their name use the all-year species list."))
        right.addWidget(self.loc_box)
        reset = QPushButton("Reset all settings to defaults")
        reset.setToolTip("Min scores, model parameters, processing and saved-output options back to their\n"
                         "defaults. Kept: which models are ticked, extra outputs, the location filter and\n"
                         "species list (on/off and their values) and the results folder.")
        reset.clicked.connect(self.reset_defaults)
        right.addWidget(reset)
        right.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        return scroll

    def _add_advanced(self, f: QFormLayout, key, cls):
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
            tip = spec.get("help", "") + f"\nDefault: {spec['default']}"
            w.setToolTip(tip.strip())
            lab = QLabel(spec["label"])
            lab.setToolTip(tip.strip())
            f.addRow(lab, w)
            self.adv[key][spec["key"]] = (spec, w)

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

    # -- summary of what differs from the defaults
    def _watch(self, root: QWidget):
        for w in root.findChildren(QWidget):
            for sig in ("toggled", "valueChanged", "currentIndexChanged", "textChanged", "dateChanged"):
                if hasattr(w, sig) and not isinstance(w, (QScrollArea, QLabel)):
                    try:
                        getattr(w, sig).connect(self.update_summary)
                    except (TypeError, RuntimeError, AttributeError):
                        pass

    def changed_advanced(self) -> list[str]:
        out = []
        for key, items in self.adv.items():
            name = (DETECTORS[key].name if key in DETECTORS else AcousticIndices.name).split(" (")[0]
            for k, (spec, _) in items.items():
                v = self._adv_values(key)[k]
                d = spec["default"]
                same = (abs(float(v) - float(d)) < 1e-9) if spec["kind"] in ("float", "int") else str(v) == str(d)
                if not same:
                    out.append(f"{name}: {spec['label'].split(' (')[0].lower()} {v}")
        if self.loc_box.isChecked():
            out.append(f"location filter on ({self.lat.value():.2f}, {self.lon.value():.2f})")
        if self.sl_box.isChecked():
            out.append(f"species list: {self.sl_mode.currentText().lower()}")
        if self.quiet_on.isChecked():
            out.append(f"skip windows below {self.quiet.value():.0f} dBFS")
        if abs(self.grid.value() - 3.0) > 1e-9:
            out.append(f"agreement grid {self.grid.value():g} s")
        if not self.embeddings.isChecked():
            out.append("embeddings not saved")
        if not self.raven.isChecked():
            out.append("no Raven tables")
        return out

    def update_summary(self, *_):
        if self._loading:
            return
        ch = self.changed_advanced()
        if ch:
            self.adv_summary.setText("<b>Advanced settings changed:</b><br>" + "<br>".join(f"&bull; {c}" for c in ch))
        else:
            self.adv_summary.setText("Advanced settings: all at their defaults.")

    def reset_defaults(self):
        for key, cls in DETECTORS.items():
            self.rows[key][1].setValue(cls.default_threshold)
        for key in self.adv:
            for k, (spec, _) in self.adv[key].items():
                self._adv_set(key, k, spec["default"])
        self.min_occ.setValue(0.03)
        self.grid.setValue(3.0)
        self.quiet_on.setChecked(False); self.quiet.setValue(-70)
        self.embeddings.setChecked(True)
        self.raven.setChecked(True)
        self.threads.setValue(max(1, (os.cpu_count() or 2) - 1))
        self._save()
        self.update_summary()

    def _browse_species_list(self):
        p, _ = QFileDialog.getOpenFileName(self, "Species list", self.sl_path.text() or str(Path.home()),
                                           "Text files (*.txt *.csv);;All files (*)")
        if p:
            self.sl_path.setText(p)

    def _browse_results_root(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for saved runs", self.results_root.text())
        if d:
            self.results_root.setText(d)
            self._save()

    def _open_results_root(self):
        p = Path(self.results_root.text())
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def results_dir(self) -> Path:
        return Path(self.results_root.text().strip() or default_results_root())

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
                           quiet_dbfs=round(self.quiet.value(), 1) if self.quiet_on.isChecked() else None,
                           events=self.events.isChecked(),
                           events_spectrogram=self.events.isChecked() and self.events_spec.isChecked())

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
        q.setValue("opt/events", self.events.isChecked())
        q.setValue("opt/events_spec", self.events_spec.isChecked())
        q.setValue("opt/threads", self.threads.value())
        q.setValue("runs/root", self.results_root.text())
        q.setValue("runs/autosave", self.autosave.isChecked())
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
        self.events.setChecked(b(q.value("opt/events"), False))
        self.events_spec.setChecked(b(q.value("opt/events_spec"), False))
        if q.value("opt/threads") is not None:
            self.threads.setValue(int(f(q.value("opt/threads"), self.threads.value())))
        self.results_root.setText(str(q.value("runs/root") or default_results_root()))
        self.autosave.setChecked(b(q.value("runs/autosave"), True))
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

        big = "QPushButton { min-height: 30px; font-size: 10.5pt; padding: 4px 14px; }"
        self.open_btn = theme.role(QPushButton("Open audio file..."), "accent")
        self.open_btn.setToolTip("Open a recording to see its spectrogram (Ctrl+O).\n"
                                 "You can also drag a file onto the window.")
        self.run_btn = theme.role(QPushButton("Run models"), "primary")
        self.run_btn.setToolTip("Run the ticked models on the whole file (Ctrl+R).\n"
                                "The results are saved automatically to the results folder (left panel).")
        self.run_btn.setEnabled(False)
        self.stop_btn = theme.role(QPushButton("Stop"), "danger")
        self.stop_btn.setToolTip("Stop the running job (nothing is saved for a stopped run)")
        self.stop_btn.setEnabled(False)
        self.load_btn = QPushButton("Load results...")
        self.load_btn.setToolTip("Open the results of an earlier run (single file or batch) without running\n"
                                 "the models again (Ctrl+L). The recording is opened with them.")
        self.save_btn = QPushButton("Open run folder")
        self.save_btn.setToolTip("Open the folder where this run's tables were saved")
        self.save_btn.setEnabled(False)
        for w in (self.open_btn, self.run_btn, self.stop_btn, self.load_btn, self.save_btn):
            w.setStyleSheet(big) if not w.objectName() else w.setMinimumHeight(34)
        self.run_settings = None  # settings used for the current result
        self.run_dir: Path | None = None  # where the current result is saved
        self._pending = None  # loaded result waiting for its spectrogram
        self.file_label = QLabel("No file loaded - open a recording or drag one onto the window")
        bar = QHBoxLayout()
        for w in (self.open_btn, self.run_btn, self.stop_btn):
            bar.addWidget(w)
        bar.addSpacing(16)
        for w in (self.load_btn, self.save_btn):
            bar.addWidget(w)
        bar.addSpacing(12)
        bar.addWidget(self.file_label, 1)

        self.play_btn = theme.role(QPushButton("Play selection"), "accent")
        self.play_btn.setToolTip("Play / stop the selected stretch or box (Ctrl+P).\n"
                                 "Click the spectrogram or a table row to select a stretch.")
        self.play_btn.setEnabled(False)
        self.analyse_btn = QPushButton("Analyse selection")
        self.analyse_btn.setToolTip("Run the ticked models on just the selection or box and show what each\n"
                                    "model thinks is there (temporary - nothing is saved). Ctrl+E")
        self.analyse_btn.setEnabled(False)
        self.analyse_band = QCheckBox("filter to box band")
        self.analyse_band.setToolTip("Also band-pass the audio to the box's frequency range before analysis.\n"
                                     "Usually WORSE: the models were trained on unfiltered audio (on a Wood\n"
                                     "Thrush test BirdNET no longer recognised it after filtering to 2-4.5 kHz).")
        self.analyse_band.setEnabled(False)
        self.analyse_band.setVisible(False)
        self._dialogs = []
        from .view_controls import ViewControls

        self.view = ViewControls()
        self.view.setVisible(False)
        self.view_btn = QToolButton()
        self.view_btn.setText("View && listening options")
        self.view_btn.setCheckable(True)
        self.view_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.view_btn.setArrowType(Qt.RightArrow)
        self.view_btn.setToolTip("Colours, contrast, frequency range and FFT size of the spectrogram;\n"
                                 "playback speed, hiss reduction and band filter for listening")
        self.view_btn.toggled.connect(lambda on: (self.view.setVisible(on),
                                                  self.view_btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow),
                                                  QSettings("BioTool", "BioTool").setValue("ui/view_open", on)))
        sel = QHBoxLayout()
        sel.addWidget(QLabel("<b>Selection:</b>"))
        self.view.box_btn.setObjectName("")  # secondary tool: normal button style
        for w in (self.play_btn, self.analyse_btn, self.analyse_band, self.view.box_btn, self.view.export_btn):
            sel.addWidget(w)
        sel.addStretch(1)
        sel.addWidget(self.view_btn)
        self.view_btn.setChecked(str(QSettings("BioTool", "BioTool").value("ui/view_open", "false")).lower() == "true")
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
        self._legend_html = legend
        self.legend = QLabel(f"Detections per model (colour = sound group, opacity = score):  {legend}")

        self.details = QLabel("Click on the spectrogram or the detection lanes to see what every model reported there.")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet("padding: 4px; background: palette(alternate-base);")

        self.det_table = DataFrameView(on_row=self._row_clicked)
        self.agree_table = DataFrameView(on_row=self._row_clicked)
        self.idx_table = DataFrameView()
        self.ev_table = DataFrameView(on_row=self._row_clicked)
        self.res_tabs = QTabWidget()
        self.res_tabs.addTab(self.agree_table, "Model agreement (per 3 s window)")
        self.res_tabs.addTab(self.det_table, "All detections")
        self.res_tabs.addTab(self.idx_table, "Acoustic indices (per minute)")
        self.ev_tab = self.res_tabs.addTab(self.ev_table, "Sound intervals")
        self.res_tabs.setTabVisible(self.ev_tab, False)

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
        lay.addLayout(sel)
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
        self.save_btn.clicked.connect(self.open_run_folder)
        self.load_btn.clicked.connect(lambda: self.load_results())
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
        self.ev_table.set_frame(pd.DataFrame())
        self.res_tabs.setTabVisible(self.ev_tab, False)
        self.lanes.clear()
        self.file_label.setText(Path(path).name)
        self.run_dir = None
        self._pending = None
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
        self.load_btn.setEnabled(not busy)
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
        self.analyse_band.setVisible(on)
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
            if self._pending is not None and self.S_raw is not None:  # spectrogram of a loaded run is ready
                res, settings, run_dir = self._pending
                self._pending = None
                self.run_settings, self.run_dir = settings, run_dir
                self._show_result(res)
                self.save_btn.setEnabled(True)
                self.file_label.setText(self.file_label.text() + f"  |  results loaded from {run_dir.name}")
                self.main.log(f"Loaded results from {run_dir}")
            return
        self._show_result(res)
        self.run_dir = None
        if self.main.panel.autosave.isChecked():
            try:
                self.run_dir = save_single_run(res, self.main.panel.results_dir(), self.run_settings)
                self.main.log(f"Saved to {self.run_dir}")
                self.main.add_recent(self.run_dir)
                if self.run_settings.save_embeddings and res.embeddings:
                    self.main.explore.set_folder(str(self.run_dir))
            except Exception as e:
                self.main.log(f"ERROR: could not save the results to {self.main.panel.results_dir()}: {e}")
                QMessageBox.warning(self, "Save results", f"Could not save the results:\n{e}\n\n"
                                    "Choose another results folder on the left (3. Results are saved to).")
        else:
            self.main.log("Not saved (automatic saving is off) - use 'Save results...' to keep them.")
        self.save_btn.setText("Open run folder" if self.run_dir else "Save results...")
        self.save_btn.setToolTip("Open the folder where this run's tables were saved" if self.run_dir else
                                 "Save this run's tables to a new folder in the results folder")
        self.save_btn.setEnabled(True)

    def _show_result(self, res):
        self.result = res
        det = res.detections
        self.det_table.set_frame(det.drop(columns=["file"], errors="ignore"))
        grid = self.run_settings.grid_s if self.run_settings else 3.0
        agree = merge.species_agreement(det, grid)
        self.res_tabs.setTabText(0, f"Model agreement (per {grid:g} s window)")
        self.agree_table.set_frame(agree.drop(columns=["file"], errors="ignore"))
        self.idx_table.set_frame(res.indices.drop(columns=["file"], errors="ignore") if len(res.indices) else res.indices)
        ev = getattr(res, "events", None)
        self.res_tabs.setTabVisible(self.ev_tab, ev is not None)
        if ev is not None:
            self.ev_table.set_frame(ev.drop(columns=["file"], errors="ignore"))
            lanes = ev.rename(columns={"max_score": "score"})
        else:
            lanes = det
        what = "Sound intervals" if ev is not None else "Detections"
        self.legend.setText(f"{what} per model (colour = sound group, opacity = score):  {self._legend_html}")
        self._draw_lanes(lanes)
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

    def open_run_folder(self):
        if self.result is None:
            return
        if self.run_dir is None:  # automatic saving was off: save now
            try:
                self.run_dir = save_single_run(self.result, self.main.panel.results_dir(),
                                               self.run_settings or self.main.panel.settings())
            except Exception as e:
                QMessageBox.warning(self, "Save results", f"Could not save the results:\n{e}")
                return
            self.main.add_recent(self.run_dir)
            self.main.log(f"Saved to {self.run_dir}")
            self.save_btn.setText("Open run folder")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_dir)))

    def load_results(self, run_dir: str | None = None):
        if self.main.busy_job:
            QMessageBox.information(self, "Busy", "Wait for the current job to finish (or press Stop).")
            return
        if not run_dir:
            start = self.main.panel.results_dir()
            run_dir = QFileDialog.getExistingDirectory(self, "Choose the folder of a saved run",
                                                       str(start if start.exists() else self.main.last_dir))
            if not run_dir:
                return
        run_dir = Path(run_dir)
        settings, files = read_run(run_dir)
        if not files:
            QMessageBox.information(self, "Load results", f"No saved results found in\n{run_dir}\n\n"
                                    "Choose a run folder (it contains run_config.json and a 'per_file' folder).")
            return
        entry = files[0]
        if len(files) > 1:  # batch run: choose one recording
            names = [f"{Path(f['file']).name or f['stem']}" for f in files]
            name, ok = QInputDialog.getItem(self, "Load results", f"{len(files)} recordings in this run - "
                                            "which one should be shown?", names, 0, False)
            if not ok:
                return
            entry = files[names.index(name)]
        audio = Path(entry["file"]) if entry.get("file") else None
        if audio is None or not audio.is_file():
            QMessageBox.information(self, "Load results", f"The recording of this run was not found:\n{audio}\n\n"
                                    "Please point to it (it may have been moved or be on another computer).")
            exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))
            p, _ = QFileDialog.getOpenFileName(self, f"Where is {audio.name if audio else 'the recording'}?",
                                               self.main.last_dir, f"Audio ({exts});;All files (*)")
            if not p:
                return
            audio = Path(p)
        res = read_file_result(run_dir, entry, audio)
        if settings is None:  # older run without run_config.json: lanes for the models found in it
            settings = RunSettings(models={m: {} for m in res.detections["model"].unique()})
        self.main.tabs.setCurrentIndex(0)
        self.open_file(str(audio))
        if self.path == str(audio):
            self._pending = (res, settings, run_dir)
            self.main.add_recent(run_dir)


# ----------------------------------------------------------------------------- batch
class BatchTab(QWidget):
    def __init__(self, main: "MainWindow"):
        super().__init__()
        self.main = main
        self._worker = None
        self.in_edit = QLineEdit()
        self.out_edit = QLineEdit()
        in_btn = QPushButton("Browse...")
        in_btn.setToolTip("Choose the folder with the recordings")
        out_btn = QPushButton("Browse...")
        out_btn.setToolTip("Choose where the results of this batch are written")
        self.in_edit.setToolTip("Folder with the recordings (you can also drag a folder onto the window)")
        self.out_edit.setToolTip("Results of this batch. Keep the same folder to resume a stopped run:\n"
                                 "finished files are skipped.")
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setToolTip("Also process recordings in subfolders")
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
        self.start_btn.setToolTip("Run the ticked models on every recording in the folder")
        self.stop_btn.setToolTip("Stop after the current file; start again later to resume")
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.setToolTip("Open the output folder in the file explorer")
        self.load_btn = QPushButton("Show a file's results...")
        self.load_btn.setToolTip("Pick one recording of this batch and show its spectrogram and results\n"
                                 "in the Single file tab")
        self.load_btn.clicked.connect(lambda: self.out_edit.text() and self.main.single.load_results(self.out_edit.text()))
        btns = QHBoxLayout()
        for b in (self.start_btn, self.stop_btn, self.open_btn, self.load_btn):
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
            if is_input:
                self.out_edit.setText(str(self.main.panel.results_dir() / f"batch_{Path(d).name}"))

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
        self.main.add_recent(self.out_edit.text())
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
        self.adv_tab = self.tabs.addTab(self.panel.advanced_page, "Advanced settings")
        self.tabs.setTabToolTip(0, "Look at and listen to one recording, run the models on it")
        self.tabs.setTabToolTip(1, "Run the models on every recording in a folder (resumable)")
        self.tabs.setTabToolTip(2, "Group similar sounds using the saved embeddings")
        self.tabs.setTabToolTip(self.adv_tab, "Model parameters, filters and processing options")
        self.panel.open_advanced.connect(lambda: self.tabs.setCurrentIndex(self.adv_tab))

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
        lr = QAction("Load results of a previous run...", self)
        lr.setShortcut(QKeySequence("Ctrl+L"))
        lr.triggered.connect(lambda: self.single.load_results())
        m.addAction(lr)
        self.recent_menu = m.addMenu("Recent results")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        orf = QAction("Open results folder", self)
        orf.triggered.connect(self.panel._open_results_root)
        m.addAction(orf)
        m.addSeparator()
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
            "Ctrl+O  open audio file\nCtrl+R  run models on the open file\nCtrl+L  load results of a previous run\n"
            "Ctrl+P  play / stop selection\n"
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
                                    "Choose the file in the Advanced settings tab, or untick 'Species list' there.")
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
                                "coordinates in the Advanced settings tab, or untick 'Location / season filter' there.")
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
            self.batch.out_edit.setText(str(self.panel.results_dir() / f"batch_{p.name}"))
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

    # -- recent runs
    def add_recent(self, run_dir):
        q = QSettings("BioTool", "BioTool")
        rec = [str(run_dir)] + [r for r in (q.value("runs/recent") or []) if r != str(run_dir)]
        q.setValue("runs/recent", rec[:12])

    def _fill_recent(self):
        self.recent_menu.clear()
        rec = [r for r in (QSettings("BioTool", "BioTool").value("runs/recent") or []) if Path(r).is_dir()]
        if not rec:
            a = self.recent_menu.addAction("(no saved runs yet)")
            a.setEnabled(False)
        for r in rec:
            a = self.recent_menu.addAction(Path(r).name)
            a.setToolTip(r)
            a.triggered.connect(lambda _=False, r=r: self.single.load_results(r))

    def log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.statusBar().showMessage(msg.splitlines()[0][:200])

    def _about(self):
        lines = [f"<b>{c.name}</b>: {c.description}<br><i>{c.licence_note}</i>" for c in DETECTORS.values()]
        lines.append(f"<b>{AcousticIndices.name}</b>: {AcousticIndices.description}")
        lines.append("<br>Indices:<br>" + "<br>".join(f"<b>{k}</b> - {v}" for k, v in INDEX_HELP.items()))
        QMessageBox.about(self, "Models and licences", "<br><br>".join(lines))
