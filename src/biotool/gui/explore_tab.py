"""'Explore sounds' tab: group similar windows (embedding clusters), browse, listen, name groups."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from .. import explore
from ..audio import to_mono
from . import theme, workers
from .table import DataFrameView


class _ClusterWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, run_dir, model, scope, mcs):
        super().__init__()
        self.args = (run_dir, model, scope, mcs)

    def run(self):
        try:
            run_dir, model, scope, mcs = self.args
            self.log.emit(f"Loading {model} embeddings from {run_dir} ...")
            meta, X = explore.load_windows(run_dir, model)
            w, s = explore.cluster(meta, X, scope, mcs, progress=self.log.emit)
            self.finished.emit((w, s))
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def _snippet_spectrogram(path: str, start: float, end: float, pad: float = 0.5, nfft: int = 512):
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr = f.samplerate
        s0 = max(0.0, start - pad)
        f.seek(int(s0 * sr))
        x = to_mono(f.read(int((end - start + 2 * pad) * sr), dtype="float32", always_2d=False))
    hop = nfft // 2
    n = max(1, (len(x) - nfft) // hop + 1)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    fr = x[np.clip(idx, 0, len(x) - 1)] * np.hanning(nfft)[None, :]
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-6)
    S -= np.median(S, axis=1, keepdims=True)
    return S, sr, s0, len(x) / sr


class ExploreTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.windows: pd.DataFrame | None = None
        self.summary: pd.DataFrame | None = None
        self.current_cluster = None
        self.current_window = None
        self._player = None

        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Results folder of a batch run (with embeddings)")
        browse = QPushButton("Browse...")
        self.model = QComboBox()
        self.scope = QComboBox()
        for k, v in explore.SCOPES.items():
            self.scope.addItem(v, k)
        self.mcs = QSpinBox(); self.mcs.setRange(2, 500); self.mcs.setValue(8)
        self.mcs.setToolTip("Smallest number of similar windows that counts as a group")
        self.run_btn = theme.role(QPushButton("Find sound groups"), "primary")
        self.export_btn = QPushButton("Export groups (CSV)")
        self.export_btn.setEnabled(False)
        self.status = QLabel("")
        self.status.setObjectName("muted")

        step = lambda t: (lambda l: (l.setObjectName("step"), l)[1])(QLabel(t))
        r1 = QHBoxLayout()
        r1.addWidget(step("1")); r1.addWidget(QLabel("Results folder"))
        r1.addWidget(self.dir_edit, 1); r1.addWidget(browse)
        r2 = QHBoxLayout()
        for w in (step("2"), QLabel("Embeddings"), self.model, QLabel("Windows"), self.scope,
                  QLabel("Min. group size"), self.mcs, self.run_btn, step("4"), self.export_btn):
            r2.addWidget(w)
        r2.addStretch(1)
        help_ = QLabel("Groups windows that sound alike, using the embeddings saved by a batch run. "
                       "<b>3</b> Click a group (table or map) to see and hear its windows, give it a name, "
                       "then <b>4</b> export. Grey points = unassigned one-off sounds. "
                       "Perch embeddings usually separate sounds best.")
        help_.setWordWrap(True)
        help_.setObjectName("muted")

        self.groups = DataFrameView(on_row=self._group_clicked)
        self.members = DataFrameView(on_row=self._member_clicked)
        self.map = pg.PlotWidget()
        self.map.hideAxis("left"); self.map.hideAxis("bottom")
        self.map.setTitle("Sound map (similar sounds are close together)")
        self.scatter = pg.ScatterPlotItem(size=7, pen=None)
        self.scatter.sigClicked.connect(self._point_clicked)
        self.map.addItem(self.scatter)

        self.snip = pg.PlotWidget()
        self.snip.setLabel("left", "kHz"); self.snip.setLabel("bottom", "s")
        self.snip.setMouseEnabled(x=False, y=False)
        self.snip_img = pg.ImageItem(); self.snip_img.setColorMap(pg.colormap.get("viridis"))
        self.snip.addItem(self.snip_img)
        self.snip_region = pg.LinearRegionItem(movable=False, brush=(255, 255, 255, 30))
        self.snip.addItem(self.snip_region)
        self.play_btn = QPushButton("Play window")
        self.play_btn.setEnabled(False)
        self.rep_btn = QPushButton("Play group example")
        self.rep_btn.setEnabled(False)
        self.name_edit = QLineEdit(); self.name_edit.setPlaceholderText("Name this group, e.g. 'cicada A' or 'generator'")
        self.name_btn = QPushButton("Set name"); self.name_btn.setEnabled(False)
        self.snip_label = QLabel("")
        self.snip_label.setWordWrap(True)

        name_row = QHBoxLayout()
        name_row.addWidget(self.name_edit, 1); name_row.addWidget(self.name_btn)
        play_row = QHBoxLayout()
        play_row.addWidget(self.play_btn); play_row.addWidget(self.rep_btn); play_row.addWidget(self.snip_label, 1)

        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Sound groups")); ll.addWidget(self.groups, 1); ll.addLayout(name_row)
        right_top = QSplitter(Qt.Horizontal)
        right_top.addWidget(self.map)
        snipw = QWidget(); sl = QVBoxLayout(snipw); sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.snip, 1); sl.addLayout(play_row)
        right_top.addWidget(snipw)
        right = QSplitter(Qt.Vertical)
        right.addWidget(right_top)
        mw = QWidget(); ml = QVBoxLayout(mw); ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(QLabel("Windows in the selected group (click to view / listen)")); ml.addWidget(self.members)
        right.addWidget(mw)
        right.setSizes([380, 260])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left); split.addWidget(right)
        split.setSizes([480, 900])

        lay = QVBoxLayout(self)
        lay.addLayout(r1); lay.addLayout(r2); lay.addWidget(help_); lay.addWidget(self.status)
        lay.addWidget(split, 1)
        self.play_btn.setObjectName("accent"); self.rep_btn.setObjectName("accent")
        self._names_dirty = False
        self._loaded_key = None

        browse.clicked.connect(self._browse)
        self.dir_edit.editingFinished.connect(self._refresh_models)
        self.model.currentIndexChanged.connect(lambda _: self._settings_changed())
        self.scope.currentIndexChanged.connect(lambda _: self._settings_changed())
        self.mcs.valueChanged.connect(lambda _: self._settings_changed())
        self._clear_results("Choose a results folder, then click 'Find sound groups'.")
        self.run_btn.clicked.connect(self.run)
        self.export_btn.clicked.connect(self._export)
        self.play_btn.clicked.connect(self._play_window)
        self.rep_btn.clicked.connect(self._play_rep)
        self.name_btn.clicked.connect(self._set_name)
        self.name_edit.returnPressed.connect(self._set_name)

    # -- folder / models
    def set_folder(self, folder: str):
        if folder:
            self.dir_edit.setText(folder)
            self._refresh_models()

    def refresh_enabled(self, busy: bool):
        self.run_btn.setEnabled(not busy and self.model.count() > 0)

    # -- clearing
    def _confirm_discard(self) -> bool:
        if not self._names_dirty:
            return True
        r = QMessageBox.question(self, "Explore", "You named groups but have not exported them. "
                                                  "Discard the names?")
        return r == QMessageBox.Yes

    def _clear_results(self, message: str = ""):
        self.windows = self.summary = None
        self.current_cluster = self.current_window = None
        self._names_dirty = False
        self._loaded_key = None
        self.groups.set_frame(pd.DataFrame())
        self.members.set_frame(pd.DataFrame())
        self.scatter.clear()
        self.snip_img.clear()
        self.snip_region.hide()
        self.snip_label.setText("")
        self.name_edit.clear()
        for b in (self.export_btn, self.name_btn, self.play_btn, self.rep_btn):
            b.setEnabled(False)
        self.status.setText(message)

    def _settings_changed(self):
        if self.windows is not None:
            if not self._confirm_discard():
                return
            self._clear_results("Settings changed - click 'Find sound groups' to group again.")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose results folder", self.dir_edit.text() or self.main.last_dir)
        if d:
            self.set_folder(d)

    def _refresh_models(self):
        folder = self.dir_edit.text().strip()
        if folder == getattr(self, "_folder", None) and self.model.count():
            return
        if self.windows is not None and not self._confirm_discard():
            self.dir_edit.setText(self._folder or "")
            return
        self._folder = folder
        ms = explore.available_models(folder) if folder and Path(folder).is_dir() else []
        self.model.blockSignals(True)
        self.model.clear()
        pref = ["perch_v2", "birdnet", "yamnet"]
        for m in sorted(ms, key=lambda m: pref.index(m) if m in pref else 9):
            self.model.addItem(m)
        self.model.blockSignals(False)
        if folder and not ms:
            msg = "No embeddings in this folder - run a batch with 'Save embeddings' ticked."
        elif ms:
            n = len(list(Path(folder, "embeddings").glob(f"*__{self.model.currentText()}.npz")))
            msg = f"Embeddings found for: {', '.join(ms)} ({n} files). Click 'Find sound groups'."
        else:
            msg = "Choose a results folder, then click 'Find sound groups'."
        self._clear_results(msg)
        self.refresh_enabled(self.main.busy_job)

    # -- clustering
    def run(self):
        d, m = self.dir_edit.text().strip(), self.model.currentText()
        if self.main.busy_job:
            return
        if not d or not m:
            QMessageBox.information(self, "Explore", "Choose a results folder that contains embeddings.")
            return
        if not self._confirm_discard():
            return
        self._clear_results("Grouping ...")
        self.main.set_busy(True)
        self._worker = _ClusterWorker(d, m, self.scope.currentData(), self.mcs.value())
        self._worker.log.connect(self.main.log)
        self._worker.finished.connect(self._done)
        self._worker.failed.connect(self._failed)
        workers.start(self._worker, self)

    def _failed(self, msg):
        self.main.set_busy(False)
        first = msg.splitlines()[0]
        self.status.setText(first)
        self.main.log("Explore: " + first)
        if "nothing to group" in first:
            first += "\n\nTry 'All windows' - in busy recordings almost every window already has a species."
            QMessageBox.information(self, "Explore", first)
        else:
            QMessageBox.critical(self, "Explore", msg[:3000])

    def _done(self, res):
        self.main.set_busy(False)
        self.windows, self.summary = res
        self.export_btn.setEnabled(True)
        self.groups.set_frame(self.summary)
        self._draw_map()
        self.map.autoRange()
        n = int((self.summary["cluster"] >= 0).sum())
        self.status.setText(f"{n} sound groups from {len(self.windows)} windows "
                            f"({int((self.windows['cluster'] < 0).sum())} unassigned). "
                            "Click a group to listen; name it; export when done.")
        if len(self.summary):
            self._select_cluster(int(self.summary.iloc[0]["cluster"]))

    def _color(self, c: int):
        return pg.mkBrush(170, 170, 170, 90) if c < 0 else pg.mkBrush(pg.intColor(c, hues=12, values=2, alpha=220))

    def _draw_map(self, highlight: int | None = None):
        w = self.windows
        c = w["cluster"].to_numpy()
        if not hasattr(self, "_brush_cache") or self._brush_cache[0] is not w:
            uniq = {int(k): self._color(int(k)) for k in np.unique(c)}
            self._brush_cache = (w, [uniq[int(k)] for k in c])
        sel = (c == highlight) if highlight is not None else np.zeros(len(c), bool)
        size = np.where(sel, 11, np.where(c < 0, 5, 7))
        edge = pg.mkPen(self.main.theme.plot_fg, width=1)
        pens = [edge if s_ else None for s_ in sel]
        self.scatter.setData(x=w["x"].to_numpy(), y=w["y"].to_numpy(), data=np.arange(len(w)),
                             brush=self._brush_cache[1], size=size, pen=pens)

    # -- selection
    def _select_cluster(self, c: int):
        self.current_cluster = c
        d = self.windows[self.windows["cluster"] == c]
        cols = ["stem", "start_s", "end_s", "labels", "representative", "source_file"]
        self.members.set_frame(d[cols].sort_values(["representative", "stem", "start_s"],
                                                   ascending=[False, True, True]))
        row = self.summary[self.summary["cluster"] == c]
        self.name_edit.setText(str(row.iloc[0]["name"]) if len(row) and c >= 0 else "")
        self.name_btn.setEnabled(c >= 0)
        self.rep_btn.setEnabled(c >= 0)
        self._draw_map(c)
        rep = d[d["representative"]]
        self._show_window((rep if len(rep) else d).iloc[0].to_dict())

    def _group_clicked(self, row: dict):
        self._select_cluster(int(row["cluster"]))

    def _point_clicked(self, _, points, *args):
        if not points:
            return
        r = self.windows.iloc[int(points[0].data())]
        if int(r["cluster"]) != self.current_cluster:
            self._select_cluster(int(r["cluster"]))
        self._show_window(r.to_dict())

    def _member_clicked(self, row: dict):
        self._show_window(row)

    def _show_window(self, row: dict):
        self.current_window = row
        path = row.get("source_file", "")
        if not path or not Path(path).exists():
            self.snip_label.setText(f"Audio file not found: {path}")
            self.play_btn.setEnabled(False)
            return
        s, e = float(row["start_s"]), float(row["end_s"])
        S, sr, s0, dur = _snippet_spectrogram(path, s, e)
        v = self.main.single.view  # same colours / brightness / contrast as the Single file view
        self.snip_img.setColorMap(pg.colormap.get(v.colormap_name()))
        self.snip_img.setImage(S, autoLevels=False)
        lo, hi = np.percentile(S, [40, 99.8])
        self.snip_img.setLevels(list(v.levels(lo, hi)))
        self.snip_region.show()
        self.snip_img.setRect(QRectF(s0, 0, dur, sr / 2000))
        self.snip.setXRange(s0, s0 + dur, padding=0); self.snip.setYRange(0, sr / 2000, padding=0)
        self.snip_region.setRegion([s, e])
        self.snip_label.setText(f"<b>{row.get('stem')}</b> {s:.1f}-{e:.1f} s<br>{row.get('labels') or 'no model detection'}")
        self.play_btn.setEnabled(True)

    # -- audio
    def _player_obj(self):
        if self._player is None:
            from .player import SegmentPlayer
            self._player = SegmentPlayer()
        return self._player

    def _play(self, row):
        v = self.main.single.view  # same speed / band settings as the Single file tab
        err = self._player_obj().play(row["source_file"], float(row["start_s"]), float(row["end_s"]),
                                      speed=v.playback_speed(), band=v.playback_band_hz(),
                                      hiss=v.hiss_strength())
        if err:
            QMessageBox.warning(self, "Playback", err)

    def _play_window(self):
        if self.current_window:
            self._play(self.current_window)

    def _play_rep(self):
        d = self.windows[(self.windows["cluster"] == self.current_cluster) & self.windows["representative"]]
        if len(d):
            r = d.iloc[0].to_dict()
            self._show_window(r)
            self._play(r)

    # -- naming / export
    def _set_name(self):
        if self.current_cluster is None or self.current_cluster < 0:
            return
        self.summary.loc[self.summary["cluster"] == self.current_cluster, "name"] = self.name_edit.text().strip()
        self.groups.set_frame(self.summary)
        self._names_dirty = True

    def _export(self):
        p = explore.export(self.dir_edit.text(), self.windows, self.summary, self.model.currentText())
        self._names_dirty = False
        self.main.log(f"Exported sound groups to {p}")
        self.status.setText(f"Exported to {p.name} (and {p.stem}_summary.csv) in the results folder.")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
