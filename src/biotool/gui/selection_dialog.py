"""Small window showing what each model thinks is in the selected span (temporary, not saved)."""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .table import DataFrameView


class SelectionDialog(QDialog):
    def __init__(self, parent, title: str, df: pd.DataFrame, note: str, settings_note: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 420)
        self.df = df
        head = QLabel(f"<b>{title}</b><br>{settings_note}")
        head.setWordWrap(True)
        expl = QLabel("Top labels per model for this stretch, analysed with windows every 0.5 s. "
                      "<i>above_min_score</i> = would be reported with your current settings; "
                      "<i>windows_found</i> = windows where it passes the min score. "
                      "Nothing is saved - use Copy to paste into a spreadsheet.")
        expl.setWordWrap(True)
        expl.setObjectName("muted")
        table = DataFrameView()
        table.set_frame(df if len(df) else pd.DataFrame({"result": ["no labels found"]}))
        notes = QLabel(note)
        notes.setWordWrap(True)
        notes.setObjectName("muted")
        copy = QPushButton("Copy table")
        copy.setToolTip("Copy the table to the clipboard (paste into Excel)")
        copy.clicked.connect(self._copy)
        close = QPushButton("Close")
        close.setToolTip("Close this window (nothing was saved)")
        close.clicked.connect(self.close)
        btns = QHBoxLayout()
        btns.addWidget(copy); btns.addStretch(1); btns.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(head); lay.addWidget(expl); lay.addWidget(table, 1)
        if note:
            lay.addWidget(notes)
        lay.addLayout(btns)

    def _copy(self):
        QApplication.clipboard().setText(self.df.to_csv(sep="\t", index=False))
