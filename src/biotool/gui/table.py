"""pandas DataFrame -> Qt table with sorting and text filtering."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QTableView, QVBoxLayout, QWidget)


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None):
        super().__init__()
        self._df = pd.DataFrame() if df is None else df

    def set_frame(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        v = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(v, float):
                return "" if pd.isna(v) else f"{v:.4g}"
            return str(v)
        if role == Qt.UserRole:  # raw value for sorting - plain Python types only (numpy scalars crash Qt)
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and v != v:
                return None
            return v if isinstance(v, (int, float, str, bool)) else str(v)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


class _Proxy(QSortFilterProxyModel):
    def lessThan(self, left, right):
        a, b = left.data(Qt.UserRole), right.data(Qt.UserRole)
        if a is None or b is None:
            return a is None and b is not None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


class DataFrameView(QWidget):
    """Table with a filter box. `row_activated(dict)` gives the clicked row as a dict."""

    def __init__(self, placeholder: str = "Filter (matches any column)...", on_row=None):
        super().__init__()
        self.model = DataFrameModel()
        self.proxy = _Proxy()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.verticalHeader().setDefaultSectionSize(22)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(placeholder)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.count = QLabel("")
        top = QHBoxLayout()
        top.addWidget(self.filter, 1)
        top.addWidget(self.count)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.view)
        self._on_row = on_row
        self.view.clicked.connect(self._clicked)
        self.proxy.layoutChanged.connect(self._update_count)
        self.proxy.modelReset.connect(self._update_count)
        self.filter.textChanged.connect(lambda _: self._update_count())

    def _update_count(self):
        self.count.setText(f"{self.proxy.rowCount()} / {self.model.rowCount()} rows")

    def set_frame(self, df: pd.DataFrame):
        # A fresh model each time: resetting a sorted model whose columns change can crash Qt.
        old = self.model
        self.model = DataFrameModel(df.reset_index(drop=True))
        self.proxy.setSourceModel(self.model)
        old.deleteLater()
        self.proxy.sort(-1)  # keep the frame's own order until the user clicks a header
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.view.resizeColumnsToContents()
        for c in range(self.model.columnCount()):
            if self.view.columnWidth(c) > 260:
                self.view.setColumnWidth(c, 260)
        self._update_count()

    def _clicked(self, idx):
        if self._on_row is None:
            return
        src = self.proxy.mapToSource(idx)
        self._on_row(self.model.frame().iloc[src.row()].to_dict())
