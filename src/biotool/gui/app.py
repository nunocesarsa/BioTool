"""Entry point for the desktop app:  biotool-gui  (or  python -m biotool.gui.app)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _ensure_streams():
    """Without a console (pythonw.exe) sys.stdout/stderr are None and libraries that print
    progress bars (kagglehub/tqdm) crash. Send everything to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from ..paths import app_data_dir

    log = open(app_data_dir() / "biotool.log", "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def main():
    _ensure_streams()
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("BioTool")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        w.single.open_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
