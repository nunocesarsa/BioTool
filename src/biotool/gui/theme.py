"""Interface themes and button styles.

Themes: Classic (light window, dark plots - the original look), Light (white everywhere),
Grey (mid-grey, between light and dark) and Dark.
Buttons get a role via setObjectName: "primary" (green: run / start), "danger" (red: stop),
"accent" (blue: open / play). Their text is slightly larger so the main actions stand out.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    base: str
    alt_base: str
    text: str
    button: str
    highlight: str
    plot_bg: str
    plot_fg: str
    muted: str
    link: str


THEMES = {
    "Classic": Theme("Classic", "#efefef", "#ffffff", "#f5f5f5", "#1e1e1e", "#e6e6e6", "#3a7bd5",
                     "#000000", "#c8c8c8", "#7a7a7a", "#1a5fb4"),
    "Light": Theme("Light", "#fafafa", "#ffffff", "#f3f5f7", "#1b1b1b", "#f0f0f0", "#3a7bd5",
                   "#ffffff", "#333333", "#808080", "#1a5fb4"),
    "Grey": Theme("Grey", "#5b6067", "#4b4f55", "#53585e", "#f2f2f2", "#666b73", "#6aa3ff",
                  "#2c2f33", "#d6d6d6", "#b8bcc2", "#9cc4ff"),
    "Dark": Theme("Dark", "#2b2d30", "#1f2023", "#26282b", "#e8e8e8", "#3a3c40", "#4c8dff",
                  "#141518", "#c0c0c0", "#9a9a9a", "#8ab4f8"),
}
DEFAULT = "Classic"

BUTTON_CSS = """
QPushButton#primary { background: #2e7d32; color: white; font-weight: 600; font-size: 10.5pt;
                      padding: 5px 16px; border: 1px solid #1b5e20; border-radius: 4px; }
QPushButton#primary:hover { background: #388e3c; }
QPushButton#primary:pressed { background: #1b5e20; }
QPushButton#danger { background: #c62828; color: white; font-weight: 600; font-size: 10.5pt;
                     padding: 5px 14px; border: 1px solid #8e0000; border-radius: 4px; }
QPushButton#danger:hover { background: #d32f2f; }
QPushButton#accent { background: #1565c0; color: white; font-weight: 600; font-size: 10.5pt;
                     padding: 5px 14px; border: 1px solid #0d47a1; border-radius: 4px; }
QPushButton#accent:hover { background: #1976d2; }
QPushButton#accent:checked { background: #0d47a1; }
QPushButton#primary:disabled, QPushButton#danger:disabled, QPushButton#accent:disabled {
    background: palette(button); color: palette(mid); border: 1px solid palette(mid); }
QLabel#muted { color: %(muted)s; }
QLabel#step { color: %(link)s; font-weight: 600; }
"""


def role(widget: QWidget, name: str) -> QWidget:
    """Give a button a colour role ('primary', 'danger', 'accent')."""
    widget.setObjectName(name)
    return widget


def apply(app: QApplication, name: str) -> Theme:
    t = THEMES.get(name, THEMES[DEFAULT])
    app.setStyle("Fusion")
    p = QPalette()
    c = QColor
    for group in (QPalette.Active, QPalette.Inactive):
        p.setColor(group, QPalette.Window, c(t.window))
        p.setColor(group, QPalette.WindowText, c(t.text))
        p.setColor(group, QPalette.Base, c(t.base))
        p.setColor(group, QPalette.AlternateBase, c(t.alt_base))
        p.setColor(group, QPalette.Text, c(t.text))
        p.setColor(group, QPalette.Button, c(t.button))
        p.setColor(group, QPalette.ButtonText, c(t.text))
        p.setColor(group, QPalette.Highlight, c(t.highlight))
        p.setColor(group, QPalette.HighlightedText, c("#ffffff"))
        p.setColor(group, QPalette.ToolTipBase, c(t.base))
        p.setColor(group, QPalette.ToolTipText, c(t.text))
        p.setColor(group, QPalette.PlaceholderText, c(t.muted))
        p.setColor(group, QPalette.Link, c(t.link))
        p.setColor(group, QPalette.Mid, c(t.muted))
    for role_ in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role_, c(t.muted))
    p.setColor(QPalette.Disabled, QPalette.Base, c(t.alt_base))
    p.setColor(QPalette.Disabled, QPalette.Button, c(t.button))
    app.setPalette(p)
    app.setStyleSheet(BUTTON_CSS % {"muted": t.muted, "link": t.link})
    pg.setConfigOption("background", t.plot_bg)
    pg.setConfigOption("foreground", t.plot_fg)
    return t


def style_plots(root: QWidget, t: Theme) -> None:
    """Re-colour existing pyqtgraph plots (new ones pick up the config automatically)."""
    for w in root.findChildren(pg.PlotWidget):
        w.setBackground(t.plot_bg)
        item = w.getPlotItem()
        for ax in ("left", "bottom", "right", "top"):
            a = item.getAxis(ax)
            a.setPen(pg.mkPen(t.plot_fg))
            a.setTextPen(pg.mkPen(t.plot_fg))
        if item.titleLabel is not None and item.titleLabel.text:
            item.setTitle(item.titleLabel.text, color=t.plot_fg)
