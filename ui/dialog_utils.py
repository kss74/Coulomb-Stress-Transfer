# -*- coding: utf-8 -*-
"""
Shared helpers for QDialog sizing/window-chrome, used by every dialog in
this plugin that can grow tall (DistributedSlipDialog, SlipInversionDialog,
and any future one) -- factored out here rather than duplicated per-dialog.

Motivating bug reports (2026-08-15c): popup dialogs opened at a fixed
`self.resize(w, h)` that could exceed the actual screen size on smaller
displays, had no minimize/maximize window-chrome (most window managers
only give a QDialog a close button unless explicitly asked), and in at
least one report shrinking the window caused it to effectively vanish
(almost certainly the window collapsing to a near-zero-height frame once
a stacked-QVBoxLayout content column's cumulative minimumSizeHint was
fought below what the layout could actually lay out, with no scroll area
to absorb the difference).

configure_resizable_dialog() fixes the window-chrome/never-bigger-than-
screen part; wrap_in_scroll_area() fixes the "content taller than the
dialog can shrink to" part by moving a dialog's main content into a
QScrollArea so the DIALOG can be resized freely (down to a small,
explicit minimum) while its content simply gains scrollbars instead of
fighting the layout for space or disappearing.
"""

from qgis.PyQt.QtWidgets import QDialog, QScrollArea, QWidget, QVBoxLayout, QSizePolicy
from qgis.PyQt.QtCore import Qt


def configure_resizable_dialog(dialog: QDialog, default_width: int, default_height: int,
                               min_width: int = 360, min_height: int = 280):
    """
    Give `dialog` proper minimize/maximize/close window-chrome, a
    sensible minimum size it can actually be shrunk to, a size grip in
    the corner, and an initial size clamped to fit within the available
    screen (so it never opens larger than the screen, which is what was
    silently happening before on smaller displays). Call this INSTEAD of
    a bare `self.resize(w, h)` in a dialog's __init__.
    """
    dialog.setWindowFlags(
        dialog.windowFlags()
        | Qt.WindowMinMaxButtonsHint
        | Qt.WindowSystemMenuHint
        | Qt.WindowCloseButtonHint
    )
    dialog.setSizeGripEnabled(True)
    dialog.setMinimumSize(min(min_width, default_width), min(min_height, default_height))

    screen = None
    try:
        w = dialog.window()
        screen = w.screen() if hasattr(w, "screen") else None
    except Exception:
        screen = None
    if screen is None:
        try:
            from qgis.PyQt.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
        except Exception:
            screen = None

    if screen is not None:
        avail = screen.availableGeometry()
        # Leave a margin so the dialog's own title bar/frame and any
        # taskbar/dock are never pushed off-screen.
        max_w = max(min_width, int(avail.width() * 0.92))
        max_h = max(min_height, int(avail.height() * 0.88))
        width = min(default_width, max_w)
        height = min(default_height, max_h)
    else:
        width, height = default_width, default_height

    dialog.resize(width, height)


def wrap_in_scroll_area(dialog: QDialog, build_content):
    """
    Set `dialog`'s own top-level layout to a single QScrollArea holding
    a fresh QWidget, and return that inner QWidget for the caller to lay
    content out on (via `build_content(inner_widget)`, or just use the
    returned widget directly as the parent for a QVBoxLayout as usual).

    This is the fix for "shrinking the dialog makes tall stacked content
    disappear": the scroll area absorbs any shortfall between the
    dialog's actual size and its content's natural size by scrolling,
    instead of the layout system fighting over impossible space and
    (per the bug report) the window collapsing.

    build_content : callable(inner_widget) -> None, responsible for
                    creating the inner widget's own layout and adding
                    everything to it (exactly what the caller would
                    otherwise have done directly on `dialog`).
    """
    outer_layout = QVBoxLayout(dialog)
    outer_layout.setContentsMargins(0, 0, 0, 0)

    scroll = QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)

    inner = QWidget()
    inner.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    scroll.setWidget(inner)
    outer_layout.addWidget(scroll)

    build_content(inner)
    return inner


def wrap_widget_in_scroll_area(widget, parent=None):
    """
    Wrap an ALREADY-BUILT widget in a QScrollArea for embedding as one
    region of a larger layout -- e.g. a settings column that sits beside
    a plot rather than the plot being stacked below it (AftershockMCTestDialog,
    2026-08-18). Unlike wrap_in_scroll_area(), this does NOT take over
    the dialog's own top-level layout (calling QVBoxLayout(dialog) a
    second time on a dialog that already has one is a Qt error), so it
    composes with sibling non-scrolling content the caller adds
    elsewhere in its own layout.

    widget : a fully-built widget (already has its own layout and
             children) that should become scrollable in place.
    Returns the QScrollArea; caller adds IT (not `widget` directly) to
    their own layout with addWidget().
    """
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setWidget(widget)
    return scroll
