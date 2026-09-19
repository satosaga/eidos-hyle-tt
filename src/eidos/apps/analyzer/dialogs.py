"""
eidos.apps.analyzer.dialogs -- Small QDialog subclasses, both non-modal
(shown via show(), never exec() -- see each class's own docstring).

CalibrationDiagnosticsDialog: shows eidos.lib.calibration_diagnostics'
trade-off/non-identifiability figures for a finished Auto Fit run. Mirrors
eidos.apps.viewer.dialogs' naming/scope convention (small QDialog
subclasses live in a dialogs.py sibling of window.py).
"""

import logging

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.calibrator import CalibrationResult, SobolSensitivityTrials
from eidos.lib import calibration_diagnostics as diag

logger = logging.getLogger(__name__)


def _size_scroll_area_to_content(scroll: QScrollArea, content_w: int, content_h: int) -> None:
    """
    Give a setWidgetResizable(False) QScrollArea a minimumSize that
    shows a content_w x content_h child with NEITHER scrollbar visible.

    frameWidth() alone (`2 * scroll.frameWidth()`) isn't enough: on at
    least one real style/platform this codebase runs on, QScrollArea
    also reserves
    real layout space for a scrollbar's own track (QStyle's
    PM_ScrollBarExtent) as part of computing whether the child fits --
    even for the "would fit without it" case a plain frame-only margin
    was aiming for, this can round the wrong way and leave both
    scrollbars visible regardless of content actually fitting.
    Reserving that extent up front on both axes sidesteps the
    fits-or-doesn't-fit judgment call entirely.
    """
    frame = 2 * scroll.frameWidth()
    scrollbar_extent = scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent, None, scroll)
    scroll.setMinimumSize(content_w + frame + scrollbar_extent, content_h + frame + scrollbar_extent)


class CalibrationDiagnosticsDialog(QDialog):
    """
    Dialog showing eidos.lib.calibration_diagnostics' figure for a single
    Auto Fit CalibrationResult. Shown non-modally by its caller
    (TTAnalyzerWindow._on_show_diagnostics uses show(), not exec()) so
    several of these can be open at once -- e.g. one per Rebuild, to
    compare diagnostics side by side.

    The figure is built once, in __init__, via
    calibration_diagnostics.plot_calibration_diagnostics -- this dialog
    does no re-simulation and never touches calibrate() itself, it only
    visualizes a result TTAnalyzerWindow already has in hand.

    plot_calibration_diagnostics returns exactly one figure (Relative Std
    and Correlations are mutually exclusive, and Pairwise was folded into
    the Correlations heatmap's click-and-hold popup -- see
    _wire_pair_popup), so this dialog shows it directly, no tab bar.
    plot_calibration_diagnostics raises ValueError when there is too
    little data; _build_content catches that to show a word-wrapped
    explanation instead of a figure.

    Pooling method toggle
    -----------------------
    A QComboBox above the figure lets the viewer switch which near-best-
    fit level-set filter builds every panel here (see eidos.lib.
    calibration_diagnostics' module docstring for "rmse_tolerance" vs.
    "mahalanobis"). Switching it calls _build_content again, which
    rebuilds the figure/canvas/wiring from scratch (a fresh FigureCanvas
    each time) and re-sizes the dialog. _build_content shows a
    word-wrapped QLabel instead of a figure when
    plot_calibration_diagnostics raises ValueError (too few usable
    points) -- the label is word-wrapped and width-capped at
    _MIN_WIDTH_PX so a long message doesn't stretch the whole dialog to
    fit it on one line.

    The canvas is wrapped in a QScrollArea with a FIXED size
    (canvas.setFixedSize, pinned to the Figure's own natural pixel size --
    get_size_inches() * dpi) and setWidgetResizable(False) -- resizing
    this dialog's window changes how much of the canvas is VISIBLE
    (scrolling), never the canvas's own size or aspect ratio. Do NOT
    switch to setWidgetResizable(True) + canvas.setMinimumSize
    (grow-to-fill): FigureCanvasQTAgg resizes its underlying Figure,
    including aspect ratio, to match whatever Qt gives it, so a
    correlation heatmap's cells would stretch or squash depending on
    nothing more meaningful than how the dialog window was last dragged.
    A fixed canvas size sidesteps this: the
    figure only ever renders at the ONE pixel size (and aspect ratio)
    its own figsize specifies, at the cost of a little unused margin or
    a scrollbar when the dialog and canvas sizes don't match exactly --
    see _size_to_content, which still sizes the dialog to fit the canvas
    without needing to scroll in the common case.
    """

    # Display label -> eidos.lib.calibration_diagnostics pooling_method
    # value, in the order shown in the combo box. A tuple of (label,
    # value) pairs rather than a dict so the combo box's item order is
    # explicit and stable, independent of dict insertion-order relying on
    # a reader's trust that nothing reorders it later.
    _POOLING_METHOD_CHOICES = (
        ("RMSE tolerance", "rmse_tolerance"),
        ("Mahalanobis (MCD)", "mahalanobis"),
    )

    def __init__(self, result: CalibrationResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Fit Diagnostics")

        self._result = result
        # Click-and-hold popup state (see _wire_pair_popup) -- at most one
        # is ever open at a time, tracked here so a second press can close
        # a still-open one before opening the next. Also closed by
        # _build_content on every rebuild (see the pooling toggle) since
        # an already-open popup would otherwise go on referencing the
        # PREVIOUS pooling method's figure.
        self._pair_popup: QWidget | None = None
        # The single widget currently occupying _content_index below --
        # either the QScrollArea+canvas or the plain QLabel error message
        # -- tracked so _build_content can remove/replace it in place
        # without disturbing the pooling row above or the close row below.
        self._content_widget: QWidget | None = None

        self._layout = QVBoxLayout(self)

        pooling_row = QHBoxLayout()
        pooling_row.addWidget(QLabel("Pooling:"))
        self._pooling_combo = QComboBox()
        for label, method in self._POOLING_METHOD_CHOICES:
            self._pooling_combo.addItem(label, userData=method)
        self._pooling_combo.currentIndexChanged.connect(self._on_pooling_method_changed)
        pooling_row.addWidget(self._pooling_combo)
        pooling_row.addStretch(1)
        self._layout.addLayout(pooling_row)

        # Fixed index of the figure/message content within self._layout
        # (0 is pooling_row, just added above) -- _build_content
        # removes/re-inserts exactly this index on every (re)build; the
        # close row, added once below and never touched again, stays
        # after it regardless of how many times content is rebuilt.
        self._content_index = 1

        self._build_content(self._pooling_combo.currentData())

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        self._layout.addLayout(close_row)

    def _on_pooling_method_changed(self, _index: int) -> None:
        """QComboBox.currentIndexChanged handler -- see class docstring's
        "Pooling method toggle" section."""
        self._build_content(self._pooling_combo.currentData())

    def _build_content(self, pooling_method: str) -> None:
        """
        (Re)builds whatever occupies self._content_index -- either the
        Correlations/Eigenvalues/Loadings/VIFs figure for `pooling_method`,
        or, if plot_calibration_diagnostics raises ValueError (not enough
        data), a word-wrapped explanation instead. Called once from
        __init__ and again whenever the pooling combo box changes (see
        class docstring) -- always tears down whatever was there before,
        including a stale pair-popup, and builds fresh.
        """
        self._hide_pair_popup()
        if self._content_widget is not None:
            item = self._layout.takeAt(self._content_index)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
            self._content_widget = None

        result = self._result
        has_figure = False
        try:
            figs = diag.plot_calibration_diagnostics(result, pooling_method=pooling_method)
        except ValueError as exc:
            logger.warning("CalibrationDiagnosticsDialog: %s", exc)
            label = QLabel(str(exc))
            # Both required: an unwrapped QLabel sizes itself to its whole
            # string on one line, and wordWrap alone still leaves Qt free
            # to pick a wide wrapping width.
            label.setWordWrap(True)
            label.setMaximumWidth(self._MIN_WIDTH_PX - 40)
            self._layout.insertWidget(self._content_index, label)
            self._content_widget = label
            self.setMinimumSize(self._MIN_WIDTH_PX, self._MIN_HEIGHT_PX)
            self.resize(self._MIN_WIDTH_PX, self._MIN_HEIGHT_PX)
        else:
            has_figure = True
            # plot_calibration_diagnostics returns exactly one figure
            # (see its docstring) -- no tab bar needed, see class
            # docstring for why the earlier multi-tab layout was dropped.
            key, fig = next(iter(figs.items()))
            canvas = FigureCanvas(fig)
            # setFixedSize (not setMinimumSize), pinned to the Figure's
            # own natural pixel size -- see the class docstring for why.
            width_px = int(fig.get_size_inches()[0] * fig.dpi)
            height_px = int(fig.get_size_inches()[1] * fig.dpi)
            canvas.setFixedSize(width_px, height_px)
            if key == "correlation_heatmap":
                # The Correlations<->Loadings and Loadings<->VIFs gaps
                # come out visibly uneven in THIS app's real Qt rendering
                # even after calibration_diagnostics was tuned against
                # its own dev-time renders -- a gap tuned against one
                # renderer doesn't reliably carry over to another's font
                # metrics (see equalize_correlation_heatmap_gaps). Fixed
                # here against the actual FigureCanvasQTAgg: draw() first
                # forces a real layout pass at this canvas's fixed pixel
                # size, then equalize_correlation_heatmap_gaps measures
                # and closes the gaps using THIS canvas's own real
                # positions, then a second draw() re-renders with the
                # correction -- before the dialog is ever shown.
                canvas.draw()
                diag.equalize_correlation_heatmap_gaps(fig)
                # equalize_correlation_heatmap_gaps trims fig's own width
                # down to its now-repositioned content (see its own
                # docstring) -- re-read get_size_inches() and re-apply to
                # the canvas, or the canvas stays at the WIDER pre-trim
                # size and the dead space it just closed reopens as a
                # gap between the canvas's real content and its own
                # right edge instead.
                width_px = int(fig.get_size_inches()[0] * fig.dpi)
                height_px = int(fig.get_size_inches()[1] * fig.dpi)
                canvas.setFixedSize(width_px, height_px)
                canvas.draw()
            scroll = QScrollArea()
            scroll.setWidget(canvas)
            scroll.setWidgetResizable(False)
            # QScrollArea's own sizeHint() does NOT reflect a
            # setWidgetResizable(False) child's actual fixed size (it's a
            # generic "reasonable scrollable area" guess) -- without this,
            # _size_to_content's adjustSize() call below undersizes the
            # whole dialog to that generic guess, showing scrollbars for
            # a canvas that would otherwise fit outright.
            _size_scroll_area_to_content(scroll, canvas.width(), canvas.height())
            self._layout.insertWidget(self._content_index, scroll)
            self._content_widget = scroll
            if key == "correlation_heatmap":
                self._wire_pair_popup(canvas, fig, result, pooling_method)
                self._wire_row_crosshair(canvas, fig, result)
                self._wire_clickable_hover_affordance(canvas, fig, result)

        if has_figure:
            self._size_to_content()

    def _wire_pair_popup(
        self, canvas: FigureCanvas, fig, result: CalibrationResult, pooling_method: str,
    ) -> None:
        """
        Click-and-hold a Correlations heatmap cell to preview that pair's
        scatter; release to close it -- so a trade-off spotted in the
        heatmap (or the Loadings column next to it, sharing the same row
        axis) can be checked against the actual point cloud.

        The Correlations axes is found via its "correlations_heatmap"
        label rather than fig.axes ordering, so this doesn't break if
        plot_parameter_correlation_heatmap's internal axes-creation order
        ever changes.

        event.xdata/event.ydata land in the same data coordinates as the
        heatmap's tick positions (0..n_free-1, one per cell) because the
        underlying imshow uses its default extent -- rounding to the
        nearest integer recovers the (row, col) cell the same way the
        heatmap's own tick labels do.

        pooling_method: MUST be the same value _build_content just built
        `fig` with, so the popup's title Pearson r matches the cell the
        viewer clicked (see plot_single_pair_scatter's docstring).
        """
        ax = next((a for a in fig.axes if a.get_label() == "correlations_heatmap"), None)
        if ax is None:
            logger.warning(
                "_wire_pair_popup: no axes labeled 'correlations_heatmap' in "
                "this figure -- pair-preview-on-click will not be available."
            )
            return
        keys = result.free_keys
        n_free = len(keys)

        def on_press(event):
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            j = int(round(event.xdata))
            i = int(round(event.ydata))
            if not (0 <= i < n_free and 0 <= j < n_free):
                return  # off-grid
            # Diagonal cells (a key vs. itself) pop up too -- treating
            # every cell the same way is easier to predict when clicking
            # around the grid than a diagonal that silently does nothing.
            pair_fig = diag.plot_single_pair_scatter(
                result, keys[j], keys[i], pooling_method=pooling_method,
            )
            if pair_fig is not None:
                self._show_pair_popup(pair_fig, canvas, event)

        def on_release(_event):
            self._hide_pair_popup()

        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("button_release_event", on_release)

    def _wire_row_crosshair(self, canvas: FigureCanvas, fig, result: CalibrationResult) -> None:
        """
        Hover tracking across the row-aligned panels (Correlations,
        Loadings, VIFs all share the same free_keys row axis via sharey)
        plus the direction-aligned pair (Loadings/Eigenvalues share a
        principal-direction column axis instead, via sharex). Helps track
        which free_key row you're on across this multi-panel figure.

        Outline only (facecolor="none"): a filled highlight would tint
        whatever heatmap cells it covers, making them read as a
        different value against the shared colorbar than they actually
        are. A white stroke behind the thin pale-magenta edge keeps the
        box visible against both the colormap's dark and light ends.
        Same hue family as _wire_clickable_hover_affordance's own border
        (deliberately related) but thin/pale where that one is
        thick/saturated -- this box means "same row/column," not
        "clickable."

        Hovering any row-aligned axes draws a horizontal box spanning
        that row across all three of them. Hovering Correlations
        additionally draws a vertical box within Correlations alone;
        hovering Loadings or Eigenvalues additionally draws a vertical
        box across both of them. VIFs has no second meaningful axis to
        box -- an x position there is a bar VALUE, not a parameter/
        direction index.

        Axes are found by label, same as _wire_pair_popup. Redraws only
        when the hovered (row, column) INDEX changes, not on every raw
        motion_notify_event, to avoid re-rendering the whole figure on
        every tiny mouse jitter.
        """
        by_label = {a.get_label(): a for a in fig.axes}
        row_axes = [
            by_label[name] for name in
            ("correlations_heatmap", "loadings_heatmap", "vifs_bar")
            if name in by_label
        ]
        if len(row_axes) < 2:
            logger.warning(
                "_wire_row_crosshair: fewer than 2 row-aligned axes found in "
                "this figure -- hover crosshair will not be available."
            )
            return
        ax_corr = by_label.get("correlations_heatmap")
        ax_load = by_label.get("loadings_heatmap")
        ax_eig = by_label.get("eigenvalues_bar")
        col_axes = {a for a in (ax_load, ax_eig) if a is not None}
        n_free = len(result.free_keys)

        artists: list[Rectangle] = []
        last: dict[str, int | str | None] = {"row": None, "col_ax_key": None, "col": None}

        def _clear() -> None:
            for artist in artists:
                artist.remove()
            artists.clear()

        def _box(ax, xa: float, xb: float, ya: float, yb: float) -> None:
            x0, x1 = sorted((xa, xb))
            y0, y1 = sorted((ya, yb))
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="none", edgecolor="#e8a0c4", linewidth=1.6, zorder=20,
            )
            rect.set_path_effects([withStroke(linewidth=3.5, foreground="white")])
            ax.add_patch(rect)
            artists.append(rect)

        def on_move(event) -> None:
            in_tracked_axes = event.inaxes in row_axes or event.inaxes is ax_eig
            if not in_tracked_axes:
                if last["row"] is not None:
                    _clear()
                    last["row"] = last["col_ax_key"] = last["col"] = None
                    canvas.draw_idle()
                return

            row = None
            if event.inaxes in row_axes and event.ydata is not None:
                i = int(round(event.ydata))
                if 0 <= i < n_free:
                    row = i

            # "col_ax_key" distinguishes Correlations' own column axis from
            # Loadings/Eigenvalues' shared direction axis -- same integer
            # index means something different in each, so a plain
            # (row, col) comparison could wrongly skip a redraw when the
            # viewer moves from one to the other at the same index.
            col_ax_key: str | None = None
            col: int | None = None
            if event.inaxes is ax_corr and event.xdata is not None:
                j = int(round(event.xdata))
                if 0 <= j < n_free:
                    col_ax_key, col = "correlations", j
            elif event.inaxes in col_axes and event.xdata is not None:
                j = int(round(event.xdata))
                if 0 <= j < n_free:
                    col_ax_key, col = "direction", j

            if (row, col_ax_key, col) == (last["row"], last["col_ax_key"], last["col"]):
                return
            last["row"], last["col_ax_key"], last["col"] = row, col_ax_key, col

            _clear()
            if row is not None:
                for a in row_axes:
                    x0, x1 = a.get_xlim()
                    _box(a, x0, x1, row - 0.5, row + 0.5)
            if col_ax_key == "correlations" and ax_corr is not None:
                assert col is not None  # travels with col_ax_key -- see the tuple assignment above
                y0, y1 = ax_corr.get_ylim()
                _box(ax_corr, col - 0.5, col + 0.5, y0, y1)
            elif col_ax_key == "direction":
                assert col is not None  # travels with col_ax_key -- see the tuple assignment above
                for a in col_axes:
                    y0, y1 = a.get_ylim()
                    _box(a, col - 0.5, col + 0.5, y0, y1)
            canvas.draw_idle()

        def on_leave(_event) -> None:
            if last["row"] is not None or last["col"] is not None:
                _clear()
                last["row"] = last["col_ax_key"] = last["col"] = None
                canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("figure_leave_event", on_leave)

    def _wire_clickable_hover_affordance(self, canvas: FigureCanvas, fig, result: CalibrationResult) -> None:
        """
        Cursor + border affordance distinguishing this figure's one
        click-and-hold-for-popup element (Correlations cells) from its
        look-only ones (Loadings cells, VIFs bars): hovering a clickable
        cell switches the cursor to a pointing hand and outlines that ONE
        cell in a bright, thick accent border; hovering anything else
        leaves the cursor and border alone. No elevation/shadow effect --
        matplotlib has no native concept of one, and the row crosshair
        below already gets a similar payoff more simply with a border.

        A DIFFERENT feature from _wire_row_crosshair: that one boxes the
        whole hovered ROW across every row-aligned panel regardless of
        clickability, in a neutral black+white-stroke style. This one
        boxes a single ELEMENT in a deliberately different, saturated
        color, so a viewer can't mistake "this row is aligned with that
        one" for "this thing is clickable" -- both boxes can be visible
        on the same cell at once, which is intentional.

        Runs as its own motion_notify_event/figure_leave_event pair,
        independent of _wire_row_crosshair's -- matplotlib dispatches
        every connected callback per event, so both fire without
        interfering with each other.
        """
        by_label = {a.get_label(): a for a in fig.axes}
        ax_corr = by_label.get("correlations_heatmap")
        if ax_corr is None:
            logger.warning(
                "_wire_clickable_hover_affordance: no axes labeled "
                "'correlations_heatmap' found in this figure -- "
                "clickable-element hover affordance will not be available."
            )
            return
        n_free = len(result.free_keys)

        artists: list[Rectangle] = []
        last_key = None  # (id(axes), row, col) for the currently-highlighted element, or None

        def _clear() -> None:
            for artist in artists:
                artist.remove()
            artists.clear()

        def on_move(event) -> None:
            nonlocal last_key
            key = None
            box = None  # (ax, x0, x1, y0, y1)

            if event.inaxes is ax_corr and event.xdata is not None and event.ydata is not None:
                j = int(round(event.xdata))
                i = int(round(event.ydata))
                if 0 <= i < n_free and 0 <= j < n_free:
                    key = (id(ax_corr), i, j)
                    box = (ax_corr, j - 0.5, j + 0.5, i - 0.5, i + 0.5)

            if key == last_key:
                return
            last_key = key

            _clear()
            if box is not None:
                ax, x0, x1, y0, y1 = box
                rect = Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor="none", edgecolor="#d6006f", linewidth=2.4, zorder=30,
                )
                ax.add_patch(rect)
                artists.append(rect)
                canvas.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                canvas.unsetCursor()
            canvas.draw_idle()

        def on_leave(_event) -> None:
            nonlocal last_key
            if last_key is not None:
                last_key = None
                _clear()
                canvas.unsetCursor()
                canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("figure_leave_event", on_leave)

    def _show_pair_popup(self, pair_fig, source_canvas: FigureCanvas, event) -> None:
        """Show pair_fig in a small frameless popup near the click."""
        self._hide_pair_popup()

        popup = QWidget(self, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        popup.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(1, 1, 1, 1)

        popup_canvas = FigureCanvas(pair_fig)
        width_px = int(pair_fig.get_size_inches()[0] * pair_fig.dpi)
        height_px = int(pair_fig.get_size_inches()[1] * pair_fig.dpi)
        popup_canvas.setFixedSize(width_px, height_px)
        popup_layout.addWidget(popup_canvas)
        popup.adjustSize()

        # matplotlib event coords are bottom-left-origin (event.y measured
        # up from the canvas's bottom edge); Qt widget coords are
        # top-left-origin -- flip before mapping to a global position.
        local = QPoint(int(event.x), int(source_canvas.height() - event.y))
        global_pos = source_canvas.mapToGlobal(local) + QPoint(12, 12)
        # Clamp into the screen the popup would actually land on -- an
        # unclamped position can fall outside every screen's geometry on
        # a multi-monitor setup (e.g. a secondary display with a
        # negative-offset origin), which Qt reassigns to the primary
        # screen WITHOUT making the popup visible there (a real, reported
        # bug). screenAt(global_pos) can itself return None right when
        # this matters most, so fall back to the source canvas's own
        # screen, which is always real.
        screen = QApplication.screenAt(global_pos) or source_canvas.screen()
        if screen is not None:
            popup_size = popup.size()
            avail = screen.availableGeometry()
            x = max(avail.left(), min(global_pos.x(), avail.right() - popup_size.width()))
            y = max(avail.top(), min(global_pos.y(), avail.bottom() - popup_size.height()))
            global_pos = QPoint(x, y)
        popup.move(global_pos)
        popup.show()
        self._pair_popup = popup

    def _hide_pair_popup(self) -> None:
        if self._pair_popup is not None:
            self._pair_popup.close()
            self._pair_popup.deleteLater()
            self._pair_popup = None

    def closeEvent(self, event) -> None:
        self._hide_pair_popup()
        super().closeEvent(event)

    # Floor so a small result (few free_keys, mostly narrow figures)
    # doesn't open in a cramped window.
    _MIN_WIDTH_PX = 600
    _MIN_HEIGHT_PX = 470

    def _size_to_content(self) -> None:
        """
        Size the dialog window to what its own layout actually needs
        (Qt's adjustSize(), driven by the now-fully-built layout's real
        sizeHint) instead of a hand-guessed CHROME_WIDTH_PX/HEIGHT_PX
        constant added to the canvas size -- a fixed margin guess for
        QVBoxLayout margins + QScrollArea frame + the close-button row
        can't account for per-platform chrome variation the way
        adjustSize() does. Still capped to a fraction of the available
        screen so an oversized figure can't be asked to exceed the
        monitor.
        """
        self.adjustSize()
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        max_w = int(avail.width() * 0.92)
        max_h = int(avail.height() * 0.88)
        if self.width() > max_w or self.height() > max_h:
            self.resize(min(self.width(), max_w), min(self.height(), max_h))


class SobolS2DetailsDialog(QDialog):
    """
    Non-modal dialog showing eidos.lib.calibration_diagnostics.
    plot_sobol_s2_heatmap for a finished Sobol' sensitivity screen -- see
    TTAnalyzerWindow._on_show_sensitivity_details, the "Check S2"
    button next to the Sensitivity column's Sobol'/Morris controls.

    Same conventions as CalibrationDiagnosticsDialog: built fresh on
    every click (not cached -- cheap matplotlib Figures over an already-
    computed SobolSensitivityTrials, no re-simulation), shown via show()
    not exec() so more than one can be open at once, WA_DeleteOnClose
    frees the figure on close, and the canvas is a FIXED size (not
    resizable) for the identical reason documented on that class's own
    docstring.

    Click-and-hold a matrix cell to preview that cell's own scatter (the
    S1 effect scatter on the diagonal, the S2 interaction scatter off
    it); release to close it -- same click-and-hold-a-heatmap-cell
    convention as CalibrationDiagnosticsDialog._wire_pair_popup, adapted
    here for a SobolSensitivityTrials' own sample_x/sample_y rather than
    a CalibrationResult's pooled trials. Kept as its own duplicated
    implementation rather than a shared base class with
    CalibrationDiagnosticsDialog: the two dialogs' popups are wired to
    different axes/data (a correlation-heatmap cell vs. a Sobol' matrix
    cell) and have no other coupling worth abstracting over.
    """

    def __init__(self, result: SobolSensitivityTrials, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sobol' Second-Order Interaction Matrix")
        self._cell_popup: QWidget | None = None

        layout = QVBoxLayout(self)
        fig = diag.plot_sobol_s2_heatmap(result)
        if fig is None:
            self.resize(self._MIN_WIDTH_PX, self._MIN_HEIGHT_PX)
            layout.addWidget(QLabel(
                "Need at least 2 free (checked) parameters for a pairwise "
                "interaction matrix."
            ))
        else:
            canvas = FigureCanvas(fig)
            # Read AFTER FigureCanvas(fig) above, and self.resize() below
            # derived from these same ints, not a second independent
            # fig.dpi read.
            width_px = int(fig.get_size_inches()[0] * fig.dpi)
            height_px = int(fig.get_size_inches()[1] * fig.dpi)
            canvas.setFixedSize(width_px, height_px)
            scroll = QScrollArea()
            scroll.setWidget(canvas)
            scroll.setWidgetResizable(False)
            # See CalibrationDiagnosticsDialog's identical comment --
            # QScrollArea's own sizeHint() ignores a setWidgetResizable
            # (False) child's real size otherwise.
            _size_scroll_area_to_content(scroll, canvas.width(), canvas.height())
            layout.addWidget(scroll)
            self._wire_cell_popup(canvas, fig, result)
            self._wire_cell_crosshair(canvas, fig, result)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        if fig is not None:
            self._size_to_content()

    def _wire_cell_popup(self, canvas: FigureCanvas, fig, result: SobolSensitivityTrials) -> None:
        """
        Click-and-hold an OFF-DIAGONAL matrix cell to preview that
        pair's own S2 interaction scatter (plot_sensitivity_interaction_
        scatter) -- unlike the inline bar's own popup (TTAnalyzerWindow.
        _on_sensitivity_bar_clicked), which always shows key's own S1/ST
        effect scatter and never any per-partner S2 view at all, this
        dialog lets the viewer pick any pair directly off the matrix,
        including a pair with no finite S2 (its gray cell) -- the raw
        sampled points are still worth a look even when SALib's own
        index for that pair came back NaN.

        The diagonal is a no-op, deliberately: plot_sobol_s2_heatmap
        already draws it as N/A (S2 has no defined value for a parameter
        paired with itself -- see that function's docstring), so there
        is no S2 statistic for a diagonal cell's popup to be ABOUT. A
        click only ever surfaces the SAME statistic this matrix itself
        is showing, never a different-order one (e.g. that parameter's
        own S1 effect scatter) smuggled in through the click handler.

        Axes found by its "sobol_s2_heatmap" label, same convention
        _wire_pair_popup uses for "correlations_heatmap".
        """
        ax = next((a for a in fig.axes if a.get_label() == "sobol_s2_heatmap"), None)
        if ax is None:
            logger.warning(
                "_wire_cell_popup: no axes labeled 'sobol_s2_heatmap' in "
                "this figure -- cell-preview-on-click will not be available."
            )
            return
        keys = list(result.s1.keys())
        n = len(keys)

        def on_press(event):
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            j = int(round(event.xdata))
            i = int(round(event.ydata))
            if not (0 <= i < n and 0 <= j < n):
                return  # off-grid
            if i == j:
                return  # diagonal is N/A -- no S2 popup for a parameter paired with itself
            key_i, key_j = keys[i], keys[j]
            pair = (key_i, key_j) if key_i < key_j else (key_j, key_i)
            s2_val = result.s2.get(pair)
            title = (
                f"Sobol' S2 = {s2_val:.3g}±{result.s2_conf[pair]:.3g}"
                if s2_val is not None else None
            )
            cell_fig = diag.plot_sensitivity_interaction_scatter(
                key_i, key_j, result.sample_x[key_i], result.sample_x[key_j],
                result.sample_y, title=title,
            )
            self._show_cell_popup(cell_fig, canvas, event)

        def on_release(_event):
            self._hide_cell_popup()

        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("button_release_event", on_release)

    def _wire_cell_crosshair(self, canvas: FigureCanvas, fig, result: SobolSensitivityTrials) -> None:
        """
        Hover tracking over the S2 matrix, same two-part treatment
        CalibrationDiagnosticsDialog uses on the Correlations heatmap
        (_wire_row_crosshair + _wire_clickable_hover_affordance combined
        into one method here, since both track the same single axes):

        - A crosshair -- a full-width box across the hovered ROW plus a
          full-height box across the hovered COLUMN -- pale pink
          (#e8a0c4), same color _wire_row_crosshair uses. That one spans
          THREE separate axes (Correlations/Loadings/VIFs share one row
          axis); this dialog has a single axes, so both bars are drawn
          directly on it instead.
        - The single cell exactly at the crosshair's intersection (row
          AND col both valid, i.e. the actual cell under the cursor) --
          not merely near it -- gets its own thicker, brighter border on
          top, saturated magenta (#d6006f), same color/weight
          _wire_clickable_hover_affordance uses for its own "this one
          cell" highlight. Suppressed on the diagonal (row == col):
          _wire_cell_popup no longer opens anything for a diagonal cell
          (see its own docstring -- the diagonal is N/A, nothing to
          click through to), so highlighting it as clickable would be
          misleading. The row/column crosshair itself still draws as
          normal over a diagonal cell -- only this one cell-level
          "clickable" cue is withheld.

        Redraws only when the hovered (row, col) index actually
        changes, not on every raw motion_notify_event, to avoid
        re-rendering the whole figure on every tiny mouse jitter -- same
        convention _wire_row_crosshair uses.
        """
        ax = next((a for a in fig.axes if a.get_label() == "sobol_s2_heatmap"), None)
        if ax is None:
            logger.warning(
                "_wire_cell_crosshair: no axes labeled 'sobol_s2_heatmap' in "
                "this figure -- hover crosshair will not be available."
            )
            return
        n = len(result.s1)

        artists: list[Rectangle] = []
        last: dict[str, int | None] = {"row": None, "col": None}

        def _clear() -> None:
            for artist in artists:
                artist.remove()
            artists.clear()

        def _box(xa: float, xb: float, ya: float, yb: float, *, edgecolor: str, linewidth: float, zorder: int) -> None:
            x0, x1 = sorted((xa, xb))
            y0, y1 = sorted((ya, yb))
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="none", edgecolor=edgecolor, linewidth=linewidth, zorder=zorder,
            )
            rect.set_path_effects([withStroke(linewidth=3.5, foreground="white")])
            ax.add_patch(rect)
            artists.append(rect)

        def on_move(event) -> None:
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                if last["row"] is not None or last["col"] is not None:
                    _clear()
                    last["row"] = last["col"] = None
                    canvas.draw_idle()
                return
            j = int(round(event.xdata))
            i = int(round(event.ydata))
            row = i if 0 <= i < n else None
            col = j if 0 <= j < n else None
            if (row, col) == (last["row"], last["col"]):
                return
            last["row"], last["col"] = row, col

            _clear()
            if row is not None or col is not None:
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                if row is not None:
                    _box(x0, x1, row - 0.5, row + 0.5, edgecolor="#e8a0c4", linewidth=1.6, zorder=20)
                if col is not None:
                    _box(col - 0.5, col + 0.5, y0, y1, edgecolor="#e8a0c4", linewidth=1.6, zorder=20)
                if row is not None and col is not None and row != col:
                    _box(
                        col - 0.5, col + 0.5, row - 0.5, row + 0.5,
                        edgecolor="#d6006f", linewidth=2.4, zorder=30,
                    )
            canvas.draw_idle()

        def on_leave(_event) -> None:
            if last["row"] is not None or last["col"] is not None:
                _clear()
                last["row"] = last["col"] = None
                canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("figure_leave_event", on_leave)

    def _show_cell_popup(self, cell_fig, source_canvas: FigureCanvas, event) -> None:
        """Show cell_fig in a small frameless popup near the click -- see
        CalibrationDiagnosticsDialog._show_pair_popup's identical
        implementation for the multi-monitor clamping reasoning."""
        self._hide_cell_popup()

        popup = QWidget(self, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        popup.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(1, 1, 1, 1)

        popup_canvas = FigureCanvas(cell_fig)
        width_px = int(cell_fig.get_size_inches()[0] * cell_fig.dpi)
        height_px = int(cell_fig.get_size_inches()[1] * cell_fig.dpi)
        popup_canvas.setFixedSize(width_px, height_px)
        popup_layout.addWidget(popup_canvas)
        popup.adjustSize()

        local = QPoint(int(event.x), int(source_canvas.height() - event.y))
        global_pos = source_canvas.mapToGlobal(local) + QPoint(12, 12)
        screen = QApplication.screenAt(global_pos) or source_canvas.screen()
        if screen is not None:
            popup_size = popup.size()
            avail = screen.availableGeometry()
            x = max(avail.left(), min(global_pos.x(), avail.right() - popup_size.width()))
            y = max(avail.top(), min(global_pos.y(), avail.bottom() - popup_size.height()))
            global_pos = QPoint(x, y)
        popup.move(global_pos)
        popup.show()
        self._cell_popup = popup

    def _hide_cell_popup(self) -> None:
        if self._cell_popup is not None:
            self._cell_popup.close()
            self._cell_popup.deleteLater()
            self._cell_popup = None

    def closeEvent(self, event) -> None:
        self._hide_cell_popup()
        super().closeEvent(event)

    _MIN_WIDTH_PX = 600
    _MIN_HEIGHT_PX = 470

    def _size_to_content(self) -> None:
        """See CalibrationDiagnosticsDialog._size_to_content's identical
        docstring -- same reasoning, same replacement of a guessed
        CHROME_WIDTH_PX/HEIGHT_PX constant with Qt's own adjustSize()."""
        self.adjustSize()
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        max_w = int(avail.width() * 0.92)
        max_h = int(avail.height() * 0.88)
        if self.width() > max_w or self.height() > max_h:
            self.resize(min(self.width(), max_w), min(self.height(), max_h))


if __name__ == "__main__":
    # Dev-only layout preview: opens CalibrationDiagnosticsDialog against a
    # synthetic CalibrationResult shaped like a real 12-free_key/500-trial
    # Auto Fit run, WITHOUT running calibrate() itself. A real 12-free_key
    # Auto Fit takes a few minutes per attempt, unworkable as a "tweak
    # layout code, check the result" loop; this builds a synthetic
    # CalibrationResult with the same shape and roughly the same
    # correlation structure a real run shows (a cda/air_density trade-off,
    # a rider/bike mass-split trade-off -- both real findings), so the
    # dialog opens in about a second showing something worth looking at.
    #
    # Run: `python -m eidos.apps.analyzer.dialogs`
    #
    # Note: like the rest of eidos.apps.analyzer, this needs a real
    # PySide6/Qt runtime (a plain `pip install PySide6` without the
    # system-level OpenGL/EGL libraries it links against at import time
    # will fail).
    import sys
    import types

    import numpy as np

    from core.calibrator import bounds_from_schema
    from core.simulators import DEFAULT_SIMULATOR_KEY

    rng = np.random.default_rng(0)
    free_keys = [
        "cda", "mu", "brake_usability", "wind_speed", "wind_direction", "air_density",
        "crr", "rider_weight", "bike_weight", "f_max",
        "brake_lookahead", "gravity_accel",
    ]
    n_trials = 500

    all_trials = []
    for _ in range(n_trials):
        x = {}

        # cda/air_density trade-off (product held roughly constant) --
        # mirrors the real run's Direction 1-ish finding.
        cda_lo, cda_hi = bounds_from_schema(DEFAULT_SIMULATOR_KEY, "cda")
        cda = rng.uniform(cda_lo, cda_hi)
        ad_lo, ad_hi = bounds_from_schema(DEFAULT_SIMULATOR_KEY, "air_density")
        ad_center = (ad_lo + ad_hi) / 2
        cda_center = (cda_lo + cda_hi) / 2
        x["cda"] = cda
        x["air_density"] = float(np.clip(
            ad_center * cda_center / cda + rng.normal(0, 0.01), ad_lo, ad_hi,
        ))

        # rider/bike mass-split trade-off (sum held roughly constant).
        rw_lo, rw_hi = bounds_from_schema(DEFAULT_SIMULATOR_KEY, "rider_weight")
        bw_lo, bw_hi = bounds_from_schema(DEFAULT_SIMULATOR_KEY, "bike_weight")
        total_center = (rw_lo + rw_hi) / 2 + (bw_lo + bw_hi) / 2
        rider = rng.uniform(rw_lo, rw_hi)
        x["rider_weight"] = rider
        x["bike_weight"] = float(np.clip(total_center - rider, bw_lo, bw_hi))

        # wind_direction: a tight cluster straddling the 0/360 seam (a north
        # wind), baked into every trial the same way cda/air_density and
        # rider/bike_weight are above -- exercises _pool_trials' unwrap-
        # around-x_best fix (see calibration_diagnostics.py's docstring)
        # without needing a real north-wind Auto Fit run to test against.
        # Left as plain uniform noise like every other key below, this
        # would only straddle the seam by chance.
        x["wind_direction"] = float(rng.normal(358.0, 4.0)) % 360.0

        # Every other free_key: independent, uniform within its own
        # schema bounds -- no engineered structure, just plausible noise.
        for k in free_keys:
            if k in x:
                continue
            lo, hi = bounds_from_schema(DEFAULT_SIMULATOR_KEY, k)
            x[k] = rng.uniform(lo, hi)

        vec = np.array([x[k] for k in free_keys])
        fun = 0.3 + abs(rng.normal(0, 0.03))
        all_trials.append(types.SimpleNamespace(x=vec, fun=fun, success=True, de_success=True))

    x_matrix = np.array([t.x for t in all_trials])
    best = min(all_trials, key=lambda t: t.fun)

    result = CalibrationResult(
        free_keys=free_keys,
        x_best=np.asarray(best.x),
        rmse_mps=float(best.fun),
        physics_overrides={},
        n_trials=len(all_trials),
        n_converged=len(all_trials),
        x_std={k: float(np.std(x_matrix[:, i])) for i, k in enumerate(free_keys)},
        all_trials=all_trials,
    )

    app = QApplication(sys.argv)
    dlg = CalibrationDiagnosticsDialog(result)
    dlg.exec()
