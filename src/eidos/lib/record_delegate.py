#########################
# record_delegate.py
#########################
import math
from typing import Optional

from PySide6.QtCore import QEvent, QMargins, QModelIndex, QObject, QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QWidget,
)


class RecordListDelegate(QStyledItemDelegate):
    """
    Custom list delegate that renders each strategy record with a color marker,
    checkbox, radio button, and a set of value columns (segment/seed, finish
    time, strategy set, run id) in a single row -- no repeated field-name labels.
    """
    CHECKBOX_SIZE = 16
    RADIO_SIZE = 14
    INNER_GAP = 5
    MARGIN = QMargins(1, 1, 1, 1)

    # (header, alignment, diagonal_header) for each value column, in draw
    # order. Alignment applies to the data cells only. diagonal_header marks
    # columns whose header (built by the Viewer) is drawn on a rising
    # diagonal so the full title fits above a column sized tight to its short
    # data (N Seg/Seed) -- the rest draw their header horizontally, and are
    # widened as needed to fit their full title (see _compute_column_widths).
    COLUMNS = [
        ("N Seg", Qt.AlignmentFlag.AlignLeft, True),
        ("Seed", Qt.AlignmentFlag.AlignLeft, True),
        ("Time", Qt.AlignmentFlag.AlignRight, False),
        ("Strategy Set Dir", Qt.AlignmentFlag.AlignLeft, False),
        ("Run Set ID", Qt.AlignmentFlag.AlignLeft, False),
    ]
    COLUMN_PADDING = 4
    # Small, subdued style for the Viewer's column-header row -- matches the
    # app's existing color:#999999/9px label convention (e.g. the Analyzer's
    # Sobol/Morris parameter labels).
    HEADER_FONT_PIXEL_SIZE = 9
    # QFontMetrics.horizontalAdvance() on the full string can undershoot what
    # elidedText()/drawText() actually need by a couple of px (kerning/right
    # bearing on the last glyph) -- a small buffer keeps max-width values from
    # eliding themselves.
    MEASURE_SLOP = 3

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.TIME_KEYS = ['output', 'results', 'kpis', 'total_time_s']
        self.MARKER_SIZE = 10
        self.MARKER_PADDING = 5
        self.LIST_FONT_SIZE = 13

        # Column widths are sized to the widest value actually present in the
        # model (computed once here, since the delegate is recreated alongside
        # the model on every list rebuild), so columns stay as tight as
        # possible instead of using generous fixed guesses.
        model = parent.model() if parent is not None and hasattr(parent, 'model') else None
        self._column_widths = self._compute_column_widths(model)

    def _compute_column_widths(self, model) -> list:
        """Size each column to fit its widest value. Columns with a horizontal
        header (diagonal_header=False) also enforce their header label's width as a
        minimum, so the full title always fits; diagonal-header columns (N Seg/Seed)
        are sized to their data alone, since the diagonal rise makes room for the title."""
        font = QFont("Menlo", self.LIST_FONT_SIZE)
        metrics = QFontMetrics(font)
        header_font = QFont()
        header_font.setPixelSize(self.HEADER_FONT_PIXEL_SIZE)
        header_metrics = QFontMetrics(header_font)
        STRATEGY_ATTR_ROLE = Qt.ItemDataRole.UserRole + 1

        content_widths = [
            0 if diagonal else header_metrics.horizontalAdvance(title)
            for title, _align, diagonal in self.COLUMNS
        ]
        if model is not None:
            for row in range(model.rowCount()):
                meta = model.index(row, 0).data(STRATEGY_ATTR_ROLE)
                for c, value in enumerate(self.column_values(meta)):
                    content_widths[c] = max(content_widths[c], metrics.horizontalAdvance(value))

        return [w + 2 * self.COLUMN_PADDING + self.MEASURE_SLOP for w in content_widths]

    def prefix_width(self) -> int:
        """Width of the left margin + color marker + checkbox + radio button, before the value columns start.

        Assumes a marker is drawn (true for every real record); used by the
        Viewer to align its column-header row with these columns.
        """
        return (
            5 +
            self.MARKER_SIZE + self.MARKER_PADDING +
            self.CHECKBOX_SIZE + self.INNER_GAP +
            self.RADIO_SIZE + self.INNER_GAP + 5
        )

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Render the record row: color marker, checkbox, radio button, and value columns."""
        STRATEGY_ATTR_ROLE = Qt.ItemDataRole.UserRole + 1
        ACTIVE_STATE_ROLE = Qt.ItemDataRole.UserRole + 2

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        model = index.model()
        meta = index.data(STRATEGY_ATTR_ROLE)
        is_selected = option.state & QStyle.StateFlag.State_Selected

        # Background color: selected > latest seed-0 > other seed-0 > default
        is_seed0 = meta and str(meta.get('Seed_file')) == '0'
        is_latest_seed0 = False
        if is_seed0:
            current_run_id = str(meta.get('run_set_id', ''))
            if current_run_id == getattr(model, 'latest_seed0_run_id', ""):
                is_latest_seed0 = True

        if is_selected:
            bg_color = QColor("#EF7E1B")
            current_text_color = QColor("#FFFFFF")
        elif is_latest_seed0:
            bg_color = QColor("#029BB6")
            current_text_color = QColor("#FFFFFF")
        elif is_seed0:
            bg_color = QColor("#4C7EA0")
            current_text_color = QColor("#FFFFFF")
        else:
            bg_color = option.palette.color(QPalette.ColorRole.Base)
            current_text_color = option.palette.color(QPalette.ColorRole.Text)

        painter.fillRect(option.rect, bg_color)

        x = option.rect.left() + 5

        # 1. Color marker
        line_color = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(line_color, QColor) and line_color.isValid():
            marker_rect = QRect(x, option.rect.center().y() - self.MARKER_SIZE // 2, self.MARKER_SIZE, self.MARKER_SIZE)
            painter.setBrush(QBrush(line_color))
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawRect(marker_rect)
            x += self.MARKER_SIZE + self.MARKER_PADDING

        # 2. Checkbox
        checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
        checkbox_rect = QRect(x, option.rect.center().y() - self.CHECKBOX_SIZE // 2, self.CHECKBOX_SIZE, self.CHECKBOX_SIZE)

        check_opt = QStyleOptionButton()
        check_opt.rect = checkbox_rect
        check_opt.state = QStyle.StateFlag.State_Enabled | (QStyle.StateFlag.State_On if checked else QStyle.StateFlag.State_Off)
        QApplication.style().drawControl(QStyle.ControlElement.CE_CheckBox, check_opt, painter)
        x += self.CHECKBOX_SIZE + self.INNER_GAP

        # 3. Radio button (active record selection)
        is_active = index.data(ACTIVE_STATE_ROLE)
        radio_rect = QRect(x, option.rect.center().y() - self.RADIO_SIZE // 2, self.RADIO_SIZE, self.RADIO_SIZE)

        painter.setPen(QPen(QColor("#CCCCCC"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(radio_rect)

        if is_active:
            painter.setBrush(QColor("#00FE77"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(radio_rect.adjusted(3, 3, -3, -3))

        x += self.RADIO_SIZE + self.INNER_GAP + 5

        # 4. Value columns (no repeated field-name labels; a thin separator marks each boundary)
        font = QFont("Menlo", self.LIST_FONT_SIZE)
        metrics = QFontMetrics(font)
        painter.setFont(font)
        sep_pen = QPen(QColor(255, 255, 255, 60), 1)

        for (_, align, _diagonal), width, value in zip(self.COLUMNS, self._column_widths, self.column_values(meta)):
            col_rect = QRect(x, option.rect.top(), width, option.rect.height())
            inner_rect = col_rect.adjusted(self.COLUMN_PADDING, 0, -self.COLUMN_PADDING, 0)
            elided = metrics.elidedText(value, Qt.TextElideMode.ElideRight, inner_rect.width())
            painter.setPen(current_text_color)
            painter.drawText(inner_rect, align | Qt.AlignmentFlag.AlignVCenter, elided)

            x += width
            painter.setPen(sep_pen)
            painter.drawLine(x, option.rect.top() + 4, x, option.rect.bottom() - 4)

        painter.restore()

    @staticmethod
    def column_values(meta: Optional[dict]) -> list:
        """Return display strings for COLUMNS, in order, from a STRATEGY_ATTR_ROLE
        dict. meta is None when the index has no data yet (e.g. a transient
        paint during a model reset) -- blank columns in that case only.
        Once meta is a real STRATEGY_ATTR_ROLE dict, every field below is
        structurally guaranteed (see StrategyRecordModel.data), so a missing
        one is a real bug, not a display default."""
        if not meta:
            return ['', '', '', '', '']
        return [
            str(meta["N_seg_file"]),
            str(meta["Seed_file"]),
            f"{meta['time']:.2f}s",
            meta["strategy_set_dir"],
            meta["run_set_id"],
        ]

    def control_rects(self, row_rect: QRect, index: QModelIndex):
        """Return (checkbox_rect, radio_rect) for a row, matching the paint() layout."""
        line_color = index.data(Qt.ItemDataRole.DecorationRole)
        marker_w = (self.MARKER_SIZE + self.MARKER_PADDING) if isinstance(line_color, QColor) and line_color.isValid() else 0

        x_start = row_rect.left() + 5
        x_checkbox = x_start + marker_w
        x_radio = x_checkbox + self.CHECKBOX_SIZE + self.INNER_GAP

        checkbox_rect = QRect(x_checkbox, row_rect.center().y() - self.CHECKBOX_SIZE // 2, self.CHECKBOX_SIZE, self.CHECKBOX_SIZE)
        radio_rect = QRect(x_radio, row_rect.center().y() - self.RADIO_SIZE // 2, self.RADIO_SIZE, self.RADIO_SIZE)
        return checkbox_rect, radio_rect

    def hits_control(self, row_rect: QRect, index: QModelIndex, pos) -> bool:
        """True if pos (row-relative or view-relative, matching row_rect's coordinate space) falls on the checkbox or radio button."""
        checkbox_rect, radio_rect = self.control_rects(row_rect, index)
        return checkbox_rect.contains(pos) or radio_rect.contains(pos)

    def editorEvent(self, event: QEvent, model, option: QStyleOptionViewItem, index: QModelIndex) -> bool:
        """Handle mouse clicks on the checkbox and radio button areas."""
        if event.type() == QEvent.Type.MouseButtonRelease and index.isValid():
            pos = event.pos()
            checkbox_rect, radio_rect = self.control_rects(option.rect, index)

            # Checkbox click
            if checkbox_rect.contains(pos):
                new_state = Qt.CheckState.Unchecked if index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked else Qt.CheckState.Checked
                model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)
                return True

            # Radio button click
            if radio_rect.contains(pos):
                current_active = index.data(Qt.ItemDataRole.UserRole + 2)
                model.setData(index, not current_active, Qt.ItemDataRole.UserRole + 2)
                return True

        return super().editorEvent(event, model, option, index)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """
        Compute row size from the fixed widget/column widths.
        Layout: left margin(5) + marker + checkbox + radio + value columns + right margin(10).
        """
        line_color = index.data(Qt.ItemDataRole.DecorationRole)
        marker_w = (self.MARKER_SIZE + self.MARKER_PADDING) if isinstance(line_color, QColor) and line_color.isValid() else 0

        columns_width = sum(self._column_widths)

        total_width = (
            5 +
            marker_w +
            self.CHECKBOX_SIZE + self.INNER_GAP +
            self.RADIO_SIZE + self.INNER_GAP + 5 +
            columns_width +
            10
        )
        return QSize(total_width, 35)


class RecordListHeader(QWidget):
    """Column-header strip for a RecordListDelegate.

    Columns marked diagonal_header (N Seg/Seed -- see RecordListDelegate.COLUMNS)
    draw their title on a rising diagonal (bottom-left to upper-right) so the
    full title fits above a column sized tight to its short data; the rest
    draw horizontally, left-aligned, in columns widened to fit their title
    (RecordListDelegate._compute_column_widths enforces that minimum).
    """
    ANGLE_DEG = 30
    # Room reserved past the last column, in case a diagonal column ever ends
    # up last (today N Seg/Seed -- the only diagonal ones -- come first, so
    # their rise has room to their right already).
    RIGHT_OVERHANG = 20

    def __init__(self, delegate: RecordListDelegate, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._delegate = delegate

        font = QFont()
        font.setPixelSize(delegate.HEADER_FONT_PIXEL_SIZE)
        self._font = font
        metrics = QFontMetrics(font)
        ascent, descent = metrics.ascent(), metrics.descent()
        theta = math.radians(self.ANGLE_DEG)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # Exact pixel extent above/below the shared baseline, across every
        # column's glyph corners (rotated for diagonal columns), so the strip
        # is sized to the true rendered bounds -- no arbitrary margin.
        top_extent = 0.0
        bottom_extent = 0.0
        for title, _align, diagonal in delegate.COLUMNS:
            if diagonal:
                w = metrics.horizontalAdvance(title)
                for lx, ly in ((0, -ascent), (w, -ascent), (0, descent), (w, descent)):
                    py = -lx * sin_t + ly * cos_t  # corner's y-offset from baseline after rotate(-ANGLE_DEG)
                    top_extent = max(top_extent, -py)
                    bottom_extent = max(bottom_extent, py)
            else:
                top_extent = max(top_extent, ascent)
                bottom_extent = max(bottom_extent, descent)

        self._baseline_y = int(math.ceil(top_extent))
        self.setFixedHeight(self._baseline_y + int(math.ceil(bottom_extent)))
        self.setFixedWidth(delegate.prefix_width() + sum(delegate._column_widths) + self.RIGHT_OVERHANG)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)
        painter.setPen(QColor("#999999"))

        x = self._delegate.prefix_width()
        baseline_y = self._baseline_y
        for (title, _align, diagonal), width in zip(self._delegate.COLUMNS, self._delegate._column_widths):
            painter.save()
            painter.translate(x + self._delegate.COLUMN_PADDING, baseline_y)
            if diagonal:
                painter.rotate(-self.ANGLE_DEG)
            # Same baseline anchor for every column (diagonal or flat) keeps
            # them all sitting on one shared bottom edge.
            painter.drawText(0, 0, title)
            painter.restore()
            x += width

        painter.end()
