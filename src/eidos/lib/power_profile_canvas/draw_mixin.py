"""
eidos.lib.power_profile_canvas.draw_mixin -- QPainter-based rendering.

_PowerProfileDrawMixin is one of three mixins combined into
PowerProfileCanvas (see canvas.py) -- see data_mixin.py's docstring for
the general caveat about mixins not being standalone classes. The
largest of the three: all the actual paintEvent / QPainter drawing code.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF

import eidos.lib.visual_profile as vp
from eidos.lib.power_profile_canvas.plot_limits import (
    ACTIVE_COLOR_HEX,
    DESIGNER_COLOR_HEX,
    PlotLimits,
)

logger = logging.getLogger(__name__)

# mypy can't see across mixins on its own: this mixin references attributes
# (self.limits, self.MARGIN_LEFT, self.map_data_to_widget, ...) that live on
# PowerProfileCanvas or the other two mixins, not on this class itself. The
# TYPE_CHECKING-only base below tells mypy "for type-checking purposes,
# assume self has this interface" without changing anything at runtime (the
# import and base class only exist while TYPE_CHECKING is True). It points
# at _canvas_selftype's stand-in rather than the real PowerProfileCanvas
# because canvas.py imports this module at load time to build the class --
# importing the real class back here would be a genuine cycle that mypy
# can't resolve even under TYPE_CHECKING. See _canvas_selftype.py's
# docstring for the full explanation.
if TYPE_CHECKING:
    from eidos.lib.power_profile_canvas._canvas_selftype import _CanvasSelfType
    _DrawMixinBase = _CanvasSelfType
else:
    _DrawMixinBase = object


class _PowerProfileDrawMixin(_DrawMixinBase):
    """QPainter-based rendering.

    Not a usable class on its own -- see data_mixin.py's docstring.
    """

    def paintEvent(self, event):
        """
        - Graph background: render all Selected records (e.g. 51 items)
        - Dynamic HUD/lines: synchronised to the single Active record (radio button)
        """
        try:
            # --- 1. Static graph cache (all Selected records) ---
            if getattr(self, '_cache_dirty', True) or not hasattr(self, '_graph_cache') or not self._graph_cache:
                self._rebuild_graph_cache()

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw the background (all graphs) in one shot
            painter.drawPixmap(0, 0, self._graph_cache)
            
            # --- 2. Static graph overlay (Active record) ---
            active_idx = next((i for i, r in enumerate(self.records) if r.get('is_active')), None)
            if active_idx is not None and active_idx < len(self.data_cache):
                active_data = self.data_cache[active_idx].copy()
                active_data['COLOR_HEX'] = ACTIVE_COLOR_HEX
                self.draw_data_on_all_subplots(painter, [active_data])

            # --- 3. Dynamic overlay (Active record) ---
            try:
                active_idx = next((i for i, r in enumerate(self.records) if r.get('is_active')), None)
                # Draw HUD only when an Active record exists and the cache has been built
                if active_idx is not None and active_idx < len(self.data_cache):
                    # Use active_idx instead of the hardwired data_cache[0]
                    active_data = self.data_cache[active_idx]
                    
                    # Draw the vertical cursor line
                    self._draw_cursor_line(painter)
                    
                    # Draw HUD (physical quantities for the Active record overlaid on the graph)
                    # Passing active_data ensures the correct single record's values appear
                    # even when 51 records are shown simultaneously.
                    self._draw_distributed_huds(painter, active_data)
                else:
                    # Guard: draw at least the cursor line when there is no Active record
                    if hasattr(self, 'cursor_x'):
                        self._draw_cursor_line(painter)

            except Exception as e:
                logger.error("Error in dynamic drawing: %s", e)

            # --- 4. External drawing (Designer etc.) ---
            if getattr(self, 'external_painter', None):
                self.external_painter(painter)

            painter.end()

        except Exception as e:
            if 'painter' in locals() and painter.isActive():
                painter.end()
            logger.error("FATAL: Error in paintEvent: %s", e)

    def _rebuild_graph_cache(self):
        """Render the heavy graph drawing into a Pixmap and update the cache."""
        self._graph_cache = QPixmap(self.size())
        self._graph_cache.fill(Qt.GlobalColor.white)
        
        cache_painter = QPainter(self._graph_cache)
        cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Prepare draw data (retrieve sorted list)
        sorted_data_cache, sorted_records = self._get_sorted_plot_data()

        # 2. Compute segment boundaries once (eliminates 5× redundant computation)
        boundary_x = self._get_segment_boundaries()

        # 3. Compute drawing area (always, regardless of data presence)
        full_plot_rect, _, plot_names = self._update_layout_geometry()

        # 4. Draw each subplot
        for i, name in enumerate(plot_names):
            current_rect = self.subplot_rects[name]
            current_limits = self.limits.get(name, self.limits['Course'])
            is_last = (i == len(plot_names) - 1)

            # (A) Draw the frame (axes, grid, segment dividers)
            self._draw_subplot_base(cache_painter, current_rect, name, current_limits, is_last, boundary_x)

            # (B) Plot the data
            if sorted_data_cache:
                self._draw_subplot_data(cache_painter, current_rect, name, current_limits, 
                                        sorted_data_cache, sorted_records)

            # (C) Draw the legend
            self._draw_subplot_legend(cache_painter, current_rect, name)

        # 5. Show message when there is no data at all
        if not self.data_cache and not self.external_painter:
             cache_painter.setPen(Qt.GlobalColor.gray)
             cache_painter.drawText(full_plot_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, "No background data")

        cache_painter.end()
        self._cache_dirty = False

    def request_refresh(self):
        """Request a cache rebuild without changing self.records -- called after set_target_records/set_axis_mode update the data cache, and after Designer's own calculate_plot_limits changes. Resize triggers a rebuild separately, via _cache_dirty set directly in resizeEvent."""
        self._cache_dirty = True
        self.update()

    def draw_data_on_all_subplots(self, painter: QPainter, data_list: list):
        """Draw data passed from outside (Designer etc.) across all subplots in logical order."""
        plot_names = ['Course', 'Power', 'Velocity', 'WPrime', 'Cumulative']
        data_tuple = tuple(data_list)
        dummy_records = tuple([None] * len(data_list))
        
        for name in plot_names:
            rect = self.subplot_rects.get(name)
            limits = self.limits.get(name, self.limits['Course'])
            if not rect:
                continue
            self._draw_segment_boundaries(painter, rect, limits, data_tuple)
            self._draw_subplot_data(painter, rect, name, limits, data_tuple, dummy_records)

    # ------------------------------------------------
    # 5. Primitive Drawing
    # ------------------------------------------------
    def _draw_subplot_base(self, painter: QPainter, rect: QRect, plot_name: str, 
                           limits: PlotLimits, draw_x_axis: bool, boundary_x: list):
        """Draw the axes, grid, and segment boundary lines passed as an argument."""
        # 1. Draw axes and grid
        self._draw_subplot_axes(painter, rect, plot_name, limits, draw_x_axis)
        
        # 2. Draw segment dividers (vertical dashed lines)
        alpha_value = 100
        segment_color = QColor(Qt.GlobalColor.gray)
        segment_color.setAlpha(alpha_value)
        pen = QPen(segment_color, 1)
        pen.setDashPattern([5.0, 3.0])
        painter.setPen(pen)

        for x_edge in boundary_x:
            if x_edge < limits.X_min or x_edge > limits.X_max: 
                continue
            # Convert X coordinate to widget space (Y is irrelevant here; fix to Y_min)
            point = self.map_data_to_widget(x_edge, limits.Y1_min, rect, limits)
            painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))

    def _draw_subplot_axes(self, painter: QPainter, rect: QRect, plot_name: str, limits: PlotLimits, draw_x_axis: bool):
        """
        Draw tick marks, tick labels, and axis grid lines for a single subplot.

        Renders X ticks along the bottom, Y1 ticks on the left, and (for the Course
        subplot only) Y2 ticks for altitude on the right.
        """
        # --- 0. Title definitions and common settings ---
        Y_AXIS_TITLES = {
            'Course': ('Grade (%)', 'Altitude (m)'), 
            'Power': ('Power (W)', None),
            'Velocity': ('Velocity (km/h)', None),
            'WPrime': ("W' Balance (J)", None),
            'Cumulative': ('Dist (km)' if self.is_time_mode else 'Time (min)', None)
        }
        title_y1, title_y2 = Y_AXIS_TITLES[plot_name]
        painter.setFont(QFont("Arial", vp.TICK_FONTSIZE))

        # --- 1. Axis lines (left and bottom) ---
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawLine(rect.topLeft(), rect.bottomLeft())  # left axis
        
        if draw_x_axis:
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())  # bottom axis
            
            # X-axis ticks and labels
            x_ticks = self._get_x_ticks(limits)
            for x_tick in x_ticks:
                point_x = self.map_data_to_widget(x_tick, limits.Y1_min, rect, limits).x()
                label = f"{x_tick:.0f}" 
                painter.drawText(int(point_x - 25), rect.bottom() + 5, 50, 20, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, label)

            # X-axis title
            x_title = "Time (sec)" if self.is_time_mode else "Distance (m)"
            painter.drawText(
                QRect(rect.left(), rect.bottom() + 20, rect.width(), self.MARGIN_BOTTOM - 20),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, x_title
            )
        else:
            # Bottom edge of intermediate subplots
            painter.setPen(QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.SolidLine))
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        # --- 2. Y1 axis (left): ticks, grid, and labels ---
        y1_ticks, _ = self._get_y_ticks(plot_name, limits)
        for y_tick in y1_ticks:
            point = self.map_data_to_widget(limits.X_min, y_tick, rect, limits)
            
            # Grid lines
            if plot_name == 'Course' and abs(y_tick) < 0.1: 
                painter.setPen(QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine))
            else:
                painter.setPen(QPen(Qt.GlobalColor.gray, 0.5, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(rect.left(), point.y()), QPointF(rect.right(), point.y()))
            
            # Labels
            painter.setPen(Qt.GlobalColor.black)
            text_rect = QRect(rect.left() - self.MARGIN_LEFT, int(point.y() - 5), self.MARGIN_LEFT - 5, 10)
            
            if plot_name in ['Power', 'WPrime']:
                label = f"{y_tick:.0f}"
            else:
                label = f"{y_tick:.1f}"
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

        # Y1 axis title (rotated)
        painter.save()
        painter.translate(rect.left() - self.MARGIN_LEFT + 15, rect.center().y())
        painter.rotate(-90)
        painter.setFont(QFont("Arial", vp.YLABEL_FONTSIZE))
        painter.drawText(QRect(-rect.height()//2, -10, rect.height(), 20), Qt.AlignmentFlag.AlignCenter, title_y1)
        painter.restore()
        
        # --- 3. Y2 axis (right — Course only) ---
        if title_y2 and plot_name == 'Course':
            _, y2_ticks = self._get_y_ticks(plot_name, limits)
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawLine(rect.topRight(), rect.bottomRight())
            
            for y_tick in y2_ticks:
                point = self.map_data_to_widget(limits.X_max, y_tick, rect, limits, use_y2=True)
                painter.drawLine(QPointF(point.x() - 4, point.y()), QPointF(point.x(), point.y()))
                text_rect = QRect(rect.right() + 5, int(point.y() - 5), self.MARGIN_RIGHT - 5, 10)
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{y_tick:.0f}")

            # Y2 axis title
            painter.save()
            x_center = rect.right() + self.MARGIN_RIGHT * 0.7 
            painter.translate(QPointF(x_center, rect.center().y()))
            painter.rotate(90)
            painter.setFont(QFont("Arial", vp.YLABEL_FONTSIZE))
            painter.drawText(QRect(int(-rect.height() / 2), int(-self.MARGIN_RIGHT / 2), rect.height(), self.MARGIN_RIGHT), Qt.AlignmentFlag.AlignCenter, title_y2)
            painter.restore()

    def _draw_subplot_data(self, painter: QPainter, rect: QRect, plot_name: str,
                           limits: PlotLimits, sorted_data_cache: Tuple[Dict[str, Any], ...],
                           sorted_records: Tuple[Optional[Dict[str, Any]], ...]):
        """Draw data lines only.

        sorted_records isn't actually read below (kept for signature
        symmetry with sorted_data_cache) -- draw_data_on_all_subplots
        relies on that to pass an all-None placeholder tuple when called
        with externally-supplied data that has no per-record metadata.
        """
        
        for i, data in reversed(list(enumerate(sorted_data_cache))):
            line_color = QColor(data['COLOR_HEX'])
            line_color_target = QColor(line_color.red(), line_color.green(), line_color.blue(), 127)

            # --- 2. Retrieve arrays from the Data Logic layer ---
            x_array, y1_array, y2_array = self._get_plot_arrays(data, plot_name)
            if x_array is None or x_array.size == 0: continue

            # --- Execute actual drawing (drawing commands only; no computation here) ---
            if plot_name == 'Course':
                # Y1: grade (dotted), Y2: altitude (solid)
                self._draw_line_profile(painter, rect, limits, x_array, y1_array, limits.Y1_min, 
                                        line_color, vp.LINE_WIDTH_PROF, Qt.PenStyle.DotLine)
                self._draw_line_profile(painter, rect, limits, x_array, y2_array, limits.Y2_min, 
                                        line_color, vp.LINE_WIDTH_PROF, Qt.PenStyle.SolidLine, use_y2=True)
            
            elif plot_name == 'Power':
                # Target power (step profile)
                self._draw_step_profile(painter, rect, limits, data['TargetL_m'], data['TargetP_W'], 
                                        line_color_target, vp.LINE_WIDTH_TARGET, 
                                        actual_dist=data['DISTANCE'], actual_time=data['TIME'])
                # Actual power
                self._draw_line_profile(painter, rect, limits, x_array, y1_array, limits.Y1_min, 
                                        line_color, vp.LINE_WIDTH_PROF, Qt.PenStyle.DotLine)
                # CP reference line
                if data['CP_REF'] is not None:
                    self._draw_horizontal_line(painter, rect, limits, data['CP_REF'], line_color)

            elif plot_name == 'Velocity':
                # Computed velocity in km/h
                self._draw_line_profile(painter, rect, limits, x_array, y1_array, limits.Y1_min, 
                                        line_color, vp.LINE_WIDTH_PROF, Qt.PenStyle.SolidLine)

            elif plot_name == 'WPrime':
                # W' Balance
                self._draw_line_profile(painter, rect, limits, x_array, y1_array, limits.Y1_min, 
                                        line_color, vp.LINE_WIDTH_PROF, Qt.PenStyle.SolidLine)
                # W' Max reference line
                if data['W_PRIME_MAX'] is not None:
                    self._draw_horizontal_line(painter, rect, limits, data['W_PRIME_MAX'], line_color)
                # Zero reference line
                self._draw_horizontal_line(painter, rect, limits, 0.0, Qt.GlobalColor.black, Qt.PenStyle.DotLine, 1)

            elif plot_name == 'Cumulative':
                # Computed cumulative value (distance or time)
                self._draw_line_profile(painter, rect, limits, x_array, y1_array, limits.Y1_min, 
                                        line_color, vp.LINE_WIDTH_PROF, Qt.PenStyle.SolidLine)

    def _draw_line_profile(self, painter: QPainter, rect: QRect, limits: PlotLimits, 
                           x_data: np.ndarray, y_data: np.ndarray, y_ref: float, 
                           color: QColor, width: float, style: Qt.PenStyle, use_y2: bool = False):
        """Draw a continuous polyline for the given Y data array within plot_rect."""
        if x_data.size <= 1 or y_data.size != x_data.size:
            return

        painter.setPen(QPen(color, width, style))

        # --- 1. Motion-retention logic (unchanged) ---
        sampling_density = 2.0
        target_points = max(1.0, rect.width() * sampling_density)
        
        if x_data.size > target_points:
            step = max(1, int(x_data.size / target_points))
            x_draw = x_data[::step]
            y_draw = y_data[::step]
            if (x_data.size - 1) % step != 0:
                x_draw = np.append(x_draw, x_data[-1])
                y_draw = np.append(y_draw, y_data[-1])
        else:
            x_draw, y_draw = x_data, y_data

        # --- 2. Vectorised coordinate transform (optimised for PySide6) ---
        x_min, x_max = limits.X_min, limits.X_max
        y_min = limits.Y2_min if use_y2 else limits.Y1_min
        y_max = limits.Y2_max if use_y2 else limits.Y1_max

        dx = (x_max - x_min) if x_max != x_min else 1.0
        dy = (y_max - y_min) if y_max != y_min else 1.0

        # rect.bottom() (not rect.top() + rect.height()) to match
        # map_data_to_widget exactly -- QRect.bottom() is top()+height()-1,
        # so the two formulas differ by 1px if not kept in sync, which
        # they previously were not (every polyline drawn 1px below every
        # gridline/tick/step-profile/cursor-line, all of which go through
        # map_data_to_widget).
        px = rect.left() + ((x_draw - x_min) / dx) * rect.width()
        py = rect.bottom() - ((y_draw - y_min) / dy) * rect.height()

        # --- 3. Fast drawing object (PySide6 compliant) ---
        polygon = QPolygonF([QPointF(x, y) for x, y in zip(px, py)])
        
        painter.drawPolyline(polygon)

    def _draw_step_profile(self, painter: QPainter, rect: QRect, limits: PlotLimits,
                          target_L: np.ndarray, target_P: np.ndarray,
                          color: QColor, width: float,
                          actual_dist: Optional[np.ndarray] = None, actual_time: Optional[np.ndarray] = None):
        """
        Draw a step (staircase) profile for the target power strategy within plot_rect.

        Each horizontal segment spans one strategy segment length; a vertical riser
        connects adjacent segments.
        """
        if target_L.size == 0 or target_P.size != target_L.size:
            return

        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine))
        
        # 1. Distance-based segment edges (always needed)
        segment_edges_dist = np.concatenate([[0], np.cumsum(target_L)])
        
        # 2. Determine X-axis coordinates
        if self.is_time_mode and actual_dist is not None and actual_time is not None:
            # Time mode: look up time from distance
            # Ensure actual_dist is sorted ascending before interpolation
            segment_edges = np.interp(segment_edges_dist, actual_dist, actual_time)
        else:
            # Distance mode: use distance directly
            segment_edges = segment_edges_dist

        # 3. Drawing loop
        for j in range(target_P.size):
            x_start = segment_edges[j]
            x_end = segment_edges[j+1]
            p_val = target_P[j]
            
            # Skip off-screen segments for speed
            if x_end < limits.X_min or x_start > limits.X_max:
                continue

            p1 = self.map_data_to_widget(x_start, p_val, rect, limits)
            p2 = self.map_data_to_widget(x_end, p_val, rect, limits)
            
            # Draw horizontal bar
            painter.drawLine(p1, p2)
            
            # Draw vertical step
            if j > 0:
                p_prev = target_P[j-1]
                p_prev_point = self.map_data_to_widget(x_start, p_prev, rect, limits)
                painter.drawLine(p_prev_point, p1)

    def _draw_horizontal_line(self, painter: QPainter, rect: QRect, limits: PlotLimits,
                               y_val: float, color: "QColor | Qt.GlobalColor", style: Qt.PenStyle = Qt.PenStyle.SolidLine,
                               width: Optional[float] = None):
        """Draw a horizontal reference line at the specified Y value."""
        if width is None:
            width = vp.LINE_WIDTH_CP_WPRIME
            
        painter.setPen(QPen(color, width, style))
        # Retrieve coordinates spanning the display range (X_min to X_max) and draw the line
        p_start = self.map_data_to_widget(limits.X_min, y_val, rect, limits)
        p_end = self.map_data_to_widget(limits.X_max, y_val, rect, limits)
        painter.drawLine(p_start, p_end)

    def _draw_plot_legend(self, painter: QPainter, rect: QRect, items: List[str], position: str):
        """
        Draw a legend box inside plot_rect at the specified position.

        Args:
            painter: Active QPainter to draw with.
            rect: The plot's own rect the legend is positioned inside.
            items: list of keys into LEGEND_DEFINITIONS.
            position: one of 'TopLeft', 'TopRight', 'BottomLeft', 'BottomRight'.
        """
        if not items:
            return

        LEGEND_MARKER_LEN = 20                  # line length (slightly shorter)
        LEGEND_HEIGHT = 14                      # row height, reduced for tighter spacing (18 -> 14)
        LEGEND_PADDING = 5                      # inner padding, reduced (8 -> 5)
        TEXT_GAP = 3                            # gap between line marker and text, reduced
        LEGEND_FONT_SIZE = vp.TICK_FONTSIZE - 2 # smaller than the tick font (12 -> 10)
        
        painter.setFont(QFont("Arial", LEGEND_FONT_SIZE))

        # Compute legend box size
        max_label_width = 0
        for key in items:
            if key in self.LEGEND_DEFINITIONS:
                label = self.LEGEND_DEFINITIONS[key]['label']
                max_label_width = max(max_label_width, painter.fontMetrics().horizontalAdvance(label))
        
        box_width = LEGEND_MARKER_LEN + TEXT_GAP + max_label_width + 2 * LEGEND_PADDING
        box_height = len(items) * LEGEND_HEIGHT + 2 * LEGEND_PADDING
        
        # Compute legend box position
        if position == 'TopLeft':
            x = rect.left() + LEGEND_PADDING
            y = rect.top() + LEGEND_PADDING
        elif position == 'TopRight':
            x = rect.right() - box_width - LEGEND_PADDING
            y = rect.top() + LEGEND_PADDING
        elif position == 'BottomLeft':
            x = rect.left() + LEGEND_PADDING
            y = rect.bottom() - box_height - LEGEND_PADDING
        elif position == 'BottomRight':
            x = rect.right() - box_width - LEGEND_PADDING
            y = rect.bottom() - box_height - LEGEND_PADDING
        else:
            return
            
        legend_rect = QRect(int(x), int(y), int(box_width), int(box_height))
        
        # Draw legend box background
        painter.setBrush(QBrush(QColor(255, 255, 255, 180)))  # semi-transparent white
        painter.setPen(QPen(Qt.GlobalColor.gray, 1))
        painter.drawRect(legend_rect)
        
        current_y = legend_rect.top() + LEGEND_PADDING
        
        for key in items:
            if key not in self.LEGEND_DEFINITIONS: continue
            
            def_ = self.LEGEND_DEFINITIONS[key]
            label = def_['label']
            style = def_['style']
            width = def_['width']
            color = def_['color']
            
            # Draw line marker
            line_start = QPointF(legend_rect.left() + LEGEND_PADDING, current_y + LEGEND_HEIGHT / 2)
            line_end = QPointF(line_start.x() + LEGEND_MARKER_LEN, line_start.y())
            
            painter.setPen(QPen(color, width, style))
            painter.drawLine(line_start, line_end)

            # Draw text
            painter.setPen(Qt.GlobalColor.black)
            text_x = line_end.x() + TEXT_GAP
            text_rect = QRect(int(text_x), int(current_y), int(max_label_width), int(LEGEND_HEIGHT))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            
            current_y += LEGEND_HEIGHT

    def _draw_cursor_line(self, painter):
        """Lightweight function that draws only the cursor line, ignoring the background."""
        if not hasattr(self, 'cursor_dist') or self.cursor_dist is None:
            return

        # Compute drawing area (same constants as paintEvent)
        widget_width = self.width()
        widget_height = self.height()
        full_plot_rect = QRect(
            self.MARGIN_LEFT, self.MARGIN_TOP,
            widget_width - self.MARGIN_LEFT - self.MARGIN_RIGHT,
            widget_height - self.MARGIN_TOP - self.MARGIN_BOTTOM
        )
        num_subplots = 5
        plot_height = (full_plot_rect.height() - self.PLOT_GAP * (num_subplots - 1)) / num_subplots

        cursor_color = QColor("#008000") 
        painter.setPen(QPen(cursor_color, 2, Qt.PenStyle.SolidLine))
        for i, name in enumerate(['Course', 'Power', 'Velocity', 'WPrime', 'Cumulative']):
            top = full_plot_rect.top() + i * (plot_height + self.PLOT_GAP)
            current_rect = QRect(full_plot_rect.left(), int(top), full_plot_rect.width(), int(plot_height))
            
            # Coordinate calculation
            pt = self.map_data_to_widget(self.cursor_dist, 0.0, current_rect, self.limits[name])
            painter.drawLine(int(pt.x()), current_rect.top(), int(pt.x()), current_rect.bottom())

    def _draw_segment_boundaries(self, painter: QPainter, rect: QRect, limits: PlotLimits, data_cache: tuple):
        """
        [3. Geometry layer]
        Inspect COLOR_HEX and render Designer control lines (dark green) and history lines (fog) differently.
        """
        if not data_cache:
            return

        painter.save()

        for data in data_cache:
            if data.get('TargetL_m') is None or data['TargetL_m'].size == 0:
                continue

            # --- 1. Role determination and pen selection ---
            current_hex = data.get('COLOR_HEX', '#808080').upper()
            
            if current_hex == DESIGNER_COLOR_HEX.upper():
                # [Primary: Designer control line] Sharp dark green, solid
                pen = QPen(QColor(DESIGNER_COLOR_HEX), 1, Qt.PenStyle.SolidLine)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                is_active = True
            elif current_hex == ACTIVE_COLOR_HEX.upper():
                pen = QPen(QColor(ACTIVE_COLOR_HEX), 1, Qt.PenStyle.SolidLine)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                is_active = True
            else:
                # [Background: density fog] Gray, dashed, semi-transparent
                fog_color = QColor(Qt.GlobalColor.gray)
                fog_color.setAlpha(100)
                pen = QPen(fog_color, 1, Qt.PenStyle.DashLine)
                pen.setDashPattern([5.0, 3.0])
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                is_active = False

            painter.setPen(pen)

            # --- 2. Coordinate calculation (Time/Distance hybrid) ---
            dist_edges = np.concatenate([[0], np.cumsum(data['TargetL_m'])])
            
            if self.is_time_mode and data.get('TIME') is not None and data['TIME'].size > 0:
                # Project distance-based boundaries onto the time axis using trajectory data
                x_edges = np.interp(dist_edges, data['DISTANCE'], data['TIME'])
            else:
                x_edges = dist_edges

            # --- 3. Draw execution ---
            for x_edge in x_edges:
                if limits.X_min <= x_edge <= limits.X_max:
                    point = self.map_data_to_widget(x_edge, 0, rect, limits)
                    
                    # Snap Designer lines to integer pixels to prevent blurring
                    draw_x = int(point.x()) if is_active else point.x()
                    painter.drawLine(QPointF(draw_x, rect.top()), QPointF(draw_x, rect.bottom()))

        painter.restore()

    def _draw_subplot_legend(self, painter: QPainter, rect: QRect, plot_name: str):
        """Decoration layer: draw the legend independently."""
        legend_items, position = self._get_legend_config(plot_name)
        if legend_items:
            self._draw_plot_legend(painter, rect, legend_items, position)

    # ------------------------------------------------
    # 6. UI/HUD Layer
    # ------------------------------------------------
    def _draw_distributed_huds(self, painter: QPainter, data: dict):
        """
        Overlay per-subplot HUD tooltips adjacent to the cursor position.

        Reads current_stats and draws a rounded-rectangle info box for each subplot
        showing the metric value at the cursor (grade, power, velocity, W' balance,
        and cumulative distance or time).
        """
        if not hasattr(self, 'current_stats') or self.current_stats is None:
            return

        # --- 1. Common calculations ---
        st = self.current_stats
        # Use the data argument passed in directly
        limits = self.limits['Course']
        plot_width = self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT
        target_x = getattr(self, 'current_cursor_x', self.current_dist)
        rel_x = (target_x - limits.X_min) / (limits.X_max - limits.X_min)
        cursor_x = self.MARGIN_LEFT + rel_x * plot_width

        full_h = self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM
        num_subplots = 5
        plot_h = (full_h - self.PLOT_GAP * (num_subplots - 1)) / num_subplots

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg_color = QColor(20, 20, 20, 160)
        text_color = QColor(0, 255, 0)
        font = QFont("Menlo", 11, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()

        # --- 2. Identify current segment ---
        target_L = data['TargetL_m']
        target_P = data['TargetP_W']
        segment_edges_dist = np.concatenate([[0], np.cumsum(target_L)])
        
        idx = np.searchsorted(segment_edges_dist, st['dist'], side='right') - 1
        idx = max(0, min(idx, target_P.size - 1))

        if self.is_time_mode:
            actual_dist = data['DISTANCE']
            actual_time = data['TIME']
            # Convert segment boundary distances to the time axis
            t_edges = np.interp([segment_edges_dist[idx], segment_edges_dist[idx+1]], actual_dist, actual_time)
            seg_len_str = f"{t_edges[1] - t_edges[0]:.1f}s"
        else:
            seg_len_str = f"{target_L[idx]:.0f}m"

        # --- 3. Build HUD content ---
        elapsed_min, elapsed_sec = divmod(int(round(max(0.0, st['time']))), 60)
        hud_contents = [
            f"Grade: {st['grade']:>5.1f}%\nAlt: {st['alt']:>5.1f}m",
            f"Act: {st['actual']:>4.0f}W\nTgt: {target_P[idx]:>4.0f}W ({seg_len_str})\nCP: {st['cp']:>4.0f}W",
            f"Velo: {st['vel']:>5.1f}km/h",
            f"W'Bal: {st['w_bal']:>5.0f}J\nW'Max: {st['w_max']:>5.0f}J",
            f"{'Dist' if self.is_time_mode else 'Time'}: " +
            (f"{st['dist']/1000.0:.2f}km" if self.is_time_mode else f"{elapsed_min}m{elapsed_sec}s")
        ]

        # --- 4. Draw loop ---
        for i, text in enumerate(hud_contents):
            top = self.MARGIN_TOP + i * (plot_h + self.PLOT_GAP)
            lines = text.split('\n')
            max_w = max([fm.horizontalAdvance(line) for line in lines])
            total_h = fm.height() * len(lines)
            hud_w, hud_h = max_w + 12, total_h + 6
            x_pos = cursor_x + 10
            if x_pos + hud_w > self.width():
                x_pos = cursor_x - hud_w - 10
            hud_rect = QRectF(x_pos, top + 5, hud_w, hud_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(hud_rect, 4, 4)
            painter.setPen(text_color)
            painter.drawText(hud_rect.adjusted(6, 2, -6, -2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, text)

        # --- 5. X-axis footer label ---
        x_info_str = f"{target_x:.1f} sec" if self.is_time_mode else f"{target_x/1000.0:.3f} km"
        x_info_w = fm.horizontalAdvance(x_info_str) + 12
        x_info_h = fm.height() + 4
        x_info_x = cursor_x - x_info_w / 2
        x_info_rect = QRectF(x_info_x, self.height() - self.MARGIN_BOTTOM + 2, x_info_w, x_info_h)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(x_info_rect, 3, 3)
        painter.setPen(text_color)
        painter.drawText(x_info_rect, Qt.AlignmentFlag.AlignCenter, x_info_str)
