"""
eidos.lib.power_profile_canvas.layout_mixin -- Coordinate mapping and widget geometry.

_PowerProfileLayoutMixin is one of three mixins combined into
PowerProfileCanvas (see canvas.py) -- see data_mixin.py's docstring for
the general caveat about mixins not being standalone classes.
"""

import math
from typing import TYPE_CHECKING, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRect, QSize

from eidos.lib.power_profile_canvas.plot_limits import PlotLimits, nice_ticks

# See draw_mixin.py's comment above its own _DrawMixinBase (and
# _canvas_selftype.py's docstring): same TYPE_CHECKING-only trick, pointed
# at the cycle-free stand-in class rather than the real PowerProfileCanvas,
# so mypy knows self also has the other mixins' attributes
# (self.is_time_mode, ...) without any real inheritance or circular import
# at runtime.
if TYPE_CHECKING:
    from eidos.lib.power_profile_canvas._canvas_selftype import _CanvasSelfType
    _LayoutMixinBase = _CanvasSelfType
else:
    _LayoutMixinBase = object


class _PowerProfileLayoutMixin(_LayoutMixinBase):
    """Coordinate mapping and widget-geometry calculation.

    Not a usable class on its own -- see data_mixin.py's docstring.
    """

    def sizeHint(self):
        """Return the preferred widget size (750 x 800 px)."""
        return QSize(750, 800)
    
    def resizeEvent(self, event):
        """Invalidate the cache when the window is resized."""
        self._cache_dirty = True
        # paintEvent is called automatically after resizeEvent,
        # so an explicit update() call here is not needed.
        super().resizeEvent(event)

    # ------------------------------------------------
    # 2. Data Logic
    # ------------------------------------------------
    def _get_x_ticks(self, limits):
        """Generate tick values at a fixed interval (step)."""
        duration = limits.X_max - limits.X_min
        if duration <= 0: return []
        
        # Choose a round step value
        if self.is_time_mode:
            # 60 s (1 min), 300 s (5 min), 600 s (10 min) increments
            if duration <= 600:    step = 60.0
            elif duration <= 1800: step = 300.0
            else:                  step = 600.0
        else:
            # 1 km (1000 m), 2 km, 5 km increments
            if duration <= 10000:  step = 1000.0
            elif duration <= 20000: step = 2000.0
            else:                   step = 5000.0
            
        start = math.ceil(limits.X_min / step) * step
        return nice_ticks(start, limits.X_max, step)

    def _get_y_ticks(self, plot_name: str, limits: PlotLimits) -> Tuple[np.ndarray, np.ndarray]:
        # Y1 axis (left) interval settings
        """Return Y1 and Y2 tick arrays for the given plot at fixed metric-appropriate intervals."""
        if plot_name == 'Course': 
            interval = 10.0
        elif plot_name == 'Power': 
            interval = 100.0
        elif plot_name == 'Velocity': 
            interval = 10.0
        elif plot_name == 'WPrime': 
            interval = 5000.0
        elif plot_name == 'Cumulative':
            # Use fixed step instead of 4-division
            if self.is_time_mode:
                # Y-axis is distance (km): 2 km or 5 km increments
                interval = 2.0 if limits.Y1_max <= 15 else 5.0
            else:
                # Y-axis is time (min): 2 min or 5 min increments
                interval = 2.0 if limits.Y1_max <= 15 else 5.0
        else: 
            # Fallback for undefined plots: divide into 4 equal parts
            interval = (limits.Y1_max - limits.Y1_min) / 4 if limits.Y1_max != limits.Y1_min else 1.0

        y1_min = limits.Y1_min
        y1_max = limits.Y1_max
        y1_ticks = nice_ticks(y1_min, y1_max, interval)

        # Y2 axis (right)
        y2_ticks = np.array([])
        if plot_name == 'Course':
            # Altitude: 50 m increments
            interval_y2 = 50.0
            y2_min = limits.Y2_min
            y2_max = limits.Y2_max
            y2_ticks = nice_ticks(y2_min, y2_max, interval_y2)
            if y2_ticks.size < 3:
                y2_ticks = np.linspace(y2_min, y2_max, 3)

        return y1_ticks, y2_ticks

    def _update_layout_geometry(self):
        """Finalise the drawing rectangle (QRect) for each subplot based on the current widget size."""
        full_plot_rect = QRect(
            self.MARGIN_LEFT, self.MARGIN_TOP, 
            self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT, 
            self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM
        )
        
        num_subplots = 5
        plot_height = (full_plot_rect.height() - self.PLOT_GAP * (num_subplots - 1)) / num_subplots
        plot_names = ['Course', 'Power', 'Velocity', 'WPrime', 'Cumulative']

        self.subplot_rects = {}
        for i, name in enumerate(plot_names):
            top = full_plot_rect.top() + i * (plot_height + self.PLOT_GAP)
            self.subplot_rects[name] = QRect(
                full_plot_rect.left(), int(top), 
                full_plot_rect.width(), int(plot_height)
            )
        
        return full_plot_rect, plot_height, plot_names

    def map_data_to_widget(self, x_data: float, y_data: float, plot_rect: QRect, limits: PlotLimits, use_y2: bool = False) -> QPointF:
        """
        Map a (x_data, y_data) point to widget pixel coordinates.

        Args:
            x_data: X value in data units (distance m or time s).
            y_data: Y value in data units.
            plot_rect: bounding QRect of the subplot.
            limits: axis limits for this subplot.
            use_y2: if True, use Y2 axis limits (altitude) instead of Y1.

        Returns:
            QPointF in widget coordinates.
        """
        x_range = limits.X_max - limits.X_min
        
        if use_y2:
            y_min, y_max = limits.Y2_min, limits.Y2_max
        else:
            y_min, y_max = limits.Y1_min, limits.Y1_max
            
        y_range = y_max - y_min

        if x_range <= 0 or y_range <= 0:
            return QPointF(plot_rect.left(), plot_rect.bottom())

        x_normalized = (x_data - limits.X_min) / x_range
        x_widget = plot_rect.left() + x_normalized * plot_rect.width()

        y_normalized = (y_data - y_min) / y_range
        y_widget = plot_rect.bottom() - y_normalized * plot_rect.height()
        
        return QPointF(x_widget, y_widget)

    def map_widget_to_data(self, x_px: float, y_px: float, rect: QRect, limits: PlotLimits):
        """
        Inverse-map widget coordinates to physical quantities (Distance or Time, and Power/etc.).
        X-axis physical quantity is switched automatically based on is_time_mode.
        """
        if rect.width() <= 0 or rect.height() <= 0:
            return 0.0, 0.0

        # --- X-axis inverse mapping ---
        rel_x = (x_px - rect.left()) / rect.width()
        # limits.X_min/max already hold mode-appropriate values (distance or time) at calculation time
        data_x = limits.X_min + rel_x * (limits.X_max - limits.X_min)

        # --- Y-axis inverse mapping ---
        rel_y = (rect.bottom() - y_px) / rect.height()
        data_y = limits.Y1_min + rel_y * (limits.Y1_max - limits.Y1_min)

        return data_x, data_y

    # ------------------------------------------------
    # 4. Rendering Pipeline
    # ------------------------------------------------
