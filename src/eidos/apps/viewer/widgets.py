"""
eidos.apps.viewer.widgets -- CourseMapWidget, the 2D top-down course map.

Renders the course polyline and rider position marker (plus true-wind
vector and CdA-vs-heading polar plot overlays) via raw QPainter -- no
matplotlib involved, unlike eidos.apps.analyzer.canvas's AnalysisCanvas.
"""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class CourseMapWidget(QWidget):
    """
    Fixed-size widget that renders a 2D top-down course map with a rider position marker.

    Scales GPS (lat/lon) coordinates to widget pixels and paints the course
    polyline and current-position dot on each update.
    """
    def __init__(self, parent=None):
        """Initialize the course map with empty state and a fixed 300x300 size policy."""
        super().__init__(parent)
        self.points = []
        self.raw_dist = None
        self.current_idx = 0
        # Fixed size policy prevents the widget from being stretched by the layout
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_true_wind(self, wind_speed, wind_direction):
        """Store the true wind speed and direction and request a repaint."""
        self.true_v_wind = wind_speed
        self.true_d_wind = wind_direction
        self.update()  # trigger repaint

    def sizeHint(self):
        """Return the preferred widget size (300x300) to the layout manager."""
        return QSize(300, 300)

    def minimumSizeHint(self):
        """Return the minimum widget size (300x300)."""
        return QSize(300, 300)

    def set_course_latlon(self, lats: np.ndarray, lons: np.ndarray, dists: np.ndarray):
        """Project lat/lon to flat-plane coordinates scaled to fit the 300x300 widget."""
        if lats is None or lats.size < 2: return
        self.raw_dist = dists

        # 1. Map projection (equirectangular)
        lat_mid = np.radians(np.mean(lats))
        m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
        m_per_lon = 111412.84 * np.cos(lat_mid)
        
        x_m = (lons - lons[0]) * m_per_lon
        y_m = (lats - lats[0]) * m_per_lat

        # 2. Scale to fit the 300x300 draw area without distortion
        margin = 20
        # Use sizeHint dimensions rather than the actual widget size
        draw_w = 300 - 2 * margin
        draw_h = 300 - 2 * margin
        
        x_min, x_max = x_m.min(), x_m.max()
        y_min, y_max = y_m.min(), y_m.max()
        
        range_x = max(1.0, x_max - x_min)
        range_y = max(1.0, y_max - y_min)
        
        # Common scale that preserves the aspect ratio
        scale = min(draw_w / range_x, draw_h / range_y)
        
        # Offsets to centre the scaled course within the draw area
        off_x = (draw_w - range_x * scale) / 2
        off_y = (draw_h - range_y * scale) / 2

        # 3. Build coordinate list (includes Y-axis flip)
        self.points = [
            QPointF(margin + off_x + (xi - x_min) * scale, 
                    300 - (margin + off_y + (yi - y_min) * scale))
            for xi, yi in zip(x_m, y_m)
        ]
        self.update()

    def update_cursor(self, dist_m):
        """Update the highlighted course position to the first point at or past dist_m (bisect_left, not the nearest of the two neighbors)."""
        if self.raw_dist is None or len(self.points) == 0: return
        import bisect
        idx = bisect.bisect_left(self.raw_dist, dist_m)
        self.current_idx = max(0, min(idx, len(self.points) - 1))
        self.update()

    def set_wind_data(self, wind_data, cda_ratios):
        """Store wind and CdA ratio data received from the viewer for painting."""
        self.wind_data = wind_data
        self.cda_ratios = cda_ratios
        
        self.sim_dist = wind_data['DISTANCE']
        self.yaw_raw = wind_data['WIND_YAW']
        self.v_app = wind_data['WIND_V_APP']
        
        self.heading_dist = wind_data['HEADING_DIST']
        self.heading_vals = wind_data['HEADING']

        self.update()

    def paintEvent(self, event):
        """Paint the course polyline, rider position, wind vectors, and CdA polar plot."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Draw background and border
        p.fillRect(self.rect(), QColor("#E0E0E0"))
        p.setPen(QPen(QColor("#AAAAAA"), 1))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # 2. Nothing to draw if no course points
        if not self.points:
            return

        # 3. Draw course polyline
        p.setPen(QPen(QColor("#000080"), 2, Qt.PenStyle.SolidLine))
        p.drawPolyline(self.points)

        # 4. Draw rider position and apparent wind
        if 0 <= self.current_idx < len(self.points):
            self._draw_rider_and_apparent_wind(p)

        # 5. Draw true wind vector
        if hasattr(self, 'true_v_wind') and self.true_v_wind > 0:
            self._draw_true_wind_vector(p)

    def _draw_true_wind_vector(self, p: QPainter):
        """Draw the true wind vector as a green arrow in the top-right corner."""
        p.save()
        
        # Anchor to the top-right corner
        margin = 40
        p.translate(self.width() - margin, margin)
        
        # Rotate so that 270° (west) points right on screen
        p.rotate(self.true_d_wind - 270)
        
        # Arrow length proportional to wind speed
        length = self.true_v_wind * 3.0
        
        # Green pen
        p.setPen(QPen(QColor(0, 180, 0), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        
        # Draw shaft and arrowhead
        p.drawLine(0, 0, length, 0)
        arrow_size = 8
        p.drawLine(length, 0, length - arrow_size, -arrow_size // 2)
        p.drawLine(length, 0, length - arrow_size,  arrow_size // 2)
        
        # Wind speed label
        p.setPen(QColor("#666666"))
        p.drawText(QPointF(-15, 25), f"{self.true_v_wind:.1f}m/s")
        
        p.restore()

    def _draw_rider_and_apparent_wind(self, p):
        """Draw true wind (green) and apparent wind (blue/red) vectors centred on the rider position."""
        if self.sim_dist is None or len(self.sim_dist) == 0:
            return
        if self.v_app is None or len(self.v_app) == 0:
            return

        pos = self.points[self.current_idx]
        d = self.raw_dist[self.current_idx]

        # 1. Interpolate current values
        v_val = np.interp(d, self.sim_dist, self.v_app)
        yw_val = np.interp(d, self.sim_dist, self.yaw_raw)
        hd_val = np.interp(d, self.heading_dist, self.heading_vals)

        # 2. Choose colour based on current CdA ratio -- no CdA data (e.g.
        # a strategy whose simulator has no aero-drag model at all, or an
        # older-format strategy) means no ratio to compute a colour from;
        # fall back to a neutral factor rather than crashing. Skipping the
        # polar plot below is then the visual equivalent of that factor
        # never varying (a "no yaw dependence" circle of radius 1), so no
        # separate message is warranted.
        has_cda_data = self.cda_ratios is not None and len(self.cda_ratios) > 0
        if has_cda_data:
            abs_yw = min(180, max(0, abs(yw_val)))
            current_factor = self.cda_ratios[int(abs_yw)]
        else:
            current_factor = 1.0

        # 3. Draw CdA polar plot
        if has_cda_data:
            self._draw_cda_polar_plot(p, pos, hd_val, self.cda_ratios)

        # 4. True wind vector (green): points in the direction the wind comes from
        if hasattr(self, 'true_v_wind') and self.true_v_wind > 0:
            # Use compass bearing directly (0=N, 90=E, 270=W)
            # At 270°, sin=-1 cos=0, so the vector points left (west)
            t_angle_rad = np.radians(self.true_d_wind)
            t_len = min(40, self.true_v_wind * 3.0)
            
            t_end_pt = QPointF(
                pos.x() + t_len * np.sin(t_angle_rad),
                pos.y() - t_len * np.cos(t_angle_rad)
            )
            
            p.setPen(QPen(QColor("#008000"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(pos, t_end_pt)

        # 5. Apparent wind vector (blue/red): wind direction as felt by the rider
        # heading + yaw angle = absolute apparent wind bearing
        bar_color = QColor("blue") if current_factor < 1.0 else QColor("red")
        v_len = min(40, v_val * 3.0) 
        angle_rad = np.radians(hd_val + yw_val)
        
        end_pt = QPointF(
            pos.x() + v_len * np.sin(angle_rad), 
            pos.y() - v_len * np.cos(angle_rad)
        )
        
        p.setPen(QPen(bar_color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(pos, end_pt)

        # 6. Draw rider dot
        p.setBrush(QColor("#008000"))
        p.setPen(QPen(Qt.GlobalColor.white, 2))
        p.drawEllipse(pos, 5, 5)

    def _draw_cda_polar_plot(self, p, center_pos, hd_deg, cda_ratios):
        """Draw a CdA polar plot around center_pos, colour-coded cyan/magenta relative to hd_deg."""
        if cda_ratios is None or len(cda_ratios) == 0:
            return

        display_base_r = 25.0
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        # Draw full circle in 1° increments
        for a in range(0, 360, 1):
            # 1. Compute angle relative to heading
            # 2. Mirror to symmetric 0-180° index
            # a is 0-360; treat directly as relative deflection angle
            rel_angle = a if a <= 180 else 360 - a
            
            ratio = cda_ratios[int(rel_angle)]
            r = display_base_r * ratio
            
            # Cyan if CdA ratio < 1.0 (beneficial), magenta otherwise
            if ratio < 1.0:
                p.setBrush(QColor(0, 255, 255, 50))
            else:
                p.setBrush(QColor(255, 0, 255, 30))

            # 3. Convert to absolute Qt drawing angle
            # Qt: north (0°) maps to 90° in arc coordinates
            # Pie start = heading + relative angle
            qt_start_angle = int((90 - (hd_deg + a + 5)) * 16)
            p.drawPie(QRectF(center_pos.x() - r, center_pos.y() - r, 
                             r * 2, r * 2), qt_start_angle, int(5 * 16))

        # 4. Draw reference circle (ratio = 1.0) as dashed line
        p.setPen(QPen(QColor(50, 50, 50, 180), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center_pos, display_base_r, display_base_r)

        p.restore()

