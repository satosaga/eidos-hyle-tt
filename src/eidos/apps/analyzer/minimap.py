"""
eidos.apps.analyzer.minimap -- Course minimap widget.

CourseMinimapWidget is the small top-down course map with a cursor dot,
driven by TTAnalyzerWindow's distance slider (see eidos.apps.analyzer.window).
Shares no code with AnalysisCanvas beyond the SCENARIO_COLORS palette
(imported below); conceptually standalone.
"""

import bisect

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from eidos.apps.analyzer.canvas import SCENARIO_COLORS

# ---------------------------------------------------------------------------
# Course minimap
# ---------------------------------------------------------------------------

class CourseMinimapWidget(QWidget):
    """
    Small top-down course map with cursor position dot(s), driven by the
    distance slider (see TTAnalyzerWindow._on_slider_moved).

    Draws two overlaid polylines when an Activity is loaded: the GPX
    course (green, matching "Strategy" elsewhere in this app) and the
    Activity's own recorded GPS track (white, matching "Activity"). Each
    gets its own cursor dot, positioned at the SAME pct (see
    core.activity_correspondence) but evaluated against ITS OWN
    distance total — the course dot via course_s_p's own total, the
    activity dot via activity_raw.distance_m's own total. Line-choice
    through corners and GPS-spline smoothing bias mean a real position at
    a given pct can still differ between the two by a few metres (see
    core.activity_parser.find_course_matches's own docstring for why);
    the two dots visibly separating or reconverging as the slider moves
    makes that reconciliation problem directly visible on the course
    shape. Rebuild scenarios get no dot of their own here — course
    geometry is always GPX-derived, so a Rebuild's position always
    coincides with the green course dot by construction.

    Deliberately NOT eidos.apps.viewer's CourseMapWidget, despite the
    near-identical projection math below: importing it would pull
    eidos.apps.analyzer into eidos.apps.viewer's much heavier import
    graph just to reuse one class that also carries wind-vector/CdA-
    polar-plot painting this feature has no use for. This is the
    minimal polyline + dot version, themed for this file's dark palette.
    """

    SIZE = 280
    MARGIN = 20

    COURSE_COLOR = SCENARIO_COLORS[0]   # green — matches "Strategy" everywhere else
    ACTIVITY_COLOR = "#FFFFFF"          # white — matches "Activity" everywhere else
    RUNG_COLOR = "#FFCC00"              # yellow — matches AnalysisCanvas.CURSOR_COLOR

    # Half-width [m] of the local window shown in follow mode (visible
    # span is 2x this) — small enough that a typical few-to-15 m
    # ride-vs-course gap reads as a meaningful fraction of the view,
    # not swallowed by the dot markers as it is in the whole-course
    # overview.
    FOLLOW_HALF_WIDTH_M = 20.0
    # Index-window slack multiplier: how far past FOLLOW_HALF_WIDTH_M
    # (same units, along each track's own distance axis) to slice points
    # for, so a polyline segment can still enter/exit the visible square
    # from a diagonal without a visible gap at the edge.
    FOLLOW_SLICE_FACTOR = 1.5

    DOT_RADIUS = 5
    HIGHLIGHT_RUNG_WIDTH = 4  # px -- the current cursor's own rung, vs 1px for the rest

    def __init__(self, parent=None):
        super().__init__(parent)
        # Highlighted Rebuild's colour, set via set_highlight_color() from
        # TTAnalyzerWindow._refresh_canvas. None when no Rebuild is
        # highlighted (falls back to COURSE_COLOR / Strategy green) — see
        # _current_course_color and set_highlight_color's docstring for
        # which drawn elements this does and doesn't apply to.
        self._highlight_color: str | None = None

        # Raw lat/lon/dist for the GPX course — kept around (not just the
        # projected points) so _reproject() can rebuild both polylines
        # sharing one projection whenever the Activity track changes.
        self._course_lats: np.ndarray | None = None
        self._course_lons: np.ndarray | None = None
        self.raw_dist: np.ndarray | None = None
        # raw_dist as a fraction of its own last element -- see
        # core.activity_correspondence for why cursor/rung matching
        # uses pct (each track's own fraction of ITS OWN total distance)
        # rather than a shared raw distance value: the course's and
        # Activity's own distance totals routinely disagree by a percent
        # or more. Recomputed in _reproject().
        self.course_pct: np.ndarray | None = None

        self._activity_lats: np.ndarray | None = None
        self._activity_lons: np.ndarray | None = None
        self.activity_dist: np.ndarray | None = None
        self.activity_pct_arr: np.ndarray | None = None

        # Dense counterparts (see set_activity_track's dense_lats/lons/
        # dists docstring) -- drawn as the Activity polyline INSTEAD of
        # the sparse arrays above, which stay reserved for cursor/rung
        # matching (activity_current_idx indexes self.activity_points,
        # not this). None if no dense track was given, falling back to
        # the sparse polyline.
        self._activity_dense_lats: np.ndarray | None = None
        self._activity_dense_lons: np.ndarray | None = None
        self.activity_dense_dist: np.ndarray | None = None
        self.activity_dense_pct: np.ndarray | None = None

        self.points: list[QPointF] = []            # projected course polyline (overview)
        self.activity_points: list[QPointF] = []    # projected activity polyline (overview) -- cursor-indexed, sparse
        self.activity_dense_points: list[QPointF] = []  # drawn INSTEAD of activity_points when available (see above)
        self.current_idx = 0
        self.activity_current_idx = -1               # -1: no Activity dot to draw

        # Meter-space (unscaled) coordinates, shared origin — kept
        # alongside the overview's already-widget-scaled self.points so
        # follow mode (see set_follow_mode) can re-project a small local
        # window around the cursor on every move without recomputing the
        # equirectangular lat/lon math each time.
        self._course_x_m: np.ndarray | None = None
        self._course_y_m: np.ndarray | None = None
        self._activity_x_m: np.ndarray | None = None
        self._activity_y_m: np.ndarray | None = None
        self._activity_dense_x_m: np.ndarray | None = None
        self._activity_dense_y_m: np.ndarray | None = None

        # Follow mode: recomputed by update_cursor() on every move (see
        # its docstring) rather than in paintEvent, matching this app's
        # general "cache in the update call, cheap read in paint" split
        # (c.f. AnalysisCanvas.set_cursor_pct).
        self.follow_mode: bool = False
        self._last_pct: float = 0.0
        self._follow_course_points: list[QPointF] = []
        self._follow_activity_points: list[QPointF] = []
        self._follow_activity_dot: QPointF | None = None
        self._follow_matched_course_dot: QPointF | None = None
        self._follow_gap_m: float | None = None
        self._follow_rungs: list[tuple[QPointF, QPointF]] = []

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(self.SIZE, self.SIZE)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.SIZE, self.SIZE)

    def set_course(self, lats: np.ndarray, lons: np.ndarray, dists: np.ndarray):
        """
        Set the GPX course track and reproject.

        Static per-strategy data — called once (see
        TTAnalyzerWindow._apply_strategy), never touched again by
        Activity/Rebuild changes. Any previously-set Activity track is
        cleared, since it would otherwise be reprojected against a
        course it no longer belongs to.
        """
        if lats is None or len(lats) < 2:
            return
        self._course_lats = np.asarray(lats)
        self._course_lons = np.asarray(lons)
        self.raw_dist = np.asarray(dists)
        self._activity_lats = None
        self._activity_lons = None
        self.activity_dist = None
        self.activity_current_idx = -1
        self._reproject()

    def set_activity_track(
        self,
        lats: "np.ndarray | None", lons: "np.ndarray | None", dists: "np.ndarray | None",
        dense_lats: "np.ndarray | None" = None, dense_lons: "np.ndarray | None" = None,
        dense_dists: "np.ndarray | None" = None,
    ):
        """
        Overlay (or clear) the Activity's own recorded GPS track.

        Args:
            lats/lons: ActivityRecord.lat_deg/lon_deg (native FIT sample
                spacing, NOT the distance-rescaled axis used for
                Rebuild's PowerBlocks). Pass None (or too short an array)
                to clear the overlay. Reserved for cursor dot/rung
                matching (activity_current_idx indexes the polyline built
                from these) — never for the drawn line itself when a
                dense track is available (see dense_lats below).
            dists: ActivityRecord.distance_m — same basis update_cursor()
                uses to place the Activity dot.
            dense_lats/dense_lons: ActivityRecord.dense_lat_deg/
                dense_lon_deg — the SAME GPS-spline curve as lats/lons,
                sampled far more finely. Drawn as the Activity polyline
                INSTEAD of lats/lons when given, so the line follows the
                fitted curve's actual shape rather than straight chords
                between sparse samples. Falls back to lats/lons for the
                line when omitted or too short.
            dense_dists: ActivityRecord.dense_distance_m — dense_lats/
                dense_lons' own distance axis, used only to window the
                dense polyline in follow mode (see
                _recompute_follow_view); never for cursor placement.

        Reprojects all tracks together (see _reproject) so the drawn gap
        reflects the real spatial offset rather than two independently-
        normalized shapes that merely look similar. NaN GPS samples
        (sensor dropout) are dropped rather than plotted, since fed as-is
        they would break the projection's min/max or draw a spurious
        line segment leaping through the widget.
        """
        if lats is None or len(lats) < 2:
            self._activity_lats = None
            self._activity_lons = None
            self.activity_dist = None
            self._activity_dense_lats = None
            self._activity_dense_lons = None
            self.activity_dense_dist = None
            self.activity_current_idx = -1
            self._reproject()
            return

        lats = np.asarray(lats)
        lons = np.asarray(lons)
        dists = np.asarray(dists)
        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.sum() < 2:
            self._activity_lats = None
            self._activity_lons = None
            self.activity_dist = None
            self._activity_dense_lats = None
            self._activity_dense_lons = None
            self.activity_dense_dist = None
            self.activity_current_idx = -1
            self._reproject()
            return

        self._activity_lats = lats[valid]
        self._activity_lons = lons[valid]
        self.activity_dist = dists[valid]

        if dense_lats is not None and len(dense_lats) >= 2:
            dense_lats = np.asarray(dense_lats)
            dense_lons = np.asarray(dense_lons)
            dense_dists = np.asarray(dense_dists)
            dense_valid = ~(np.isnan(dense_lats) | np.isnan(dense_lons))
            if dense_valid.sum() >= 2:
                self._activity_dense_lats = dense_lats[dense_valid]
                self._activity_dense_lons = dense_lons[dense_valid]
                self.activity_dense_dist = dense_dists[dense_valid]
            else:
                self._activity_dense_lats = None
                self._activity_dense_lons = None
                self.activity_dense_dist = None
        else:
            self._activity_dense_lats = None
            self._activity_dense_lons = None
            self.activity_dense_dist = None

        self._reproject()

    def _reproject(self):
        """
        (Re)project course + (if present) activity lat/lon into shared
        widget-space coordinates.

        One equirectangular projection (origin + scale) is computed from
        the COMBINED bounding box of both tracks, then applied to each
        separately. Projecting each track independently (its own min/max,
        its own scale) would center-and-fit each shape on its own,
        visually hiding exactly the real-world spatial divergence
        (corner-cutting, odometer drift) this overlay exists to show —
        two similarly-shaped but differently-scaled lines would end up
        drawn on top of each other regardless of their actual offset.
        """
        if self._course_lats is None:
            return

        lat_mid_parts = [self._course_lats]
        has_activity = self._activity_lats is not None
        if has_activity:
            lat_mid_parts.append(self._activity_lats)

        all_lats = np.concatenate(lat_mid_parts)

        # Equirectangular projection, same formula as
        # eidos.apps.viewer's CourseMapWidget.set_course_latlon.
        lat_mid = np.radians(np.mean(all_lats))
        m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
        m_per_lon = 111412.84 * np.cos(lat_mid)

        # Origin pinned to the course's own first point (not the combined
        # bbox's corner) purely so re-Browse-ing a new FIT against the
        # same course doesn't shift the course polyline's own pixel
        # position — only origin, not scale, is course-anchored; scale
        # below still comes from the combined bbox.
        origin_lat, origin_lon = self._course_lats[0], self._course_lons[0]

        def project(lats, lons):
            x_m = (lons - origin_lon) * m_per_lon
            y_m = (lats - origin_lat) * m_per_lat
            return x_m, y_m

        x_course, y_course = project(self._course_lats, self._course_lons)
        has_dense = has_activity and self._activity_dense_lats is not None
        x_all_parts, y_all_parts = [x_course], [y_course]
        if has_activity:
            x_act, y_act = project(self._activity_lats, self._activity_lons)
            x_all_parts.append(x_act)
            y_all_parts.append(y_act)
        else:
            x_act = y_act = None
        if has_dense:
            x_dense, y_dense = project(self._activity_dense_lats, self._activity_dense_lons)
            x_all_parts.append(x_dense)
            y_all_parts.append(y_dense)
        else:
            x_dense = y_dense = None
        x_all = np.concatenate(x_all_parts)
        y_all = np.concatenate(y_all_parts)

        draw_w = self.SIZE - 2 * self.MARGIN
        draw_h = self.SIZE - 2 * self.MARGIN

        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()
        range_x = max(1.0, x_max - x_min)
        range_y = max(1.0, y_max - y_min)

        scale = min(draw_w / range_x, draw_h / range_y)
        off_x = (draw_w - range_x * scale) / 2
        off_y = (draw_h - range_y * scale) / 2

        def to_points(x_m, y_m):
            return [
                QPointF(self.MARGIN + off_x + (xi - x_min) * scale,
                        self.SIZE - (self.MARGIN + off_y + (yi - y_min) * scale))
                for xi, yi in zip(x_m, y_m)
            ]

        self.points = to_points(x_course, y_course)
        self.activity_points = to_points(x_act, y_act) if has_activity else []
        self.activity_dense_points = to_points(x_dense, y_dense) if has_dense else []

        # Meter-space coordinates, kept for follow mode (see
        # _recompute_follow_view) -- same x_course/y_course/x_act/y_act/
        # x_dense/y_dense already computed above, just not thrown away.
        self._course_x_m, self._course_y_m = x_course, y_course
        self._activity_x_m = x_act if has_activity else None
        self._activity_y_m = y_act if has_activity else None
        self._activity_dense_x_m = x_dense if has_dense else None
        self._activity_dense_y_m = y_dense if has_dense else None

        self.course_pct = self.raw_dist / self.raw_dist[-1]
        self.activity_pct_arr = self.activity_dist / self.activity_dist[-1] if has_activity else None
        self.activity_dense_pct = (
            self.activity_dense_dist / self.activity_dense_dist[-1] if has_dense else None
        )

        self.update_cursor(self._last_pct)  # re-run with the new geometry
        self.update()

    def set_highlight_color(self, color: "str | None"):
        """
        Set (or clear) the highlighted-Rebuild colour. Called from
        TTAnalyzerWindow._refresh_canvas alongside AnalysisCanvas's own
        selected_scenario handling, so this widget and the five plot
        panels highlight the same Rebuild at the same time.

        Affects the course POLYLINE (both views) and the follow-mode
        "matched course dot" (see _recompute_follow_view) — these
        represent "the course, as currently being compared." Deliberately
        does NOT affect either ringed cursor dot: those are a *slider*
        position indicator first (green because the distance slider's
        own handle is green), not a representation of "the course," so
        they stay Strategy green regardless of what's highlighted.
        """
        self._highlight_color = color
        self.update()

    def _current_course_color(self) -> str:
        """COURSE_COLOR, or the highlighted Rebuild's colour if set — see
        set_highlight_color's docstring for exactly which drawn elements
        use this vs. staying fixed at COURSE_COLOR."""
        return self._highlight_color or self.COURSE_COLOR

    def set_follow_mode(self, enabled: bool):
        """
        Toggle between the whole-course overview and a zoomed, cursor-
        centred local view (see _recompute_follow_view).

        The overview's fixed SIZE x SIZE frame maps the *entire* course
        into ~240 drawable px, so a real Activity/course gap of a few to
        ~15 m (see FOLLOW_HALF_WIDTH_M) works out to only 1-4 px —
        smaller than the DOT_RADIUS-sized cursor markers, so the course
        dot (drawn on top) fully occludes the activity dot and the gap
        this widget exists to show ends up invisible, even though it's
        computed correctly. Follow mode re-projects a small real-world
        window around the cursor instead, so the same gap reads as a
        real fraction of the visible width.
        """
        self.follow_mode = enabled
        self._recompute_follow_view(self._last_pct)
        self.update()

    def update_cursor(self, pct: float):
        """
        Move the course cursor dot (always, at pct of course_s_p's own
        total) and, if an Activity track is loaded, the activity cursor
        dot too (at that SAME pct of activity_raw.distance_m's own
        total — see core.activity_correspondence for why pct, not a
        shared raw distance value, is the correct correspondence). The
        two dots can therefore sit at visibly different points along
        their respective lines for the same pct — see the class
        docstring for why that's deliberate, not a bug.

        Also recomputes the follow-mode local view (cheap: a couple of
        bisects plus slicing a few dozen points — see
        _recompute_follow_view) regardless of whether follow mode is
        currently on, so toggling it on mid-drag shows the right window
        immediately rather than one stale slider-tick behind.
        """
        self._last_pct = pct

        if self.course_pct is not None and self.points:
            idx = bisect.bisect_left(self.course_pct, pct)
            self.current_idx = max(0, min(idx, len(self.points) - 1))

        if self.activity_pct_arr is not None and self.activity_points:
            idx_a = bisect.bisect_left(self.activity_pct_arr, pct)
            self.activity_current_idx = max(0, min(idx_a, len(self.activity_points) - 1))
        else:
            self.activity_current_idx = -1

        self._recompute_follow_view(pct)
        self.update()

    def _recompute_follow_view(self, pct: float):
        """
        Re-project a FOLLOW_HALF_WIDTH_M-radius window around the
        course's current cursor point into local screen coordinates. The
        course point at self.current_idx is pinned to the widget's exact
        centre by construction. The window radius itself is a spatial
        magnification (a fixed real-metre zoom), unrelated to pct -- only
        WHICH point sits at that centre, and which Activity samples count
        as "nearby" on its own distance axis, are pct-driven.

        Matching convention: the rungs and Δ below connect each Activity
        sample to the course geometry AT THAT SAME PCT (see
        core.activity_correspondence) -- the same convention
        update_plots' Δt panel uses (Rebuild's PowerBlocks, and therefore
        x_traj, are rescaled to strategy.course_distance_m, i.e. the SAME
        correspondence). Do NOT switch this to spatial nearest-point
        matching: it gives a smaller, "more correct" Δ (the rider's GPS
        track barely deviates from the course line), but decouples the
        map from what Δt/Power/Velocity/Altitude actually compare, making
        it useless as an explanation of their numbers. A systematic
        Activity-odometer/GPX-geometry disagreement (e.g. wheel-sensor
        calibration, not a routing difference) SHOULD show up here as a
        growing Δ -- it's the same pct correspondence the rest of the
        app's shared x-axis uses, and this map is what makes it legible.
        Course/activity polylines are sliced by INDEX along their own
        distance axis (not a spatial bounding-box filter) so a course
        that loops back near itself can't splice in an unrelated segment.
        """
        self._follow_course_points = []
        self._follow_activity_points = []
        self._follow_activity_dot = None
        self._follow_matched_course_dot = None
        self._follow_gap_m = None
        self._follow_rungs = []

        if self._course_x_m is None or self.raw_dist is None or not len(self.raw_dist):
            return
        assert self._course_y_m is not None  # set together with _course_x_m

        half = self.FOLLOW_HALF_WIDTH_M
        scale = (self.SIZE - 2 * self.MARGIN) / (2 * half)
        cx = float(self._course_x_m[self.current_idx])
        cy = float(self._course_y_m[self.current_idx])

        def to_local(x_m, y_m):
            return QPointF(self.SIZE / 2 + (x_m - cx) * scale, self.SIZE / 2 - (y_m - cy) * scale)

        def slice_indices(dist_axis, center_m):
            slack = half * self.FOLLOW_SLICE_FACTOR
            lo = max(0, bisect.bisect_left(dist_axis, center_m - slack))
            hi = min(len(dist_axis), bisect.bisect_right(dist_axis, center_m + slack))
            return lo, hi

        dist_m_course = pct * self.raw_dist[-1]
        lo_c, hi_c = slice_indices(self.raw_dist, dist_m_course)
        self._follow_course_points = [
            to_local(xi, yi)
            for xi, yi in zip(self._course_x_m[lo_c:hi_c], self._course_y_m[lo_c:hi_c])
        ]

        if self._activity_x_m is not None and self.activity_dist is not None:
            assert self._activity_y_m is not None  # set together with _activity_x_m
            activity_total_m = self.activity_dist[-1]
            dist_m_activity = pct * activity_total_m
            lo_a, hi_a = slice_indices(self.activity_dist, dist_m_activity)
            act_d_slice = self.activity_dist[lo_a:hi_a]
            act_x_slice = self._activity_x_m[lo_a:hi_a]
            act_y_slice = self._activity_y_m[lo_a:hi_a]

            # Line drawn from the dense track when available (see
            # set_activity_track's dense_lats docstring) -- windowed by
            # its OWN distance axis (same total as activity_dist, just
            # far more finely sampled) the same way as the sparse arrays
            # above. act_d_slice/act_x_slice/act_y_slice (sparse) stay
            # untouched below for the dot/rung matching, which must stay
            # tied to real recorded samples.
            if self._activity_dense_x_m is not None and self.activity_dense_dist is not None:
                assert self._activity_dense_y_m is not None  # set together with _activity_dense_x_m
                lo_d, hi_d = slice_indices(self.activity_dense_dist, dist_m_activity)
                line_x_slice = self._activity_dense_x_m[lo_d:hi_d]
                line_y_slice = self._activity_dense_y_m[lo_d:hi_d]
            else:
                line_x_slice = act_x_slice
                line_y_slice = act_y_slice
            self._follow_activity_points = [
                to_local(xi, yi) for xi, yi in zip(line_x_slice, line_y_slice)
            ]

            # Ladder rungs: one per VISIBLE Activity sample (static across
            # the whole window, not just the highlighted one), each
            # connecting that real recorded point to the course position
            # at the SAME pct (act_d_slice / activity_total_m, mapped onto
            # the course's own total). Only the course side is
            # interpolated -- same "only interpolate the reference side"
            # convention the Δt panel and calibrator use. Every Activity
            # sample has a valid counterpart since pct is always in [0, 1]
            # by construction. A rung's length is the pct-matching gap at
            # that sample -- real cross-track deviation plus whatever the
            # Activity/course distance-scale mismatch contributes.
            course_d_at_act = (act_d_slice / activity_total_m) * self.raw_dist[-1]
            course_x_at_act = np.interp(course_d_at_act, self.raw_dist, self._course_x_m)
            course_y_at_act = np.interp(course_d_at_act, self.raw_dist, self._course_y_m)
            self._follow_rungs = [
                (to_local(axi, ayi), to_local(cxi, cyi))
                for axi, ayi, cxi, cyi in zip(
                    act_x_slice, act_y_slice, course_x_at_act, course_y_at_act,
                )
            ]

            # Highlighted pair: mirrors exactly what core.calibrator.
            # objective_calibration compares -- a REAL, un-interpolated
            # Activity sample (v_actual_grid is read directly off real
            # samples) against the simulated/course trace interpolated at
            # THAT sample's own exact pct, never at a pct interpolated on
            # both sides. Singled out here as two distinct dots (one
            # specific rung above) so it reads clearly against the rest.
            ax = float(self._activity_x_m[self.activity_current_idx])
            ay = float(self._activity_y_m[self.activity_current_idx])
            act_own_pct = float(self.activity_dist[self.activity_current_idx]) / activity_total_m
            self._follow_activity_dot = to_local(ax, ay)
            mcx = float(np.interp(act_own_pct * self.raw_dist[-1], self.raw_dist, self._course_x_m))
            mcy = float(np.interp(act_own_pct * self.raw_dist[-1], self.raw_dist, self._course_y_m))
            self._follow_matched_course_dot = to_local(mcx, mcy)
            self._follow_gap_m = float(((ax - mcx) ** 2 + (ay - mcy) ** 2) ** 0.5)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(self.rect(), QColor("#2a2a2a"))
        p.setPen(QPen(QColor("#555555"), 1))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if not self.points:
            return

        if self.follow_mode:
            self._paint_follow(p)
        else:
            self._paint_overview(p)

    def _paint_overview(self, p: QPainter):
        """Whole-course view — see class docstring."""
        # Activity drawn first (under), course drawn second (over) —
        # course is the fixed reference and should stay visually on top
        # where the two lines nearly coincide.
        # Dense track drawn INSTEAD of the sparse one when available (see
        # set_activity_track's dense_lats docstring) -- same line, just
        # following the fitted curve's actual shape instead of straight
        # chords between sparse samples. The cursor dot below still
        # indexes activity_points (sparse), never this.
        line_points = self.activity_dense_points or self.activity_points
        if line_points:
            # Dashed, matching AnalysisCanvas's linestyle="--" convention
            # for every other Activity line in this app.
            p.setPen(QPen(QColor(self.ACTIVITY_COLOR), 2, Qt.PenStyle.DashLine))
            p.drawPolyline(line_points)

        p.setPen(QPen(QColor(self._current_course_color()), 2, Qt.PenStyle.SolidLine))
        p.drawPolyline(self.points)

        if 0 <= self.activity_current_idx < len(self.activity_points):
            pos = self.activity_points[self.activity_current_idx]
            p.setBrush(QColor(self.ACTIVITY_COLOR))
            p.setPen(QPen(QColor("#333333"), 2))
            p.drawEllipse(pos, self.DOT_RADIUS, self.DOT_RADIUS)

        if 0 <= self.current_idx < len(self.points):
            pos = self.points[self.current_idx]
            p.setBrush(QColor(self.COURSE_COLOR))
            p.setPen(QPen(Qt.GlobalColor.white, 2))
            p.drawEllipse(pos, self.DOT_RADIUS, self.DOT_RADIUS)

    def _paint_follow(self, p: QPainter):
        """Zoomed, cursor-centred local view — see set_follow_mode."""
        if self._follow_activity_points:
            p.setPen(QPen(QColor(self.ACTIVITY_COLOR), 2, Qt.PenStyle.DashLine))
            p.drawPolyline(self._follow_activity_points)

        if self._follow_course_points:
            p.setPen(QPen(QColor(self._current_course_color()), 2, Qt.PenStyle.SolidLine))
            p.drawPolyline(self._follow_course_points)

        # Ladder rungs — static across the whole visible window, not just
        # the highlighted cursor pair (see _recompute_follow_view).
        # Drawn after both rails so they're not hidden underneath either
        # line, but before the dots so the dots stay on top of everything.
        if self._follow_rungs:
            p.setPen(QPen(QColor(self.RUNG_COLOR), 1, Qt.PenStyle.SolidLine))
            for activity_pt, course_pt in self._follow_rungs:
                p.drawLine(activity_pt, course_pt)

        # Highlighted rung: the current cursor's own Activity/matched-course
        # pair, drawn thicker than the regular rungs above so its Δ
        # doesn't get lost among every other visible sample's own rung.
        if self._follow_activity_dot is not None and self._follow_matched_course_dot is not None:
            p.setPen(QPen(QColor(self.RUNG_COLOR), self.HIGHLIGHT_RUNG_WIDTH, Qt.PenStyle.SolidLine))
            p.drawLine(self._follow_activity_dot, self._follow_matched_course_dot)

        if self._follow_activity_dot is not None:
            p.setBrush(QColor(self.ACTIVITY_COLOR))
            p.setPen(QPen(QColor("#333333"), 2))
            p.drawEllipse(self._follow_activity_dot, self.DOT_RADIUS, self.DOT_RADIUS)

        # Matched course dot: the course point corresponding to the
        # highlighted Activity sample (see _recompute_follow_view), drawn
        # alongside it. Plain fill, no ring — unlike the view-centring
        # course dot below, so the two green dots stay visually distinct.
        if self._follow_matched_course_dot is not None:
            p.setBrush(QColor(self._current_course_color()))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(self._follow_matched_course_dot, self.DOT_RADIUS, self.DOT_RADIUS)

        # View-centring course dot: always exactly at the widget centre
        # by construction (see _recompute_follow_view) — drawn last so
        # it stays on top. Ringed, to distinguish it from the matched
        # course dot above even where the two happen to coincide.
        center = QPointF(self.SIZE / 2, self.SIZE / 2)
        p.setBrush(QColor(self.COURSE_COLOR))
        p.setPen(QPen(Qt.GlobalColor.white, 2))
        p.drawEllipse(center, self.DOT_RADIUS, self.DOT_RADIUS)

        if self._follow_gap_m is not None:
            p.setPen(QColor("#cccccc"))
            p.drawText(6, self.SIZE - 8, f"Δ {self._follow_gap_m:.1f} m")
