"""
eidos.apps.analyzer.canvas -- Plot canvas widget.

AnalysisCanvas is the five-panel matplotlib comparison plot, driven by
TTAnalyzerWindow's distance slider (see eidos.apps.analyzer.window). The
course minimap widget it's drawn alongside lives in
eidos.apps.analyzer.minimap.
"""

import sys
from typing import Optional

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QSizePolicy

from core.activity_correspondence import activity_pct, pct_to_activity_time
from core.activity_parser import ActivityRecord
from core.data_manager import format_time_mmss
from eidos.apps.analyzer.models import Scenario, SimTrace, StrategyRecord

# ---------------------------------------------------------------------------
# I. Constants
# ---------------------------------------------------------------------------

# Index 0 is Strategy's bright saturated green, outside the Set3 family so
# it's never confused with a Rebuild. Indices 1+ rotate through
# matplotlib's 12-color "Set3" qualitative colormap for mutual
# distinguishability (see _add_rebuild's _next_rebuild_num). Regenerate via:
#   [matplotlib.colors.to_hex(matplotlib.colormaps['Set3'](i)) for i in range(12)]
SCENARIO_COLORS = [
    "#00CC00",
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
]

MONOSPACE_FONT = "Menlo" if sys.platform == "darwin" else "Consolas"


class AnalysisCanvas(FigureCanvas):
    """
    Five-panel matplotlib canvas displaying the Analyzer comparison plots.

    Panels (shared pct-of-own-course axis — see core.activity_correspondence):
        0. Altitude     — GPX-derived course elevation, drawn per scenario
                          so a selected Rebuild's highlight applies here too
        1. Power        — planned target steps + actual FIT power + scenario traces
        2. Velocity     — scenario traces + activity FIT speed
        3. W' remaining — scenario traces
        4. Δ time       — cumulative time delta vs strategy (activity − predicted)

    The x-axis is pct: each series' own distance divided by that SAME
    series' own total distance (Strategy/Rebuild: course_distance_m /
    trace.x_traj[-1]; Activity: activity_pct(), since Activity's own
    GPS-spline-derived total distance routinely disagrees with the
    course's by a percent or more). Plotting raw distance instead would
    misrepresent how far into ITS OWN ride each series actually was at a
    shared x position. Every series spans exactly [0, 1] by construction,
    so no per-series clipping is needed anywhere in this file.

    All scenario SimTraces, and the FIT "Activity" overlay, are
    interpolated onto a common pct grid before plotting so multiple traces
    overlay cleanly. This is a display-only grid, resampled fresh on every
    redraw, independent of the physics engine's own PowerBlocks input,
    which stays on the FIT file's native sample spacing (see
    core.activity_parser.build_zoh_power_blocks). Course geometry is
    always GPX-derived (see Scenario's docstring).

    A slider-driven cursor (see TTAnalyzerWindow._on_slider_moved) is drawn
    as an axvline on every panel via set_cursor_pct(), which also floats a
    small value label next to the cursor on each panel, at each visible
    series' own data value (Strategy/Activity/highlighted Rebuild),
    colour-matched to that series' line. Both are a deliberately cheap
    path (existing artists moved/re-labelled + plain np.interp lookups
    against each trace's own native arrays) that never touches the shared
    grid above, so dragging the slider doesn't trigger a full redraw.
    """

    GRID_DS_M = 0.01     # target physical resolution [m] of the pct grid (see _compute_pct_grid)

    PANEL_TITLES = [
        "Altitude [m]", "Power [W]", "Velocity [km/h]", "W' Balance [J]",
        "Δ Time [s]  (vs Strategy)",
    ]

    CURSOR_COLOR = "#FFCC00"

    def __init__(self, parent=None):
        # No tight_layout=True: matplotlib's persistent layout engine
        # recomputes margins on every draw, including every frame of a
        # legend drag. A dragged legend's rendered bounding box (not just
        # its anchor point) feeds into that recalculation and crushes
        # every other panel's plot area too, since they share one
        # figure-level layout — clamping the anchor alone (see
        # LEGEND_POS_MIN/MAX below) isn't enough, since an in-bounds
        # anchor can still have an oversized rendered box. Layout is
        # instead computed once from the first real update_plots() call
        # (see "_layout_frozen" below) and never touched again.
        self._fig = Figure(figsize=(10, 8))
        self._layout_frozen = False
        super().__init__(self._fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._axes = self._fig.subplots(5, 1, sharex=True)
        self._fig.patch.set_facecolor("#1e1e1e")
        self._style_axes()

        # Per-axes dragged-legend position (axes-fraction (x, y), matplotlib's
        # own draggable-legend coordinate convention), keyed by Axes object
        # identity. clear_plots()'s ax.cla() destroys the Legend object every
        # redraw, so a plain set_draggable(True) alone would reset to the
        # default corner on every redraw. Captured on mouse release below
        # and re-applied in _build_legends() so a dragged position survives.
        self._legend_positions = {}
        self.mpl_connect('button_release_event', self._on_button_release)

        # --- Cursor state (see set_cursor_pct) ---
        # Whether update_plots() has ever run — read by
        # TTAnalyzerWindow._on_slider_moved/_refresh_selection_buttons to
        # enable the slider once there's anything to scrub through. The
        # slider's own 0-1000 ratio IS the pct value directly (see
        # set_cursor_pct) — every series spans exactly [0, 1] by construction.
        self.has_data: bool = False
        # One axvline per axis, recreated at the end of every update_plots()
        # call (ax.cla() in clear_plots() destroys the previous ones).
        self._cursor_lines: list = []
        # Per-(panel, series) value labels floated next to the cursor line
        # at that series' own data value — see _create_cursor_artists's
        # "--- Cursor value labels ---" block and set_cursor_pct.
        self._cursor_texts: dict = {}
        # Cheap-lookup sources, cached at the end of every update_plots()
        # call so set_cursor_pct doesn't need any arguments beyond a pct.
        self._cursor_strategy_trace = None
        self._cursor_activity_raw = None
        self._cursor_activity_gps_speed_ms = None
        self._cursor_selected_trace = None
        self._cursor_selected_label = None
        self._cursor_selected_color = None
        self._cursor_course_s_p = None
        self._cursor_course_altitude_m = None

        # --- Activity-altitude-lag state (see set_activity_altitude_lag) ---
        # The Line2D drawn in _plot_activity_overlay's "FIT activity
        # measurements" block (None if there's no activity_raw or no valid
        # altitude samples). Moving the lag spinbox should only ever
        # reposition THIS one artist, not re-run the full update_plots().
        self._activity_alt_line = None
        # Cached source arrays (same t_src/d_src/alt_src/valid
        # _plot_activity_overlay computed) so set_activity_altitude_lag can
        # redo its one np.interp call without the full argument list.
        self._activity_alt_t_src = None
        self._activity_alt_d_src = None
        self._activity_alt_alt_src = None
        self._activity_alt_valid = None
        # activity_raw.total_distance_m, cached to convert d_src (metres)
        # to pct.
        self._activity_alt_total_distance_m = None

    # Clamp range for a dragged legend's axes-fraction anchor position —
    # an out-of-bounds legend would otherwise crush the plot area via
    # tight_layout's margin recalculation (see __init__).
    LEGEND_POS_MIN = -0.05
    LEGEND_POS_MAX = 1.05

    @classmethod
    def _clamp_legend_pos(cls, pos: tuple[float, float]) -> tuple[float, float]:
        x, y = pos
        return (
            min(max(x, cls.LEGEND_POS_MIN), cls.LEGEND_POS_MAX),
            min(max(y, cls.LEGEND_POS_MIN), cls.LEGEND_POS_MAX),
        )

    def _on_button_release(self, event):
        """Schedule a (clamped) snapshot of any dragged legend's new position.

        Deferred via QTimer.singleShot(0, ...) rather than reading
        legend._loc synchronously: matplotlib's own draggable-legend
        machinery finalizes the dropped position in ITS OWN
        button_release_event handler, connected after this one, so plain
        mpl dispatch order would read legend._loc one drag late.
        """
        QTimer.singleShot(0, self._clamp_dragged_legends)

    def _clamp_dragged_legends(self):
        """Clamp every axis' legend position and snap any out-of-bounds
        one back immediately (see _on_button_release for why this runs
        deferred, not directly from the release event)."""
        changed = False
        for ax in self._axes:
            legend = ax.get_legend()
            if legend is not None and isinstance(legend._loc, tuple):
                clamped = self._clamp_legend_pos(legend._loc)
                self._legend_positions[ax] = clamped
                if clamped != legend._loc:
                    legend._loc = clamped
                    changed = True
        if changed:
            self.draw_idle()

    def _style_axes(self):
        """Apply titles, grid, and dark-theme colors to all five axes.

        Shared by __init__ and clear_plots(): ax.cla() resets these to
        matplotlib's defaults, so this must re-run after every clear.
        color= goes straight to set_title() rather than a separate
        ax.title.set_color() call: with loc="left", the text actually
        drawn is ax._left_title, not ax.title — .set_color() on ax.title
        was a silent no-op there.
        """
        for ax, title in zip(self._axes, self.PANEL_TITLES):
            ax.set_facecolor("#2a2a2a")
            ax.set_title(title, fontsize=9, loc="left", color="#cccccc")
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")
        self._axes[-1].set_xlabel("Course position [%]")
        self._axes[-1].xaxis.label.set_color("#cccccc")

    @staticmethod
    def _legend_sort_key(label: str) -> int:
        """Strategy first, Activity second, everything else unchanged.

        See _build_legends for why this exists — pulled out as its own
        method only so that block reads as "sort by this rule" rather
        than an inline lambda.
        """
        if label.startswith("Strategy"):
            return 0
        if label.startswith("Activity"):
            return 1
        return 2

    def clear_plots(self):
        """Remove all plotted lines but keep axes structure."""
        for ax in self._axes:
            ax.cla()
        self._style_axes()

    def update_plots(
        self,
        strategy: StrategyRecord,
        scenarios: list[Scenario],
        activity_raw: ActivityRecord | None,
        selected_scenario: Scenario | None = None,
        activity_altitude_lag_s: float = 0.0,
    ):
        """
        Redraw all panels from current scenarios and optional FIT activity.

        This is an orchestrator: the actual per-block plotting logic lives
        in the private `_plot_*`/`_build_*`/`_cache_*`/`_create_*` helpers
        below, one per block, called in sequence.

        Args:
            strategy:     Loaded StrategyRecord (provides planned power steps).
            scenarios:    List of Scenario instances with populated trace fields.
            activity_raw: Optional raw (un-resampled) ActivityRecord for the
                          "Activity" FIT overlay.
            selected_scenario: The Rebuild currently selected in the list,
                          if any. When set, that Rebuild's lines are drawn
                          thicker/full-opacity and every other Rebuild is
                          dimmed, across all five panels. Strategy is
                          never dimmed — it is a fixed reference line, not
                          one of the Rebuilds being compared.
            activity_altitude_lag_s: Manual shift [s] applied only to the
                          Activity altitude line (see _plot_activity_overlay).
                          Barometric altitude characteristically lags
                          GPS/speed by several seconds, device- and
                          course-dependent; delay estimation lives in
                          hyle.apps.fit2gpx_converter and is not duplicated
                          here, so this parameter only ever applies a
                          caller-supplied shift (0.0 default = no shift).
        """
        self.clear_plots()
        axes = ax_alt, ax_p, ax_v, ax_w, ax_dt = self._axes

        # clear_plots()'s ax.cla() just destroyed whatever Line2D this
        # pointed to (if any) -- reset so a stale reference doesn't linger
        # if this call has no activity_raw / no valid altitude samples to
        # re-populate it. Re-set below if the line is actually redrawn
        # (see _plot_activity_overlay).
        self._activity_alt_line = None
        self._activity_alt_t_src = None
        self._activity_alt_d_src = None
        self._activity_alt_alt_src = None
        self._activity_alt_valid = None
        self._activity_alt_total_distance_m = None

        # --- Reference trace (first scenario with a trace; this is the
        # auto-added "Strategy" scenario, override-free Planned power) ---
        baseline_trace: Optional[SimTrace] = None
        for sc in scenarios:
            if sc.trace is not None:
                baseline_trace = sc.trace
                break

        pct_grid = self._compute_pct_grid(strategy, scenarios, activity_raw)

        self._plot_planned_power(ax_p, strategy)
        activity_gps_speed_ms = self._plot_activity_overlay(
            ax_alt, ax_p, ax_v, activity_raw, activity_altitude_lag_s
        )
        self._plot_scenario_traces(
            axes, strategy, scenarios, baseline_trace, activity_raw, pct_grid, selected_scenario
        )
        self._plot_reference_lines(ax_p, ax_w, strategy, selected_scenario)
        self._build_legends()
        self._cache_cursor_sources(
            strategy, baseline_trace, activity_raw, activity_gps_speed_ms, selected_scenario
        )
        self._create_cursor_artists(axes, baseline_trace, selected_scenario)

        # One-time layout freeze (see __init__): computed once from this
        # first call's real content (titles, tick-label widths, legends),
        # then left alone for every future redraw and legend drag.
        if not self._layout_frozen:
            self._fig.tight_layout()
            self._layout_frozen = True

        self.draw_idle()

    def _compute_pct_grid(
        self,
        strategy: StrategyRecord,
        scenarios: list[Scenario],
        activity_raw: ActivityRecord | None,
    ) -> np.ndarray:
        """--- Shared pct grid ---

        Set self.has_data and return the shared pct grid (see class
        docstring) used to resample scenario traces below. Every series
        spans exactly [0, 1] by construction, so only a real-distance
        reference is needed to size the grid's resolution: the longest
        available distance among course_distance_m, Activity's own
        recorded extent, and any scenario's finish distance.
        """
        ref_distance_m = strategy.course_distance_m
        if activity_raw is not None:
            ref_distance_m = max(ref_distance_m, float(activity_raw.total_distance_m))
        for sc in scenarios:
            if sc.trace is not None:
                ref_distance_m = max(ref_distance_m, float(sc.trace.x_traj[-1]))
        self.has_data = True

        # linspace(count computed once) rather than arange+append: every
        # cell, including the last, is exactly 1/n_segments long, and 1.0
        # is always the exact last grid point rather than a
        # near-but-not-quite-1.0 neighbour that could otherwise create a
        # degenerate near-zero-length final cell.
        n_segments = max(1, round(ref_distance_m / self.GRID_DS_M))
        return np.linspace(0.0, 1.0, n_segments + 1)

    def _plot_planned_power(self, ax_p, strategy: StrategyRecord):
        """--- Planned power steps (grey) ---

        where='post': PowerBlocks' own convention — block i's power holds
        over [seg_edges[i], seg_edges[i+1]), i.e. effective from the START
        of the segment. This differs from p_traj/Activity's convention
        (see _plot_scenario_traces/_plot_activity_overlay), where the
        value represents the interval ENDING at that sample.
        """
        seg_edges = np.concatenate([[0.0], np.cumsum(strategy.planned_power_blocks.length)])
        pct_edges = seg_edges / strategy.course_distance_m
        planned_p = strategy.planned_power_blocks.power
        ax_p.step(pct_edges, np.append(planned_p, planned_p[-1]), where='post',
                  color="#66AA66", linewidth=5.0)

    def _plot_activity_overlay(
        self,
        ax_alt,
        ax_p,
        ax_v,
        activity_raw: ActivityRecord | None,
        activity_altitude_lag_s: float,
    ) -> "np.ndarray | None":
        """--- FIT activity measurements ---

        Draw the Activity's Power/Velocity/Altitude overlays and return its
        GPS-derived speed array (or None), which update_plots threads into
        _cache_cursor_sources so set_cursor_pct sees the same value.
        """
        activity_gps_speed_ms = None
        if activity_raw is not None:
            act_time_str = format_time_mmss(activity_raw.time_s[-1])
            act_lbl = f"Activity ({act_time_str}/{activity_raw.total_distance_m:.0f}m)"
            act_pct = activity_pct(activity_raw)

            # ANT+/BLE convention: power_w[i] is backward-looking, covering
            # [distance_m[i-1], distance_m[i]) (see build_zoh_power_blocks).
            # To draw it with the same forward-looking where='post'
            # convention as Planned above, re-pair it the way
            # build_zoh_power_blocks does: drop the leading sample and shift
            # the index by one, so power_w[i+1] pairs with the interval
            # starting at distance_m[i]. x/y must stay the same length
            # (act_pct in full, power_w[1:] with its own last value
            # repeated) or matplotlib's 'post' step has no next x to extend
            # the final block's power to.
            ax_p.step(act_pct,
                      np.append(activity_raw.power_w[1:], activity_raw.power_w[-1]), where='post',
                      color="#FFFFFF", linestyle="--", linewidth=0.8, alpha=0.7, label=act_lbl)

            # GPS-position-derived speed, replacing the FIT file's own
            # speed_ms outright rather than overlaying both -- speed_ms is
            # the less trustworthy of the two (slow to react at a
            # standing-start launch, device-side oversmoothing elsewhere).
            # None here (too few samples, or a lap shorter than the knot
            # spacing) means the trace is simply not drawn -- it does NOT
            # fall back to speed_ms, which would reintroduce the bias this
            # switch exists to remove.
            #
            # Read directly off the record (see ActivityRecord.gps_speed_ms)
            # -- computed once, in core.activity_parser._make_activity_record,
            # from the SAME GPS-spline curve lat_deg/lon_deg/distance_m
            # already came from -- rather than fitting a second spline on
            # top of that already-smoothed position data, which would
            # double-smooth. No re-fitting fallback for a None here: every
            # real ActivityRecord constructor populates gps_speed_ms.
            activity_gps_speed_ms = activity_raw.gps_speed_ms
            if activity_gps_speed_ms is not None:
                ax_v.plot(act_pct, activity_gps_speed_ms * 3.6, color="#FFFFFF",
                          linestyle="--", linewidth=0.8, alpha=0.7, label=act_lbl)

            # Activity altitude — raw barometric reading, manually
            # shiftable via activity_altitude_lag_s. lag is always >= 0
            # (the sensor only ever reads stale, never early): reading i is
            # repositioned to distance_at(time_i - lag_s) rather than left
            # at distance_m[i] -- a per-sample shift using each reading's
            # own recorded time, not a single whole-ride-average-speed
            # offset, which would be wrong wherever local pace differs
            # from that average.
            #
            # Uses pad_time_s/pad_distance_m/pad_altitude_m, not the
            # lap-trimmed arrays: a corrected position at the goal needs
            # readings recorded up to ALTITUDE_LAG_MAX_S seconds after the
            # goal (time_i - lag_s == goal_time only for a reading recorded
            # that much later) -- the padded arrays retain it, the
            # lap-trimmed ones don't. No unpadded fallback: every
            # activity_raw this canvas ever receives comes from
            # find_course_matches, which always populates these (see
            # activity_gps_speed_ms's own analogous note above) -- asserts
            # are for mypy's narrowing, not a real runtime possibility.
            assert activity_raw.pad_time_s is not None
            assert activity_raw.pad_distance_m is not None
            assert activity_raw.pad_altitude_m is not None
            t_src = activity_raw.pad_time_s
            d_src = activity_raw.pad_distance_m
            alt_src = activity_raw.pad_altitude_m
            valid = ~np.isnan(alt_src)
            # Cached so set_activity_altitude_lag() can recompute d_for_alt
            # on its own, without re-running update_plots.
            self._activity_alt_t_src = t_src
            self._activity_alt_d_src = d_src
            self._activity_alt_alt_src = alt_src
            self._activity_alt_valid = valid
            self._activity_alt_total_distance_m = activity_raw.total_distance_m
            if valid.any():
                d_for_alt = np.interp(t_src - activity_altitude_lag_s, t_src, d_src)
                pct_for_alt = d_for_alt / self._activity_alt_total_distance_m
                # Clip to [0, 1] before plotting: pad_distance_m runs
                # negative before the lap start and past total_distance_m
                # after the goal, so pct_for_alt routinely extends outside
                # that range, more so as the lag grows -- an unclipped line
                # would drag every panel's x-axis around as the lag
                # spinbox moves (all 5 share one x-axis, sharex=True, no
                # explicit set_xlim).
                in_range = (pct_for_alt >= 0.0) & (pct_for_alt <= 1.0)
                disp = valid & in_range
                if disp.any():
                    line, = ax_alt.plot(pct_for_alt[disp], alt_src[disp], color="#FFFFFF",
                                         linestyle="--", linewidth=0.8, alpha=0.7, label=act_lbl)
                    self._activity_alt_line = line

        return activity_gps_speed_ms

    def _plot_scenario_traces(
        self,
        axes,
        strategy: StrategyRecord,
        scenarios: list[Scenario],
        baseline_trace: Optional[SimTrace],
        activity_raw: ActivityRecord | None,
        pct_grid: np.ndarray,
        selected_scenario: Scenario | None,
    ):
        """--- Scenario traces ---

        Draw each scenario's Altitude/Power/Velocity/W'/Δt lines, highlighting
        the selected Rebuild and dimming every other one (see update_plots'
        selected_scenario docstring for that convention).

        Every trace spans exactly [0, 1] of pct by construction, so no
        per-trace clip against pct_grid is needed anywhere below.
        """
        ax_alt, ax_p, ax_v, ax_w, ax_dt = axes
        # Strategy's own baseline, as pct -- the same for every scenario
        # iteration below, computed once rather than per-iteration.
        baseline_pct = baseline_trace.x_traj / baseline_trace.x_traj[-1] if baseline_trace is not None else None
        for sc in scenarios:
            if sc.trace is None:
                continue
            tr = sc.trace
            tr_pct = tr.x_traj / tr.x_traj[-1]
            v_i = np.interp(pct_grid, tr_pct, tr.v_traj)
            w_i = np.interp(pct_grid, tr_pct, tr.w_traj)
            t_i = np.interp(pct_grid, tr_pct, tr.t_traj)

            finish_str = format_time_mmss(tr.finish_time_s)
            lbl = f"{sc.label}  ({finish_str}/{tr.x_traj[-1]:.0f}m)"

            # Highlight the selected Rebuild, dim the others — never
            # applied to Strategy, which stays a fixed full-opacity
            # reference line regardless of what's selected in the list.
            if selected_scenario is not None and sc.label != "Strategy":
                is_selected = (sc is selected_scenario)
                lw = 3.0 if is_selected else 1.2
                alpha = 1.0 if is_selected else 0.3
            else:
                lw = 1.2
                alpha = 1.0

            # Altitude: GPX-derived course geometry is identical for every
            # scenario, so this isn't new per-scenario data the way
            # Power/Velocity/W' are — drawn once per scenario anyway, in
            # that scenario's own colour, so this panel's legend and the
            # selected-Rebuild highlight stay consistent with the rest.
            ax_alt.plot(strategy.course_s_p / strategy.course_distance_m, strategy.course_altitude_m,
                        color=sc.color, linewidth=lw, alpha=alpha, label=lbl)

            # where='post': p_traj[i] holds over the forward half-interval
            # [t_i, t_i+1), same convention as Planned above.
            ax_p.step(tr_pct, tr.p_traj, where='post', color=sc.color, linewidth=lw, alpha=alpha, label=lbl)
            ax_v.plot(pct_grid, v_i * 3.6, color=sc.color, linewidth=lw, alpha=alpha, label=lbl)
            ax_w.plot(pct_grid, w_i, color=sc.color, linewidth=lw, alpha=alpha, label=lbl)

            if baseline_trace is not None:
                assert baseline_pct is not None  # computed from baseline_trace above, travels with it
                # Δtime: Activity vs Strategy, on Activity's own native FIT
                # sample times. Only Strategy (dense, ~0.1s native step) is
                # interpolated to match; resampling Activity itself would
                # mix in an artifact unrelated to real pacing error.
                if activity_raw is not None:
                    pct_native = activity_pct(activity_raw)
                    t_actual_native = activity_raw.time_s
                    t_baseline_at_native = np.interp(pct_native, baseline_pct, baseline_trace.t_traj)
                    raw_delta_native = t_actual_native - t_baseline_at_native

                    act_time_str = format_time_mmss(activity_raw.time_s[-1])
                    expected_label = f"Activity ({act_time_str}/{activity_raw.total_distance_m:.0f}m)"

                    if not any(line.get_label().startswith("Activity (") for line in ax_dt.get_lines()):
                        ax_dt.plot(pct_native, raw_delta_native, color="#FFFFFF", linestyle="--",
                                   linewidth=1.0, alpha=0.85, label=expected_label)

                # Δt(pct) = time actual reaches pct minus time baseline reaches pct
                t_baseline_at_pct = np.interp(pct_grid, baseline_pct, baseline_trace.t_traj)
                delta = t_i - t_baseline_at_pct
                ax_dt.plot(pct_grid, delta, color=sc.color, linewidth=lw, alpha=alpha, label=lbl)
                ax_dt.axhline(0, color="#555555", linewidth=0.8)

    def _plot_reference_lines(self, ax_p, ax_w, strategy: StrategyRecord, selected_scenario: Scenario | None):
        """--- CP / W' max reference lines ---

        Tracks the selected Rebuild's own cp/w_prime override, if any,
        falling back to the strategy's baseline CP/W' when nothing is
        selected. Read from raw_physiological rather than the resolved
        simulator's own params shape (which may lack CP/W' fields
        entirely, e.g. core.simulators.sim_stub.DummyPhysicsParams);
        raw_physiological is always guaranteed to have cp/w_prime, since
        every SIMULATOR_REGISTRY entry's PhysiologicalSettings is required
        to define them (see core.schema.PhysiologicalSettingsBase).
        """
        cp_ref = strategy.raw_physiological["cp"]
        w_prime_ref = strategy.raw_physiological["w_prime"]
        if selected_scenario is not None:
            cp_ref = selected_scenario.physics_overrides.get("cp", cp_ref)
            w_prime_ref = selected_scenario.physics_overrides.get("w_prime", w_prime_ref)

        ax_p.axhline(
            cp_ref, color="#FF4444",
            linewidth=0.8, linestyle="--", label=f"CP = {cp_ref:.0f} W",
        )

        ax_w.axhline(
            w_prime_ref, color="#FF4444",
            linewidth=0.8, linestyle="--",
            label=f"W' = {w_prime_ref:.0f} J",
        )

    def _build_legends(self):
        """--- Legends ---

        Draggable so a legend that overlaps busy plot content can be moved
        out of the way; dragged positions persist across redraws via
        self._legend_positions (see __init__/_on_button_release).
        Checks get_legend_handles_labels() rather than plain get_lines():
        ax.legend() with zero labelled artists prints a spurious
        "No artists with labels found" UserWarning on every redraw.
        """
        for ax in self._axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # Reorder to Strategy / Activity / everything else (without
                # this, Activity ends up first purely as an accident of
                # draw order). sorted() is stable, so entries within each
                # bucket (e.g. multiple Rebuilds) keep their relative order.
                order = sorted(range(len(labels)), key=lambda i: self._legend_sort_key(labels[i]))
                handles = [handles[i] for i in order]
                labels = [labels[i] for i in order]
                loc = self._legend_positions.get(ax, "upper right")
                legend = ax.legend(handles, labels, fontsize=7, loc=loc,
                                    facecolor="#333333", labelcolor="#cccccc",
                                    framealpha=0.7)
                legend.set_draggable(True)

    def _cache_cursor_sources(
        self,
        strategy: StrategyRecord,
        baseline_trace: Optional[SimTrace],
        activity_raw: ActivityRecord | None,
        activity_gps_speed_ms: "np.ndarray | None",
        selected_scenario: Scenario | None,
    ):
        """--- Cursor state (see set_cursor_pct) ---

        Cache the cheap-lookup sources: plain references to objects the
        caller already has as locals/params, no new computation. Refreshed
        on every redraw so a stale selected/baseline trace from a
        since-removed Rebuild doesn't linger.
        """
        self._cursor_strategy_trace = baseline_trace
        self._cursor_activity_raw = activity_raw
        self._cursor_activity_gps_speed_ms = activity_gps_speed_ms
        self._cursor_selected_trace = selected_scenario.trace if selected_scenario is not None else None
        self._cursor_selected_label = selected_scenario.label if selected_scenario is not None else None
        self._cursor_selected_color = selected_scenario.color if selected_scenario is not None else None
        self._cursor_course_s_p = strategy.course_s_p
        self._cursor_course_altitude_m = strategy.course_altitude_m

    def _create_cursor_artists(
        self,
        axes,
        baseline_trace: Optional[SimTrace],
        selected_scenario: Scenario | None,
    ):
        """(Re)create the cursor axvline on every axis, then the per-
        (panel, series) cursor value label Text artists.

        Mandatory every update_plots() call: clear_plots()'s ax.cla()
        destroys the previous ones along with everything else. Called
        after _build_legends so these label-less artists never appear in
        any legend.
        """
        ax_alt, ax_p, ax_v, ax_w, ax_dt = axes

        # Preserve the previous cursor position across the redraw rather
        # than resetting to the left edge every time.
        prev_x = self._cursor_lines[0].get_xdata()[0] if self._cursor_lines else 0.0
        self._cursor_lines = [
            ax.axvline(prev_x, color=self.CURSOR_COLOR, linewidth=1.0, alpha=0.9, zorder=10)
            for ax in self._axes
        ]

        # --- Cursor value labels ---
        # One Text artist per (panel, series) slot that might show a
        # value, floated next to the cursor line at that series' own data
        # value, colour-matched to it. set_cursor_pct positions/shows/hides
        # these on every cursor move; all start invisible/blank here until
        # the next set_cursor_pct call turns on whichever are available.
        strategy_color = baseline_trace.color if baseline_trace is not None else SCENARIO_COLORS[0]
        rebuild_color = selected_scenario.color if selected_scenario is not None else "#FFFFFF"
        text_kwargs = dict(
            fontsize=7, va="center", zorder=11, visible=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#1e1e1e", alpha=0.85, edgecolor="none"),
        )
        self._cursor_texts = {
            "altitude": ax_alt.text(0, 0, "", color="#cccccc", **text_kwargs),
            "power_strategy": ax_p.text(0, 0, "", color=strategy_color, **text_kwargs),
            "power_activity": ax_p.text(0, 0, "", color="#FFFFFF", **text_kwargs),
            "power_rebuild": ax_p.text(0, 0, "", color=rebuild_color, **text_kwargs),
            "vel_strategy": ax_v.text(0, 0, "", color=strategy_color, **text_kwargs),
            "vel_activity": ax_v.text(0, 0, "", color="#FFFFFF", **text_kwargs),
            "vel_rebuild": ax_v.text(0, 0, "", color=rebuild_color, **text_kwargs),
            "wbal_strategy": ax_w.text(0, 0, "", color=strategy_color, **text_kwargs),
            "wbal_rebuild": ax_w.text(0, 0, "", color=rebuild_color, **text_kwargs),
            "dt_activity": ax_dt.text(0, 0, "", color="#FFFFFF", **text_kwargs),
            "dt_rebuild": ax_dt.text(0, 0, "", color=rebuild_color, **text_kwargs),
        }

    def set_activity_altitude_lag(self, lag_s: float):
        """
        Reposition the Activity-altitude dashed line for a new manual lag,
        without touching anything else.

        Deliberately cheap — safe to call on every spinbox tick, unlike a
        full update_plots() (ax.cla() on all 5 panels, full re-plot, legend
        rebuild), since this only concerns one dashed line. Recomputes only
        that line's x-data, via the same np.interp + range clip
        _plot_activity_overlay used, from the arrays it cached.

        No-op if there's no line to move (no activity loaded, or no valid
        altitude samples at all).
        """
        if self._activity_alt_line is None:
            return
        t_src = self._activity_alt_t_src
        d_src = self._activity_alt_d_src
        alt_src = self._activity_alt_alt_src
        valid = self._activity_alt_valid
        d_for_alt = np.interp(t_src - lag_s, t_src, d_src)
        pct_for_alt = d_for_alt / self._activity_alt_total_distance_m
        in_range = (pct_for_alt >= 0.0) & (pct_for_alt <= 1.0)
        disp = valid & in_range
        self._activity_alt_line.set_data(pct_for_alt[disp], alt_src[disp])
        self.draw_idle()

    def set_cursor_pct(self, pct: float) -> dict:
        """
        Move the cursor line and its floating value labels on every panel
        (see _place_cursor_texts), and return the same Strategy/Activity/
        selected-Rebuild/Altitude values at pct (see class docstring).

        Deliberately cheap — safe to call on every slider-drag tick. Only
        moves the axvline Line2D objects created by _create_cursor_artists
        (no cla(), no re-plotting, no legend rebuild), and reads each
        series via plain np.interp against its own native arrays, never
        the display grid update_plots() builds for a full redraw. All
        lookup sources are cached by _cache_cursor_sources.

        Every series spans exactly [0, 1] of pct by construction, so no
        series is ever "past its extent" at a valid (clamped) pct -- only
        its own presence (loaded or not) gates whether a value is returned.

        Power is read back via linear np.interp for simplicity even
        though it is drawn as a ZOH step — right at a step edge the
        numeric readout can read a touch smoothed vs. the plotted line.
        Velocity/W'/Altitude are continuous and match the drawn line
        exactly.

        Returns:
            {} if nothing has been plotted yet. Otherwise a dict with
            "pct" (clamped to [0, 1]), "altitude_m" (float or None), and
            "strategy"/"activity"/"rebuild" — each either None (series
            not loaded) or a dict of "power_w"/"velocity_kmh"/"wbal_j",
            plus "delta_t_s" for "activity"/"rebuild" (omitted when no
            Strategy trace is cached to diff against), plus
            "label"/"color" for "rebuild". "activity"'s own
            "velocity_kmh" is likewise omitted when no GPS-derived speed
            could be obtained for this activity (see
            _plot_activity_overlay's own gps_speed_ms handling).
        """
        if not self.has_data or not self._cursor_lines:
            return {}

        pct = max(0.0, min(pct, 1.0))
        for line in self._cursor_lines:
            line.set_xdata([pct, pct])

        altitude_m = None
        if self._cursor_course_s_p is not None:
            altitude_m = float(np.interp(
                pct * self._cursor_course_s_p[-1], self._cursor_course_s_p, self._cursor_course_altitude_m
            ))

        strategy_vals = None
        strat_tr = self._cursor_strategy_trace
        if strat_tr is not None:
            d_strat = pct * strat_tr.x_traj[-1]
            strategy_vals = {
                "power_w": float(np.interp(d_strat, strat_tr.x_traj, strat_tr.p_traj)),
                "velocity_kmh": float(np.interp(d_strat, strat_tr.x_traj, strat_tr.v_traj)) * 3.6,
                "wbal_j": float(np.interp(d_strat, strat_tr.x_traj, strat_tr.w_traj)),
            }

        activity_vals = None
        act = self._cursor_activity_raw
        if act is not None:
            act_pct = activity_pct(act)
            activity_vals = {
                "power_w": float(np.interp(pct, act_pct, act.power_w)),
            }
            # GPS-derived, same series as the drawn line in
            # _plot_activity_overlay. Key omitted (not just None), same
            # convention as delta_t_s below, when no GPS-derived speed
            # could be obtained for this activity.
            gps_speed_ms = self._cursor_activity_gps_speed_ms
            if gps_speed_ms is not None:
                activity_vals["velocity_kmh"] = float(np.interp(pct, act_pct, gps_speed_ms)) * 3.6
            if strat_tr is not None:
                t_act = float(pct_to_activity_time(act, pct))
                t_strat = float(np.interp(pct * strat_tr.x_traj[-1], strat_tr.x_traj, strat_tr.t_traj))
                activity_vals["delta_t_s"] = t_act - t_strat

        rebuild_vals = None
        reb_tr = self._cursor_selected_trace
        if reb_tr is not None:
            d_reb = pct * reb_tr.x_traj[-1]
            rebuild_vals = {
                "label": self._cursor_selected_label,
                "color": self._cursor_selected_color,
                "power_w": float(np.interp(d_reb, reb_tr.x_traj, reb_tr.p_traj)),
                "velocity_kmh": float(np.interp(d_reb, reb_tr.x_traj, reb_tr.v_traj)) * 3.6,
                "wbal_j": float(np.interp(d_reb, reb_tr.x_traj, reb_tr.w_traj)),
            }
            if strat_tr is not None:
                t_reb = float(np.interp(d_reb, reb_tr.x_traj, reb_tr.t_traj))
                t_strat = float(np.interp(pct * strat_tr.x_traj[-1], strat_tr.x_traj, strat_tr.t_traj))
                rebuild_vals["delta_t_s"] = t_reb - t_strat

        self._place_cursor_texts(pct, altitude_m, strategy_vals, activity_vals, rebuild_vals)
        self.draw_idle()

        return {
            "pct": pct,
            "altitude_m": altitude_m,
            "strategy": strategy_vals,
            "activity": activity_vals,
            "rebuild": rebuild_vals,
        }

    def _place_cursor_texts(
        self, pct: float, altitude_m: float | None,
        strategy_vals: dict | None, activity_vals: dict | None, rebuild_vals: dict | None,
    ):
        """
        Position/show/hide the per-(panel, series) value labels created in
        _create_cursor_artists, floating each one next to the cursor line
        at that series' own data value, colour-matched to it.

        Anchor side flips (right-of-cursor near the left edge of the
        course, left-of-cursor near the right edge) so a label never runs
        off the panel when the cursor nears either end.
        """
        if not self._cursor_texts:
            return

        ha = "left" if pct < 0.5 else "right"
        offset = 0.015
        text_x = pct + offset if ha == "left" else pct - offset

        def place(key: str, y, text_str):
            txt = self._cursor_texts.get(key)
            if txt is None:
                return
            if y is None or text_str is None:
                txt.set_visible(False)
                return
            txt.set_position((text_x, y))
            txt.set_ha(ha)
            txt.set_text(text_str)
            txt.set_visible(True)

        place("altitude", altitude_m, f"{altitude_m:.0f}m" if altitude_m is not None else None)

        def place_group(vals, specs):
            for key, field, fmt in specs:
                v = vals.get(field) if vals is not None else None
                place(key, v, fmt.format(v) if v is not None else None)

        place_group(strategy_vals, [
            ("power_strategy", "power_w", "{:.0f}W"),
            ("vel_strategy", "velocity_kmh", "{:.1f}km/h"),
            ("wbal_strategy", "wbal_j", "{:.0f}J"),
        ])
        place_group(activity_vals, [
            ("power_activity", "power_w", "{:.0f}W"),
            ("vel_activity", "velocity_kmh", "{:.1f}km/h"),
            ("dt_activity", "delta_t_s", "{:+.1f}s"),
        ])
        place_group(rebuild_vals, [
            ("power_rebuild", "power_w", "{:.0f}W"),
            ("vel_rebuild", "velocity_kmh", "{:.1f}km/h"),
            ("wbal_rebuild", "wbal_j", "{:.0f}J"),
            ("dt_rebuild", "delta_t_s", "{:+.1f}s"),
        ])

