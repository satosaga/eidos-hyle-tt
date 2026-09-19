"""
eidos.apps.analyzer.widgets -- Analyzer-specific control-panel Qt widgets.

NoScrollDoubleSpinBox/NoScrollSpinBox, SensitivityBarWidget (one row's
Morris/Sobol' bar, embedded in PhysicsOverridePanel next to its Auto Fit
checkbox), PhysicsOverridePanel (the physics-parameter override form
embedded in the Rebuilds panel), and RebuildColorDelegate (keeps a
Rebuild's own color visible when its list row is selected).
"""

from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

import core.calibrator as calibrator
from core.schema import bounds_from_field, field_display_label, gui_decimals_from_field
from core.simulators import (
    SimulatorSpec,
    calibratable_physical_keys,
    physiological_only_keys,
)

# ---------------------------------------------------------------------------
# VII. Physics parameter override panel
# ---------------------------------------------------------------------------

class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox where only typing a value changes it.

    Simulation results aren't live/reactive to these controls — there's no
    benefit to incremental nudging (wheel, spin arrows, Up/Down keys) over
    just typing the number, and every one of those is also an easy way to
    silently mutate a value without meaning to. For a tool built on "Exact
    Intelligent Design" (explicit, deliberate parameter values — see
    EIDOS^TT's own naming and architecture notes), that's not acceptable.

    stepBy() is the single method Qt routes ALL step-based interactions
    through (spin arrows, mouse wheel, and Up/Down arrow keys), so
    overriding it as a no-op blocks all three at once. wheelEvent() is
    still overridden separately to ignore (not just no-op) the event, so
    the scroll passes through to the parent scroll area instead of being
    swallowed here.
    """

    def stepBy(self, steps):
        pass

    def wheelEvent(self, event):
        event.ignore()


class NoScrollSpinBox(QSpinBox):
    """Integer sibling of NoScrollDoubleSpinBox -- same reasoning, same
    stepBy()-no-op/wheelEvent()-ignore mechanism, for integer-valued
    controls (e.g. TTAnalyzerWindow's Morris r / Sobol' N spinbox)."""

    def stepBy(self, steps):
        pass

    def wheelEvent(self, event):
        event.ignore()


class SensitivityBarWidget(QWidget):
    """
    One physics-param row's Morris/Sobol' sensitivity, drawn as two
    stacked horizontal-bar lanes -- embedded directly in
    PhysicsOverridePanel's own QGridLayout, next to that row's Auto Fit
    checkbox (see PhysicsOverridePanel.show_morris_sensitivity/
    show_sobol_sensitivity/clear_sensitivity). Plain QPainter, not
    matplotlib: this widget IS a cell in the panel's own grid, so its
    row is pixel-aligned with that row's label/spinbox/checkbox by
    construction -- no cross-toolkit alignment to solve the way a
    separate matplotlib figure would need.

    Two mutually-exclusive display modes (whichever show_* was called
    last wins) plus an empty state (clear(), e.g. before the first
    sensitivity run completes or while a fresh one is running). Each
    mode uses two independent lanes rather than one overlaid bar,
    because Sobol' S1 can come out above ST for a noisy/under-converged
    estimate (S1 <= ST only holds in exact theory, not for finite-sample
    bootstrap estimates) -- an overlaid single-bar layout would then
    show the "front" bar poking out past the "back" one, reading as a
    rendering bug rather than the genuine estimation noise it is. Two
    independent lanes represent that case as plainly as any other:
    just two bars of different lengths.

    - Morris: top lane is mu_star with a whisker for its own
      mu_star_conf (SALib's bootstrap CI half-width); bottom lane is
      sigma -- not a measurement-uncertainty interval around mu_star
      (sigma flags nonlinear/interaction-driven effects, it's a
      different statistic, not mu_star's error bar), but ALWAYS given a
      zero-width whisker at its own bar's tip anyway, purely so every
      lane in both modes draws the same end-cap tick mark regardless of
      whether that row happens to have a real interval -- the
      alternative (only draw a whisker where SALib gives one) means
      mu_star and sigma render with a different silhouette on every row
      where mu_star's own CI happens to be near zero too, which reads
      as broken rather than as "no interval here". mu_star and sigma
      share ONE vmax across both lanes (see show_morris_sensitivity) --
      both are already in the same physical units ("RMSE change per
      full sweep of this key's own range," see sample_morris_
      sensitivity), so a shared scale makes a row's own mu_star-bar-vs-
      sigma-bar length directly comparable, matching the classic Morris
      (mu_star, sigma) scatter's sigma=mu_star diagonal (calibration_
      diagnostics.plot_morris_mustar_sigma_scatter, opened by pressing
      either lane) -- a separate per-statistic scale here would make
      that comparison meaningless on the bar while the popup scatter
      right next to it draws a diagonal implying it IS meaningful.
    - Sobol': top lane is S1 (darker color, this parameter's own share
      alone), bottom lane is ST (lighter color, total effect including
      interactions; S1 <= ST in theory). Both on the fixed [0, 1] scale
      S1/ST are already bounded to, each with its own whisker for
      s1_conf/st_conf.

    Every whisker (both modes) uses the same flat, quiet color.

    The two lanes sit flush against each other (no gap) so the pair
    reads as one bar split in half, not two unrelated bars.
    """
    _MORRIS_MU_COLOR = QColor("#4bb062")
    _MORRIS_SIGMA_COLOR = QColor("#8a8a8a")
    _SOBOL_ST_COLOR = QColor("#87b082")
    _SOBOL_S1_COLOR = QColor("#1f6f36")
    _WHISKER_COLOR = QColor("#f0f0f0")
    _EMPTY_COLOR = QColor("#3a3a3a")
    _LANE_GAP = 0
    _WHISKER_HEIGHT_FRAC = 2 / 3
    # No numbers spelled out here (e.g. "Morris μ*=1.2±0.3 σ=0.4") --
    # that would duplicate exactly what pressing the bar already reveals
    # in the scatter popup's own title (see calibration_diagnostics.
    # plot_morris_mustar_sigma_scatter/plot_sensitivity_effect_scatter/
    # plot_sensitivity_interaction_scatter); the hover just needs to say
    # that pressing does.
    _TOOLTIP_TEXT = "Press and hold for details"

    # press-and-hold-to-peek, not click-to-toggle: `clicked` on mouse
    # down opens a scatter popup of this row's raw evaluated points
    # (TTAnalyzerWindow, via PhysicsOverridePanel.sensitivity_bar_
    # clicked, which adds which row's own free_key this was); `released`
    # on mouse up closes it again. No top/bottom lane distinction --
    # neither Morris (mu_star/sigma share the same underlying (x, y), so
    # there's only ever one plot) nor Sobol' (S1 and ST are shown
    # together as a pair regardless of which half was pressed) reads
    # position within the bar. `clicked` is never emitted while empty
    # (clear()'d, no _mode) -- there's no data to show yet, so no press
    # target either (see clear()/show_morris/show_sobol's cursor
    # handling); `released` always fires on any left-button release,
    # even if _mode became None in between, since hiding an
    # already-hidden popup is a harmless no-op.
    clicked = Signal()
    released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(90)
        self.setMinimumHeight(28)
        self._mode: str | None = None  # None | "morris" | "sobol"
        # morris
        self._mu_frac = 0.0
        self._mu_whisker: tuple[float, float] | None = None
        self._mu_whisker_clipped: tuple[bool, bool] = (False, False)
        self._sigma_frac = 0.0
        self._sigma_whisker: tuple[float, float] | None = None
        # sobol
        self._s1_frac = 0.0
        self._s1_whisker: tuple[float, float] | None = None
        self._s1_whisker_clipped: tuple[bool, bool] = (False, False)
        self._st_frac = 0.0
        self._st_whisker: tuple[float, float] | None = None
        self._st_whisker_clipped: tuple[bool, bool] = (False, False)

    def clear(self) -> None:
        self._mode = None
        self.setToolTip("")
        self.unsetCursor()
        self.update()

    def show_morris(
        self, mu_star: float, mu_star_conf: float, sigma: float, vmax: float,
    ) -> None:
        """vmax: ONE shared scale for both mu_star and sigma, across
        every row in the panel (see class docstring for why shared, not
        per-statistic) -- 0 (nothing to show yet) draws an empty lane
        rather than dividing by zero. vmax is expected to already
        include mu_star_conf's extent (see show_morris_sensitivity) so
        the whisker itself is rarely clipped."""
        self._mode = "morris"
        self._mu_frac = 0.0 if vmax <= 0 else min(mu_star / vmax, 1.0)
        if vmax <= 0:
            self._mu_whisker = None
            self._mu_whisker_clipped = (False, False)
        else:
            raw_lo, raw_hi = mu_star - mu_star_conf, mu_star + mu_star_conf
            self._mu_whisker_clipped = (raw_lo < 0.0, raw_hi > vmax)
            lo = max(0.0, raw_lo) / vmax
            hi = min(raw_hi, vmax) / vmax
            self._mu_whisker = (min(lo, 1.0), min(hi, 1.0))
        self._sigma_frac = 0.0 if vmax <= 0 else min(sigma / vmax, 1.0)
        # Zero-width -- sigma has no real CI (see class docstring) -- but
        # still a whisker tuple, so the bottom lane gets the same
        # end-cap tick every top lane gets, never a bare bar.
        self._sigma_whisker = (self._sigma_frac, self._sigma_frac)
        self.setToolTip(self._TOOLTIP_TEXT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def show_sobol(self, s1: float, s1_conf: float, st: float, st_conf: float) -> None:
        # Bar tips clipped to [0, 1] for DRAWING only -- S1/ST can in
        # principle read outside [0, 1] (a real, if rare, SALib
        # bootstrap-estimate artifact, same one that lets S1 > ST -- see
        # class docstring); the tooltip still states the real, unclipped
        # values. Whisker ends are ALSO clipped for drawing (same [0, 1]
        # axis), but here whether each end was actually clipped is kept
        # (self._s1_whisker_clipped/_st_whisker_clipped) -- _draw_lane
        # omits the end-cap glyph on a clipped end entirely (see its own
        # docstring for why: there is no real position to mark there).
        self._mode = "sobol"
        self._s1_frac = max(0.0, min(s1, 1.0))
        self._st_frac = max(0.0, min(st, 1.0))
        s1_raw_lo, s1_raw_hi = s1 - s1_conf, s1 + s1_conf
        self._s1_whisker = (max(0.0, min(s1_raw_lo, 1.0)), max(0.0, min(s1_raw_hi, 1.0)))
        self._s1_whisker_clipped = (s1_raw_lo < 0.0, s1_raw_hi > 1.0)
        st_raw_lo, st_raw_hi = st - st_conf, st + st_conf
        self._st_whisker = (max(0.0, min(st_raw_lo, 1.0)), max(0.0, min(st_raw_hi, 1.0)))
        self._st_whisker_clipped = (st_raw_lo < 0.0, st_raw_hi > 1.0)
        self.setToolTip(self._TOOLTIP_TEXT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._mode is not None:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.released.emit()
        super().mouseReleaseEvent(event)

    def _draw_lane(
        self, painter: QPainter, rect: QRect, frac: float, color: QColor,
        whisker: tuple[float, float] | None = None,
        whisker_clipped: tuple[bool, bool] = (False, False),
    ) -> None:
        painter.fillRect(rect, self._EMPTY_COLOR)
        # Sub-pixel (QRectF), not round()-ed to whole pixels -- with
        # antialiasing on (see paintEvent), two whisker ends only ~1px
        # apart in true value still render as two distinguishable,
        # smoothly-blended marks instead of both snapping to the same
        # integer pixel and merging into one blob. This is strictly a
        # rendering-precision improvement -- it does not change what any
        # of these numbers mean, only how faithfully a narrow-but-real
        # gap between two values shows up on screen.
        bar_w = rect.width() * frac
        painter.fillRect(QRectF(rect.x(), rect.y(), bar_w, rect.height()), color)
        if whisker is None:
            return
        lo_frac, hi_frac = whisker
        lo_clipped, hi_clipped = whisker_clipped
        x_lo = max(float(rect.x()), min(rect.x() + rect.width() * lo_frac, float(rect.right())))
        x_hi = max(float(rect.x()), min(rect.x() + rect.width() * hi_frac, float(rect.right())))
        cap_h = max(1.0, rect.height() * self._WHISKER_HEIGHT_FRAC)
        cap_y = rect.y() + (rect.height() - cap_h) / 2
        mid_y = rect.y() + rect.height() / 2
        cap_w = 1.2
        # A clipped end has no real position to mark -- the true bound
        # lies somewhere past the edge of what this lane can show, not
        # AT x_lo/x_hi. Drawing a cap there (in any color) would assert
        # a specific endpoint that doesn't exist. The line itself still
        # runs up to that edge -- only the end-cap glyph is omitted.
        if not lo_clipped:
            painter.fillRect(QRectF(x_lo - cap_w / 2, cap_y, cap_w, cap_h), self._WHISKER_COLOR)
        if not hi_clipped:
            painter.fillRect(QRectF(x_hi - cap_w / 2, cap_y, cap_w, cap_h), self._WHISKER_COLOR)
        line_w = max(x_hi - x_lo, cap_w)
        painter.fillRect(QRectF(x_lo, mid_y - 0.6, line_w, 1.2), self._WHISKER_COLOR)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outer = self.rect().adjusted(1, 3, -1, -3)
        lane_h = (outer.height() - self._LANE_GAP) // 2
        top = QRect(outer.x(), outer.y(), outer.width(), lane_h)
        bottom = QRect(outer.x(), outer.bottom() - lane_h + 1, outer.width(), lane_h)

        if self._mode == "morris":
            self._draw_lane(
                painter, top, self._mu_frac, self._MORRIS_MU_COLOR,
                self._mu_whisker, self._mu_whisker_clipped,
            )
            self._draw_lane(
                painter, bottom, self._sigma_frac, self._MORRIS_SIGMA_COLOR, self._sigma_whisker,
            )
        elif self._mode == "sobol":
            self._draw_lane(
                painter, top, self._s1_frac, self._SOBOL_S1_COLOR,
                self._s1_whisker, self._s1_whisker_clipped,
            )
            self._draw_lane(
                painter, bottom, self._st_frac, self._SOBOL_ST_COLOR,
                self._st_whisker, self._st_whisker_clipped,
            )
        else:
            painter.fillRect(outer, self._EMPTY_COLOR)

        painter.end()


class PhysicsOverridePanel(QWidget):
    """
    Widget of QDoubleSpinBox controls for overriding physical parameters.

    Embedded directly inside the "Rebuilds" panel (see
    TTAnalyzerWindow._build_rebuilds_panel) rather than being its own
    bordered group box, so Label / physics params / Add button read as
    one form.

    Built for one specific simulator, passed in at construction
    (`simulator_spec`) -- since one TTAnalyzerWindow is always opened
    against exactly one StrategyRecord for its whole lifetime (there is
    no "load a different strategy" flow), and that strategy's own
    simulator never changes underneath this panel, its PARAM_SPECS/
    WBAL_ONLY_KEYS are computed ONCE, per instance, from that simulator's
    own physical_param_model/physiological_param_model -- not from a
    fixed default. Covers every one of `simulator_spec`'s own
    `core.simulators.calibratable_physical_keys(simulator_spec.key)`
    keys, so a Rebuild produced by Auto Fit is always fully inspectable
    here too: clicking it (_on_rebuild_item_clicked -> set_values) and
    its list-item tooltip (format_overrides_tooltip) both iterate this
    same PARAM_SPECS, so whichever calibratable key Auto Fit touched
    (e.g. rider mass) is visible in both places, not silently dropped.
    Each row has its own Reset button that restores just that one
    parameter to the strategy JSON's value — resetting one shouldn't
    discard edits made to the others.

    Also covers `simulator_spec`'s further physiological params — e.g.
    sim_kiritsubo's cp, w_prime, p_max, w_prime_recovery_rate,
    vitality_loss_rate (WBAL_ONLY_KEYS below, which for a different
    simulator entry can be a different, possibly smaller, set — see
    core.simulators.sim_stub.PhysiologicalSettings for a real example) —
    that feed W' Balance (see core.simulators.sim_kiritsubo's own
    CP_eff/W_prime/P_max_physio/K/V_slope locals)
    but never enter the velocity/position ODE, so they have zero effect
    on Auto Fit's objective (velocity RMSE against the FIT recording). No
    Auto Fit checkbox is offered for these rows — there's no velocity
    signal for DE to fit them against — but Reset still applies, same as
    every other row.
    """

    # Emitted when a row's own SensitivityBarWidget is pressed -- key --
    # and released (no key needed there; TTAnalyzerWindow only ever has
    # one scatter popup open at a time, so "the mouse was released
    # somewhere" is enough to know to close it). This panel is the one
    # place that knows which key each SensitivityBarWidget belongs to
    # (see self._sensitivity_widgets), so it's the one that adds that
    # key onto the bare press signal before handing it up to
    # TTAnalyzerWindow, which opens the actual scatter popup.
    sensitivity_bar_clicked = Signal(str)
    sensitivity_bar_released = Signal()

    # Emitted whenever anything that feeds a Sensitivity run changes:
    # any row's "Fit" checkbox (bound to QCheckBox.toggled, not clicked,
    # so a direct click, _uncheck_auto_fit's own setChecked(), and
    # set_auto_fit_checks()'s bulk restore all fire this the same way),
    # or any Auto Fit-eligible row's spinbox value -- free or fixed alike
    # (bound to QDoubleSpinBox.valueChanged, so typing, Reset, and
    # set_values()'s bulk restore all fire this too; see
    # _build_param_row). Every one of those is a real input to
    # calibrator.build_calibration_inputs (free_keys via which boxes are
    # checked, everything else via get_overrides()'s fixed baseline), so
    # none of them may go unwired here -- a row whose edits never reached
    # this signal would silently screen against a stale value. The W'
    # Balance-only rows are deliberately NOT wired to this signal at
    # all -- see WBAL_ONLY_KEYS: those keys never enter the velocity/
    # position ODE, so no value they hold can ever move the RMSE this
    # signal's two listeners both key off (Sensitivity screening here,
    # and PhysicsOverridePanel's own fit-confirmed styling).
    #
    # TTAnalyzerWindow connects this to rerun Sensitivity screening
    # against whatever's checked and entered NOW, cancelling whatever run
    # is already in flight first (see TTAnalyzerWindow._run_sensitivity)
    # -- that cancellation is what makes firing on every intermediate
    # state safe: a burst of these (e.g. set_auto_fit_checks()/
    # set_values() flipping several rows in a row, or Qt emitting
    # valueChanged once per keystroke while typing a number) each
    # supersede the previous one, so only the LAST -- the actual final,
    # settled state -- ever finishes and reaches the screen. See
    # TTAnalyzerWindow.SENSITIVITY_INPUT_DEBOUNCE_MS for the debounce
    # that collapses such a burst into one rerun instead of one per
    # emission.
    sensitivity_inputs_changed = Signal()

    # PARAM_SPECS' per-key decimals (built in __init__, from whichever
    # simulator_spec was passed in) come from core.schema.
    # gui_decimals_from_field, read live off that same simulator's own
    # PhysicalSettings/PhysiologicalSettings Field(ge=..., le=...). No
    # per-key step: NoScrollDoubleSpinBox's stepBy()/wheelEvent()
    # overrides, plus setButtonSymbols(NoButtons) in _build_param_row,
    # already block every interaction Qt would consult singleStep for
    # (spin arrows, wheel, Up/Down keys), so a step value would have no
    # visible effect. min/max are likewise read live -- Auto Fit-eligible
    # rows via calibrator.bounds_from_schema() in _build_param_row below,
    # the same call calibrator.py itself uses for DE bounds; WBAL_ONLY_KEYS
    # rows via core.schema.bounds_from_field directly (see class docstring
    # above) -- either way, reading live off core.schema removes the
    # possibility of drift a hardcoded copy would risk.
    #
    # No label column either -- _build_param_row derives it from
    # core.schema.field_display_label() (the same title/unit metadata
    # eidos.apps.manager's GUI Form Editor reads), rather than a second,
    # independently-drifting hand-authored copy.

    _HEADER_STYLE = "color: #a0a0a0; font-weight: bold; font-size: 11px;"
    _BLOCK_TITLE_STYLE = "color: #808080; font-size: 11px;"

    # A checked row's Value whose number hasn't been through a fit under
    # the CURRENTLY checked free-parameter set -- see _refresh_fit_
    # confirmed_style. Muted text, not a disabled look: the box is still
    # exactly as editable/meaningful as ever (it's the next fit's
    # starting point, and instantly becomes the real fixed value the
    # moment its own checkbox is unchecked) -- this only says "not
    # confirmed by a fit yet", not "off-limits". background-color/border
    # are restated rather than left to the ancestor QSS (see _init_ui's
    # setStyleSheet on QDoubleSpinBox) because a widget-level styleSheet
    # is not guaranteed to inherit properties the ancestor sheet set for
    # the same selector.
    _UNFITTED_VALUE_STYLE = (
        "QDoubleSpinBox { background-color: #2a2a2a; color: #808080; "
        "border: 1px solid #555; font-style: italic; }"
    )

    def __init__(self, simulator_spec: SimulatorSpec, parent=None):
        """
        Args:
            simulator_spec: The resolved SimulatorSpec of the ONE strategy
                this panel's owning TTAnalyzerWindow was opened against --
                fixed for this panel's whole lifetime (see class docstring
                for why no "switch strategies" flow means no runtime
                rebuild is needed, unlike eidos.apps.manager.editor_panes.
                ConfigurationEditorPane's dynamic form). Determines this
                instance's own PARAM_SPECS/WBAL_ONLY_KEYS.
            parent: Passed straight through to QWidget.__init__.
        """
        super().__init__(parent)
        self._simulator_key = simulator_spec.key
        self._physical_param_model = simulator_spec.physical_param_model
        self._physiological_param_model = simulator_spec.physiological_param_model

        # Keys that affect W' Balance only (never enter the velocity/
        # position ODE — see core.physics_overrides.
        # build_overridden_params), so they have no Auto Fit checkbox and
        # are not in core.simulators.calibratable_physical_keys(
        # simulator_spec.key). Bounds for these come straight from this
        # simulator's own PhysiologicalSettings via core.schema.
        # bounds_from_field, the same "read live" mechanism calibrator.
        # bounds_from_schema uses for the calibratable rows. Same source
        # core.calibration_cache.compute_cache_key's own W' Balance
        # exclusion reads off of (core.simulators.physiological_only_
        # keys) -- one derivation, not two independently-drifting copies.
        self.WBAL_ONLY_KEYS = frozenset(physiological_only_keys(simulator_spec.key))
        # calibratable_physical_keys already returns physical_param_model's
        # own field-declaration order -- kept as a list (not just the
        # frozenset above) so PARAM_SPECS below can read rows out in that
        # same order, rather than an incidental one.
        calibratable_keys_ordered = calibratable_physical_keys(simulator_spec.key)

        # Calibratable physical keys first, then WBAL_ONLY_KEYS -- each
        # block in its own model's own field-declaration order. decimals
        # per key comes from core.schema.gui_decimals_from_field, read
        # live off that key's own Field(ge=..., le=...) (or an explicit
        # json_schema_extra={"decimals": ...} override) -- see the
        # PARAM_SPECS comment above for why there's no per-key step.
        self.PARAM_SPECS = [
            (k, gui_decimals_from_field(self._physical_param_model, k))
            for k in calibratable_keys_ordered
        ] + [
            (k, gui_decimals_from_field(self._physiological_param_model, k))
            for k in self._physiological_param_model.model_fields
        ]

        self._spinboxes: dict[str, QDoubleSpinBox] = {}
        self._json_defaults: dict[str, float] = {}
        self._reset_buttons: dict[str, QPushButton] = {}
        self._auto_fit_checkboxes: dict[str, QCheckBox] = {}
        self._sensitivity_widgets: dict[str, SensitivityBarWidget] = {}
        self._decimals: dict[str, int] = {}
        self._value_tooltips: dict[str, str] = {}
        # The exact (free-key set, full spinbox snapshot) a completed fit
        # last covered, or None if nothing's been fit yet -- see
        # mark_fit_confirmed/_current_fit_signature/_refresh_fit_
        # confirmed_style. A row shows confirmed (not muted) iff this
        # signature EQUALS the panel's live signature right now, checked
        # fresh on every sensitivity_inputs_changed emission rather than
        # invalidated by the mere fact that some signal fired -- toggling
        # a checkbox off and back on (or checking, then unchecking, some
        # OTHER row) with nothing else touched is a net no-op on both
        # components of the signature, and must land back on "confirmed",
        # not get stuck muted just because something happened in between.
        self._fit_confirmed_signature: tuple | None = None

        outer = QVBoxLayout(self)
        outer.setSpacing(6)
        outer.setContentsMargins(0, 0, 0, 0)

        # The Auto Fit-eligible params live in their own bordered
        # block with a column header row -- distinguishing them at a
        # glance from the W' Balance-only rows below (which have no
        # Auto Fit/Sensitivity columns at all), and putting "Reset" /
        # "Fit" / "Sensitivity" in the header once each rather than
        # repeated as a label on every row. Each block
        # also gets its own title, named for what it feeds -- "Physics"
        # (the velocity/position ODE) vs. "W' Balance".
        autofit_specs = [s for s in self.PARAM_SPECS if s[0] not in self.WBAL_ONLY_KEYS]
        wbal_specs = [s for s in self.PARAM_SPECS if s[0] in self.WBAL_ONLY_KEYS]

        physics_label = QLabel("Physics")
        physics_label.setStyleSheet(self._BLOCK_TITLE_STYLE)
        outer.addWidget(physics_label)

        autofit_frame = QFrame()
        # Scoped to #autofit_frame specifically -- QLabel is itself a
        # QFrame subclass in Qt, so a bare "QFrame { border: ... }" rule
        # here would also match (and box) every plain QLabel nested
        # inside this frame, not just the frame itself.
        autofit_frame.setObjectName("autofit_frame")
        autofit_frame.setStyleSheet(
            "QFrame#autofit_frame { border: 1px solid #4a4a4a; border-radius: 6px; }"
        )
        autofit_grid = QGridLayout(autofit_frame)
        autofit_grid.setSpacing(4)
        autofit_grid.setContentsMargins(6, 6, 6, 6)
        for col, title in enumerate(["Parameter", "Value", "Reset", "Fit"]):
            header = QLabel(title)
            header.setStyleSheet(self._HEADER_STYLE)
            # Explicit top-align -- this row is as tall as sensitivity_
            # header_box (see below: "Sensitivity" PLUS the controls row
            # TTAnalyzerWindow fills sensitivity_controls_container with,
            # stacked underneath it), and each plain header is only one
            # line tall. Top-aligning them lines their own text up with
            # "Sensitivity" specifically -- the top line of that taller
            # cell -- rather than centering them against the cell's full
            # height, which would sit them below it, roughly level with
            # the controls row instead.
            autofit_grid.addWidget(header, 0, col, Qt.AlignmentFlag.AlignTop)
        # Kept as an instance attr (unlike the other header labels) so
        # show_morris_sensitivity/show_sobol_sensitivity/clear_sensitivity
        # can rewrite it to name whichever statistic pair is currently on
        # screen, color-matched to that pair's own bar colors -- so the
        # header doubles as this column's legend, not just its title.
        sensitivity_header_box = QWidget()
        sensitivity_header_col = QVBoxLayout(sensitivity_header_box)
        sensitivity_header_col.setContentsMargins(0, 0, 0, 0)
        sensitivity_header_col.setSpacing(2)

        sensitivity_title_row = QHBoxLayout()
        sensitivity_title_row.setContentsMargins(0, 0, 0, 0)
        sensitivity_title_row.setSpacing(4)

        self._sensitivity_header_label = QLabel("Sensitivity")
        self._sensitivity_header_label.setStyleSheet(self._HEADER_STYLE)
        self._sensitivity_header_label.setTextFormat(Qt.TextFormat.RichText)
        # AlignTop, not the addWidget default -- the "Check S2" button
        # TTAnalyzerWindow puts in sensitivity_details_button_container
        # (this row's other item) is taller than this label, so without
        # an explicit alignment Qt centers the label within that taller
        # row height instead of pinning it to the row's top, dropping
        # its text below the Parameter/Value/Reset/Fit headers' own
        # (AlignTop'd) baseline in the shared grid row above.
        sensitivity_title_row.addWidget(self._sensitivity_header_label, 0, Qt.AlignmentFlag.AlignTop)
        sensitivity_title_row.addStretch(1)

        # Empty placeholder, to the RIGHT of the "Sensitivity" label
        # itself (this row), not stacked below it -- TTAnalyzerWindow
        # builds its "Check S2" button directly into this container (see
        # TTAnalyzerWindow._build_sensitivity_controls). Its own
        # QHBoxLayout is built here so a caller can just grab .layout()
        # and addWidget -- same "left empty here, so the panel stays
        # unaware of what fills it" reasoning as
        # sensitivity_controls_container below.
        self.sensitivity_details_button_container = QWidget()
        sensitivity_details_button_row = QHBoxLayout(self.sensitivity_details_button_container)
        sensitivity_details_button_row.setContentsMargins(0, 0, 0, 0)
        sensitivity_title_row.addWidget(self.sensitivity_details_button_container)

        sensitivity_header_col.addLayout(sensitivity_title_row)

        # Empty placeholder, stacked directly under the title row above
        # -- TTAnalyzerWindow builds its Sobol'/Morris method radios,
        # N/r spinbox, and discard-status label directly into this
        # container (see TTAnalyzerWindow._build_sensitivity_controls).
        # Left empty here so this panel stays unaware of
        # SensitivityWorker or those controls' own contents.
        self.sensitivity_controls_container = QWidget()
        sensitivity_header_col.addWidget(self.sensitivity_controls_container)

        autofit_grid.addWidget(sensitivity_header_box, 0, 4, Qt.AlignmentFlag.AlignTop)
        for row, spec in enumerate(autofit_specs, start=1):
            self._build_param_row(autofit_grid, row, spec, with_autofit=True)
        # Explicit, not left to QGridLayout's own default heuristic --
        # Parameter/Value/Reset/Auto Fit all stay their natural (tight)
        # width, and any leftover horizontal space this panel is given
        # goes entirely into the Sensitivity column, since a wider bar
        # is strictly more useful (more sub-pixel precision -- see the
        # antialiasing work on SensitivityBarWidget) than wider gaps
        # anywhere else in the row.
        for col in range(4):
            autofit_grid.setColumnStretch(col, 0)
        autofit_grid.setColumnStretch(4, 1)
        outer.addWidget(autofit_frame)

        wbal_title_label = QLabel("W' Balance")
        wbal_title_label.setStyleSheet(self._BLOCK_TITLE_STYLE)
        outer.addWidget(wbal_title_label)

        # Its own bordered block too, same reasoning as autofit_frame --
        # otherwise these rows just float loose below the Physics block
        # with nothing marking them as a group of their own.
        wbal_frame = QFrame()
        wbal_frame.setObjectName("wbal_frame")
        wbal_frame.setStyleSheet(
            "QFrame#wbal_frame { border: 1px solid #4a4a4a; border-radius: 6px; }"
        )
        wbal_grid = QGridLayout(wbal_frame)
        wbal_grid.setSpacing(4)
        wbal_grid.setContentsMargins(6, 6, 6, 6)
        for col, title in enumerate(["Parameter", "Value", "Reset"]):
            header = QLabel(title)
            header.setStyleSheet(self._HEADER_STYLE)
            wbal_grid.addWidget(header, 0, col)
        for row, spec in enumerate(wbal_specs, start=1):
            self._build_param_row(wbal_grid, row, spec, with_autofit=False)
        # Same policy as autofit_grid above: Parameter/Value/Reset stay
        # tight; this block has no Sensitivity-equivalent column worth
        # widening, so leftover space goes into a dedicated empty
        # trailing column instead -- not into Parameter or Value, which
        # would otherwise drift apart from their own Reset button.
        wbal_grid.setColumnStretch(0, 0)
        wbal_grid.setColumnStretch(1, 0)
        wbal_grid.setColumnStretch(2, 0)
        wbal_grid.setColumnStretch(3, 1)

        outer.addWidget(wbal_frame)

        # See _fit_confirmed_signature's own docstring for why this
        # panel re-derives its own confirmed/muted styling from its own
        # signal rather than relying on TTAnalyzerWindow to call
        # something back in.
        self.sensitivity_inputs_changed.connect(self._refresh_fit_confirmed_style)

    def _build_param_row(
        self, layout: QGridLayout, row: int, spec: tuple, with_autofit: bool,
    ) -> None:
        """Build one PARAM_SPECS row (label/spinbox/reset[/auto-fit
        checkbox/sensitivity bar]) into `layout` at `row` -- shared by
        both the Auto Fit-eligible block and the W' Balance-only block
        (see __init__), with_autofit selecting which trailing columns
        this row gets."""
        key, dec = spec
        if key in self.WBAL_ONLY_KEYS:
            model_cls, field_name = self._physiological_param_model, key
            lo, hi = bounds_from_field(model_cls, key)
        else:
            model_cls, field_name = self._physical_param_model, key
            lo, hi = calibrator.bounds_from_schema(self._simulator_key, key)
        label = field_display_label(model_cls, field_name)
        self._decimals[key] = dec
        # Single-line title only, no raw-key second line -- this panel's
        # width is a hard setFixedWidth in TTAnalyzerWindow with no room
        # to spare, and calibration_diagnostics.py's S2/Correlations
        # matrices already use this same display label, so there's
        # nothing to cross-reference a raw key against.
        layout.addWidget(QLabel(label), row, 0)
        sb = NoScrollDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(dec)
        sb.setEnabled(False)      # enabled only when strategy is loaded
        # No spin arrows — with scroll already disabled, the arrows are
        # the only remaining click-and-forget way to nudge a value
        # without noticing. Typing a value, or the Reset button, are
        # the only ways to change one now.
        sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # A value typed outside [lo, hi] isn't clamped to the nearest
        # bound — Qt rejects it and keeps whatever the last valid value
        # was (which, mid-edit while deleting digits, can be a
        # surprising intermediate value, not the original one). Given
        # that, the valid range needs to be visible up front rather
        # than discovered by trial and error.
        self._value_tooltips[key] = f"Valid range: {lo:g} – {hi:g}"
        sb.setToolTip(self._value_tooltips[key])
        sb.valueChanged.connect(lambda _value, k=key: self._update_reset_button(k))
        sb.valueChanged.connect(lambda _value, k=key: self._uncheck_auto_fit(k))
        # sensitivity_inputs_changed wiring for this spinbox lives in the
        # `if with_autofit:` block below, NOT here -- see WBAL_ONLY_KEYS's
        # own comment: those keys never enter the velocity/position ODE,
        # so they can't move the RMSE Auto Fit/Sensitivity are computed
        # against. Firing on their edits would only waste a recompute AND
        # (via _current_fit_signature including them) wrongly mute an
        # otherwise still-valid confirmed fit the next time the signal
        # fires for an unrelated reason.
        layout.addWidget(sb, row, 1)
        self._spinboxes[key] = sb

        # Icon-only -- the column header above already says "Reset"
        # once, so the word doesn't need to repeat on every row.
        btn_reset = QPushButton("↺")
        btn_reset.setFixedWidth(28)
        btn_reset.setStyleSheet("padding: 2px 4px;")
        btn_reset.setToolTip(f"Reset {label} to strategy JSON value")
        btn_reset.setEnabled(False)
        btn_reset.clicked.connect(lambda checked=False, k=key: self._reset_param(k))
        layout.addWidget(btn_reset, row, 2)
        self._reset_buttons[key] = btn_reset

        # No Auto Fit checkbox for W' Balance-only params — see
        # WBAL_ONLY_KEYS: they don't enter the velocity ODE, so DE
        # calibration against velocity RMSE has no signal to fit
        # them against.
        if with_autofit:
            sb.valueChanged.connect(self.sensitivity_inputs_changed)
            cb_auto_fit = QCheckBox()
            cb_auto_fit.setChecked(False)
            cb_auto_fit.setEnabled(False)
            if key == "gravity_accel":
                cb_auto_fit.setToolTip(
                    "Check to enable auto fit\n"
                    "Usually left unchecked — gravity is not normally a free parameter."
                )
            else:
                cb_auto_fit.setToolTip("Check to enable auto fit")
            cb_auto_fit.toggled.connect(self.sensitivity_inputs_changed)
            layout.addWidget(cb_auto_fit, row, 3)
            self._auto_fit_checkboxes[key] = cb_auto_fit

            bar = SensitivityBarWidget()
            bar.clicked.connect(lambda k=key: self.sensitivity_bar_clicked.emit(k))
            bar.released.connect(self.sensitivity_bar_released)
            layout.addWidget(bar, row, 4)
            self._sensitivity_widgets[key] = bar

    def free_keys(self) -> list[str]:
        """
        Keys whose "Auto Fit" checkbox is checked — these are searched
        rather than held fixed at the spinbox value. Two callers: see
        TTAnalyzerWindow._on_add_rebuild (an empty list means "Add
        Rebuild" behaves as a manual override, every row fixed; a
        non-empty list routes the click through AutoFitWorker instead)
        and _run_sensitivity, which screens exactly this same set as its
        free parameters, every other row held fixed at its spinbox value.
        """
        return [key for key, cb in self._auto_fit_checkboxes.items() if cb.isChecked()]

    def mark_fit_confirmed(self, keys) -> None:
        """Record that a fit was just completed (or an existing Rebuild
        carrying a past fit's own values was just loaded) jointly
        covering exactly `keys` -- called by TTAnalyzerWindow right after
        the matching set_values()/set_auto_fit_checks() pair, so the
        panel's live signature (see _current_fit_signature) already
        reflects that fit's own checked set and fixed baseline. A no-op
        call with an empty/None `keys` (a manual, non-Auto-Fit Rebuild)
        stores None -- nothing here was ever fit, so no row can ever
        compare equal to it."""
        self._fit_confirmed_signature = self._current_fit_signature() if keys else None
        self._refresh_fit_confirmed_style()

    def _current_fit_signature(self) -> tuple:
        """(free-key set, Auto Fit-eligible spinbox snapshot) right now
        -- the two things a fit result actually depends on: which keys
        it jointly searched, and what every OTHER Auto Fit-eligible key
        was held fixed at. The W' Balance-only keys are excluded from
        the snapshot -- see WBAL_ONLY_KEYS: they never enter the
        velocity/position ODE, so no value they hold can move the RMSE
        a fit was scored against, and including them here would make an
        edit to one of them silently invalidate an otherwise still-valid
        confirmed fit the next time this signature gets compared (see
        _build_param_row's matching decision to not wire those rows to
        sensitivity_inputs_changed in the first place). Compared for
        exact equality against self._fit_confirmed_signature, not diffed
        piecewise, so a detour that ends up back at the exact same state
        (e.g. toggling one row off then back on, with nothing else
        touched in between) reads as still-confirmed rather than
        muted -- see that attribute's own docstring."""
        overrides = {
            k: v for k, v in self.get_overrides().items() if k not in self.WBAL_ONLY_KEYS
        }
        return (frozenset(self.free_keys()), tuple(sorted(overrides.items())))

    def _refresh_fit_confirmed_style(self) -> None:
        """Mute (see _UNFITTED_VALUE_STYLE) exactly the checked rows
        whose value isn't confirmed by a completed fit right now -- i.e.
        every checked row, whenever the panel's LIVE signature doesn't
        exactly equal self._fit_confirmed_signature (see
        _current_fit_signature). Never a per-key membership test: a
        mismatch anywhere (a different checked set, or any fixed value
        having moved) means NONE of the checked rows are confirmed,
        since Auto Fit searches every checked key jointly against every
        fixed key's current value -- there's no such thing as "half of
        this fit still applies." A row that isn't checked at all is
        always shown plain -- its Value is simply the fixed number the
        user set, no fit concept ever applies to it."""
        confirmed = (
            self._fit_confirmed_signature is not None
            and self._current_fit_signature() == self._fit_confirmed_signature
        )
        for key, cb in self._auto_fit_checkboxes.items():
            sb = self._spinboxes[key]
            if cb.isChecked() and not confirmed:
                sb.setStyleSheet(self._UNFITTED_VALUE_STYLE)
                sb.setToolTip(
                    f"{self._value_tooltips[key]}\n\n"
                    "Not yet fit under the current Fit selection -- "
                    "click Add Rebuild to fit it."
                )
            else:
                sb.setStyleSheet("")
                sb.setToolTip(self._value_tooltips[key])

    def show_morris_sensitivity(self, result: "calibrator.MorrisSensitivityTrials") -> None:
        """
        Render result's mu*/sigma into every row's SensitivityBarWidget,
        replacing whatever that row previously showed (Sobol' or empty).

        mu_star and sigma share ONE vmax across every row and both lanes
        (see SensitivityBarWidget's class docstring for why: they're the
        same physical unit, and this keeps a row's own mu_star-bar-vs-
        sigma-bar length comparison consistent with the classic Morris
        scatter's diagonal, which the same press-either-lane popup also
        opens). Includes mu_star_conf's extent so each row's whisker is
        rarely clipped by the scale itself.
        """
        vmax = max(
            max((result.mu_star[k] + result.mu_star_conf[k] for k in result.mu_star), default=0.0),
            max((result.sigma[k] for k in result.sigma), default=0.0),
        )
        for key, widget in self._sensitivity_widgets.items():
            if key in result.mu_star:
                widget.show_morris(
                    result.mu_star[key], result.mu_star_conf[key], result.sigma[key], vmax,
                )
            else:
                widget.clear()
        self._set_sensitivity_header(
            "μ*", SensitivityBarWidget._MORRIS_MU_COLOR, "σ", SensitivityBarWidget._MORRIS_SIGMA_COLOR,
        )

    def show_sobol_sensitivity(self, result: "calibrator.SobolSensitivityTrials") -> None:
        """Render result's S1/ST (and their bootstrap CIs) into every
        row's SensitivityBarWidget -- see show_morris_sensitivity's
        identical shape/reasoning."""
        for key, widget in self._sensitivity_widgets.items():
            if key in result.s1:
                widget.show_sobol(
                    result.s1[key], result.s1_conf[key], result.st[key], result.st_conf[key],
                )
            else:
                widget.clear()
        self._set_sensitivity_header(
            "S1", SensitivityBarWidget._SOBOL_S1_COLOR,
            "ST", SensitivityBarWidget._SOBOL_ST_COLOR,
        )

    def clear_sensitivity(self) -> None:
        """Blank every row's SensitivityBarWidget -- e.g. a new Activity
        was just selected (see TTAnalyzerWindow._select_candidate), so
        any previously-shown result no longer describes this ride."""
        for widget in self._sensitivity_widgets.values():
            widget.clear()
        self._sensitivity_header_label.setText("Sensitivity")

    def set_sensitivity_busy(self, busy: bool) -> None:
        """Dim every row's SensitivityBarWidget while a Sensitivity run
        is in flight (see TTAnalyzerWindow._run_sensitivity), undim once
        it lands -- a plain opacity drop on the bars themselves, not the
        whole panel (the Value/Reset/Fit columns next to them stay fully
        interactive and legible throughout), so it reads as "this
        specific result is stale/being replaced" rather than "the panel
        is locked." A fresh QGraphicsOpacityEffect per call rather than
        toggling one shared instance's opacity -- Qt effects are tied to
        a single widget each; sharing one across every bar isn't an
        option here."""
        for widget in self._sensitivity_widgets.values():
            if busy:
                effect = QGraphicsOpacityEffect(widget)
                effect.setOpacity(0.35)
                widget.setGraphicsEffect(effect)
            else:
                widget.setGraphicsEffect(None)

    def _set_sensitivity_header(
        self, top_name: str, top_color: QColor, bottom_name: str, bottom_color: QColor,
    ) -> None:
        """Rewrite the "Sensitivity" column header to name+color-match
        whichever statistic pair show_morris_sensitivity/
        show_sobol_sensitivity just drew -- so the header also serves as
        this column's legend (top lane's name/color, then bottom lane's),
        not just a static title."""
        self._sensitivity_header_label.setText(
            f'Sensitivity (<span style="color:{top_color.name()}">{top_name}</span> / '
            f'<span style="color:{bottom_color.name()}">{bottom_name}</span>)'
        )

    def setAutoFitEnabled(self, enabled: bool):
        """Enable/disable every Auto Fit checkbox at once (e.g. while a
        calibration run is in progress)."""
        for cb in self._auto_fit_checkboxes.values():
            cb.setEnabled(enabled)

    def populate(self, raw_physical: dict, raw_physiological: dict):
        """Fill spinboxes with values from the loaded strategy JSON."""
        combined = {**raw_physical, **raw_physiological}
        self._json_defaults = {
            key: float(combined[key]) for key in self._spinboxes if key in combined
        }
        for key, sb in self._spinboxes.items():
            if key in self._json_defaults:
                sb.setValue(self._json_defaults[key])
            sb.setEnabled(True)
        for key, cb in self._auto_fit_checkboxes.items():
            cb.setEnabled(key in self._json_defaults)
        # Explicit pass rather than relying solely on valueChanged: setValue
        # above is a no-op (no signal fired) for any spinbox already sitting
        # at its default, which is normally every one of them right after a
        # fresh populate() — Reset needs to end up disabled in that case,
        # not keep whatever pre-populate() state it had.
        for key in self._reset_buttons:
            self._update_reset_button(key)

    def get_overrides(self) -> dict:
        """
        Return the current value of every spinbox, keyed by parameter name.

        Includes parameters left unchanged from the strategy default, not
        only ones the user modified — harmless downstream, since callers
        merge this into the strategy's raw_physical/raw_physiological dict,
        so an unchanged value simply overwrites itself.
        """
        return {key: sb.value() for key, sb in self._spinboxes.items()}

    def _update_reset_button(self, key: str):
        """
        Enable key's Reset button only if its spinbox's current value
        differs from the strategy JSON's value. Connected to every
        spinbox's valueChanged and also called explicitly from
        populate()/set_values()/_reset_param(), since setValue() doesn't
        emit valueChanged when the value already matches the current one.

        Args:
            key: A PARAM_SPECS key.
        """
        if key not in self._reset_buttons:
            return
        if key not in self._json_defaults:
            self._reset_buttons[key].setEnabled(False)
            return
        current = self._spinboxes[key].value()
        default = self._json_defaults[key]
        # Half a display unit of tolerance (e.g. dec=3 -> 0.0005) so a
        # value that only differs from the default in a digit the spinbox
        # doesn't even show isn't treated as "changed".
        tol = 0.5 * 10 ** (-self._decimals[key])
        self._reset_buttons[key].setEnabled(abs(current - default) > tol)

    def _uncheck_auto_fit(self, key: str):
        """Uncheck key's "Auto Fit" checkbox, if it has one -- connected
        to every spinbox's valueChanged, same as _update_reset_button.
        Typing a Value by hand (or pressing Reset, which also goes
        through setValue()) is a manual override; leaving Fit checked
        would claim this row is still free for the next Auto Fit search
        to move, when the number actually on screen is now the user's
        own choice. No-op for W' Balance-only keys, which have no Fit
        checkbox at all (see WBAL_ONLY_KEYS).

        setChecked(False) on a row that was actually checked fires that
        checkbox's own toggled -- which sensitivity_inputs_changed is
        connected to directly (see _build_param_row) -- so this needs no
        separate emit of its own; Qt already no-ops setChecked(False) on
        a box that's already unchecked, so nothing fires THAT signal for
        a row whose Fit box was never checked to begin with -- but the
        valueChanged that got this method called in the first place is
        itself also wired straight to sensitivity_inputs_changed (see
        _build_param_row), so a fixed row's edit still reaches it, just
        via that connection instead of this one."""
        cb = self._auto_fit_checkboxes.get(key)
        if cb is not None:
            cb.setChecked(False)

    def _reset_param(self, key: str):
        """Reset a single spinbox to its strategy JSON value."""
        if key in self._json_defaults:
            self._spinboxes[key].setValue(self._json_defaults[key])
            self._update_reset_button(key)

    def set_values(self, overrides: dict):
        """Load a saved Rebuild's physics_overrides into the spinboxes.

        Used when the user clicks an existing Rebuild in the list, to
        inspect/reuse its parameters as the starting point for a new
        one. The clicked Rebuild itself is never modified — this only
        pre-fills the panel; "Add Rebuild" always creates a new entry.
        Silently skips any key not present as a spinbox, or not in
        overrides.
        """
        for key, sb in self._spinboxes.items():
            if key in overrides:
                sb.setValue(float(overrides[key]))
        for key in self._reset_buttons:
            self._update_reset_button(key)

    def set_auto_fit_checks(self, keys):
        """Check exactly the given keys' "Auto Fit" checkboxes, uncheck the rest.

        Companion to set_values() -- called alongside it when the user clicks
        an existing Rebuild in the list (see
        TTAnalyzerWindow._on_rebuild_item_clicked), so that Rebuild's own
        Auto Fit checkbox state (Scenario.auto_fit_keys) is restored too, not
        just its spinbox values. `keys` is typically empty (a plain manual
        Rebuild) or the set of keys an Auto Fit run was calibrated against.

        Every row whose checked state actually flips here fires
        sensitivity_inputs_changed (see that signal's own docstring for
        why firing once per flipped row, mid-restore, is fine)."""
        keys = set(keys)
        for key, cb in self._auto_fit_checkboxes.items():
            cb.setChecked(key in keys)

    def format_overrides_tooltip(self, overrides: dict) -> str:
        """Format a Rebuild's saved overrides for a list-item tooltip."""
        lines = []
        for key, dec in self.PARAM_SPECS:
            if key in overrides:
                if key in self.WBAL_ONLY_KEYS:
                    model_cls, field_name = self._physiological_param_model, key
                else:
                    model_cls, field_name = self._physical_param_model, key
                label = field_display_label(model_cls, field_name)
                lines.append(f"{label}: {overrides[key]:.{dec}f}")
        return "\n".join(lines)

    def reset_to_strategy(self, raw_physical: dict, raw_physiological: dict):
        """Reset all spinboxes to the strategy's original values."""
        self.populate(raw_physical, raw_physiological)


class RebuildColorDelegate(QStyledItemDelegate):
    """List item delegate that keeps a Rebuild's own color visible when selected.

    Plain QSS isn't enough on its own — Qt's item views substitute the
    palette's HighlightedText role (white, by default) for selected-row
    text regardless of what QListWidgetItem.setForeground() set, unless
    a stylesheet rule *also* pins `color` for that state. Since each
    Rebuild's color is assigned dynamically per row, it can't be
    hardcoded into a static QSS rule, so this overrides the palette's
    HighlightedText color per-item, right before painting.
    """

    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if brush is not None:
            option.palette.setColor(QPalette.HighlightedText, brush.color())
        super().paint(painter, option, index)
