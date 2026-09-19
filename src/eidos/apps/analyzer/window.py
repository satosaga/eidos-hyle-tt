"""
eidos.apps.analyzer.window -- TTAnalyzerWindow, the Analyzer's main window.

Kept as one class rather than split further: QMainWindow construction,
panel building, and event handlers are all tightly coupled to shared
instance state, so splitting them apart would obscure the flow rather
than clarify it.
"""

import concurrent.futures
import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import Literal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

import core.calibrator as calibrator
from core.activity_parser import (
    ALTITUDE_LAG_MAX_S,
    ActivityCandidate,
    ActivityRecord,
    find_course_matches,
    parse_fit_file,
)
from core.data_manager import format_time_mmss
from core.io_config import BASE_ACTIVITIES_DIR, BASE_CONFIGS_DIR
from core.schema import RunValidationModel
from eidos.apps.analyzer.canvas import SCENARIO_COLORS, AnalysisCanvas
from eidos.apps.analyzer.dialogs import (
    CalibrationDiagnosticsDialog,
    SobolS2DetailsDialog,
)
from eidos.apps.analyzer.minimap import CourseMinimapWidget
from eidos.apps.analyzer.models import Scenario, SimTrace, StrategyRecord
from eidos.apps.analyzer.widgets import (
    NoScrollSpinBox,
    PhysicsOverridePanel,
    RebuildColorDelegate,
)
from eidos.apps.analyzer.workers import (
    AutoFitWorker,
    SensitivityWorker,
    SimulationWorker,
)
from eidos.lib import calibration_diagnostics as diag
from eidos.lib.branding import window_title

logger = logging.getLogger(__name__)

# Where _on_generate_config writes new configs -- resources/configs/templates/,
# not BASE_CONFIGS_DIR itself, so a Rebuild-generated config shows up as a
# selectable base in eidos.apps.manager's GenerationManagerPane.button_new
# (scoped to TEMPLATES_DIR) instead of sitting one level up, invisible to
# that picker.
_CONFIGS_TEMPLATES_DIR = os.path.join(BASE_CONFIGS_DIR, "templates")


def _sensitivity_discard_status(
    n_discarded: int, n_requested: int, singular: str, plural: str,
    implicated_counts: dict[str, int],
    n_unattributable: int, unattributable_label: str,
) -> str:
    """Return "" if nothing was discarded; otherwise a note of how many
    extra Morris trajectories/Sobol' replicates were drawn and discarded
    for hitting infeasible physics before reaching n_requested, naming
    which free_key(s) were implicated (see calibrator.sample_morris_
    sensitivity/sample_sobol_sensitivity for what "implicated" means per
    method; the final mu*/sigma/S1/ST always use the full n_requested
    regardless -- this is purely informational).

    n_unattributable/unattributable_label cover a discard with no single
    free_key to blame (MorrisSensitivityTrials.n_baseline_infeasible /
    SobolSensitivityTrials.n_base_point_infeasible).

    singular/plural are passed explicitly rather than derived by
    appending "s" -- English pluralization isn't that regular.
    """
    if n_discarded == 0:
        return ""
    unit = singular if n_discarded == 1 else plural
    parts = [f"{k}×{v}" for k, v in implicated_counts.items() if v]
    if n_unattributable:
        parts.append(f"{n_unattributable} {unattributable_label}")
    detail = f": {', '.join(parts)}" if parts else ""
    return (
        f"{n_discarded} {unit} discarded and replaced (hit infeasible "
        f"physics{detail}) -- still used the full {n_requested} requested."
    )


def _fmt_device_local_clock(start_time_utc, utc_offset_s: float | None) -> str:
    """Format a FIT start_time at the recording device's own local time.

    Uses utc_offset_s (from the FIT Activity message's local_timestamp,
    see core.activity_parser.parse_fit_file) rather than the analyzing
    machine's system timezone — this is the wall-clock time at the place
    the ride happened, which is what matters when a rider is picking
    between course-match candidates. Falls back to plain UTC, labelled as
    such, if the FIT has no Activity message or lacks local_timestamp.
    """
    if utc_offset_s is None:
        return start_time_utc.strftime("%H:%M:%S UTC")
    local_dt = start_time_utc + timedelta(seconds=utc_offset_s)
    return local_dt.strftime("%H:%M:%S")


def _next_config_filename(base_name: str) -> str:
    """Return the next unused "{base_name}_NN.json" filename in
    _CONFIGS_TEMPLATES_DIR.

    Mirrors eidos.apps.manager's ConfigFileManager._generate_new_filename's
    own sequential-numbering scheme (not imported directly — that class
    pulls in the rest of eidos.apps.manager's GUI, which this module has
    no other reason to depend on) so a Rebuild-generated config sits alongside
    manually-created templates (e.g. sample_config.json) without colliding.
    """
    pattern = re.compile(r"^" + re.escape(base_name) + r"_(\d+)\.json$")
    max_num = 0
    if os.path.isdir(_CONFIGS_TEMPLATES_DIR):
        for fname in os.listdir(_CONFIGS_TEMPLATES_DIR):
            m = pattern.match(fname)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return f"{base_name}_{max_num + 1:02d}.json"


# ---------------------------------------------------------------------------
# VIII. Main window
# ---------------------------------------------------------------------------

class TTAnalyzerWindow(QMainWindow):
    """
    Main window for the EIDOS^TT Analyzer.

    The strategy record is fixed for the lifetime of the process: it is
    resolved from CLI arguments (same 4-positional format as
    eidos.apps.trainer — see _parse_cli_args()) and loaded once in main()
    before the window is constructed. There is no in-app strategy switcher;
    re-running against a different strategy means relaunching the process.

    Layout:

    - Left -- AnalysisCanvas (five-panel plot)
    - Right -- Control panel (four panels: Strategy / Activity / Rebuilds / Course Map)

      - Strategy (read-only: which strategy is loaded)
      - Activity (Browse FIT + matched-candidate selection)
      - Rebuilds (form: Label/Parameters/Add Rebuild, then the list +
        Remove Rebuild; Strategy isn't in this list -- see
        _run_strategy_scenario)
      - Course Map (see _build_minimap_panel)
    """

    # Wide enough for the Sensitivity header row (label + Check S2
    # button) and the Sobol'/Morris controls row (radios + N/r
    # spinboxes) to show their full text without clipping -- see
    # _init_ui's right_widget/right_scroll setFixedWidth calls.
    DEFAULT_WIDTH = 1560
    DEFAULT_HEIGHT = 900

    # The Matched Activity combo's own placeholder text for every
    # no-match state (initial/no valid records/no course match alike --
    # see _set_no_candidates) -- one constant so all three stay
    # textually identical by construction, never drift apart.
    NO_ACTIVITY_MATCHED = "(no activity matched)"

    # How long any Sensitivity input (a Fit checkbox, or a spinbox value)
    # must sit still before _on_sensitivity_inputs_changed actually
    # launches a rerun. PhysicsOverridePanel.set_auto_fit_checks()/
    # set_values() (restoring a clicked Rebuild's state) flip/set several
    # rows synchronously, each its own signal emission, and typing a
    # multi-digit number fires valueChanged once per keystroke; without
    # debouncing, each emission would launch (and immediately supersede)
    # its own SensitivityWorker instead of settling on one run for the
    # final state. Short enough that an isolated checkbox click still
    # feels immediate.
    SENSITIVITY_INPUT_DEBOUNCE_MS = 200

    def __init__(self, strategy: StrategyRecord, parent=None):
        super().__init__(parent)
        self.setWindowTitle(window_title("Analyzer"))
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

        self._strategy: StrategyRecord = strategy
        self._activity_raw: ActivityRecord | None = None
        self._candidates: list[ActivityCandidate] = []
        self._auto_fit_worker: AutoFitWorker | None = None
        self._auto_fit_start_time: float = 0.0
        self._strategy_scenario: Scenario | None = None
        self._rebuilds: list[Scenario] = []
        # Monotonic within one Activity's Rebuilds, never reused (not
        # len(self._rebuilds)+1) — otherwise removing #2 from {#1,#2,#3}
        # makes the next Add compute "3" again, colliding with the #3
        # still on screen (label AND color, both indexed off this).
        # Reset to 1 on Activity change instead, since that clears the
        # whole list — nothing remains on screen to collide with.
        self._next_rebuild_num: int = 1
        self._last_selected_row: int = -1
        self._workers: list[SimulationWorker] = []
        # Manual Altitude-panel Activity-line shift — see
        # _build_activity_panel's spinbox and AnalysisCanvas.update_plots'
        # activity_altitude_lag_s docstring.
        self._altitude_lag_s: float = 0.0
        self._sensitivity_worker: SensitivityWorker | None = None
        self._morris_result: "calibrator.MorrisSensitivityTrials | None" = None
        self._sobol_result: "calibrator.SobolSensitivityTrials | None" = None
        self._scatter_popup: QWidget | None = None
        # One pool for this window's whole lifetime, reused across every
        # SensitivityWorker run -- see SensitivityWorker's own `pool`
        # docstring. Created eagerly (not lazily on first use): Process
        # PoolExecutor doesn't actually spawn its worker processes until
        # the first submit(), so this costs nothing up front.
        self._sensitivity_pool = concurrent.futures.ProcessPoolExecutor()
        # See SENSITIVITY_INPUT_DEBOUNCE_MS.
        self._sensitivity_input_debounce = QTimer(self)
        self._sensitivity_input_debounce.setSingleShot(True)
        self._sensitivity_input_debounce.timeout.connect(self._on_sensitivity_inputs_settled)

        self._init_ui()
        self._apply_strategy()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        """Build and wire all widgets."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #cccccc; }
            QGroupBox {
                font-weight: bold; border: 1px solid #555; margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; font-size: 12px; }
            QPushButton {
                background-color: #3a3a3a; color: #cccccc;
                border: 1px solid #555; border-radius: 4px; padding: 4px 10px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:disabled { color: #666; }
            QComboBox, QDoubleSpinBox, QListWidget {
                background-color: #2a2a2a; color: #cccccc; border: 1px solid #555;
            }
            QListWidget::item:selected {
                background-color: #444444;
            }
            QLabel { color: #cccccc; }

            /* Distance slider (below the canvas) */
            QSlider::groove:horizontal {
                border: 1px solid #555; height: 6px; background: #333;
                margin: 2px 0; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00CC00; border: 1px solid #5c5c5c;
                width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
            }
            QSlider::handle:horizontal:hover { background: #00AA00; }
            QSlider:disabled::handle:horizontal { background: #555555; }
        """)

        # Central splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left: plot canvas + distance slider
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(2)

        self._canvas = AnalysisCanvas()
        canvas_layout.addWidget(self._canvas, stretch=1)

        # 0-1000 ratio, directly proportional to pct (see
        # AnalysisCanvas/_on_slider_moved) — every series spans exactly
        # [0, 1] of pct by construction, so this needs no further scaling
        # against anything that changes as Activity/Rebuilds change.
        self._dist_slider = QSlider(Qt.Horizontal)
        self._dist_slider.setRange(0, 1000)
        self._dist_slider.setEnabled(False)
        self._dist_slider.valueChanged.connect(self._on_slider_moved)

        # Inset from the window edges: at value=0 the handle would
        # otherwise sit flush against the window's left/bottom edge,
        # making it very easy to grab the OS window-resize corner instead
        # of the handle.
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(15, 0, 15, 10)
        slider_row.addWidget(self._dist_slider)
        canvas_layout.addLayout(slider_row)

        splitter.addWidget(canvas_container)

        # Right: control panel, scrollable — today's panels already run
        # close to DEFAULT_HEIGHT, and the Course Map panel below pushes
        # total content past it at the default window size.
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(6, 6, 6, 6)
        # 500px so the Sensitivity header/controls rows fit -- see
        # DEFAULT_WIDTH's own comment.
        right_widget.setFixedWidth(500)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(520)
        right_scroll.setFrameShape(QScrollArea.NoFrame)

        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # — Strategy selection —
        right_layout.addWidget(self._build_strategy_panel())

        # — Activity selection —
        right_layout.addWidget(self._build_activity_panel())

        # — Rebuilds (label + physics overrides + Add, then the list + Remove) —
        right_layout.addWidget(self._build_rebuilds_panel())

        # — Course minimap —
        right_layout.addWidget(self._build_minimap_panel())

        right_layout.addStretch()

    def _build_strategy_panel(self) -> QGroupBox:
        box = QGroupBox("Strategy")
        layout = QVBoxLayout(box)

        self._strategy_label = QLabel("(loading…)")
        self._strategy_label.setWordWrap(True)
        layout.addWidget(self._strategy_label)
        return box

    def _build_activity_panel(self) -> QGroupBox:
        box = QGroupBox("Activity")
        layout = QVBoxLayout(box)

        btn_browse = QPushButton("Browse FIT…")
        btn_browse.clicked.connect(self._on_browse_fit)
        layout.addWidget(btn_browse)

        self._activity_file_label = QLabel("(no FIT loaded)")
        self._activity_file_label.setWordWrap(True)
        layout.addWidget(self._activity_file_label)

        layout.addWidget(QLabel("Matched Activity:"))

        self._candidate_combo = QComboBox()
        # No separate status label below this combo -- whatever it shows
        # (real candidates, or this placeholder) IS the status. Diagnostic
        # detail for why nothing matched goes to a QMessageBox instead
        # (see _set_no_candidates), transient rather than permanent layout.
        self._candidate_combo.addItem(self.NO_ACTIVITY_MATCHED)
        self._candidate_combo.setEnabled(False)
        self._candidate_combo.currentIndexChanged.connect(self._on_candidate_selected)
        layout.addWidget(self._candidate_combo)

        # Manual nudge for the Altitude panel's Activity line only — see
        # AnalysisCanvas.update_plots' activity_altitude_lag_s docstring
        # for why this is a plain user-supplied number rather than an
        # auto-estimated correction: the project's one authoritative
        # delay-ESTIMATION implementation lives in
        # hyle.apps.fit2gpx_converter and is deliberately not duplicated
        # here. Default 0.0 (no shift), not a guessed value, since the
        # real lag is device/course-dependent. Range is [0,
        # ALTITUDE_LAG_MAX_S], not symmetric: barometric altitude only
        # ever LAGS true position (device architecture rules out the
        # reverse), so a negative shift has no physical meaning here.
        lag_row = QHBoxLayout()
        lag_row.addWidget(QLabel("Altitude lag [s]:"))
        self._altitude_lag_spin = QDoubleSpinBox()
        self._altitude_lag_spin.setRange(0.0, ALTITUDE_LAG_MAX_S)
        self._altitude_lag_spin.setSingleStep(0.5)
        self._altitude_lag_spin.setValue(0.0)
        self._altitude_lag_spin.setToolTip(
            "Manual shift for the Activity altitude line only (raw FIT\n"
            "barometric reading, uncorrected) — the barometer's own\n"
            "reading characteristically lags several seconds behind\n"
            "actual position. Not auto-estimated; adjust by eye against\n"
            "the course line above."
        )
        self._altitude_lag_spin.valueChanged.connect(self._on_altitude_lag_changed)
        lag_row.addWidget(self._altitude_lag_spin)
        layout.addLayout(lag_row)

        return box

    def _build_rebuilds_panel(self) -> QGroupBox:
        """Build the single "Rebuilds" panel: input form, then the list.

        One box rather than two — "make a new Rebuild" is fundamentally
        "add one to the Rebuilds set", so the input form and the list it
        feeds live together. Order is deliberately input-first (Label /
        Parameters / Add Rebuild, then the list / Remove Rebuild) so a
        freshly-added row appears right below the form that made it.
        """
        box = QGroupBox("Rebuilds")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # Power source is always "actual" for user-created Rebuilds — see
        # Scenario's docstring: overriding physics params on Planned power
        # replays a DE-optimized strategy under conditions it was never
        # optimized for, which isn't a meaningful comparison. Only the
        # auto-added "Strategy" scenario (not a Rebuild — see
        # _run_strategy_scenario) uses Planned, as the Δt panel's
        # reference line.

        # --- Input form: Label / Parameters / Add Rebuild ---
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self._rebuild_label_edit = QLineEdit()
        self._rebuild_label_edit.setText(self._auto_label())
        self._rebuild_label_edit.setStyleSheet(
            "background-color: #2a2a2a; color: #cccccc; border: 1px solid #555;"
        )
        label_row.addWidget(self._rebuild_label_edit)
        layout.addLayout(label_row)

        self._physics_panel = PhysicsOverridePanel(self._strategy.simulator_spec)
        layout.addWidget(self._physics_panel)

        # Sensitivity screening runs against whichever parameters
        # currently have their Auto Fit checkbox checked (PhysicsOverride
        # Panel.free_keys(), same set _on_add_rebuild uses) -- every other
        # row stays fixed at its spinbox value. Results render directly
        # into those same checkboxes' rows via PhysicsOverridePanel.
        # show_morris_sensitivity/show_sobol_sensitivity -- no separate
        # "apply" step.
        #
        # Any checkbox/spinbox change (direct click, typed value, Reset,
        # or a bulk restore from clicking an existing Rebuild) routes
        # through PhysicsOverridePanel.sensitivity_inputs_changed and is
        # debounced (SENSITIVITY_INPUT_DEBOUNCE_MS) so a burst settles
        # into one run; Enter in the N/r spinbox or a method switch
        # (_on_sensitivity_method_changed) recomputes immediately, since
        # those are already single, deliberate actions. All paths funnel
        # through _run_sensitivity, which cancels any run already in
        # flight first. Zero boxes checked logs that instead of launching
        # a worker. Selecting a new Activity only clears the stale result
        # (_select_candidate) without triggering a recompute.
        #
        # Its method/N-or-r controls live inline, always visible, built
        # directly into self._physics_panel's "Sensitivity" column header
        # (see _build_sensitivity_controls) -- no separate popup to open.
        self._build_sensitivity_controls()
        self._physics_panel.sensitivity_bar_clicked.connect(self._on_sensitivity_bar_clicked)
        self._physics_panel.sensitivity_bar_released.connect(self._hide_scatter_popup)
        self._physics_panel.sensitivity_inputs_changed.connect(self._on_sensitivity_inputs_changed)

        # A single button covers both cases: if any row's "Auto Fit"
        # checkbox is checked, click routes through AutoFitWorker with
        # those keys free and every other row fixed at its spinbox value;
        # if none are checked, it's exactly a manual Add Rebuild (every
        # row fixed). See PhysicsOverridePanel.free_keys/_on_add_rebuild.
        add_row = QHBoxLayout()
        btn_add = QPushButton("Add Rebuild")
        btn_add.setEnabled(False)
        btn_add.clicked.connect(self._on_add_rebuild)
        self._btn_run = btn_add
        add_row.addWidget(btn_add)

        btn_auto_fit_cancel = QPushButton("Cancel")
        btn_auto_fit_cancel.setVisible(False)
        btn_auto_fit_cancel.clicked.connect(self._on_auto_fit_cancel_clicked)
        self._btn_auto_fit_cancel = btn_auto_fit_cancel
        add_row.addWidget(btn_auto_fit_cancel)
        layout.addLayout(add_row)

        self._auto_fit_progress_label = QLabel("")
        self._auto_fit_progress_label.setStyleSheet("color: #999999;")
        layout.addWidget(self._auto_fit_progress_label)

        # --- List: what already exists, then Remove Rebuild ---
        self._rebuild_list = QListWidget()
        # Up/Down arrow-key navigation: handled in eventFilter() below,
        # deliberately NOT via QListWidget's native currentItemChanged --
        # that signal ordering conflicts with itemClicked's toggle-off
        # logic (see _on_rebuild_item_clicked's docstring). The event-
        # filter path never touches currentItemChanged, so it can't
        # reintroduce that conflict.
        self._rebuild_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._rebuild_list.installEventFilter(self)
        self._rebuild_list.setFixedHeight(120)
        self._rebuild_list.setItemDelegate(RebuildColorDelegate(self._rebuild_list))
        self._rebuild_list.itemClicked.connect(self._on_rebuild_item_clicked)
        layout.addWidget(self._rebuild_list)

        # Remove Rebuild / Diagnose Fit / Generate Config all act on
        # whichever Rebuild is highlighted -- all three start disabled and
        # are kept in sync by _refresh_selection_buttons, called everywhere
        # self._last_selected_row changes (Rebuild click, arrow-key nav,
        # Remove Rebuild, a new Rebuild's auto-select, Activity/candidate
        # switches that clear the list).
        list_btn_row = QHBoxLayout()
        btn_remove = QPushButton("Remove Rebuild")
        btn_remove.setEnabled(False)
        btn_remove.clicked.connect(self._on_remove_rebuild)
        self._btn_remove_rebuild = btn_remove
        list_btn_row.addWidget(btn_remove)

        # Diagnose Fit has the extra condition that the highlighted
        # Rebuild must also carry its own calibration_result (i.e. it was
        # produced by Auto Fit, not added manually).
        btn_diagnostics = QPushButton("Diagnose Fit")
        btn_diagnostics.setEnabled(False)
        btn_diagnostics.clicked.connect(self._on_show_diagnostics)
        self._btn_diagnostics = btn_diagnostics
        list_btn_row.addWidget(btn_diagnostics)

        btn_gen_config = QPushButton("Generate Config")
        btn_gen_config.setEnabled(False)
        btn_gen_config.setToolTip(
            "Write the selected Rebuild's rider/environment parameters "
            "as a new configs/*.json"
        )
        btn_gen_config.clicked.connect(self._on_generate_config)
        self._btn_gen_config = btn_gen_config
        list_btn_row.addWidget(btn_gen_config)
        layout.addLayout(list_btn_row)

        return box

    def _build_minimap_panel(self) -> QGroupBox:
        box = QGroupBox("Course Map")
        layout = QVBoxLayout(box)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._minimap = CourseMinimapWidget()
        layout.addWidget(self._minimap)

        # See CourseMinimapWidget.set_follow_mode: at the whole-course
        # overview scale, a real Activity-vs-course gap works out to
        # fewer pixels than the cursor dot markers themselves, so it's
        # invisible in practice even though correctly computed. This
        # checkbox switches to a zoomed, cursor-centred local view.
        self._follow_checkbox = QCheckBox("Follow cursor (zoom)")
        self._follow_checkbox.toggled.connect(self._minimap.set_follow_mode)
        layout.addWidget(self._follow_checkbox)

        return box

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _apply_strategy(self):
        """Populate the UI from the strategy loaded at startup (see main())."""
        self._strategy_label.setText(os.path.basename(self._strategy.record_dir))
        self._minimap.set_course(
            self._strategy.course_profile.lat_fine,
            self._strategy.course_profile.lon_fine,
            self._strategy.course_s_p,
        )
        self._physics_panel.populate(self._strategy.raw_physical, self._strategy.raw_physiological)
        self._btn_run.setEnabled(True)
        self._sensitivity_controls_ready = True
        self._refresh_sensitivity_controls_enabled()

        # Auto-run the Strategy scenario. It isn't a Rebuild: it doesn't
        # appear in the Rebuilds list and can't be removed from there —
        # see _run_strategy_scenario().
        self._run_strategy_scenario()

    @Slot()
    def _set_no_candidates(self, message: str) -> None:
        """Common cleanup for every "no match" FIT-load outcome (parse
        produced no valid records, or find_course_matches found none) --
        clears any prior successful match's state (candidates/
        activity_raw/minimap track) and resets the combo to a single
        disabled self.NO_ACTIVITY_MATCHED placeholder item, so no prior
        match's combo entries/graphs are left stuck on screen. `message`
        goes to a QMessageBox rather than a permanent label, so it leaves
        no trace in the layout once dismissed."""
        QMessageBox.warning(self, "No Activity Match", message)
        self._candidates = []
        self._activity_raw = None
        self._candidate_combo.blockSignals(True)
        self._candidate_combo.clear()
        self._candidate_combo.addItem(self.NO_ACTIVITY_MATCHED)
        self._candidate_combo.setEnabled(False)
        self._candidate_combo.blockSignals(False)
        self._minimap.set_activity_track(None, None, None)

    def _on_browse_fit(self):
        """Open a file dialog to select an activity FIT.

        Defaults to BASE_ACTIVITIES_DIR — eidos.apps.trainer writes
        recorded FIT files there — but any path can be chosen.
        """
        start_dir = BASE_ACTIVITIES_DIR if os.path.isdir(BASE_ACTIVITIES_DIR) else "."
        fit_path, _ = QFileDialog.getOpenFileName(
            self, "Select Activity FIT", start_dir, "FIT files (*.fit)"
        )
        if fit_path:
            self._load_fit(fit_path)

    def _load_fit(self, fit_path: str):
        """Parse a FIT file, run course matching, and select the best candidate."""
        try:
            fit_data = parse_fit_file(fit_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "FIT Parse Error", str(exc))
            return

        self._activity_file_label.setText(os.path.basename(fit_path))

        if fit_data is None:
            self._set_no_candidates("No valid records found in FIT.")
            return

        self._candidates = find_course_matches(
            fit_data,
            self._strategy.course_distance_m,
            self._strategy.course_latlons,
        )

        if not self._candidates:
            self._set_no_candidates(
                "No course match found in FIT.\n"
                "Check that the FIT contains the target course."
            )
            return

        # Each combo entry carries its own distance/time/start/match
        # fields directly in its text -- no separate status label, so
        # entries must be self-explanatory on their own: labeled, not
        # just bare numbers in a row.
        self._candidate_combo.blockSignals(True)
        self._candidate_combo.clear()
        for i, cand in enumerate(self._candidates):
            label = (
                f"#{i+1}  {cand.record.total_distance_m/1000:.2f} km"
                f"  {format_time_mmss(cand.record.elapsed_time_s)}"
                f"  start {_fmt_device_local_clock(cand.record.start_time, cand.record.utc_offset_s)}"
                f"  match {cand.combined_score:.2f}"
            )
            self._candidate_combo.addItem(label, userData=i)
        self._candidate_combo.setEnabled(True)
        self._candidate_combo.blockSignals(False)

        # Select the top candidate
        self._select_candidate(0)

    @Slot(float)
    def _on_altitude_lag_changed(self, value: float):
        """Reposition the Activity-altitude line for the new manual shift.

        Deliberately calls the cheap AnalysisCanvas.set_activity_altitude_lag
        rather than _refresh_canvas() — this knob only concerns one
        dashed line and shouldn't trigger a full 5-panel redraw, which
        would also visibly drag every panel's shared x-axis as the value
        changes. self._altitude_lag_s is still updated so any future
        full redraw (e.g. a new Rebuild) keeps using the current lag.
        """
        self._altitude_lag_s = value
        self._canvas.set_activity_altitude_lag(value)

    @Slot(int)
    def _on_candidate_selected(self, index: int):
        """Switch the active activity to the selected candidate."""
        if index < 0 or index >= len(self._candidates):
            return
        self._select_candidate(index)

    def _select_candidate(self, index: int):
        """Set the activity to the candidate at the given index.

        Clears all Rebuilds first: a Rebuild's trace is a re-simulation of
        the PREVIOUS activity_raw's power (see run_scenario), so it would
        otherwise sit on screen mismatched against whichever Activity
        overlay is now selected. Strategy is unaffected — it doesn't
        depend on activity_raw. Label and Parameters in the Rebuilds form
        also reset to their defaults, so a value left over from typing or
        from inspecting a clicked Rebuild doesn't silently carry over.
        """
        if index < 0 or index >= len(self._candidates):
            return
        cand = self._candidates[index]

        self._rebuilds.clear()
        self._rebuild_list.clear()
        self._last_selected_row = -1
        self._next_rebuild_num = 1
        self._rebuild_label_edit.setText(self._auto_label())
        self._physics_panel.reset_to_strategy(self._strategy.raw_physical, self._strategy.raw_physiological)
        self._refresh_selection_buttons()
        # A new FIT has no continuity with wherever the cursor sat on the
        # previous Activity — start over at the left edge.
        self._dist_slider.setValue(0)
        # A barometer lag tuned by eye for the PREVIOUS Activity has no
        # reason to still be right for this one (different device/ride).
        # setValue(0.0) is a no-op if already 0.0 (no spurious
        # _refresh_canvas call), and re-triggers _on_altitude_lag_changed
        # via valueChanged otherwise.
        self._altitude_lag_spin.setValue(0.0)

        # Raw record (variable interval), used to build PowerBlocks.
        # Distance/Time/Start/Match score are all already visible on the
        # combo item that's now selected (see _load_fit) -- there is no
        # separate status label to also update/clear.
        self._activity_raw = cand.record

        # Overlay this candidate's own recorded GPS track on the minimap,
        # alongside the (already-set, static) GPX course line — see
        # CourseMinimapWidget's class docstring for why the two get
        # separate cursor dots rather than being forced onto one line.
        self._minimap.set_activity_track(
            self._activity_raw.lat_deg,
            self._activity_raw.lon_deg,
            self._activity_raw.distance_m,
            dense_lats=self._activity_raw.dense_lat_deg,
            dense_lons=self._activity_raw.dense_lon_deg,
            dense_dists=self._activity_raw.dense_distance_m,
        )

        self._refresh_canvas()

        # A new Activity means any previously-shown sensitivity result no
        # longer describes this ride -- clear it immediately, and reset
        # both N/r spinboxes and their confirmed values back to defaults.
        # No automatic run: Activity selection doesn't touch the Auto Fit
        # checkboxes that _run_sensitivity screens, so the next result
        # comes from an explicit trigger (Enter in a spinbox, or a method
        # switch), same as any later recompute.
        self._morris_result = None
        self._sobol_result = None
        self._physics_panel.clear_sensitivity()
        self._sensitivity_confirmed_param = {
            "sobol": calibrator.DEFAULT_SOBOL_N, "morris": calibrator.DEFAULT_MORRIS_R,
        }
        self._sensitivity_morris_spin.setValue(calibrator.DEFAULT_MORRIS_R)
        # Signals blocked -- switching the radio here would otherwise
        # route through _on_sensitivity_method_changed and trigger an
        # unwanted recompute as a side effect of this reset.
        self._sensitivity_sobol_radio.blockSignals(True)
        self._sensitivity_sobol_radio.setChecked(True)
        self._sensitivity_sobol_radio.blockSignals(False)
        self._sensitivity_sobol_spin.setValue(calibrator.DEFAULT_SOBOL_N)
        self._refresh_sensitivity_controls_enabled()

    @Slot()
    def _on_add_rebuild(self):
        """
        Build and enqueue a new scenario from the current UI state.

        Always replays Activity (FIT-recorded) power. If no
        PhysicsOverridePanel row has its "Auto Fit" checkbox checked,
        this is a plain manual Rebuild: every parameter fixed at its
        spinbox value, added immediately. If any rows are checked, this
        instead launches an AutoFitWorker with those parameters free
        and every other row fixed; the Rebuild is added once calibration
        finishes (_on_auto_fit_finished).
        """
        if self._activity_raw is None:
            QMessageBox.warning(
                self, "No Activity",
                "Load an activity FIT before adding a rebuild."
            )
            return

        overrides = self._physics_panel.get_overrides()
        free_keys = self._physics_panel.free_keys()
        label_text = self._rebuild_label_edit.text().strip() or self._auto_label()

        if not free_keys:
            self._add_rebuild("actual", overrides, label_text, auto_fit_keys=[])
            return

        self._btn_run.setEnabled(False)
        self._physics_panel.setAutoFitEnabled(False)
        self._btn_auto_fit_cancel.setVisible(True)
        self._auto_fit_progress_label.setText("Starting…")
        self._auto_fit_start_time = time.monotonic()

        worker = AutoFitWorker(
            self._strategy,
            self._activity_raw,
            overrides,
            free_keys,
            parent=self,
        )
        self._auto_fit_n_seeds_requested = calibrator.calculate_num_trials(len(free_keys))
        worker.progress.connect(self._on_auto_fit_progress)
        worker.finished.connect(self._on_auto_fit_finished)
        worker.error.connect(self._on_auto_fit_error)
        worker.cancelled.connect(self._on_auto_fit_cancelled)
        self._auto_fit_worker = worker
        self._workers.append(worker)
        worker.start()

    def _auto_fit_reset_controls(self):
        """Re-enable the Add Rebuild button / Auto Fit checkboxes after a
        run ends (success, error, or cancellation all funnel through here)."""
        self._btn_run.setEnabled(True)
        self._physics_panel.setAutoFitEnabled(True)
        self._btn_auto_fit_cancel.setVisible(False)
        self._btn_auto_fit_cancel.setEnabled(True)
        self._auto_fit_worker = None

    @Slot(int, int, object)
    def _on_auto_fit_progress(self, n_done: int, n_total: int, best_rmse_mps):
        elapsed = time.monotonic() - self._auto_fit_start_time
        if best_rmse_mps is None:
            self._auto_fit_progress_label.setText(f"trial {n_done}/{n_total} — elapsed {elapsed:.0f}s")
        else:
            self._auto_fit_progress_label.setText(
                f"trial {n_done}/{n_total} — best velocity RMSE={best_rmse_mps:.3f}m/s — elapsed {elapsed:.0f}s"
            )

    @Slot(object)
    def _on_auto_fit_finished(self, result: "calibrator.CalibrationResult"):
        """
        Add the calibrated result as a normal Rebuild.

        Re-simulates once more via the ordinary _add_rebuild/SimulationWorker
        path rather than reusing calibrate()'s internal trace — a second
        run, but it keeps calibration results indistinguishable from a
        manual Rebuild everywhere downstream (list item, plot, color,
        tooltip), with no separate code path to maintain. `result` itself
        is still attached to the new Scenario (see _add_rebuild's
        calibration_result param) so "Diagnose Fit" can show it later.
        """
        self._auto_fit_reset_controls()

        requested = getattr(self, "_auto_fit_n_seeds_requested", result.n_trials)
        status = f"Done — velocity RMSE={result.rmse_mps:.3f}m/s over {result.n_trials}/{requested} trials"
        if result.from_cache:
            status += f" (cached, computed {result.cached_at})"
        if result.n_trials < requested:
            status += f" ({requested - result.n_trials} crashed — see console)"
        if result.n_converged < result.n_trials:
            status += f", {result.n_trials - result.n_converged} hit maxiter"
        self._auto_fit_progress_label.setText(status)

        # Full diagnostic detail goes to the console rather than the UI —
        # x_std in particular needs the per-key bounds context to read
        # meaningfully (see CalibrationResult.x_std docstring), which
        # doesn't fit in a one-line status label.
        logger.info(
            "Auto Fit done: rmse_mps=%.4f, n_trials=%d/%d, n_converged=%d, x_best=%s, x_std=%s",
            result.rmse_mps, result.n_trials, requested, result.n_converged,
            dict(zip(result.free_keys, result.x_best)), result.x_std,
        )

        label_text = self._rebuild_label_edit.text().strip() or self._auto_label()
        self._add_rebuild(
            "actual", result.physics_overrides, label_text,
            auto_fit_keys=result.free_keys, calibration_result=result,
        )

    @Slot(str)
    def _on_auto_fit_error(self, msg: str):
        self._auto_fit_reset_controls()
        self._auto_fit_progress_label.setText("")
        QMessageBox.critical(self, "Auto Fit Error", msg)

    @Slot()
    def _on_auto_fit_cancelled(self):
        self._auto_fit_reset_controls()
        self._auto_fit_progress_label.setText("Cancelled.")

    @Slot()
    def _on_auto_fit_cancel_clicked(self):
        if self._auto_fit_worker is not None:
            self._auto_fit_progress_label.setText("Cancelling…")
            self._btn_auto_fit_cancel.setEnabled(False)
            self._auto_fit_worker.requestInterruption()

    @Slot()
    def _on_show_diagnostics(self):
        """
        Open CalibrationDiagnosticsDialog for the highlighted Rebuild's own
        Auto Fit result. Built fresh on every click, not cached -- these
        are cheap matplotlib Figures over already-computed CalibrationResult
        data (no re-simulation).

        Shown non-modally (show(), not exec()) so a second click opens an
        independent extra window instead of blocking until the first
        closes -- useful for comparing two Rebuilds' diagnostics side by
        side. WA_DeleteOnClose frees the figure when its dialog closes.

        _btn_diagnostics is only enabled when the highlighted Rebuild has a
        calibration_result, so the guard below is defensive.
        """
        scenario = (
            self._rebuilds[self._last_selected_row]
            if 0 <= self._last_selected_row < len(self._rebuilds) else None
        )
        if scenario is None or scenario.calibration_result is None:
            QMessageBox.information(self, "Auto Fit Diagnostics", "Highlight a Rebuild made with Auto Fit first.")
            return
        dialog = CalibrationDiagnosticsDialog(scenario.calibration_result, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    @Slot()
    def _on_show_sensitivity_details(self) -> None:
        """"Check S2" was pressed -- open SobolS2DetailsDialog for
        self._sobol_result. Built fresh on every click (not cached, same
        reasoning as _on_show_diagnostics) and shown non-modally.
        _btn_sensitivity_details is only enabled when self._sobol_result
        exists with >= 2 free_keys (see _refresh_sensitivity_details_
        button_enabled), so the guard below is defensive."""
        if self._sobol_result is None:
            return
        dialog = SobolS2DetailsDialog(self._sobol_result, self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()

    def _build_sensitivity_controls(self) -> None:
        """Build (once) the Sobol'/Morris radio choice and each method's
        own labeled N/r spinbox, directly into self._physics_panel.
        sensitivity_controls_container -- always visible in the
        "Sensitivity" column header, no popup to open first. A single
        persistent set of widgets, not rebuilt per strategy load: _on_
        sensitivity_finished/_on_sensitivity_error/_refresh_sensitivity_
        display all read/write these same widgets regardless of what's
        currently loaded. Discard-count and progress status go to the
        logger (INFO) rather than a status label.

        No Recompute button -- only the radio-selected method's spinbox
        is enabled (_refresh_sensitivity_controls_enabled), and pressing
        Enter in it recomputes immediately and CONFIRMS that value
        (_on_sensitivity_param_entered, self._sensitivity_confirmed_
        param). Each spinbox has its own adjacent label naming its number
        (Sobol's N sample size vs. Morris's r trajectory count) -- the
        two are unrelated quantities with different scales/defaults.

        Switching the radio selection always recomputes immediately with
        the newly-selected method's last confirmed value (see _on_
        sensitivity_method_changed), and snaps the just-deselected
        method's spinbox back to ITS last-confirmed value, discarding any
        edit typed but never confirmed with Enter.
        """
        container = self._physics_panel.sensitivity_controls_container
        controls_row = QHBoxLayout(container)
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(4)

        sobol_radio = QRadioButton("Sobol'")
        morris_radio = QRadioButton("Morris")
        # macOS's native style ignores QRadioButton::indicator sizing in
        # a stylesheet (fixed ~16px regardless), so these two switch to
        # Fusion instead, which does honor it -- shrunk to fit the
        # header row alongside "Sensitivity" and the other column
        # headers. self._sensitivity_radio_style keeps the QStyle object
        # alive: setStyle() doesn't take ownership, so an unreferenced
        # one would be garbage-collected out from under the widgets.
        self._sensitivity_radio_style = QStyleFactory.create("Fusion")
        radio_qss = (
            "QRadioButton { color: #cccccc; font-size: 9px; spacing: 2px; }"
            "QRadioButton::indicator { width: 7px; height: 7px; border-radius: 4px; "
            "border: 1px solid #888888; background-color: transparent; }"
            "QRadioButton::indicator:checked { background-color: #cccccc; border: 1px solid #cccccc; }"
        )
        for radio in (sobol_radio, morris_radio):
            radio.setStyle(self._sensitivity_radio_style)
            radio.setStyleSheet(radio_qss)
        method_group = QButtonGroup(container)
        method_group.addButton(sobol_radio)
        method_group.addButton(morris_radio)
        # Checked, but not yet wired to _on_sensitivity_method_changed
        # (connected at the very end of this method instead) -- this
        # initial True is the widget's own default state, not a real
        # switch, and that handler assumes self._sensitivity_confirmed_
        # param/_sensitivity_sobol_spin/etc. already exist.
        sobol_radio.setChecked(True)
        self._sensitivity_sobol_radio = sobol_radio
        self._sensitivity_morris_radio = morris_radio

        # Short labels ("N"/"r", not "N (samples)"/"r (trajectories)") --
        # the inline header row has limited width; each spinbox's own
        # tooltip spells out the full meaning.
        sobol_n_label = QLabel("N")
        sobol_n_label.setStyleSheet("color: #999999; font-size: 9px;")
        sobol_spin = NoScrollSpinBox()
        sobol_spin.setRange(4, 100000)
        sobol_spin.setValue(calibrator.DEFAULT_SOBOL_N)
        sobol_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        sobol_spin.setMaximumWidth(45)
        sobol_spin.setStyleSheet("font-size: 9px;")
        sobol_spin.setToolTip(
            "Sobol' sample size N. Total evaluations = N × (free params + 2). "
            "Press Enter to recompute."
        )
        sobol_spin.lineEdit().returnPressed.connect(
            lambda: self._on_sensitivity_param_entered("sobol")
        )
        self._sensitivity_sobol_spin = sobol_spin

        morris_r_label = QLabel("r")
        morris_r_label.setStyleSheet("color: #999999; font-size: 9px;")
        morris_spin = NoScrollSpinBox()
        morris_spin.setRange(1, 1000)
        morris_spin.setValue(calibrator.DEFAULT_MORRIS_R)
        morris_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        morris_spin.setMaximumWidth(45)
        morris_spin.setStyleSheet("font-size: 9px;")
        morris_spin.setToolTip(
            "Morris trajectory count r. Total evaluations = r × (free params + 1). "
            "Press Enter to recompute."
        )
        morris_spin.lineEdit().returnPressed.connect(
            lambda: self._on_sensitivity_param_entered("morris")
        )
        self._sensitivity_morris_spin = morris_spin

        controls_row.addWidget(sobol_radio)
        controls_row.addWidget(sobol_n_label)
        controls_row.addWidget(sobol_spin)
        controls_row.addWidget(morris_radio)
        controls_row.addWidget(morris_r_label)
        controls_row.addWidget(morris_spin)
        controls_row.addStretch(1)

        # Built into sensitivity_details_button_container -- the empty
        # placeholder PhysicsOverridePanel leaves to the RIGHT of its
        # "Sensitivity" header label (a sibling row of controls_row, not
        # part of it) -- so the button sits next to "Sensitivity", not at
        # the far end of the Sobol'/Morris controls row below. Opens
        # SobolS2DetailsDialog for the S1/S2 interaction matrix SALib
        # already computed for this screen but the inline bars only
        # summarize to one number per parameter. Sobol'-only (Morris has
        # no second-order term -- see MorrisSensitivityTrials' docstring),
        # so its enabled state is refreshed alongside the display rather
        # than tied to _sensitivity_controls_ready -- see _refresh_
        # sensitivity_details_button_enabled.
        # Plain "S2", not a subscript rendering -- Sobol' notation's
        # subscript is the INPUT VARIABLE's own index (S_i means
        # "variable i"), not an "order" label, so a literal digit
        # subscript here would misread as "index of variable #2" rather
        # than "the second-order interaction indices". Same reasoning at
        # the S1/ST title built in _on_sensitivity_bar_clicked below.
        btn_sensitivity_details = QPushButton("Check S2")
        btn_sensitivity_details.setStyleSheet("font-size: 9px; padding: 1px 6px;")
        btn_sensitivity_details.setToolTip(
            "Sobol' S1/S2 interaction matrix for the current screen "
            "(Sobol' method only, needs ≥ 2 checked Fit boxes)."
        )
        btn_sensitivity_details.clicked.connect(self._on_show_sensitivity_details)
        self._btn_sensitivity_details = btn_sensitivity_details
        details_container_layout = self._physics_panel.sensitivity_details_button_container.layout()
        assert details_container_layout is not None  # PhysicsOverridePanel always gives this container a layout
        details_container_layout.addWidget(btn_sensitivity_details)

        # Set directly (not via _apply_strategy yet -- that hasn't run
        # at construction time) so the initial refresh below correctly
        # starts everything disabled until a strategy actually loads.
        self._sensitivity_controls_ready = False
        self._sensitivity_confirmed_param = {
            "sobol": calibrator.DEFAULT_SOBOL_N, "morris": calibrator.DEFAULT_MORRIS_R,
        }
        self._refresh_sensitivity_controls_enabled()
        self._refresh_sensitivity_details_button_enabled()
        sobol_radio.toggled.connect(self._on_sensitivity_method_changed)

    @Slot(str)
    def _on_sensitivity_bar_clicked(self, key: str) -> None:
        """A row's SensitivityBarWidget was pressed -- open a popup of
        scatter(s) built from whichever *SensitivityTrials the
        currently-selected method last computed (self._morris_result/
        _sobol_result; None is a silent no-op). Closed again on release
        (sensitivity_bar_released -> _hide_scatter_popup).

        Morris: always exactly one plot, the classic (mu_star, sigma)
        whole-run scatter (plot_morris_mustar_sigma_scatter) with `key`
        highlighted.

        Sobol': always exactly one plot -- the plain 2D effect scatter
        (plot_sensitivity_effect_scatter, key's sampled value vs. RMSE),
        titled with both S1 and ST so the figure answers whether the
        parameter matters on its own (S1) and overall once every
        interaction it takes part in is folded in (ST). The per-partner
        S2 interaction view lives separately in SobolS2DetailsDialog
        ("Check S2"), where the viewer picks the pair off the matrix.
        """
        method = self._sensitivity_method()
        if method == "morris":
            if self._morris_result is None or key not in self._morris_result.mu_star:
                return
            figs = [diag.plot_morris_mustar_sigma_scatter(
                self._morris_result.mu_star, self._morris_result.sigma,
                self._morris_result.mu_star_conf, highlight_key=key,
            )]
        else:
            if self._sobol_result is None or key not in self._sobol_result.sample_x:
                return
            figs = [diag.plot_sensitivity_effect_scatter(
                key, self._sobol_result.sample_x[key], self._sobol_result.sample_y,
                title=(
                    f"Sobol' S1 = {self._sobol_result.s1[key]:.3g}"
                    f"±{self._sobol_result.s1_conf[key]:.3g}, "
                    f"ST = {self._sobol_result.st[key]:.3g}"
                    f"±{self._sobol_result.st_conf[key]:.3g}"
                ),
            )]
        self._show_scatter_popup(figs)

    def _show_scatter_popup(self, figs: list) -> None:
        """Show one or more matplotlib Figures (from
        _on_sensitivity_bar_clicked) side by side in a small frameless
        popup near the cursor -- same convention CalibrationDiagnostics
        Dialog._show_pair_popup uses for its click-a-heatmap-cell popup,
        adapted for a plain Qt widget press: SensitivityBarWidget.clicked
        carries no position of its own, so QCursor.pos() stands in."""
        self._hide_scatter_popup()

        popup = QWidget(self, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        popup.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        popup_layout = QHBoxLayout(popup)
        popup_layout.setContentsMargins(1, 1, 1, 1)
        popup_layout.setSpacing(2)

        for fig in figs:
            canvas = FigureCanvas(fig)
            width_px = int(fig.get_size_inches()[0] * fig.dpi)
            height_px = int(fig.get_size_inches()[1] * fig.dpi)
            canvas.setFixedSize(width_px, height_px)
            popup_layout.addWidget(canvas)
        popup.adjustSize()

        # See CalibrationDiagnosticsDialog._show_pair_popup's identical
        # clamping comment -- same multi-monitor reasoning.
        global_pos = QCursor.pos() + QPoint(12, 12)
        screen = QApplication.screenAt(global_pos) or self.screen()
        if screen is not None:
            popup_size = popup.size()
            avail = screen.availableGeometry()
            x = max(avail.left(), min(global_pos.x(), avail.right() - popup_size.width()))
            y = max(avail.top(), min(global_pos.y(), avail.bottom() - popup_size.height()))
            global_pos = QPoint(x, y)
        popup.move(global_pos)
        popup.show()
        self._scatter_popup = popup

    @Slot()
    def _hide_scatter_popup(self) -> None:
        if self._scatter_popup is not None:
            self._scatter_popup.close()
            self._scatter_popup.deleteLater()
            self._scatter_popup = None

    def _sensitivity_method(self) -> str:
        return "sobol" if self._sensitivity_sobol_radio.isChecked() else "morris"

    def _refresh_sensitivity_controls_enabled(self) -> None:
        """Only the radio-selected method's spinbox is enabled -- typing
        into the other one would do nothing (Enter only recomputes the
        selected method, see _on_sensitivity_param_entered). Both
        spinboxes and both radios are also disabled while a run is
        already in flight, so a second Enter/method switch can't overlap
        it, and while no strategy has loaded yet (self.
        _sensitivity_controls_ready)."""
        ready = self._sensitivity_controls_ready and self._sensitivity_worker is None
        self._sensitivity_sobol_radio.setEnabled(ready)
        self._sensitivity_morris_radio.setEnabled(ready)
        self._sensitivity_sobol_spin.setEnabled(ready and self._sensitivity_sobol_radio.isChecked())
        self._sensitivity_morris_spin.setEnabled(ready and self._sensitivity_morris_radio.isChecked())

    def _refresh_sensitivity_details_button_enabled(self) -> None:
        """"Check S2" only makes sense for a finished Sobol' screen
        with >= 2 free parameters -- Morris has no second-order term
        (see MorrisSensitivityTrials' docstring), and a single free
        parameter has no other parameter for S2 to pair with. Called
        wherever self._sobol_result or the selected method can change,
        kept separate from _refresh_sensitivity_controls_enabled's
        "ready" gate since this button's enabled state also depends on
        _sobol_result."""
        self._btn_sensitivity_details.setEnabled(
            self._sensitivity_method() == "sobol"
            and self._sobol_result is not None
            and len(self._sobol_result.s1) >= 2
        )

    @Slot(bool)
    def _on_sensitivity_method_changed(self, sobol_checked: bool) -> None:
        """Switching the radio selection flips which spinbox is editable,
        snaps the just-deselected method's spinbox back to its own last
        confirmed value (discarding any typed-but-never-Entered edit),
        and unconditionally recomputes the newly-selected method with its
        last confirmed value, so switching methods always shows a result
        for the one now on screen rather than a stale/empty one."""
        if sobol_checked:
            method = "sobol"
            self._sensitivity_morris_spin.setValue(self._sensitivity_confirmed_param["morris"])
        else:
            method = "morris"
            self._sensitivity_sobol_spin.setValue(self._sensitivity_confirmed_param["sobol"])
        self._refresh_sensitivity_controls_enabled()
        self._refresh_sensitivity_display()
        self._run_sensitivity(method, self._sensitivity_confirmed_param[method])

    @Slot()
    def _on_sensitivity_inputs_changed(self) -> None:
        """A Fit checkbox or a spinbox's value changed -- by any means
        (see PhysicsOverridePanel.sensitivity_inputs_changed for the
        full list). (Re)starts self._sensitivity_input_debounce instead
        of rerunning Sensitivity immediately (SENSITIVITY_INPUT_
        DEBOUNCE_MS), so a burst of these settles into exactly one
        _run_sensitivity call for the final state."""
        self._sensitivity_input_debounce.start(self.SENSITIVITY_INPUT_DEBOUNCE_MS)

    def _on_sensitivity_inputs_settled(self) -> None:
        """self._sensitivity_input_debounce fired -- the Fit checkbox/
        spinbox state has held still for SENSITIVITY_INPUT_DEBOUNCE_MS,
        so rerun Sensitivity now with the currently-selected method's
        last confirmed N/r, same deliberate auto-pipeline exception _on_
        sensitivity_method_changed already makes, so the bars on screen
        always describe the free-parameter set and fixed values that are
        actually on screen right now rather than whatever they were the
        last time someone pressed Enter."""
        method = self._sensitivity_method()
        self._run_sensitivity(method, self._sensitivity_confirmed_param[method])

    def _refresh_sensitivity_display(self) -> None:
        method = self._sensitivity_method()
        if method == "sobol" and self._sobol_result is not None:
            self._physics_panel.show_sobol_sensitivity(self._sobol_result)
        elif method == "morris" and self._morris_result is not None:
            self._physics_panel.show_morris_sensitivity(self._morris_result)
        else:
            self._physics_panel.clear_sensitivity()
        self._refresh_sensitivity_details_button_enabled()

    def _run_sensitivity(self, method: str, param: int) -> None:
        """Launch a SensitivityWorker for `method` ("morris"/"sobol") with
        `param` (r for Morris, N for Sobol') against whichever parameters
        currently have their Auto Fit checkbox checked (PhysicsOverride
        Panel.free_keys()) -- every other row stays fixed at its spinbox
        value, same split _on_add_rebuild uses. A free_key with no effect
        on RMSE at all is a genuinely indeterminate 0/0 for Sobol's
        variance-normalized S1/ST; calibrator.sample_sobol_sensitivity
        resolves it by construction (see calibrator._tie_breaker) so
        S1/ST -> 0 exactly as real RMSE variance -> 0. Zero boxes checked
        has nothing to screen, so that alone logs a line and clears any
        stale bars instead of launching a worker.

        A run already in flight gets requestInterruption()'d first rather
        than left to finish alongside the new one. The new SensitivityWorker
        is handed self._sensitivity_pool -- one ProcessPoolExecutor shared
        across every run for this window's lifetime -- so superseding a
        run doesn't also spin up a second pool; the new run's chunks just
        queue onto the same pool the old one's already-dispatched chunks
        are still finishing on. The old worker's finished/error/cancelled
        handlers all check `w is self._sensitivity_worker` first and
        silently ignore a stale worker's signal, since interruption isn't
        instantaneous and a late-arriving old result must not clobber a
        newer one."""
        if self._activity_raw is None:
            return
        if self._sensitivity_worker is not None:
            self._sensitivity_worker.requestInterruption()
        free_keys = self._physics_panel.free_keys()
        if not free_keys:
            self._morris_result = None
            self._sobol_result = None
            self._physics_panel.clear_sensitivity()
            self._physics_panel.set_sensitivity_busy(False)
            self._refresh_sensitivity_details_button_enabled()
            logger.info("Check at least 1 Fit box to run sensitivity screening.")
            return
        logger.debug("Running %s sensitivity…", "Sobol'" if method == "sobol" else "Morris")
        kwargs = {"n": param} if method == "sobol" else {"r": param}
        worker = SensitivityWorker(
            self._strategy, self._activity_raw, self._physics_panel.get_overrides(),
            free_keys, method, kwargs, pool=self._sensitivity_pool, parent=self,
        )
        worker.finished.connect(lambda result, m=method, w=worker: self._on_sensitivity_finished(w, m, result))
        worker.error.connect(lambda msg, w=worker: self._on_sensitivity_error(w, msg))
        worker.cancelled.connect(lambda w=worker: self._on_sensitivity_cancelled(w))
        self._sensitivity_worker = worker
        self._physics_panel.set_sensitivity_busy(True)
        self._refresh_sensitivity_controls_enabled()
        worker.start()

    def _on_sensitivity_finished(self, worker: "SensitivityWorker", method: str, result) -> None:
        if worker is not self._sensitivity_worker:
            return  # Superseded by a newer run -- see _run_sensitivity.
        self._sensitivity_worker = None
        self._physics_panel.set_sensitivity_busy(False)
        self._refresh_sensitivity_controls_enabled()
        if method == "morris":
            self._morris_result = result
            status = _sensitivity_discard_status(
                result.n_trajectories_discarded, result.n_trajectories_requested,
                "trajectory", "trajectories", result.infeasible_entry_counts,
                result.n_baseline_infeasible, "already infeasible at baseline",
            )
        else:
            self._sobol_result = result
            status = _sensitivity_discard_status(
                result.n_replicates_discarded, result.n_replicates_requested,
                "replicate", "replicates", result.infeasible_ab_counts,
                result.n_base_point_infeasible, "base A/B point(s) infeasible",
            )
        if status:
            logger.info(status)
        if self._sensitivity_method() == method:
            self._refresh_sensitivity_display()
        else:
            # _refresh_sensitivity_display (which also refreshes this
            # button) only runs above when the finished run matches
            # whichever method is currently on screen -- refreshed here
            # too so a Sobol' run finishing while Morris is selected
            # still updates the button the moment it's switched back to.
            self._refresh_sensitivity_details_button_enabled()

    def _on_sensitivity_error(self, worker: "SensitivityWorker", msg: str) -> None:
        if worker is not self._sensitivity_worker:
            return  # Superseded by a newer run -- see _run_sensitivity.
        self._sensitivity_worker = None
        self._physics_panel.set_sensitivity_busy(False)
        self._refresh_sensitivity_controls_enabled()
        QMessageBox.critical(self, "Sensitivity Error", msg)

    def _on_sensitivity_cancelled(self, worker: "SensitivityWorker") -> None:
        """The common case: `worker` was requestInterruption()'d by
        _run_sensitivity because a newer run superseded it, so `worker is
        not self._sensitivity_worker` here and this is a silent no-op --
        the newer run already owns self._sensitivity_worker and its own
        controls-enabled state."""
        if worker is not self._sensitivity_worker:
            return
        self._sensitivity_worker = None
        self._physics_panel.set_sensitivity_busy(False)
        self._refresh_sensitivity_controls_enabled()

    def _on_sensitivity_param_entered(self, method: str) -> None:
        """Enter in the (only-editable) N/r spinbox confirms its current
        value (self._sensitivity_confirmed_param) and recomputes
        immediately -- no separate Recompute button. Only reachable for
        the radio-selected method in practice, since the other spinbox is
        disabled, but `method` is still passed explicitly per spinbox
        rather than re-derived from the radio state, so this can't
        silently confirm/recompute the wrong one if that invariant is
        ever broken."""
        if self._activity_raw is None:
            QMessageBox.warning(
                self, "No Activity",
                "Load an activity FIT before checking sensitivity."
            )
            return
        spin = self._sensitivity_sobol_spin if method == "sobol" else self._sensitivity_morris_spin
        self._sensitivity_confirmed_param[method] = spin.value()
        self._run_sensitivity(method, spin.value())

    def _refresh_selection_buttons(self):
        """Enable/disable Remove Rebuild, Generate Config, and Diagnose Fit
        per the highlighted Rebuild.

        Remove Rebuild and Generate Config need only a highlighted Rebuild
        to exist; Diagnose Fit also requires it carry its own
        calibration_result (produced by Auto Fit, not added manually).
        Called everywhere self._last_selected_row changes, so all three
        always reflect whichever Rebuild is currently highlighted.
        """
        scenario = (
            self._rebuilds[self._last_selected_row]
            if 0 <= self._last_selected_row < len(self._rebuilds) else None
        )
        self._btn_remove_rebuild.setEnabled(scenario is not None)
        self._btn_gen_config.setEnabled(scenario is not None)
        self._btn_diagnostics.setEnabled(scenario is not None and scenario.calibration_result is not None)

    def _start_worker(self, scenario: Scenario, on_done, on_error):
        """Launch a SimulationWorker for `scenario` and wire its callbacks.

        Shared by _run_strategy_scenario and _add_rebuild — the two differ
        only in what happens on completion (Strategy has no list item to
        update; a Rebuild does), so on_done/on_error carry that difference
        in as closures built at each call site.
        """
        worker = SimulationWorker(self._strategy, scenario, self._activity_raw, self)
        worker.finished.connect(on_done)
        worker.error.connect(on_error)
        self._workers.append(worker)
        worker.start()

    def _run_strategy_scenario(self):
        """Run the auto-added Strategy scenario (planned power, no overrides).

        Unlike a Rebuild, this scenario is tracked in self._strategy_scenario
        rather than self._rebuilds/self._rebuild_list — it never appears
        in the Rebuilds panel and has no remove control, since the Δt
        panel always needs a reference line. Reserves SCENARIO_COLORS[0];
        see _add_rebuild's offset.
        """
        scenario = Scenario(
            label="Strategy",
            color=SCENARIO_COLORS[0],
            power_source="planned",
            physics_overrides={},
        )
        self._strategy_scenario = scenario
        self._start_worker(
            scenario,
            on_done=lambda trace, s=scenario: self._on_strategy_scenario_done(trace, s),
            on_error=self._on_strategy_scenario_error,
        )

    @Slot(object, object)
    def _on_strategy_scenario_done(self, trace: SimTrace, scenario: Scenario):
        """Receive the Strategy scenario's finished SimTrace and redraw."""
        scenario.trace = trace
        self._refresh_canvas()

    @Slot(str)
    def _on_strategy_scenario_error(self, msg: str):
        """Handle a Strategy scenario simulation failure."""
        QMessageBox.critical(self, "Simulation Error", f"Strategy: {msg}")

    def _auto_label(self) -> str:
        """Generate the default label for the next Rebuild.

        Reads self._next_rebuild_num without incrementing it — incrementing
        happens once, in _add_rebuild, when a Rebuild is actually created.
        No time suffix here — that's added separately once the sim
        finishes (Rebuilds list item, plot legend), so putting one here
        too would double up (e.g. "Rebuild #1 (00:00)  (Sim: 02:15 / ...)").
        """
        return f"Rebuild #{self._next_rebuild_num}"

    def _load_physics_state(self, overrides: dict, auto_fit_keys: list) -> None:
        """Push `overrides`/`auto_fit_keys` into self._physics_panel --
        the one path every caller that loads a Rebuild's (or a just-
        completed fit's) saved state into the panel goes through, instead
        of calling set_values()/set_auto_fit_checks()/mark_fit_confirmed()
        separately and risking one drifting out of sync. `auto_fit_keys`
        doubles as "was this actually fit": non-empty means overrides
        came from a completed Auto Fit, so those keys' values are
        confirmed, not muted as pending (PhysicsOverridePanel.
        mark_fit_confirmed); empty means a plain manual Rebuild and
        mark_fit_confirmed is a no-op."""
        self._physics_panel.set_values(overrides)
        self._physics_panel.set_auto_fit_checks(auto_fit_keys)
        self._physics_panel.mark_fit_confirmed(auto_fit_keys)

    def _add_rebuild(
        self,
        power_source: Literal["planned", "actual"],
        overrides: dict,
        label: str,
        auto_fit_keys: list | None = None,
        calibration_result: "calibrator.CalibrationResult | None" = None,
    ):
        """Create a Rebuild Scenario, start the worker, and add it to the list.

        auto_fit_keys: keys whose Auto Fit checkbox was checked to produce
        `overrides` (empty/None for a plain manual Rebuild). Stored on the
        Scenario so _on_rebuild_item_clicked can restore the checkbox state,
        not just the spinbox values, when this Rebuild is clicked again.
        calibration_result: the CalibrationResult that produced `overrides`,
        if any (None for a plain manual Rebuild) -- stored on the Scenario
        so "Diagnose Fit" can show it when this Rebuild is highlighted.
        """
        # Cycle through SCENARIO_COLORS[1:] only — index 0 is reserved for
        # Strategy and must never be reachable here, not even on
        # wraparound (a plain "% len(SCENARIO_COLORS)" would collide with
        # Strategy's color every 13th Rebuild, since SCENARIO_COLORS has
        # 13 entries). Indexed off
        # _next_rebuild_num (monotonic), not len(self._rebuilds) — see
        # that field's docstring for why list length collides after a
        # remove.
        n_rotating = len(SCENARIO_COLORS) - 1
        color = SCENARIO_COLORS[1 + (self._next_rebuild_num - 1) % n_rotating]
        scenario = Scenario(
            label=label,
            color=color,
            power_source=power_source,
            physics_overrides=overrides,
            auto_fit_keys=list(auto_fit_keys) if auto_fit_keys else [],
            calibration_result=calibration_result,
        )
        self._rebuilds.append(scenario)
        self._next_rebuild_num += 1
        self._rebuild_label_edit.setText(self._auto_label())

        # Add placeholder to list widget
        item = QListWidgetItem(f"⏳ {label}")
        item.setForeground(QColor(color))
        item.setToolTip(self._physics_panel.format_overrides_tooltip(overrides))
        self._rebuild_list.addItem(item)
        list_idx = self._rebuild_list.count() - 1

        # Auto-select the new Rebuild: parameters are known immediately
        # (no need to wait for the simulation to finish), and this drives
        # update_plots' highlight/dim treatment once _refresh_canvas runs.
        # Set directly rather than via QListWidget signals — see
        # _on_rebuild_item_clicked's docstring for why.
        self._rebuild_list.setCurrentRow(list_idx)
        self._last_selected_row = list_idx
        self._load_physics_state(overrides, scenario.auto_fit_keys)
        self._refresh_selection_buttons()

        self._btn_run.setEnabled(False)

        self._start_worker(
            scenario,
            on_done=lambda trace, s=scenario, i=list_idx: self._on_rebuild_done(trace, s, i),
            on_error=lambda msg, i=list_idx: self._on_rebuild_error(msg, i),
        )

    @Slot(object, object, int)
    def _on_rebuild_done(self, trace: SimTrace, scenario: Scenario, list_idx: int):
        """Receive finished SimTrace, update scenario, refresh plots."""
        scenario.trace = trace
        item = self._rebuild_list.item(list_idx)
        if item:
            item.setText(f"✓ {trace.label}  ({format_time_mmss(trace.finish_time_s)})")

        self._btn_run.setEnabled(True)
        self._refresh_canvas()

    @Slot(str, int)
    def _on_rebuild_error(self, msg: str, list_idx: int):
        """Handle a worker error: update list item and show a message."""
        item = self._rebuild_list.item(list_idx)
        if item:
            item.setText("✗ Error")
        self._btn_run.setEnabled(True)
        QMessageBox.critical(self, "Simulation Error", msg)

    def closeEvent(self, event):
        """
        Block until every still-running background QThread
        (SimulationWorker/AutoFitWorker/SensitivityWorker) has actually
        finished before letting this window close.

        Qt requires a QThread to have finished before its wrapper object
        is destroyed -- destroying one still running aborts the whole
        process ("QThread: Destroyed while thread is still running"), not
        a Python exception this class could catch. main() does
        `sys.exit(app.exec())` with no wait of its own, so without this
        override, closing the window while a re-simulation/Auto Fit/
        sensitivity run is still in flight crashes the process.

        requestInterruption() lets AutoFitWorker/SensitivityWorker stop
        early -- AutoFitWorker checks isInterruptionRequested() between
        calibration trials, SensitivityWorker roughly once per calibrator.
        _SENSITIVITY_TARGET_CHUNK_S. SimulationWorker has no such hook, so
        requestInterruption() is a no-op for it and wait() simply blocks
        until it finishes naturally.

        self._sensitivity_pool (the ProcessPoolExecutor every
        SensitivityWorker shares) is shut down after every QThread using
        it has actually stopped, never before -- shutting down a pool a
        still-running SensitivityWorker is mid-submit()/as_completed()
        against would be undefined.
        """
        running = [w for w in self._workers if w.isRunning()]
        for w in (self._auto_fit_worker, self._sensitivity_worker):
            if w is not None and w.isRunning():
                running.append(w)
        if running:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                for w in running:
                    w.requestInterruption()
                for w in running:
                    w.wait()
            finally:
                QApplication.restoreOverrideCursor()
        self._sensitivity_pool.shutdown(wait=True, cancel_futures=True)
        event.accept()

    def eventFilter(self, obj, event):
        """
        Up/Down arrow-key navigation for the Rebuild list.

        Installed on self._rebuild_list rather than wired through
        QListWidget's native currentItemChanged, to avoid the
        mouse/keyboard ordering conflict described in
        _on_rebuild_item_clicked's docstring -- that signal fires for
        mouse clicks too, so sharing it with the same toggle-off handler
        would break click-to-select. This path only triggers from a real
        Key_Up/Key_Down press, so it can't interact with that logic.

        Mirrors _on_rebuild_item_clicked's non-toggle branch: move the
        highlighted row, load its saved parameters, refresh the plot
        highlight. Arrow keys have no toggle-off concept, so that part
        of the click handler is intentionally not mirrored here.
        """
        if obj is self._rebuild_list and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Up, Qt.Key_Down):
                count = self._rebuild_list.count()
                if count > 0:
                    row = self._rebuild_list.currentRow()
                    if row < 0:
                        row = 0
                    else:
                        row = max(0, min(count - 1, row + (1 if key == Qt.Key_Down else -1)))
                    self._rebuild_list.setCurrentRow(row)
                    self._last_selected_row = row
                    if 0 <= row < len(self._rebuilds):
                        self._load_physics_state(
                            self._rebuilds[row].physics_overrides, self._rebuilds[row].auto_fit_keys,
                        )
                    self._refresh_selection_buttons()
                    self._refresh_canvas()
                return True
        return super().eventFilter(obj, event)

    @Slot(QListWidgetItem)
    def _on_rebuild_item_clicked(self, item: QListWidgetItem):
        """
        Select a Rebuild: load its saved parameters into the panel and
        refresh the plot highlight. Re-clicking the already-selected row
        toggles it off instead.

        Lets the user inspect an existing Rebuild's physics_overrides and
        tweak a few before adding a new one — the clicked Rebuild itself
        is never modified; "Add Rebuild" always creates a separate entry.
        Its Auto Fit checkbox state is restored too via set_auto_fit_checks,
        so re-running Auto Fit starts from the same free/fixed split as
        when this Rebuild was created. The Label field is deliberately
        NOT set to the clicked Rebuild's name -- it stays on the
        auto-numbered default, so it's clear the new one is a distinct
        Rebuild, not a rename.

        QListWidget's default SingleSelection has no built-in toggle-off,
        so self._last_selected_row tracks what was selected *before* this
        click to tell a second click on the same row apart from a first
        click on a new one.

        Mouse-click only (itemClicked), deliberately: wiring
        QListWidget.currentItemChanged too (for keyboard nav) fires
        before itemClicked on an ordinary mouse click, which would update
        _last_selected_row before this method's toggle-off check runs,
        making every click look like a repeat click on the same row.
        Arrow-key navigation is handled separately via eventFilter()
        instead, which never touches currentItemChanged.
        """
        row = self._rebuild_list.row(item)
        if row == self._last_selected_row:
            self._rebuild_list.clearSelection()
            self._rebuild_list.setCurrentRow(-1)
            self._last_selected_row = -1
        else:
            self._last_selected_row = row
            if 0 <= row < len(self._rebuilds):
                self._load_physics_state(
                    self._rebuilds[row].physics_overrides, self._rebuilds[row].auto_fit_keys,
                )
        self._refresh_selection_buttons()
        self._refresh_canvas()

    @Slot()
    def _on_remove_rebuild(self):
        """Remove the currently selected scenario from the list and plots."""
        row = self._rebuild_list.currentRow()
        if row < 0 or row >= len(self._rebuilds):
            return
        self._rebuilds.pop(row)
        self._rebuild_list.takeItem(row)
        self._last_selected_row = -1
        self._rebuild_label_edit.setText(self._auto_label())
        self._refresh_selection_buttons()
        self._refresh_canvas()

    @Slot()
    def _on_generate_config(self):
        """
        Write the selected Rebuild's physical/physiological parameters out
        as a new configs/templates/*.json, in the same PhysicalSettings/
        PhysiologicalSettings/RunSettings shape eidos.apps.manager/designer/
        trainer already read — so a hand-tuned or Auto-Fit-calibrated
        Rebuild can seed a fresh optimization run instead of only ever
        being a read-only comparison line. Written into configs/templates/
        specifically so it's immediately pickable from eidos.apps.manager's
        template selector.

        PhysicalSettings/PhysiologicalSettings are read straight off
        scenario.physics_overrides, which PhysicsOverridePanel.get_overrides
        already populates for every field except cda_yaw_table_filename
        (always the strategy's own value — no Rebuild control changes
        it). RunSettings and Engine (including optimizer_params) are never
        touched by a Rebuild and are copied from the strategy unchanged —
        Engine has no default in eidos.apps.generator.load_config_jsons,
        so omitting it would silently skip the written config at generate
        time.
        """
        row = self._rebuild_list.currentRow()
        if row < 0 or row >= len(self._rebuilds):
            QMessageBox.warning(
                self, "No Rebuild Selected",
                "Select a Rebuild in the list before generating a config."
            )
            return
        scenario = self._rebuilds[row]
        overrides = scenario.physics_overrides
        simulator_spec = self._strategy.simulator_spec

        physical_settings = {
            k: overrides[k] for k in simulator_spec.physical_param_model.model_fields
            if k != "cda_yaw_table_filename"
        }
        physiological_settings = {k: overrides[k] for k in simulator_spec.physiological_param_model.model_fields}
        physical_settings["cda_yaw_table_filename"] = self._strategy.raw_physical["cda_yaw_table_filename"]
        run_settings = {k: self._strategy.raw_run[k] for k in RunValidationModel.model_fields}

        config = {
            "PhysicalSettings": physical_settings,
            "PhysiologicalSettings": physiological_settings,
            "RunSettings": run_settings,
            "Engine": dict(self._strategy.raw_engine),
        }

        base_name = re.sub(r"[^A-Za-z0-9]+", "_", scenario.label).strip("_") or "rebuild"
        filename = _next_config_filename(base_name)
        os.makedirs(_CONFIGS_TEMPLATES_DIR, exist_ok=True)
        out_path = os.path.join(_CONFIGS_TEMPLATES_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=4, ensure_ascii=False)

        QMessageBox.information(self, "Config Generated", f"Wrote configs/templates/{filename}")

    def _refresh_canvas(self):
        """Redraw the canvas from the Strategy scenario plus current Rebuilds.

        Strategy is prepended so AnalysisCanvas.update_plots picks it as
        the Δt panel's reference line (baseline_trace = first scenario
        with a trace) — see that method's docstring.
        """
        plot_scenarios = []
        if self._strategy_scenario is not None:
            plot_scenarios.append(self._strategy_scenario)
        plot_scenarios.extend(self._rebuilds)

        selected_scenario = None
        if 0 <= self._last_selected_row < len(self._rebuilds):
            selected_scenario = self._rebuilds[self._last_selected_row]

        self._canvas.update_plots(
            self._strategy, plot_scenarios, self._activity_raw, selected_scenario,
            activity_altitude_lag_s=self._altitude_lag_s,
        )

        # Course Map's line/matched-dot colour follows the same
        # highlighted Rebuild as the five plot panels above — see
        # CourseMinimapWidget.set_highlight_color's docstring for which
        # drawn elements this does (and deliberately doesn't) apply to.
        self._minimap.set_highlight_color(
            selected_scenario.color if selected_scenario is not None else None
        )

        # Re-sync the cursor/minimap at whatever pct the slider is already
        # sitting at: a full redraw can change what set_cursor_pct would
        # show there (Rebuild finished/selected/removed, Activity changed)
        # without the user having touched the slider. Cheap, so calling it
        # again here costs nothing.
        self._dist_slider.setEnabled(self._canvas.has_data)
        self._on_slider_moved(self._dist_slider.value())

    @Slot(int)
    def _on_slider_moved(self, value: int):
        """Move the plot cursor (and its on-graph value labels) + minimap dot.

        Deliberately does NOT call _refresh_canvas()/update_plots() — see
        AnalysisCanvas.set_cursor_pct's docstring for why that path must
        stay cheap enough to run on every slider-drag tick.
        """
        if not self._canvas.has_data:
            return
        pct = value / 1000.0
        self._canvas.set_cursor_pct(pct)
        self._minimap.update_cursor(pct)

