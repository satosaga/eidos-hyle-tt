"""
eidos.apps.viewer.window -- TTSimulatorViewer, the Viewer's main window.

Kept as one class (rather than split further), same rationale as
eidos.apps.analyzer.window.TTAnalyzerWindow: QMainWindow construction,
panel building, and event handlers here are all tightly coupled to shared
instance state.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QPoint,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.data_manager import build_course_profile
from core.git_info import check_reproducibility
from core.io_config import (
    BASE_EXPORTS_DIR,
    BASE_STRATEGIES_DIR,
    find_strategy_json_path,
)
from core.schema import PowerBlocks, RunSettings
from core.simulators import resolve_simulator
from eidos.apps.viewer.dialogs import (
    ExportReportDialog,
    RecordDetailDialog,
    StrategySelectorDialog,
)
from eidos.apps.viewer.helpers import load_initial_records
from eidos.apps.viewer.widgets import CourseMapWidget
from eidos.lib.branding import window_title
from eidos.lib.power_profile_canvas import PowerProfileCanvas
from eidos.lib.record_delegate import RecordListDelegate, RecordListHeader
from eidos.lib.record_model import StrategyRecordModel

logger = logging.getLogger(__name__)


# --------------------------------------------
# IV. Main window class (TTSimulatorViewer)
# --------------------------------------------
class TTSimulatorViewer(QMainWindow):
    """Main window for visualising and analysing EIDOS TT simulation results."""
    
    # --- Class constants ---
    DEFAULT_WIDTH = 1350
    DEFAULT_HEIGHT = 800
    MONOSPACE_FONT = "Menlo" if sys.platform == "darwin" else "Consolas"

    # Signals
    trigger_pane_update = Signal()

    def __init__(self, parent=None):
        """Initialize the main window and trigger full data load and UI setup."""
        self.record_model: StrategyRecordModel  # assigned in initialize(), called below
        super().__init__(parent)
        self.setWindowTitle(window_title("Viewer"))
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        # Load data and run all initialisation steps at startup
        self.initialize()

    def initialize(self, scroll_to_key: Optional[Tuple[str, str, str]] = None):
        """Load records, build all widgets, connect signals, and perform the initial canvas update.

        scroll_to_key: an optional StrategyRecordModel.record_key() to center the
        list view on once rebuilt (e.g. a strategy just added via Create Design).
        """
        records = load_initial_records()
        if not records:
            logger.info("No strategy is selected.")
            return

        previous_state = self.record_model.capture_state() if hasattr(self, 'record_model') else None
        self.record_model = StrategyRecordModel(records, previous_state=previous_state)
        # Before _create_widgets(): PowerProfileCanvas.__init__ eagerly calls
        # refresh_from_model(), which would otherwise run _rebuild_data_cache
        # against the newly-selected record while it's still is_index_only
        # (that record's own full JSON hasn't been loaded yet at this point).
        self._ensure_selected_records_loaded()
        self._init_styles()
        self._create_widgets()
        self._setup_layouts()
        self._connect_signals()
        self.update_power_profile()
        self.update_export_pane()
        self.list_view.viewport().installEventFilter(self)
        if scroll_to_key is not None:
            self._scroll_list_to_record(scroll_to_key)

    def eventFilter(self, obj, event):
        """Left-click on a strategy row (outside its checkbox/radio) opens the row's context menu."""
        if (
            obj is self.list_view.viewport()
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            pos = event.pos()
            index = self.list_view.indexAt(pos)
            if index.isValid():
                delegate = self.list_view.itemDelegate()
                row_rect = self.list_view.visualRect(index)
                if not delegate.hits_control(row_rect, index, pos):
                    self.show_context_menu(pos)
                    return True
        return super().eventFilter(obj, event)

    def _scroll_list_to_record(self, key: Tuple[str, str, str]):
        """Center the record list view on the row matching key (a StrategyRecordModel.record_key())."""
        for row, record in enumerate(self.record_model._records):
            if StrategyRecordModel.record_key(record) == key:
                list_view = self.list_view
                model_index = self.record_model.index(row, 0)
                QTimer.singleShot(
                    0,
                    lambda: list_view.scrollTo(model_index, QAbstractItemView.PositionAtCenter)
                )
                break

    def _init_styles(self):
        """Set up the application-wide stylesheet and monospace font."""
        self.ui_font = QFont(self.MONOSPACE_FONT, 12)
        self.setStyleSheet("""
            /* Group box base style */
            QGroupBox { 
                font-weight: bold; 
                border: 1px solid #999; 
                margin-top: 10px; 
                padding-top: 10px; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 3px; 
                font-size: 15px; 
            }

            /* List view */
            QListView { 
                background-color: #606060; 
                color: white; 
                border: 1px solid #ccc; 
            }

            /* --- Unified action button style --- */
            /* Orange (#E67E22) base for visibility */
            QPushButton[class="action-button"] {
                background-color: #E67E22; 
                color: #FFFFFF;             /* fixed white text */
                font-family: 'Menlo'; 
                font-size: 13px;
                font-weight: bold; 
                border-radius: 5px; 
                min-height: 0px;
                padding: 5px 15px;
            }
            QPushButton[class="action-button"]:hover { 
                background-color: #D35400; 
            }
            QPushButton[class="action-button"]:pressed { 
                background-color: #A04000; 
            }
            QPushButton[class="action-button"]:disabled {
                background-color: #333333;
                color: #777777;
            }

            /* Secondary buttons sharing the action-button box model (padding/
               radius/font) so they line up with it pixel-for-pixel -- a plain
               QPushButton uses the native macOS bezel, which reserves extra
               invisible chrome and doesn't align with a stylesheet button of
               the same reported height. */
            QPushButton[class="toggle-button"] {
                background-color: #4a4a4a;
                color: #FFFFFF;
                font-family: 'Menlo';
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 0px;
                padding: 5px 15px;
            }
            QPushButton[class="toggle-button"]:hover {
                background-color: #5a5a5a;
            }
            QPushButton[class="toggle-button"]:pressed {
                background-color: #333333;
            }

            /* Slider style */
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 6px;
                background: #333;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #008000;
                border: 1px solid #5c5c5c;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #006000;
            }
        """)

    def _create_widgets(self):
        """Instantiate all widgets used by the viewer."""
        # Graph canvas
        self.power_profile_canvas = PowerProfileCanvas(self.record_model)
        
        # Course map widget
        self.course_map = CourseMapWidget()

        # Distance slider
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setRange(0, 1000)
        self.dist_slider.setEnabled(False)
        sp = self.dist_slider.sizePolicy()
        sp.setRetainSizeWhenHidden(True) 
        self.dist_slider.setSizePolicy(sp)
        
        # Debounce timer for canvas redraws
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100)

        # Record list view
        self.list_view = QListView()
        self.list_view.setModel(self.record_model)
        self.list_view.setItemDelegate(RecordListDelegate(self.list_view))
        self.list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_view.setMouseTracking(False)
        self.list_view.setWordWrap(False)
        self.list_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setSelectionMode(QListView.NoSelection)

        # Data panel
        self.data_panel = QGroupBox("Data at Cursor")
        self.snapshot_display = QLabel("Select a single record and move the slider.")
        self.snapshot_display.setFont(self.ui_font)

    def _setup_layouts(self):
        """Assemble left (graph+slider) and right (selector+map+actions) panes into the main splitter."""
        # --- Left pane: main graph ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.addWidget(self.power_profile_canvas, stretch=1)
        # --- Slider row: axis toggle buttons + slider ---
        slider_row_layout = QHBoxLayout()
        slider_row_layout.setContentsMargins(0, 0, 0, 0)
        slider_row_layout.setSpacing(0)
        # 1. Axis toggle button container
        # Fixed width pushes the slider right to align with the graph's zero point
        axis_ctrl_container = QWidget()
        axis_ctrl_layout = QVBoxLayout(axis_ctrl_container)
        axis_ctrl_layout.setContentsMargins(5, 0, 5, 0)
        axis_ctrl_layout.setSpacing(2)
        self.btn_dist_mode = QRadioButton("📏 Dist")
        self.btn_time_mode = QRadioButton("⏱️ Time")
        # Font and style
        mode_style = "QRadioButton { font-size: 10px; font-weight: bold; color: #CCCCCC; }"
        self.btn_dist_mode.setStyleSheet(mode_style)
        self.btn_time_mode.setStyleSheet(mode_style)
        self.btn_dist_mode.setChecked(True)
        axis_ctrl_layout.addWidget(self.btn_dist_mode)
        axis_ctrl_layout.addWidget(self.btn_time_mode)
        # Adjust this value to align the slider's left edge with the graph zero point
        # ~80-90 px typically aligns it flush with the Y-axis label margin
        axis_ctrl_container.setFixedWidth(85) 
        slider_row_layout.addWidget(axis_ctrl_container)
        # 2. Slider widget
        slider_row_layout.addWidget(self.dist_slider)
        # Small right spacing aligns the slider end with the graph right margin
        slider_row_layout.addSpacing(85) 
        left_layout.addLayout(slider_row_layout)

        # --- Right pane: control panel ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        
        # 1. Record selector panel
        self.record_selector = self.create_record_selector_panel()
        right_layout.addWidget(self.record_selector, stretch=1)
        
        # 2. Course map
        self.map_group = QGroupBox("Course Map")
        map_layout = QVBoxLayout(self.map_group)
        map_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        map_layout.addWidget(self.course_map)
        right_layout.addWidget(self.map_group)
        
        # 3. Action panel
        self.action_panel = self.create_action_panel()
        right_layout.addWidget(self.action_panel)

        # Splitter setup
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(left_container)
        self.main_splitter.addWidget(right_container)
        self.main_splitter.setStretchFactor(0, 1)  # left pane stretches
        self.main_splitter.setStretchFactor(1, 0)  # right pane holds fixed width
        right_container.setFixedWidth(370) 
        self.setCentralWidget(self.main_splitter)

    def _connect_signals(self):
        """Connect all signals to their handler slots."""
        # 1. Centralise redraw requests through the debounce timer
        # Any model change triggers a timer-debounced repaint request
        self.record_model.dataChanged.connect(self._request_update)
        self.record_model.dataChanged.connect(lambda: self.trigger_pane_update.emit())

        # 2. Bulk selection buttons
        # Fire a single repaint request after bulk processing, rather than blocking signals
        def handle_bulk_selection(state):
            """Set all records selected or deselected, then request a single repaint."""
            self.record_model.set_all_records_selected(state)
            self._request_update()  # one repaint after all updates

        self.btn_show_all.clicked.connect(lambda: handle_bulk_selection(True))
        self.btn_hide_all.clicked.connect(lambda: handle_bulk_selection(False))

        # --- Remaining signal connections ---
        self.snapshot_button.clicked.connect(self.handle_create_snapshot)
        self.trigger_pane_update.connect(self.update_export_pane)
        self.update_timer.timeout.connect(self.update_power_profile)
        
        self.btn_dist_mode.toggled.connect(self.update_axis_mode)
        self.btn_time_mode.toggled.connect(self.update_axis_mode)

        self.dist_slider.valueChanged.connect(self.sync_cursor_from_slider)
        self.main_splitter.splitterMoved.connect(self._request_update)
        self.list_view.activated.connect(self._request_update)
        if self.list_view.selectionModel():
            self.list_view.selectionModel().selectionChanged.connect(self._request_update)

        self.export_button.clicked.connect(self.handle_export)
        self.check_config_button.clicked.connect(self.handle_check_config)
        self.launch_nav_button.clicked.connect(self.handle_launch_navigator)

    def update_axis_mode(self):
        """Switch the graph between distance and time axis modes and re-sync the cursor."""
        is_time = self.btn_time_mode.isChecked()
        self.power_profile_canvas.set_axis_mode(is_time)  # redraws the graph
        self.sync_cursor_from_slider(self.dist_slider.value())  # sync cursor position

    def _request_update(self):
        """Schedule a canvas repaint; reset the timer on each call (debounce)."""
        self.update_timer.stop() 
        self.update_timer.start(50)

    def _update_distributed_huds(self, stats, dist_m, x_val): 
        """Pass stats and position to the canvas for the HUD update."""
        self.power_profile_canvas.current_stats = stats
        self.power_profile_canvas.current_dist = dist_m
        # Store the correct X coordinate on the graph
        self.power_profile_canvas.current_cursor_x = x_val 
        
        self.power_profile_canvas.update()

    def create_record_selector_panel(self):
        """Build and return the Strategy Record Selector panel with Show All, Hide All, and Snapshot buttons."""
        group = QGroupBox("Strategy Record Selector")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(2)

        # --- Three buttons in a single horizontal row ---
        btn_layout = QHBoxLayout()
        
        self.btn_show_all = QPushButton("Show All")
        self.btn_hide_all = QPushButton("Hide All")
        # Same box model as action-button (see toggle-button in _init_styles)
        # so these two line up exactly with Snapshot -- a plain QPushButton
        # renders with native chrome that a stylesheet button doesn't share.
        self.btn_show_all.setProperty("class", "toggle-button")
        self.btn_hide_all.setProperty("class", "toggle-button")
        self.btn_show_all.setCursor(Qt.PointingHandCursor)
        self.btn_hide_all.setCursor(Qt.PointingHandCursor)
        # Snapshot button on the right
        self.snapshot_button = QPushButton("📸 Snapshot")
        self.snapshot_button.setProperty("class", "action-button")
        self.snapshot_button.setCursor(Qt.PointingHandCursor)

        # Order: Show All -> Hide All -> Snapshot
        btn_layout.addWidget(self.btn_show_all)
        btn_layout.addWidget(self.btn_hide_all)
        btn_layout.addWidget(self.snapshot_button)
        
        layout.addLayout(btn_layout)

        # Column header row (small, subdued -- matches the app's existing
        # color:#999999/9px label convention), aligned to the delegate's columns
        # and scrolled in lockstep with the list so headers stay over their
        # column when the row content is scrolled horizontally.
        self.list_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        layout.addWidget(self._build_record_list_header())

        # List widget
        layout.addWidget(self.list_view, stretch=1)

        return group

    def _build_record_list_header(self) -> QWidget:
        """Build a clipped strip showing RecordListDelegate's full column titles on a
        rising diagonal (see RecordListHeader), aligned to its columns.

        The header content is wider than the visible strip and is shifted left
        by the list's horizontal scrollbar value, so it scrolls with the rows
        instead of staying fixed while the row content slides underneath it.
        """
        delegate = self.list_view.itemDelegate()
        content = RecordListHeader(delegate)

        viewport = QWidget()
        viewport.setFixedHeight(content.height())
        content.setParent(viewport)
        content.move(-self.list_view.horizontalScrollBar().value(), 0)

        self.list_view.horizontalScrollBar().valueChanged.connect(lambda v: content.move(-v, 0))

        # Keep references alive and reachable (Qt parent-child ownership already
        # keeps them alive, but this makes intent explicit and aids debugging).
        self._record_header_content = content
        self._record_header_viewport = viewport
        return viewport

    def create_action_panel(self):
        """Build and return the action panel QGroupBox with Check, Export, and Go buttons."""
        self.action_panel = QGroupBox("Selected Strategy Record Actions")
        layout = QVBoxLayout(self.action_panel)
        layout.setContentsMargins(6, 2, 6, 2)
        btn_layout = QHBoxLayout()

        self.check_config_button = QPushButton("🔍 Check")
        self.export_button = QPushButton("📤 Export")
        self.launch_nav_button = QPushButton("🏁 Go")

        # Common button settings
        for btn in [self.check_config_button, self.export_button, self.launch_nav_button]:
            btn.setProperty("class", "action-button") 
            btn.setCursor(Qt.PointingHandCursor)
        
        # Add to layout
        btn_layout.addWidget(self.check_config_button)
        btn_layout.addWidget(self.export_button)
        btn_layout.addWidget(self.launch_nav_button)

        layout.addLayout(btn_layout)
        return self.action_panel

    def load_full_record_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Load the full JSON for record, unpack compressed fields, and re-run
        the simulation if output data is absent. Raises on any failure (stale
        parquet index pointing at a deleted/moved file, a corrupt/truncated
        JSON, a simulator key no longer in SIMULATOR_REGISTRY, a malformed
        course/physics section, ...) rather than logging and returning an
        incomplete record -- a record downstream code (_rebuild_data_cache)
        structurally assumes is fully loaded is itself a real bug, not
        something to route around."""
        # 1. Resolve the JSON file path
        strategy_set_dir, run_set_id = record.get('strategy_set_dir'), record.get('run_set_id')
        n_seg, seed = record.get('N_seg_file'), record.get('Seed_file')
        if strategy_set_dir is None or run_set_id is None or n_seg is None or seed is None:
            raise ValueError(
                f"record is missing strategy_set_dir/run_set_id/N_seg_file/Seed_file "
                f"(full_path={record.get('full_path', '?')})"
            )
        f_path = find_strategy_json_path(strategy_set_dir, run_set_id, n_seg, seed)

        # 2. Load raw JSON
        with open(f_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        # 3. Unpack compressed data packets
        # This ensures arrays like 'v_limit_list' are restored in course_profile
        from core.data_manager import unpack_input_data
        full_data = unpack_input_data(full_data)

        # 4. Merge the unpacked data into the record dict
        record.update(full_data)

        # --- Re-run simulation only when output data is absent ---
        out = record.get('output', {})
        if 'data' not in out:
            pure_input = record['input']
            reconstructed_output = record['output']

            # This reconstructs plot data by re-simulating the stored
            # strategy with the simulator that actually produced it
            # (input.settings.engine.simulator, resolved below) --
            # git_state.commit_hash/is_dirty still matter, though: even
            # the *same* registered kernel's own code can have changed
            # since this strategy was generated. See core.git_info's
            # module docstring.
            git_state = pure_input.get('git_state', {})
            repro_warning = check_reproducibility(
                git_state.get('commit_hash'), git_state.get('is_dirty')
            )
            if repro_warning:
                logger.warning(repro_warning)

            simulator_spec = resolve_simulator(pure_input['settings']['engine']['simulator'])

            c_p = pure_input['data']['course_profile']
            p_s = pure_input['settings']['physical']
            q_s = pure_input['settings']['physiological']
            run_s = pure_input['settings']['run']

            # Build physics params from the restored arrays. model_construct(),
            # not model_validate() -- see core.physics_overrides' module
            # docstring for why an already-validated strategy JSON's own
            # settings are reconstructed without re-running validators.
            physical_settings = simulator_spec.physical_param_model.model_construct(**p_s)
            course = build_course_profile(c_p)
            course = simulator_spec.recompute_course_physics(course, physical_settings)
            physics = simulator_spec.build_physics_params(
                physical_settings,
                simulator_spec.physiological_param_model.model_construct(**q_s),
                RunSettings(**run_s),
                course,
            )

            strategy_data = reconstructed_output['results']['strategy']
            power_blocks = PowerBlocks(
                power=np.array(strategy_data['target_power_list']),
                length=np.array(strategy_data['target_length_list'])
            )

            sim_res = simulator_spec.kernel(
                0.0, power_blocks, physics, True, False, True
            )

            out['data'] = {
                'trace': {
                    'time_s_list': sim_res.t_traj.tolist(),
                    'distance_p_m_list': sim_res.x_traj.tolist(),
                    'speed_mps_list': sim_res.v_traj.tolist(),
                    'actual_p_w_list': sim_res.p_traj.tolist(),
                    'current_w_prime_j_list': sim_res.w_traj.tolist(),
                    'wind_v_apparent_mps_list': sim_res.v_w_app_traj.tolist(),
                    'wind_yaw_deg_list': sim_res.psi_w_app_traj.tolist()
                }
            }

        record['is_index_only'] = False
        return record

    @Slot(QPoint)
    def show_context_menu(self, position):
        """Show a left-click context menu offering Create Design, Show CLI Arguments, and (for seed-0 records) Remove Design."""
        index = self.list_view.indexAt(position)
        if not index.isValid():
            return

        # 1. Highlight the clicked row immediately
        self.list_view.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
        
        # Use UserRole (STRATEGY_ENTITY_ROLE) to get the real dict, not the display proxy
        # This keeps the Designer in sync with the actual record object
        STRATEGY_ENTITY_ROLE = Qt.ItemDataRole.UserRole
        record = index.data(STRATEGY_ENTITY_ROLE) 
        
        if not record:
            return

        # --- Build context menu ---
        menu = QMenu()
        # Action items
        create_action = menu.addAction("🎨 Create Design")
        
        is_s0 = str(record['Seed_file']) == '0'
        remove_action = None
        if is_s0:
            menu.addSeparator()
            remove_action = menu.addAction("🗑️ Remove Design")

        menu.addSeparator()
        show_cli_action = menu.addAction("📋 Show CLI Arguments")

        # 2. Show menu
        action = menu.exec(self.list_view.viewport().mapToGlobal(position))

        # 3. Clear selection highlight
        self.list_view.clearSelection()

        # 4. Execute selected action
        if action == create_action:
            # Pass the real dict reference so Designer changes are reflected immediately
            self.launch_designer_overlay(record)
        elif remove_action and action == remove_action:
            self.execute_remove_design(record)
        elif action == show_cli_action:
            self.show_record_detail(index)

    def execute_remove_design(self, record):
        """Delete the record's JSON file, remove it from the Parquet index, and re-initialise the viewer."""
        if not record:
            return

        # 1. Reconstruct the file path using the same rules as when saved
        strategy_set_dir = record['strategy_set_dir']
        run_set_id = record['run_set_id']
        n_seg = record['N_seg_file']
        seed = record['Seed_file']

        strategy_set_dir_path = os.path.join(BASE_STRATEGIES_DIR, strategy_set_dir)
        strategy_file_name = f"strategy_{run_set_id}_N{n_seg}_S{seed}.json"
        file_path = os.path.join(strategy_set_dir_path, strategy_file_name)

        # 2. Check physical file existence
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            # File may still exist in the Parquet index even if JSON is missing;
            # proceed to Parquet cleanup regardless
        else:
            # 3. Confirm with user
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Remove Design",
                f"Are you sure you want to PERMANENTLY remove this record?\n\n{strategy_file_name}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

            # 4. Delete the physical file
            try:
                os.remove(file_path)
                logger.info("Deleted JSON: %s", file_path)
            except Exception as e:
                logger.error("Failed to delete JSON file: %s", e)

        # 5. Remove from Parquet index
        try:
            self.remove_from_index_parquet(record, strategy_set_dir_path)
        except Exception as e:
            logger.error("Failed to update Parquet index: %s", e)

        # 6. Re-initialise to reflect changes in the UI
        logger.info("Re-initializing Viewer...")
        self.initialize()

    def remove_from_index_parquet(self, record, strategy_set_dir_path):
        """Remove the entry matching record from the _index.parquet file in strategy_set_dir_path."""
        import pandas as pd
        index_path = os.path.join(strategy_set_dir_path, "_index.parquet")
        
        if not os.path.exists(index_path):
            return

        df = pd.read_parquet(index_path)
        initial_count = len(df)

        target_id = record['run_set_id']
        target_n  = record['N_seg_file']
        target_s  = record['Seed_file']

        # str() on both sides avoids dtype mismatch issues (matches snapshot logic)
        mask = (
            (df["run_set_id"].astype(str) == str(target_id)) &
            (df["output.metadata.n_seg"].astype(str) == str(target_n)) &
            (df["output.metadata.seed"].astype(str) == str(target_s))
        )

        # Invert mask to keep only non-matching rows
        df_filtered = df[~mask]

        if len(df_filtered) < initial_count:
            df_filtered.to_parquet(index_path, index=False)
            logger.info("Updated Index: Removed 1 record. (Total: %d)", len(df_filtered))
        else:
            logger.warning("Record not found in Parquet index. ID=%s, N=%s, S=%s", target_id, target_n, target_s)

    @Slot(QModelIndex)
    def show_record_detail(self, index):
        """Copy CLI arguments to the clipboard and show the detail dialog."""
        record = self.record_model._records[index.row()]
        
        # Fail fast: treat missing keys as exceptions
        strategy_set_dir = record['strategy_set_dir']
        run_id  = record['run_set_id']
        n_seg   = record['N_seg_file']
        seed    = record['Seed_file']

        cli_args = f"{strategy_set_dir} {run_id} {n_seg} {seed}"
        
        # Copy to clipboard
        QApplication.clipboard().setText(cli_args)
        
        # Show detail window (stays open until closed)
        self.detail_win = RecordDetailDialog(cli_args, self)
        self.detail_win.show()

    def _ensure_selected_records_loaded(self) -> list:
        """Collect the currently selected + active records and fully load
        any still is_index_only, so a downstream _rebuild_data_cache (which
        assumes every record it's given is complete) never sees a partial
        one. Returns the resulting cache_records list."""
        records = self.record_model._records
        selected_records = [r for r in records if r.get('is_selected')]
        active_record = next((r for r in records if r.get('is_active')), None)
        # Include active record in cache even if not selected
        if active_record and active_record not in selected_records:
            cache_records = selected_records + [active_record]
        else:
            cache_records = selected_records

        for r in cache_records:
            if r.get('is_index_only', False):
                self.load_full_record_data(r)

        return cache_records

    @Slot()
    def update_power_profile(self):
        """Reload canvas data, update UI visibility, inject wind/heading data, and sync the cursor."""
        cache_records = self._ensure_selected_records_loaded()

        # Supply records to the canvas
        current_ids = [id(r) for r in cache_records]  # use object identity
        last_ids = getattr(self, "_last_ids_val", [])

        if current_ids != last_ids or not self.power_profile_canvas.data_cache:
            self.power_profile_canvas.set_target_records(cache_records)
            self._last_ids_val = current_ids

        # Locate the active record index within the canvas (1:1 with data_cache)
        active_idx = next((i for i, r in enumerate(self.power_profile_canvas.records) 
                          if r.get('is_active')), None)
        is_operable = (
            active_idx is not None and
            active_idx < len(self.power_profile_canvas.data_cache)
        )

        # 5. Update UI state
        self.map_group.setVisible(is_operable)
        self.dist_slider.setEnabled(is_operable)
        
        if is_operable:
            try:
                # active_idx was already validated by is_operable above
                r_active = self.power_profile_canvas.records[active_idx]
                current_data = self.power_profile_canvas.data_cache[active_idx]
                
                # Retrieve physics data
                input_data = r_active['input']
                cp = input_data['data']['course_profile']
                physical = input_data['settings']['physical']

                # Inject heading data into cache for slider synchronisation
                current_data['HEADING'] = np.array(cp['heading_deg_list'])
                current_data['HEADING_DIST'] = np.array(cp['distance_p_m_list'])

                self.current_analysis_data = current_data

                # Wire data to widgets. .get(..., 0.0): a simulator with no
                # wind model (e.g. core.simulators.sim_stub) has neither
                # field on its own PhysicalSettings at all -- 0.0/0.0 is
                # the same "no wind" state as an explicit zero, not a
                # distinct case worth a different message.
                self.course_map.set_true_wind(physical.get('wind_speed', 0.0), physical.get('wind_direction', 0.0))
                # current_data['CDA_RATIOS'] (eidos.lib.power_profile_canvas.
                # data_mixin._rebuild_data_cache, via PROFILE_KEYS) is always
                # a real key -- an empty array when input.data.cda_ratios is
                # absent from the strategy JSON, never actually missing.
                # CourseMapWidget treats an empty array as "skip the
                # CdA-dependent drawing", not a crash.
                self.course_map.set_wind_data(
                    current_data,
                    current_data['CDA_RATIOS']
                )
                self.course_map.set_course_latlon(
                    current_data['LAT'], 
                    current_data['LON'], 
                    current_data['COURSE_DIST']
                )
            except Exception as e:
                logger.error("CRITICAL ERROR in update_power_profile: %s", e)

        self.is_operable = is_operable

        # 6. Repaint and sync cursor
        self.power_profile_canvas.update()
        if is_operable:
            self.sync_cursor_from_slider(self.dist_slider.value())

    @Slot()
    def handle_create_snapshot(self):
        """Copy selected records and their index rows into a new timestamped snapshot directory."""
        selected_records = [r for r in self.record_model._records if r.get('is_selected', False)]
        if not selected_records: return

        new_dir_name = datetime.now().strftime("S%Y%m%d_%H%M%S")
        new_strategy_set_dir = os.path.join(BASE_STRATEGIES_DIR, new_dir_name)

        try:
            os.makedirs(new_strategy_set_dir, exist_ok=False)
            snapshot_rows = []

            for record in selected_records:
                # 1. Resolve the full home directory for this record
                relative_strategy_set_dir = record.get('strategy_set_dir')
                full_home_dir = os.path.join(BASE_STRATEGIES_DIR, relative_strategy_set_dir)
                home_index_path = os.path.join(full_home_dir, "_index.parquet")

                # 2. Search keys available in the Viewer's record
                target_n  = record.get('N_seg_file')
                target_s  = record.get('Seed_file')
                target_id = record.get('run_set_id')

                # 3. Copy JSON file to snapshot directory
                src_json = find_strategy_json_path(full_home_dir, target_id, target_n, target_s)

                if src_json and os.path.exists(src_json):
                    shutil.copy2(src_json, os.path.join(new_strategy_set_dir, os.path.basename(src_json)))

                    # 4. Extract matching row from home index
                    if os.path.exists(home_index_path):
                        home_df = pd.read_parquet(home_index_path)
                        
                        # String conversion avoids dtype mismatch
                        mask = (
                            (home_df["run_set_id"].astype(str) == str(target_id)) &
                            (home_df["output.metadata.n_seg"].astype(str) == str(target_n)) &
                            (home_df["output.metadata.seed"].astype(str) == str(target_s))
                        )
                        
                        hit = home_df[mask]
                        if not hit.empty:
                            snapshot_rows.append(hit)

            # 5. Save snapshot index
            if snapshot_rows:
                snapshot_df = pd.concat(snapshot_rows, ignore_index=True)
                snapshot_df.to_parquet(os.path.join(new_strategy_set_dir, "_index.parquet"))
                QMessageBox.information(self, "Snapshot", f"Created: {new_dir_name}")
            else:
                raise ValueError(f"Index extraction failed. Path: {home_index_path}\nSearch key: N={target_n}, S={target_s}")

            if hasattr(self, 'filter_pane'):
                self.filter_pane.refresh_history_lists()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {str(e)}")

    @Slot()
    def handle_export(self):
        """On button press: run the export and show the raw log in a popup dialog."""
        # Run export and capture log output
        execution_log = self.start_sync_export()
        
        if not execution_log:
            return

        # Show results dialog
        self.export_dialog = ExportReportDialog("Detailed Export Log", execution_log, self)
        self.export_dialog.exec()

    @Slot()
    def update_export_pane(self):
        """Refresh the action panel state by reading the latest model state directly."""
        all_records = self.record_model._records
        selected_records = [r for r in all_records if r.get('is_selected', False)]
        
        has_active = any(r.get('is_active', False) for r in all_records)
        
        # Logical state
        is_any = (len(selected_records) >= 1)
        is_visible = has_active  # visibility follows active record presence

        # --- Visibility control ---
        if hasattr(self, 'action_panel'):
            self.action_panel.setVisible(is_visible)

        self.dist_slider.setVisible(is_visible)

        # --- Button and guide text control ---
        self.snapshot_button.setEnabled(is_any)

        if is_visible:
            self.export_button.setEnabled(True)
            self.snapshot_display.setText("Move the slider to see details.")
        else:
            self.export_button.setEnabled(False)
            self.snapshot_display.setText("Select records and an active target for details.")
            if hasattr(self, 'power_profile_canvas'):
                self.power_profile_canvas.set_cursor_position(None)

    @Slot()
    def start_sync_export(self):
        """Run the FIT/ZWO/PDF export for the active record and return the captured log."""
        all_records = self.record_model._records
        record = next((r for r in all_records if r.get('is_active', False)), None)
        if not record: return "No active record selected."

        strategy_set_dir = record.get('strategy_set_dir')
        runset_id = record.get('run_set_id')

        self.export_button.setEnabled(False)
        QApplication.processEvents()

        log_content = ""  # accumulates stdout/stderr output
        try:
            # See _launch_script's docstring for why -m (module invocation)
            # is used instead of a resolved sibling file path.
            command = [sys.executable, "-m", "eidos.apps.exporter", strategy_set_dir, runset_id, str(record['N_seg_file']), str(record['Seed_file'])]
            
            # Run exporter and capture output
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # The exporter's own logging already carries timestamp/level
            # tags (see core/logging_setup.py), so stdout reads fine on its
            # own -- no extra "STDOUT" header needed. stderr is normally
            # empty here (logging is routed to stdout via configure_logging),
            # so only show it if something outside logging wrote to it.
            log_content = result.stdout
            if result.stderr:
                log_content += f"\n--- stderr ---\n{result.stderr}"

        except subprocess.CalledProcessError as e:
            log_content = f"🛑 Export Failed with Exit Code {e.returncode}\n\n{e.stdout}"
            if e.stderr:
                log_content += f"\n--- stderr ---\n{e.stderr}"
        except Exception as e:
            log_content = f"🛑 Unexpected Error:\n{str(e)}"

        finally:
            self.export_button.setEnabled(True)
            return log_content

    @Slot(int)
    def sync_cursor_from_slider(self, value):
        """Synchronise all displays based on the active record's physics at the slider position."""
        # 1. Get current analysis data
        data = getattr(self, 'current_analysis_data', None)
        if data is None: return

        # DISTANCE/COURSE_DIST/TIME are always real keys in a data_cache
        # entry (eidos.lib.power_profile_canvas.data_mixin._rebuild_data_cache,
        # via PROFILE_KEYS) -- an empty array when output.data.trace hasn't
        # been materialised yet, never actually missing. .size == 0, not a
        # truthiness check, since `not <nonempty ndarray>` raises a NumPy
        # ambiguous-comparison error.
        dists = data['DISTANCE']
        if dists.size == 0: dists = data['COURSE_DIST']
        if dists.size == 0: return

        # 2. Map slider value to physical coordinate (time or distance mode)
        ratio = value / 1000.0
        is_time_mode = self.btn_time_mode.isChecked()

        if is_time_mode:
            times = data['TIME']
            if times.size == 0: return
            x_val = ratio * times[-1]
            dist_m = np.interp(x_val, times, dists)
        else:
            x_val = ratio * dists[-1]
            dist_m = x_val

        # 3. Update cursor line on canvas
        self.power_profile_canvas.set_cursor_position(x_val)

        # 4. Update HUD and map at the current distance (get_data_at_dist
        # resolves the active record's index internally -- see its own
        # docstring -- so there's nothing to locate here first)
        stats = self.power_profile_canvas.get_data_at_dist(dist_m)

        if stats:
            self._update_distributed_huds(stats, dist_m, x_val)
            self.course_map.update_cursor(dist_m)

    @Slot()
    def handle_check_config(self):
        """Show the resolved config file path and a formatted settings report for the active record."""
        all_records = self.record_model._records
        record = next((r for r in all_records if r.get('is_active', False)), None)
        if not record: return "No active record selected."
        
        # --- Resolve path via shared config module ---
        try:
            from core.io_config import find_strategy_json_path
            f_path = find_strategy_json_path(
                record['strategy_set_dir'],
                record['run_set_id'],
                record['N_seg_file'],
                record['Seed_file']
            )
        except Exception:
            # The JSON file itself may be gone from disk (moved/deleted by
            # hand) even though the record is otherwise complete -- a
            # best-effort path display isn't worth failing this action over.
            f_path = record.get('full_path', 'Path could not be resolved')

        report = [
            "=" * min(len(f_path), 80), "",
            f"{f_path}",
            "=" * min(len(f_path), 80), "",
        ]

        def dict_to_report(d, indent=0):
            """Recursively format dict d into report lines with the given indentation level."""
            exclude_keys = ["data", "is_selected", "full_path", "run_set_id",
                            "N_seg_file", "Seed_file", "run_set_id", "input"]
            for key, value in d.items():
                if key in exclude_keys: continue
                if isinstance(value, dict):
                    report.append("  " * indent + f"[{key}]")
                    dict_to_report(value, indent + 1)
                else:
                    report.append("  " * indent + f"{key:<28}: {value}")

        # input.versions holds a milestone label per role (data_manager/
        # simulator/optimizer), but a label only means something paired
        # with WHICH one it's a milestone of -- that key lives separately,
        # in input.settings.engine (see eidos.apps.generator.
        # create_json_input_dict's own docstring). Composed here, display-
        # only, as "role—key" -- data_manager has no engine key of its own
        # (only one implementation exists), so it falls back to its plain
        # role name.
        input_data = dict(record.get("input", {}))
        versions = input_data.get("versions")
        engine = input_data.get("settings", {}).get("engine", {})
        if isinstance(versions, dict):
            composed_versions = {}
            for role, milestone in versions.items():
                key = engine.get(role)
                label = f"{role}—{key}" if key is not None else role
                composed_versions[label] = milestone
            input_data["versions"] = composed_versions

        dict_to_report(input_data)
        
        self.config_dialog = ExportReportDialog("Record Configuration Profile", "\n".join(report), self)
        self.config_dialog.exec()

    @Slot()
    def handle_launch_navigator(self):
        """Locate exported FIT files for the active record and open the StrategySelectorDialog."""
        all_records = self.record_model._records
        record = next((r for r in all_records if r.get('is_active', False)), None)
        if not record: return "No active record selected."

        strategy_set_dir_name = str(record['strategy_set_dir'])
        run_set_id   = str(record['run_set_id'])
        n_seg        = str(record['N_seg_file'])
        seed         = str(record['Seed_file'])

        # exports / {strategy_set_dir} / {run_set_id}_N{n}_S{s} / fit
        sub_dir = f"{run_set_id}_N{n_seg}_S{seed}"
        target_dir = os.path.join(BASE_EXPORTS_DIR, strategy_set_dir_name, sub_dir, "fit")

        fit_files = []
        if os.path.exists(target_dir):
            fit_files = [f for f in os.listdir(target_dir) if f.endswith(".fit")]
            fit_files.sort()

        # Build the run_info tuple in the format StrategySelectorDialog expects
        run_info = (strategy_set_dir_name, run_set_id, n_seg, seed)

        dialog = StrategySelectorDialog(fit_files, run_info, self)
        dialog.exec()

    def launch_designer_overlay(self, record):
        """Launch the StrategyDesigner overlay for record, loading full data first if needed."""
        if not record:
            return

        # Same load-on-demand logic as update_power_profile. load_full_record_data
        # raises on failure (see its own docstring) rather than leaving record
        # incomplete, so no post-call completeness guard is needed here.
        if record.get('is_index_only', False):
            self.load_full_record_data(record)

        from eidos.apps.designer import StrategyDesignerController
        
        # Pass the fully loaded record to the designer
        self.designer_controller = StrategyDesignerController(
            record, 
            self.power_profile_canvas, 
            self
        )
        
        self.designer_controller.connect_canvas()

