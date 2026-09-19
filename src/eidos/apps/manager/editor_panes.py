"""
eidos.apps.manager.editor_panes -- ConfigurationEditorPane and FilterEditorPane.

The two large form-editing panels: ConfigurationEditorPane builds a
dynamic form from PhysicalSettings/PhysiologicalSettings/OptimizerParams/
RunValidationModel/EngineValidationModel (physical/physiological/optimizer-
tuning/run/engine settings for strategy generation) -- PhysicalSettings/
PhysiologicalSettings are resolved per the currently-selected
Engine.simulator, and OptimizerParams per the currently-selected
Engine.optimizer (see ConfigurationEditorPane._get_schema_map), not fixed
models -- each core.simulators.SIMULATOR_REGISTRY/eidos.lib.optimizer.
OPTIMIZER_REGISTRY entry defines its own field set. OptimizerParams is
also the one dynamic section whose data lives NESTED in config_data
(config_data['Engine']['optimizer_params'], not a top-level
config_data['OptimizerParams'] key -- see
ConfigurationEditorPane._section_path) since that's where
Engine.optimizer_params actually lives in the JSON. FilterEditorPane
builds one from eidos.lib.strategy_selector's TYPE_MAP (passed in as a
constructor argument, not imported directly here -- see
FilterEditorPane.__init__).
"""

import json
import logging
import os
import re
from typing import Any, Dict, Literal, Optional, get_args, get_origin

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from core.io_config import BASE_STRATEGIES_DIR
from core.schema import (
    RunValidationModel,
    field_display_label,
    gui_decimals_from_field,
    preset_value_from_field,
)
from eidos.apps.manager.constants import SECTION_MAP
from eidos.apps.manager.file_managers import ConfigFileManager
from eidos.apps.manager.validators import (
    FixedDoubleSpinBox,
    FixedIntSpinBox,
    OptionalFloatValidator,
    OptionalIntValidator,
)

logger = logging.getLogger(__name__)


def _set_line_edit_text(widget: QLineEdit, text: str) -> None:
    """Set text and reset the cursor to position 0.

    setText() alone leaves the cursor at the end, and Qt scrolls the
    viewport to keep it visible -- so a long value (a path, a glob
    pattern) shows only its tail in a narrow box instead of the more
    identifying head.
    """
    widget.setText(text)
    widget.setCursorPosition(0)


class ConfigurationEditorPane(QWidget):
    """
    Editor pane for a single JSON configuration file (rider, environment, or run settings).

    Presents the settings as a validated form; emits editor_closed when the user
    dismisses the pane.
    """
    editor_closed = Signal()

    # 'Engine' is the JSON section name; not 'EngineSettings' like the
    # other three. PhysicalSettings/PhysiologicalSettings are resolved
    # dynamically per the currently selected Engine.simulator -- see
    # _get_schema_map, not a class-level constant here (each core.
    # simulators.SIMULATOR_REGISTRY entry defines its own field set).
    # 'Engine' itself isn't here either, even though its own schema is
    # fixed regardless of registry selection: EngineValidationModel now
    # lives in eidos.lib.optimizer (see that module's docstring), so
    # importing it at class-body/module level would defeat the whole
    # point of _get_schema_map's own lazy eidos.lib.optimizer import
    # below -- see that method's docstring.
    _STATIC_SCHEMA_MAP = {
        'RunSettings': RunValidationModel,
    }

    # Engine fields select a registry key rather than accepting free
    # text -- populated from core.simulators.SIMULATOR_REGISTRY /
    # eidos.lib.optimizer.OPTIMIZER_REGISTRY, imported lazily inside
    # _build_form_ui() to keep numba out of the manager process until a
    # config is actually opened for editing (manager otherwise never
    # imports core.simulators/eidos.lib.optimizer -- generator.py runs as a
    # subprocess precisely to keep those heavy imports out of the GUI
    # process; see constants.py's SCRIPT_MODULE_MAP docstring).
    _ENGINE_REGISTRY_FIELDS = ('simulator', 'optimizer')

    _DECIMAL_PLACES: Dict[str, int] = {} 

    def __init__(self, file_manager: ConfigFileManager, parent=None):
        """Initialize with the given file_manager and build the editor UI."""
        super().__init__(parent)
        self.file_manager = file_manager
        self.current_file_path: Optional[str] = None
        self.config_data: Dict[str, Any] = {}
        self.widgets: Dict[str, Dict[str, QWidget]] = {} 
        self.tab_widget: Optional[QTabWidget] = None 

        self._setup_ui()
        self.title_label.setText("Configuration Editor")

    def _setup_ui(self):
        """Build the main layout with a tabbed form/JSON editor and Save/Cancel buttons."""
        main_layout = QVBoxLayout(self)
        self.title_label = QLabel("Configuration Editor")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        
        self.tab_widget = QTabWidget()
        
        self.form_scroll_widget = QWidget() 
        self.form_layout = QVBoxLayout(self.form_scroll_widget)
        self.form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._build_form_ui() 
        
        self.form_scroll_area = QScrollArea()
        self.form_scroll_area.setWidgetResizable(True) 
        self.form_scroll_area.setWidget(self.form_scroll_widget)

        self.json_editor = QPlainTextEdit()
        editor_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        editor_font.setPointSize(14) 
        self.json_editor.setFont(editor_font)

        self.tab_widget.addTab(self.form_scroll_area, "GUI Form Editor")
        self.tab_widget.addTab(self.json_editor, "Raw JSON Editor")
        self.json_editor.textChanged.connect(self._sync_json_to_form) 
        
        button_layout = QHBoxLayout()
        self.save_and_back_button = QPushButton("💾 Save Config and Close")
        self.cancel_and_back_button = QPushButton("❌ Cancel")
        
        button_layout.addWidget(self.save_and_back_button)
        button_layout.addWidget(self.cancel_and_back_button)
        
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.tab_widget) 
        main_layout.addLayout(button_layout)
        
        self.save_and_back_button.clicked.connect(self._save_and_back)
        self.cancel_and_back_button.clicked.connect(self._cancel_and_back)

    # Muted-secondary-text style for the raw field_name shown alongside
    # each row's pretty title -- matches the convention already
    # established in eidos.apps.analyzer.widgets (e.g. its
    # sobol_n_label/morris_r_label), reused here rather than inventing a
    # new gray/size pairing.
    _RAW_KEY_STYLE = "color: #999999; font-size: 9px;"

    # Marks a field _initialize_form_from_data just seeded from
    # core.schema.preset_value_from_field rather than an actual value
    # from config_data -- same amber this app already uses elsewhere
    # (window.py's TTManagerGUI.LOG_COLOR_WARNING) for "this needs your
    # attention", not the app's own arbitrary color. Cleared the instant
    # the user
    # edits the field themselves (see the widget.*Changed.connect calls
    # in _build_form_ui below) -- a preset is a starting point to review,
    # not a value the user is meant to leave unexamined.
    _PRESET_VALUE_STYLE = "border: 1px solid #e5c07b; color: #e5c07b;"

    def _clear_preset_highlight(self, widget: QWidget):
        """Drop _PRESET_VALUE_STYLE the instant the user actually edits
        this field themselves. Connected to each dynamic-section widget's
        own change signal in _build_form_ui, downstream of that same
        signal's existing _sync_form_to_json connection -- both fire on
        every real user edit, never on _initialize_form_from_data's own
        programmatic sets (those run under widget.blockSignals(True))."""
        widget.setStyleSheet("")

    def _build_field_label(self, schema_class, field_name: str) -> QLabel:
        """Build one form row's label: schema.field_display_label()'s
        pretty title, plus the raw field_name in small muted text right
        after it -- so this row can still be correlated with the same
        key in the Raw JSON Editor tab, which shows raw field names with
        no title substitution."""
        title = field_display_label(schema_class, field_name)
        return QLabel(
            f'{title}: <span style="{self._RAW_KEY_STYLE}">({field_name})</span>'
        )

    def _get_schema_map(self) -> Dict[str, Any]:
        """Build this pane's full section_name -> Pydantic model map, with
        PhysicalSettings/PhysiologicalSettings resolved against whichever
        simulator self.config_data's own Engine.simulator currently names
        (falling back to core.simulators.DEFAULT_SIMULATOR_KEY if absent or
        not a registered key -- e.g. a brand-new config being created from
        scratch, or a malformed one, should still open with a sensible
        default form rather than crashing), and OptimizerParams resolved
        the same way against Engine.optimizer / eidos.lib.optimizer.
        DEFAULT_OPTIMIZER_KEY.

        Reads self.config_data (the backing dict this pane keeps in sync
        with the JSON editor / form widgets), not the Engine simulator/
        optimizer combo box widgets themselves -- this must work even
        before those widgets exist (the very first _build_form_ui() call,
        from __init__, when no section's widgets have been built yet).
        Imported lazily, same reasoning as the Engine combo box's own
        SIMULATOR_REGISTRY/OPTIMIZER_REGISTRY imports in _build_form_ui:
        keep numba out of the manager process until a config is actually
        opened for editing. EngineValidationModel itself is fetched here
        too (not from _STATIC_SCHEMA_MAP) purely for that same import-cost
        reason -- its own schema doesn't vary by registry selection like
        PhysicalSettings/PhysiologicalSettings/OptimizerParams do.
        """
        from core.simulators import DEFAULT_SIMULATOR_KEY, SIMULATOR_REGISTRY
        from eidos.lib.optimizer import (
            DEFAULT_OPTIMIZER_KEY,
            OPTIMIZER_REGISTRY,
            EngineValidationModel,
        )

        engine_data = self.config_data.get('Engine') or {}
        simulator_key = engine_data.get('simulator')
        if simulator_key not in SIMULATOR_REGISTRY:
            simulator_key = DEFAULT_SIMULATOR_KEY
        spec = SIMULATOR_REGISTRY[simulator_key]

        optimizer_key = engine_data.get('optimizer')
        if optimizer_key not in OPTIMIZER_REGISTRY:
            optimizer_key = DEFAULT_OPTIMIZER_KEY
        optimizer_spec = OPTIMIZER_REGISTRY[optimizer_key]

        return {
            'PhysicalSettings': spec.physical_param_model,
            'PhysiologicalSettings': spec.physiological_param_model,
            'OptimizerParams': optimizer_spec.param_model,
            'Engine': EngineValidationModel,
            **self._STATIC_SCHEMA_MAP,
        }

    # Every dynamic/static section's own data lives at
    # config_data[section_name] directly -- a single-element path -- EXCEPT
    # OptimizerParams, which lives nested at config_data['Engine']
    # ['optimizer_params'] (that's where Engine.optimizer_params actually
    # is in the JSON; there is no top-level config_data['OptimizerParams']
    # key). Centralised here so _prune_stale_dynamic_fields/
    # _initialize_form_from_data/_sync_form_to_json can all stay generic
    # loops over self.widgets.items() without each special-casing
    # OptimizerParams' own storage location.
    _SECTION_PATHS = {
        'OptimizerParams': ('Engine', 'optimizer_params'),
    }

    def _section_path(self, section_name: str) -> tuple:
        """Path of keys from config_data's own root down to section_name's
        own dict. See _SECTION_PATHS' comment for why this isn't always
        just (section_name,)."""
        return self._SECTION_PATHS.get(section_name, (section_name,))

    def _get_section_data(self, config_data: Dict[str, Any], section_name: str) -> Dict[str, Any]:
        """Read section_name's own dict out of config_data, walking
        _section_path(section_name). Returns the SAME (mutable) nested
        dict object if the whole path already exists, or a fresh, not-yet-
        linked-in empty dict if any step is missing/not a dict -- exactly
        the semantics config_data.get(section_name, {}) already had for
        every section before OptimizerParams' nesting existed. A caller
        that fills in the fresh-dict case must write it back via
        _set_section_data for the change to actually reach config_data."""
        node: Any = config_data
        for key in self._section_path(section_name):
            if not isinstance(node, dict):
                return {}
            node = node.get(key)
        return node if isinstance(node, dict) else {}

    def _set_section_data(self, config_data: Dict[str, Any], section_name: str, section_data: Dict[str, Any]):
        """Write section_data into config_data at _section_path(section_name),
        creating any missing intermediate dict (e.g. a from-scratch config
        with no 'Engine' key yet) along the way."""
        node = config_data
        path = self._section_path(section_name)
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = section_data

    def _ensure_section_data(self, config_data: Dict[str, Any], section_name: str) -> Dict[str, Any]:
        """Like _get_section_data, but creates any missing dict along
        _section_path(section_name) (via setdefault at each step) so the
        returned dict is always the REAL object already linked into
        config_data -- callers that mutate it in place (e.g.
        _initialize_form_from_data seeding a preset value) need this,
        unlike _get_section_data's read-only "fresh empty dict if missing"
        semantics."""
        node = config_data
        for key in self._section_path(section_name):
            node = node.setdefault(key, {})
        return node

    # Sections whose own field set can change out from under
    # self.config_data at runtime (a simulator/optimizer switch) and
    # therefore need _prune_stale_dynamic_fields/_initialize_form_from_data's
    # prune-then-refill treatment. RunSettings/Engine never do -- their
    # own schema is fixed regardless of which registry entries are
    # selected -- so they're deliberately excluded here even though they
    # go through the same generic _build_form_ui loop.
    _DYNAMIC_SECTIONS = ('PhysicalSettings', 'PhysiologicalSettings', 'OptimizerParams')

    def _prune_stale_dynamic_fields(self):
        """Drop any _DYNAMIC_SECTIONS key in self.config_data that has no
        corresponding widget in the just-rebuilt self.widgets -- call this
        right after _build_form_ui() rebuilds against a (possibly new)
        schema.

        Needed because _sync_form_to_json merges each section's new widget
        values ON TOP OF whatever was already in self.config_data for that
        section, rather than replacing it outright. That merge is harmless
        as long as a section's own field set never changes -- true for
        RunSettings/Engine always, but not for PhysicalSettings/
        PhysiologicalSettings/OptimizerParams: a simulator switch can
        shrink or reshape either of the first two (e.g. sim_kiritsubo's
        full 5-field PhysiologicalSettings -> sim_stub's field-less one,
        just the inherited cp/w_prime), and an optimizer switch can do
        the same to the third
        (opt_tenchi's 11-field TenchiParams -> opt_stub's field-less
        OptStubParams). Without pruning, a field the new schema no longer
        has would survive indefinitely in self.config_data, invisible in
        the form (no widget reads it) but still written to the saved
        JSON -- which then fails that section's own extra="forbid"
        validation the next time anything re-loads this config (e.g.
        eidos.apps.generator.load_config_jsons).

        Also refreshes self.json_editor's displayed text from the now-
        pruned self.config_data -- required, not cosmetic:
        _save_config_only() writes whatever self.json_editor.toPlainText()
        currently holds, NOT self.config_data directly, so pruning
        self.config_data alone would leave stale fields visible-gone in
        the GUI Form tab while still fully intact (and saved) in the Raw
        JSON Editor tab.
        """
        pruned_any = False
        for section_name in self._DYNAMIC_SECTIONS:
            section_data = self._get_section_data(self.config_data, section_name)
            if not section_data:
                continue
            valid_fields = set(self.widgets.get(section_name, {}))
            for stale_key in [k for k in section_data if k not in valid_fields]:
                del section_data[stale_key]
                pruned_any = True

        if pruned_any:
            # blockSignals() returns the PREVIOUS state -- restore that,
            # not unconditionally False: load_config() calls this method
            # from inside its own blockSignals(True)/(False) bracket, and
            # unconditionally unblocking here would re-enable signals
            # before load_config's own closing call runs.
            was_blocked = self.json_editor.blockSignals(True)
            self.json_editor.setPlainText(json.dumps(self.config_data, indent=4, ensure_ascii=False))
            self.json_editor.blockSignals(was_blocked)

    @Slot(str)
    def _on_registry_selection_changed(self, _new_key: str):
        """Rebuild whichever dynamic section(s) depend on the Engine field
        that just changed -- PhysicalSettings/PhysiologicalSettings for
        Engine.simulator, OptimizerParams for Engine.optimizer -- so the
        form always matches the currently selected registry entries' own
        field sets -- see core.simulators.SimulatorSpec.physical_param_model/
        physiological_param_model and eidos.lib.optimizer.OptimizerSpec.
        param_model, each of which can differ in shape between registry
        entries (e.g. core.simulators.sim_stub's minimal
        PhysiologicalSettings vs. sim_kiritsubo's full one; eidos.lib.
        optimizers.opt_stub's field-less OptStubParams vs. opt_tenchi's
        11-field TenchiParams). One shared slot, connected to both the
        simulator and optimizer combo boxes in _build_form_ui, since the
        rebuild sequence itself doesn't need to know which one fired --
        _get_schema_map re-resolves both PhysicalSettings/
        PhysiologicalSettings AND OptimizerParams from self.config_data's
        current Engine.simulator/Engine.optimizer every time regardless.

        _sync_form_to_json() first commits the new selection (and every
        other current widget value) into self.config_data -- the backing
        dict _get_schema_map reads from -- then _build_form_ui() tears
        down and rebuilds every section's widgets against the (possibly
        new) schema, and _initialize_form_from_data() repopulates them
        from that same self.config_data. Field values that still exist
        under the new simulator/optimizer (RunSettings/Engine always;
        PhysicalSettings/OptimizerParams fields whenever the new entry's
        own shape happens to share a name) survive the rebuild; fields the
        new entry doesn't have are simply absent from the rebuilt widget
        set.
        """
        self._sync_form_to_json()
        self._build_form_ui()
        self._prune_stale_dynamic_fields()
        self._initialize_form_from_data(self.config_data)

    def _build_form_ui(self):
        """Rebuild the GUI form from _get_schema_map(), creating grouped spinbox/lineedit widgets."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.widgets.clear()

        # OptimizerParams renders NESTED inside Engine's own box, as a
        # full-width row of Engine's own QFormLayout (see
        # _build_section_group_box's call below for 'Engine'), rather
        # than as a sibling top-level GroupBox like PhysicalSettings/
        # PhysiologicalSettings/RunSettings/Engine itself -- matching
        # config_data['Engine']['optimizer_params']'s real JSON nesting
        # (see _SECTION_PATHS). So OptimizerParams is skipped in this
        # top-level loop and handled specially inside the 'Engine'
        # iteration instead.
        schema_map = self._get_schema_map()
        for section_name, schema_class in schema_map.items():
            if section_name == 'OptimizerParams':
                continue

            group_box = self._build_section_group_box(section_name, schema_class)
            if group_box is None:
                continue

            if section_name == 'Engine':
                nested_box = self._build_section_group_box(
                    'OptimizerParams', schema_map.get('OptimizerParams'), title="Optimizer Params",
                )
                if nested_box is not None:
                    group_box.layout().addRow(nested_box)

            self.form_layout.addWidget(group_box)

    def _build_section_group_box(self, section_name: str, schema_class, title: Optional[str] = None) -> Optional[QGroupBox]:
        """Build one section's own QGroupBox (title bar + QFormLayout of
        field rows), registering its widgets into self.widgets[section_name]
        along the way. Returns None -- building nothing, registering an
        empty self.widgets[section_name] -- if schema_class has no fields
        at all (e.g. OptimizerParams resolved against eidos.lib.optimizers.
        opt_stub.OptStubParams, which has none): matches this method's own
        previous inline behavior of never attaching an empty section's box
        to the form. `title` defaults to section_name split at capitals
        ("PhysicalSettings" -> "Physical Settings"); OptimizerParams'
        own caller in _build_form_ui passes "Optimizer Params"
        explicitly anyway, though the same split would already produce
        it -- just being explicit at the one nested call site.
        """
        fields = getattr(schema_class, "model_fields", {})
        self.widgets[section_name] = {}
        if not fields:
            return None

        if title is None:
            title = re.sub(r'([A-Z])', r' \1', section_name).strip()
            title = title.replace('Settings', ' Settings')

        group_box = QGroupBox(title)
        form_layout = QFormLayout(group_box)

        for field_name, field_info in fields.items():
            field_type = field_info.annotation
            widget: QWidget | None = None

            # Extract ge/le (and gt/lt -- an exclusive bound still needs
            # to surface here, or a gt/lt-only field silently falls back
            # to the -1000000/1000000 default range below and shows
            # "None" in its own tooltip, even though a real bound
            # exists) for the spinbox range and tooltip. A field never
            # has both ge and gt (nor both le and lt) at once, so gt/lt
            # only apply here when the ge/le check didn't already set a
            # value.
            ge_val, le_val = None, None
            for m in field_info.metadata:
                if hasattr(m, 'ge'): ge_val = m.ge
                elif hasattr(m, 'gt'): ge_val = m.gt
                if hasattr(m, 'le'): le_val = m.le
                elif hasattr(m, 'lt'): le_val = m.lt

            # Label: pretty title (from core.schema's Field(title=...))
            # with the raw field_name shown small/muted alongside it,
            # so a row here can still be correlated with the same key
            # in the Raw JSON Editor tab.
            label_widget = self._build_field_label(schema_class, field_name)

            # Build widget by field type
            if field_type is float:
                widget = FixedDoubleSpinBox()
                decimals = gui_decimals_from_field(schema_class, field_name)
                widget.setDecimals(decimals)
                widget.setSingleStep(10**-decimals)
                widget.setMinimum(float(ge_val) if ge_val is not None else -1000000.0)
                widget.setMaximum(float(le_val) if le_val is not None else 1000000.0)
                
                self._DECIMAL_PLACES[f'{section_name}.{field_name}'] = decimals
                widget.valueChanged.connect(self._sync_form_to_json)
                widget.valueChanged.connect(lambda _v, w=widget: self._clear_preset_highlight(w))

            elif field_type is int:
                widget = FixedIntSpinBox()
                widget.setRange(
                    int(ge_val) if ge_val is not None else -1000000,
                    int(le_val) if le_val is not None else 1000000
                )
                widget.valueChanged.connect(self._sync_form_to_json)
                widget.valueChanged.connect(lambda _v, w=widget: self._clear_preset_highlight(w))

            elif field_type is str and section_name == 'Engine' and field_name in self._ENGINE_REGISTRY_FIELDS:
                if field_name == 'simulator':
                    from core.simulators import SIMULATOR_REGISTRY
                    registry_keys = sorted(SIMULATOR_REGISTRY)
                else:
                    from eidos.lib.optimizer import OPTIMIZER_REGISTRY
                    registry_keys = sorted(OPTIMIZER_REGISTRY)

                widget = QComboBox()
                widget.addItems(registry_keys)
                widget.currentTextChanged.connect(self._sync_form_to_json)
                # Rebuild whichever dynamic section(s) depend on this
                # field -- PhysicalSettings/PhysiologicalSettings for
                # simulator, OptimizerParams for optimizer -- see
                # _on_registry_selection_changed's own docstring.
                # Connected after _sync_form_to_json above so
                # self.config_data already reflects the new selection
                # by the time this runs.
                widget.currentTextChanged.connect(self._on_registry_selection_changed)
                if field_info.description:
                    widget.setToolTip(field_info.description)

            elif get_origin(field_type) is Literal:
                # Literal[...] fields render as a selection dropdown, e.g.
                # TenchiParams.de_strategy's fixed set of scipy.optimize.
                # differential_evolution strategy names.
                choices = [str(c) for c in get_args(field_type)]
                widget = QComboBox()
                widget.addItems(choices)
                widget.currentTextChanged.connect(self._sync_form_to_json)
                widget.currentTextChanged.connect(lambda _t, w=widget: self._clear_preset_highlight(w))
                if field_info.description:
                    widget.setToolTip(field_info.description)

                form_layout.addRow(label_widget, widget)
                self.widgets[section_name][field_name] = widget
                continue

            elif field_type is str:
                str_container = QWidget()
                str_layout = QHBoxLayout(str_container)
                str_layout.setContentsMargins(0, 0, 0, 0)
                str_layout.setSpacing(2)
                
                widget = QLineEdit()
                widget.textChanged.connect(self._sync_form_to_json)
                widget.textChanged.connect(lambda _t, w=widget: self._clear_preset_highlight(w))
                # QAbstractSpinBox (float/int fields above) fires
                # valueChanged -- and therefore clears the preset
                # highlight -- the moment the user presses Enter/tabs
                # away, even if they didn't actually change the number
                # (interpretText() re-commits the value on every such
                # commit). QLineEdit's own textChanged has no equivalent:
                # it only fires when the text content itself actually
                # differs. editingFinished (Enter or focus-out) is
                # QLineEdit's matching "user committed this field" signal,
                # so it gets the same highlight-clearing treatment here --
                # without this, a str field would stay highlighted forever
                # unless its text is literally retyped, unlike every
                # numeric field beside it.
                widget.editingFinished.connect(lambda w=widget: self._clear_preset_highlight(w))
                str_layout.addWidget(widget)
                
                if field_name in ['gpx_filename', 'cda_yaw_table_filename']:
                    select_btn = QToolButton()
                    select_btn.setText("▼")
                    str_layout.addWidget(select_btn)

                    def on_select_click(checked=False, f_n=field_name, tw=widget, btn=select_btn):
                        """Show a dropdown menu of available files and set the target line edit on selection."""
                        menu = QMenu(self)
                        menu.setStyleSheet("QMenu { menu-scrollable: 1; }")
                        menu.setMaximumHeight(400)
                        items = self.file_manager.get_available_gpx_files() if 'gpx' in f_n else \
                                self.file_manager.get_available_cda_tables()
                        if not items:
                            menu.addAction("No Files Found")
                        else:
                            for item in items:
                                action = menu.addAction(item)
                                action.triggered.connect(lambda _, v=item, t=tw: _set_line_edit_text(t, v))
                        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

                    select_btn.clicked.connect(on_select_click)
                
                # Tooltip for str fields
                if field_info.description:
                    widget.setToolTip(field_info.description)
                    
                form_layout.addRow(label_widget, str_container)
                self.widgets[section_name][field_name] = widget
                continue

            # Tooltip: aggregate description and range
            if widget:
                tip_elements = []
                if field_info.description:
                    tip_elements.append(field_info.description)
                if ge_val is not None or le_val is not None:
                    tip_elements.append(f"Range: [{ge_val} to {le_val}]")
                
                if tip_elements:
                    widget.setToolTip("\n".join(tip_elements))

                form_layout.addRow(label_widget, widget)
                self.widgets[section_name][field_name] = widget

        return group_box

    def _block_form_signals(self, block: bool):
        """Block or unblock signals on all form widgets to suppress redundant sync callbacks."""
        for section in self.widgets.values():
            for widget in section.values():
                widget.blockSignals(block)
                
    def _initialize_form_from_data(self, config_data: Dict[str, Any]):
        """
        Populate the form widgets from config_data. Decimal precision is
        determined by _build_form_ui.

        The counterpart to _prune_stale_dynamic_fields, for the opposite
        direction: a field the just-rebuilt widget set has but config_data
        doesn't (e.g. a simulator switch that introduces a field the
        previously selected simulator's own model didn't have -- sim_stub
        -> sim_kiritsubo's w_prime_recovery_rate/vitality_loss_rate) is
        seeded from that field's own core.schema.preset_value_from_field
        instead of being left at whatever bare value widget construction
        happened to give it. Without this, the widget would show a value
        (Qt's own zero-ish default) that config_data never actually
        received, so the saved JSON would fail the new simulator's own
        "Field required" validation on next load.

        Writes the seed into config_data too (not just the widget) for the
        same reason _prune_stale_dynamic_fields writes back into
        self.json_editor: this pane's Raw JSON Editor tab has its own text
        buffer, independent of the widgets, and _save_config_only() saves
        that text -- not config_data directly -- so a value that only
        reached the widget would look present in the GUI Form tab while
        still missing from the saved file.
        """
        schema_map = self._get_schema_map()
        seeded_any = False
        for section_name, field_widgets in self.widgets.items():
            section_data = self._ensure_section_data(config_data, section_name)
            schema_class = schema_map.get(section_name)

            for field_name, widget in field_widgets.items():
                value = section_data.get(field_name)
                is_preset = False

                if value is None and schema_class is not None:
                    value = preset_value_from_field(schema_class, field_name)
                    if value is not None:
                        section_data[field_name] = value
                        seeded_any = True
                        is_preset = True

                if value is not None:
                    # Block signals to avoid triggering unnecessary saves
                    widget.blockSignals(True)
                    try:
                        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                            widget.setValue(value)
                        elif isinstance(widget, QComboBox):
                            widget.setCurrentText(str(value))
                        elif isinstance(widget, QLineEdit):
                            _set_line_edit_text(widget, str(value))
                        # Freshly built widgets (_build_form_ui always runs
                        # right before this method) start unstyled, so the
                        # "else ''" branch is only reached for genuinely
                        # non-preset fields -- see _PRESET_VALUE_STYLE's own
                        # docstring for why this is cleared the instant the
                        # user edits the field themselves.
                        widget.setStyleSheet(self._PRESET_VALUE_STYLE if is_preset else "")
                    finally:
                        widget.blockSignals(False)

        if seeded_any:
            # blockSignals() returns the PREVIOUS state -- restore that,
            # not unconditionally False. See _prune_stale_dynamic_fields's
            # own comment on the same pattern: load_config() calls this
            # method from inside its own blockSignals(True)/(False)
            # bracket, and an unconditional unblock here would re-enable
            # json_editor's textChanged signal before that closing call.
            was_blocked = self.json_editor.blockSignals(True)
            self.json_editor.setPlainText(json.dumps(self.config_data, indent=4, ensure_ascii=False))
            self.json_editor.blockSignals(was_blocked)

    @Slot()
    def _sync_json_to_form(self):
        """Parse the JSON editor text and update all form widgets to match."""
        self._block_form_signals(True) 
        try:
            json_text = self.json_editor.toPlainText()
            if not json_text.strip():
                self.config_data = {}
                return
            new_config_data = json.loads(json_text)
            self.config_data = new_config_data
            # Rebuild PhysicalSettings/PhysiologicalSettings/OptimizerParams'
            # rows in case the user hand-edited Engine.simulator/
            # Engine.optimizer directly in the Raw JSON Editor tab -- see
            # _get_schema_map's docstring. Pruned the same way as
            # _on_registry_selection_changed: a hand-typed dynamic-section
            # field that doesn't belong to the now-resolved simulator/
            # optimizer has no widget to show it, but would otherwise
            # survive untouched in self.config_data and get written back
            # out on the next form edit -- see _prune_stale_dynamic_fields's
            # own docstring.
            self._build_form_ui()
            self._prune_stale_dynamic_fields()
            self._initialize_form_from_data(self.config_data)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error("Error syncing JSON to form: %s", e)
        finally:
            self._block_form_signals(False)

    @Slot()
    def _sync_form_to_json(self):
        """Read all form widget values and update the JSON editor to match."""
        current_data = self.config_data.copy()
        try:
            for section_name, field_widgets in self.widgets.items():
                section_data = self._get_section_data(current_data, section_name)

                for field_name, widget in field_widgets.items():
                    value = None
                    key = f'{section_name}.{field_name}'
                    if isinstance(widget, QDoubleSpinBox):
                        value = widget.value()
                        decimals = self._DECIMAL_PLACES.get(key, 4)
                        # Strip trailing zeros while preserving precision
                        # e.g. 0.100000 -> 0.1 / -0.001000 -> -0.001
                        formatted_value = float(f"{value:.{decimals}f}".rstrip('0').rstrip('.'))
                        value = formatted_value

                    elif isinstance(widget, QSpinBox):
                        value = int(widget.value())
                    elif isinstance(widget, QComboBox):
                        value = widget.currentText()
                    elif isinstance(widget, QLineEdit):
                        value = widget.text()
                        
                    if value is not None:
                        section_data[field_name] = value

                self._set_section_data(current_data, section_name, section_data)

            self.config_data = current_data
            
            # Block JSON editor signals while updating to avoid re-triggering sync
            self.json_editor.blockSignals(True)
            self.json_editor.setPlainText(json.dumps(self.config_data, indent=4, ensure_ascii=False))
            self.json_editor.blockSignals(False)
        except Exception as e:
            logger.error("FATAL ERROR syncing form to JSON: %s", e)
            pass
            
    def load_config(self, filename: str) -> bool:
        """Load filename into the editor; return True on success, False on error."""
        try:
            file_path = os.path.join(self.file_manager.CONFIGS_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json_text = f.read()
            config_data = json.loads(raw_json_text)
            
            self.config_data = config_data
            self.title_label.setText(f"Editing Config File: {filename}")
            self.current_file_path = file_path

            self.json_editor.blockSignals(True)
            self.json_editor.setPlainText(raw_json_text)

            # Rebuild PhysicalSettings/PhysiologicalSettings/OptimizerParams'
            # rows against THIS file's own Engine.simulator/Engine.optimizer
            # before populating -- see _get_schema_map's docstring. Without
            # this, a config naming a non-default simulator/optimizer would
            # show the wrong field set until the user happened to touch the
            # simulator/optimizer combo box themselves.
            self._build_form_ui()
            # Drop any dynamic-section field this file has that doesn't
            # belong to its own Engine.simulator/Engine.optimizer (e.g. a
            # hand-edited file), so a subsequent form edit (see
            # _sync_form_to_json's merge behavior) doesn't perpetuate it.
            # If anything is actually pruned, this also refreshes the Raw
            # JSON Editor tab's displayed text to match -- see this
            # method's own docstring for why that refresh is required,
            # not cosmetic.
            self._prune_stale_dynamic_fields()
            self._initialize_form_from_data(config_data)
            
            self.json_editor.blockSignals(False)
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"An error occurred while loading the config file: {e}")
            return False

    def _save_config_only(self) -> bool:
        """Validate and write current JSON editor content to disk; return True on success."""
        if not self.current_file_path:
            QMessageBox.warning(self, "Save Error", "The file being edited cannot be identified.")
            return False
            
        filename = os.path.basename(self.current_file_path)
        try:
            final_json_text = self.json_editor.toPlainText()
            final_config_data = json.loads(final_json_text)
            
            self.file_manager.save_config_json(filename, final_config_data)
            self.config_data = final_config_data 
            return True
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Save Error", "JSON format is invalid. Save failed.")
            return False
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the file: {e}")
            return False

    @Slot()
    def _save_and_back(self):
        """Save the config and emit editor_closed to return to the manager pane."""
        if self._save_config_only():
            self.editor_closed.emit()

    @Slot()
    def _cancel_and_back(self):
        """Discard changes and emit editor_closed to return to the manager pane."""
        self.editor_closed.emit()

# ----------------------
# B-2. FilterEditorPane 
# ----------------------
class FilterEditorPane(QWidget):
    """
    Editor pane for the result-filter JSON file used by the Viewer.

    Displays a type-annotated form driven by type_map and emits editor_closed
    when the user dismisses the pane.
    """
    editor_closed = Signal()

    def __init__(self, file_manager, type_map: Dict[str, str], parent=None):
        """Initialize with the given file_manager and type_map schema, then build the filter editor UI."""
        super().__init__(parent)
        self.file_manager = file_manager
        self.TYPE_MAP = type_map 
        self.current_filename: Optional[str] = None
        self.widgets: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.config_data: Dict[str, Any] = {}
        
        self.checkbox_container_map: Dict[QCheckBox, QWidget] = {} 

        self._setup_ui() 

    # --------------------------------------------------
    # Helper methods for UI construction
    # --------------------------------------------------

    def _setup_ui(self):
        """Build the filter editor layout with a tabbed form/JSON editor and Save/Cancel buttons."""
        main_layout = QVBoxLayout(self)
        self.title_label = QLabel("Filter Configuration Editor")
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        main_layout.addWidget(self.title_label)

        self.tab_widget = QTabWidget()
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        form_widget = QWidget()
        self.form_layout_content = QVBoxLayout(form_widget)
        self.scroll_area.setWidget(form_widget)
        self._build_form_ui() 

        self.json_editor = QPlainTextEdit()
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(14)
        self.json_editor.setFont(font)
        
        self.tab_widget.addTab(self.scroll_area, "⚙️ GUI Form Editor") 
        self.tab_widget.addTab(self.json_editor, "📝 Raw JSON Editor")
        main_layout.addWidget(self.tab_widget)
        
        button_layout = QHBoxLayout()
        self.save_and_back_button = QPushButton("💾 Save Filter and Close")
        self.cancel_and_back_button = QPushButton("❌ Cancel")
        
        button_layout.addWidget(self.save_and_back_button)
        button_layout.addWidget(self.cancel_and_back_button)
        main_layout.addLayout(button_layout)
        
        self.save_and_back_button.clicked.connect(self.save_and_back)
        self.cancel_and_back_button.clicked.connect(self.cancel_editing)
        
        self.json_editor.textChanged.connect(self.sync_json_to_form) 

    def _get_group_name(self, field_key: str) -> str:
        """Return the section display name for a given field key."""
        if field_key == 'strategy_set_dir':
            return SECTION_MAP.get('run_set_id', 'Execution Metadata')

        if field_key.startswith('input.settings.physiological'):
            section_prefix = 'input.settings.physiological'
        elif field_key.startswith('input.settings.physical'):
            section_prefix = 'input.settings.physical'
        elif field_key.startswith('input.settings.run'):
            section_prefix = 'input.settings.run'
        elif field_key.startswith('input.versions'):
            section_prefix = 'input.versions'
        elif field_key.startswith('input.git_state'):
            section_prefix = 'input.git_state'
        elif field_key.startswith('output.metadata'):
            section_prefix = 'output.metadata'
        elif field_key.startswith('output.results'):
            section_prefix = 'output.results'
        elif field_key == 'run_set_id':
            section_prefix = 'run_set_id'
        else:
            section_prefix = field_key
        return SECTION_MAP.get(section_prefix, 'Other')

    def _get_display_label(self, field_key: str) -> str:
        """Return a human-readable label string for field_key."""
        if field_key == 'run_set_id':
            return "Run Set ID"
        if field_key == 'input.settings.run.gpx_filename':
            return "GPX Filename"
        # Extract last word segment, capitalize
        display_name = field_key.split('.')[-1]
        return ' '.join(word.capitalize() for word in display_name.split('_'))

    @Slot(int)
    def set_filter_enabled(self, state: int):
        """Enable or disable the input container for the sender checkbox based on its check state."""
        checkbox = self.sender()
        if isinstance(checkbox, QCheckBox) and checkbox in self.checkbox_container_map:
            container = self.checkbox_container_map[checkbox]
            is_checked = (state == Qt.CheckState.Checked.value)
            
            container.setEnabled(is_checked)
            self.sync_form_to_json()

    def _build_form_ui(self):
        """Dynamically build the filter input form from TYPE_MAP."""
        for i in reversed(range(self.form_layout_content.count())):
            item = self.form_layout_content.itemAt(i)
            if item.widget():
                item.widget().setParent(None)

        self.widgets.clear()
        self.checkbox_container_map.clear()

        # Group fields by section; prepend strategy_set_dir (physical path, not in TYPE_MAP)
        groups = {}
        sorted_keys = ['strategy_set_dir'] + [k for k in self.TYPE_MAP.keys() if k != 'strategy_set_dir']

        for field_key in sorted_keys:
            group_name = self._get_group_name(field_key)
            if group_name not in groups:
                groups[group_name] = []
            field_type = 'string' if field_key == 'strategy_set_dir' else self.TYPE_MAP.get(field_key, 'string')
            groups[group_name].append((field_key, field_type))

        for group_name, fields in groups.items():
            group_box = QGroupBox(group_name)
            form_layout = QFormLayout(group_box)
            self.widgets[group_name] = {}

            for field_key, field_type in fields:
                self.widgets[group_name][field_key] = {}
                self._create_filter_row(group_name, field_key, field_type, form_layout)

            self.form_layout_content.addWidget(group_box)

        self.form_layout_content.addStretch(1)

    def _connect_filter_signals(self, checkbox: QCheckBox, container: QWidget):
        """Connect widget change signals to sync_form_to_json."""
        checkbox.stateChanged.connect(self.set_filter_enabled)
        for line_edit in container.findChildren(QLineEdit):
            line_edit.textChanged.connect(self.sync_form_to_json)
        for combo in container.findChildren(QComboBox):
            combo.currentIndexChanged.connect(self.sync_form_to_json)

    def _show_selector_popup(self, items, target_edit, btn):
        """Display a scrollable popup menu of items; clicking one sets target_edit's text."""
        if not items:
            items = ["(No Data)"]

        popup = QMenu(self)
        list_widget = QListWidget()
        list_widget.addItems(items)
        list_widget.setFixedSize(300, 400)
        
        action = QWidgetAction(popup)
        action.setDefaultWidget(list_widget)
        popup.addAction(action)

        def on_item_clicked(*args):
            """Set the target line edit to the clicked item text and close the popup."""
            item = list_widget.currentItem()
            if item:
                if item.text() != "(No Data)":
                    try:
                        _set_line_edit_text(target_edit, item.text())
                    except AttributeError as e:
                        logger.error("target_edit is NOT QLineEdit anymore: %s", e)
                    
                    if hasattr(self, 'sync_form_to_json'):
                        self.sync_form_to_json()
            popup.close()

        list_widget.itemClicked.connect(on_item_clicked)
        
        popup.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _create_filter_row(self, group_name: str, field_key: str, field_type: str, form_layout: QFormLayout):
        """Build one filter row: checkbox + input widget(s)."""
        label_text = self._get_display_label(field_key)
        checkbox = QCheckBox("")
        checkbox.setChecked(False)

        label_container = QWidget()
        label_hbox = QHBoxLayout(label_container)
        label_hbox.setContentsMargins(0, 0, 0, 0)
        label_hbox.addWidget(checkbox)
        label_hbox.addWidget(QLabel(label_text))
        label_hbox.addStretch(1)

        widget_container = QWidget()
        container_layout = QHBoxLayout(widget_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        widgets_dict: Dict[str, Any] = {}

        # Numeric type (Min/Max range)
        if field_type in ['integer', 'float']:
            min_w = QLineEdit()
            max_w = QLineEdit()
            validator = OptionalIntValidator() if field_type == 'integer' else OptionalFloatValidator()
            min_w.setValidator(validator)
            max_w.setValidator(validator)
            min_w.setPlaceholderText(field_type.capitalize())
            max_w.setPlaceholderText(field_type.capitalize())

            widgets_dict.update({'min': min_w, 'max': max_w})
            container_layout.addWidget(QLabel("Min:"))
            container_layout.addWidget(min_w)
            container_layout.addWidget(QLabel("Max:"))
            container_layout.addWidget(max_w)

        # String type (glob pattern / selector)
        elif field_type == 'string':
            edit_w = QLineEdit()
            edit_w.setPlaceholderText("Glob Pattern (e.g. 2026*)")
            widgets_dict['pattern'] = edit_w
            container_layout.addWidget(edit_w)

            # strategy_set_dir: show physical folder history
            if field_key == 'strategy_set_dir':
                history_btn = QToolButton()
                history_btn.setText("▼")
                container_layout.addWidget(history_btn)

                def on_dir_click(_, target=edit_w, btn=history_btn):
                    """Show a popup of existing strategy-set directories for the strategy_set_dir field."""
                    if not os.path.exists(BASE_STRATEGIES_DIR):
                        items = []
                    else:
                        items = [d for d in os.listdir(BASE_STRATEGIES_DIR)
                                 if os.path.isdir(os.path.join(BASE_STRATEGIES_DIR, d)) and not d.startswith('.')]
                        items.sort(reverse=True)
                    self._show_selector_popup(items, target, btn)

                history_btn.clicked.connect(on_dir_click)

            # run_set_id: show logical ID history
            elif field_key == 'run_set_id':
                history_btn = QToolButton()
                history_btn.setText("▼")
                container_layout.addWidget(history_btn)

                def on_id_click(_, target=edit_w, btn=history_btn):
                    """Show a popup of known run_set_id values for the run_set_id field."""
                    items = self.file_manager.get_all_run_set_ids()
                    self._show_selector_popup(items, target, btn)

                history_btn.clicked.connect(on_id_click)

        # Boolean type
        elif field_type == 'boolean':
            combo = QComboBox()
            combo.addItems(["True", "False", "Any"])
            combo.setCurrentIndex(2)
            widgets_dict['value'] = combo
            container_layout.addWidget(combo)

        # Register
        self.widgets[group_name][field_key].update({
            'checkbox': checkbox,
            'container': widget_container
        })
        self.widgets[group_name][field_key].update(widgets_dict)
        self.checkbox_container_map[checkbox] = widget_container

        form_layout.addRow(label_container, widget_container)
        widget_container.setEnabled(False)
        self._connect_filter_signals(checkbox, widget_container)

    def _block_form_signals(self, block: bool):
        """Block or unblock signals on all filter form widgets and the JSON editor."""
        for group_widgets in self.widgets.values():
            for field_widgets in group_widgets.values():
                for widget in field_widgets.values():
                    if hasattr(widget, 'blockSignals'):
                        widget.blockSignals(block)
        
        self.json_editor.blockSignals(block)

    def load_config(self, filename: str) -> bool:
        """Load filename into the filter editor; return True on success, False on error."""
        try:
            config_data = self.file_manager.load_config_json(filename)
            self.current_filename = filename
            self.title_label.setText(f"Editing Filter File: {filename}")
            self._initialize_form_from_data(config_data)
            # Every filter value here is a short flat array (a [min, max]
            # range pair, or a single-element [True]/[False] -- see
            # sync_form_to_json's own filter_value construction), which
            # json.dumps(indent=4) would otherwise spread across several
            # indented lines each. These two regexes collapse each array
            # back onto one line ("[\n    1,\n    2\n]" -> "[1, 2]") for a
            # more compact, readable manual-editor view -- first pulling
            # the whitespace right inside the brackets, then collapsing
            # the comma-newline-indent between elements to ", ".
            raw_json = json.dumps(config_data, indent=4, ensure_ascii=False)
            json_str = re.sub(r'\[\s+([^\[\]]*?)\s+\]', r'[\1]', raw_json)
            json_str = re.sub(r',\s+(?=[^\[\]]*?\])', r', ', json_str)
            self.json_editor.blockSignals(True)
            self.json_editor.setPlainText(json_str) 
            self.json_editor.blockSignals(False)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load filter file '{filename}': {e}")
            self.current_filename = None
            self.title_label.setText("Filter Configuration Editor")
            return False

    def _initialize_form_from_data(self, filter_data: Dict[str, Any]):
        """Populate form widgets from filter_data; missing keys leave widgets at their defaults."""
        self._block_form_signals(True)
        try:
            for group_widgets in self.widgets.values():
                for field_key, widgets_dict in group_widgets.items():
                    filter_value = filter_data.get(field_key)
                    checkbox: Optional[QCheckBox] = widgets_dict.get('checkbox')
                    container: Optional[QWidget] = widgets_dict.get('container')

                    if checkbox:
                        is_enabled = filter_value is not None
                        checkbox.setChecked(is_enabled)
                        if container: container.setEnabled(is_enabled)
                        
                    
                    if 'min' in widgets_dict:
                        min_w: QLineEdit = widgets_dict['min']
                        max_w: QLineEdit = widgets_dict['max']

                        if isinstance(filter_value, list) and len(filter_value) == 2:
                            min_val, max_val = filter_value[0], filter_value[1]
                            
                            _set_line_edit_text(min_w, str(min_val) if min_val is not None else "")
                            _set_line_edit_text(max_w, str(max_val) if max_val is not None else "")
                        else:
                            min_w.setText("")
                            max_w.setText("")

                    elif 'pattern' in widgets_dict:
                        pattern_w: QLineEdit = widgets_dict['pattern']
                        if isinstance(filter_value, str):
                            _set_line_edit_text(pattern_w, filter_value)
                        else:
                            pattern_w.setText("")

                    elif 'value' in widgets_dict:
                        combo_w: QComboBox = widgets_dict['value']
                        if isinstance(filter_value, list):
                            if filter_value == [True]: combo_w.setCurrentIndex(0)
                            elif filter_value == [False]: combo_w.setCurrentIndex(1)
                            else: combo_w.setCurrentIndex(2) 
                        else:
                            combo_w.setCurrentIndex(2)
                            
        except Exception as e:
            QMessageBox.warning(self, "Form Initialization Error", f"Error during form setup: {e}")
        finally:
            self._block_form_signals(False)

    @Slot()
    def sync_form_to_json(self):
        """Build JSON data from the current form state and update the JSON editor."""
        self.json_editor.blockSignals(True)

        try:
            current_data = {}

            for group_widgets in self.widgets.values():
                for field_key, widgets_dict in group_widgets.items():
                    checkbox: QCheckBox = widgets_dict.get('checkbox')

                    if checkbox is None or not checkbox.isChecked():
                        continue

                    filter_value = None

                    if 'min' in widgets_dict:
                        min_w: QLineEdit = widgets_dict['min']
                        max_w: QLineEdit = widgets_dict['max']
                        min_val, max_val = None, None

                        is_float_field = self.TYPE_MAP.get(field_key) == 'float'

                        min_text = min_w.text().strip()
                        max_text = max_w.text().strip()

                        try:
                            if not min_text:
                                min_val = None
                            elif is_float_field:
                                min_val = float(min_text)
                            else:
                                min_val = int(min_text)
                        except ValueError:
                            min_val = None

                        try:
                            if not max_text:
                                max_val = None
                            elif is_float_field:
                                max_val = float(max_text)
                            else:
                                max_val = int(max_text)
                        except ValueError:
                            max_val = None


                        if min_val is not None or max_val is not None:
                            if is_float_field:
                                filter_value = [round(v, 4) if isinstance(v, float) else v for v in [min_val, max_val]]
                            else:
                                filter_value = [min_val, max_val]

                    elif 'pattern' in widgets_dict:
                        pattern_w: QLineEdit = widgets_dict['pattern']
                        pattern_text = pattern_w.text().strip()
                        if pattern_text:
                            filter_value = pattern_text

                    elif 'value' in widgets_dict:
                        combo_w: QComboBox = widgets_dict['value']
                        index = combo_w.currentIndex()
                        if index == 0: filter_value = [True]
                        elif index == 1: filter_value = [False]
                    
                    if filter_value is not None:
                        current_data[field_key] = filter_value

            # Collapse each flat array onto one line -- see load_config's
            # own comment on the same two regexes for the full rationale.
            raw_json = json.dumps(current_data, indent=4, ensure_ascii=False)
            json_output = re.sub(r'\[\s+([^\[\]]*?)\s+\]', r'[\1]', raw_json)
            json_output = re.sub(r',\s+(?=[^\[\]]*?\])', r', ', json_output)
            self.json_editor.setPlainText(json_output)
            self.config_data = current_data
            
            # Save button is always enabled when data is valid
            self.save_and_back_button.setEnabled(True)
            self.save_and_back_button.setText("💾 Save Filter and Close")
            
        except Exception:
            self.save_and_back_button.setEnabled(False)
            self.save_and_back_button.setText("❌ Save Filter and Close (Internal Error)")

        self.json_editor.blockSignals(False)


    @Slot()
    def sync_json_to_form(self):
        """Parse the JSON editor text and repopulate all filter form widgets."""
        self._block_form_signals(True)
        
        try:
            json_text = self.json_editor.toPlainText()
            if not json_text.strip(): 
                self._initialize_form_from_data({})
                return

            data = json.loads(json_text)
            self._initialize_form_from_data(data)

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error("Error syncing JSON to form: %s", e)

        self._block_form_signals(False)

        self.sync_form_to_json()

    @Slot()
    def save_and_back(self):
        """Validate and save the current filter JSON, then emit editor_closed."""
        final_json_text = self.json_editor.toPlainText()

        if not self.current_filename:
            QMessageBox.warning(self, "Save Error", "The file being edited cannot be identified.")
            return

        try:
            final_config_data = json.loads(final_json_text)
            
            self.file_manager.save_config_json(self.current_filename, final_config_data)
            self.editor_closed.emit()
            
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Save Error", "Raw JSON format is invalid. Save failed. Please check the Raw JSON Editor tab.")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving the file: {e}")

    @Slot()
    def cancel_editing(self):
        """Discard changes and emit editor_closed to return to the manager pane."""
        self.editor_closed.emit()
