"""
eidos.apps.manager.manager_panes -- BaseManagerPane and its two subclasses.

BaseManagerPane is the run/manage panel shared by strategy generation and
viewing (config file list, run/duplicate/edit/delete buttons, log
viewer); GenerationManagerPane and ViewingManagerPane extend it with
their own pane-specific extras.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from eidos.apps.manager.constants import GENERATOR_SCRIPT, VIEWER_SCRIPT
from eidos.apps.manager.file_managers import ConfigFileManager, FilterFileManager
from eidos.apps.manager.helpers import format_script_name


class BaseManagerPane(QWidget):
    """
    Base widget that manages a single external Python script (generator or viewer).

    Provides a process table, run/stop controls, and optional configuration editor
    access. Subclasses specialise for generation or viewing workflows.
    """
    status_bar_message = Signal(str)
    run_set_triggered = Signal(list) 
    open_editor_triggered = Signal(str)
    stop_triggered = Signal(str)
    
    def __init__(self, file_manager: ConfigFileManager, script_name: str, pane_title: str, editor_available: bool, has_stop_button: bool = True, parent=None):
        """Initialize with file_manager, script name, pane title, and editor availability flag.

        has_stop_button: whether this pane gets a Stop button at all. Panes
        that allow multiple concurrent runs of their script (see
        ViewingManagerPane) have no single process to stop, so they skip it.
        """
        super().__init__(parent)
        self.file_manager = file_manager
        self.script_name = script_name
        self.pane_title = pane_title
        self.editor_available = editor_available
        self.has_stop_button = has_stop_button
        self.create_widgets()
        self.create_layout()
        self.connect_signals()
        self.load_config_files()

        # Shared log viewer for process stdout/stderr
        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        log_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        log_font.setPointSize(12)
        self.log_viewer.setFont(log_font)
        pane_layout = self.layout()
        if pane_layout:
            pane_layout.addWidget(QLabel("Execution Log:"))
            pane_layout.addWidget(self.log_viewer)
            self.config_list_widget.setMaximumHeight(100)  # limit file list height to 100px
            self.config_list_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            self.log_viewer.setMaximumHeight(16777215)  # Qt maximum: allow log viewer to expand
            # setStretch is QBoxLayout-specific (not on the generic QLayout
            # that self.layout() is typed to return) -- but create_layout()
            # above always builds this pane's layout as a QVBoxLayout, so
            # this always holds in practice.
            if isinstance(pane_layout, QVBoxLayout):
                pane_layout.setStretch(pane_layout.count() - 1, 10)

        if self.has_stop_button:
            self.add_stop_button()

    def add_stop_button(self):
        """Add a Stop button to the button row and connect it to stop_triggered."""
        self.stop_button = QPushButton("🛑 Stop")
        self.stop_button.setStyleSheet("""
            QPushButton:disabled {
                background-color: #505050;
                color: #aaaaaa;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QPushButton:enabled {
                background-color: #d32f2f;
                color: white;
                border-radius: 5px;
            }
        """)
        self.stop_button.setEnabled(False)
        
        if hasattr(self, 'button_run'):
            parent_layout = self.button_run.parentWidget().layout()
            if parent_layout:
                parent_layout.addWidget(self.stop_button)
        
        self.stop_button.clicked.connect(lambda: self.stop_triggered.emit(self.script_name))

    def create_widgets(self):
        """Instantiate all list and button widgets for the manager pane."""
        self.config_list_widget = QListWidget()
        self.button_new = QPushButton("🆕 Add Config File")
        self.button_delete = QPushButton("❌ Delete Config File")
        self.button_edit = QPushButton("✏️ Edit Config File")
        self.button_duplicate = QPushButton("📄 Duplicate Config File")
        self.button_run = QPushButton(f"🚀 Run {format_script_name(self.script_name)}")
        self.button_run.setStyleSheet("""
            QPushButton {
                background-color: #E67E22;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #D35400;
            }
            QPushButton:pressed {
                background-color: #A04000;
            }
        """)

    def create_layout(self):
        """Arrange widgets in the pane layout."""
        main_layout = QVBoxLayout(self)

        button_row_layout = QHBoxLayout()
        button_row_layout.addWidget(self.button_new)
        button_row_layout.addWidget(self.button_duplicate)
        if self.editor_available:
            button_row_layout.addWidget(self.button_edit)
        button_row_layout.addWidget(self.button_delete)

        main_layout.addLayout(button_row_layout)
        main_layout.addWidget(QLabel("Files in use:"))
        main_layout.addWidget(self.config_list_widget)
        main_layout.addWidget(self.button_run)

    def connect_signals(self):
        """Connect button clicked signals to their corresponding slot methods."""
        self.button_new.clicked.connect(self.select_template_and_create) 
        self.button_delete.clicked.connect(self.delete_config)      
        self.button_duplicate.clicked.connect(self.duplicate_config)    
        self.button_edit.clicked.connect(self.open_config_editor)      
        self.button_run.clicked.connect(self.run_simulation_set)

    def load_config_files(self, select_filename: Optional[str] = None):
        """Reload the file list from disk; optionally select a specific filename."""
        self.config_list_widget.clear()
        try:
            files = self.file_manager.get_available_config_files() 
            self.config_list_widget.addItems(files)
            if select_filename:
                items = self.config_list_widget.findItems(select_filename, Qt.MatchFlag.MatchExactly)
                if items:
                    self.config_list_widget.setCurrentItem(items[0])
        except Exception as e:
            self.status_bar_message.emit(f"Failed to load files from {self.file_manager.CONFIGS_DIR}: {e}")

    def _get_selected_filename(self) -> Optional[str]:
        """Return the currently selected filename from the list widget, or None."""
        current_item = self.config_list_widget.currentItem()
        return current_item.text() if current_item else None

    def _get_filename_for_edit(self) -> Optional[str]:
        """Return the selected filename, or the first file in the list if
        nothing is selected (None if the list is empty).

        Without this fallback, Edit would depend on QListWidget's focus-in
        auto-current quirk: whichever list last had keyboard focus reports
        a current item even without a click, the other doesn't.
        """
        filename = self._get_selected_filename()
        if filename is not None:
            return filename
        if self.config_list_widget.count() == 0:
            return None
        self.config_list_widget.setCurrentRow(0)
        return self.config_list_widget.item(0).text()

    @Slot()
    def select_template_and_create(self):
        """Prompt the user to pick a template, then create a new numbered config file from it."""
        templates = self.file_manager.get_template_config_files()

        if not templates:
            template_dir = self.file_manager.TEMPLATES_DIR
            QMessageBox.warning(
                self,
                "No Templates Found",
                f"No template files found in:\n{template_dir}\n\n"
                "Please place a base JSON file in the templates directory first."
            )
            return

        selected_template, ok = QInputDialog.getItem(
            self,
            "Select Template",
            "Select a configuration template to use as a base:",
            templates, 0, False
        )

        if ok and selected_template:
            try:
                new_filename = self.file_manager.create_new_config_json(selected_template)
                self.status_bar_message.emit(f"Created new file: {new_filename} (from {selected_template})")
                self.load_config_files(select_filename=new_filename)
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Creation Error", 
                    f"An error occurred while creating the file:\n{str(e)}"
                )

    @Slot()
    def delete_config(self):
        """Delete the selected config file after user confirmation."""
        filename_to_delete = self._get_selected_filename()
        if not filename_to_delete:
            QMessageBox.warning(self, "Deletion Error", "Please select a file to delete.")
            return
        
        if QMessageBox.question(self, 'Confirm', f"Do you want to delete file '{filename_to_delete}'?",
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            try:
                self.file_manager.delete_config_json(filename_to_delete)
                self.status_bar_message.emit(f"File '{filename_to_delete}' deleted.")
                self.load_config_files() 
            except Exception as e:
                QMessageBox.critical(self, "Deletion Error", f"An error occurred during file deletion: {e}")

    @Slot()
    def duplicate_config(self):
        """Duplicate the selected config file with the next sequential filename."""
        original_filename = self._get_selected_filename()
        if not original_filename:
            QMessageBox.warning(self, "Duplication Error", "Please select a file to duplicate.")
            return
        try:
            new_filename = self.file_manager.duplicate_config_json(original_filename)
            self.status_bar_message.emit(f"File '{original_filename}' duplicated to: {new_filename}")
            self.load_config_files(select_filename=new_filename)
        except Exception as e:
            QMessageBox.critical(self, "Duplication Error", f"An error occurred during file duplication: {e}")

    @Slot()
    def open_config_editor(self):
        """Emit open_editor_triggered with the selected filename, or the first file if none is selected."""
        filename = self._get_filename_for_edit()
        if not filename:
            QMessageBox.warning(self, "Edit Error", "No files exist to edit.")
            return
        self.open_editor_triggered.emit(filename)

    @Slot()
    def run_simulation_set(self):
        """Emit run_set_triggered with the config files in the order shown in the 'Files in use' list."""
        try:
            all_config_files = [
                self.config_list_widget.item(i).text()
                for i in range(self.config_list_widget.count())
            ]
            if not all_config_files:
                QMessageBox.warning(self, "Run Error", "No files exist to run.")
                return
            # run_set_triggered emits [script_name, file1, file2, ...]
            self.run_set_triggered.emit([self.script_name] + all_config_files)
        except Exception as e:
            QMessageBox.critical(self, "Run Error", f"An error occurred while retrieving the run list: {e}")

class GenerationManagerPane(BaseManagerPane):
    """
    Manager pane for the optimization generator script.

    go_to_viewing_triggered is declared below but currently unused --
    nothing emits it and window.py's connect_signals doesn't wire it to
    anything, unlike every other pane signal there. No 'Go to Viewer'
    shortcut currently exists in the UI.
    """
    go_to_viewing_triggered = Signal()

    def __init__(self, file_manager: ConfigFileManager, parent=None):
        """Initialize with the generation script name and editor enabled."""
        super().__init__(
            file_manager=file_manager,
            script_name=GENERATOR_SCRIPT,
            pane_title="Strategy Generation Management",
            editor_available=True, 
            parent=parent
        )

class ViewingManagerPane(BaseManagerPane):
    """
    Manager pane for the result viewer script.

    Extends BaseManagerPane with filter-editor access, emitting
    open_filter_editor_triggered when the user requests filter changes.

    Unlike GenerationManagerPane, multiple viewer.py instances may run at
    once (see TTManagerGUI.start_external_process), so this pane has no
    Stop button -- there's no single process for it to target.
    """
    open_filter_editor_triggered = Signal(str)

    def __init__(self, file_manager: FilterFileManager, parent=None):
        """Initialize with the viewer script name and filter editor enabled."""
        super().__init__(
            file_manager=file_manager,
            script_name=VIEWER_SCRIPT,
            pane_title="Strategy Viewing Management",
            editor_available=True,
            has_stop_button=False,
            parent=parent
        )
        self.button_new.setText("🆕 Add Filter File")
        self.button_duplicate.setText("📄 Duplicate Filter File")
        self.button_edit.setText("✏️ Edit Filter File")
        self.button_delete.setText("❌ Delete Filter File")

    @Slot()
    def open_config_editor(self):
        """Emit open_filter_editor_triggered with the selected filename, or the first file if none is selected."""
        filename = self._get_filename_for_edit()
        if not filename:
            QMessageBox.warning(self, "Edit Error", "No files exist to edit.")
            return
        self.open_filter_editor_triggered.emit(filename)
