"""
eidos.apps.manager.window -- TTManagerGUI, the Manager's main window.

Owns the shared ConfigFileManager/FilterFileManager instances, hosts the
editor and run/manage panes in tabs, and launches Generator/Viewer as
subprocesses (see start_external_process and
eidos.apps.manager.constants.SCRIPT_MODULE_MAP).
"""

import os
import signal
from typing import List

from PySide6.QtCore import QEvent, QProcess, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
)

from core.io_config import (
    BASE_ACTIVITIES_DIR,
    BASE_CDA_YAW_TABLES_DIR,
    BASE_CONFIGS_DIR,
    BASE_EXPORTS_DIR,
    BASE_FILTERS_DIR,
    BASE_STRATEGIES_DIR,
)
from core.io_config import (
    BASE_GPX_DIR as BASE_GPX_DATA_DIR,
)
from core.pydantic_mapper import TYPE_MAP
from eidos.apps.manager.constants import GENERATOR_SCRIPT, SCRIPT_MODULE_MAP
from eidos.apps.manager.editor_panes import ConfigurationEditorPane, FilterEditorPane
from eidos.apps.manager.file_managers import ConfigFileManager, FilterFileManager
from eidos.apps.manager.manager_panes import GenerationManagerPane, ViewingManagerPane
from eidos.lib.branding import window_title


class TTManagerGUI(QMainWindow):

    """
    Top-level main window for EIDOS^TT Manager.

    Hosts the GenerationManagerPane and ViewingManagerPane in a tab widget,
    wires inter-pane signals, and owns the shared ConfigFileManager instances.
    """
    def __init__(self, parent=None):
        """Initialize the main window, file managers, process table, and all panes."""
        super().__init__(parent)
        self.setWindowTitle(window_title("Manager"))
        self.resize(500, 800)

        self.menuBar().setNativeMenuBar(False)

        self.config_file_manager = ConfigFileManager(target_dir=BASE_CONFIGS_DIR)
        self.filter_file_manager = FilterFileManager()

        self.processes = {}  # { "script_name": [QProcess, ...] }
        # GENERATOR_SCRIPT is single-instance (list holds 0 or 1 entries,
        # enforced in start_external_process). VIEWER_SCRIPT allows
        # multiple concurrent runs, so its list can hold several.
        # Per-process, per-channel partial-line buffers, keyed by
        # (id(process), "out"|"err"). QProcess delivers output in
        # arbitrary-sized chunks that can split a line in the middle;
        # handle_process_output holds back an incomplete trailing line
        # here until the rest arrives, so level-tag detection (looking
        # for e.g. "[BANNER]" in a *complete* line) never sees a truncated
        # line. Entries are cleaned up in handle_process_finished.
        self._line_buffers = {}
        # Same keying as _line_buffers: the color to use for a line with
        # no "[LEVELNAME]" tag of its own -- e.g. a traceback body line
        # under logger.exception()'s one-line "[ERROR] ..." header --
        # so the whole multi-line record reads as one color instead of
        # the header alone. See _append_process_lines.
        self._line_carry_colors = {}
        self.setStatusBar(QStatusBar(self))

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.create_panes()
        self.create_layout()
        self.connect_signals()

        self.create_resource_menu()

    def create_resource_menu(self):
        """Add a Resources menu to the menu bar with shortcuts to key data directories."""
        bar = self.menuBar()
        resource_menu = bar.addMenu("📁 Resources")

        actions = [
            ("📍 Open GPX Data", BASE_GPX_DATA_DIR),
            ("🚴 Open CdA Yaw Tables", BASE_CDA_YAW_TABLES_DIR),
            ("⚙️ Open Configs", BASE_CONFIGS_DIR),
            ("🔍 Open Filters", BASE_FILTERS_DIR),
            None,
            ("📊 Open Strategies", BASE_STRATEGIES_DIR),
            ("📦 Open Exports", BASE_EXPORTS_DIR),
            ("⚡ Open Activities", BASE_ACTIVITIES_DIR),
        ]

        def open_in_finder(path):
            """Open path in the OS file manager, creating it first if necessary."""
            import platform
            import subprocess
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)

            if platform.system() == "Darwin":
                subprocess.run(["open", abs_path])
            elif platform.system() == "Windows":
                os.startfile(abs_path)
            else:
                subprocess.run(["xdg-open", abs_path])

        for item in actions:
            if item is None:
                resource_menu.addSeparator()
                continue

            label, target_path = item
            action = resource_menu.addAction(label)
            action.triggered.connect(lambda checked=False, p=target_path: open_in_finder(p))

    def create_panes(self):
        """Instantiate all manager and editor panes and arrange them into stacked widgets."""
        self.generation_manager_pane = GenerationManagerPane(self.config_file_manager, self)
        self.config_editor_pane = ConfigurationEditorPane(self.config_file_manager, self)
        self.viewing_manager_pane = ViewingManagerPane(self.filter_file_manager, self)
        self.filter_editor_pane = FilterEditorPane(self.filter_file_manager, type_map=TYPE_MAP, parent=self)

        # Generation stack: index 0 = list, index 1 = editor
        self.stack_gen = QStackedWidget()
        self.stack_gen.addWidget(self.generation_manager_pane)
        self.stack_gen.addWidget(self.config_editor_pane)

        # Viewing stack: index 0 = list, index 1 = editor
        self.stack_view = QStackedWidget()
        self.stack_view.addWidget(self.viewing_manager_pane)
        self.stack_view.addWidget(self.filter_editor_pane)

    def create_layout(self):
        """Add generation and viewing stacked widgets as tabs in the main tab widget."""
        self.tab_widget.addTab(self.stack_gen, "🧬 Strategy Generation")
        self.tab_widget.addTab(self.stack_view, "👀 Strategy Viewing")

    def connect_signals(self):
        """Wire all cross-pane signals to their handler slots."""
        self.generation_manager_pane.status_bar_message.connect(self.statusBar().showMessage)
        self.viewing_manager_pane.status_bar_message.connect(self.statusBar().showMessage)
        self.generation_manager_pane.run_set_triggered.connect(self.start_external_process)
        self.viewing_manager_pane.run_set_triggered.connect(self.start_external_process)

        self.generation_manager_pane.open_editor_triggered.connect(self.show_config_editor)
        self.config_editor_pane.editor_closed.connect(self.back_to_config_manager)

        self.viewing_manager_pane.open_filter_editor_triggered.connect(self.show_filter_editor)
        self.filter_editor_pane.editor_closed.connect(self.back_to_viewing_manager)

        self.generation_manager_pane.stop_triggered.connect(self.stop_external_process)
        # ViewingManagerPane has no Stop button (has_stop_button=False) --
        # multiple viewer.py instances can run at once, so there's no
        # single process for a Stop button to target.

    @Slot(str)
    def show_config_editor(self, filename: str):
        """Load filename into the config editor and switch the generation stack to editor view."""
        if self.config_editor_pane.load_config(filename):
            self.stack_gen.setCurrentIndex(1)

    @Slot(str)
    def show_filter_editor(self, filename: str):
        """Load filename into the filter editor and switch the viewing stack to editor view."""
        if self.filter_editor_pane.load_config(filename):
            self.stack_view.setCurrentIndex(1)

    def changeEvent(self, event):
        """
        Reload both file lists whenever the window becomes the active window.

        Lets external changes made in Finder (renames, additions, deletions
        in the Configs/Filters directories) show up automatically, without
        requiring the user to open/close an editor or restart the app.
        """
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.generation_manager_pane.load_config_files()
            self.viewing_manager_pane.load_config_files()
        super().changeEvent(event)

    @Slot()
    def back_to_config_manager(self):
        """Reload the config file list and switch the generation stack back to manager view."""
        self.generation_manager_pane.load_config_files()
        self.stack_gen.setCurrentIndex(0)

    @Slot()
    def back_to_viewing_manager(self):
        """Reload the filter file list and switch the viewing stack back to manager view."""
        self.viewing_manager_pane.load_config_files()
        self.stack_view.setCurrentIndex(0)

    @Slot(list)
    def start_external_process(self, run_info: List[str]):
        """
        Launch a script in an isolated process group via a one-liner relay.
        Using os.setpgrp() ensures all child processes (parallel workers) can be
        terminated cleanly on stop.
        """
        script_name = run_info[0]

        # GENERATOR_SCRIPT is single-instance -- block a second launch
        # while one is running. VIEWER_SCRIPT allows multiple concurrent
        # instances, so no such guard applies to it.
        if script_name == GENERATOR_SCRIPT:
            running = self.processes.get(script_name, [])
            if running and running[0].state() == QProcess.ProcessState.Running:
                QMessageBox.warning(self, "Warning", f"⚠️ {script_name} is already running.")
                return

        python_executable = self.config_file_manager.PYTHON_EXECUTABLE
        module_name = SCRIPT_MODULE_MAP[script_name]

        # Route output to the appropriate log viewer
        if script_name == GENERATOR_SCRIPT:
            target_log = self.generation_manager_pane.log_viewer
            target_log.clear()
        else:
            target_log = self.viewing_manager_pane.log_viewer
            # Viewer allows multiple concurrent runs sharing one log, so a
            # fresh launch doesn't clear output still coming from other
            # running instances -- just mark where this run's output starts.
            #
            # Uses _append_lines (the same cursor.insertText() path as
            # ordinary process output) rather than appendPlainText(): Qt's
            # appendPlainText() always inserts its own paragraph break
            # before the new text, which added a blank line here that
            # core.logging_setup.log_banner()'s own "[BANNER]" section
            # headers -- also written via _append_lines -- don't get,
            # making the two look inconsistent.
            from datetime import datetime
            start_time = datetime.now().strftime("%H:%M:%S")
            border = "=" * 25
            self._append_lines(
                target_log, [border, f"🚀 STARTED at {start_time}", border], self.LOG_COLOR_INFO
            )

        process = QProcess(self)
        self.processes.setdefault(script_name, []).append(process)

        # stdout/stderr are read via their dedicated readAllStandardOutput()/
        # readAllStandardError() calls (not the generic readAll(), which
        # only ever reads from QProcess's current read channel -- stdout by
        # default -- and would silently drop stderr content). Each channel
        # gets its own color in the log viewer so real errors (tracebacks,
        # warnings, etc., which land on stderr since the app's own logger
        # is configured to write to stdout) stand out from normal output.
        process.readyReadStandardOutput.connect(
            lambda p=process, log=target_log: self.handle_process_output(
                p, log, QProcess.ProcessChannel.StandardOutput
            )
        )
        process.readyReadStandardError.connect(
            lambda p=process, log=target_log: self.handle_process_output(
                p, log, QProcess.ProcessChannel.StandardError
            )
        )
        process.finished.connect(
            lambda exit_code, exit_status, name=script_name, proc=process: self.handle_process_finished(
                exit_code, exit_status, name, proc
            )
        )

        # Relay one-liner: os.setpgrp() makes this the PGID leader,
        # then subprocess runs the target module under it
        launcher_code = (
            f"import os, sys, subprocess; "
            f"os.setpgrp(); "
            f"subprocess.run([r'{python_executable}', '-u', '-m', r'{module_name}'], check=True)"
        )

        command_list = [python_executable, "-u", "-c", launcher_code]
        process.start(command_list[0], command_list[1:])
        self.statusBar().showMessage(f"🚀 Running {script_name} with Group Isolation...", 0)

        # Update UI buttons. Viewer has no stop button and stays enabled
        # for repeated launches (multiple instances are allowed).
        if script_name == GENERATOR_SCRIPT:
            self.generation_manager_pane.stop_button.setEnabled(True)
            self.generation_manager_pane.button_run.setEnabled(False)

    def stop_external_process(self, script_name: str):
        """
        Stop button handler: use os.killpg to terminate all processes in the
        relay process group.

        Only GENERATOR_SCRIPT has a Stop button (single-instance), so
        script_name's process list here always holds at most one entry.
        """
        running = self.processes.get(script_name, [])
        process = running[0] if running else None
        if not (process and process.state() == QProcess.ProcessState.Running):
            return

        target_pane = self.generation_manager_pane if script_name == GENERATOR_SCRIPT else self.viewing_manager_pane
        target_log = target_pane.log_viewer

        from datetime import datetime
        stop_time = datetime.now().strftime("%H:%M:%S")

        # See the matching comment on the STARTED banner in
        # start_external_process for why this uses _append_lines() rather
        # than appendPlainText().
        border = "=" * 25
        self._append_lines(
            target_log, [border, f"🛑 TERMINATED BY USER at {stop_time}", border], self.LOG_COLOR_INFO
        )

        pid = process.processId()
        if pid > 0:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:
                target_log.appendPlainText(f"⚠️ Kill error: {str(e)}")

        process.terminate()
        if not process.waitForFinished(1000):
            process.kill()

        self.statusBar().showMessage(f"🛑 Process Group '{script_name}' fully stopped.", 3000)
        target_pane.stop_button.setEnabled(False)
        target_pane.button_run.setEnabled(True)

    # Execution Log text colors. Two layers:
    #
    #   1. Channel default -- LOG_COLOR_INFO for stdout, LOG_COLOR_RAW_STDERR
    #      for stderr. Since core.logging_setup.configure_logging() routes
    #      the app's own logger (all levels) to stdout regardless of level,
    #      plain stderr content that reaches here bypassed that logger
    #      entirely -- an unhandled traceback, a Qt runtime warning, a
    #      crashed subprocess's raw output. LOG_COLOR_RAW_STDERR stays a
    #      dedicated red so those don't blend into ordinary output.
    #   2. Level-tag override -- lines from the app's own logger carry a
    #      "[LEVELNAME]" tag (LOG_FORMAT's %(levelname)s) and get colored
    #      by level regardless of which channel/process hop produced them,
    #      via _LEVEL_TAG_COLORS below. BANNER (log_banner()/log_subbanner(),
    #      the custom level between INFO and WARNING) keeps its own color so
    #      section/phase boundaries -- including the hand-off when
    #      eidos.apps.viewer launches a grandchild like Trainer/Navigator/
    #      Analyzer -- stand out from ordinary INFO/WARNING/ERROR lines too.
    LOG_COLOR_INFO = QColor("#d4d4d4")  # matches log_viewer's base foreground
    LOG_COLOR_WARNING = QColor("#e5c07b")
    LOG_COLOR_ERROR = QColor("#ff9d5c")
    LOG_COLOR_BANNER = QColor("#5cc8ff")
    LOG_COLOR_RAW_STDERR = QColor("#ff5c5c")

    @property
    def _LEVEL_TAG_COLORS(self):
        # Order doesn't matter -- the tags are mutually exclusive
        # substrings (matches logging_setup.LOG_FORMAT's %(levelname)s).
        return {
            "[BANNER]": self.LOG_COLOR_BANNER,
            "[ERROR]": self.LOG_COLOR_ERROR,
            "[WARNING]": self.LOG_COLOR_WARNING,
            "[INFO]": self.LOG_COLOR_INFO,
        }

    def _line_tag_color(self, line):
        """Return the color for `line`'s own "[LEVELNAME]" tag, or None if
        it doesn't have one (see _LEVEL_TAG_COLORS)."""
        for tag, tag_color in self._LEVEL_TAG_COLORS.items():
            if tag in line:
                return tag_color
        return None

    def _append_lines(self, target_log, lines, base_color):
        """Insert complete, self-contained lines (a start/stop banner, the
        synthetic exit-code error message) into target_log, coloring each
        one by its own "[LEVELNAME]" tag if it has one, otherwise
        base_color. No carry-over between lines -- for that (a live
        process stream, where one log record such as logger.exception()'s
        traceback can span several untagged lines), use
        _append_process_lines instead."""
        if not lines:
            return
        line_colors = [(line, self._line_tag_color(line) or base_color) for line in lines]
        self._append_colored_lines(target_log, line_colors)

    def _append_process_lines(self, target_log, lines, buf_key, base_color):
        """Insert complete lines read from a live process channel,
        carrying a tagged line's color forward onto the untagged lines
        that follow it -- e.g. logger.exception()'s one-line "[ERROR] ..."
        header followed by an untagged traceback body -- so the whole
        record reads as one color instead of just the header. Carry state
        persists across calls (and resets on a channel default when a new
        tag appears) keyed by buf_key -- see self._line_carry_colors.

        base_color is the fallback for a line with no tag of its own and
        no carried-over color yet (the channel default: white for stdout,
        red for stderr -- see the class-level comment above).
        """
        line_colors = []
        carry = self._line_carry_colors.get(buf_key)
        for line in lines:
            tag_color = self._line_tag_color(line)
            if tag_color is not None:
                carry = tag_color
            color = carry if carry is not None else base_color
            line_colors.append((line, color))
        self._line_carry_colors[buf_key] = carry
        self._append_colored_lines(target_log, line_colors)

    def _append_colored_lines(self, target_log, line_colors):
        """Insert [(line, color), ...] into target_log, one QTextCharFormat
        per line."""
        if not line_colors:
            return

        cursor = target_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for line, color in line_colors:
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            cursor.setCharFormat(fmt)
            cursor.insertText(line + "\n")
        target_log.setTextCursor(cursor)
        target_log.moveCursor(QTextCursor.MoveOperation.End)

        # Reset the widget's default format back to the stdout/INFO color so
        # that later plain appendPlainText() calls (termination banners in
        # stop_external_process) don't inherit a lingering format from this
        # chunk.
        default_fmt = QTextCharFormat()
        default_fmt.setForeground(self.LOG_COLOR_INFO)
        target_log.setCurrentCharFormat(default_fmt)

    def handle_process_output(self, process, target_log, channel):
        """Read one channel (stdout or stderr), split it into complete
        lines (holding back any trailing partial line for the next chunk
        -- see self._line_buffers), and append each line to the target
        log viewer colored by its own "[LEVELNAME]" tag, a carried-over
        color from a preceding tagged line (e.g. a logger.exception()
        traceback body), or the channel default -- see
        _append_process_lines.

        Reads via readAllStandardOutput()/readAllStandardError() -- the
        channel-specific methods -- rather than the generic readAll(),
        which only reads QProcess's current read channel (stdout by
        default) regardless of which readyRead* signal fired, and would
        silently drop stderr data.
        """
        if channel == QProcess.ProcessChannel.StandardOutput:
            raw_data = process.readAllStandardOutput().data()
            base_color = self.LOG_COLOR_INFO
            buf_key = (id(process), "out")
        else:
            raw_data = process.readAllStandardError().data()
            base_color = self.LOG_COLOR_RAW_STDERR
            buf_key = (id(process), "err")

        if not raw_data:
            return

        try:
            data = raw_data.decode('utf-8')
        except UnicodeDecodeError:
            data = raw_data.decode('utf-8', errors='replace')

        pending = self._line_buffers.get(buf_key, "")
        text = pending + data
        parts = text.split("\n")
        if text.endswith("\n"):
            complete_lines, pending = parts[:-1], ""
        else:
            complete_lines, pending = parts[:-1], parts[-1]
        self._line_buffers[buf_key] = pending

        self._append_process_lines(target_log, complete_lines, buf_key, base_color)

    def _flush_process_buffers(self, process, target_log):
        """Flush any trailing partial lines left in self._line_buffers for
        this process (both channels) -- called when the process exits, so
        a final line with no trailing newline isn't silently dropped.
        Also drops that process's carry-color state (self._line_carry_colors)
        now that both buf_keys are done, so a future process object whose
        id() happens to be reused never inherits a stale carried color."""
        for suffix, base_color in (("out", self.LOG_COLOR_INFO), ("err", self.LOG_COLOR_RAW_STDERR)):
            buf_key = (id(process), suffix)
            pending = self._line_buffers.pop(buf_key, "")
            if pending:
                self._append_process_lines(target_log, [pending], buf_key, base_color)
            self._line_carry_colors.pop(buf_key, None)

    @Slot(int, QProcess.ExitStatus, str, QProcess)
    def handle_process_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus, script_name: str, process: QProcess
    ):
        """Update UI state and status bar after the external process exits.

        `process` identifies which of the (possibly several, for
        VIEWER_SCRIPT) concurrent instances finished, since script_name
        alone no longer uniquely identifies one.
        """
        target_pane = self.generation_manager_pane if script_name == GENERATOR_SCRIPT else self.viewing_manager_pane

        self._flush_process_buffers(process, target_pane.log_viewer)

        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self.statusBar().showMessage(f"✅ {script_name} completed.", 3000)
            if script_name == GENERATOR_SCRIPT:
                self.viewing_manager_pane.load_config_files()
        else:
            self.statusBar().showMessage(f"❌ {script_name} failed (Code: {exit_code})", 5000)
            self._append_lines(
                target_pane.log_viewer,
                ["", f"[ERROR] {script_name} stopped with exit code {exit_code}"],
                self.LOG_COLOR_ERROR,
            )

        running = self.processes.get(script_name)
        if running and process in running:
            running.remove(process)
            if not running:
                del self.processes[script_name]

        # Viewer has no stop button and stays enabled for repeated launches.
        if script_name == GENERATOR_SCRIPT:
            target_pane.stop_button.setEnabled(False)
            target_pane.button_run.setEnabled(True)
