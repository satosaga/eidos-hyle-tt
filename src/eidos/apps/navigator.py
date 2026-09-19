#########################
# navigator.py
#########################
import glob
import json
import logging
import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import pyautogui
import pytesseract

# fit_tool imports
from fit_tool.fit_file import FitFile
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Project-internal modules
from core.io_config import BASE_TEMP_DIR, create_strategy_export_dir
from core.logging_setup import configure_logging
from eidos.lib.branding import window_title

logger = logging.getLogger(__name__)

# ------------------ Constants ------------------
SCAN_INTERVAL_MS = 180
HUD_WIDTH = 320
HUD_BG_COLOR = "#0A0A0A"
HUD_ACTIVE_COLOR = "#00FF00"
HUD_WAIT_COLOR = "#888888"
HUD_SYNC_COLOR = "#FF8C00"

SCANNER_COLOR = "rgba(255,0,0,150)"

# Absolute path (via core.io_config.BASE_TEMP_DIR) rather than a bare
# relative filename -- this app is launched both directly (as a console
# script, from whatever directory the user is in) and as a subprocess from
# eidos.apps.viewer, so it can't rely on the process cwd being the repo root.
CONFIG_FILE = os.path.join(BASE_TEMP_DIR, "hud_layout_config.json")

INITIAL_SYNC_REQUIRED_FRAMES = 2
INITIAL_SYNC_TOL_DISTANCE_KM = 0.01
MAX_VALID_SPEED_KMPH = 100 

OCR_CONFIG = (
    "--psm 7 --oem 1 "
    "-c tessedit_char_whitelist=0123456789. "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)

# ------------------ Helper functions ------------------
def find_target_fit(strategy_set_dir: str, run_set_id: str, n_seg: int, seed: int, if_value: str) -> Optional[str]:
    """Search for a FIT file matching the given parameters."""
    trial_id = f"N{n_seg}_S{seed}"

    base_dir = create_strategy_export_dir(strategy_set_dir, f"{run_set_id}_{trial_id}")

    fit_dir = os.path.join(base_dir, "fit")
    try:
        if_val_float = float(if_value)
        if_code = f"{int(round(if_val_float * 100)):03d}"
    except ValueError:
        if_code = if_value 
    
    pattern = os.path.join(fit_dir, f"*_IF{if_code}.fit")
    files = glob.glob(pattern)
    return files[0] if files else None

# ------------------ HUD ------------------
class TTNavigatorHUD(QWidget):
    """
    Always-on-top HUD widget that provides real-time pacing guidance during a time trial.

    Loads a pre-optimised strategy from a FIT file, tracks elapsed distance via a
    screen OCR loop, and displays elapsed time, the FIT course point's target
    power label, and raw distance progress along the course. Manages the OCR
    capture-region overlay (ScannerWindow) and OCR thread.
    """
    def __init__(self, fit_path):
        """Initialize the HUD: load FIT strategy, build UI, start scanner and OCR thread."""
        super().__init__()
        self.setWindowTitle(window_title("Navigator"))
        self.setFixedWidth(HUD_WIDTH)
        self.setStyleSheet(f"background-color: {HUD_BG_COLOR}; color: white;")
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        self.target_points = []
        self.max_strategy_dist = 0.0
        self.is_navigating = False
        self.is_origin_set = False
        self.start_raw_distance = 0.0
        
        self.is_racing = False
        self.start_time_perf = None
        self.finish_time_record = None

        self.current_raw_distance = None  
        self.latest_binary = None
        self.ocr_running = True
        self.force_sync_request = False
        self.lock = threading.Lock()

        self.last_confirmed_dist = None  
        self.last_confirmed_time = None
        self.last_dist = None
        self.dist_check_counter = 0

        self.load_optimization_results_from_fit(fit_path)
        self.setup_ui(fit_path)

        self.scanner = ScannerWindow()
        self.load_layout()
        self.scanner.show()

        threading.Thread(target=self.ocr_loop, daemon=True).start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(SCAN_INTERVAL_MS)

    def setup_ui(self, fit_path):
        """Build the HUD widget layout: labels, progress bar, Start/Stop button, and debug display."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        file_name = os.path.basename(fit_path)
        self.file_info_label = QLabel(f"{file_name}", self)
        self.file_info_label.setStyleSheet(f"font-size: 11px; color: {HUD_WAIT_COLOR}; font-weight: bold;")
        self.file_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_info_label)

        self.time_label = QLabel("00:00.00", self)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(f"font-size: 32px; color: {HUD_WAIT_COLOR}; font-family: 'Courier New';")
        layout.addWidget(self.time_label)

        self.power_label = QLabel("--- W", self)
        self.power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.power_label.setStyleSheet(f"font-size: 48px; color: {HUD_WAIT_COLOR}; font-weight: bold;")
        layout.addWidget(self.power_label)

        self.segment_bar = QFrame(self)
        self.segment_bar.setFixedHeight(2)
        self.segment_bar.setStyleSheet(f"background-color: {HUD_ACTIVE_COLOR}; border: none;")
        self.segment_bar.setFixedWidth(0)
        layout.addWidget(self.segment_bar)
        layout.addSpacing(2)

        self.progress_label = QLabel("READY", self)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet(f"font-size: 24px; color: {HUD_WAIT_COLOR};")
        layout.addWidget(self.progress_label)

        self.btn = QPushButton("START", self)
        self.btn.setFixedHeight(64)
        self.btn.clicked.connect(self.toggle_sync)
        self.btn.setStyleSheet("""
            QPushButton {
                color: white; font-size: 18px; font-weight: bold;
                border-radius: 15px; border: 1px solid transparent; 
                background-clip: border;
            }
            QPushButton[state="orange"] { background-color: #FF8C00; }
            QPushButton[state="stop"] { background-color: #C0392B; }
        """)
        self.update_button_style()
        layout.addWidget(self.btn)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444444; border: none;")
        layout.addWidget(sep)

        self.vision_label = QLabel(self)
        self.vision_label.setFixedHeight(40)
        self.vision_label.setStyleSheet("background-color: black; border: none;")
        self.vision_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.vision_label)

        self.debug_info_label = QLabel(self)
        self.debug_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.debug_info_label.setStyleSheet("padding-left: 10px; background-color: rgba(0,0,0,50);")
        self.update_debug_text(None, None, None)
        layout.addWidget(self.debug_info_label)

    def load_layout(self):
        """Restore HUD and scanner window positions from the JSON layout config file."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    c = json.load(f)
                self.move(c.get("hud_x", 100), c.get("hud_y", 100))
                self.scanner.setGeometry(c.get("sc_x", 400), c.get("sc_y", 150), c.get("sc_w", 115), c.get("sc_h", 35))
            except Exception: pass

    def save_layout(self):
        """Persist current HUD and scanner window geometry to the JSON layout config file."""
        try:
            config = {
                "hud_x": self.x(), "hud_y": self.y(),
                "sc_x": self.scanner.x(), "sc_y": self.scanner.y(),
                "sc_w": self.scanner.width(), "sc_h": self.scanner.height()
            }
            os.makedirs(BASE_TEMP_DIR, exist_ok=True)
            with open(CONFIG_FILE, "w") as f: json.dump(config, f)
        except Exception: pass

    def update_loop(self):
        """Timer callback: refresh the OCR preview, debug text, and all HUD display elements."""
        if self.latest_binary is not None:
            h, w = self.latest_binary.shape
            img = QImage(self.latest_binary.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(img)
            self.vision_label.setPixmap(pix.scaled(self.vision_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.update_debug_text(self.current_raw_distance, 
                               self.start_raw_distance if self.is_origin_set else None,
                               self.last_confirmed_dist)

        if not self.is_navigating:
            self.progress_label.setText("READY")
            self.progress_label.setStyleSheet(f"font-size: 24px; color: {HUD_WAIT_COLOR};")
            self.time_label.setText("00:00.00")
            self.power_label.setText("--- W")
            self.segment_bar.setFixedWidth(0)
            self.is_racing = False
        
        elif not self.is_origin_set:
            self.progress_label.setText("INITIALIZING")
            self.progress_label.setStyleSheet(f"font-size: 24px; color: {HUD_SYNC_COLOR};")
            self.power_label.setText("--- W")
            self.segment_bar.setFixedWidth(0)
        
        else:
            with self.lock:
                current_d = self.last_confirmed_dist
                start_d = self.start_raw_distance
                confirmed_time = self.last_confirmed_time
            if current_d is not None:
                pos = current_d - start_d

                if not self.is_racing and pos > 0.0:
                    self.is_racing = True
                    self.start_time_perf = confirmed_time

                if self.is_racing:
                    elapsed_s = time.perf_counter() - self.start_time_perf if pos < self.max_strategy_dist else (self.finish_time_record or (time.perf_counter() - self.start_time_perf))
                    if pos >= self.max_strategy_dist and self.finish_time_record is None:
                        self.finish_time_record = elapsed_s
                    
                    self.time_label.setStyleSheet(f"font-size: 32px; color: {HUD_ACTIVE_COLOR}; font-family: 'Courier New';")
                    mm, ss = divmod(elapsed_s, 60)
                    self.time_label.setText(f"{int(mm):02d}:{int(ss):02d}:{int((ss-int(ss))*100):02d}")

                self.progress_label.setText(f"{pos:.2f} km / {self.max_strategy_dist:.2f} km")
                self.progress_label.setStyleSheet(f"font-size: 24px; color: {HUD_ACTIVE_COLOR};")
                
                if pos >= self.max_strategy_dist:
                    self.power_label.setText("--- W")
                    self.power_label.setStyleSheet(f"font-size: 48px; color: {HUD_WAIT_COLOR}; font-weight: bold;")
                    self.progress_label.setText("FINISHED")
                    self.segment_bar.setFixedWidth(0)
                else:
                    current_label = self.target_points[0]['target_label'] if self.target_points else "--- W"
                    seg_start_dist = 0.0
                    next_pt = None
                    for p in self.target_points:
                        if pos >= p['dist']:
                            current_label = p['target_label'] 
                            seg_start_dist = p['dist']
                        else:
                            next_pt = p
                            break
                    self.power_label.setText(current_label)
                    self.power_label.setStyleSheet(f"font-size: 48px; color: {HUD_ACTIVE_COLOR}; font-weight: bold;")

                    target_dist = next_pt['dist'] if next_pt else self.max_strategy_dist
                    seg_total = target_dist - seg_start_dist
                    seg_remaining = target_dist - pos
                    if seg_total > 0:
                        ratio = max(0.0, min(1.0, seg_remaining / seg_total))
                        self.segment_bar.setFixedWidth(int((HUD_WIDTH - 20) * ratio))

    def toggle_sync(self):
        """Toggle navigation on/off; reset sync state when starting."""
        if not self.is_navigating:
            self.last_confirmed_dist = self.last_confirmed_time = self.last_dist = None
            self.dist_check_counter = 0
            self.is_origin_set = False 
            self.is_navigating = True
            self.force_sync_request = True
            self.is_racing = False
        else:
            self.is_navigating = self.is_origin_set = False
        self.update_button_style()

    def update_button_style(self):
        """Update the Start/Stop button label and style property to match is_navigating."""
        state = "stop" if self.is_navigating else "orange"
        self.btn.setText("STOP" if self.is_navigating else "START")
        if self.btn.property("state") != state:
            self.btn.setProperty("state", state)
            self.btn.style().unpolish(self.btn)
            self.btn.style().polish(self.btn)

    def load_optimization_results_from_fit(self, file_path):
        """Parse a FIT file and populate target_points and max_strategy_dist from its records."""
        try:
            fit_file = FitFile.from_file(file_path)
        except Exception as e:
            logger.error("Failed to open FIT: %s", e)
            return

        dist_list = []
        self.target_points = []
        
        for record in fit_file.records:
            msg = record.message
            msg_name = msg.__class__.__name__

            # Collect distance data
            if "RecordMessage" in msg_name:
                d_m = getattr(msg, 'distance', None)
                if d_m is not None:
                    dist_list.append(float(d_m) / 1000.0)

            # Collect instructions (course points)
            if "CoursePointMessage" in msg_name:
                cp_name = getattr(msg, 'course_point_name', "")
                cp_dist = getattr(msg, 'distance', None)
                if isinstance(cp_name, bytes):
                    cp_name = cp_name.decode('utf-8', errors='ignore')
                if cp_name and cp_dist is not None:
                    self.target_points.append({
                        'dist': float(cp_dist) / 1000.0,
                        'target_label': str(cp_name)
                    })

        # Set total course distance
        if dist_list:
            self.max_strategy_dist = max(dist_list)
        elif self.target_points:
            self.max_strategy_dist = max(p['dist'] for p in self.target_points)

        self.target_points.sort(key=lambda x: x['dist'])

    def update_debug_text(self, raw, origin, confirmed):
        """Render raw, origin, and confirmed distance values into the debug label as HTML."""
        def fmt(v): return f"{v:6.2f} km" if v is not None else "---.-- km"
        is_valid = (raw is not None and confirmed is not None and abs(raw - confirmed) < INITIAL_SYNC_TOL_DISTANCE_KM)
        raw_color = "#FF4444" if (self.is_navigating and raw is not None and not is_valid) else "#CCCCCC"
        text = (f"<div style='font-family: Arial; font-size: 11px; text-align: center; color: #888888;'>"
                f"Recognized: <span style='color: {raw_color};'>{fmt(raw)}</span> | "
                f"Origin: <span style='color: #00AAFF;'>{fmt(origin)}</span> | "
                f"Valid: <span style='color: #00AA88;'>{fmt(confirmed)}</span></div>")
        self.debug_info_label.setText(text)

    def ocr_loop(self):
        """Background thread: capture the scanner region, run OCR, and validate distance readings.

        Validates each new reading with one of two different strategies,
        depending on whether a confirmed baseline distance already exists
        (self.last_confirmed_dist):

        - No baseline yet: there's nothing to compute a plausible speed
          against, so instead this waits for OCR to sync onto the real
          displayed number -- INITIAL_SYNC_REQUIRED_FRAMES consecutive
          readings must agree within INITIAL_SYNC_TOL_DISTANCE_KM of each
          other before the latest of them is trusted as the baseline. A
          single garbled/noisy OCR frame won't match its predecessor, so
          the counter resets and confirmation is delayed until real
          digits stabilize.
        - Baseline already confirmed: each new reading is checked for
          physical plausibility instead -- the implied speed since the
          last confirmed reading must fall in [0, MAX_VALID_SPEED_KMPH),
          rejecting both a backward jump (negative) and an implausible
          leap (a misread digit).
        """
        while self.ocr_running:
            try:
                geom = self.scanner.geometry()
                screenshot = pyautogui.screenshot(region=(geom.x(), geom.y(), geom.width(), geom.height()))
                img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if cv2.countNonZero(binary) < binary.size / 2: binary = cv2.bitwise_not(binary)
                self.latest_binary = binary
                txt = pytesseract.image_to_string(binary, config=OCR_CONFIG).strip()
                if txt:
                    try:
                        val = float(txt)
                        self.current_raw_distance = val 
                        t_now = time.perf_counter()
                        if self.force_sync_request:
                            self.last_confirmed_dist = self.last_confirmed_time = None
                            self.last_dist, self.dist_check_counter = val, 0
                            self.force_sync_request = False
                            continue
                        is_valid = False
                        if self.last_confirmed_dist is None:
                            if self.last_dist is not None and abs(val - self.last_dist) <= INITIAL_SYNC_TOL_DISTANCE_KM:
                                self.dist_check_counter += 1
                                if self.dist_check_counter >= INITIAL_SYNC_REQUIRED_FRAMES: is_valid = True
                            else: self.dist_check_counter = 0
                            self.last_dist = val
                        else:
                            v_kmh = ((val - self.last_confirmed_dist) / (t_now - self.last_confirmed_time)) * 3600
                            if 0 <= v_kmh < MAX_VALID_SPEED_KMPH: is_valid = True
                        if is_valid:
                            with self.lock:
                                self.last_confirmed_dist, self.last_confirmed_time = val, t_now
                                if self.is_navigating and not self.is_origin_set:
                                    self.start_raw_distance, self.is_origin_set = val, True
                    except ValueError: pass
                else: self.current_raw_distance = None
            except Exception: pass
            time.sleep(SCAN_INTERVAL_MS / 1000)

    def closeEvent(self, event):
        """Save layout, stop the OCR thread, and close the scanner window on HUD close."""
        self.save_layout()
        self.ocr_running = False
        if hasattr(self, 'scanner'): self.scanner.close()
        event.accept()

class ScannerWindow(QWidget):
    """
    Frameless, always-on-top overlay that defines the OCR capture region.

    The user positions this transparent rectangle over the distance readout on
    screen; the Navigator's OCR thread periodically captures this region.
    """
    def __init__(self):
        """Initialize the frameless, always-on-top scanner overlay window."""
        super().__init__()
        self.setWindowTitle("Scanner")
        self.setGeometry(400, 150, 115, 35)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(f"background-color: {SCANNER_COLOR}; border: none;")
        self.offset = None
    def mousePressEvent(self, event):
        """Record the drag offset when the left mouse button is pressed."""
        if event.button() == Qt.LeftButton: self.offset = event.globalPosition() - self.pos()
    def mouseMoveEvent(self, event):
        """Move the scanner window while the left mouse button is held."""
        if event.buttons() & Qt.LeftButton and self.offset is not None:
            delta = event.globalPosition() - self.offset
            self.move(int(delta.x()), int(delta.y()))

def main() -> None:
    configure_logging()
    if len(sys.argv) < 6:
        print("Usage: eidos-navigator <StrategySetDir> <RunID> <Nseg> <Seed> <IF>")
        sys.exit(1)

    # 1. Parse arguments
    strategy_set_dir, run_id, n, s, if_val = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    # 2. Search for FIT file
    fit_file = find_target_fit(strategy_set_dir, run_id, int(n), int(s), if_val)

    if fit_file:
        app = QApplication(sys.argv)
        hud = TTNavigatorHUD(fit_file)
        hud.show()
        sys.exit(app.exec())
    else:
        logger.error("FIT file not found: StrategySetDir=%s, IF=%s", strategy_set_dir, if_val)


if __name__ == "__main__":
    main()