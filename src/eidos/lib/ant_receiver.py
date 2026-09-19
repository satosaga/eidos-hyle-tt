######################
# ant_receiver.py
######################
import logging
import math
import threading
import time

from openant.devices import ANTPLUS_NETWORK_KEY
from openant.devices.scanner import Scanner
from openant.easy.channel import Channel
from openant.easy.node import Node

from core.logging_setup import log_subbanner

logger = logging.getLogger(__name__)


# =========================
# Health Monitor
# =========================
class ANTHealthMonitor:
    """Tracks ANT+ error and timeout events for diagnostics."""

    def __init__(self):
        self.error_count = 0
        self.timeout_count = 0
        self.last_error = None

    def report(self, code, detail=None):
        """Record an error event and print a log line."""
        self.error_count += 1
        self.last_error = {
            "time": time.time(),
            "code": code,
            "detail": detail
        }
        logger.warning("%s | %s", code, detail)

# =========================
# Scanner
# =========================
class EidosScanner(Scanner):
    """ANT+ device scanner that appends discovered power meter IDs to shared state."""

    def __init__(self, node, shared_state, lock):
        # If node is None (no dongle), skip parent __init__ to avoid AttributeError
        self.shared_state = shared_state
        self.lock = lock
        if node is not None:
            super().__init__(node, device_id=0, device_type=11)
        else:
            self.node = None

    def on_found(self, device_tuple):
        """Called when a new device is detected; adds its ID to found_devices."""
        device_id, device_type, trans_type = device_tuple
        with self.lock:
            if device_id not in self.shared_state.found_devices:
                self.shared_state.found_devices.append(device_id)
                logger.info("Power meter found: ID %s", device_id)

# =========================
# ANTPowerReceiver
# =========================
class ANTPowerReceiver(threading.Thread):
    """
    Daemon thread that receives ANT+ power data and writes it to shared_state.

    Supports both page 0x10 (instantaneous power) and page 0x12 (torque-based power).
    Falls back to virtual-only mode when no USB dongle is present.
    Implements automatic channel recovery after consecutive timeouts.
    """

    def __init__(self, shared_state, state_lock):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.lock = state_lock

        self.node = None
        self.node_thread = None
        self.scanner = None
        self.data_channel = None

        self._stop_event = threading.Event()

        # Power calculation state for page 0x10 and 0x12
        self._last_10_power = None
        self._last_10_time = 0
        self._last_12_event = None
        self._last_12_torque = None
        self._last_12_period = None
        self._last_12_power = None
        self._last_12_time = 0

        self.health = ANTHealthMonitor()

    def stop_receiver(self):
        """Signal the thread to stop and clean up resources."""
        self._stop_event.set()
        self._cleanup()

    def run(self):
        """Initialize the ANT+ node and run the main receive loop.

    Starts the node in a background thread, opens a bidirectional channel,
    then polls for power data every 0.5 s. Handles virtual-only mode when
    no USB dongle is present, detects 6 s data gaps, and attempts automatic
    channel recovery after three consecutive timeouts."""
        logger.info("ANT+ Initializing Node (Hybrid Mode: USB/Virtual)...")
        try:
            self.node = Node()
            self.node_thread = threading.Thread(target=self.node.start, daemon=True)
            self.node_thread.start()

            time.sleep(2.0)
            self.node.set_network_key(0x00, ANTPLUS_NETWORK_KEY)

            self.data_channel = self.node.new_channel(Channel.Type.BIDIRECTIONAL_RECEIVE)
            logger.info("ANT+ Single Channel Instance Created.")

        except Exception:
            # No dongle present; continue in virtual-only mode
            self.health.report("PHYSICAL_NODE_MISSING", "Operating in Virtual-Only mode.")

        # Initialize scanner (safe even when node is None)
        self.scanner = EidosScanner(self.node, self.shared_state, self.lock)
        self.scanner.on_found((0, 11, 0))

        current_target = None
        timeout_consecutive_count = 0

        while not self._stop_event.is_set():
            with self.lock:
                target_id = self.shared_state.target_device_id

            if target_id != current_target:
                self._apply_channel_settings(target_id)
                current_target = target_id
                timeout_consecutive_count = 0

            if current_target == 0:  # virtual power meter
                with self.lock:
                    self.shared_state.last_data_time = time.time()
                    if self.shared_state.ant_status != "Connected":
                        self.shared_state.ant_status = "Connected"

            if current_target is not None:
                now = time.time()
                # Set power to 0 when data stops (rider stops pedaling)
                if now - self.shared_state.last_data_time > 6.0:
                    timeout_consecutive_count += 1
                    self.health.report("ANT_TIMEOUT", f"Count: {timeout_consecutive_count}")

                    self._reset_histories()
                    with self.lock:
                        self.shared_state.actual_power = 0
                        self.shared_state.ant_status = "No Data"

                    if timeout_consecutive_count >= 3:
                        log_subbanner(logger, f"ATTEMPTING ANT+ RECOVERY (Target: {current_target})")
                        self._apply_channel_settings(current_target)
                        timeout_consecutive_count = 0
                        with self.lock:
                            self.shared_state.last_data_time = time.time()

            self._select_best_power()
            time.sleep(0.5)

    def _apply_channel_settings(self, target_id):
        """Configure the ANT+ channel for the given target device ID."""
        # Skip if no physical channel (no dongle)
        if self.data_channel is None:
            if target_id == 0:
                with self.lock:
                    self.shared_state.ant_status = "Connected"
            return

        self._reset_histories()
        try:
            self.data_channel.on_broadcast_data = None
            try:
                self.data_channel.close()
            except Exception:
                pass

            time.sleep(0.5)

            if target_id == 0:
                with self.lock:
                    self.shared_state.ant_status = "Connected"
                return

            logger.info("Applying Settings to Channel: ID %s", target_id)
            self.data_channel.on_broadcast_data = self._on_power_data
            self.data_channel.set_id(target_id, 11, 0)
            self.data_channel.set_period(8182)
            self.data_channel.set_rf_freq(57)

            time.sleep(0.2)
            self.data_channel.open()
            logger.info("Channel %s Open Command Sent.", target_id)

            with self.lock:
                self.shared_state.ant_status = "Connected"

        except Exception as e:
            self.health.report("SETTING_APPLY_ERROR", str(e))

    def _reset_histories(self):
        """Clear accumulated power history for both page types."""
        self._last_10_power = None
        self._last_10_time = 0
        self._last_12_event = None
        self._last_12_torque = None
        self._last_12_period = None
        self._last_12_power = None
        self._last_12_time = 0

    def _on_power_data(self, data):
        """
        Parse incoming ANT+ broadcast data.

        Page 0x10: instantaneous power [W] from bytes 6-7.
        Page 0x12: torque-based power computed from delta torque and delta period.
        """
        try:
            payload = getattr(data, 'payload', data)
            if len(payload) < 8:
                return

            now = time.time()
            page = payload[0]

            with self.lock:
                self.shared_state.last_data_time = now
                self.shared_state.ant_status = "Connected"

            if page == 0x10:
                power = payload[6] | (payload[7] << 8)
                if 0 <= power < 5000:
                    self._last_10_power = float(power)
                    self._last_10_time = now
                return

            if page == 0x12:
                event = payload[1]
                period = payload[4] | (payload[5] << 8)
                torque = payload[6] | (payload[7] << 8)

                if self._last_12_event is None:
                    self._last_12_event = event
                    self._last_12_torque = torque
                    self._last_12_period = period
                    return

                delta_event = (event - self._last_12_event) & 0xFF
                delta_torque = (torque - self._last_12_torque) & 0xFFFF
                delta_period = (period - self._last_12_period) & 0xFFFF

                self._last_12_event = event
                self._last_12_torque = torque
                self._last_12_period = period

                if delta_event == 0 or delta_period == 0:
                    return

                power = (delta_torque * 128.0 * math.pi) / delta_period

                if 0 <= power < 5000:
                    self._last_12_power = power
                    self._last_12_time = now
                return

        except Exception as e:
            self.health.report("DATA_HANDLER_EXCEPTION", str(e))

    def _select_best_power(self):
        """
        Write the most recent valid power reading to shared_state.
        Page 0x10 (2s window) takes priority over page 0x12 (4s window).
        """
        now = time.time()
        chosen = None
        if self._last_10_power is not None and now - self._last_10_time < 2.0:
            chosen = self._last_10_power
        elif self._last_12_power is not None and now - self._last_12_time < 4.0:
            chosen = self._last_12_power

        if chosen is not None:
            with self.lock:
                self.shared_state.actual_power = float(chosen)

    def _cleanup(self):
        """Close the ANT+ channel and stop the node gracefully."""
        if self.data_channel:
            try:
                self.data_channel.close()
            except Exception as e:
                self.health.report("CLEAN_CH_ERR", str(e))

        if self.node:
            try:
                self.node.stop()
                if self.node_thread and self.node_thread.is_alive():
                    self.node_thread.join(timeout=2.0)
            except Exception as e:
                self.health.report("CLEAN_NODE_ERR", str(e))