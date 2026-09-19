########################
# designer.py
########################
import datetime
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from numba import njit
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from scipy.optimize import minimize

from core.data_manager import (
    build_course_profile,
    save_strategy_to_json,
    unpack_input_data,
)
from core.git_info import get_git_commit_hash, is_relevant_code_dirty
from core.io_config import BASE_STRATEGIES_DIR, find_strategy_json_path
from core.logging_setup import configure_logging
from core.schema import PowerBlocks, RunSettings
from core.simulators import resolve_simulator
from eidos.apps.generator import create_json_output_dict
from eidos.lib.branding import window_title
from eidos.lib.power_profile_canvas import DESIGNER_COLOR_HEX, PowerProfileCanvas

logger = logging.getLogger(__name__)


def _can_split_segment(current_lengths: list[float], idx: int, l_min: float) -> bool:
    """
    Whether current_lengths[idx] can be split into two segments without
    creating any segment shorter than l_min anywhere in the strategy.

    Two independent conditions, both required:
      - LOCAL: the segment being split must itself hold two l_min-sized
        pieces (>= 2*l_min) -- otherwise the split is forced to produce
        one piece < l_min regardless of where the split point falls.
      - GLOBAL: course_distance must still exceed (n_seg + 1) * l_min
        after n_seg grows by one from this split -- otherwise the course
        no longer has enough total distance for every segment to reach
        l_min, even though this split's own two pieces are individually
        fine. A segment can individually satisfy the LOCAL condition
        while the strategy as a whole has no slack left for one more
        segment -- these are genuinely different constraints, not one
        implying the other.
    """
    if current_lengths[idx] < 2 * l_min:
        return False
    course_distance = np.sum(current_lengths)
    n_seg = len(current_lengths)
    return course_distance > (n_seg + 1) * l_min


# ----------------------------------------------------------------
# View/Controller
# ----------------------------------------------------------------
class StrategyDesignerController:
    """
    Controller layer for the interactive strategy designer.
    Handles mouse events, overlay drawing, and coordination between
    the model (StrategyDesignerModel) and the canvas (PowerProfileCanvas).
    """
    def __init__(self, base_data: dict, target_canvas: PowerProfileCanvas, parent_viewer=None):
        from eidos.apps.designer import StrategyDesignerModel
        self.model = StrategyDesignerModel(base_data)
        self.canvas = target_canvas
        self.parent_viewer = parent_viewer

        # UI state
        self.dragging_idx = -1
        self.dragging_mode = None
        self.hud_offset = QPoint(15, 25)
        self.is_dragging_hud = False
        self.last_mouse_pos = QPoint()
        self.hover_idx = -1

        # Button hit areas
        self.save_btn_rect = QRect()
        self.refine_btn_rect = QRect()
        self.reset_btn_rect = QRect()
        # Button press/hover state, by name ('save'/'refine'/'reset'), or None
        self.active_btn: str | None = None
        self.hover_btn: str | None = None

    def connect_canvas(self):
        """Hook into the canvas painter, mouse/leave events, then run initial simulation."""
        self.canvas.external_painter = self.draw_overlay
        self.canvas.mousePressEvent = self.on_press
        self.canvas.mouseMoveEvent = self.on_move
        self.canvas.mouseReleaseEvent = self.on_release
        self.canvas.leaveEvent = self.on_leave
        self.canvas.setMouseTracking(True)
        self.model.update_simulation()
        self.sync_to_viewer_limits()

    # --- Mouse Event Handlers ---
    def on_press(self, event):
        """Arm a HUD button press (fired on release), or handle HUD dragging
        and graph segment manipulation."""
        pos = event.position()
        px, py = int(pos.x()), int(pos.y())

        if self.save_btn_rect.contains(px, py):
            self.active_btn = 'save'; self.canvas.update(); return
        if self.refine_btn_rect.contains(px, py):
            self.active_btn = 'refine'; self.canvas.update(); return
        if self.reset_btn_rect.contains(px, py):
            self.active_btn = 'reset'; self.canvas.update(); return

        rect_hud_ref = self.canvas.subplot_rects.get('Cumulative')
        if rect_hud_ref and QRect(rect_hud_ref.left() + self.hud_offset.x(), rect_hud_ref.top() + self.hud_offset.y(), 280, 110).contains(px, py):
            self.is_dragging_hud = True; self.last_mouse_pos = QPoint(px, py); return

        rect_p = self.canvas.subplot_rects.get('Power')
        limits = self.canvas.limits.get('Power')
        if not rect_p or not rect_p.contains(px, py) or not limits: return

        val_x, val_y = self.canvas.map_widget_to_data(px, py, rect_p, limits)

        # Convert pixel tolerances to physical units
        x0, y0 = self.canvas.map_widget_to_data(px, py, rect_p, limits)
        x_edge, y_edge = self.canvas.map_widget_to_data(px + 8, py + 15, rect_p, limits)

        tol_x = abs(x_edge - x0)   # physical distance corresponding to 8 px
        tol_y = abs(y_edge - y0)   # physical power corresponding to 15 px

        boundaries = np.cumsum(self.model.current_lengths)

        # Determine hit target in current axis mode
        if self.canvas.is_time_mode:
            out = self.model.last_out
            if out is None: return
            b_times = np.interp(boundaries[:-1], out.x_traj, out.t_traj)
            diffs_phys = np.abs(b_times - val_x)
            data_x = np.interp(val_x, out.t_traj, out.x_traj)
        else:
            diffs_phys = np.abs(boundaries[:-1] - val_x)
            data_x = val_x

        idx = np.searchsorted(boundaries, data_x)
        idx = min(idx, len(self.model.current_powers) - 1)
        diff_y_phys = np.abs(val_y - self.model.current_powers[idx])

        near_power = diff_y_phys < tol_y
        near_boundary = len(diffs_phys) > 0 and np.min(diffs_phys) < tol_x

        if event.button() == Qt.RightButton:
            l_min = self.model.base_run_data['input']['settings']['run']['seg_length_min']
            # Decide merge-vs-split from t_idx's actual neighbors, not idx --
            # idx (searchsorted) can flip between the two segments straddling
            # a boundary on sub-pixel rounding, and near_power is unreliable
            # for a thin segment (its flat bar and both risers can sit within
            # a few pixels of each other). If either neighbor is too short to
            # ever satisfy _can_split_segment, split can't be the intent.
            t_idx = np.argmin(diffs_phys) if near_boundary else -1
            neighbors_splittable = (
                _can_split_segment(self.model.current_lengths, t_idx, l_min)
                and _can_split_segment(self.model.current_lengths, t_idx + 1, l_min)
            ) if near_boundary else True

            if near_boundary and (not near_power or not neighbors_splittable):
                # Merge segment at boundary
                self.model.current_lengths[t_idx] += self.model.current_lengths[t_idx + 1]
                self.model.current_lengths.pop(t_idx + 1); self.model.current_powers.pop(t_idx + 1)
            else:
                # Split segment at click position
                if not _can_split_segment(self.model.current_lengths, idx, l_min):
                    return
                orig_len = self.model.current_lengths[idx]
                prev_b = boundaries[idx-1] if idx > 0 else 0
                split_pos = max(l_min, min(data_x - prev_b, orig_len - l_min))
                self.model.current_lengths[idx] = split_pos
                self.model.current_lengths.insert(idx + 1, orig_len - split_pos)
                self.model.current_powers.insert(idx + 1, self.model.current_powers[idx])
            self.model.update_simulation(); self.canvas.update(); return

        if event.button() == Qt.LeftButton:
            if near_boundary and not near_power:
                self.dragging_idx = np.argmin(diffs_phys); self.dragging_mode = 'boundary'
            else:
                self.dragging_idx = idx; self.dragging_mode = 'power'

    def on_move(self, event):
        """Update HUD position, drag a boundary/power value, or track hover for
        the HUD readout and the HUD buttons."""
        pos = event.position()
        px, py = int(pos.x()), int(pos.y())
        self._update_btn_hover(px, py)
        if self.is_dragging_hud:
            diff = QPoint(px, py) - self.last_mouse_pos
            self.hud_offset += diff; self.last_mouse_pos = QPoint(px, py); self.canvas.update(); return

        if self.dragging_idx == -1:
            self._update_hover(px, py)

        if self.dragging_idx != -1:
            rect = self.canvas.subplot_rects.get('Power'); limits = self.canvas.limits.get('Power')
            if not rect or not limits: return
            data_x, data_y = self.canvas.map_widget_to_data(px, py, rect, limits)
            target_dist = data_x
            if self.canvas.is_time_mode:
                out = self.model.last_out
                if out: target_dist = np.interp(data_x, out.t_traj, out.x_traj)

            if self.dragging_mode == 'boundary':
                l_min = self.model.base_run_data['input']['settings']['run']['seg_length_min']
                boundaries = np.cumsum(self.model.current_lengths)
                prev = boundaries[self.dragging_idx - 1] if self.dragging_idx > 0 else 0
                nxt = boundaries[self.dragging_idx + 1]
                new_b = max(prev + l_min, min(target_dist, nxt - l_min))
                self.model.current_lengths[self.dragging_idx] = new_b - prev
                self.model.current_lengths[self.dragging_idx + 1] = nxt - new_b
            elif self.dragging_mode == 'power':
                self.model.current_powers[self.dragging_idx] = max(0.0, min(1000.0, data_y))

            self.model.update_simulation(); self.canvas.update()

    def _btn_rect(self, name: str) -> QRect:
        return {'save': self.save_btn_rect, 'refine': self.refine_btn_rect, 'reset': self.reset_btn_rect}[name]

    def _run_btn_action(self, name: str):
        """Run the action for HUD button `name`, invoked on release."""
        if name == 'save':
            self.execute_save_and_exit()
        elif name == 'refine':
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                if self.model.refine_strategy():
                    self.sync_to_viewer_limits(); self.canvas.update()
            finally:
                QApplication.restoreOverrideCursor()
        elif name == 'reset':
            self.model.reset_to_initial(); self.sync_to_viewer_limits(); self.canvas.update()

    def on_release(self, event):
        """Fire a HUD button's action if released back over the button it was
        pressed on, otherwise finalize a drag and sync axis limits."""
        if self.active_btn is not None:
            pos = event.position()
            px, py = int(pos.x()), int(pos.y())
            name = self.active_btn
            self.active_btn = None
            if self._btn_rect(name).contains(px, py):
                self._run_btn_action(name)
            # Always repaint here, not inside _run_btn_action: a no-op
            # refine skips its own canvas.update(), which would otherwise
            # leave the pressed darkening stuck.
            self.canvas.update()
            return
        if self.dragging_idx != -1: self.sync_to_viewer_limits()
        self.dragging_idx = -1; self.dragging_mode = None; self.is_dragging_hud = False

    def on_leave(self, event):
        """Clear hover/press state when the cursor leaves the canvas; a
        held HUD button is cancelled, not fired."""
        if self.hover_idx != -1:
            self.hover_idx = -1; self.canvas.update()
        if self.hover_btn is not None:
            self.hover_btn = None; self.canvas.update()
        if self.active_btn is not None:
            self.active_btn = None; self.canvas.update()

    def _update_btn_hover(self, px: int, py: int):
        """Track which HUD button (if any) is under the cursor, for the same
        hover-darkening EIDOS^TT's QPushButton[...]:hover style gives real
        buttons elsewhere."""
        new_hover = next((n for n in ('save', 'refine', 'reset') if self._btn_rect(n).contains(px, py)), None)
        if new_hover != self.hover_btn:
            self.hover_btn = new_hover; self.canvas.update()

    def _update_hover(self, px: int, py: int):
        """Set hover_idx to the segment under the cursor in the Power subplot
        (by X-position only; unlike on_press's click hit-test, no y-tolerance)."""
        rect_p = self.canvas.subplot_rects.get('Power')
        limits = self.canvas.limits.get('Power')
        new_hover = -1
        if rect_p and limits and rect_p.contains(px, py) and self.model.current_lengths:
            val_x, _ = self.canvas.map_widget_to_data(px, py, rect_p, limits)
            if self.canvas.is_time_mode:
                out = self.model.last_out
                data_x = np.interp(val_x, out.t_traj, out.x_traj) if out is not None else None
            else:
                data_x = val_x
            if data_x is not None:
                boundaries = np.cumsum(self.model.current_lengths)
                idx = np.searchsorted(boundaries, data_x)
                new_hover = int(min(idx, len(self.model.current_powers) - 1))
        if new_hover != self.hover_idx:
            self.hover_idx = new_hover; self.canvas.update()

    def draw_overlay(self, painter: QPainter):
        """Draw the Designer strategy overlay and HUD status panel onto the canvas."""
        out = self.model.last_out
        if out is None: return

        course_in = self.model.base_run_data['input']['data']['course_profile']
        dist_list = np.array(course_in['distance_p_m_list'])
        t_course = np.interp(dist_list, out.x_traj, out.t_traj)

        physiological = self.model.base_run_data['input']['settings']['physiological']
        overlay_data = {
            'COLOR_HEX': DESIGNER_COLOR_HEX,
            'TIME': out.t_traj, 'DISTANCE': out.x_traj, 'POWER': out.p_traj,
            'SPEED': out.v_traj, 'WPRIME': out.w_traj,
            # Every SIMULATOR_REGISTRY entry's own PhysiologicalSettings is
            # required to have cp/w_prime (see core.schema.
            # PhysiologicalSettingsBase) -- unconditional indexing, not
            # .get(): a strategy whose simulator somehow lacks either would
            # indicate a real bug, not a normal condition to degrade
            # around quietly.
            'CP_REF': physiological['cp'],
            'W_PRIME_MAX': physiological['w_prime'],
            'TargetL_m': np.array(self.model.current_lengths),
            'TargetP_W': np.array(self.model.current_powers),
            'SLOPE': np.array(course_in['slope_ratio_list']),
            'ALTITUDE': np.array(course_in['altitude_list']),
            'COURSE_DIST': dist_list, 'TIME_COURSE': t_course, 'TotalTime_s': out.finish_time
        }
        self.canvas.draw_data_on_all_subplots(painter, [overlay_data])

        # HUD: status display and control buttons
        rect_cum = self.canvas.subplot_rects.get('Cumulative')
        if not rect_cum: return
        hud_x, hud_y = rect_cum.left() + self.hud_offset.x(), rect_cum.top() + self.hud_offset.y()
        hover_line = None
        if 0 <= self.hover_idx < len(self.model.current_lengths):
            hover_line = (f"Seg {self.hover_idx + 1}/{len(self.model.current_lengths)}:  "
                          f"L={self.model.current_lengths[self.hover_idx]:.0f}m   "
                          f"P={self.model.current_powers[self.hover_idx]:.0f}W")
        hud_w, hud_h = 280, 110 + (22 if hover_line else 0)

        is_pnlty = self.model.current_penalty > 1.001
        delta = self.model.current_time - self.model.initial_time
        if is_pnlty:
            color = QColor(255, 30, 30); status_txt = "PENALTY !"
        elif delta < -0.001:
            color = QColor(60, 190, 255); status_txt = "IMPROVED"
        elif delta < 0.001:
            color = Qt.GlobalColor.white; status_txt = "BASELINE"
        elif delta < 1.0:
            color = QColor(30, 255, 30); status_txt = "FAIR"
        else:
            color = QColor(255, 165, 30); status_txt = "DEGRADED"

        # Background panel
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(0, 0, 0, 180))
        painter.drawRoundedRect(hud_x, hud_y, hud_w, hud_h, 5, 5)

        painter.setPen(QPen(color))
        painter.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        txt = (f"{status_txt}\n"
               f"Time: {self.model.current_time:.2f}s\n"
               f"Diff: {delta:+.3f}s")
        painter.drawText(hud_x + 10, hud_y + 15, 170, 90, Qt.AlignmentFlag.AlignLeft, txt)

        # Control buttons
        btn_x, btn_w, btn_h = hud_x + 185, 85, 26
        self.save_btn_rect   = QRect(btn_x, hud_y + 10, btn_w, btn_h)
        self.refine_btn_rect = QRect(btn_x, hud_y + 42, btn_w, btn_h)
        self.reset_btn_rect  = QRect(btn_x, hud_y + 74, btn_w, btn_h)

        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        btn_list = [
            ('save',   self.save_btn_rect,   "SAVE & EXIT", QColor(46, 125, 50)),
            ('refine', self.refine_btn_rect, "REFINE",      QColor(40, 100, 200)),
            ('reset',  self.reset_btn_rect,  "RESET",       QColor(150, 40, 40))
        ]
        for name, r, t, c in btn_list:
            # Matches EIDOS^TT's QPushButton[...]:hover/:pressed convention
            # elsewhere: darker while held, lighter darkening on hover.
            if self.active_btn == name:
                c = c.darker(150)
            elif self.hover_btn == name:
                c = c.darker(115)
            painter.setBrush(c); painter.setPen(Qt.PenStyle.NoPen); painter.drawRoundedRect(r, 3, 3)
            painter.setPen(Qt.GlobalColor.white); painter.drawText(r, Qt.AlignmentFlag.AlignCenter, t)

        if hover_line:
            painter.setPen(QPen(Qt.GlobalColor.yellow))
            painter.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
            painter.drawText(hud_x + 10, hud_y + 100, hud_w - 20, 20,
                              Qt.AlignmentFlag.AlignLeft, hover_line)

    def sync_to_viewer_limits(self):
        """Recompute axis limits using the current simulation output as a temporary record."""
        out = self.model.last_out
        if out is None: return
        course_data = self.model.base_run_data['input']['data']['course_profile']
        physiological = self.model.base_run_data['input']['settings']['physiological']
        temporary_record = {
            'TIME': np.ravel(out.t_traj), 'DISTANCE': np.ravel(out.x_traj), 'POWER': np.ravel(out.p_traj),
            'WPRIME': np.ravel(out.w_traj), 'SPEED': np.ravel(out.v_traj),
            'SLOPE': np.ravel(course_data['slope_ratio_list']), 'ALTITUDE': np.ravel(course_data['altitude_list']),
            'CP_REF': physiological['cp'],
            'W_PRIME_MAX': physiological['w_prime'],
            'TargetP_W': np.ravel(self.model.current_powers)
        }
        if self.canvas.calculate_plot_limits(temporary_record=temporary_record):
            self.canvas.request_refresh()
        else:
            self.canvas.update()

    def execute_save_and_exit(self):
        """Save the current strategy as a strategy JSON and return to the Viewer."""
        out = self.model.last_out
        if not out: return

        run_set_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_seg = len(self.model.current_powers)
        seed = 0
        # A Designer save may be hand-edited, Refine-polished, or both, so
        # settings.engine.optimizer's original value (paired with
        # optimizer_params/versions.optimizer) can no longer accurately
        # claim how target_power_list/target_length_list were produced --
        # override it. git_state is re-stamped fresh too: Designer always
        # re-simulates with whatever code is checked out right now, not
        # necessarily the commit the base strategy was generated at (see
        # core.git_info's module docstring).
        input_data = self.model.base_run_data['input']
        saved_input = {
            **input_data,
            "versions": {**input_data['versions'], "optimizer": "N/A"},
            "git_state": {"commit_hash": get_git_commit_hash(), "is_dirty": is_relevant_code_dirty()},
            "settings": {
                **input_data['settings'],
                "engine": {**input_data['settings']['engine'], "optimizer": "manual", "optimizer_params": {}},
            },
        }
        final_data = {
            "run_set_id": run_set_id,
            "input": saved_input,
            "output": create_json_output_dict(n_seg, seed, 1, True, out.finish_time, 1, 1,
                                              np.array(self.model.current_powers), np.array(self.model.current_lengths))
        }

        strategy_set_dir = self.model.base_run_data.get('strategy_set_dir', 'manual_design')
        strategy_set_dir_path = os.path.join(BASE_STRATEGIES_DIR, strategy_set_dir)
        os.makedirs(strategy_set_dir_path, exist_ok=True)
        file_path = os.path.join(strategy_set_dir_path, f"strategy_{run_set_id}_N{n_seg}_S{seed}.json")

        save_strategy_to_json(final_data, file_path)
        logger.info("Saved Designer Strategy: %s", file_path)
        self.update_index_parquet(final_data, strategy_set_dir_path)

        if self.parent_viewer:
            # Called from Viewer: unhook painter and mouse events, then re-initialize
            self.canvas.external_painter = None
            self.canvas.mousePressEvent = self.canvas.__class__.mousePressEvent
            self.canvas.mouseMoveEvent = self.canvas.__class__.mouseMoveEvent
            self.canvas.mouseReleaseEvent = self.canvas.__class__.mouseReleaseEvent

            if hasattr(self.parent_viewer, 'initialize'):
                self.parent_viewer.initialize(scroll_to_key=(str(run_set_id), str(n_seg), str(seed)))
            logger.info("Designer session ended. Returning to Viewer.")
        else:
            # Standalone: close the window
            top_level = self.canvas.window()
            if top_level:
                top_level.close()

    def update_index_parquet(self, final_data, full_base_path):
        """Append the new strategy record to the strategy set's _index.parquet file."""
        index_path = os.path.join(full_base_path, "_index.parquet")
        if not os.path.exists(index_path):
            logger.warning(
                "No _index.parquet in %s; this new record won't show "
                "in the Viewer's index-based list until you run "
                "'hyle-strategy-doctor --fix' to rebuild it.",
                full_base_path,
            )
            return
        try:
            from core.pydantic_mapper import extract_flat_data
            df_idx = pd.read_parquet(index_path)
            df_new = pd.DataFrame([extract_flat_data(final_data)])
            pd.concat([df_idx, df_new], ignore_index=True).to_parquet(index_path, index=False)
            logger.info("Index updated.")
        except Exception:
            logger.exception("Failed to update strategy index")

# ----------------------------------------------------------------
# Refine: local power polish (Nelder-Mead)
# ----------------------------------------------------------------
# Deliberately independent of eidos.lib.optimizer.OPTIMIZER_REGISTRY:
# Refine polishes the *current*, possibly hand-edited strategy around its
# own point -- a fixed operation unrelated to whichever optimizer (if
# any) originally produced the strategy being edited, so it must not
# depend on the strategy's own settings.engine.optimizer/optimizer_params
# (see StrategyDesignerModel.refine_strategy).
REFINE_XATOL_W = 1.0
REFINE_FATOL_S = 0.01
REFINE_RESTART_EPS_S = 0.01 * 1e-3
REFINE_MAX_RESTARTS = 100
REFINE_IMPROVEMENT_EPSILON_S = 0.01


@njit
def _refine_objective(powers: np.ndarray, fixed_lengths: np.ndarray, physics, kernel):
    """Score a power allocation (segment lengths fixed) for Nelder-Mead."""
    power_blocks = PowerBlocks(power=powers, length=fixed_lengths)
    output = kernel(0.0, power_blocks, physics, False, False, True)
    return output.finish_time * output.penalty_factor


def _refine_powers_locally(initial_powers, fixed_lengths, seg_power_min, seg_power_max, physics, kernel):
    """
    Polish power allocation with Nelder-Mead, holding segment lengths fixed.

    Iteratively restarts the simplex rather than seeking convergence in a
    single run: each restart typically improves on the last by far less
    than REFINE_FATOL_S (a single run's own convergence tolerance) -- the
    real gain comes from accumulating many such small improvements across
    restarts. Stops once improvement falls below the much finer
    REFINE_RESTART_EPS_S; stopping at REFINE_FATOL_S itself would cut
    this accumulation off after just 1-2 restarts.
    """
    def objective_unscaled(p_actual):
        p_clipped = np.clip(p_actual, seg_power_min, seg_power_max)
        return _refine_objective(p_clipped, fixed_lengths, physics, kernel)

    p_bounds = [(seg_power_min, seg_power_max)] * len(initial_powers)
    current_p = initial_powers
    best_time = float('inf')
    for i in range(REFINE_MAX_RESTARTS):
        res = minimize(
            fun=objective_unscaled, x0=current_p, method='Nelder-Mead', bounds=p_bounds,
            options={'adaptive': True, 'xatol': REFINE_XATOL_W, 'fatol': REFINE_FATOL_S},
        )
        improvement = best_time - res.fun
        if res.fun < best_time:
            best_time = res.fun
            current_p = res.x
        if i > 0 and improvement < REFINE_RESTART_EPS_S:
            break
    return np.clip(current_p, seg_power_min, seg_power_max), best_time


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------
class StrategyDesignerModel:
    """
    Model layer: holds the current strategy state and runs physics simulations.
    Exposes current_powers and current_lengths as mutable lists that the
    controller edits directly in response to user interaction.
    """
    def __init__(self, base_run_data: dict):
        self.base_run_data = base_run_data
        self.simulator_spec = resolve_simulator(base_run_data['input']['settings']['engine']['simulator'])
        self.physics_params = self._build_physics_params(base_run_data)
        res = base_run_data['output']['results']['strategy']
        self.initial_powers = list(res['target_power_list'])
        self.initial_lengths = list(res['target_length_list'])
        self.initial_time = base_run_data['output']['results']['kpis']['total_time_s']
        self.current_powers = list(self.initial_powers)
        self.current_lengths = list(self.initial_lengths)
        self.current_time, self.current_penalty, self.current_score = 0.0, 1.0, 0.0
        self.last_out = None
        self.update_simulation()

    def reset_to_initial(self):
        """Restore powers and lengths to the original loaded strategy."""
        self.current_powers, self.current_lengths = list(self.initial_powers), list(self.initial_lengths)
        self.update_simulation()

    def _build_physics_params(self, rd: dict):
        """Construct PhysicsParams from the strategy JSON input section, via the
        resolved simulator_spec's own builder (whatever shape it expects).

        model_construct(), not the normal constructor -- s['physical']/
        s['physiological'] are already-validated (round-tripped from a
        strategy JSON originally built from validated settings), so
        re-running PhysicalSettings.cda_yaw_table_filename's CSV file-I/O
        validator here has no correctness benefit, only cost. See
        core.physics_overrides' module docstring for the same reasoning."""
        s, course_data = rd['input']['settings'], rd['input']['data']['course_profile']
        physical_s = s['physical']
        physical_settings = self.simulator_spec.physical_param_model.model_construct(**physical_s)
        course = build_course_profile(course_data)
        course = self.simulator_spec.recompute_course_physics(course, physical_settings)
        return self.simulator_spec.build_physics_params(
            physical_settings,
            self.simulator_spec.physiological_param_model.model_construct(**s['physiological']),
            RunSettings(**s['run']),
            course,
        )

    def update_simulation(self):
        """Run the physics simulator with the current strategy and update KPIs."""
        power_blocks = PowerBlocks(power=np.asarray(self.current_powers, dtype=np.float64),
                                   length=np.asarray(self.current_lengths, dtype=np.float64))
        self.last_out = self.simulator_spec.kernel(0.0, power_blocks, self.physics_params, True, False, True)
        self.current_time = self.last_out.finish_time
        self.current_penalty = self.last_out.penalty_factor
        self.current_score = self.current_time * self.current_penalty

    def refine_strategy(self):
        """
        Locally polish the current strategy's power allocation via
        _refine_powers_locally (segment lengths held fixed). Returns True
        if the refined strategy improved the objective.
        """
        run_s = self.base_run_data['input']['settings']['run']
        refined_powers, obj_val = _refine_powers_locally(
            np.array(self.current_powers), np.array(self.current_lengths),
            run_s['seg_power_min'], run_s['seg_power_max'],
            self.physics_params, self.simulator_spec.kernel,
        )
        if (self.current_score - obj_val) > REFINE_IMPROVEMENT_EPSILON_S:
            self.current_powers = list(refined_powers)
            self.update_simulation(); return True
        return False

# ----------------------------------------------------------------
# Main Window & Entry Point
# ----------------------------------------------------------------
class DesignerWindow(QMainWindow):
    """Standalone window wrapping the Designer canvas and controller."""
    def __init__(self, base_data: dict, existing_canvas=None, parent_viewer=None):
        super().__init__()
        self.setWindowTitle(window_title("Designer"))
        self.canvas = existing_canvas if existing_canvas else PowerProfileCanvas(None)
        self.controller = StrategyDesignerController(base_data, self.canvas, parent_viewer)
        self.canvas.model = self.controller.model
        self.controller.connect_canvas()
        if not existing_canvas:
            container = QWidget(); layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(self.canvas)
            self.setCentralWidget(container); self.resize(1200, 800)

def execute_designer():
    """Entry point: load a strategy JSON and launch the Designer window."""
    if len(sys.argv) < 5:
        print("Usage: eidos-designer <StrategySetDir> <RunID> <Nseg> <Seed>")
        sys.exit(1)
    strategy_set_dir, run_id, n_seg, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    json_path = find_strategy_json_path(strategy_set_dir, run_id, n_seg, seed)

    with open(json_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    base_data = unpack_input_data(base_data)

    base_data['strategy_set_dir'] = strategy_set_dir  # needed for save path resolution
    app = QApplication(sys.argv)
    window = DesignerWindow(base_data)
    window.show()
    sys.exit(app.exec())

def main() -> None:
    configure_logging()
    execute_designer()


if __name__ == '__main__':
    main()