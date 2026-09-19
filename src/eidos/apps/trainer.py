#######################
# trainer.py
#######################
import glob
import importlib
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage, Sport, SubSport
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import eidos.lib.view_opengl as view
from core.activity_parser import LEAD_IN_TIME_S, TRAIL_OUT_TIME_S
from core.data_manager import (
    build_course_profile,
    unpack_input_data,
)
from core.fit_combiner import combine_fit_files
from core.git_info import check_reproducibility
from core.io_config import (
    BASE_ACTIVITIES_DIR,
    create_strategy_export_dir,
)
from core.logging_setup import configure_logging, log_subbanner
from core.schema import PowerBlocks, RunSettings
from core.simulators import resolve_simulator
from eidos.lib.ant_receiver import (
    ANTPowerReceiver,  # project-internal ANT+ receiver module
)
from eidos.lib.branding import window_title

logger = logging.getLogger(__name__)


# --- Runtime state container ---
class SimulationState:
    """
    Mutable state container for the real-time simulation loop in the Trainer.

    Holds elapsed time, distance, velocity, W' balance (anaerobic energy reserve, J),
    and target power; updated each tick by the simulation thread.
    """
    def __init__(self):
        """Initialize all simulation state fields to their default values."""
        self.elapsed_time = 0.0
        self.distance = 0.0
        self.velocity = 0.0
        self.w_prime_bal = 0.0
        self.target_power = 0.0
        self.actual_power = 0.0 
        self.course_coords = None  
        self.course_headings_deg = None  
        self.distance_step = 1.0
        self.segment_boundaries = [] 
        self.target_p_list = []      
        self.road_vertices = None  
        self.side_lines = None 
        self.reset_requested = False
        self.is_running = True
        self.start_requested = False
        self.waiting_for_start = True
        self.countdown_value = 0
        self.finish_time_str = ""
        self.ant_status = "Scanning"
        self.ant_recv = None
        self.last_data_time = 0.0
        self.found_devices = []
        self.target_device_id = 0 
        self.ghost_x_traj = None
        self.ghost_v_traj = None
        self.ghost_t_traj = None
        self.ghost_curr_x = 0.0    
        self.ghost_curr_v = 0.0
        self.v_wind = 0.0
        self.d_wind = 0.0
        self.heading_rad = 0.0  # rider's current heading in radians
        self.center_v = None    # center-line vertices (1D float32 array)
        self.rv_draw = None     # road cross-line vertices (decimated array)

class SimulationResetException(Exception):
    """Custom exception used to interrupt the physics loop on a reset request."""
    pass

shared_state = SimulationState()
state_lock = threading.Lock()

def init_graphics_settings():
    """Configure OpenGL depth test and linear fog for the 3D scene."""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_FOG)
    
    # Unify background and fog colour
    fog_color = [0.05, 0.05, 0.1, 1.0]
    glClearColor(*fog_color)
    glFogfv(GL_FOG_COLOR, fog_color)
    
    # Linear fog (simplest and most intuitive calculation)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 50.0)      # start fading at 50 m
    glFogf(GL_FOG_END, 500.0)       # fully obscured at 500 m
    glHint(GL_FOG_HINT, GL_NICEST)  # maximum quality hint

def real_time_sync_logic(p_block, t, v, x, W):
    """Sync callback injected into the simulator: sleeps until wall-clock matches simulation time, then exchanges power values with shared_state."""
    if shared_state.reset_requested:
        if hasattr(real_time_sync_logic, "start_wall"):
            delattr(real_time_sync_logic, "start_wall")
        raise SimulationResetException()

    if not hasattr(real_time_sync_logic, "start_wall"):
        real_time_sync_logic.start_wall = time.perf_counter()
    
    expected_wall = real_time_sync_logic.start_wall + t
    wait_wait = expected_wall - time.perf_counter()
    if wait_wait > 0: 
        time.sleep(wait_wait)

    real_time_sync_logic.last_phys_wall = time.perf_counter()
    
    with state_lock:
        p_input = shared_state.actual_power
        shared_state.elapsed_time, shared_state.distance = t, x
        shared_state.velocity, shared_state.w_prime_bal = v, W
        shared_state.target_power = p_block

        if shared_state.ghost_x_traj is not None:
            shared_state.ghost_curr_x = np.interp(t, shared_state.ghost_t_traj, shared_state.ghost_x_traj)
            shared_state.ghost_curr_v = np.interp(t, shared_state.ghost_t_traj, shared_state.ghost_v_traj)

    return p_input

def angle_lerp(a0, a1, w):
    """Interpolate between two angles a0 and a1 with weight w, handling wrap-around correctly."""
    da = (a1 - a0 + 180.0) % 360.0 - 180.0
    return a0 + da * w

def display():
    """GLUT display callback: interpolate physics state, render the 3D scene, and swap buffers."""
    if not hasattr(display, "prev_time"):
        display.prev_time = time.perf_counter()
        display.fps = 0.0

    now = time.perf_counter()
    frame_dt = now - display.prev_time
    display.prev_time = now
    if frame_dt > 0:
        display.fps = display.fps * 0.9 + (1.0 / frame_dt) * 0.1

    # --- 1. Clear screen ---
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # --- 2. 3D projection setup (dynamic aspect ratio) ---
    viewport = glGetIntegerv(GL_VIEWPORT)
    vw, vh = viewport[2], viewport[3]
    aspect_ratio = vw / float(vh) if vh != 0 else 1.0

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, aspect_ratio, 0.1, 15000.0) 
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # --- 3. Physics state interpolation and camera calculation ---
    with state_lock:
        if shared_state.course_coords is None: return
        last_phys = getattr(real_time_sync_logic, "last_phys_wall", now)
        dt_fraction = now - last_phys
        if dt_fraction > 0.2: dt_fraction = 0.0
        
        # Predictive interpolation for the ego rider's position
        d_interp = shared_state.distance + shared_state.velocity * dt_fraction
        # Predictive interpolation for the ghost rider's position
        g_d_interp = None
        if shared_state.ghost_x_traj is not None:
            g_d_interp = shared_state.ghost_curr_x + shared_state.ghost_curr_v * dt_fraction

        d, stride = d_interp, shared_state.distance_step
        cx, cy, cz = shared_state.course_coords
        h_deg = shared_state.course_headings_deg
        
        f = d / stride
        i0 = int(f)
        if i0 >= len(cx) - 1:
            curr_x, curr_y, curr_z = cx[-1], cy[-1], cz[-1]
            rad = np.radians(h_deg[-1])
            ux, uz = np.sin(rad), -np.cos(rad)
        else:
            i1 = i0 + 1
            w = f - i0
            curr_x = cx[i0] * (1 - w) + cx[i1] * w
            curr_y = cy[i0] * (1 - w) + cy[i1] * w
            curr_z = cz[i0] * (1 - w) + cz[i1] * w
            h = angle_lerp(h_deg[i0], h_deg[i1], w)
            rad = np.radians(h)
            ux, uz = np.sin(rad), -np.cos(rad)
            
        v, t, W, p_target = shared_state.velocity, shared_state.elapsed_time, shared_state.w_prime_bal, shared_state.target_power
        boundaries = shared_state.segment_boundaries
        p_list = shared_state.target_p_list

    # Camera setup
    cam_x, cam_y, cam_z = curr_x - ux * 7.0, curr_y + 2.8, curr_z - uz * 7.0
    gluLookAt(cam_x, cam_y, cam_z, curr_x + ux * 40, curr_y + 1.2, curr_z + uz * 40, 0, 1, 0)

    # --- Draw environment grid ---
    view.draw_environment_grid(curr_x, curr_z)

    # --- Draw road ---
    view.draw_road()

    # --- Draw segment gates ---
    for i, b_dist in enumerate(boundaries):
        b_idx = min(int(b_dist / stride), len(cx) - 1)
        is_start = (i == 0)
        is_finish = (i == len(boundaries) - 1)
        if is_start:
            label = "START"
        elif is_finish:
            label = "FINISH"
        else:
            label = f"{p_list[i]:.0f}W"
        color = (0.0, 0.0, 1.0) if is_start else ((1.0, 0.8, 0.0) if is_finish else (1.0, 0.5, 0.0))
        view.draw_gate(cx[b_idx], cy[b_idx], cz[b_idx], color, label, h_deg[b_idx], is_start_gate=is_start)

    # --- 4.5 Draw ghost rider ---
    gx, gy, gz = None, None, None
    if g_d_interp is not None:
        gf = g_d_interp / stride
        gi0 = int(gf)
        if gi0 < len(cx) - 1:
            gi1 = gi0 + 1
            gw = gf - gi0
            gx = cx[gi0] * (1 - gw) + cx[gi1] * gw
            gy = cy[gi0] * (1 - gw) + cy[gi1] * gw
            gz = cz[gi0] * (1 - gw) + cz[gi1] * gw
    view.draw_ghost_rider(gx, gy, gz)

    # --- Draw ego rider ---
    view.draw_ego_rider(curr_x, curr_y, curr_z)

    # --- Draw wind vectors (hidden once the run has finished) ---
    with state_lock:
        if not shared_state.finish_time_str:
            view.draw_wind_vectors_3d(
                shared_state.v_wind,    # true wind speed
                shared_state.d_wind,    # true wind direction
                v,                      # rider's ground speed
                curr_x, curr_y, curr_z, # rider's current position
                rad                     # rider's heading in radians
            )
            view.draw_wind_particles(
                shared_state.v_wind, shared_state.d_wind, v, rad,
                curr_x, curr_y, curr_z, frame_dt
            )

    # --- Draw minimap ---
    with state_lock:
        # Pass ghost coordinates if a ghost rider is active
        gx_pos, gz_pos = (gx, gz) if 'gx' in locals() else (None, None)
        view.draw_minimap(cx, cz, curr_x, curr_z, gx_pos, gz_pos)

    # --- Draw HUD ---
    upcoming = [b for b in boundaries if b > d]
    view.draw_hud(t, d, v, W, p_target, upcoming[0] - d if upcoming else 0)

    glutSwapBuffers()

def timer(_):
    """GLUT timer callback: request a redisplay every 16 ms (~60 fps)."""
    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


def special_keys(key, x, y):
    """GLUT special-key callback: handle arrow keys for IF adjustment while at the start."""
    with state_lock:
        # 1. Allow IF adjustment only while waiting at the start
        if shared_state.waiting_for_start:
            if key == GLUT_KEY_UP:
                real_time_sync_logic.if_scale = min(1.0, real_time_sync_logic.if_scale + 0.05)
            elif key == GLUT_KEY_DOWN:
                real_time_sync_logic.if_scale = max(0.5, real_time_sync_logic.if_scale - 0.05)
            
            # Round and print only when a change occurred
            if key in [GLUT_KEY_UP, GLUT_KEY_DOWN]:
                real_time_sync_logic.if_scale = round(real_time_sync_logic.if_scale, 2)
                logger.info("Current IF Scale: %.2f", real_time_sync_logic.if_scale)

        # 2. Add any actions permitted during the ride below this line
        # if key == GLUT_KEY_LEFT: ...
    glutPostRedisplay()

def keyboard_keys(key, x, y):
    """GLUT keyboard callback: handle Space (start), Enter (reset), W/S (power), Esc (quit)."""
    code = ord(key)
    char = key.decode("utf-8").lower() if isinstance(key, bytes) else key.lower()

    if '1' <= char <= '5':
        idx = int(char) - 1
        with state_lock:
            if idx < len(shared_state.found_devices):
                new_id = shared_state.found_devices[idx]
                shared_state.target_device_id = new_id
                logger.info("Target Device Locked to: %s", new_id)
        return  # handled; no further processing

    # 1. Key actions
    if code == 13: # Enter
        with state_lock: shared_state.reset_requested = True
    elif char == ' ': # Space
        with state_lock:
            if shared_state.waiting_for_start:
                shared_state.start_requested = True    
    elif char == 'w':  # W key: increase power
        with state_lock: shared_state.actual_power += 10.0
    elif char == 's':  # S key: decrease power
        with state_lock: shared_state.actual_power = max(0.0, shared_state.actual_power - 10.0)
    
    # 2. Exit handling
    elif code == 27: # Esc
        log_subbanner(logger, "Finalizing and Exiting...")

        # Stop ANT+ receiver
        if shared_state.ant_recv:
            logger.info("Stopping ANT+ Receiver...")
            try:
                # Call the stop method implemented in eidos.lib.ant_receiver
                shared_state.ant_recv.stop_receiver()
                # Short timeout; 1 second is sufficient
                shared_state.ant_recv.join(timeout=1.0)
            except Exception as e:
                logger.error("Error during shutdown: %s", e)

        logger.info("ANT+ released. Program terminated.")

        # Merge FIT files
        log_subbanner(logger, "COMBINING: Merging trial FIT files")
        try:
            combine_fit_files(Path(BASE_ACTIVITIES_DIR) / "to_combine", Path(BASE_ACTIVITIES_DIR))
        except Exception as e:
            logger.error("Error combining FIT files: %s", e)
        logger.info("ANT+ released. Program terminated.")

        os._exit(0)

# eidos.apps.analyzer's find_course_matches (core.activity_parser) needs
# LEAD_IN_TIME_S of real history before the standing-start stop and
# TRAIL_OUT_TIME_S past the goal (see those constants' own docstrings for
# the full derivation) -- but the physics kernel only simulates the timed
# course segment itself (x=0 at t=0 through the goal), so save_to_fit's
# own trajectory has neither. LEAD_IN_PAD_S/TRAIL_OUT_PAD_S synthesize
# both margins as extra records, each with a small buffer over the bare
# minimum to clear find_course_matches's strict-inequality boundary
# check with room to spare against 1Hz sample-timing rounding.
#
# Start margin: stationary (v=0, fixed position) padding -- physically
# accurate for waiting at the line, and the standing start's own first
# samples (v~0) can't give a trustworthy extrapolated heading anyway.
#
# Goal margin: constant-velocity, constant-heading extrapolation from the
# trial's own last two real records (extrapolate_uniform_motion), not
# stationary padding -- a rider crossing the line at speed doesn't
# teleport to a dead stop at the goal coordinate, and padding frozen
# exactly there can tie or beat the real crossing sample's own
# distance-to-goal, misleading find_course_matches's Stage 1 (which picks
# a lap's goal crossing as the closest-to-goal sample in its near-goal
# cluster). Extrapolated motion instead keeps moving away from the goal
# each second, so it can't out-compete the real crossing.
#
# core.fit_combiner.combine_fit_files resolves any resulting timestamp
# overlap between adjacent trials (by shifting a trial's timestamps, never
# by dropping records).
LEAD_IN_PAD_S = LEAD_IN_TIME_S + 5.0
TRAIL_OUT_PAD_S = TRAIL_OUT_TIME_S + 5.0


def extrapolate_uniform_motion(prev: RecordMessage, last: RecordMessage, k: int) -> RecordMessage:
    """
    Constant-velocity, constant-heading extrapolation of a RecordMessage
    k seconds past `last`, continuing the prev->last step linearly in
    every field -- position, altitude, distance, and a speed re-derived
    from that same distance delta (not copied from last.speed) so every
    field agrees with the one straight-line model. power=0 (coasting).
    prev/last are assumed 1s apart (this project's 1 Hz cadence); k need
    not be.
    """
    d_lat, d_lon = last.position_lat - prev.position_lat, last.position_long - prev.position_long
    d_alt, d_dist = last.altitude - prev.altitude, last.distance - prev.distance
    msg = RecordMessage()
    msg.timestamp = last.timestamp + k * 1000
    msg.position_lat = last.position_lat + d_lat * k
    msg.position_long = last.position_long + d_lon * k
    msg.altitude = last.altitude + d_alt * k
    msg.distance = last.distance + d_dist * k
    msg.speed = d_dist  # d_dist is already a per-second rate (prev/last are 1s apart)
    msg.power = 0
    return msg


def save_to_fit(result, file_path, course_profile, start_time: datetime):
    """
    Save simulation result to a FIT file.
    Positions are stored in degrees; the fit-tool library converts
    to Semicircles internally.

    Prepends LEAD_IN_PAD_S seconds of synthetic stationary records before
    the timed start and appends TRAIL_OUT_PAD_S seconds of extrapolated
    (not stationary) records after the timed goal -- see those constants'
    own docstring for why they're built so differently.

    start_time: real wall-clock time the trial's physics actually began
    (captured by the caller right before the timed kernel.py_func call,
    not "now" -- this function is called after the trial has already
    finished, so using datetime.now() here would anchor every record to
    the trial's END instead of its start, shifting the whole trial into
    a fictional future window and corrupting inter-trial gaps in any
    FIT file these trials are later combined into; see
    core.fit_combiner).
    """
    # Follow sample code conventions: auto_define and min_string_size
    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    # 1. FileIdMessage
    file_id_msg = FileIdMessage()
    file_id_msg.type = 4  # Activity
    file_id_msg.manufacturer = 255 # Development
    file_id_msg.product_name = "EIDOS^TT Trainer"
    file_id_msg.serial_number = 1  # arbitrary value
    start_time_ms = round(start_time.timestamp() * 1000)
    file_id_msg.time_created = start_time_ms
    builder.add(file_id_msg)

    # Unpack trajectory arrays
    t_traj, x_traj, v_traj, p_traj = result.t_traj, result.x_traj, result.v_traj, result.p_traj
    ref_s_p = course_profile.s_p_fine
    
    # Compute stride for 1-second sampling
    dt = t_traj[1] - t_traj[0]
    step = int(max(1, round(1.0 / dt))) 

    records = []

    # 2. Lead-in: LEAD_IN_PAD_S seconds of stationary records at the course
    # start position, ending 1s before t=0 -- see LEAD_IN_PAD_S's own
    # docstring above.
    start_lat = float(np.interp(0.0, ref_s_p, course_profile.lat_fine))
    start_lon = float(np.interp(0.0, ref_s_p, course_profile.lon_fine))
    start_alt = float(np.interp(0.0, ref_s_p, course_profile.altitude))
    n_lead_in = int(round(LEAD_IN_PAD_S))
    for k in range(n_lead_in, 0, -1):
        msg = RecordMessage()
        msg.timestamp = start_time_ms - k * 1000
        msg.distance = 0.0
        msg.speed = 0.0
        msg.power = 0
        msg.position_lat = start_lat
        msg.position_long = start_lon
        msg.altitude = start_alt
        records.append(msg)

    # 3. Build RecordMessages for the timed trial itself
    for i in range(0, len(t_traj), step):
        msg = RecordMessage()

        # Timestamp and basic physics quantities
        msg.timestamp = start_time_ms + round(t_traj[i] * 1000)
        curr_x = float(x_traj[i])
        msg.distance = curr_x
        msg.speed = float(max(0.0, v_traj[i]))
        msg.power = int(round(max(0.0, p_traj[i])))

        # Assign interpolated degrees directly
        # The library handles the Semicircles conversion and validation internally
        msg.position_lat = float(np.interp(curr_x, ref_s_p, course_profile.lat_fine))
        msg.position_long = float(np.interp(curr_x, ref_s_p, course_profile.lon_fine))
        msg.altitude = float(np.interp(curr_x, ref_s_p, course_profile.altitude))

        records.append(msg)

    # 4. Trail-out: TRAIL_OUT_PAD_S seconds of extrapolated (not
    # stationary) records past the timed goal -- see TRAIL_OUT_PAD_S's
    # own docstring for the model and why it isn't the same treatment as
    # the lead-in above.
    prev_real, last_real = records[-2], records[-1]
    for k in range(1, int(round(TRAIL_OUT_PAD_S)) + 1):
        records.append(extrapolate_uniform_motion(prev_real, last_real, k))

    # 5. Bulk-add records
    builder.add_all(records)

    # 6. Session & Activity
    last_timestamp = start_time_ms + int(t_traj[-1] * 1000)
    
    session_msg = SessionMessage()
    session_msg.timestamp = last_timestamp
    session_msg.start_time = start_time_ms
    session_msg.total_elapsed_time = float(t_traj[-1])
    session_msg.total_timer_time = float(t_traj[-1])
    session_msg.total_distance = float(x_traj[-1])
    session_msg.sport = Sport.CYCLING
    session_msg.sub_sport = SubSport.VIRTUAL_ACTIVITY
    # Also assign start/end positions in degrees
    session_msg.start_position_lat = records[0].position_lat
    session_msg.start_position_long = records[0].position_long
    builder.add(session_msg)

    activity_msg = ActivityMessage()
    activity_msg.timestamp = last_timestamp
    activity_msg.num_sessions = 1
    activity_msg.total_timer_time = float(t_traj[-1])
    builder.add(activity_msg)

    # 7. Write file
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fit_file = builder.build()
    fit_file.to_file(file_path)

    # Optionally export CSV for debugging (uncomment to enable)
    # fit_file.to_csv(file_path.replace('.fit', '.csv'))

    log_subbanner(logger, f"SUCCESS: GPS SYNCED (DEGREE-MODE): {file_path}")

# --- Physics execution main loop ---
def physics_runner(p_powers, p_lengths, physics_params, base_dir, course, course_label, kernel, simulator_module, initial_w_prime):
    """
    Physics thread: loop over trials, run countdown and simulation, save FIT results.

    kernel: the resolved simulator's @njit physics kernel (core.simulators
        .SimulatorSpec.kernel). simulator_module: the actual module kernel
        is defined in (resolved via kernel.py_func.__module__), needed
        because the live-session real_time_sync_logic hookup below works by
        monkeypatching that module's own `sync_hook` global and calling
        `kernel.py_func` (the pre-JIT pure-Python function, which looks up
        globals dynamically) rather than the compiled dispatcher (which
        bakes globals in at compile time) -- see sync_hook's own docstring
        in core.simulators.sim_kiritsubo.
    initial_w_prime: rider's raw w_prime (from the strategy's own
        physiological settings dict, not physics_params.w_prime -- a
        future simulator's own params shape isn't guaranteed to carry a
        w_prime field just because its PhysiologicalSettings has one),
        used only to seed the HUD's W' balance display at the start of
        each trial. Every SIMULATOR_REGISTRY entry's own
        PhysiologicalSettings is guaranteed to have w_prime regardless
        (see core.schema.PhysiologicalSettingsBase).
    """
    trial_count = 0  # number of attempts in this session

    combine_dir = os.path.join(BASE_ACTIVITIES_DIR, "to_combine")
    if os.path.exists(combine_dir):
        # Delete all existing .fit files
        for f in glob.glob(os.path.join(combine_dir, "*.fit")):
            try:
                os.remove(f)
            except Exception as e:
                logger.error("Error cleaning %s: %s", f, e)
    else:
        os.makedirs(combine_dir)
    log_subbanner(logger, f"CLEANED: {combine_dir} for a new session")


    while shared_state.is_running:
        try:
            # 1. Reset shared state
            trial_count += 1
            with state_lock:
                shared_state.elapsed_time = 0.0
                shared_state.distance = 0.0
                shared_state.velocity = 0.0
                shared_state.actual_power = 0.0
                shared_state.w_prime_bal = initial_w_prime
                shared_state.reset_requested = False
                shared_state.start_requested = False
                shared_state.waiting_for_start = True
                shared_state.countdown_value = 0
                shared_state.finish_time_str = ""
                shared_state.ghost_x_traj = None
                shared_state.ghost_v_traj = None
                shared_state.ghost_t_traj = None

                current_if = getattr(real_time_sync_logic, "if_scale", 1.0)
                shared_state.target_p_list = (p_powers * current_if).tolist()
                shared_state.target_power = shared_state.target_p_list[0]

            if hasattr(real_time_sync_logic, "start_wall"):
                delattr(real_time_sync_logic, "start_wall")

            # 2. Wait at start line (Space key)
            log_subbanner(logger, "READY AT START LINE: Push [SPACE] to start countdown")
            while True:
                with state_lock:
                    if shared_state.reset_requested: raise SimulationResetException()
                    # Keep updating target power display while waiting
                    current_if = getattr(real_time_sync_logic, "if_scale", 1.0)
                    # Apply IF scale to first segment target power for display
                    if p_powers is not None and len(p_powers) > 0:
                        shared_state.target_power = p_powers[0] * current_if
                        shared_state.target_p_list = (p_powers * current_if).tolist()
                    if shared_state.start_requested: 
                        shared_state.waiting_for_start = False
                        break
                time.sleep(0.1)

            # Apply IF scale to strategy powers
            current_if = getattr(real_time_sync_logic, "if_scale", 1.0)
            adjusted_powers = p_powers * current_if
            with state_lock:
                shared_state.target_p_list = adjusted_powers.tolist()
            p_blocks = PowerBlocks(adjusted_powers, p_lengths)
            
            # Pre-compute ghost rider trajectory (ideal pace)
            logger.info("Calculating Ghost Rider with IF: %.2f...", current_if)
            # A. use_sync_hook=False below (the kernel's 5th positional arg)
            # is what makes this run at full speed instead of real-time --
            # with it False the kernel never reads simulator_module.sync_hook
            # at all, so patching it to a no-op here has no effect on this
            # particular call; kept only so nothing downstream sees a stale
            # hook reference while this block runs.
            orig_hook = simulator_module.sync_hook
            simulator_module.sync_hook = lambda p, t, v, x, W: 0
            ghost_res = kernel.py_func(
                0.0, p_blocks, physics_params, True, False, True
            )
            with state_lock:
                shared_state.ghost_x_traj = ghost_res.x_traj.copy()
                shared_state.ghost_v_traj = ghost_res.v_traj.copy()
                shared_state.ghost_t_traj = ghost_res.t_traj.copy()
                shared_state.ghost_curr_x = 0.0
                shared_state.ghost_curr_v = 0.0
            simulator_module.sync_hook = orig_hook
            logger.info("Ghost Calculation Done! Target Time: %.2fs", ghost_res.finish_time)

            # 3. 5-second countdown
            for i in range(5, 0, -1):
                with state_lock:
                    shared_state.countdown_value = i
                    if shared_state.reset_requested: raise SimulationResetException()
                logger.info("Countdown: %d", i)
                time.sleep(1.0)  # one-second intervals

            with state_lock:
                shared_state.countdown_value = 0 # Go!

            # 4. Start simulation
            log_subbanner(logger, "GO!")
            # Captured here, at the real start of the timed run (not after
            # it finishes) -- see save_to_fit's start_time docstring.
            trial_start_time = datetime.now(timezone.utc)
            result = kernel.py_func(
                0.0, p_blocks, physics_params, True, True, False
            )
            
            # 5. Post-finish: wait for reset
            log_subbanner(logger, f"Goal! Time: {result.finish_time:.2f}s")

            with state_lock:
                # Format finish time as MM:SS.ff
                m, s = divmod(result.finish_time, 60)
                shared_state.finish_time_str = f"{int(m):02d}:{s:05.2f}"

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            if not os.path.exists(BASE_ACTIVITIES_DIR):
                os.makedirs(BASE_ACTIVITIES_DIR)

            # 1. Define output filename
            file_base_name = f"{timestamp}_{course_label}_try{trial_count:02d}.fit"
            full_path = os.path.join(BASE_ACTIVITIES_DIR, file_base_name)
            # 2. Save simulation result as FIT
            logger.info("About to save...")
            save_to_fit(result, full_path, course, trial_start_time)
            logger.info("Saved.")
            # 3. Copy to the combiner staging directory
            combine_path = os.path.join(combine_dir, file_base_name)
            try:
                shutil.copy2(full_path, combine_path)
                logger.info("Copied for combine: %s", combine_path)
            except Exception as e:
                logger.error("Error copying to combine: %s", e)
            logger.info("Activity saved: %s", full_path)

            while not shared_state.reset_requested:
                if not shared_state.is_running: return
                time.sleep(0.1)
            raise SimulationResetException()

        except SimulationResetException:
            log_subbanner(logger, "Returning to Start Gate")
            continue

# --- Main entry point ---
def run_trainer():
    """Parse CLI arguments, load strategy JSON, initialise shared state, and start the GLUT main loop."""
    if len(sys.argv) < 5:
        print("Usage: eidos-trainer <StrategySetDir> <RunID> <Nseg> <Seed>")
        return

    # 1. Parse arguments
    # strategy_set_dir: e.g. "NisekoClassic2026" or "_20260301_170105"
    # ts: timestamp embedded in the strategy filename, e.g. "20260301_170105"
    strategy_set_dir, ts, n, s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    trial_id = f"N{n}_S{s}"

    # 2. Export directory (under exports/ hierarchy). Read the strategy JSON
    # from HERE, not from resources/strategies/ directly: Viewer only ever
    # offers "Launch Trainer" after listing an already-exported FIT file
    # for this trial (see eidos.apps.viewer.window.handle_launch_navigator),
    # so an export always exists by the time this runs -- and unlike the
    # exports/ copy, the original resources/strategies/ JSON is not durable
    # (Viewer's "Remove Design" permanently deletes it, independent of any
    # export). Exporter places this copy via shutil.copy2 under the exact
    # same filename (core.io_config.find_strategy_json_path's pattern), so
    # the filename below matches what's on disk in either location.
    base_dir = create_strategy_export_dir(strategy_set_dir, f"{ts}_{trial_id}")

    # Filename: strategy_{ts}_{trial_id}.json
    strategy_filename = f"strategy_{ts}_{trial_id}.json"
    strategy_path = os.path.join(base_dir, strategy_filename)

    log_subbanner(logger, "LOADING STRATEGY")
    logger.info("Directory: %s", strategy_set_dir)
    logger.info("File     : %s", strategy_filename)

    if not os.path.exists(strategy_path):
        logger.error("Strategy file not found: %s", strategy_path)
        return

    with open(strategy_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    data = unpack_input_data(raw_data)

    # Trainer's whole point is a "physics-identical" live replay of this
    # strategy -- it re-simulates with the simulator that actually produced
    # it (input.settings.engine.simulator, resolved below), but
    # git_state.commit_hash/is_dirty still matter: even the *same*
    # registered kernel's own code can have changed since this strategy
    # was generated. See core.git_info's module docstring.
    git_state = data.get('input', {}).get('git_state', {})
    repro_warning = check_reproducibility(git_state.get('commit_hash'), git_state.get('is_dirty'))
    if repro_warning:
        logger.warning(repro_warning)

    simulator_spec = resolve_simulator(data['input']['settings']['engine']['simulator'])
    kernel = simulator_spec.kernel
    # sync_hook's monkeypatch trick (see physics_runner's docstring) needs
    # the actual module kernel is defined in, not just the kernel itself.
    simulator_module = importlib.import_module(kernel.py_func.__module__)

    gpx_full_name = data['input']['settings']['run']['gpx_filename']
    course_label = os.path.splitext(gpx_full_name)[0]

    physical_s, physiological_s, run_s = data['input']['settings']['physical'], data['input']['settings']['physiological'], data['input']['settings']['run']
    cp_data = data['input']['data']['course_profile']
    physical_settings = simulator_spec.physical_param_model.model_construct(**physical_s)
    course = build_course_profile(cp_data)
    course = simulator_spec.recompute_course_physics(course, physical_settings)
    
    h_deg = np.array(cp_data['heading_deg_list'])
    lats, lons, alts = np.array(cp_data['latitude_list']), np.array(cp_data['longitude_list']), np.array(cp_data['altitude_list'])
    lat_mid = np.radians(np.mean(lats))
    m_per_lat, m_per_lon = (111132.92 - 559.82 * np.cos(2 * lat_mid)), (111412.84 * np.cos(lat_mid))
    course_x, course_y, course_z = (lons - lons[0]) * m_per_lon, alts, -(lats - lats[0]) * m_per_lat 

    rad = np.radians(h_deg)
    road_half_width = 2.5
    nx, nz = np.cos(rad) * road_half_width, np.sin(rad) * road_half_width
    left_v = np.column_stack([course_x - nx, course_y, course_z - nz]).astype(np.float32)
    right_v = np.column_stack([course_x + nx, course_y, course_z + nz]).astype(np.float32)
    road_vertices = np.empty((len(course_x) * 2, 3), dtype=np.float32)
    road_vertices[0::2, :], road_vertices[1::2, :] = left_v, right_v
    side_lines = (left_v, right_v)

    p_powers, p_lengths = np.array(data['output']['results']['strategy']['target_power_list']), np.array(data['output']['results']['strategy']['target_length_list'])
    boundaries_with_start = [0.0] + np.cumsum(p_lengths).tolist()

    # --- Initialize shared state ---
    with state_lock:
        shared_state.course_coords, shared_state.course_headings_deg = (course_x, course_y, course_z), h_deg
        shared_state.road_vertices, shared_state.side_lines = road_vertices, side_lines
        shared_state.distance_step, shared_state.segment_boundaries = course.distance_step, boundaries_with_start
        shared_state.target_p_list = p_powers.tolist()
        shared_state.actual_power = 0.0
        # .get(..., 0.0): a simulator with no wind model (e.g.
        # core.simulators.sim_stub) has neither field on its own
        # PhysicalSettings at all -- 0.0/0.0 is the same "no wind" state
        # as an explicit zero.
        shared_state.v_wind = physical_s.get('wind_speed', 0.0)
        shared_state.d_wind = math.radians(physical_s.get('wind_direction', 0.0))

    physics = simulator_spec.build_physics_params(
        physical_settings,
        simulator_spec.physiological_param_model.model_construct(**physiological_s),
        RunSettings(**run_s),
        course,
    )

    # --- Prepare communication and physics threads ---
    logger.info("Initializing ANT+ Power Receiver...")
    ant_recv = ANTPowerReceiver(shared_state, state_lock)
    shared_state.ant_recv = ant_recv  # store reference so it can be stopped externally
    ant_recv.start()

    logger.info("Starting Physics Engine Thread...")
    real_time_sync_logic.if_scale = 1.0
    simulator_module.sync_hook = real_time_sync_logic
    threading.Thread(
        target=physics_runner,
        args=(p_powers, p_lengths, physics, base_dir, course, course_label, kernel, simulator_module, physiological_s['w_prime']),
        daemon=True,
    ).start()

    # --- Pre-compute draw arrays for rendering performance ---
    cx, cy, cz = shared_state.course_coords
    # 1. Flatten center-line coordinates into a single float32 array
    shared_state.center_v = np.column_stack((cx, cy, cz)).astype(np.float32).flatten()
    # 2. Pre-extract road cross-lines at ~2 m intervals
    # With distance_step=1.0, each cross-line pair (left+right) occupies 6 elements
    # Reshape road_vertices (N, 3) so each row-pair is one cross-line set
    rv_reshaped = shared_state.road_vertices.reshape(-1, 2, 3) 
    # Slice [::2] over vertex pairs to achieve ~2 m spacing
    shared_state.rv_draw = rv_reshaped[::2].astype(np.float32).flatten()

    # --- Initialise graphics ---
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1280, 720)
    glutCreateWindow(window_title("Trainer").encode())
    view.init_view_context(shared_state, state_lock, real_time_sync_logic)
    init_graphics_settings()
    
    glEnable(GL_DEPTH_TEST)
    glutDisplayFunc(display)
    glutTimerFunc(0, timer, 0)
    
    glutKeyboardFunc(keyboard_keys)
    glutSpecialFunc(special_keys)

    logger.info("Ready. Start pedaling to move!")

    glutMainLoop()

def main() -> None:
    configure_logging()
    run_trainer()


if __name__ == "__main__":
    main()