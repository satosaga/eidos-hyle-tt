#####################
# view_opengl.py
#####################
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT import GLUT_STROKE_ROMAN

shared_state = None
state_lock = None
real_time_sync_logic = None
_course_cx = None
_course_bounds = None

_N_WIND_PARTICLES = 100
_WIND_BOX = np.array([25.0, 4.0, 25.0])  # half-extents (m) of the particle volume around the rider
_wind_particles = None  # (N, 3) offsets relative to the rider

def init_view_context(state, lock, sync_logic):
    """Bind module-level state, lock, and sync logic, and cache the course X/Z bounds."""
    global shared_state, state_lock, real_time_sync_logic
    global _course_cx, _course_bounds
    shared_state = state
    state_lock = lock
    real_time_sync_logic = sync_logic
    with lock:
        cx = state.course_coords[0]
        cz = state.course_coords[2]
        _course_cx = cx
        min_x, max_x = np.min(cx), np.max(cx)
        min_z, max_z = np.min(cz), np.max(cz)
        _course_bounds = (min_x, max_x, min_z, max_z)

def draw_minimap(cx, cz, curr_x, curr_z, ghost_x=None, ghost_z=None):
    """
    Render a minimap in the top-right corner of the screen.

    cx, cz      : full course coordinate arrays
    curr_x/z    : current rider position
    ghost_x/z   : ghost rider position (optional)
    """
    if cx is None or len(cx) == 0:
        return
    # 1. Save viewport and set minimap rectangle
    viewport = glGetIntegerv(GL_VIEWPORT)
    vw, vh = viewport[2], viewport[3]
    map_size = int(min(vw, vh) * 0.25)  # 25% of the shorter screen dimension
    margin = 20
    glViewport(vw - map_size - margin, vh - map_size - margin, map_size, map_size)
    # 2. Set 2D projection covering the full course extent
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    min_x, max_x, min_z, max_z = _course_bounds
    range_x = max_x - min_x
    range_z = max_z - min_z
    center_x = (max_x + min_x) / 2.0
    center_z = (max_z + min_z) / 2.0

    # Uniform square extent with 10% margin
    max_range = max(range_x, range_z) * 1.1
    half_side = max_range / 2.0
    gluOrtho2D(center_x - half_side, center_x + half_side,
               center_z + half_side, center_z - half_side)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    # 3. Disable depth test and lighting for 2D overlay
    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_POINT_BIT)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # 4. Semi-transparent background panel
    glColor4f(0.0, 0.0, 0.0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(center_x - half_side, center_z - half_side)
    glVertex2f(center_x + half_side, center_z - half_side)
    glVertex2f(center_x + half_side, center_z + half_side)
    glVertex2f(center_x - half_side, center_z + half_side)
    glEnd()
    # 5. Course path (light gray)
    glColor4f(0.6, 0.6, 0.6, 0.8)
    glLineWidth(1.5)
    step = max(1, len(cx)//2000)   # limit to ~2000 points
    glBegin(GL_LINE_STRIP)
    for i in range(0, len(cx), step):
        glVertex2f(cx[i], cz[i])
    glEnd()
    # 6. Ghost rider (red)
    if ghost_x is not None and ghost_z is not None:
        glPointSize(7.0)
        glColor3f(1.0, 0.2, 0.2)
        glBegin(GL_POINTS)
        glVertex2f(ghost_x, ghost_z)
        glEnd()
    # 7. Ego rider (blue)
    glPointSize(9.0)
    glColor3f(0.2, 0.5, 1.0)
    glBegin(GL_POINTS)
    glVertex2f(curr_x, curr_z)
    glEnd()
    # 8. Restore matrices and viewport
    glPopAttrib()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glViewport(0, 0, vw, vh)

def draw_wind_vectors_3d(v_wind, d_wind, v, x, y, z, heading_rad):
    """
    Render true and apparent wind as 3D cones above the rider's head, with
    polar reference rings at 5 m/s and 10 m/s. Each cone's apex sits at
    the rider's head (draw_bright_cone's base-then-180-degree-flip
    construction) with its base out at distance |vector| in the vector's
    own direction -- so, unlike draw_wind_particles' ambient drift
    vector (which points where the air actually moves), d_wind here is
    used un-negated on purpose: the cone visually reads as "wind arriving
    from that direction," not as a motion vector, so it does not need
    draw_wind_particles' FROM-to-TO negation.

    Apparent wind (orange) adds the rider's own ground speed to true wind
    (blue) -- the geometric mirror of draw_wind_particles' subtraction,
    for the same apex-at-rider convention.

    Args:
        v_wind: true wind speed (m/s).
        d_wind: true wind direction (rad, meteorological convention).
        v: rider's ground speed (m/s).
        x, y, z: rider world position.
        heading_rad: rider heading (rad).
    """
    VECTOR_SCALE = 0.25
    # Compute apparent and true wind vectors in world coordinates
    ux, uz = np.sin(heading_rad), -np.cos(heading_rad)
    tw_x = v_wind * np.sin(d_wind) * VECTOR_SCALE
    tw_z = -v_wind * np.cos(d_wind) * VECTOR_SCALE
    aw_x = tw_x + v * ux * VECTOR_SCALE
    aw_z = tw_z + v * uz * VECTOR_SCALE
    glPushMatrix()
    glTranslatef(x, y + 1.1, z)  # translate to head center
    # 1. Polar rings at 5 m/s and 10 m/s
    glDisable(GL_LIGHTING)
    glColor4f(0.2, 0.2, 0.2, 0.5)
    for m_per_sec in [5.0, 10.0]:
        radius = m_per_sec * VECTOR_SCALE
        glBegin(GL_LINE_LOOP)
        for i in range(36):
            theta = 2.0 * np.pi * i / 36.0
            glVertex3f(radius * np.cos(theta), 0.0, radius * np.sin(theta))
        glEnd()

    def draw_bright_cone(vx, vz, color):
        """Render a lit solid cone pointing in the direction (vx, vz) with the given color."""
        length = np.sqrt(vx**2 + vz**2)
        if length < 0.05: return
        glPushMatrix()
        angle = np.degrees(np.arctan2(vx, vz))
        glRotatef(angle, 0, 1, 0)
        # Shift so cone apex is at origin (glutSolidCone base is at origin)
        glTranslatef(0, 0, length)
        glRotatef(180, 0, 1, 0)
        # Enable emission so the cone is visible in dark environments
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT, GL_EMISSION, color)
        glColor4f(*color)
        glutSolidCone(0.03 * length, length, 12, 1)
        glMaterialfv(GL_FRONT, GL_EMISSION, (0, 0, 0, 1))
        glDisable(GL_LIGHTING)
        glPopMatrix()
    # 2. Apparent wind (orange)
    draw_bright_cone(aw_x, aw_z, (1.0, 0.4, 0.0, 1.0))
    # 3. True wind (blue)
    draw_bright_cone(tw_x, tw_z, (0.0, 0.5, 1.0, 1.0))
    glPopMatrix()

def draw_wind_particles(v_wind, d_wind, v, heading_rad, x, y, z, dt):
    """
    Render ambient wind as short streaks drifting through a box around the rider.

    Shows apparent (rider-relative) wind: true wind minus the rider's own
    velocity vector (Galilean subtraction), computed directly from
    v_wind/d_wind/v/heading rather than shared_state's per-step v_w_ax/v_w_cr,
    so it stays valid before the physics thread starts stepping (v=0 while
    waiting or counting down). While riding, v > 0 makes air stream past
    from ahead, as a headwind should.

    d_wind is the meteorological "coming from" bearing; particle velocity is
    the negated true-wind vector in that convention, minus the rider's own
    forward velocity.

    Particle offsets advect each frame and wrap through the opposite box
    face on exit, for continuous flow instead of pop-in respawns. Skipped
    entirely when the apparent wind is effectively calm.

    Args:
        v_wind: true wind speed (m/s).
        d_wind: true wind direction (rad, meteorological convention).
        v: rider's ground speed (m/s).
        heading_rad: rider heading (rad).
        x, y, z: rider world position.
        dt: seconds elapsed since the previous frame.
    """
    global _wind_particles
    ux, uz = np.sin(heading_rad), -np.cos(heading_rad)
    wx = -v_wind * np.sin(d_wind) - v * ux
    wz = v_wind * np.cos(d_wind) - v * uz
    v_w_ap = np.hypot(wx, wz)
    if v_w_ap < 0.3:
        return
    if _wind_particles is None:
        rng = np.random.default_rng()
        _wind_particles = (rng.random((_N_WIND_PARTICLES, 3)) * 2.0 - 1.0) * _WIND_BOX

    dt = min(dt, 0.1)  # guard against large jumps after a stall/resize
    _wind_particles[:, 0] += wx * dt
    _wind_particles[:, 2] += wz * dt
    _wind_particles[:, 0] = (_wind_particles[:, 0] + _WIND_BOX[0]) % (2 * _WIND_BOX[0]) - _WIND_BOX[0]
    _wind_particles[:, 2] = (_wind_particles[:, 2] + _WIND_BOX[2]) % (2 * _WIND_BOX[2]) - _WIND_BOX[2]

    streak_len = min(1.0, 0.1 * v_w_ap)
    dxs, dzs = -wx / v_w_ap * streak_len, -wz / v_w_ap * streak_len

    # Per-vertex alpha fades each streak from its head to fully transparent
    # at the tail -- a uniform-alpha segment reads as a rigid needle under
    # perspective foreshortening, especially streaks heading toward the camera.
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LINE_BIT)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glLineWidth(1.0)
    glPushMatrix()
    glTranslatef(x, y + 3.0, z)
    glBegin(GL_LINES)
    for px, py, pz in _wind_particles:
        glColor4f(0.85, 0.93, 1.0, 0.6)
        glVertex3f(px, py, pz)
        glColor4f(0.85, 0.93, 1.0, 0.0)
        glVertex3f(px + dxs, py, pz + dzs)
    glEnd()
    glPopMatrix()
    glPopAttrib()

def draw_gate(x, y, z, color, base_label, heading_deg, is_start_gate=False):
    """
    Render a start or finish gate frame with label and dynamic status text.

    The gate frame is a wire cube scaled to 6 m wide x 4 m tall. For the start
    gate, the first segment target power (W) is shown. Status text (READY /
    countdown / GO / finish time) is displayed when applicable.

    Args:
        x, y, z: gate world position.
        color: RGB tuple for the frame.
        base_label: 'START' or 'FINISH'.
        heading_deg: gate facing direction (degrees).
        is_start_gate: True for the start gate, False for the finish gate.
    """
    angle_for_opengl = -heading_deg
    glPushMatrix()
    glTranslatef(x, y + 2.0, z)
    glRotatef(angle_for_opengl, 0, 1, 0)
    # 1. Gate frame (height 4.0, width 6.0)
    glColor3f(*color)
    glPushMatrix()
    glScalef(6.0, 4.0, 0.5)
    glutWireCube(1.0)
    glPopMatrix()
    # 2. Label rendering
    m_scale_base = 0.0045
    glLineWidth(2.5)
    glColor3f(1.0, 1.0, 1.0)
    label_y_pos = 0.6
    t_width_base = sum(glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c)) for c in base_label) * m_scale_base
    x_off = -2.7 if base_label == "START" else (2.7 - t_width_base if base_label == "FINISH" else -(t_width_base / 2.0))
    # A: Main label
    glPushMatrix()
    glTranslatef(x_off, label_y_pos, -0.26)
    glScalef(m_scale_base, m_scale_base, m_scale_base)
    for c in base_label: glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
    glPopMatrix()
    # B: Start gate only — show first segment target power
    if is_start_gate:
        with state_lock:
            p_val = shared_state.target_p_list[0] if shared_state.target_p_list else 0
        p_txt = f"{p_val:.0f}W"
        t_width_p = sum(glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c)) for c in p_txt) * m_scale_base
        glPushMatrix()
        glTranslatef(-(t_width_p / 2.0), label_y_pos, -0.26)
        glScalef(m_scale_base, m_scale_base, m_scale_base)
        for c in p_txt: glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
        glPopMatrix()
    # 3. Dynamic status (READY / countdown / finish time)
    if is_start_gate or (base_label == "FINISH" and shared_state.finish_time_str):
        status_text = ""
        status_color = (1.0, 1.0, 1.0)
        with state_lock:
            if is_start_gate:
                if shared_state.waiting_for_start:
                    status_text, status_color = "READY", (1.0, 0.5, 0.0)
                elif shared_state.countdown_value > 0:
                    status_text, status_color = str(shared_state.countdown_value), (1.0, 0.0, 0.0)
                elif 0 < shared_state.elapsed_time < 2.0:
                    status_text, status_color = "GO", (0.4, 1.0, 0.2)
            elif base_label == "FINISH":
                status_text, status_color = shared_state.finish_time_str, (0.4, 1.0, 0.2)
        if status_text:
            m_scale_dyn = 0.0035
            t_width_dyn = sum(glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c)) for c in status_text) * m_scale_dyn
            glPushMatrix()
            glTranslatef(-t_width_dyn / 2.0, 1.5, -0.26)
            glScalef(m_scale_dyn, m_scale_dyn, m_scale_dyn)
            glLineWidth(3.0)
            glColor3f(*status_color)
            for c in status_text: glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
            glLineWidth(1.0)
            glPopMatrix()
    glPopMatrix()

def draw_stroke_centered(s, x, y, scale, color):
    """Render a GLUT stroke string centered horizontally at (x, y) in screen space."""
    viewport = glGetIntegerv(GL_VIEWPORT)
    avg_dim = (viewport[2] + viewport[3]) / 2.0
    w = sum(glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c)) for c in s) * scale
    glLineWidth(max(1.0, avg_dim / 600.0))  # no caller has ever asked for an explicit line width
    glColor4f(*color)
    glPushMatrix()
    glTranslatef(x - (w / 2), y, 0)
    glScalef(scale, scale, 1.0)
    for c in s: glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
    glPopMatrix()

def draw_stroke_left(s, x, y, scale, color):
    """Render a GLUT stroke string left-aligned at (x, y) in screen space."""
    glColor4f(*color)
    glPushMatrix()
    glTranslatef(x, y, 0)
    glScalef(scale, scale, 1.0)
    glLineWidth(1.0)
    for c in s: glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
    glLineWidth(1.0)
    glPopMatrix()   

def draw_hud_sensor_list(vw, vh, scale, spacing):
    """Render ANT+ connection status and device ID list in the top-left corner."""
    with state_lock:
        devices = list(shared_state.found_devices)
        target_id = shared_state.target_device_id
        ant_status = shared_state.ant_status
    x_pos = 30 * (vw / 1280.0)
    y_pos = vh - (50 * (vh / 720.0))
    # Connection status color
    if "Connected" in ant_status:
        status_color = (0.4, 1.0, 0.2, 0.9)
    elif any(s in ant_status for s in ["Scanning", "Initializing"]):
        status_color = (1.0, 0.9, 0.2, 0.9)
    else:
        status_color = (1.0, 0.2, 0.2, 0.9)
    # Row 1: Power meter status
    draw_stroke_left(f"Power Meter: {ant_status}", x_pos, y_pos, scale * 1.1, status_color)
    y_pos -= (spacing * 1.5)
    # Row 2+: Discovered device IDs
    for i, dev_id in enumerate(devices[:5]):
        is_selected = (dev_id == target_id)
        color = (0.8, 0.8, 0.8, 1.0) if is_selected else (0.8, 0.8, 0.8, 0.5)
        prefix = f"[{i+1}]"
        if dev_id == 0:
            draw_stroke_left(f"{prefix} W/S Virtual Power Meter", x_pos, y_pos, scale, color)
        else:
            draw_stroke_left(f"{prefix} ANT+ Power Meter (ID: {dev_id})", x_pos, y_pos, scale, color)
        y_pos -= spacing * 1.2

def draw_hud_controls(vw, vh, scale, spacing):
    """
    Render a stacked, low-key key-binding hint in the bottom-right corner.

    One key per line, sharing a single left edge (sized so the widest
    line's right edge touches the margin) -- this keeps the "=" separators
    aligned for easy scanning and avoids a wide single-row block that could
    overlap the 3D start-gate model near the start line. Keys with no
    current effect are omitted: W/S only changes power in virtual-meter
    mode (target_id==0; see eidos.lib.ant_receiver), since ANT+ broadcasts
    overwrite actual_power every frame otherwise.
    """
    with state_lock:
        waiting = shared_state.waiting_for_start
        target_id = shared_state.target_device_id
        n_devices = len(shared_state.found_devices)
    if waiting:
        lines = ["SPACE=Start", "ENTER=Reset", "UP/DN=Adjust IF", "ESC=Quit"]
        if n_devices > 1:
            lines.append("1-5=Meter")
    else:
        lines = ["ENTER=Reset", "ESC=Quit"]
        if target_id == 0:
            lines.insert(0, "W/S=Adjust Power")

    widths = [sum(glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c)) for c in line) * scale for line in lines]
    x_margin = 30 * (vw / 1280.0)
    x_pos = vw - x_margin - max(widths)
    y_pos = 30 * (vh / 720.0)
    color = (0.8, 0.8, 0.8, 0.5)
    for line in lines:
        draw_stroke_left(line, x_pos, y_pos, scale, color)
        y_pos += spacing

def draw_hud(t, d, v, W, p_target, dist_to_next):
    """
    Render the 2D heads-up display over the 3D scene.

    Switches to an orthographic 2D projection, then draws the sensor list
    (top-left), target and current power (center bottom), and secondary stats
    (elapsed time, distance, velocity, W' balance (anaerobic energy reserve, J))
    in the upper center area.

    Args:
        t: elapsed time (s).
        d: distance covered (m).
        v: current velocity (m/s).
        W: W' balance remaining (J).
        p_target: target power for the current segment (W).
        dist_to_next: distance to the next segment boundary (m).
    """
    viewport = glGetIntegerv(GL_VIEWPORT)
    vw, vh = viewport[2], viewport[3]
    # 2D projection
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, vw, 0, vh)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glDisable(GL_FOG)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Scaling
    avg_dim = (vw + vh) / 2.0
    text_scale = avg_dim / 3500.0
    label_scale = text_scale * 0.45
    sub_label_scale = label_scale * 0.9
    line_spacing = 60 * text_scale
    center_x = vw // 2
    base_y = int(vh * 0.20)
    main_gap = vw * 0.07
    sub_gap = vw * 0.10
    # 1. Sensor list (top-left)
    draw_hud_sensor_list(vw, vh, sub_label_scale, line_spacing)
    # 1b. Key-binding hint (bottom-right)
    draw_hud_controls(vw, vh, sub_label_scale, line_spacing)
    # 2. PRIMARY: POWER (center bottom)
    p_actual = shared_state.actual_power
    p_diff = p_actual - p_target
    label_offset = 85 * text_scale
    # Target power
    draw_stroke_centered(f"{round(p_target)}", center_x - main_gap, base_y, text_scale, (1, 1, 1, 0.9))
    draw_stroke_centered("Target Power [W]", center_x - main_gap, base_y - label_offset, label_scale, (0.7, 0.7, 0.7, 0.8))
    current_if = getattr(real_time_sync_logic, "if_scale", 1.0)
    if_label_y = base_y - (label_offset + 80 * text_scale)
    draw_stroke_centered(f"({current_if:.2f} X {round(p_target/current_if)} W)", center_x - main_gap, if_label_y, label_scale, (0.7, 0.7, 0.7, 0.8))
    # Current power
    abs_diff = abs(p_diff)
    act_col = (0.4, 1.0, 0.2, 0.9) if abs_diff <= 10 else (1.0, 0.9, 0.2, 0.9) if abs_diff <= 20 else (1.0, 0.2, 0.2, 0.9)
    draw_stroke_centered(f"{round(p_actual)}", center_x + main_gap, base_y, text_scale, act_col)
    draw_stroke_centered("Current Power [W]", center_x + main_gap, base_y - label_offset, label_scale, (0.7, 0.7, 0.7, 0.8))
    # 3. SECONDARY: STATS (center upper area)
    sub_y = base_y + int(350 * text_scale)
    sub_scale = text_scale * 0.65
    sub_label_offset = 80 * text_scale
    # Time
    draw_stroke_centered(f"{round(t)}", center_x - sub_gap * 1.5, sub_y, sub_scale, (0.9, 0.9, 0.9, 0.8))
    draw_stroke_centered("Time [s]", center_x - sub_gap * 1.5, sub_y - sub_label_offset, sub_label_scale, (0.6, 0.6, 0.6, 0.7))
    # Distance
    draw_stroke_centered(f"{d/1000:.2f}", center_x - sub_gap * 0.5, sub_y, sub_scale, (0.9, 0.9, 0.9, 0.8))
    draw_stroke_centered("Distance [km]", center_x - sub_gap * 0.5, sub_y - sub_label_offset, sub_label_scale, (0.6, 0.6, 0.6, 0.7))
    # Velocity
    draw_stroke_centered(f"{v*3.6:4.1f}", center_x + sub_gap * 0.5, sub_y, sub_scale, (0.9, 0.9, 0.9, 0.8))
    draw_stroke_centered("Velocity [km/h]", center_x + sub_gap * 0.5, sub_y - sub_label_offset, sub_label_scale, (0.6, 0.6, 0.6, 0.7))
    # W' Balance
    w_col = (0.9, 0.9, 0.9, 0.8) if W >= 0 else (1.0, 0.2, 0.2, 1.0)
    draw_stroke_centered(f"{round(W)}", center_x + sub_gap * 1.5, sub_y, sub_scale, w_col)
    draw_stroke_centered("W' Balance [J]", center_x + sub_gap * 1.5, sub_y - sub_label_offset, sub_label_scale, (0.6, 0.6, 0.6, 0.7))
    glPopAttrib()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_road():
    """
    Draw road geometry (side lines, cross lines, center line) from shared_state vertex arrays.
    """
    global _course_cx
    if shared_state is None or shared_state.side_lines is None:
        return
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)
    # Side lines and cross lines
    glColor3f(0.4, 0.4, 0.4)
    left_v, right_v = shared_state.side_lines
    # Left side line
    glVertexPointer(3, GL_FLOAT, 0, left_v)
    glDrawArrays(GL_LINE_STRIP, 0, len(left_v))
    # Right side line
    glVertexPointer(3, GL_FLOAT, 0, right_v)
    glDrawArrays(GL_LINE_STRIP, 0, len(right_v))
    # Cross lines
    if hasattr(shared_state, 'rv_draw') and shared_state.rv_draw is not None:
        glVertexPointer(3, GL_FLOAT, 0, shared_state.rv_draw)
        glDrawArrays(GL_LINES, 0, len(shared_state.rv_draw) // 3)
    # Center line
    if hasattr(shared_state, 'center_v') and shared_state.center_v is not None:
        glVertexPointer(3, GL_FLOAT, 0, shared_state.center_v)
        glDrawArrays(GL_LINE_STRIP, 0, len(_course_cx))
    glDisableClientState(GL_VERTEX_ARRAY)
    glPopAttrib()

def draw_environment_grid(curr_x, curr_z):
    """
    Draw a ground grid centered on the current rider position to convey speed sensation.
    Covers a 300 m x 300 m area at 10 m intervals.
    """
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT)
    glDisable(GL_LIGHTING)
    glBegin(GL_LINES)
    glColor3f(0.15, 0.15, 0.15)
    g_step = 10.0
    start_x = (int(curr_x / g_step) - 15) * g_step
    end_x   = (int(curr_x / g_step) + 15) * g_step
    start_z = (int(curr_z / g_step) - 15) * g_step
    end_z   = (int(curr_z / g_step) + 15) * g_step
    # Lines along X axis
    xi = start_x
    while xi <= end_x:
        glVertex3f(xi, 0.0, start_z)
        glVertex3f(xi, 0.0, end_z)
        xi += g_step
    # Lines along Z axis
    zi = start_z
    while zi <= end_z:
        glVertex3f(start_x, 0.0, zi)
        glVertex3f(end_x, 0.0, zi)
        zi += g_step
    glEnd()
    glPopAttrib()

def draw_ego_rider(x, y, z):
    """
    Draw the ego rider: a foot circle and a head sphere at the given position.
    y=0.01 for the foot circle avoids Z-fighting with the ground plane.
    """
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT)
    glDisable(GL_LIGHTING)
    glPushMatrix()
    glTranslatef(x, y, z)
    # Foot circle (dark blue)
    glColor4f(0.0, 0.0, 0.7, 1.0)
    glBegin(GL_LINE_LOOP)
    for j in range(32):
        th = 2.0 * np.pi * j / 32
        glVertex3f(0.25 * np.cos(th), 0.01, 0.25 * np.sin(th))
    glEnd()
    # Head sphere (bright blue)
    glColor3f(0.0, 0.0, 1.0)
    glTranslatef(0, 1.1, 0)
    glutWireSphere(0.25, 16, 16)
    glPopMatrix()
    glPopAttrib()

def draw_ghost_rider(gx, gy, gz):
    """
    Draw the ghost rider (reference target) as a red symbol at the given position.
    """
    if gx is None or gy is None or gz is None:
        return
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT)
    glDisable(GL_LIGHTING)
    glPushMatrix()
    glTranslatef(gx, gy, gz)
    # Foot circle (dark red)
    glColor4f(0.7, 0.0, 0.0, 1.0)
    glBegin(GL_LINE_LOOP)
    for j in range(32):
        th = 2.0 * np.pi * j / 32
        glVertex3f(0.25 * np.cos(th), 0.01, 0.25 * np.sin(th))
    glEnd()
    # Head sphere (bright red)
    glTranslatef(0, 1.1, 0)
    glColor3f(1.0, 0.0, 0.0)
    glutWireSphere(0.25, 16, 16)
    glPopMatrix()
    glPopAttrib()