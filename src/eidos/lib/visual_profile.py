########################
# visual_profile.py
########################
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# I. Constants
# --------------------------------------------------
PROFILE_KEYS = {
    'CDA_RATIOS': 'input.data.cda_ratios',
    'COURSE_DIST': 'input.data.course_profile.distance_p_m_list',
    'SLOPE': 'input.data.course_profile.slope_ratio_list',
    'ALTITUDE': 'input.data.course_profile.altitude_list',
    'LAT': 'input.data.course_profile.latitude_list',
    'LON': 'input.data.course_profile.longitude_list',
    'HEADING': 'input.data.course_profile.heading_deg_list',
    'DISTANCE': 'output.data.trace.distance_p_m_list',
    'TIME': 'output.data.trace.time_s_list',
    'POWER': 'output.data.trace.actual_p_w_list',
    'SPEED': 'output.data.trace.speed_mps_list',
    'WPRIME': 'output.data.trace.current_w_prime_j_list',
    'WIND_V_APP': 'output.data.trace.wind_v_apparent_mps_list',
    'WIND_YAW': 'output.data.trace.wind_yaw_deg_list',
}

# Visualization parameters
YLABEL_FONTSIZE = 14
TICK_FONTSIZE = 12

# Line width constants
LINE_WIDTH_PROF = 2.0
LINE_WIDTH_TARGET = 7.0
LINE_WIDTH_CP_WPRIME = 1.5

# --------------------------------------------------
# II. Helpers
# --------------------------------------------------
def get_record_style(record: dict, total_time_s: float, all_times: List[float], N: int) -> Tuple:
    """
    Compute display color (color_rgb) for a strategy record based on its
    finish time.

    Designer records receive a fixed green color. All other records are
    colored using the 'plasma' colormap, normalized between the fastest
    and slowest times in all_times.
    """
    if record.get("is_designer"):
        return (0.0, 1.0, 0.0)

    if N == 0:
        return (0.0, 0.0, 0.0)

    all_times_array = np.array([t for t in all_times if t is not None])

    time_min = np.nanmin(all_times_array) if len(all_times_array) > 0 else 0
    time_max = np.nanmax(all_times_array) if len(all_times_array) > 0 and np.nanmax(all_times_array) > time_min else time_min + 1

    CMAP = plt.colormaps['plasma']

    if time_max == time_min:
        norm_val = 1.0
    else:
        norm_val = (total_time_s - time_min) / (time_max - time_min)

    color_rgb = CMAP(0.9 * norm_val)[:3]

    return color_rgb