######################
# data_manager.py
######################
import base64
import json
import logging
import os
import pickle
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from core.io_config import BASE_CDA_YAW_TABLES_DIR, BASE_GPX_DIR
from core.schema import CoursePoints, CourseProfile, ExportTarget, RunSettings

logger = logging.getLogger(__name__)

# --------------------------------------------------
# I. Version
# --------------------------------------------------
# Bump policy: bump for a change you'd want to be able to look back and
# identify later -- not for pure refactors/renames (e.g. a file move).
# Enforced by the pre-commit framework (scripts/check_code_version_bump.sh,
# wired in .pre-commit-config.yaml), which blocks a commit touching this
# file unless this line is part of the same commit -- use
# `git commit --no-verify` for a deliberate no-bump change. See
# core.git_info's module docstring for the separate, automatic
# git_commit/git_dirty reproducibility tracking this hand-maintained
# string does *not* need to (and isn't meant to) replace.
DATA_MANAGER_VERSION = "v1.3.2"

# --------------------------------------------------
# II. Internal helpers
# --------------------------------------------------
def format_time_mmss(seconds: float) -> str:
    """Format a duration in seconds as zero-padded 'MM:SS' (e.g. 65 -> '01:05').

    Shared by eidos.lib.pdf_exporter and eidos.apps.analyzer so the two
    can't silently diverge on padding.
    """
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _parse_gpx_to_ndarray(xml_text: str) -> np.ndarray:
    """Parse a GPX XML string and return a (N, 3) array of [lat, lon, elevation]."""
    root = ET.fromstring(xml_text)
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    pts = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat_str, lon_str = trkpt.get('lat'), trkpt.get('lon')
        if lat_str is None or lon_str is None:
            raise ValueError("GPX <trkpt> is missing a required lat/lon attribute")
        lat, lon = float(lat_str), float(lon_str)
        ele_elem = trkpt.find('gpx:ele', ns)
        ele = float(ele_elem.text) if (ele_elem is not None and ele_elem.text is not None) else 0.0
        pts.append([lat, lon, ele])
    return np.array(pts)

def _clean_and_project(points: np.ndarray, d_epsilon: float):
    """
    Project geographic coordinates to a local flat-earth Cartesian frame and
    remove duplicate points closer than d_epsilon [m] by iterative averaging.

    Returns (x, y, z, s_h, s_p) arrays in meters.
    """
    # Drop non-finite rows (e.g. a GPS dropout sample with NaN lat/lon in a
    # real device recording -- observed on real activity FIT files, not
    # GPX course definitions) before projecting. Without this, a single
    # NaN row poisons every downstream cumsum entry from that point on
    # (s_h/s_p go NaN for the entire remainder of the track, not just
    # that one point), which then blows up fit_uniform_b_spline's domain
    # computation (ceil(NaN) -> ValueError).
    finite_mask = np.all(np.isfinite(points), axis=1)
    if not finite_mask.all():
        points = points[finite_mask]

    # lat_mid anchors the projection's scale factors at the SAME point
    # (points[0]) the x/y offsets are already anchored at -- not the
    # track's mean latitude -- so this forward projection is the exact
    # inverse of core.course_geometry.project_meters_to_latlon, which
    # recomputes m_per_lat/m_per_lon from CoursePoints.origin_lat (that
    # same first point, see load_course below). Using the mean instead
    # would leave the two using slightly different scale factors,
    # breaking the round trip by an amount that grows with how far the
    # track's mean latitude drifts from its start point.
    lat_mid = np.radians(points[0, 0])
    m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
    m_per_lon = 111412.84 * np.cos(lat_mid)
    x = (points[:, 1] - points[0, 1]) * m_per_lon
    y = (points[:, 0] - points[0, 0]) * m_per_lat
    z = points[:, 2]

    x_l, y_l, z_l = x.tolist(), y.tolist(), z.tolist()

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(x_l) - 1:
            dist_h = np.sqrt((x_l[i+1] - x_l[i])**2 + (y_l[i+1] - y_l[i])**2)
            if dist_h < d_epsilon:
                x_l[i] = (x_l[i] + x_l[i+1]) / 2.0
                y_l[i] = (y_l[i] + y_l[i+1]) / 2.0
                z_l[i] = (z_l[i] + z_l[i+1]) / 2.0
                x_l.pop(i+1); y_l.pop(i+1); z_l.pop(i+1)
                changed = True
                continue
            i += 1

    x_clean, y_clean, z_clean = np.array(x_l), np.array(y_l), np.array(z_l)
    dx, dy, dz = np.diff(x_clean), np.diff(y_clean), np.diff(z_clean)
    dist_h = np.sqrt(dx**2 + dy**2)
    dist_p = np.sqrt(dx**2 + dy**2 + dz**2)
    s_h = np.insert(np.cumsum(dist_h), 0, 0.0)
    s_p = np.insert(np.cumsum(dist_p), 0, 0.0)

    return x_clean, y_clean, z_clean, s_h, s_p

def unpack_input_data(data_dict: dict) -> dict:
    """
    Decompress a compressed_packet and restore array data into course_profile.

    The packet is removed after decompression and is_index_only is set to False.
    If no compressed_packet is present, the dict is returned unchanged.
    """
    input_part = data_dict["input"]
    data_part = input_part["data"]

    if "compressed_packet" in data_part:
        # 1. Decompress
        compressed_data = base64.b64decode(data_part["compressed_packet"])
        unpacked_arrays = pickle.loads(zlib.decompress(compressed_data))

        # 2. Merge arrays into course_profile (distance_step already present)
        cp = data_part["course_profile"]
        cp.update(unpacked_arrays)

        # 3. Restore cda_ratios to data_part level (not inside course_profile)
        if "cda_ratios" in unpacked_arrays:
            data_part["cda_ratios"] = np.array(unpacked_arrays["cda_ratios"], dtype=np.float64)
            if "cda_ratios" in cp:
                del cp["cda_ratios"]

        # 4. Remove packet and mark as fully loaded
        del data_part["compressed_packet"]
        data_dict['is_index_only'] = False

    return data_dict

# --------------------------------------------------
# III. Public interface
# --------------------------------------------------

def load_cda_yaw_table(table_name: str) -> np.ndarray:
    """
    Load a CdA yaw multiplier CSV from the cda_yaw_tables directory and return
    a 181-element array of multipliers for yaw angles 0-180 degrees.
    """
    table_path = os.path.join(BASE_CDA_YAW_TABLES_DIR, table_name)

    df = pd.read_csv(table_path, header=None).sort_values(by=0)
    interp_func = interp1d(df[0], df[1], kind='linear', bounds_error=False,
                           fill_value=(df[1].values[0], df[1].values[-1]))

    return interp_func(np.arange(0, 181, 1)).astype(np.float64)

def extract_gpx_base_name(gpx_filename: str) -> str:
    """Return the stem (filename without extension) of a GPX file path."""
    base_name_with_ext = os.path.basename(gpx_filename)
    return os.path.splitext(base_name_with_ext)[0]

def build_course_profile(course_data: dict) -> CourseProfile:
    """
    Reconstruct a CourseProfile from a deserialized course_data dictionary
    (typically extracted from a strategy JSON after decompression).

    v_limit is reused as stored (this strategy's own original speed-limit
    result); cos_phi/sin_phi are set to neutral "no wind" placeholders
    (1.0/0.0) -- both are provisional. No wind_direction argument here --
    not every simulator requires one. Callers that need this simulator's
    own correct v_limit/cos_phi/sin_phi should follow this with
    simulator_spec.recompute_course_physics(course, physical) -- see
    e.g. eidos.apps.exporter, which does exactly that.
    """
    s_p_fine = np.array(course_data['distance_p_m_list'])
    s_h_fine = np.array(course_data['distance_h_m_list'])
    slope_ratio = np.array(course_data['slope_ratio_list'])
    heading = np.radians(np.array(course_data['heading_deg_list']))
    n_fine = len(s_p_fine)

    return CourseProfile(
        distance=s_p_fine[-1],
        distance_step=course_data['distance_step'],
        s_p_fine=s_p_fine,
        s_h_fine=s_h_fine,
        lat_fine=np.array(course_data['latitude_list']),
        lon_fine=np.array(course_data['longitude_list']),
        slope=np.arctan(slope_ratio),
        kappa=np.array(course_data['kappa_list']),
        v_limit=np.array(course_data['v_limit_list']),
        altitude=np.array(course_data['altitude_list']),
        heading=heading,
        cos_phi=np.ones(n_fine),
        sin_phi=np.zeros(n_fine),
    )

def load_course(gpx_file: str, run_settings: RunSettings) -> CoursePoints:
    """
    Load a GPX file, clean duplicate points, project to Cartesian coordinates,
    validate against RunSettings, and return a CoursePoints instance.
    """
    try:
        gpx_full_path = os.path.join(BASE_GPX_DIR, gpx_file)
        with open(gpx_full_path, 'r', encoding='utf-8') as f:
            xml_text = f.read()
    except Exception as e:
        logger.error("Error loading GPX: %s", e)
        raise

    raw_pts = _parse_gpx_to_ndarray(xml_text)
    x_c, y_c, z_c, s_h_c, s_p_c = _clean_and_project(raw_pts, 1.0)

    l_total_m = s_p_c[-1]
    run_settings.validate_with_course(l_total_m)

    return CoursePoints(
        x=x_c, y=y_c, z=z_c, s_h=s_h_c, s_p=s_p_c,
        distance=s_p_c[-1], origin_lat=raw_pts[0, 0], origin_lon=raw_pts[0, 1]
    )

def extract_export_target(strategy_json_path: str) -> ExportTarget:
    """
    Load a strategy JSON, decompress its arrays, and return an ExportTarget instance
    ready for FIT/ZWO export.

    Raises KeyError if required course arrays are missing after decompression.
    """
    with open(strategy_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = unpack_input_data(data)

    # Verify required arrays are present after decompression
    course_p_dict = data.get('input', {}).get('data', {}).get('course_profile', {})
    required_keys = ['v_limit_list', 'slope_ratio_list', 'heading_deg_list']
    missing = [k for k in required_keys if k not in course_p_dict]
    if missing:
        raise KeyError(f"Missing required arrays after unpack: {missing}")

    run_set_id = data['run_set_id']
    out_meta = data['output']['metadata']
    out_res = data['output']['results']
    physiological_s = data['input']['settings']['physiological']
    run_s = data['input']['settings']['run']

    # cp/w_prime read unconditionally: every SIMULATOR_REGISTRY entry's
    # own PhysiologicalSettings is required to define cp/w_prime (see
    # core.schema.PhysiologicalSettingsBase), regardless of whether that
    # simulator's own kernel physics reads either -- no per-simulator
    # fallback or precondition check needed here.
    return ExportTarget(
        run_set_id=run_set_id,
        n_seg_used=out_meta['n_seg'],
        seed_used=out_meta['seed'],
        target_power_list=np.array(out_res['strategy']['target_power_list']),
        target_length_list=np.array(out_res['strategy']['target_length_list']),
        gpx_filename=run_s['gpx_filename'],
        cp=physiological_s['cp'],
        w_prime=physiological_s['w_prime'],
    )


def save_strategy_to_json(data_dict: dict, file_path: str):
    """
    Compress and atomically write a strategy dictionary to a JSON file.

    Internally copies the dict, packs arrays into a compressed_packet,
    then writes to a temp file before atomically replacing the target path.
    This prevents data loss from partial writes.
    """
    import copy
    import tempfile

    packed_data = _pack_internal(copy.deepcopy(data_dict))

    dir_name = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
        json.dump(packed_data, tf, indent=4, ensure_ascii=False)
        temp_name = tf.name

    os.replace(temp_name, file_path)

def _pack_internal(data_dict: dict) -> dict:
    """
    Compress all array data in course_profile and cda_ratios into a single
    zlib+pickle+base64 compressed_packet for compact JSON storage.

    distance_step is excluded from compression and remains as a plain scalar.
    NumPy scalars are also safely converted to lists via np.atleast_1d.
    """
    data_part = data_dict["input"]["data"]
    cp = data_part["course_profile"]

    arrays_to_compress = {}

    # Extract all arrays from course_profile except distance_step
    for k in list(cp.keys()):
        v = cp[k]
        if k != 'distance_step' and isinstance(v, (list, np.ndarray, np.generic)):
            arrays_to_compress[k] = np.atleast_1d(v).tolist()
            del cp[k]

    # Include cda_ratios in the packet
    if "cda_ratios" in data_part:
        arrays_to_compress["cda_ratios"] = np.atleast_1d(data_part["cda_ratios"]).tolist()
        del data_part["cda_ratios"]

    data_part["compressed_packet"] = base64.b64encode(
        zlib.compress(pickle.dumps(arrays_to_compress))
    ).decode('utf-8')

    return data_dict