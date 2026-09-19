########################
# exporter.py
########################
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, List
from xml.dom import minidom
from xml.etree.ElementTree import Comment, Element, SubElement, tostring

import numpy as np
from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.course_message import CourseMessage
from fit_tool.profile.messages.course_point_message import CoursePointMessage
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.profile_type import (
    CoursePoint,
    Event,
    EventType,
    FileType,
    Manufacturer,
    Sport,
)
from scipy.interpolate import interp1d

from core.data_manager import (
    build_course_profile,
    extract_export_target,
    extract_gpx_base_name,
    unpack_input_data,
)
from core.git_info import check_reproducibility
from core.io_config import create_strategy_export_dir, find_strategy_json_path
from core.logging_setup import configure_logging
from core.schema import CourseProfile, ExportTarget, PowerBlocks, RunSettings
from core.simulators import resolve_simulator
from eidos.lib.pdf_exporter import build_strategy_data_from_export, export_strategy_pdf

logger = logging.getLogger(__name__)

# --------------------------------------------------
# I. Constants
# --------------------------------------------------
ZWO_AUTHOR = 'EIDOS^TT Exporter'
IF_LIST = [1.00, 0.95, 0.90, 0.70, 0.50]

# --------------------------------------------------
# II. File export (ZWO & FIT)
# --------------------------------------------------

def save_zwo(target: ExportTarget, durations: List[float], total_t: float,
             powers: np.ndarray, if_val: float, output_dir: str) -> str:
    """Build and save a Zwift-compatible ZWO workout file for the given IF."""
    c_name = extract_gpx_base_name(target.gpx_filename)
    if_code = f"{int(round(if_val * 100)):03d}"
    path = os.path.join(output_dir, f"{c_name}_CP{round(target.cp)}_IF{if_code}.zwo")

    root = Element('workout_file')
    SubElement(root, 'author').text = ZWO_AUTHOR
    SubElement(root, 'name').text = (
        f"{c_name} {target.run_set_id}_N{target.n_seg_used}_S{target.seed_used} "
        f"CP:{int(round(target.cp))}W IF:{if_val:.2f} Time:{int(round(total_t))}s"
    )

    work = SubElement(root, 'workout')

    # Lead-in (10 s at zero power)
    ss0 = SubElement(work, 'SteadyState')
    ss0.set('Duration', '10')
    ss0.set('Power', '0.0')
    ss0.set('RampType', 'Flat')

    for i, (p, dur) in enumerate(zip(powers, durations)):
        work.append(Comment(f" Seg {i+1}: {int(round(p))}W "))
        ss = SubElement(work, 'SteadyState')
        ss.set('Duration', str(int(round(dur))))
        ss.set('Power', str(p / target.cp))
        ss.set('RampType', 'Flat')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(minidom.parseString(tostring(root)).toprettyxml(indent="  "))
    return path

def save_strategy_fit(target: ExportTarget, course: CourseProfile,
                      simulation_result: Any, if_val: float, output_dir: str) -> str:
    """
    Build and save a FIT course file with power segment waypoints for the given IF.

    Uses enhanced_altitude and enhanced_speed for Garmin compatibility.
    CoursePoint labels show scaled/base power (e.g. '240W/300W') except at IF=1.00.
    seed_used is stored in serial_number for reproducibility.
    """
    builder = FitFileBuilder(auto_define=True, min_string_size=50)
    start_timestamp = round(datetime.now(timezone.utc).timestamp() * 1000)

    # 1. FileIdMessage
    msg = FileIdMessage()
    msg.type = FileType.COURSE
    msg.manufacturer = Manufacturer.DEVELOPMENT.value
    msg.product = 0
    msg.time_created = start_timestamp
    msg.serial_number = target.seed_used  # stored for experiment reproducibility
    builder.add(msg)

    # 2. CourseMessage
    # course_base_name (untruncated) is also reused below for the output
    # filename, so it must match save_zwo's c_name -- the FIT CourseMessage
    # field itself is truncated separately (course_name_field), since *that*
    # 31-char cap is a FIT course_name field-length constraint, not a
    # filename constraint. Truncating the shared variable used to also
    # shorten the .fit filename, silently diverging it from the matching
    # .zwo filename for the same IF (e.g. "..._Climb_El_CP250_IF100.fit" vs
    # "..._Climb_Elite_trim_CP250_IF100.zwo").
    course_base_name = extract_gpx_base_name(target.gpx_filename)
    c_msg = CourseMessage()
    c_msg.course_name = course_base_name[:31]
    c_msg.sport = Sport.CYCLING
    builder.add(c_msg)

    # 3. EventMessage (TIMER START)
    e_msg = EventMessage()
    e_msg.event = Event.TIMER
    e_msg.event_type = EventType.START
    e_msg.timestamp = start_timestamp
    builder.add(e_msg)

    # 4. RecordMessages (1-second interval)
    t_traj = simulation_result.t_traj
    x_traj = simulation_result.x_traj
    ref_s_p = course.s_p_fine

    dt = t_traj[1] - t_traj[0] if len(t_traj) > 1 else 1.0
    step = int(max(1, round(1.0 / dt)))
    last_record_timestamp = start_timestamp

    for i in range(0, len(t_traj), step):
        r_msg = RecordMessage()
        current_ts = start_timestamp + round(t_traj[i] * 1000)
        r_msg.timestamp = current_ts
        last_record_timestamp = current_ts

        r_msg.distance = float(x_traj[i])
        r_msg.position_lat = float(np.interp(x_traj[i], ref_s_p, course.lat_fine))
        r_msg.position_long = float(np.interp(x_traj[i], ref_s_p, course.lon_fine))
        r_msg.enhanced_altitude = float(np.interp(x_traj[i], ref_s_p, course.altitude))
        r_msg.enhanced_speed = float(simulation_result.v_traj[i])
        r_msg.power = int(round(simulation_result.p_traj[i]))
        builder.add(r_msg)

    # 5. CoursePointMessages (power segment waypoints)
    seg_start_distances = np.insert(np.cumsum(target.target_length_list)[:-1], 0, 0.0)
    p_scaled = target.target_power_list * if_val

    for dist, p_val in zip(seg_start_distances, p_scaled):
        cp_msg = CoursePointMessage()
        t_offset = np.interp(dist, x_traj, t_traj)
        cp_msg.timestamp = start_timestamp + round(t_offset * 1000)
        cp_msg.position_lat = float(np.interp(dist, ref_s_p, course.lat_fine))
        cp_msg.position_long = float(np.interp(dist, ref_s_p, course.lon_fine))
        cp_msg.distance = float(dist)

        # Label: "240W/300W" for sub-max IF, "300W" for IF=1.00
        if abs(if_val - 1.0) > 0.001:
            label = f"{int(round(p_val))}W/{int(round(p_val / if_val))}W"
        else:
            label = f"{int(round(p_val))}W"

        cp_msg.course_point_name = label
        cp_msg.type = CoursePoint.GENERIC
        builder.add(cp_msg)

    # 6. EventMessage (TIMER STOP)
    stop_msg = EventMessage()
    stop_msg.event = Event.TIMER
    stop_msg.event_type = EventType.STOP_ALL
    stop_msg.timestamp = last_record_timestamp
    builder.add(stop_msg)

    # 7. LapMessage
    elapsed_time_ms = last_record_timestamp - start_timestamp
    l_msg = LapMessage()
    l_msg.timestamp = last_record_timestamp
    l_msg.start_time = start_timestamp
    l_msg.total_elapsed_time = elapsed_time_ms / 1000.0
    l_msg.total_timer_time = elapsed_time_ms / 1000.0
    l_msg.total_distance = float(x_traj[-1])
    l_msg.start_position_lat = float(course.lat_fine[0])
    l_msg.start_position_long = float(course.lon_fine[0])
    l_msg.end_position_lat = float(course.lat_fine[-1])
    l_msg.end_position_long = float(course.lon_fine[-1])
    builder.add(l_msg)

    # 8. File naming: e.g. Fukushima2026_ITT_CP250_IF050.fit
    if_int = int(round(if_val * 100))
    file_name = f"{course_base_name}_CP{round(target.cp)}_IF{if_int:03d}.fit"
    file_path = os.path.join(output_dir, file_name)
    builder.build().to_file(file_path)

    return file_path

# --------------------------------------------------
# III. Main execution logic
# --------------------------------------------------

def execute_strategy_export():
    """Entry point: export FIT/ZWO/PDF files for a single strategy JSON."""
    if len(sys.argv) < 5:
        logger.error("Usage: eidos-exporter <StrategySetDir> <RunID> <Nseg> <Seed>")
        sys.exit(1)

    strategy_set_dir = sys.argv[1]
    run_id = sys.argv[2]
    n_seg = int(sys.argv[3])
    seed = int(sys.argv[4])
    trial_id = f"N{n_seg}_S{seed}"

    try:
        # 1. Locate strategy JSON and load it -- done before any of the
        # "Export started"/"Base directory" INFO lines below (and before
        # creating the export dir at all) purely so a repro_warning, if
        # there is one, is the first thing printed. It's the one message in
        # this whole log worth reading before anything else, so it belongs
        # above the start-of-run banner lines, not sandwiched after them.
        json_path = find_strategy_json_path(strategy_set_dir, run_id, n_seg, seed)
        target = extract_export_target(json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        data = unpack_input_data(raw_data)

        # FIT/ZWO export re-simulates the stored strategy with the simulator
        # that actually produced it (input.settings.engine.simulator,
        # resolved below) -- git_state.commit_hash/is_dirty still matter,
        # though: even the *same* registered kernel's own code can have
        # changed since this strategy was generated. See core.git_info's
        # module docstring.
        git_state = data.get('input', {}).get('git_state', {})
        repro_warning = check_reproducibility(git_state.get('commit_hash'), git_state.get('is_dirty'))
        if repro_warning:
            logger.warning(repro_warning)

        # 2. Create export directory structure
        flat_trial_id = f"{run_id}_{trial_id}"
        base_dir = create_strategy_export_dir(strategy_set_dir, flat_trial_id)

        logger.info("Export started for %s", flat_trial_id)
        logger.info("Export directory: %s", base_dir)

        fit_output_dir = os.path.join(base_dir, "fit")
        zwo_output_dir = os.path.join(base_dir, "zwo")

        # Copy JSON as archive
        shutil.copy2(json_path, os.path.join(base_dir, os.path.basename(json_path)))

        # 3. Build course profile
        simulator_spec = resolve_simulator(data['input']['settings']['engine']['simulator'])

        # model_construct(), not model_validate() -- see
        # core.physics_overrides' module docstring for why an
        # already-validated strategy JSON's own settings are reconstructed
        # without re-running PhysicalSettings.cda_yaw_table_filename's CSV
        # file-I/O validator.
        physical_settings = simulator_spec.physical_param_model.model_construct(**data['input']['settings']['physical'])
        physiological_settings = simulator_spec.physiological_param_model.model_construct(**data['input']['settings']['physiological'])
        run_settings = RunSettings(**data['input']['settings']['run'])

        # build_course_profile reuses this strategy's own stored v_limit
        # and neutral cos_phi/sin_phi placeholders -- see that function's
        # own docstring; recompute_course_physics fills in this
        # simulator's own correct v_limit/cos_phi/sin_phi (a no-op for a
        # simulator with no braking-limit or wind model, e.g.
        # core.simulators.sim_stub).
        course = build_course_profile(data['input']['data']['course_profile'])
        course = simulator_spec.recompute_course_physics(course, physical_settings)

        # 4. Simulate and export for each IF value
        sim_results = {}

        physics = simulator_spec.build_physics_params(
            physical_settings, physiological_settings, run_settings, course
        )

        for if_val in IF_LIST:
            p_scaled = target.target_power_list * if_val
            power_blocks = PowerBlocks(power=p_scaled, length=target.target_length_list)

            sim_res = simulator_spec.kernel(0.0, power_blocks, physics, True, False, True)
            sim_results[if_val] = sim_res

            # Compute segment durations for ZWO (interpolated from trajectory)
            time_at_dist = interp1d(sim_res.x_traj, sim_res.t_traj, kind='linear', fill_value="extrapolate")
            seg_ends = np.cumsum(target.target_length_list)
            durs = []
            t_curr = 0.0
            for i, dist in enumerate(seg_ends):
                t_end = sim_res.finish_time if i == len(seg_ends)-1 else float(time_at_dist(dist))
                durs.append(max(1.0, t_end - t_curr))
                t_curr = t_end

            zwo_path = save_zwo(target, durs, sim_res.finish_time, p_scaled, if_val, zwo_output_dir)
            fit_path = save_strategy_fit(target, course, sim_res, if_val, fit_output_dir)

            # Every artifact-producing step below logs as
            # "Creating <category> (<detail>) -> <relpath>", one line per
            # file, so fit/, pdf/, and zwo/ (base_dir's parallel sibling
            # subfolders) all show up the same way -- one row per file
            # written, not one row per file *type* bundled together.
            duration_s = int(round(sim_res.finish_time))
            logger.info("Creating fit (IF %.2f, %ds) -> fit/%s", if_val, duration_s, os.path.basename(fit_path))
            logger.info("Creating zwo (IF %.2f, %ds) -> zwo/%s", if_val, duration_s, os.path.basename(zwo_path))

        # 5. Generate PDF
        pdf_dir = os.path.join(base_dir, "pdf")
        input_data = data.get('input', {})
        settings = {}
        if 'versions' in input_data:
            settings['versions'] = input_data['versions']
        if 'git_state' in input_data:
            settings['git_state'] = input_data['git_state']
        settings.update(input_data.get('settings', {}))
        strategy_data = build_strategy_data_from_export(target, course, sim_results, IF_LIST, settings=settings)
        summary_pdf, tanzaku_pdf = export_strategy_pdf(strategy_data, pdf_dir)
        logger.info("Creating pdf (strategy) -> pdf/%s", os.path.basename(summary_pdf))
        logger.info("Creating pdf (stemcard) -> pdf/%s", os.path.basename(tanzaku_pdf))
        logger.info("Export complete")

    except Exception as e:
        logger.error("Export failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main() -> None:
    configure_logging()
    execute_strategy_export()


if __name__ == '__main__':
    main()