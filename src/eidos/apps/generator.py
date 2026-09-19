#########################
# generator.py
#########################
import datetime
import json
import logging
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.data_manager import (
    DATA_MANAGER_VERSION,
    load_cda_yaw_table,
    load_course,
    save_strategy_to_json,
)
from core.git_info import get_git_commit_hash, is_relevant_code_dirty
from core.io_config import (
    BASE_CONFIGS_DIR,
    BASE_STRATEGIES_DIR,
    list_json_files_by_creation_time,
)
from core.logging_setup import configure_logging, log_banner
from core.pydantic_mapper import ExperimentIndexModel, extract_flat_data
from core.schema import (
    CoursePoints,
    CourseProfile,
    OptimizationStrategy,
    RunSettings,
)
from core.simulators import (
    SimulatorSpec,
    resolve_physical_params,
    resolve_physiological_params,
    resolve_simulator,
)
from eidos.lib.optimizer import EngineSettings, OptimizerSpec, resolve_optimizer

logger = logging.getLogger(__name__)

# --------------------------------------------------
# I. Execution logic
# --------------------------------------------------

def run_full_optimization_pipeline(
    physical_settings: BaseModel,
    physiological_settings: BaseModel,
    run_settings: RunSettings,
    engine_settings: EngineSettings,
    gpx_filename: str):
    """
    Run the full optimization pipeline for a single configuration set.

    1. Load and compute course physics.
    2. Assemble PhysicsParams and OptimizationStrategy.
    3. Run sequential multi-start DE optimization across all n_seg values.
    4. Save all results and Parquet index.
    """
    simulator_spec = resolve_simulator(engine_settings.simulator)
    optimizer_spec = resolve_optimizer(engine_settings.optimizer)
    # engine_settings.optimizer_params is already validated (see
    # eidos.lib.optimizer.EngineValidationModel.validate_and_default_optimizer_params)
    # -- re-validating here is cheap (no file I/O, unlike physical_settings)
    # and keeps this the one place that turns the raw dict into the
    # optimizer's own param_model instance every downstream call needs.
    optimizer_params = optimizer_spec.param_model.model_validate(engine_settings.optimizer_params)
    log_banner(logger, "Time Trial Strategy Planner: Start Execution")

    # 1. Load course data and compute physics profile
    try:
        course_pts: CoursePoints = load_course(gpx_filename, run_settings)
        course: CourseProfile = simulator_spec.compute_course_physics(
            course_pts,
            physical_settings,
            run_settings
        )
        logger.info("Course physics calculated. v_limit check passed.")

    except ValueError as ve:
        logger.error("Critical Physics Failure: %s", ve)
        return
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return

    # 2. Assemble physics and strategy parameters (physics / tactics separation)
    physics = simulator_spec.build_physics_params(physical_settings, physiological_settings, run_settings, course)

    strategy_base = OptimizationStrategy(
        n_seg=run_settings.n_seg_min,  # placeholder: OptimizationStrategy requires a value, but neither
                                        # opt_tenchi's nor opt_stub's sequential_optimize_Nseg reads this
                                        # field back off strategy_base -- each builds its own per-n_seg
                                        # OptimizationStrategy internally instead
        seg_power_min=run_settings.seg_power_min,
        seg_power_max=run_settings.seg_power_max,
        seg_length_min=run_settings.seg_length_min
    )

    # 3. Run sequential optimization
    logger.info("Starting sequential optimization")
    optimization_start = time.perf_counter()
    all_seed_results_by_Nseg = optimizer_spec.run(
        physics=physics,
        course_distance=course.distance,
        N_seg_min=run_settings.n_seg_min,
        N_seg_max=run_settings.n_seg_max,
        strategy_base=strategy_base,
        initial_base_seed=run_settings.initial_base_seed,
        seed_factor=run_settings.seed_factor,
        simulator_key=engine_settings.simulator,
        params=optimizer_params,
    )
    optimization_wall_s = time.perf_counter() - optimization_start
    # Wall-clock only, not persisted to the saved strategy JSONs -- this
    # machine's own core count/load shifts it run to run, so it belongs in
    # the Execution Log as an observed fact about this run, not in a
    # reproducibility-tracked output file (see core.git_info's module
    # docstring on what that file's own git_commit/git_dirty fields are
    # and are not meant to capture). Logged at optimizer_spec.run's call
    # site (not inside any one OPTIMIZER_REGISTRY entry) so it applies
    # uniformly to every optimizer without touching opt_tenchi.py/
    # opt_stub.py themselves.
    logger.info("Optimization completed (wall time: %.1fs)", optimization_wall_s)

    # 4. Save results
    logger.info("Saving results")
    save_experiment_results(
        all_seed_res=all_seed_results_by_Nseg,
        course=course,
        physical_settings=physical_settings,
        physiological_settings=physiological_settings,
        run_settings=run_settings,
        engine_settings=engine_settings,
        optimizer_params=optimizer_params,
        simulator_spec=simulator_spec,
        optimizer_spec=optimizer_spec,
        data_manager_version=DATA_MANAGER_VERSION,
    )
    logger.info("All results saved successfully")

# --------------------------------------------------
# II. Configuration loading
# --------------------------------------------------

def load_config_jsons() -> list[tuple[BaseModel, BaseModel, RunSettings, EngineSettings, str, str]]:
    """
    Load and validate all JSON configuration files from BASE_CONFIGS_DIR,
    oldest Added/Duplicated first (see core.io_config.
    list_json_files_by_creation_time) -- the same order
    eidos.apps.manager's "Files in use" list shows for configs/, so a
    generation run processes them in the order a user would read down
    that list.

    Returns a list of (physical_settings, physiological_settings, RunSettings,
    EngineSettings, gpx_filename, source_filename) tuples -- source_filename
    (the config's own filename, e.g. "sample_config_01.json") is carried
    along purely so execute_strategy_generation can label its per-config-set
    log banner; it plays no role in the optimization itself. Files that fail
    Pydantic validation are skipped with a warning -- including a missing/
    invalid Engine section, which has no default and must name a registered
    simulator/optimizer (see eidos.lib.optimizer.EngineValidationModel).

    Engine is resolved before PhysicalSettings/PhysiologicalSettings
    because which Pydantic model validates those two sections depends on
    Engine.simulator -- see core.simulators.resolve_physical_params.
    """
    config_settings_list = []
    configs_dir = BASE_CONFIGS_DIR
    for filename in list_json_files_by_creation_time(configs_dir):
        file_path = os.path.join(configs_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            engine_s = EngineSettings.from_dict(data.get("Engine") or {})
            physical_s = resolve_physical_params(engine_s.simulator, data.get("PhysicalSettings"))
            physiological_s = resolve_physiological_params(engine_s.simulator, data.get("PhysiologicalSettings"))
            run_s = RunSettings.from_dict(data.get("RunSettings"))

            gpx = run_s.gpx_filename  # required, non-empty field -- from_dict above already validated it
            config_settings_list.append((physical_s, physiological_s, run_s, engine_s, gpx, filename))
        except Exception as e:
            logger.warning("Validation error in %s:", filename)
            if hasattr(e, 'errors'):
                for error in e.errors():
                    logger.warning("  field: %s, reason: %s", error["loc"], error["msg"])
            else:
                logger.warning("  -> %s", e)

    return config_settings_list

# --------------------------------------------------
# III. Result saving
# --------------------------------------------------

def create_json_input_dict(physical_settings, physiological_settings, run_settings, optimizer_params, course_data,
                           data_manager_version, simulator_key, simulator_version, optimizer_key, optimizer_version,
                           cda_ratios=None) -> dict:
    """Build the 'input' section of the strategy JSON.

    Three separate concepts around "what code produced this strategy"
    live in three separate places -- conflating them under one name would
    make this JSON confusing to read:

    - WHICH simulator/optimizer was selected: settings.engine.simulator/
      optimizer (a SIMULATOR_REGISTRY/OPTIMIZER_REGISTRY key, e.g.
      "sim_kiritsubo") -- a SETTING, matching the shape eidos.apps.
      manager's own pre-generation config JSON already uses for its
      "Engine" section. Every consumer that re-resolves this strategy's
      own simulator (eidos.apps.exporter/viewer/designer/trainer/
      analyzer, all via core.simulators.resolve_simulator(...)) reads
      the key from here, not from "versions" below -- eidos.lib.
      optimizer.resolve_optimizer(...) is only used by eidos.apps.
      generator itself, since nothing else re-runs the optimizer search
      (eidos.apps.designer's own Refine is a separate, fixed local
      polish -- see its own module comment for why it deliberately does
      not depend on settings.engine.optimizer/optimizer_params).
    - WHAT MILESTONE of the selected one/data_manager: versions.
      simulator/optimizer/data_manager -- the hand-maintained, human-
      readable SIMULATOR_VERSION/OPTIMIZER_VERSION/DATA_MANAGER_VERSION
      strings (see core.git_info's own module docstring for why these
      can't be trusted as a precise "did the code change" signal). Each
      one only means something once paired with ITS OWN key (settings.
      engine.simulator/optimizer for the first two; data_manager has no
      key of its own -- there's only one implementation today, so
      nothing to distinguish it from). No "_version" suffix on the field
      names themselves -- being inside "versions" already says so.
    - WHAT EXACT STATE the whole repo was in: git_state.commit_hash/
      is_dirty -- precise and fully automatic (core.git_info), unrelated
      to which simulator/optimizer was chosen. git_state.is_dirty
      specifically records is_relevant_code_dirty() (scoped to core.
      git_info.REPRODUCIBILITY_RELEVANT_PATHS), not a whole-repo dirty
      check -- editing an unrelated file while generating a strategy
      shouldn't taint this strategy's recorded reproducibility state.
      Read fresh here rather than passed in as parameters, unlike the
      three hand-maintained version strings -- there's no meaningful
      "caller supplies a different git state" use case the way there
      might theoretically be for the hand-maintained strings.

    settings.engine.optimizer_params persists the validated+defaulted
    optimizer parameters used for this run, purely as a record of what
    produced it -- eidos.apps.designer's Refine deliberately does NOT
    read it back (see designer.py's own module comment: Refine polishes
    the current strategy with fixed local constants, independent of
    whichever optimizer, if any, originally produced it).

    cda_ratios: this simulator's CdA yaw-multiplier table, if
    physical_settings has cda_yaw_table_filename at all (None otherwise,
    e.g. a future simulator with no aero-drag term -- see core.simulators.
    sim_stub's own section-0 comment on the cda/cda_yaw_table_filename
    pairing rule). Stored purely for eidos.apps.viewer's CdA polar plot
    (data.trace re-simulation and every other real consumer reload it
    fresh from physical.cda_yaw_table_filename via each simulator's own
    build_physics_params instead) -- omitted from "data" entirely when
    None, rather than storing an empty placeholder.
    """
    data = {"course_profile": course_data}
    if cda_ratios is not None:
        data["cda_ratios"] = cda_ratios.tolist()

    return {
        "versions": {
            "data_manager": data_manager_version,
            "simulator": simulator_version,
            "optimizer": optimizer_version,
        },
        "git_state": {
            "commit_hash": get_git_commit_hash(),
            "is_dirty": is_relevant_code_dirty(),
        },
        "settings": {
            "physical": physical_settings.model_dump(),
            "physiological": physiological_settings.model_dump(),
            "run": asdict(run_settings),
            "engine": {
                "simulator": simulator_key,
                "optimizer": optimizer_key,
                "optimizer_params": optimizer_params.model_dump(),
            },
        },
        "data": data,
    }

def create_json_output_dict(N_seg_used, seed_used, num_seed_trials, de_optimization_success,
                            total_time_s, rank_nseg, rank_runset,
                            target_power_list, target_length_list) -> dict:
    """Build the 'output' section of the strategy JSON."""
    return {
        "metadata": {
            "n_seg": N_seg_used,
            "seed": seed_used,
            "num_seed_trials": num_seed_trials,
            "de_optimization_success": bool(de_optimization_success),
        },
        "results": {
            "kpis": {
                "total_time_s": float(total_time_s),
                "strategy_rank_in_nseg": int(rank_nseg),
                "strategy_rank_in_run_set": int(rank_runset)
            },
            "strategy": {
                "target_power_list": target_power_list.tolist(),
                "target_length_list": target_length_list.tolist()
            }
        }
    }

def save_experiment_results(all_seed_res, course, physical_settings, physiological_settings, run_settings,
                             engine_settings: EngineSettings, optimizer_params: BaseModel,
                             simulator_spec: SimulatorSpec, optimizer_spec: OptimizerSpec, data_manager_version):
    """
    Rank all optimization results globally and per n_seg, then save each as a compressed
    JSON file and build a Parquet index for the Viewer.
    """
    run_set_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_dir = os.path.join(BASE_STRATEGIES_DIR, f"_{run_set_id}")
    os.makedirs(run_id_dir, exist_ok=True)

    logger.info("Results directory: %s", run_id_dir)

    index_records = []

    course_data = {
        'distance_p_m_list': course.s_p_fine.tolist(),
        'distance_h_m_list': course.s_h_fine.tolist(),
        'latitude_list': course.lat_fine.tolist(),
        'longitude_list': course.lon_fine.tolist(),
        'heading_deg_list': np.degrees(course.heading).tolist(),
        'slope_ratio_list': np.tan(course.slope).tolist(),
        'altitude_list': course.altitude.tolist(),
        'kappa_list': course.kappa.tolist(),
        'v_limit_list': course.v_limit.tolist(),
        'distance_step': course.distance_step
    }

    # Loaded here purely for eidos.apps.viewer's CdA polar plot, which
    # reads it back from the strategy JSON's own input.data.cda_ratios --
    # see create_json_input_dict's own docstring for why this is the one
    # remaining consumer that still needs a stored copy rather than
    # reloading fresh from cda_yaw_table_filename.
    cda_ratios = None
    if "cda_yaw_table_filename" in simulator_spec.physical_param_model.model_fields:
        cda_ratios = load_cda_yaw_table(physical_settings.cda_yaw_table_filename)

    input_dict = create_json_input_dict(
        physical_settings, physiological_settings, run_settings, optimizer_params, course_data,
        data_manager_version, simulator_spec.key, simulator_spec.version,
        optimizer_spec.key, optimizer_spec.version, cda_ratios,
    )

    physics = simulator_spec.build_physics_params(physical_settings, physiological_settings, run_settings, course)

    # 1. Collect results for all n_seg values and compute per-segment rankings
    all_processed_results = []
    for N_seg in range(run_settings.n_seg_min, run_settings.n_seg_max + 1):
        results_for_Nseg = all_seed_res.get(N_seg, [])
        # Counted, not predicted: results_for_Nseg IS the list
        # optimizer_spec.run() actually returned for this N_seg, so its
        # own length is the real seed-trial count -- see eidos.lib.
        # optimizer's module docstring appendix.
        num_seeds_for_Nseg = len(results_for_Nseg)

        nseg_entries = []
        for item in results_for_Nseg:
            p_blocks = optimizer_spec.decode(
                item.x, N_seg, course.distance, run_settings.seg_length_min, optimizer_params
            )
            out = simulator_spec.kernel(0.0, p_blocks, physics, True, False, True)

            res_entry = {
                'N_seg': N_seg,
                'seed': item.seed,
                'success': item.success,
                'p_blocks': p_blocks,
                'time': out.finish_time,
                'num_seeds_for_Nseg': num_seeds_for_Nseg
            }
            nseg_entries.append(res_entry)
            all_processed_results.append(res_entry)

        # Per-segment ranking
        nseg_entries.sort(key=lambda x: x['time'])
        for rank, entry in enumerate(nseg_entries, start=1):
            entry['rank_nseg'] = rank

    # 2. Global ranking across all n_seg values
    all_processed_results.sort(key=lambda x: x['time'])
    for rank, entry in enumerate(all_processed_results, start=1):
        entry['rank_runset'] = rank

    # 3. Save compressed strategy files and build index
    for item in all_processed_results:
        N_seg = item['N_seg']
        s = item['seed']

        output_dict = create_json_output_dict(
            N_seg,
            s,
            item['num_seeds_for_Nseg'],
            item['success'],
            item['time'],
            item['rank_nseg'],
            item['rank_runset'],
            item['p_blocks'].power,
            item['p_blocks'].length
        )

        final_json_data = {
            "run_set_id": run_set_id,
            "input": input_dict,
            "output": output_dict
        }

        flat_record = extract_flat_data(final_json_data)
        validated_record = ExperimentIndexModel(**flat_record).model_dump()
        index_records.append(validated_record)

        file_name = f"strategy_{run_set_id}_N{N_seg}_S{s}.json"
        file_path = os.path.join(run_id_dir, file_name)
        save_strategy_to_json(final_json_data, file_path)

    # 4. Save Parquet index
    if index_records:
        index_df = pd.DataFrame(index_records)
        index_filepath = os.path.join(run_id_dir, "_index.parquet")
        index_df.to_parquet(index_filepath, index=False)
        logger.info("Index created: %d records saved.", len(index_records))

def execute_strategy_generation():
    """Entry point: load all config JSONs and run the optimization pipeline for each."""
    logger.info("Loading all configuration JSONs")
    config_sets = load_config_jsons()

    if not config_sets:
        logger.error("No valid configurations found.")
        return

    logger.info("Loaded %d configuration sets.", len(config_sets))
    logger.info("Starting sequential optimization for %d config set(s).", len(config_sets))

    for i, (physical_s, physiological_s, run_s, engine_s, gpx_file, source_filename) in enumerate(config_sets):
        # Uses the shared log_banner() convention (core.logging_setup) --
        # keeping this per-config-set boundary from getting lost among the
        # heavy per-config DE-optimization output between it and the
        # "Start Execution" banner just below is handled by the Manager's
        # Execution Log coloring banner lines in a dedicated color, not
        # by using a different character style per banner.
        # physiological_s.cp: every SIMULATOR_REGISTRY entry's own
        # PhysiologicalSettings is required to have cp (see
        # core.schema.PhysiologicalSettingsBase), so this is always safe.
        log_banner(
            logger,
            f"CONFIG SET {i + 1}/{len(config_sets)} : {source_filename} "
            f"(CP={physiological_s.cp:.0f} W, GPX={gpx_file})",
        )
        run_full_optimization_pipeline(physical_s, physiological_s, run_s, engine_s, gpx_file)

    log_banner(logger, "Project execution finished.")

def main() -> None:
    configure_logging()
    execute_strategy_generation()


if __name__ == "__main__":
    main()