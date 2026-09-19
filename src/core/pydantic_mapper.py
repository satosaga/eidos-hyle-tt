#########################
# pydantic_mapper.py
#########################
from typing import Any, Dict, Optional

from pydantic import create_model

TYPE_MAP = {
    # --- Root ---
    'run_set_id': 'string',
    # --- Input settings ---
    # Three separate concepts, three separate JSON sections -- see
    # eidos.apps.generator.create_json_input_dict's own docstring for the
    # full reasoning:
    # - versions.data_manager/simulator/optimizer: hand-maintained,
    #   human-readable milestone/variant labels (never auto-verified).
    #   Each only means something paired with settings.engine.simulator/
    #   optimizer below (WHICH one it's a milestone OF) -- data_manager
    #   has no key of its own since there's only one implementation today.
    # - settings.engine.simulator/optimizer: the SIMULATOR_REGISTRY/
    #   OPTIMIZER_REGISTRY key actually used (WHICH one, a setting, not a
    #   version).
    # - git_state.commit_hash/is_dirty: precise, low-maintenance, and
    #   unrelated to which simulator/optimizer was chosen -- see
    #   core.git_info's module docstring for the full reproducibility
    #   story (why Viewer/Exporter/Trainer's re-simulation of a strategy's
    #   stored strategy can silently diverge from the original if the
    #   resolved core/simulators/*.py kernel has since changed, and how
    #   these two fields let that be detected). Strategies predating this
    #   feature have these as None -- see build_dynamic_model()'s
    #   Optional[...] typing.
    'input.versions.data_manager': 'string',
    'input.versions.simulator': 'string',
    'input.versions.optimizer': 'string',
    'input.settings.engine.simulator': 'string',
    'input.settings.engine.optimizer': 'string',
    'input.git_state.commit_hash': 'string',
    'input.git_state.is_dirty': 'boolean',
    # Physiological (W'-balance / power-availability) characteristics --
    # sim_kiritsubo's own PhysiologicalSettings is the union of every
    # rider field that's physiological under the is_target_power-causal
    # axis (see core.simulators' module docstring). Hardcoded to
    # sim_kiritsubo's own field names here -- this index schema is not
    # simulator-generic -- see core.simulators.sim_kiritsubo.
    # PhysiologicalSettings/PhysicalSettings.
    'input.settings.physiological.cp': 'float',
    'input.settings.physiological.w_prime': 'float',
    'input.settings.physiological.w_prime_recovery_rate': 'float',
    'input.settings.physiological.vitality_loss_rate': 'float',
    # Physical (mass/aero/braking/environment) parameters
    'input.settings.physical.rider_weight': 'float',
    'input.settings.physical.cda': 'float',
    'input.settings.physical.f_max': 'float',
    'input.settings.physical.brake_lookahead': 'float',
    'input.settings.physical.brake_usability': 'float',
    'input.settings.physical.bike_weight': 'float',
    'input.settings.physical.gravity_accel': 'float',
    'input.settings.physical.air_density': 'float',
    'input.settings.physical.crr': 'float',
    'input.settings.physical.mu': 'float',
    'input.settings.physical.wind_speed': 'float',
    'input.settings.physical.wind_direction': 'float',
    'input.settings.physical.cda_yaw_table_filename': 'string',
    # Run settings
    'input.settings.run.gpx_filename': 'string',
    'input.settings.run.time_step': 'float',
    'input.settings.run.seg_power_max': 'float',
    'input.settings.run.seg_power_min': 'float',
    'input.settings.run.seg_length_min': 'float',
    # --- Output metadata ---
    'output.metadata.n_seg': 'integer',
    'output.metadata.seed': 'integer',
    'output.metadata.de_optimization_success': 'boolean',
    # --- Output KPIs ---
    'output.results.kpis.total_time_s': 'float',
    'output.results.kpis.strategy_rank_in_run_set': 'integer',
    'output.results.kpis.strategy_rank_in_nseg': 'integer',
}

def build_dynamic_model():
    """Dynamically build a Pydantic model from TYPE_MAP with all fields optional (default None).

    A (T, None) field definition alone does *not* make a field accept
    None in pydantic v2 -- the annotation itself must be Optional[T], or
    passing an explicit None (which extract_flat_data() does for every
    missing/absent key, e.g. a field a strategy JSON predates, like
    git_state.commit_hash/is_dirty) raises a ValidationError.
    """
    field_definitions = {}
    for key, type_str in TYPE_MAP.items():
        if type_str == 'float':
            field_definitions[key] = (Optional[float], None)
        elif type_str == 'integer':
            field_definitions[key] = (Optional[int], None)
        elif type_str == 'boolean':
            field_definitions[key] = (Optional[bool], None)
        else:
            field_definitions[key] = (Optional[str], None)

    return create_model('ExperimentIndexModel', **field_definitions)

# Pydantic validation model for the experiment index
ExperimentIndexModel = build_dynamic_model()

def extract_flat_data(nested_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a flat dict from a nested JSON dict using TYPE_MAP dot-notation keys.
    Missing keys are stored as None.
    """
    flat_data: Dict[str, Any] = {}
    for dot_key in TYPE_MAP.keys():
        keys = dot_key.split('.')
        curr = nested_dict
        try:
            for k in keys:
                curr = curr[k]
            flat_data[dot_key] = curr
        except (KeyError, TypeError):
            flat_data[dot_key] = None
    return flat_data