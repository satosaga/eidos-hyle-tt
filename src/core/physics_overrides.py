##########################
# physics_overrides.py
##########################
"""
Single shared implementation of "take a strategy's raw physical/
physiological/run settings, apply a dict of scalar physics overrides,
return the resolved simulator's own params shape (whatever
SimulatorSpec.build_physics_params produces)". Lives here, not inside
eidos.apps.analyzer, so both eidos.apps.analyzer (PySide6 GUI, manual
Rebuild) and core.calibrator (headless, ProcessPoolExecutor workers,
Auto Fit) can call the same logic without core.calibrator importing
the PySide6-dependent analyzer module.

Simulator-agnostic: reconstructs PhysicalSettings/PhysiologicalSettings
(the resolved simulator's own models) with the overrides merged in and
delegates the actual params construction to simulator_spec.
build_physics_params -- the one place that's supposed to know a given
simulator's parameter shape (see core.simulators' module docstring's
design rationale). No simulator-specific field names appear here.

Reconstructed via model_construct(), not model_validate()/the normal
constructor: this function runs on every DE/Nelder-Mead evaluation
inside core.calibrator's Auto Fit (potentially thousands of calls
per run) as well as every manual Rebuild in eidos.apps.analyzer.
PhysicalSettings.cda_yaw_table_filename's field validator does real file
I/O (reads and parses a CSV) -- paying that cost on every single
evaluation instead of once at the original JSON-loading boundary would
be a severe, silent performance regression. raw_physical/
raw_physiological are already-validated dicts (read back from a
strategy JSON that was itself produced from validated settings), so
re-validating them here has no correctness benefit, only cost.
"""

from core.schema import CourseProfile, RunSettings
from core.simulators import SimulatorSpec


def build_overridden_params(
    simulator_spec: SimulatorSpec,
    raw_physical: dict,
    raw_physiological: dict,
    raw_run: dict,
    course_profile: CourseProfile,
    overrides: dict,
):
    """
    Return simulator_spec's own params (via simulator_spec.build_physics_params)
    with physics overrides applied.

    Course geometry (slope, v_limit's curvature/heading-dependent inputs)
    is always GPX-derived — see eidos.apps.analyzer's Scenario docstring
    for why. v_limit/cos_phi/sin_phi recomputation below delegates to
    simulator_spec.recompute_course_physics -- each simulator's own
    answer to "what does v_limit/wind look like for THIS set of physical
    values" -- not a shared core.course_geometry call, since not every
    simulator's PhysicalSettings carries mu/brake_usability/wind_direction
    (e.g. core.simulators.sim_stub genuinely doesn't).

    Args:
        simulator_spec:    The resolved SimulatorSpec (e.g. a StrategyRecord's
                            own `.simulator_spec`) whose physical_param_model /
                            physiological_param_model / build_physics_params /
                            recompute_course_physics determine the returned
                            params' shape and physics.
        raw_physical:       Strategy JSON's raw physical settings dict
                            (input.settings.physical).
        raw_physiological:  Strategy JSON's raw physiological settings dict
                            (input.settings.physiological).
        raw_run:            Strategy JSON's raw run dict (raw_run) — RunSettings
                            is never itself overridden, just reconstructed.
        course_profile:     core.schema.CourseProfile, already fitted for
                            this strategy's own baseline (kappa/slope/
                            s_p_fine/heading reused as-is; v_limit/cos_phi/
                            sin_phi recomputed below to match `overrides`).
        overrides:          Dict of scalar overrides; keys are the resolved
                            simulator's own PhysicalSettings / PhysiologicalSettings
                            field names (e.g. {"cda": 0.30, "wind_speed": 3.0}
                            or {"cp": 260.0}) — filtered by which model
                            actually has that field, so a single overrides
                            dict can freely mix physical and physiological keys.

    Returns:
        Whatever simulator_spec.build_physics_params returns.
    """
    physical_fields = set(simulator_spec.physical_param_model.model_fields)
    physiological_fields = set(simulator_spec.physiological_param_model.model_fields)

    physical_dict = {**raw_physical, **{k: v for k, v in overrides.items() if k in physical_fields}}
    physiological_dict = {**raw_physiological, **{k: v for k, v in overrides.items() if k in physiological_fields}}
    physical = simulator_spec.physical_param_model.model_construct(**physical_dict)
    physiological = simulator_spec.physiological_param_model.model_construct(**physiological_dict)
    run = RunSettings(**raw_run)

    overridden_course = simulator_spec.recompute_course_physics(course_profile, physical)
    return simulator_spec.build_physics_params(physical, physiological, run, overridden_course)
