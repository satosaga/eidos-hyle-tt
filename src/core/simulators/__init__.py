####################
# core/simulators/__init__.py
####################
"""
Simulator registry: maps a config's Engine.simulator key (or
DEFAULT_SIMULATOR_KEY, for callers with no config to read) to a
SimulatorSpec pairing one @njit physics kernel with the builder that
assembles its PhysicsParams from settings + course geometry, plus this
entry's own PhysicalSettings/PhysiologicalSettings Pydantic models --
kept together because a future kernel modeling something other than the
CP/W'/P_max_physio three-parameter model would need its own
PhysicsParams-equivalent shape and builder, not just a different kernel
function, and could equally need its own physical/physiological
parameter shape (see core.simulators.sim_stub.PhysiologicalSettings for
a real example of a divergent shape -- cp/w_prime are still required on
every entry, but sim_stub adds no further field at all, unlike
sim_kiritsubo's p_max/w_prime_recovery_rate/vitality_loss_rate).

Course-physics preprocessing (curvature/speed-limit/wind-geometry
physics -- how far this entry's own PhysicalSettings shape must extend
beyond what its kernel directly reads) is ALSO each entry's own
responsibility (compute_course_physics/recompute_course_physics below):
only the course-SHAPE fit itself (core.course_geometry.
fit_course_geometry_profile -- pure GPX geometry, no physical parameter)
stays genuinely shared. A simulator whose kernel has no braking-limit or
wind model (core.simulators.sim_stub) simply doesn't call the
speed-limit/wind-geometry physics at all, and its own PhysicalSettings
has no obligation to carry fields nothing it owns ever reads -- see
core.simulators.sim_kiritsubo.compute_course_physics_sim_kiritsubo vs.
core.simulators.sim_stub.compute_course_physics_sim_stub for the
concrete contrast. Physics-parameter-dependent course logic living
inside each simulator's own file also means it's automatically covered
by that file's own SIMULATOR_VERSION bump policy
(scripts/check_code_version_bump.sh) -- a separate, unversioned shared
module would let a real change to that physics escape reproducibility
tracking entirely.
"""
from typing import Any, Callable, NamedTuple, Type

from pydantic import BaseModel

from core.schema import CoursePoints, CourseProfile, PowerBlocks, RunSettings
from core.simulators.sim_kiritsubo import SIMULATOR_VERSION as _SIM_KIRITSUBO_VERSION
from core.simulators.sim_kiritsubo import PhysicalSettings as _SimKiritsuboPhysical
from core.simulators.sim_kiritsubo import (
    PhysiologicalSettings as _SimKiritsuboPhysiological,
)
from core.simulators.sim_kiritsubo import (
    build_physics_params_sim_kiritsubo,
    compute_course_physics_sim_kiritsubo,
    recompute_course_physics_sim_kiritsubo,
)
from core.simulators.sim_kiritsubo import (
    simulate_power_profile_separated_blocks as _simulate_sim_kiritsubo,
)
from core.simulators.sim_stub import SIMULATOR_VERSION as _SIM_STUB_VERSION
from core.simulators.sim_stub import PhysicalSettings as _SimStubPhysical
from core.simulators.sim_stub import PhysiologicalSettings as _SimStubPhysiological
from core.simulators.sim_stub import (
    build_physics_params_sim_stub,
    compute_course_physics_sim_stub,
    recompute_course_physics_sim_stub,
)
from core.simulators.sim_stub import (
    simulate_dummy_constant_power as _simulate_sim_stub,
)


class SimulatorSpec(NamedTuple):
    # SIMULATOR_REGISTRY's own key for this entry -- distinct from `version`
    # (the hand-maintained, pre-commit-bump-tracked milestone string, which
    # today happens to equal the registry key but isn't guaranteed to).
    # Self-describing so a caller holding a resolved SimulatorSpec (e.g.
    # eidos.apps.analyzer.models.StrategyRecord) can pass `.key` on to a
    # further callee (e.g. core.calibrator.calibrate) that itself
    # needs to re-resolve the same simulator inside a separate
    # ProcessPoolExecutor worker, without threading a second parallel
    # "which simulator" value through by hand.
    key: str
    version: str
    # This entry's own physical-parameter Pydantic model -- the JSON
    # boundary's PhysicalSettings validator AND, since every field's
    # constraints live only here, the mechanical source of truth for
    # calibratable_physical_keys below.
    physical_param_model: Type[BaseModel]
    # This entry's own physiological-parameter Pydantic model. Never a
    # calibration target (see calibratable_physical_keys) -- physiological
    # fields only affect the physiologically-clamped power path
    # (is_target_power=True), never the externally-driven replay path
    # calibration/analysis always runs (is_target_power=False).
    physiological_param_model: Type[BaseModel]
    # `Any` in place of PhysicsParams for the params argument/return type:
    # each simulator has its own parameter NamedTuple shape (e.g. sim_stub's
    # DummyPhysicsParams has far fewer fields than sim_kiritsubo.py's PhysicsParams --
    # see core.simulators.sim_stub's module docstring), so PowerBlocks/
    # SimulationOutput (shared by every simulator) are the only parts of
    # this call signature that stay precisely typed.
    kernel: Callable[[float, PowerBlocks, Any, bool, bool, bool], Any]
    # No cda_ratios argument: each simulator's own build_physics_params
    # reads physical.cda_yaw_table_filename and loads the table itself,
    # if it even has that field at all (sim_stub's own PhysicalSettings
    # doesn't -- no yaw-dependent CdA model to feed).
    build_physics_params: Callable[[BaseModel, BaseModel, RunSettings, CourseProfile], Any]
    # Full course-physics pipeline (course-shape fit + whichever further
    # physics -- speed limits, wind geometry -- this simulator's own
    # kernel actually needs) for a FRESH course: raw CoursePoints -> this
    # simulator's own CourseProfile. See core.simulators' own module
    # docstring for why this lives here (per-entry, version-tracked)
    # rather than as a single shared core.course_geometry function.
    compute_course_physics: Callable[[CoursePoints, BaseModel, RunSettings], CourseProfile]
    # Cheap partial recompute: given an ALREADY-FITTED CourseProfile (same
    # course-shape fields: kappa/slope/s_p_fine/heading/etc.) and a
    # (possibly overridden) physical settings instance, return a new
    # CourseProfile with just v_limit/cos_phi/sin_phi updated to match --
    # without repeating the expensive B-spline fit. Used by
    # core.physics_overrides.build_overridden_params (Analyzer's manual
    # Rebuild, core.calibrator's Auto Fit) and by
    # eidos.apps.exporter's FIT/ZWO export re-simulation, both of which
    # need this simulator's own answer to "what does v_limit/wind look
    # like for THIS set of physical values", not a hardcoded one.
    recompute_course_physics: Callable[[CourseProfile, BaseModel], CourseProfile]


SIMULATOR_REGISTRY: dict[str, SimulatorSpec] = {
    "sim_kiritsubo": SimulatorSpec(
        key="sim_kiritsubo",
        version=_SIM_KIRITSUBO_VERSION,
        physical_param_model=_SimKiritsuboPhysical,
        physiological_param_model=_SimKiritsuboPhysiological,
        kernel=_simulate_sim_kiritsubo,
        build_physics_params=build_physics_params_sim_kiritsubo,
        compute_course_physics=compute_course_physics_sim_kiritsubo,
        recompute_course_physics=recompute_course_physics_sim_kiritsubo,
    ),
    "sim_stub": SimulatorSpec(
        key="sim_stub",
        version=_SIM_STUB_VERSION,
        physical_param_model=_SimStubPhysical,
        physiological_param_model=_SimStubPhysiological,
        kernel=_simulate_sim_stub,
        build_physics_params=build_physics_params_sim_stub,
        compute_course_physics=compute_course_physics_sim_stub,
        recompute_course_physics=recompute_course_physics_sim_stub,
    ),
}

# Simulator key for callers with no config JSON / Engine section to read
# (hyle.* tools doing ad-hoc analysis, not eidos.apps.generator's
# experiment pipeline). A named default, not "whatever's in the registry":
# once a second SIMULATOR_REGISTRY entry exists, iteration order or
# insertion order must not silently decide which physics model those
# callers get.
DEFAULT_SIMULATOR_KEY = "sim_kiritsubo"


def resolve_simulator(name: str) -> SimulatorSpec:
    """Look up a simulator implementation by its SIMULATOR_REGISTRY key."""
    try:
        return SIMULATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown simulator '{name}'. Available: {sorted(SIMULATOR_REGISTRY)}") from None


def resolve_physical_params(name: str, raw: dict) -> BaseModel:
    """Validate `raw` (a config JSON's top-level PhysicalSettings dict)
    against the given simulator's own physical_param_model. Mirrors
    eidos.lib.optimizer.resolve_optimizer_params. Runs full Pydantic
    validation (including e.g. PhysicalSettings.cda_yaw_table_filename's
    CSV file check) -- call only at a genuine JSON-loading boundary (e.g.
    eidos.apps.generator.load_config_jsons), never per-evaluation in a hot
    loop; see core.physics_overrides' module docstring for why
    internal reconstruction sites use model_construct() instead."""
    return resolve_simulator(name).physical_param_model.model_validate(raw)


def resolve_physiological_params(name: str, raw: dict) -> BaseModel:
    """Validate `raw` (a config JSON's top-level PhysiologicalSettings dict)
    against the given simulator's own physiological_param_model. See
    resolve_physical_params' docstring."""
    return resolve_simulator(name).physiological_param_model.model_validate(raw)


def calibratable_physical_keys(name: str) -> list[str]:
    """Derive which of the given simulator's physical_param_model fields are
    valid core.calibrator free_keys: numeric fields with both a `ge`
    and `le` bound (core.schema.bounds_from_field can read a range off
    them). A string field like PhysicalSettings.cda_yaw_table_filename has
    neither, so it's automatically excluded -- no hand-maintained
    allow-list means a field can't go calibratable-but-forgotten or vice
    versa."""
    model_cls = resolve_simulator(name).physical_param_model
    keys = []
    for field_name, field_info in model_cls.model_fields.items():
        has_ge = any(hasattr(c, "ge") for c in field_info.metadata)
        has_le = any(hasattr(c, "le") for c in field_info.metadata)
        if has_ge and has_le:
            keys.append(field_name)
    return keys


def physiological_only_keys(name: str) -> list[str]:
    """Derive which of the given simulator's parameter keys are W' Balance/
    physiological-only -- i.e. never enter the velocity/position ODE (see
    physiological_param_model's own field comment above: it only affects
    the physiologically-clamped power path, is_target_power=True, never
    the externally-driven replay path calibration/analysis always runs).
    Read live off the model, same "no hand-maintained list to drift out of
    sync" reasoning as calibratable_physical_keys."""
    return list(resolve_simulator(name).physiological_param_model.model_fields)
