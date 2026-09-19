################
# schema.py
################
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from core.io_config import BASE_GPX_DIR


# --------------------------------------------------
# I. Pydantic Validation Models (JSON input sanitization)
# --------------------------------------------------
def validate_flat_filename(v: str, base_dir: str, extension: str) -> str:
    """
    Validate that a filename is a flat (non-nested) file with the expected extension
    that exists in the specified base directory.

    Raises ValueError if the filename contains path separators, the file does not
    exist, or the extension does not match.

    Shared by RunValidationModel.gpx_filename and each SIMULATOR_REGISTRY
    entry's own PhysicalSettings model (e.g. sim_kiritsubo's
    cda_yaw_table_filename) -- which physical parameters a simulator has is
    a per-entry concern, not a global one. See core.simulators' module
    docstring.
    """
    if "/" in v or "\\" in v:
        raise ValueError(f"Subdirectories are not allowed. Please provide only the filename: {v}")
    full_path = Path(base_dir) / v
    if not full_path.is_file():
        raise ValueError(f"File not found in {base_dir}: {v}")
    if full_path.suffix.lower() != extension.lower():
        raise ValueError(f"Invalid extension. Expected '{extension}', got '{full_path.suffix}'")
    return v

# -----------------------------------------------
class RunValidationModel(BaseModel):
    """
    Pydantic validation model for RunSettings JSON input.

    No knot_interval_m field: B-spline knot spacing/degree for course
    geometry smoothing is a numerical-fitting concern, not a per-run user
    setting -- hardcoded in core.course_geometry.fit_course_geometry_profile
    (COURSE_KNOT_INTERVAL_M / COURSE_SPLINE_DEGREE).

    No min_corner_radius field: it clamps curvature for a cornering
    speed-limit calculation that only core.simulators.sim_kiritsubo's own
    braking physics uses (not every simulator has a cornering-limit
    model), so it lives in sim_kiritsubo.PhysicalSettings instead. See
    core.simulators.sim_kiritsubo.compute_course_physics_sim_kiritsubo.
    """
    model_config = ConfigDict(extra="forbid")

    gpx_filename: str = Field(..., min_length=1, title="GPX filename", description="Path to the source GPX file containing course geometry")
    n_seg_min: int = Field(..., ge=0, le=100, title="N_seg min", description="Minimum number of segments for the power allocation strategy [-]")
    n_seg_max: int = Field(..., ge=0, le=100, title="N_seg max", description="Maximum number of segments for the power allocation strategy [-]")
    initial_base_seed: int = Field(..., ge=1, title="Initial base seed", description="Base seed for the pseudo-random number generator [-]")
    seed_factor: int = Field(..., ge=0, le=50, title="Seed factor", description="Multiplier for the number of independent optimization runs [-]")
    time_step: float = Field(..., ge=0.01, le=10.0, title="Time step", json_schema_extra={"unit": "s"}, description="Simulation time resolution [s]")
    seg_power_max: float = Field(..., ge=200.0, le=2000.0, title="Seg power max", json_schema_extra={"unit": "W"}, description="Upper bound for power optimization [W]")
    seg_power_min: float = Field(..., ge=0.0, le=2000.0, title="Seg power min", json_schema_extra={"unit": "W"}, description="Lower bound for power optimization [W]")
    seg_length_min: float = Field(..., ge=0.0, le=5000.0, title="Seg length min", json_schema_extra={"unit": "m"}, description="Minimum segment length [m]")
    distance_step: float = Field(..., ge=0.1, le=10.0, title="Distance step", json_schema_extra={"unit": "m"}, description="Course discretization step [m]")

    @model_validator(mode="after")
    def check_power_range(self) -> "RunValidationModel":
        """Ensure seg_power_max >= seg_power_min.

        A model_validator, not a field_validator on seg_power_max like
        check_n_seg_range below -- seg_power_max is declared BEFORE
        seg_power_min above, so a field_validator on seg_power_max would
        see info.data.get("seg_power_min") as always None (Pydantic v2
        only populates info.data with fields already validated earlier
        in declaration order), silently never firing. mode="after" runs
        once both fields exist regardless of declaration order.
        """
        if self.seg_power_max < self.seg_power_min:
            raise ValueError(f"seg_power_max ({self.seg_power_max}) must be >= seg_power_min ({self.seg_power_min})")
        return self

    @field_validator("n_seg_max")
    @classmethod
    def check_n_seg_range(cls, v: int, info) -> int:
        """Ensure n_seg_max >= n_seg_min."""
        n_min = info.data.get("n_seg_min")
        if n_min is not None and v < n_min:
            raise ValueError(f"n_seg_max ({v}) must be >= n_seg_min ({n_min})")
        return v

    @field_validator("gpx_filename")
    @classmethod
    def check_gpx_path(cls, v: str) -> str:
        """Validate that the GPX file exists in the designated directory."""
        return validate_flat_filename(v, BASE_GPX_DIR, ".gpx")

# -----------------------------------------------
class PhysiologicalSettingsBase(BaseModel):
    """
    Base class every core.simulators.SIMULATOR_REGISTRY entry's own
    PhysiologicalSettings model must inherit from, requiring at least cp
    and w_prime regardless of whether that simulator's own kernel physics
    reads either. Enforced at the type level, not per-app defensive
    handling, because both matter outside any one kernel's own physics:

    - cp: FIT/ZWO export encodes target power as a fraction of CP
      (eidos.apps.exporter.save_zwo divides by it directly).
    - w_prime: every simulator reports a W' balance trajectory
      (SimulationOutput.w_traj), even a constant one for a kernel with no
      depletion/recovery model (see core.simulators.sim_stub).

    A subclass adds its own further fields (e.g. sim_kiritsubo's p_max/
    w_prime_recovery_rate/vitality_loss_rate) and validators over the
    combined field set freely -- only cp/w_prime are required.

    This lets every downstream consumer (core.data_manager.
    extract_export_target, eidos.apps.designer/trainer/generator, eidos.
    apps.analyzer.canvas, eidos.lib.pdf_exporter) read
    physiological_settings['cp']/['w_prime'] unconditionally -- a violated
    invariant should raise a loud KeyError at the first read, not fall
    back to a silently-substituted placeholder.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    cp: float = Field(..., ge=150.0, le=600.0, title="CP", json_schema_extra={"unit": "W", "preset": 280.0}, description="Critical Power [W]")
    w_prime: float = Field(..., ge=5000.0, le=50000.0, title="W'", json_schema_extra={"unit": "J", "preset": 20000.0}, description="Anaerobic capacity [J]")


def bounds_from_field(model_cls: type[BaseModel], field_name: str) -> tuple[float, float]:
    """
    Read (ge, le) straight off a Pydantic ValidationModel Field.

    Single shared implementation, used by core.calibrator.
    bounds_from_schema (calibratable params) and directly by eidos.apps.
    analyzer's PhysicsOverridePanel (W' Balance-only params like cp/
    w_prime, never routed through calibration) -- both read live from the
    schema, no hardcoded copies.

    Args:
        model_cls:  A Pydantic BaseModel subclass (e.g. a core.simulators
                    registry entry's own PhysicalSettings validation model).
        field_name: Field name on model_cls.

    Returns:
        (lo, hi) bounds.

    Raises:
        ValueError: If the schema field has no ge/le constraint pair —
                    fail fast rather than silently falling back to an
                    unbounded range.
    """
    field_info = model_cls.model_fields[field_name]
    lo = hi = None
    for constraint in field_info.metadata:
        if hasattr(constraint, "ge"):
            lo = constraint.ge
        if hasattr(constraint, "le"):
            hi = constraint.le
    if lo is None or hi is None:
        raise ValueError(
            f"{model_cls.__name__}.{field_name} is missing a ge/le bound pair "
            f"in core.schema — cannot read bounds for '{field_name}' without explicit bounds."
        )
    return (float(lo), float(hi))


def gui_decimals_from_field(model_cls: type[BaseModel], field_name: str) -> int:
    """
    Compute a Field's GUI display decimal places, read live off the schema.

    Single shared implementation, used by eidos.apps.manager's GUI Form
    Editor and eidos.apps.analyzer's PhysicsOverridePanel -- same
    "read live, no hand-maintained per-key table" guarantee as
    bounds_from_field/field_display_label above.

    An explicit Field(json_schema_extra={"decimals": ...}) always wins --
    needed for a field whose natural range would otherwise round it to 0
    (e.g. a gt=-only field with no le=, or a field whose true precision
    needs is finer than its [ge, le] span alone would suggest; see
    eidos.lib.optimizers.opt_tenchi.TenchiParams' de_tol/refine_xatol_w/
    refine_fatol_s/refine_restart_eps_s for real examples). Otherwise,
    decimal places are derived from the [ge, le] span (falling back to an
    assumed [0, 100] span if either bound is absent) -- narrower spans get
    more decimal places, on the theory that a field usable only over a
    narrow range needs finer resolution to be usable at all.

    Args:
        model_cls:  A Pydantic BaseModel subclass.
        field_name: Field name on model_cls.

    Returns:
        Decimal places to display, in [2, 10].
    """
    field_info = model_cls.model_fields[field_name]
    extra = field_info.json_schema_extra or {}
    if isinstance(extra, dict) and "decimals" in extra:
        decimals = extra["decimals"]
        assert isinstance(decimals, int), f"decimals must be an int, got {decimals!r}"
        return decimals

    lo = hi = None
    for constraint in field_info.metadata:
        if hasattr(constraint, "ge"):
            lo = constraint.ge
        if hasattr(constraint, "le"):
            hi = constraint.le
    lo = lo if lo is not None else 0.0
    hi = hi if hi is not None else 100.0

    span = abs(float(hi) - float(lo))
    if span <= 0:
        return 2
    try:
        calculated = math.ceil(-math.log10(span / 1000.0))
        return max(2, min(calculated, 10))
    except ValueError:
        return 2


def field_display_label(model_cls: type[BaseModel], field_name: str) -> str:
    """
    Single source of truth for a Field's on-screen label, shared by
    eidos.apps.manager's GUI Form Editor and eidos.apps.analyzer's
    parameter panel. Reading both from Field(title=...,
    json_schema_extra={"unit": ...}) keeps them in sync by construction.

    No mechanical fallback if title is unset -- falls back to the raw
    field_name instead, so a field someone forgot to give a title reads
    as visibly wrong (raw snake_case in a GUI) rather than being quietly
    patched over by a plausible-looking guess.
    """
    info = model_cls.model_fields[field_name]
    title = info.title or field_name
    extra = info.json_schema_extra
    unit = extra.get("unit") if isinstance(extra, dict) else None
    return f"{title} [{unit}]" if unit else title


def preset_value_from_field(model_cls: type[BaseModel], field_name: str):
    """
    A field's own Field(json_schema_extra={"preset": ...}) value, if any --
    the starting value eidos.apps.manager's GUI Form Editor seeds a
    required field with when config_data doesn't have it yet (e.g. right
    after a simulator switch introduces a field the previous model didn't
    have -- see ConfigurationEditorPane._initialize_form_from_data).

    Returns None if the field has no preset. cp/w_prime
    (PhysiologicalSettingsBase) are the deliberate example: rider-specific
    values with no universally reasonable number must surface as a loud
    validation error when missing, not a silently-seeded guess.

    Deliberately does NOT fall back to a field's own Pydantic default --
    every field the GUI seeds stays Field(...) (required) with an explicit
    "preset" instead, including eidos.lib.optimizer.OptimizerSpec.
    param_model entries (e.g. TenchiParams). A real Pydantic default would
    let a config JSON silently omit the field and run with a value nobody
    consciously chose; an optimizer with real tunable parameters should
    always show every one for explicit review.

    Purely a GUI convenience: the field itself stays Field(...) (required)
    regardless of whether it has a preset -- this never relaxes what
    counts as a valid, complete config for any other consumer.
    """
    info = model_cls.model_fields[field_name]
    extra = info.json_schema_extra
    return extra.get("preset") if isinstance(extra, dict) else None


# --------------------------------------------------
# II. High-performance parameter containers (Numba-compatible)
# --------------------------------------------------
class PowerBlocks(NamedTuple):
    """
    Target power strategy expressed as a sequence of constant-power segments.

    Attributes:
        power:  Target power for each segment [W].
        length: Length of each segment [m].
    """
    power: np.ndarray
    length: np.ndarray

class PhysicsParams(NamedTuple):
    """
    Immutable container of pure physical and geometric constants required by the
    simulator. Excludes optimization constraints (P_min, L_min, etc.) to keep
    the physics engine independent of search strategy.

    Fields are numbered to match Numba's positional tuple access.
    """
    # Course geometry
    slope: np.ndarray            # 1  road gradient [rad]
    v_limit: np.ndarray          # 2  speed ceiling from curvature and braking [m/s]
    # Simulation control
    time_step: float             # 3  time step [s]
    distance_step: float         # 4  course discretization step [m]
    # Rider physical characteristics
    cp: float                    # 5  Critical Power [W]
    w_prime: float               # 6  anaerobic work capacity [J]
    p_max: float                 # 7  maximum instantaneous physiological power [W]
    total_weight: float          # 8  total mass (rider + bike) [kg]
    cda: float                   # 9  aerodynamic drag area [m^2]
    f_max: float                 # 10 maximum pedal force [N]
    vitality_loss_rate: float    # 11 vitality slope correction factor [-]
    w_prime_recovery_rate: float # 12 W' recovery rate constant K (Skiba & Clarke 2021, W'BAL-KODE) [-]
    brake_lookahead: float       # 13 braking lookahead window [s]
    # Environmental physical constants
    gravity_accel: float         # 14 local gravitational acceleration [m/s^2]
    air_density: float           # 15 air density [kg/m^3]
    crr: float                   # 16 rolling resistance coefficient [-]
    # Wind
    wind_speed: float            # 17 ambient wind speed [m/s]
    cda_ratios: np.ndarray       # 18 CdA multiplier table indexed by yaw angle [deg] (0-180)
    cos_phi: np.ndarray          # 19 cos(heading - wind_direction_to) per course point
    sin_phi: np.ndarray          # 20 sin(heading - wind_direction_to) per course point

class SimulationOutput(NamedTuple):
    """
    Full result container returned by the physics simulator.

    Scalar summary fields are always populated. Trajectory arrays are populated
    only when return_trajectory=True; otherwise they contain np.empty(0).
    """
    # Summary scalars
    finish_time: float      # goal time with segment-end interpolation correction [s]
    penalty_factor: float   # penalty multiplier for physiological constraint violations
    # Time-series trajectories
    t_traj: np.ndarray          # time [s]
    x_traj: np.ndarray          # cumulative road distance [m]
    v_traj: np.ndarray          # velocity [m/s]
    p_traj: np.ndarray          # propulsive power [W]
    w_traj: np.ndarray          # remaining W' [J]
    cp_eff_traj: np.ndarray     # effective CP after vitality correction [W]
    v_w_app_traj: np.ndarray    # apparent wind speed [m/s]
    psi_w_app_traj: np.ndarray  # apparent wind yaw angle [deg]

# --------------------------------------------------
# III. Settings dataclasses (input / UI layer)
# --------------------------------------------------

@dataclass(frozen=True)
class OptimizationStrategy:
    """
    Tactical constraints for the optimization search.
    Has no involvement in the physics calculation itself.
    """
    n_seg: int
    seg_power_min: float
    seg_power_max: float
    seg_length_min: float

@dataclass(frozen=True)
class RunSettings:
    """
    Simulation and algorithm execution parameters (Input.Settings.Run).
    Constructed via from_dict() which runs Pydantic validation before instantiation.
    """
    gpx_filename: str            # GPX filename (basename only, no subdirectory)
    n_seg_min: int               # minimum number of segments for the power allocation strategy [-]
    n_seg_max: int               # maximum number of segments for the power allocation strategy [-]
    initial_base_seed: int       # base seed for the PRNG [-]
    seed_factor: int             # multiplier for number of independent optimization runs [-]
    time_step: float             # simulation time resolution [s]
    seg_power_max: float         # upper bound for power optimization [W]
    seg_power_min: float         # lower bound for power optimization [W]
    seg_length_min: float        # minimum segment length [m]
    distance_step: float         # course discretization step [m]
    # No knot_interval_m/min_corner_radius fields -- see
    # RunValidationModel's docstring.

    @classmethod
    def from_dict(cls, data: dict) -> "RunSettings":
        """Validate input dict via Pydantic and return a RunSettings instance."""
        from pydantic import ValidationError
        try:
            valid_obj = RunValidationModel.model_validate(data)
            return cls(**valid_obj.model_dump())
        except ValidationError as e:
            raise e

    def validate_with_course(self, l_total_m: float) -> None:
        """
        Validate physical distance limits and DE search space feasibility.

        Raises ValueError if the course exceeds 500 km, or if the minimum
        segment length constraint makes the search space infeasible.
        """
        if l_total_m > 500.0 * 1000:
            raise ValueError(f"Distance Over: {l_total_m/1000:.1f}km exceeds 500km limit.")
        min_required = self.seg_length_min * self.n_seg_max
        if min_required >= l_total_m:
            raise ValueError(
                f"Constraint Error: L_min * N_seg_max ({min_required:.1f}m) "
                f"is greater than course distance ({l_total_m:.1f}m). "
                "Decrease seg_length_min or n_seg_max."
            )

# EngineSettings (simulator/optimizer registry selection) lives in
# eidos.lib.optimizer, not here -- see that module's docstring. It's an
# EIDOS-only concept (HYLE never selects an optimizer), so keeping it in
# core/ would mean core/ importing from eidos/lib/, which core/ may
# never do.


# --------------------------------------------------
# IV. Data structures
# --------------------------------------------------

@dataclass(frozen=True)
class CoursePoints:
    """
    Raw discrete point sequence extracted and aggregated from a GPX file.
    Coordinates are projected to a local flat-earth Cartesian frame.
    """
    x: np.ndarray     # easting from origin [m]
    y: np.ndarray     # northing from origin [m]
    z: np.ndarray     # elevation [m]
    s_h: np.ndarray   # cumulative horizontal distance [m]
    s_p: np.ndarray   # cumulative road distance along slope [m]
    distance: float   # total road distance (simulation primary axis) [m]
    origin_lat: float # reference latitude [deg]
    origin_lon: float # reference longitude [deg]

@dataclass(frozen=True)
class CourseProfile:
    """
    High-resolution physical and geometric course profile used by the simulator.
    Sampled at uniform distance_step intervals along the road distance axis.
    """
    distance: float      # total road distance [m]
    distance_step: float # sampling interval [m]
    s_p_fine: np.ndarray # road distance array [m]
    s_h_fine: np.ndarray # horizontal distance array [m]
    lat_fine: np.ndarray # latitude [deg]
    lon_fine: np.ndarray # longitude [deg]
    slope: np.ndarray    # road gradient [rad]
    kappa: np.ndarray    # curvature [1/m]
    v_limit: np.ndarray  # speed ceiling from curvature and braking [m/s]
    altitude: np.ndarray # elevation [m]
    heading: np.ndarray  # heading angle [rad], North=0, clockwise
    cos_phi: np.ndarray  # cos(heading - wind_direction_to)
    sin_phi: np.ndarray  # sin(heading - wind_direction_to)

@dataclass(frozen=True)
class ExportTarget:
    """
    All information required to export a single optimized strategy to FIT/ZWO files.
    Populated by core.data_manager.extract_export_target() from a strategy JSON.

    cp/w_prime are always present: every core.simulators.SIMULATOR_REGISTRY
    entry's own PhysiologicalSettings is required to define cp/w_prime (see
    core.schema.PhysiologicalSettingsBase), so extract_export_target reads
    them unconditionally -- no per-simulator fallback needed here.

    Holds exactly what eidos.apps.exporter/eidos.lib.pdf_exporter actually
    read. The re-simulation exporter.py performs gets a strategy's real
    physical/physiological values by reconstructing its own resolved
    SimulatorSpec's physical_param_model/physiological_param_model
    directly via model_construct() -- this dataclass isn't that path's
    source of truth, just the export-report fields.
    """
    run_set_id: str
    n_seg_used: int
    seed_used: int
    target_power_list: np.ndarray    # target power per segment [W]
    target_length_list: np.ndarray   # segment length [m]
    gpx_filename: str
    cp: float                       # Critical Power [W]
    w_prime: float                  # anaerobic work capacity [J]