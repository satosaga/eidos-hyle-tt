####################
# core/simulators/sim_kiritsubo.py
####################
"""
sim_kiritsubo simulator: the CP/W'/P_max_physio three-parameter physics
kernel plus its PhysicsParams builder. Registered in
core.simulators.SIMULATOR_REGISTRY under the "sim_kiritsubo" key.

Course-physics (braking/cornering speed limits, wind geometry) also
lives here, not in core.course_geometry -- a simulator with no
braking-limit or wind model (core.simulators.sim_stub) has no reason to
carry PhysicalSettings fields just to satisfy a shared function's
inputs. Only the course-SHAPE fit itself
(core.course_geometry.fit_course_geometry_profile -- pure GPX geometry)
is genuinely shared; this module owns what to DO with that shape
physically. Keeping the physics here also means it's covered by this
file's own SIMULATOR_VERSION bump policy
(scripts/check_code_version_bump.sh), which a separate, unversioned
module could not guarantee.
"""
import dataclasses
from pathlib import Path

import numpy as np
from numba import njit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.course_geometry import _compute_wind_geometry, fit_course_geometry_profile
from core.data_manager import load_cda_yaw_table
from core.io_config import BASE_CDA_YAW_TABLES_DIR
from core.schema import (
    CoursePoints,
    CourseProfile,
    PhysicsParams,
    PhysiologicalSettingsBase,
    PowerBlocks,
    RunSettings,
    SimulationOutput,
    validate_flat_filename,
)


# --------------------------------------------------
# 0. PhysicalSettings / PhysiologicalSettings
# --------------------------------------------------
# This simulator's own SIMULATOR_REGISTRY.physical_param_model /
# physiological_param_model -- see core.simulators' module docstring for
# the axis these two are split on: PhysicalSettings holds every value
# causally connected to speed calculation whenever power is driven
# externally (is_target_power=False -- calibration/replay always runs
# this way); PhysiologicalSettings holds only what this kernel's own
# physiological power-availability clamp (P_exerting, active when
# is_target_power=True) needs, and never affects calibration (see
# core.simulators.calibratable_physical_keys, which derives calibration
# targets from physical_param_model only).
class PhysicalSettings(BaseModel):
    """Physical (mass/aero/braking/environment) parameters -- causally
    connected to speed calculation even with power driven externally
    (is_target_power=False). Frozen: used directly as a value object
    throughout eidos.apps/eidos.lib, not just at the JSON-loading
    boundary -- see core.physics_overrides' module docstring for why
    reconstruction sites use model_construct() rather than re-running
    these field validators (in particular cda_yaw_table_filename's CSV
    file I/O below) on every simulator re-invocation.

    Verified directly against this kernel: with is_target_power=False,
    the only settings reachable from the velocity/position update are
    this class's own fields (mass, aero, rolling resistance, braking,
    wind) -- PhysiologicalSettings' cp/w_prime/p_max/etc. only ever
    feed the recorded W_traj/CP_eff trace below, never the physics
    itself, in that mode."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    rider_weight: float = Field(..., ge=30.0, le=120.0, title="Rider weight", json_schema_extra={"unit": "kg", "preset": 70.0}, description="Rider mass [kg]")
    cda: float = Field(..., ge=0.10, le=1.00, title="CdA", json_schema_extra={"unit": "m²", "preset": 0.35}, description="Aerodynamic drag area [m^2]")
    f_max: float = Field(..., ge=200.0, le=400.0, title="F_max", json_schema_extra={"unit": "N", "preset": 400.0}, description="Maximum pedal force [N]")
    brake_lookahead: float = Field(..., ge=0.5, le=5.0, title="Brake lookahead", json_schema_extra={"unit": "s", "preset": 1.2}, description="Braking prediction window [s]")
    brake_usability: float = Field(..., ge=0.0, le=1.0, title="Brake usability", json_schema_extra={"preset": 0.7}, description="Confidence in braking performance [-]")
    bike_weight: float = Field(..., ge=5.0, le=20.0, title="Bike weight", json_schema_extra={"unit": "kg", "preset": 8.0}, description="Bike mass [kg]")
    gravity_accel: float = Field(..., ge=9.76, le=9.84, title="Gravity", json_schema_extra={"unit": "m/s²", "preset": 9.80665}, description="Local gravity [m/s^2]")
    air_density: float = Field(..., ge=0.8, le=1.35, title="Air density", json_schema_extra={"unit": "kg/m³", "preset": 1.225}, description="Air density [kg/m^3]")
    crr: float = Field(..., ge=0.001, le=0.02, title="Crr", json_schema_extra={"preset": 0.004}, description="Rolling resistance coefficient [-]")
    mu: float = Field(..., ge=0.1, le=1.2, title="mu", json_schema_extra={"preset": 0.6}, description="Tire-road friction coefficient [-]")
    # Clamps GPS-derived curvature before it feeds
    # _compute_speed_limits_core's own cornering speed-limit calculation
    # below -- a genuinely simulator-specific physics choice (core.
    # simulators.sim_stub has no cornering-limit model at all), not
    # shared course geometry. See core.course_geometry.
    # fit_course_geometry_profile's own docstring and
    # compute_course_physics_sim_kiritsubo below.
    min_corner_radius: float = Field(..., ge=3.0, le=100.0, title="Min corner radius", json_schema_extra={"unit": "m", "preset": 15.0}, description="Minimum cornering radius [m]")
    wind_speed: float = Field(..., ge=0.0, le=25.0, title="Wind speed", json_schema_extra={"unit": "m/s", "preset": 0.0}, description="Ambient wind speed [m/s]")
    wind_direction: float = Field(..., ge=0.0, le=360.0, title="Wind direction", json_schema_extra={"unit": "deg", "preset": 0.0}, description="Wind direction (North=0, Clockwise) [deg]")
    cda_yaw_table_filename: str = Field(...,
        title="CdA yaw table filename",
        json_schema_extra={"preset": "constant_model.csv"},
        description=r"Multiplier $f_{table}(\psi)$ where $CdA_{total} = CdA_{rider} \times f_{table}(\psi)$."
    )

    @field_validator("cda_yaw_table_filename")
    @classmethod
    def validate_cda_yaw_table(cls, v: str) -> str:
        """Validate CdA yaw table CSV: existence, structure, and physical plausibility."""
        validate_flat_filename(v, BASE_CDA_YAW_TABLES_DIR, ".csv")
        full_path = Path(BASE_CDA_YAW_TABLES_DIR) / v
        try:
            data = np.loadtxt(str(full_path.absolute()), delimiter=",", ndmin=2)
            if data.shape[1] < 2:
                raise ValueError("CSV must have at least 2 columns (yaw, factor).")
            yaws = data[:, 0]
            factors = data[:, 1]
            if not (np.isclose(yaws[0], 0.0) and np.isclose(yaws[-1], 180.0)):  # yaw range must be [0.0, 180.0]
                raise ValueError(f"Yaw range must be [0.0, 180.0]. Found: [{yaws[0]}, {yaws[-1]}]")
            if not np.all(np.diff(yaws) > 0):  # yaw angles must be strictly increasing
                raise ValueError("1st column (yaw) must be strictly increasing.")
            if np.any((factors < 0.0) | (factors > 10.0)):  # CdA multiplier sanity check
                raise ValueError("2nd column (factors) must be within [0.0, 10.0].")
        except Exception as e:
            raise ValueError(f"Content validation failed for '{v}': {str(e)}")
        return v


class PhysiologicalSettings(PhysiologicalSettingsBase):
    """Physiological (W'-balance / power-availability) parameters -- affect
    speed only via the physiological power clamp active when
    is_target_power=True; never read during calibration/replay
    (is_target_power=False) and never a calibration target (see
    core.simulators.calibratable_physical_keys). cp/w_prime themselves are
    inherited from PhysiologicalSettingsBase -- see that class's own
    docstring for why every SIMULATOR_REGISTRY entry is required to have
    them regardless of whether its own kernel physics reads them."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    p_max: float = Field(..., ge=200.0, le=3000.0, title="P_max", json_schema_extra={"unit": "W", "preset": 1000.0}, description="Maximum instantaneous physiological power [W]")
    w_prime_recovery_rate: float = Field(..., ge=0.1, le=2.0, title="W' recovery rate", json_schema_extra={"preset": 1.28}, description="W' recovery rate constant K (Skiba & Clarke 2021, W'BAL-KODE) [-]")
    vitality_loss_rate: float = Field(..., ge=-0.005, le=0.0, title="Vitality loss rate", json_schema_extra={"preset": 0.0}, description="Vitality slope correction factor [-]")

    @model_validator(mode="after")
    def check_p_max_exceeds_cp(self) -> "PhysiologicalSettings":
        """p_max and cp each pass their own independent Field range check
        above, but nothing there stops p_max <= cp -- physiologically
        meaningless for the 3-parameter CP model (P_max is defined as
        exceeding CP), and a combination
        core.simulators.sim_kiritsubo.simulate_power_profile_separated_blocks
        assumes false every step when it divides by (P_max_physio -
        CP_eff), unguarded. Catching it here, once, is the whole reason
        Pydantic validation exists at the config-loading boundary -- the
        simulator kernel itself shouldn't need to defend against a
        physiologically invalid rider."""
        if self.p_max <= self.cp:
            raise ValueError(
                f"p_max ({self.p_max} W) must be greater than "
                f"cp ({self.cp} W) -- core.simulators.sim_kiritsubo divides by "
                "(P_max_physio - CP_eff) every step, and the 3-parameter CP "
                "model is only meaningful when max instantaneous power "
                "exceeds Critical Power."
            )
        return self


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
SIMULATOR_VERSION = "v1.6.1"

# ------------------------
# II. Course physics
# ------------------------
# Speed limits (braking/cornering) and wind geometry -- the physics-
# parameter-dependent stages that live here, not in core.course_geometry,
# because they encode a physics-MODELING choice specific to this
# simulator (brake_usability -- how much of the theoretical
# friction-limited deceleration a rider can actually achieve -- is a
# modeling assumption, not a physical constant), unlike course-SHAPE fitting
# (core.course_geometry.fit_course_geometry_profile), which is pure GPX
# geometry with no such choice to make and stays genuinely shared. See
# this module's own docstring for why living here also matters for
# reproducibility version-tracking.

@njit(cache=True)
def _compute_speed_limits_core(
    kappa: np.ndarray,
    slope: np.ndarray,
    s_p_fine: np.ndarray,
    mu: float,
    gravity_accel: float,
    brake_usability: float,
    cda: float,
    air_density: float,
    crr: float,
    rider_weight: float,
    bike_weight: float,
):
    """
    njit core of _compute_speed_limits_sim_kiritsubo -- see that
    function's docstring for the physics; this is the same computation,
    just with plain scalar args instead of a PhysicalSettings instance
    (numba's nopython mode cannot handle Python object attribute access,
    the same reason this module's own kernel takes a PhysicsParams
    NamedTuple rather than settings objects directly).

    Returns:
        (v_geo_limit, v_limit, fail_index) -- fail_index is -1 on
        success, or the index i where v_curr_sq first went negative (the
        physically-un-brakeable case _compute_speed_limits_sim_kiritsubo
        raises ValueError for). Returned here rather than raised, because
        numba's f-string support cannot format a distance value pulled
        from an array into a readable message (verified: it prints
        "<object type:float64>" instead of the number) -- the non-njit
        wrapper below raises with the real message instead.
    """
    n_fine = len(kappa)

    friction_limit_sq = mu * gravity_accel * np.cos(slope)
    max_speed_val = 40.0  # 144 km/h
    v_geo_limit = np.full_like(kappa, max_speed_val)
    mask = friction_limit_sq < (max_speed_val**2) * kappa
    v_geo_limit[mask] = np.sqrt(friction_limit_sq[mask] / (kappa[mask]))

    v_limit = np.zeros(n_fine)
    v_limit[-1] = v_geo_limit[-1]
    total_mass = bike_weight + rider_weight
    beta = -0.5 * cda * air_density / total_mass

    ds = np.diff(s_p_fine)

    for i in range(n_fine - 2, -1, -1):
        f_brake_max = -brake_usability * mu * total_mass * gravity_accel * np.cos(slope[i])
        f_grav = -total_mass * gravity_accel * np.sin(slope[i])
        f_roll = -crr * total_mass * gravity_accel * np.cos(slope[i])

        alpha = (f_brake_max + f_grav + f_roll) / total_mass

        delta_s = ds[i]
        exp_term = np.exp(-2 * beta * delta_s)

        v_next_sq = v_limit[i+1]**2
        v_curr_sq = v_next_sq * exp_term + (alpha / beta) * (exp_term - 1)
        if v_curr_sq < 0:
            return v_geo_limit, v_limit, i
        v_limit[i] = min(v_geo_limit[i], np.sqrt(v_curr_sq))

    return v_geo_limit, v_limit, -1


def _compute_speed_limits_sim_kiritsubo(
    kappa: np.ndarray,
    slope: np.ndarray,
    s_p_fine: np.ndarray,
    physical: PhysicalSettings,
) -> np.ndarray:
    """
    Compute the full braking-aware speed limit (v_limit) via a backward
    pass over the course.

    Depends only on precomputed course geometry (kappa, slope, s_p_fine)
    plus this simulator's own physical parameters (mu, brake_usability,
    CdA, air_density, Crr, masses) -- not on the raw course points or the
    B-spline fit. This is the cheap stage: safe to re-run standalone
    whenever any of those change (see recompute_course_physics_sim_kiritsubo),
    without repeating the expensive geometry fit in
    core.course_geometry.fit_course_geometry_profile. Thin wrapper around
    the @njit-compiled _compute_speed_limits_core -- unpacks physical into
    plain scalars (see that function's docstring for why) and turns its
    fail_index sentinel back into a ValueError, with the actual distance
    value formatted in plain Python (not inside njit code).

    Returns:
        v_limit [m/s], same length as kappa.

    Raises:
        ValueError: If the required deceleration into some point exceeds
                    what brake_usability/mu/masses/CdA can physically
                    provide (v_curr_sq went negative in the backward
                    pass).
    """
    _, v_limit, fail_index = _compute_speed_limits_core(
        kappa, slope, s_p_fine,
        physical.mu, physical.gravity_accel, physical.brake_usability, physical.cda,
        physical.air_density, physical.crr, physical.rider_weight, physical.bike_weight,
    )
    if fail_index >= 0:
        raise ValueError(
            f"Braking physics failure at distance {s_p_fine[fail_index]}m: v_sq is negative."
        )
    return v_limit


def recompute_course_physics_sim_kiritsubo(
    course_profile: CourseProfile, physical: PhysicalSettings,
) -> CourseProfile:
    """
    Given an already-fitted CourseProfile (kappa/slope/s_p_fine/heading
    from core.course_geometry.fit_course_geometry_profile) and a
    (possibly overridden) PhysicalSettings, return a new CourseProfile
    with v_limit/cos_phi/sin_phi recomputed to match -- without repeating
    the expensive B-spline geometry fit.

    This is the SimulatorSpec.recompute_course_physics implementation
    core.physics_overrides.build_overridden_params (Analyzer's
    manual Rebuild, core.calibrator's Auto Fit) and
    eidos.apps.exporter's FIT/ZWO re-simulation call whenever physical
    settings differ from a strategy's own baseline.
    """
    v_limit = _compute_speed_limits_sim_kiritsubo(
        course_profile.kappa, course_profile.slope, course_profile.s_p_fine, physical,
    )
    cos_phi, sin_phi = _compute_wind_geometry(course_profile.heading, physical.wind_direction)
    return dataclasses.replace(course_profile, v_limit=v_limit, cos_phi=cos_phi, sin_phi=sin_phi)


def compute_course_physics_sim_kiritsubo(
    points: CoursePoints, physical: PhysicalSettings, run: RunSettings,
) -> CourseProfile:
    """
    Full course-physics pipeline for a fresh course: fit course-shape
    geometry (core.course_geometry.fit_course_geometry_profile -- shared,
    simulator-agnostic, returns curvature RAW), clamp curvature to this
    simulator's own physical.min_corner_radius (see PhysicalSettings' own
    comment on that field for why this clamp lives here, not in
    core.course_geometry), then compute this simulator's own speed-limit/
    wind-geometry physics via recompute_course_physics_sim_kiritsubo.

    This is the SimulatorSpec.compute_course_physics implementation
    eidos.apps.generator (strategy generation) and
    hyle.apps.fit2gpx_converter call.
    """
    geom = fit_course_geometry_profile(points, run)
    n_fine = len(geom["s_p_fine"])
    placeholder = np.zeros(n_fine)
    kappa = np.minimum(geom["kappa_raw"], 1.0 / physical.min_corner_radius)
    course = CourseProfile(
        distance=geom["s_p_fine"][-1],
        distance_step=run.distance_step,
        s_p_fine=geom["s_p_fine"],
        s_h_fine=geom["s_h_fine"],
        lat_fine=geom["lat_fine"],
        lon_fine=geom["lon_fine"],
        slope=geom["slope"],
        kappa=kappa,
        v_limit=placeholder,
        altitude=geom["z_fine"],
        heading=geom["heading"],
        cos_phi=placeholder,
        sin_phi=placeholder,
    )
    return recompute_course_physics_sim_kiritsubo(course, physical)


# ------------------------
# III. Core simulation (Numba)
# ------------------------

@njit
def calc_F_prop(P, v, F_max):
    """
    Compute propulsive force from target power, velocity, and maximum pedal force.

    Returns P/v when within the force ceiling, otherwise clamps to F_max.
    This formulation avoids division-by-zero at v=0 by design: when v=0,
    P < F_max * 0 = 0 is never true for P >= 0, so F_prop = F_max is returned.
    """
    if P < F_max * v:
        F_prop = P / v
    else:
        F_prop = F_max
    return F_prop

@njit
def calc_vitality_from_TSS(TSS, vitality_slope):
    """
    Compute vitality (fatigue correction factor) from cumulative TSS.
    Returns a value in [0.0, 1.0].
    """
    vitality = 1.0 + vitality_slope * TSS
    vitality = max(0.0, min(vitality, 1.0))
    return vitality

@njit
def calc_CP_eff(NP, Time, CP, V_slope):
    """
    Compute effective Critical Power after vitality correction.
    TSS is derived from Normalized Power (NP) relative to CP.
    """
    TSS = (Time / 3600) * ((NP / CP) ** 2) * 100
    vitality = calc_vitality_from_TSS(TSS, V_slope)
    CP_eff = vitality * CP
    return CP_eff

@njit
def calc_apparent_wind(v, v_wind, c_phi, s_phi):
    """
    Decompose apparent wind into axial and cross components.

    Returns apparent wind speed (v_w_ap), axial component (v_w_ax),
    and cross component (v_w_cr).
    """
    v_w_ax = v - v_wind * c_phi   # axial wind velocity [m/s]
    v_w_cr = v_wind * s_phi        # cross wind velocity [m/s]
    v_w_ap_sq = v_w_ax**2 + v_w_cr**2
    v_w_ap = np.sqrt(v_w_ap_sq)   # apparent wind speed [m/s]
    return v_w_ap, v_w_ax, v_w_cr

@njit
def get_interpolated_value(x, data_array, stride):
    """Linear interpolation of a uniformly-sampled array at position x [m]."""
    idx_float = x / stride
    idx_lower = int(idx_float)
    idx_upper = idx_lower + 1
    if idx_upper >= len(data_array):
        return data_array[-1]
    alpha = idx_float - idx_lower
    return (1 - alpha) * data_array[idx_lower] + alpha * data_array[idx_upper]

@njit
def sync_hook(P_block, t, v, x, W):
    """
    Default no-op synchronization hook. Exists so that
    simulate_power_profile_separated_blocks compiles under Numba regardless
    of use_sync_hook's runtime value (Numba compiles all branches ahead of
    time). Overridden by eidos.apps.trainer to exchange real-time power
    data with the ANT+ receiver when use_sync_hook=True.
    """
    return 0

@njit
def simulate_power_profile_separated_blocks(
    v_init: float, power_blocks: PowerBlocks, params: PhysicsParams, return_trajectory: bool, use_sync_hook: bool, is_target_power: bool):
    """
    Physics engine: compute finish time for a given power strategy and physical constants.

    Two independent boolean flags govern the simulation mode:

    use_sync_hook and is_target_power are orthogonal. use_sync_hook controls
    whether sync_hook is called each step to source P_input and to report
    live state (a side-effect axis); is_target_power controls whether the
    physics is driven by the physiologically-clamped target power (P_exerting,
    with F_prop derived from it) or by the raw recorded power P_input
    consumed without clamping, including past the point where W' would go
    negative (a computation axis). All four combinations are meaningful:

    - use_sync_hook=False, is_target_power=True: normal strategy
      optimization (DE/NM). sync_hook is never called; P_input falls back to
      P_block. P_exerting is clamped to the physiologically available power.
    - use_sync_hook=True, is_target_power=True: drive the physics from the
      clamped target power while reporting live state to sync_hook every
      step (e.g. a ghost-rider readout during a live session). sync_hook's
      return value still feeds into the same P_exerting = min(P_input,
      P_phys_lim) clamp as the use_sync_hook=False case above -- not
      discarded. No current caller uses this combination (eidos.apps.
      trainer only uses (False,True) and (True,False)).
    - use_sync_hook=True, is_target_power=False: live training session.
      sync_hook exchanges state with the ANT+ receiver and returns the
      power actually being ridden; that value drives the physics directly,
      unclamped.
    - use_sync_hook=False, is_target_power=False: recorded-power replay
      (e.g. eidos.apps.analyzer). power_blocks already holds the measured
      power series; it drives the physics directly, unclamped. This is the
      "riding through a dragging brake" case: F_prop can still be capped
      by a_clip (cornering/braking constraints), but P_exerted (what the W'
      balance is charged for) always reflects the full recorded power --
      the shortfall is dissipated as heat/sound at the brake, not spared
      from the legs.

    Parameters
    ----------
    v_init : float
        Starting speed [m/s] at t=0, x=0.
    power_blocks : PowerBlocks
        Target power and segment lengths defining the pacing strategy
        (is_target_power=True), or a zero-order-hold reconstruction of
        recorded power and segment lengths (is_target_power=False).
    params : PhysicsParams
        Immutable physical and geometric constants.
    return_trajectory : bool
        If True, record full time-series trajectories. If False, skip array
        allocation entirely (used by the DE optimizer for speed).
    use_sync_hook : bool
        If True, call sync_hook every step to source P_input and report live
        state (used during live training sessions). See mode table above.
    is_target_power : bool
        If True, drive the physics from the physiologically-clamped target
        power. If False, drive it from the raw recorded power P_input,
        unclamped. See mode table above.
    """
    # --- Unpack PhysicsParams into local variables for Numba ---
    # Course geometry
    slope        = params.slope                  # 1
    v_limit      = params.v_limit                # 2
    # Simulation control
    dt           = params.time_step              # 3
    stride       = params.distance_step          # 4
    # Rider physical characteristics
    CP           = params.cp                     # 5
    W_prime      = params.w_prime                # 6
    P_max_physio = params.p_max                  # 7
    m            = params.total_weight           # 8
    CdA          = params.cda                    # 9
    F_max        = params.f_max                  # 10
    V_slope      = params.vitality_loss_rate     # 11
    K            = params.w_prime_recovery_rate  # 12
    tau_braking  = params.brake_lookahead        # 13
    # Environmental physical constants
    g            = params.gravity_accel          # 14
    rho          = params.air_density            # 15
    Crr          = params.crr                    # 16
    # Wind
    v_wind       = params.wind_speed             # 17
    cda_ratios   = params.cda_ratios             # 18
    cos_phi_arr  = params.cos_phi                # 19
    sin_phi_arr  = params.sin_phi                # 20


    N = len(power_blocks.length)
    x_end = 0.0
    t, x = 0.0, 0.0
    v = v_init
    W = W_prime
    P_gen_30 = 0.0
    P4_integral = 0.0
    NP = 0.0
    CP_eff = calc_CP_eff(NP, t, CP, V_slope)
    c_phi = get_interpolated_value(x, cos_phi_arr, stride)
    s_phi = get_interpolated_value(x, sin_phi_arr, stride)
    v_w_ap, v_w_ax, v_w_cr = calc_apparent_wind(v, v_wind, c_phi, s_phi)
    psi_w_ap = np.degrees(np.arctan2(v_w_cr, v_w_ax))
    penalty_factor = 1.0
    if return_trajectory:
        t_traj = [t]
        x_traj = [x]
        v_traj = [v]
        W_traj = [W]
        P_traj = [np.nan] # Placeholder to be replaced later
        CP_eff_traj = [CP_eff]
        v_w_app_traj = [v_w_ap]
        psi_w_app_traj = [psi_w_ap]

    for block_i in range(N):
        P_block = power_blocks.power[block_i]
        seg_length = power_blocks.length[block_i]
        x_seg_end = x_end + seg_length

        # Segment simulation loop
        while x <= x_seg_end:
            # sync_hook is a no-op by default (see its docstring); overridden in
            # eidos.apps.trainer to exchange current state with the ANT+ receiver
            # and receive the power actually being ridden
            if use_sync_hook:
                P_input = sync_hook(P_block, t, v, x, W)
            else:
                P_input = P_block

            # --- P_exerting = P_input, clamped to the physiologically available power P_phys_lim (3-parameter model) when is_target_power ---
            if is_target_power:
                # K_shift = W / (P_max_current - CP_eff), where
                # P_max_current = ((P_max_physio-CP_eff)/W_prime)*W + CP_eff
                # -- W cancels, reducing to the constant below. Denominator > 0
                # guaranteed by PhysiologicalSettings.check_p_max_exceeds_cp.
                K_shift = W_prime / (P_max_physio - CP_eff)
                P_phys_lim = W / (dt + K_shift) + CP_eff
                P_exerting = min(P_input, P_phys_lim)
                if P_phys_lim < P_input:  # accumulate penalty when the physiological ceiling binds below P_input
                    penalty_factor = penalty_factor + ((P_input - P_phys_lim) / P_input)**2
            else:
                P_exerting = P_input

            # --- Environmental resistance forces ---
            theta = get_interpolated_value(x, slope, stride)
            F_rolling = Crr * m * g * np.cos(theta)   # rolling resistance
            F_gravity = m * g * np.sin(theta)          # gravitational resistance
            yaw_idx = int(abs(psi_w_ap))               # aerodynamic drag
            if yaw_idx > 180:
                yaw_idx = 180
            cda_eff = CdA * cda_ratios[yaw_idx]
            F_aero = 0.5 * rho * cda_eff * v_w_ap * v_w_ax
            F_resist_env = F_rolling + F_gravity + F_aero

            # --- Acceleration / braking decision ---
            a_clip = F_max / m  # upper bound of volitional (propulsion - brake) acceleration
            a_free = -F_resist_env / m  # free deceleration with zero propulsion and braking
            brake_scan_intaval = stride * 2  # scan interval, 2x distance_step
            SCAN_STEPS = int(v * tau_braking / brake_scan_intaval)  # steps within lookahead window
            if SCAN_STEPS == 0:
                pass
            else:
                d_tau = tau_braking / SCAN_STEPS
                for i in range(1, SCAN_STEPS + 1):  # scan ahead up to tau_braking
                    tau_i = i * d_tau
                    x_tau_i = x + v * tau_i + 0.5 * a_free * tau_i**2
                    v_tau_i = max(v + a_free * tau_i, 0.0)
                    v_cap_tau_i = get_interpolated_value(x_tau_i, v_limit, stride)
                    if v_tau_i > v_cap_tau_i:
                        a_clip_tau_i = (v_cap_tau_i - v_tau_i) / tau_i
                        if a_clip_tau_i < a_clip:
                            a_clip = a_clip_tau_i

            # --- Propulsive and total resistance forces ---
            if a_clip < 0.0:  # braking required
                F_brake = -m * a_clip
                F_prop = 0.0
            else:              # no braking needed
                F_brake = 0.0
                if is_target_power:
                    F_prop = calc_F_prop(P_exerting, v, F_max)
                else:
                    F_prop = calc_F_prop(P_input, v, F_max)
                if F_prop > m * a_clip:
                    F_prop = m * a_clip
            F_resist_total = F_resist_env + F_brake

            # --- State update ---
            final_a = (F_prop - F_resist_total) / m
            v_new = v + final_a * dt
            x_new = x + v * dt + 0.5 * final_a * dt**2
            if x_new <= x:  # if unable to move forward, walk at v_min (0.1 m/s = 0.36 km/h)
                v_min = 0.1
                v_new = v_min
                x_new = x + v_min * dt
            # propulsive power
            if is_target_power:
                P_exerted = F_prop * v
            else:
                P_exerted = P_input

            # --- W' balance update (Skiba & Clarke 2021 W'BAL-KODE recovery, Eq. 16) ---
            if P_exerted > CP_eff:  # depletion
                W_new = W - (P_exerted - CP_eff) * dt
            else:                 # recovery
                W_new = W_prime - (W_prime - W) * np.exp(-K * ((CP_eff - P_exerted) / W_prime) * dt)

            if return_trajectory:
                P_traj[-1] = P_exerted  # Replace the temporary placeholder

            # --- Advance state variables ---
            t_prev = t
            t = t + dt
            x_prev = x
            x = x_new
            v = v_new
            W = W_new

            P_gen_30  = ((30.0 - dt) * P_gen_30 + dt * P_exerted) / 30.0  # 30-second rolling average of P_exerted
            P4_integral = P4_integral + (P_gen_30**4) * dt
            NP = (P4_integral / t)**(1 / 4)  # recompute NP as 4th-root mean (t > 0 always holds)

            CP_eff = calc_CP_eff(NP, t, CP, V_slope)
            c_phi = get_interpolated_value(x, cos_phi_arr, stride)
            s_phi = get_interpolated_value(x, sin_phi_arr, stride)
            v_w_ap, v_w_ax, v_w_cr = calc_apparent_wind(v, v_wind, c_phi, s_phi)
            psi_w_ap = np.degrees(np.arctan2(v_w_cr, v_w_ax))

            if return_trajectory:
                t_traj.append(t)
                x_traj.append(x)
                v_traj.append(v)
                P_traj.append(np.nan) # Temporary placeholder to be replaced later
                W_traj.append(W)
                CP_eff_traj.append(CP_eff)
                v_w_app_traj.append(v_w_ap)
                psi_w_app_traj.append(psi_w_ap)

        x_end = x_seg_end

    # --- Segment-end correction: interpolate to exact x_seg_end ---
    # t, x, v, W, CP_eff, v_w_app, psi are continuous state variables over a
    # half-open interval [t(j), t(j)+dt)), so linear interpolation to
    # x_seg_end is a first-order approximation -- exact only when final_a is
    # 0 over that interval (x_new = x + v*dt + 0.5*final_a*dt**2 is
    # quadratic in dt otherwise). P is a step signal (constant over each dt
    # interval, like PowerBlocks itself), not a boundary-sampled value --
    # there is no "next interval" at the course end, so P_traj[-1] is
    # simply carried forward from the last interval actually used
    # (P_traj[-2]), not interpolated.
    ratio = (x_seg_end - x_prev) / (x - x_prev)  # x_prev <= x_seg_end < x
    f_time = t_prev + (t - t_prev) * ratio
    if return_trajectory:
        t_traj[-1] = t_traj[-2] + (t_traj[-1] - t_traj[-2]) * ratio
        x_traj[-1] = x_seg_end
        v_traj[-1] = v_traj[-2] + (v_traj[-1] - v_traj[-2]) * ratio
        P_traj[-1] = P_traj[-2] # Replace the temporary placeholder
        W_traj[-1] = W_traj[-2] + (W_traj[-1] - W_traj[-2]) * ratio
        CP_eff_traj[-1] = CP_eff_traj[-2] + (CP_eff_traj[-1] - CP_eff_traj[-2]) * ratio
        v_w_app_traj[-1] = v_w_app_traj[-2] + (v_w_app_traj[-1] - v_w_app_traj[-2]) * ratio
        # Interpolate yaw angle with phase-wrapping correction
        diff_psi = psi_w_app_traj[-1] - psi_w_app_traj[-2]
        if diff_psi > 180.0:   # normalize difference to [-180, 180]
            diff_psi -= 360.0
        elif diff_psi < -180.0:
            diff_psi += 360.0
        psi_interp = psi_w_app_traj[-2] + diff_psi * ratio
        if psi_interp > 180.0:   # clamp result back to [-180, 180]
            psi_interp -= 360.0
        elif psi_interp < -180.0:
            psi_interp += 360.0
        psi_w_app_traj[-1] = psi_interp

        if use_sync_hook:
            sync_hook(P_block, t_traj[-1], v_traj[-1], x_traj[-1], W_traj[-1])

        output = SimulationOutput(
            finish_time=f_time,
            penalty_factor=penalty_factor,
            t_traj=np.array(t_traj),
            x_traj=np.array(x_traj),
            v_traj=np.array(v_traj),
            p_traj=np.array(P_traj),
            w_traj=np.array(W_traj),
            cp_eff_traj=np.array(CP_eff_traj),
            v_w_app_traj=np.array(v_w_app_traj),
            psi_w_app_traj=np.array(psi_w_app_traj)
        )
    else:
        # DE mode: skip all array allocation for maximum speed
        empty_arr = np.empty(0)
        output = SimulationOutput(
            finish_time=f_time,
            penalty_factor=penalty_factor,
            t_traj=empty_arr,
            x_traj=empty_arr,
            v_traj=empty_arr,
            p_traj=empty_arr,
            w_traj=empty_arr,
            cp_eff_traj=empty_arr,
            v_w_app_traj=empty_arr,
            psi_w_app_traj=empty_arr
        )

    return output

# --------------------------------------------------
# IV. PhysicsParams builder
# --------------------------------------------------

def build_physics_params_sim_kiritsubo(
    physical: PhysicalSettings,
    physiological: PhysiologicalSettings,
    run: RunSettings,
    course: CourseProfile,
) -> PhysicsParams:
    """Assemble PhysicsParams for the sim_kiritsubo kernel (CP/W'/P_max_physio three-parameter model).

    Loads physical.cda_yaw_table_filename itself, not an externally-
    supplied cda_ratios argument (see core.simulators' module docstring)
    -- this simulator's own kernel is the only thing that needs the
    resulting yaw-multiplier table."""
    cda_ratios = load_cda_yaw_table(physical.cda_yaw_table_filename)
    return PhysicsParams(
        slope=course.slope,                                        # 1
        v_limit=course.v_limit,                                    # 2
        time_step=run.time_step,                                   # 3
        distance_step=course.distance_step,                        # 4
        cp=physiological.cp,                                       # 5
        w_prime=physiological.w_prime,                             # 6
        p_max=physiological.p_max,                                 # 7
        total_weight=physical.rider_weight + physical.bike_weight, # 8
        cda=physical.cda,                                          # 9
        f_max=physical.f_max,                                      # 10
        vitality_loss_rate=physiological.vitality_loss_rate,       # 11
        w_prime_recovery_rate=physiological.w_prime_recovery_rate, # 12
        brake_lookahead=physical.brake_lookahead,                  # 13
        gravity_accel=physical.gravity_accel,                      # 14
        air_density=physical.air_density,                          # 15
        crr=physical.crr,                                          # 16
        wind_speed=physical.wind_speed,                            # 17
        cda_ratios=cda_ratios,                                     # 18
        cos_phi=course.cos_phi,                                    # 19
        sin_phi=course.sin_phi,                                    # 20
    )
