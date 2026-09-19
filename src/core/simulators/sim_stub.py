####################
# core/simulators/sim_stub.py
####################
"""
sim_stub simulator: a deliberately simplified example simulator, meant to
demonstrate how a SIMULATOR_REGISTRY entry is built -- NOT a validated
physics model, do not use for real training-strategy research.

Exists to prove core.simulators.SIMULATOR_REGISTRY genuinely supports a
second, independently-shaped entry: a different @njit kernel AND a
different parameter NamedTuple (DummyPhysicsParams below carries only
slope/time_step/distance_step/total_weight/cda/crr/gravity_accel/
air_density/cp/w_prime -- no braking-dynamics parameters, no wind
parameters, no yaw-dependent CdA table). If this only proved "two
kernels sharing one fixed PhysicsParams", it would leave the handover
doc's design point #2 (core.simulators.__init__'s own docstring: "a
future kernel modeling something other than the CP/W'/P_max_physio
three-parameter model would need its own PhysicsParams-equivalent shape
and builder") unverified.

Physics, deliberately simplified:
  - Rolling resistance + gravity + quadratic aero drag only.
  - No wind (apparent wind == ground speed), no yaw-dependent CdA.
  - The naive two-parameter W' balance model: W depletes linearly
    whenever power exceeds cp (dW = -(P - cp) * dt), and never recovers
    below cp. The physiologically available power for a step is cp +
    W/dt (always at least cp, however depleted W already is) -- the
    genuinely 2-parameter (CP, W') model this kernel commits to; see
    PhysiologicalSettings below, which accordingly carries no field
    beyond the cp/w_prime every SIMULATOR_REGISTRY entry already
    requires.
  - No cornering/braking speed limit (course.v_limit is ignored -- see
    compute_course_physics_sim_stub, which never computes a real one).

use_sync_hook / is_target_power are accepted, for call-signature
compatibility with core.simulators.SimulatorSpec.kernel's shared
six-argument convention. is_target_power is genuinely meaningful here:
it gates the cp + W/dt physiological clamp. use_sync_hook is not
meaningfully supported -- no live sync integration (it's a no-op via
the sync_hook stub below) -- it still reports the live W balance via
its own W argument on every call, though, for eidos.apps.trainer's live
HUD, the only consumer of that argument. v_init is fully supported: it
sets the starting value of v.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numba import njit
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.course_geometry import fit_course_geometry_profile
from core.data_manager import load_cda_yaw_table
from core.io_config import BASE_CDA_YAW_TABLES_DIR
from core.schema import (
    CoursePoints,
    CourseProfile,
    PhysiologicalSettingsBase,
    PowerBlocks,
    RunSettings,
    SimulationOutput,
    validate_flat_filename,
)

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
SIMULATOR_VERSION = "dummy-v9"


# --------------------------------------------------
# 0. PhysicalSettings / PhysiologicalSettings
# --------------------------------------------------
# PhysicalSettings holds just what this kernel's own physics reads
# (m = rider_weight+bike_weight, CdA=cda, Crr=crr, g=gravity_accel, rho=
# air_density) PLUS cda_yaw_table_filename -- no mu/brake_usability/
# wind_speed/wind_direction/f_max/brake_lookahead, since speed-limit/
# wind-geometry physics is each simulator's own responsibility (see
# compute_course_physics_sim_stub below) -- this kernel's
# PhysicalSettings only needs to carry what IT needs.
#
# cda_yaw_table_filename stays required, even though this kernel has no
# wind/yaw model: CdA and its yaw-multiplier table are always a PAIR, not
# two independently-optional concerns -- a simulator that models aero
# drag at all (this one does: F_aero below) always resolves it through
# the cda_ratios[yaw_idx] mechanism, even if that resolution is trivial
# here (yaw is always 0 -- no wind, see module docstring -- so only
# cda_ratios[0] is ever read). "constant_model.csv" (this field's own
# preset) is the flat, factor=1.0-everywhere table that makes this a
# no-op multiplication, the honest way to represent "this kernel doesn't
# model yaw-dependent drag" without inventing a second, field-shaped way
# to say the same thing. A simulator with NO aero drag term at all
# (unlike this one) would be the one genuinely entitled to drop
# cda/cda_yaw_table_filename together -- see eidos.apps.viewer's own CdA
# polar plot, which skips silently (not an error) for exactly that
# hypothetical case.
#
# PhysiologicalSettings adds NO field at all beyond the cp/w_prime every
# SIMULATOR_REGISTRY entry is required to have (see core.schema.
# PhysiologicalSettingsBase) -- this kernel's own naive W' balance model
# (see module docstring) needs only cp and w_prime, the genuine
# 2-parameter model this kernel commits to. Concrete proof that
# SimulatorSpec.physiological_param_model can validly differ in shape
# between registry entries, the same role DummyPhysicsParams (below)
# plays for kernel-parameter-shape independence.
class PhysicalSettings(BaseModel):
    """Physical (mass/aero/rolling-resistance) parameters -- exactly what
    this kernel's own physics reads, no more (except cda_yaw_table_filename,
    which stays required alongside cda -- see this module's own section-0
    comment for why the two are never independently optional)."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    rider_weight: float = Field(..., ge=30.0, le=120.0, title="Rider weight", json_schema_extra={"unit": "kg", "preset": 70.0}, description="Rider mass [kg]")
    cda: float = Field(..., ge=0.10, le=0.50, title="CdA", json_schema_extra={"unit": "m²", "preset": 0.25}, description="Aerodynamic drag area [m^2]")
    bike_weight: float = Field(..., ge=5.0, le=20.0, title="Bike weight", json_schema_extra={"unit": "kg", "preset": 8.0}, description="Bike mass [kg]")
    gravity_accel: float = Field(..., ge=9.76, le=9.84, title="Gravity", json_schema_extra={"unit": "m/s²", "preset": 9.80665}, description="Local gravity [m/s^2]")
    air_density: float = Field(..., ge=0.8, le=1.35, title="Air density", json_schema_extra={"unit": "kg/m³", "preset": 1.225}, description="Air density [kg/m^3]")
    crr: float = Field(..., ge=0.001, le=0.02, title="Crr", json_schema_extra={"preset": 0.004}, description="Rolling resistance coefficient [-]")
    cda_yaw_table_filename: str = Field(...,
        title="CdA yaw table filename",
        json_schema_extra={"preset": "constant_model.csv"},
        description=r"Multiplier $f_{table}(\psi)$ where $CdA_{total} = CdA_{rider} \times f_{table}(\psi)$. This kernel always evaluates it at yaw=0 (no wind model -- see module docstring), so only a flat/constant table is ever meaningful here."
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
            if not (np.isclose(yaws[0], 0.0) and np.isclose(yaws[-1], 180.0)):
                raise ValueError(f"Yaw range must be [0.0, 180.0]. Found: [{yaws[0]}, {yaws[-1]}]")
            if not np.all(np.diff(yaws) > 0):
                raise ValueError("1st column (yaw) must be strictly increasing.")
            if np.any((factors < 0.0) | (factors > 10.0)):
                raise ValueError("2nd column (factors) must be within [0.0, 10.0].")
        except Exception as e:
            raise ValueError(f"Content validation failed for '{v}': {str(e)}")
        return v


class PhysiologicalSettings(PhysiologicalSettingsBase):
    """cp/w_prime are inherited from PhysiologicalSettingsBase (required on
    every SIMULATOR_REGISTRY entry -- see that class's own docstring for
    why) and are this kernel's only physiological inputs -- both genuinely
    read by its own naive W' balance model (see module docstring). No
    further field: this kernel's model is the 2-parameter (CP, W') one,
    with no Pmax/recovery-rate/vitality-loss-rate concept to carry a field
    for."""
    model_config = ConfigDict(extra="forbid", frozen=True)


class DummyPhysicsParams(NamedTuple):
    slope: np.ndarray    # road gradient [rad]
    time_step: float     # time step [s]
    distance_step: float # course discretization step [m]
    total_weight: float  # total mass (rider + bike) [kg]
    cda: float           # aerodynamic drag area [m^2]
    crr: float           # rolling resistance coefficient [-]
    gravity_accel: float # local gravitational acceleration [m/s^2]
    air_density: float   # air density [kg/m^3]
    cp: float            # Critical Power [W] -- physiological clamp floor (see module docstring)
    w_prime: float       # anaerobic work capacity [J] -- initial W' balance, depleted by this kernel's own naive model (see module docstring)


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
    """No-op placeholder -- sim_stub has no live-training integration."""
    return 0


@njit
def simulate_dummy_constant_power(
    v_init: float, power_blocks: PowerBlocks, params: DummyPhysicsParams,
    return_trajectory: bool, use_sync_hook: bool, is_target_power: bool,
):
    slope = params.slope
    dt = params.time_step
    stride = params.distance_step
    m = params.total_weight
    CdA = params.cda
    Crr = params.crr
    g = params.gravity_accel
    rho = params.air_density
    CP = params.cp
    W_prime = params.w_prime

    v_launch_floor = 0.5  # m/s -- avoids the P/v singularity at a standing start

    N = len(power_blocks.length)
    x_end = 0.0
    t, x = 0.0, 0.0
    v = v_init
    W = W_prime
    penalty_factor = 1.0

    if return_trajectory:
        t_traj = [t]
        x_traj = [x]
        v_traj = [v]
        p_traj = [np.nan]
        W_traj = [W]

    for block_i in range(N):
        P_block = power_blocks.power[block_i]
        seg_length = power_blocks.length[block_i]
        x_seg_end = x_end + seg_length

        while x <= x_seg_end:
            if use_sync_hook:
                P_input = sync_hook(P_block, t, v, x, W)
            else:
                P_input = P_block

            if is_target_power:
                # Physiologically available power P_phys_lim for this step,
                # always at least CP however depleted W already is (see
                # module docstring for the naive depletion-only W' balance
                # model this implements). P_exerting clamps P_input to it.
                P_phys_lim = CP + W / dt
                P_exerting = min(P_input, P_phys_lim)
                if P_phys_lim < P_input:  # accumulate penalty when the physiological ceiling binds below P_input
                    penalty_factor = penalty_factor + ((P_input - P_phys_lim) / P_input) ** 2
            else:
                P_exerting = P_input

            theta = get_interpolated_value(x, slope, stride)
            F_rolling = Crr * m * g * np.cos(theta)
            F_gravity = m * g * np.sin(theta)
            F_aero = 0.5 * rho * CdA * v * abs(v)
            F_resist = F_rolling + F_gravity + F_aero

            v_for_prop = v if v > v_launch_floor else v_launch_floor
            F_prop = P_exerting / v_for_prop

            final_a = (F_prop - F_resist) / m
            v_new = v + final_a * dt
            x_new = x + v * dt + 0.5 * final_a * dt**2
            if x_new <= x:
                v_new = 0.1
                x_new = x + 0.1 * dt

            if is_target_power:
                P_exerted = F_prop * v
            else:
                P_exerted = P_input

            if return_trajectory:
                p_traj[-1] = P_exerted

            # W' balance update -- naive model: linear depletion above CP,
            # no recovery below it (see module docstring).
            if P_exerted > CP:
                W_new = W - (P_exerted - CP) * dt
            else:
                W_new = W

            t_prev = t
            t = t + dt
            x_prev = x
            x = x_new
            v = v_new
            W = W_new

            if return_trajectory:
                t_traj.append(t)
                x_traj.append(x)
                v_traj.append(v)
                p_traj.append(np.nan)
                W_traj.append(W)

        x_end = x_seg_end

    ratio = (x_seg_end - x_prev) / (x - x_prev)
    f_time = t_prev + (t - t_prev) * ratio
    if return_trajectory:
        t_traj[-1] = t_traj[-2] + (t_traj[-1] - t_traj[-2]) * ratio
        x_traj[-1] = x_seg_end
        v_traj[-1] = v_traj[-2] + (v_traj[-1] - v_traj[-2]) * ratio
        p_traj[-1] = p_traj[-2]
        W_traj[-1] = W_traj[-2] + (W_traj[-1] - W_traj[-2]) * ratio

        n_pts = len(t_traj)
        v_arr = np.array(v_traj)
        # No wind model in this stub -- v_w_app_traj/psi_w_app_traj exist
        # only for SimulationOutput shape compatibility with every consumer
        # that reads them (viewer, exporter, trainer, analyzer): apparent
        # wind equals ground speed since there is no wind; yaw is always 0.
        # cp_eff_traj stays constant (no vitality model to shift CP); w_traj
        # is the genuine depleting-W' trajectory from the loop above.
        output = SimulationOutput(
            finish_time=f_time,
            penalty_factor=penalty_factor,
            t_traj=np.array(t_traj),
            x_traj=np.array(x_traj),
            v_traj=v_arr,
            p_traj=np.array(p_traj),
            w_traj=np.array(W_traj),
            cp_eff_traj=np.full(n_pts, CP),
            v_w_app_traj=v_arr,
            psi_w_app_traj=np.zeros(n_pts),
        )
    else:
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
            psi_w_app_traj=empty_arr,
        )

    return output


# --------------------------------------------------
# Course physics
# --------------------------------------------------
# This kernel has no braking/cornering speed-limit model and no wind
# model (see module docstring) -- both v_limit and wind geometry are
# degenerate, fixed placeholders, not computed from any PhysicalSettings
# field, which is exactly why this simulator's own PhysicalSettings
# doesn't carry mu/brake_usability/wind_speed/wind_direction at all (see
# this module's section-0 comment).
_UNCONSTRAINED_V_LIMIT_MPS = 40.0  # 144 km/h -- this kernel never reads course.v_limit at all, so the exact value is inert, kept finite/plausible only so it plots sensibly if ever displayed.


def recompute_course_physics_sim_stub(
    course_profile: CourseProfile, physical: PhysicalSettings,
) -> CourseProfile:
    """SimulatorSpec.recompute_course_physics for sim_stub: a no-op.
    v_limit/cos_phi/sin_phi never depend on any PhysicalSettings field
    here (see module docstring), so there is nothing to recompute even
    when physical differs from the strategy's own baseline."""
    return course_profile


def compute_course_physics_sim_stub(
    points: CoursePoints, physical: PhysicalSettings, run: RunSettings,
) -> CourseProfile:
    """SimulatorSpec.compute_course_physics for sim_stub: fit course-shape
    geometry (core.course_geometry.fit_course_geometry_profile -- shared,
    simulator-agnostic, returns curvature RAW) and fill v_limit/cos_phi/
    sin_phi with fixed degenerate values (no braking-limit or wind model
    -- see module docstring) instead of computing them from physical at
    all. Curvature itself is used unclamped: this kernel has no
    cornering-limit model to need a min_corner_radius-style clamp for,
    and its own PhysicalSettings doesn't carry the field at all."""
    geom = fit_course_geometry_profile(points, run)
    n_fine = len(geom["s_p_fine"])
    return CourseProfile(
        distance=geom["s_p_fine"][-1],
        distance_step=run.distance_step,
        s_p_fine=geom["s_p_fine"],
        s_h_fine=geom["s_h_fine"],
        lat_fine=geom["lat_fine"],
        lon_fine=geom["lon_fine"],
        slope=geom["slope"],
        kappa=geom["kappa_raw"],
        v_limit=np.full(n_fine, _UNCONSTRAINED_V_LIMIT_MPS),
        altitude=geom["z_fine"],
        heading=geom["heading"],
        cos_phi=np.ones(n_fine),
        sin_phi=np.zeros(n_fine),
    )


def build_physics_params_sim_stub(
    physical: PhysicalSettings,
    physiological: PhysiologicalSettings,
    run: RunSettings,
    course: CourseProfile,
) -> DummyPhysicsParams:
    """Assemble DummyPhysicsParams. Loads physical.cda_yaw_table_filename
    itself, not an externally-supplied cda_ratios argument (see
    core.simulators' module docstring), and applies cda_ratios[0]
    -- this kernel always evaluates yaw at 0 (no wind model -- see module
    docstring), so that is the only entry ever meaningful here; for the
    default "constant_model.csv" table it's a 1.0 no-op, but a user-
    supplied non-flat table's own yaw=0 factor is still honored rather
    than silently ignored. cp/w_prime are both genuinely read by this
    kernel's own naive W' balance model (see module docstring), unlike
    the rest of PhysiologicalSettings' absent fields."""
    cda_ratios = load_cda_yaw_table(physical.cda_yaw_table_filename)
    return DummyPhysicsParams(
        slope=course.slope,
        time_step=run.time_step,
        distance_step=course.distance_step,
        total_weight=physical.rider_weight + physical.bike_weight,
        cda=physical.cda * cda_ratios[0],
        crr=physical.crr,
        gravity_accel=physical.gravity_accel,
        air_density=physical.air_density,
        cp=physiological.cp,
        w_prime=physiological.w_prime,
    )
