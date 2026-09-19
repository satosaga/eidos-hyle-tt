##########################
# calibrator.py
##########################
"""
EIDOS^TT Parameter Calibrator ("Auto Fit")

Given an actual ride (FIT activity, measured power) and a strategy's
course/physics baseline, finds the physics parameter values that make
the simulator's replayed velocity profile best match the rider's actual
recorded velocity profile.

Objective function
-------------------
sqrt(mean((v_sim(p_i) - v_actual(p_i))^2)) over the native (un-resampled)
FIT samples, compared at the same pct p_i = each side's own distance
divided by that SAME side's own total distance (see
core.activity_correspondence) — not at the same raw distance value: the
Activity's own GPS-spline-derived total distance and the course's own
length routinely disagree by a percent or more, and build_zoh_power_blocks
already rescales the Activity's recorded power onto the course's own
length before replaying it (see build_calibration_inputs), so comparing
the resulting v_sim/v_actual at a shared raw distance would silently
compare power recorded at one real position against a velocity reading
from a different one. v_actual(p_i) is GPS-position-derived speed
(ActivityRecord.gps_speed_ms, populated by every real ActivityRecord
constructor), not the FIT recording's own speed_ms field: speed_ms
reacts several seconds late at a standing-start launch and is
device-oversmoothed elsewhere, both of
which would otherwise bias every free parameter this module fits against
a launch/acceleration-heavy segment (e.g. f_max, brake_lookahead, mu,
brake_usability). Sampled at the native p_i indices (never interpolated);
only the dense simulated trace's velocity is ever interpolated (onto
p_i), matching the "only interpolate the simulated side" policy
eidos.apps.analyzer's Δt panel also follows.

Free parameters
----------------
Any subset of core.simulators.calibratable_physical_keys(simulator_key) --
the resolved simulator's own physical_param_model fields that carry a
ge/le bound pair (for sim_kiritsubo: cda, crr, air_density, wind_speed,
wind_direction, rider_weight, bike_weight, mu, brake_usability, f_max,
brake_lookahead, gravity_accel, min_corner_radius). Bounds are pulled
live from that model's own Field(ge=..., le=...) constraints rather than
duplicated here, so eidos.apps.analyzer's PhysicsOverridePanel and this
module can never drift apart on bounds.

Fixed parameters (every calibratable key NOT in free_keys) come from
whatever `fixed_overrides` dict the caller passes in -- for the Auto Fit
UI, the Analyzer's current spinbox state at the moment Auto Fit is
clicked.

Power source
-------------
Builds a single explicit PowerBlocks up front via
core.activity_parser.build_zoh_power_blocks (zero-order hold on the FIT
samples), then replays it through simulate_power_profile_separated_blocks
with (return_trajectory=True, use_sync_hook=False, is_target_power=False)
— identical call shape to eidos.apps.analyzer's run_scenario 'actual'
branch, so calibration and a manual 'actual' Rebuild always replay
identically for the same physics. is_target_power=False is what makes
this a recorded-power replay (physics driven by the FIT ride's own
measured power, unclamped) rather than a pacing-strategy simulation.

Physics override construction is delegated entirely to
core.physics_overrides.build_overridden_params (the same function
eidos.apps.analyzer's manual Rebuild path uses), so v_limit and
cos_phi/sin_phi recomputation (triggered by mu/brake_usability/cda/
air_density/crr/masses or wind_direction respectively) is never
reimplemented here.

Diagnostics
------------
CalibrationResult's x_std and all_trials (each an independently DE->NM-
polished trial, carrying the extra de_raw_rmse_mps/de_success/
polish_applied attributes calibrate_single_trial attaches) are already
enough to visualize parameter trade-offs and non-identifiability with no
further simulation -- see eidos.lib.calibration_diagnostics
(plot_calibration_diagnostics and friends). Kept as a separate module
specifically so importing matplotlib/scipy.stats never happens inside a
ProcessPoolExecutor worker: workers re-import this module to unpickle
calibrate_single_trial (see calibrate_multistart), so anything imported
at this module's top level is paid for by every worker whether it plots
anything or not.
"""

import concurrent.futures
import contextlib
import datetime
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from core import FLOAT_TIE_BREAKER_EPS
from core.activity_correspondence import activity_pct
from core.activity_parser import (
    ActivityRecord,
    build_zoh_power_blocks,
)
from core.physics_overrides import build_overridden_params
from core.schema import CourseProfile, PhysicsParams, PowerBlocks, bounds_from_field
from core.simulators import calibratable_physical_keys, resolve_simulator

logger = logging.getLogger(__name__)


class CalibrationCancelled(Exception):
    """Raised when should_cancel() fires -- by calibrate_multistart/
    calibrate, or by _evaluate_parallel (shared by sample_morris_
    sensitivity/sample_sobol_sensitivity)."""

# Multiplies calculate_num_trials' n_free-scaled base count to get the
# total number of independent, individually DE->Nelder-Mead-polished
# trials calibrate_multistart runs -- mirrors opt_tenchi's own
# TenchiParams.sub_seed_count (preset 20). differential_evolution's
# "randtobest1bin" recombination pulls one population toward its own
# best individual, so a single population -- however large -- tends to
# collapse onto one basin; escaping that needs several independent,
# non-communicating populations, keeping the best.
SEED_MULTIPLIER = 20

# calculate_num_trials' own seed_factor -- fixed, never varied by any
# caller. Kept as its own named module constant (rather than an inline
# literal inside calculate_num_trials) so eidos.apps.analyzer.window's
# _on_auto_fit_clicked can compute the same trial total for its
# progress display before AutoFitWorker actually starts, without a
# second, independently-drifting copy of the number.
AUTO_FIT_N_SEEDS_FACTOR = 2

# Returned by objective_calibration in place of propagating
# core.simulators.sim_kiritsubo's _compute_speed_limits_sim_kiritsubo's
# "Braking physics failure ... v_sq is negative" ValueError. See
# objective_calibration's docstring for why this is a scored penalty
# rather than a raise. Large relative to a realistic velocity RMSE (m/s)
# so DE/NM steer away from it, but finite so it doesn't abort the whole
# trial.
_INFEASIBLE_PHYSICS_PENALTY_MPS = 1.0e4

# Any evaluation with fun at or above this is an infeasible-physics-
# penalty hit (_INFEASIBLE_PHYSICS_PENALTY_MPS), not a real RMSE -- same
# cutoff eidos.lib.calibration_diagnostics' _FEASIBLE_FUN_CUTOFF_MPS uses
# for the same reason. Used by sample_morris_sensitivity/
# sample_sobol_sensitivity to exclude contaminated trajectories/
# replicates -- see their own docstrings for why DE/NM can tolerate this
# penalty's exact magnitude (only relative ordering matters there) while
# Morris/Sobol' cannot (their statistics are computed directly from the
# raw values).
_FEASIBLE_FUN_CUTOFF_MPS = _INFEASIBLE_PHYSICS_PENALTY_MPS / 2.0

# Retry cap for sample_morris_sensitivity's/sample_sobol_sensitivity's
# infeasible-trajectory/replicate REPLACEMENT loop (draw more, not just
# drop -- see those functions' own docstrings): stop asking for
# replacements once total attempted trajectories/replicates reaches this
# multiple of what was actually requested. Generous relative to an
# ordinary infeasible rate, so this only ever trips when a
# free_keys/fixed_overrides combination is overwhelmingly infeasible, not
# for an ordinary handful of corner hits.
_INFEASIBLE_RESAMPLE_ATTEMPT_FACTOR = 10

# Which keys are calibratable, and which Pydantic model backs each one's
# bounds, are derived mechanically from the resolved simulator's own
# physical_param_model (see core.simulators.calibratable_physical_keys),
# not a separately-maintained table that could drift from core.schema's
# actual field set. Every function in this module that reads a bare
# `key` also takes the `simulator_key` needed to resolve that model --
# see bounds_from_schema below.


def bounds_from_schema(simulator_key: str, key: str) -> tuple[float, float]:
    """
    Read (ge, le) for `key` straight off the resolved simulator's own
    physical_param_model Field. This is the single source of truth for
    calibration bounds — see module docstring.

    Args:
        simulator_key: core.simulators.SIMULATOR_REGISTRY key whose
                    physical_param_model backs `key`.
        key: One of core.simulators.calibratable_physical_keys(simulator_key).

    Returns:
        (lo, hi) bounds.

    Raises:
        KeyError: If key isn't a field on the resolved simulator's
                    physical_param_model.
        ValueError: If the schema field has no ge/le constraint pair —
                    fail fast rather than silently falling back to an
                    unbounded search.
    """
    return bounds_from_field(resolve_simulator(simulator_key).physical_param_model, key)


# ---------------------------------------------------------------------------
# I. Calibration inputs (built once per calibration run, reused by every
#    DE/NM evaluation and picklable for ProcessPoolExecutor workers)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationInputs:
    """
    Everything objective_calibration needs, computed once up front.

    Attributes:
        simulator_key:  core.simulators.SIMULATOR_REGISTRY key to score
                        candidates against -- resolved fresh inside each
                        ProcessPoolExecutor worker (see
                        calibrate_single_trial) rather than resolved once
                        and pickled, the same reasoning as
                        eidos.lib.optimizers.opt_tenchi.optimize_power_and_length_de_wrapper.
        base_physics:   Baseline PhysicsParams (fields not touched by
                        `fixed_overrides` or `free_keys` come from here
                        unchanged — time_step, distance_step, cp, w_prime, etc.).
        raw_physical:   Strategy JSON's raw physical settings dict
                        (fallback source for build_overridden_params).
        raw_physiological: Strategy JSON's raw physiological settings
                        dict (same).
        raw_run:        Strategy JSON's raw_run dict (same -- RunSettings
                        is never itself overridden, just reconstructed by
                        build_overridden_params on every evaluation).
        course_profile: core.schema.CourseProfile (for v_limit / wind
                        geometry recomputation inside
                        build_overridden_params).
        fixed_overrides: Overrides dict applied on every evaluation
                        before free_keys are laid on top — this is
                        where the Auto Fit UI's "spinbox state at click
                        time" enters.
        power_blocks:   ZOH PowerBlocks built from the FIT activity —
                        fixed for the whole calibration run.
        pct_grid:       The Activity's own native FIT samples, expressed
                        as pct (see core.activity_correspondence) —
                        always spans [0, 1].
        v_actual_grid:  GPS-position-derived speed [m/s] at each pct_grid
                        sample (same indices as pct_grid, never
                        interpolated) — see build_calibration_inputs and
                        this module's "Objective function" docstring
                        section for why this is not the FIT recording's
                        own speed_ms.
        free_keys:      Which core.simulators.calibratable_physical_keys(
                        simulator_key) keys are being calibrated, in the
                        order x_free is packed.
    """
    simulator_key: str
    base_physics: PhysicsParams
    raw_physical: dict
    raw_physiological: dict
    raw_run: dict
    course_profile: CourseProfile
    fixed_overrides: dict
    power_blocks: PowerBlocks
    pct_grid: np.ndarray
    v_actual_grid: np.ndarray
    free_keys: list[str] = field(default_factory=list)


def build_calibration_inputs(
    course_distance_m: float,
    simulator_key: str,
    base_physics: PhysicsParams,
    raw_physical: dict,
    raw_physiological: dict,
    raw_run: dict,
    course_profile: CourseProfile,
    fixed_overrides: dict,
    activity_raw: ActivityRecord,
    free_keys: list[str],
) -> CalibrationInputs:
    """
    Build the fixed (per-run) inputs shared by every DE/NM evaluation.

    Args:
        course_distance_m: Strategy's total road distance [m] — used
                    both to build the PowerBlocks (target_distance_m)
                    and to clip the comparison grid to the course extent.
        simulator_key: core.simulators.SIMULATOR_REGISTRY key to
                    calibrate against (see CalibrationInputs).
        base_physics: Baseline PhysicsParams (see CalibrationInputs).
        raw_physical, raw_physiological, raw_run, course_profile:
                    Passed straight through to build_overridden_params on
                    every evaluation.
        fixed_overrides: Overrides dict held constant across the whole
                    run (e.g. the Analyzer's spinbox state).
        activity_raw: Raw (un-resampled) ActivityRecord for the FIT
                    ride being calibrated against.
        free_keys:  Subset of core.simulators.calibratable_physical_keys(
                    simulator_key) to calibrate.

    Returns:
        A populated CalibrationInputs.

    Raises:
        KeyError: If free_keys contains a name not in
                    core.simulators.calibratable_physical_keys(simulator_key).
        ValueError: If activity_raw.gps_speed_ms is None -- every real
                    ActivityRecord constructor is expected to populate
                    it, so this signals a caller passing a malformed
                    record, not a case to paper over with the FIT
                    recording's own speed_ms (the biased signal this
                    switch exists to stop using as calibration ground
                    truth -- see this module's "Objective function"
                    docstring section).
    """
    unknown = [k for k in free_keys if k not in calibratable_physical_keys(simulator_key)]
    if unknown:
        raise KeyError(f"Not a calibratable parameter for simulator '{simulator_key}': {unknown}")

    power_blocks = build_zoh_power_blocks(
        activity_raw, target_distance_m=course_distance_m
    )

    # GPS-position-derived speed, not activity_raw.speed_ms — see this
    # module's "Objective function" docstring section. Read directly off
    # the record -- every real ActivityRecord constructor (core.
    # activity_parser._make_activity_record and hyle.apps.
    # fit2gpx_converter's own) populates gps_speed_ms, so there is no
    # valid activity_raw this can be missing from; this module does not
    # compute it itself.
    gps_speed_ms = activity_raw.gps_speed_ms
    if gps_speed_ms is None:
        raise ValueError(
            "activity_raw has no gps_speed_ms -- every ActivityRecord "
            "constructor is expected to populate it; Auto Fit needs this "
            "and does not fall back to speed_ms"
        )

    # Native grid: only the simulated trace is ever interpolated (in
    # objective_calibration below); the FIT samples themselves are used
    # as-is, at whatever spacing they were recorded at.
    pct_grid = activity_pct(activity_raw)
    v_actual_grid = gps_speed_ms

    return CalibrationInputs(
        simulator_key=simulator_key,
        base_physics=base_physics,
        raw_physical=raw_physical,
        raw_physiological=raw_physiological,
        raw_run=raw_run,
        course_profile=course_profile,
        fixed_overrides=dict(fixed_overrides),
        power_blocks=power_blocks,
        pct_grid=pct_grid,
        v_actual_grid=v_actual_grid,
        free_keys=list(free_keys),
    )


# ---------------------------------------------------------------------------
# II. Objective function
# ---------------------------------------------------------------------------

def pack_physics(x_free: np.ndarray, calib: CalibrationInputs, simulator_spec):
    """
    Return calib.base_physics with calib.fixed_overrides applied, then
    calib.free_keys overridden on top by x_free.

    Args:
        x_free: Candidate values, same order as calib.free_keys.
        calib:  CalibrationInputs.
        simulator_spec: core.simulators.SimulatorSpec resolved from
                calib.simulator_key by the caller (see
                calibrate_single_trial's docstring) -- determines
                the returned params' own shape.

    Returns:
        A new params instance (whatever simulator_spec.build_physics_params
        returns) via core.physics_overrides.build_overridden_params;
        base_physics is never mutated.
    """
    overrides = {**calib.fixed_overrides, **dict(zip(calib.free_keys, x_free))}
    return build_overridden_params(
        simulator_spec=simulator_spec,
        raw_physical=calib.raw_physical,
        raw_physiological=calib.raw_physiological,
        raw_run=calib.raw_run,
        course_profile=calib.course_profile,
        overrides=overrides,
    )


def objective_calibration(x_free: np.ndarray, calib: CalibrationInputs, simulator_spec) -> float:
    """
    sqrt(mean((v_sim(p) - v_actual(p))^2)) over calib.pct_grid (see this
    module's own "Objective function" docstring section for why pct, not
    a shared raw distance value, is the correct correspondence).

    Runs one fast, unclamped replay (return_trajectory=True,
    use_sync_hook=False, is_target_power=False — identical call shape to
    eidos.apps.analyzer's run_scenario 'actual' branch) with the candidate
    x_free packed into params via pack_physics.

    Args:
        x_free: Candidate parameter vector, order matches calib.free_keys.
        calib:  Fixed CalibrationInputs for this run.
        simulator_spec: core.simulators.SimulatorSpec to score against
                (resolved from calib.simulator_key by the caller -- see
                calibrate_single_trial's docstring for why this
                isn't resolved here on every evaluation instead).

    Returns:
        Velocity RMSE in m/s. Returns _INFEASIBLE_PHYSICS_PENALTY_MPS
        instead of propagating core.simulators.sim_kiritsubo's
        _compute_speed_limits_sim_kiritsubo's "Braking physics failure
        ... v_sq is negative" ValueError — see that constant's docstring
        for why this is scored, not raised.
    """
    try:
        physics = pack_physics(x_free, calib, simulator_spec)
    except ValueError:
        # mu / brake_usability / masses / cda combinations that make some
        # corner physically un-brakeable (_compute_speed_limits_sim_
        # kiritsubo raises here). During DE's blind search of a bounded
        # box, this is an expected, frequent region of the search space —
        # not a bug — so it's scored like any other infeasible-region
        # penalty rather than propagated. Letting it propagate would
        # crash that *entire* seed's DE trial (losing every evaluation
        # accumulated over up to maxiter x popsize iterations) over one
        # bad candidate, a much bigger loss than just rejecting that one
        # candidate.
        return _INFEASIBLE_PHYSICS_PENALTY_MPS

    output = simulator_spec.kernel(0.0, calib.power_blocks, physics, True, False, False)

    # Query the simulated trace at each pct_grid sample's own pct, mapped
    # onto THIS candidate's own simulated finish (output.x_traj[-1]) --
    # not calib's course_distance_m, since a candidate's own replay can
    # legitimately finish slightly short of or past it.
    d_query = calib.pct_grid * float(output.x_traj[-1])
    v_sim = np.interp(d_query, output.x_traj, output.v_traj)
    delta = v_sim - calib.v_actual_grid
    return float(np.sqrt(np.mean(delta ** 2)))


# ---------------------------------------------------------------------------
# III. DE -> Nelder-Mead two-stage optimization, one tier
#
# calibrate_multistart runs SEED_MULTIPLIER-scaled independent DE
# populations as directly-scheduled, individually-polished
# ProcessPoolExecutor tasks -- a Nelder-Mead polish costs only ~15-17% of
# one DE population's own cost, so polishing every trial is cheap. The
# populations stay independent (no shared information between them, best
# kept) because differential_evolution's "randtobest1bin" recombination
# pulls an entire population toward its current best individual, so
# *one* population -- however large -- tends to collapse onto a single
# basin.
# ---------------------------------------------------------------------------

def calibrate_de_core(seed: int, calib: CalibrationInputs, bounds: list[tuple[float, float]], simulator_spec):
    """
    Single independent DE population/trial (no polish) — same shape as
    eidos.lib.optimizers.opt_tenchi.optimize_power_and_length_de_core.

    Args:
        seed:   RNG seed for this trial.
        calib:  Fixed CalibrationInputs.
        bounds: [(lo, hi), ...] matching calib.free_keys order.
        simulator_spec: See objective_calibration's docstring. workers=1
                below means this never needs to cross a process boundary
                itself.

    Returns:
        scipy.optimize.OptimizeResult from differential_evolution.
    """
    return differential_evolution(
        func=objective_calibration,
        args=(calib, simulator_spec),
        bounds=bounds,
        strategy="randtobest1bin",
        mutation=(0.1, 1.9),
        recombination=0.9,
        popsize=5,
        maxiter=300,
        tol=0.001,
        polish=False,
        seed=seed,
        workers=1,
    )


def _objective_calibration_normalized(x_norm: np.ndarray, calib: CalibrationInputs, simulator_spec, lo: np.ndarray, hi: np.ndarray) -> float:
    """objective_calibration, evaluated at x_norm's real point (x_norm in
    [0, 1] per free_key, rescaled here via lo + x_norm*(hi-lo)).

    Exists so calibrate_single_trial's Nelder-Mead polish can run in
    [0,1]-normalized space instead of calib.free_keys' raw, wildly
    mismatched physical units/scales (wind_direction's 360-degree range
    vs. crr's 0.019 range, etc.) -- xatol/fatol below are a single
    absolute tolerance applied across every free dimension at once
    (scipy's own Nelder-Mead convergence requires max(|simplex spread|)
    <= xatol AND max(|f spread|) <= fatol together), so in raw units they
    are dimensionally meaningless for most keys (e.g. 1e-4 is ~0.00003%
    of wind_direction's range) -- forcing the simplex toward physically
    pointless precision in wide-range dimensions rather than ever cleanly
    satisfying xatol, burning iterations against maxiter=2000 for
    nothing. Same normalize-then-rescale-at-evaluation-time fix as this
    module's own Morris/Sobol' sampling (see sample_morris_sensitivity's
    docstring).
    """
    x_real = lo + x_norm * (hi - lo)
    return objective_calibration(x_real, calib, simulator_spec)


def calibrate_single_trial(seed: int, calib: CalibrationInputs, bounds: list[tuple[float, float]]):
    """
    One fully independent DE -> Nelder-Mead trial: a single DE population
    (calibrate_de_core), then polish it with a single bounded Nelder-Mead.
    If the polish regresses (NM's own simplex wandered to a worse point
    than where it started — rare but possible with the coverage/
    infeasibility penalties objective_calibration can return), the raw DE
    result is kept instead (same "Phase 3" safety check as
    eidos.lib.optimizers.opt_tenchi.optimize_power_and_length_de_wrapper).

    This runs inside each ProcessPoolExecutor worker -- see
    calibrate_multistart. calib.simulator_key is resolved to its
    SimulatorSpec fresh here, every call, rather than the resolved object
    itself being passed across the process boundary and pickled -- same
    reasoning as eidos.lib.optimizers.opt_tenchi.
    optimize_power_and_length_de_wrapper. Resolving it once per trial is
    negligible next to one DE population's own cost (a plain registry
    lookup vs. hundreds of simulator evaluations).

    Args:
        seed:   RNG seed for this trial's DE run.
        calib:  Fixed CalibrationInputs.
        bounds: [(lo, hi), ...] matching calib.free_keys order.

    Returns:
        scipy.optimize.OptimizeResult from the Nelder-Mead polish (or, if
        polishing regressed, the raw DE result) — .x is this trial's
        calibrated free-parameter vector, .fun is its velocity RMSE in
        m/s. Extra attributes, mirroring opt_tenchi's own
        res.de_raw_time / res.fun split:

        - de_raw_rmse_mps: This trial's DE result, before polishing.
        - de_success: .success of this trial's DE run — False means it
          hit maxiter without satisfying tol/atol, not that it "failed"
          outright.
        - polish_applied: Whether the NM result was kept (False means
          the safety check reverted to the raw DE result because
          polishing regressed).
    """
    simulator_spec = resolve_simulator(calib.simulator_key)
    de_result = calibrate_de_core(seed, calib, bounds, simulator_spec)
    de_fun = float(de_result.fun)

    # NM polish runs in [0,1]-normalized space, not bounds' own raw units
    # -- see _objective_calibration_normalized's docstring for why.
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    nm_result = minimize(
        _objective_calibration_normalized,
        x0=(de_result.x - lo) / (hi - lo),
        args=(calib, simulator_spec, lo, hi),
        method="Nelder-Mead",
        bounds=[(0.0, 1.0)] * len(bounds),
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000},
    )
    nm_result.x = lo + nm_result.x * (hi - lo)  # back to real units for every caller downstream

    # DEBUG, not INFO -- this fires once per trial (up to
    # calculate_num_trials(...) times per calibrate() call), so at the
    # default INFO threshold
    # (core.logging_setup.configure_logging) it would drown out everything
    # else in the Execution Log. A caller who wants actual DE/NM
    # generation counts (as opposed to the maxiter=300/2000 upper bounds
    # alone) can get them by raising just this module's logger to DEBUG,
    # e.g. logging.getLogger("core.calibrator").setLevel(logging.DEBUG).
    logger.debug(
        "calibrate_single_trial(seed=%d): DE nit=%s nfev=%s success=%s fun=%.5f "
        "| NM nit=%s nfev=%s success=%s fun=%.5f | polish_applied=%s",
        seed,
        getattr(de_result, "nit", None), getattr(de_result, "nfev", None),
        de_result.success, de_fun,
        getattr(nm_result, "nit", None), getattr(nm_result, "nfev", None),
        nm_result.success, float(nm_result.fun),
        float(nm_result.fun) < de_fun,
    )

    if float(nm_result.fun) < de_fun:
        final = nm_result
        polish_applied = True
    else:
        final = de_result
        polish_applied = False

    final.de_raw_rmse_mps = de_fun
    final.de_success = bool(de_result.success)
    final.polish_applied = polish_applied
    return final


def calculate_num_trials(n_free: int) -> int:
    """
    Total number of independent DE->Nelder-Mead trials to run, scaled to
    the number of free parameters — same spirit as
    eidos.lib.optimizers.opt_tenchi.calculate_num_seeds(n_seg,
    seed_factor), which uses seed_factor*(n_seg-1)+1 there. Calibration
    has no n_seg-like quantity (opt_tenchi's "-1" comes from its
    segment-length weights' normalization removing one degree of freedom,
    which doesn't apply here), so the base count is the simpler
    AUTO_FIT_N_SEEDS_FACTOR*n_free+1, multiplied by SEED_MULTIPLIER --
    both fixed, no caller has ever varied either.

    Args:
        n_free:          Number of free parameters (len(free_keys)).

    Returns:
        Total independent trial count.
    """
    return (AUTO_FIT_N_SEEDS_FACTOR * n_free + 1) * SEED_MULTIPLIER


def calibrate_multistart(
    calib: CalibrationInputs,
    progress_callback=None,
    should_cancel=None,
):
    """
    Run calculate_num_trials(len(calib.free_keys)) independent
    DE->Nelder-Mead trials (see calibrate_single_trial) in parallel and
    return the best. Trial seeds are 0, 1, 2, ... -- this function's
    only caller, calibrate(), has never started them anywhere else.

    Always creates its own ProcessPoolExecutor (default worker count,
    shut down on return) -- this function's only caller has never passed
    an already-running pool to submit onto instead, unlike sample_morris_
    sensitivity/sample_sobol_sensitivity's own `pool`, which a caller
    genuinely does reuse across many calls (see those functions' own
    docstrings).

    Args:
        calib:      Fixed CalibrationInputs.
        progress_callback: Optional callable(n_done: int, n_total: int,
                    best_rmse_so_far: float | None), invoked once after
                    each trial completes. Exceptions raised by the
                    callback itself are not caught.
        should_cancel: Optional callable() -> bool, polled after each
                    trial completes. If True, remaining futures are
                    cancelled and this raises CalibrationCancelled.

    Returns:
        (best, all_trials) — best is the lowest-.fun result across all
        trials; all_trials is every trial's result, for convergence/
        identifiability diagnostics (see eidos.lib.calibration_diagnostics
        — every entry here is now equally DE+NM-polished).

    Raises:
        RuntimeError: If every trial failed.
        CalibrationCancelled: If should_cancel() returned True before
                    all trials completed.
    """
    n_total = calculate_num_trials(len(calib.free_keys))
    seeds = list(range(n_total))
    bounds = [bounds_from_schema(calib.simulator_key, k) for k in calib.free_keys]

    results = []
    with concurrent.futures.ProcessPoolExecutor() as active_pool:
        futures = {
            active_pool.submit(calibrate_single_trial, seed, calib, bounds): seed
            for seed in seeds
        }
        for fut in concurrent.futures.as_completed(futures):
            seed = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception("Calibration trial (seed=%d) failed", seed)

            if progress_callback is not None:
                best_so_far = min((r.fun for r in results), default=None)
                progress_callback(len(results), len(seeds), best_so_far)

            if should_cancel is not None and should_cancel():
                active_pool.shutdown(wait=False, cancel_futures=True)
                raise CalibrationCancelled(
                    f"Cancelled after {len(results)}/{len(seeds)} trials completed."
                )

    if not results:
        raise RuntimeError("All calibration trials failed — see logged exceptions above.")

    best = min(results, key=lambda r: r.fun)
    return best, results


# ---------------------------------------------------------------------------
# IV. Pre-Auto-Fit sensitivity screening (Morris and Sobol', via SALib)
#
# Runs BEFORE calibrate()/calibrate_multistart, on whichever free_keys
# subset the caller is currently considering -- not part of an Auto Fit
# run itself. Morris and Sobol' are two INDEPENDENT, interchangeable
# views a caller picks between, not a mandatory funnel: Sobol' can be run
# directly against every calibratable parameter at once (see sample_
# sobol_sensitivity's own docstring for the evaluation-count formula),
# so there's no need to narrow the field with Morris first. See
# TTAnalyzerWindow's inline Sobol'/Morris toggle (rendered directly into
# PhysicsOverridePanel, next to the Auto Fit checkboxes), which calls
# whichever is selected against exactly whichever parameters currently
# have their Auto Fit checkbox checked, never a Morris-narrowed subset.
# Both functions remain equally usable standalone against any free_keys
# subset regardless (nothing here requires the full set), and share the
# evaluate-in-parallel-chunks helper below (_evaluate_x_chunk/
# _evaluate_parallel).
#
# SALib (not hand-rolled) for both: Sobol's Saltelli/Jansen estimator is
# easy to get subtly wrong by hand (matrix layout, sign conventions), and
# Morris' own textbook method is trajectory sampling on a discretized
# grid -- SALib is the standard, tested implementation of both rather
# than a from-scratch reimplementation. Imported inside each function
# body, not at this module's top level -- every ProcessPoolExecutor
# worker in calibrate_multistart re-imports this whole module to unpickle
# calibrate_single_trial (see module docstring), so a top-level SALib
# import would be paid by every DE/NM worker too, even though only these
# two functions -- called from the main process -- need it. Same
# reasoning eidos.lib.calibration_diagnostics' own docstring gives for
# keeping scipy.stats/matplotlib out of this module entirely.
#
# Sobol' S1/ST have a genuine 0/0 case: this module's simulator is float
# arithmetic, so RMSE(X) can come back an exact constant C over the whole
# swept region (e.g. brake_usability on a course that never brakes), and
# SALib's own Y = (Y-Y.mean())/Y.std() normalization divides 0/0 on that
# tie. sample_sobol_sensitivity sidesteps this by appending an extra
# tie-breaker variable theta as an ORDINARY (n_free+1)-th Sobol' input
# (never surfaced to a caller) and computing S1/ST/S2 against the always
# well-posed Y'(X, theta) = Y(X) + e(theta) instead of Y(X) alone -- see
# _tie_breaker's own docstring for why this is a principled fix rather
# than an arbitrary substitution, and what its result means in both the
# degenerate and non-degenerate case.
# ---------------------------------------------------------------------------

# Defaults for sample_morris_sensitivity. r=20 trajectories at
# num_levels=4 costs r*(n_free+1) evaluations (20*14=280 at n_free=13,
# every calibratable parameter at once). Only r is exposed as an
# adjustable control in TTAnalyzerWindow's inline sensitivity row;
# num_levels is fixed at this grid resolution, never varied by any
# caller.
DEFAULT_MORRIS_R = 20
DEFAULT_MORRIS_NUM_LEVELS = 4

# Default for sample_sobol_sensitivity. N=512 (a power of 2, as SALib's
# underlying scipy.stats.qmc.Sobol sampler expects for its balance
# properties); with calc_second_order=True (always used here, plus the
# appended tie-breaker variable -- see sample_sobol_sensitivity) costs
# N*(2*n_free+4) evaluations -- 512*16=8192 at n_free=6. Also exposed as
# an adjustable control, not tuned further here.
DEFAULT_SOBOL_N = 512

# _evaluate_parallel's chunk size is TIME-fixed, not point-count-fixed:
# chunk_size = _SENSITIVITY_TARGET_CHUNK_S / _SENSITIVITY_EST_EVAL_S points,
# so each ProcessPoolExecutor task takes about _SENSITIVITY_TARGET_CHUNK_S
# regardless of n or the pool's own worker count. This bounds how long a should_cancel()
# check (see _evaluate_parallel) can be stuck waiting on an
# already-dispatched chunk to finish -- the whole point of chunking at
# all here, since unlike calibrate_multistart's one-future-per-trial
# layout, there's no other natural place to check mid-run. 1s keeps that
# wait short without pushing per-task dispatch/pickling overhead
# (see _evaluate_x_chunk) high enough to matter: at n_free=13 (every
# calibratable parameter), DEFAULT_SOBOL_N=512 is 512*30=15360
# evaluations, ~307 one-second chunks -- dispatch overhead paid ~307
# times, not once per point.
_SENSITIVITY_EST_EVAL_S = 0.02
_SENSITIVITY_TARGET_CHUNK_S = 1.0


def _evaluate_x_chunk(x_chunk: np.ndarray, calib: CalibrationInputs) -> np.ndarray:
    """
    Evaluate objective_calibration for every row of x_chunk -- the
    ProcessPoolExecutor task body for _evaluate_parallel.

    One task per CHUNK, not per point: a single evaluation's own cost
    (see _SENSITIVITY_EST_EVAL_S) is negligible next to
    ProcessPoolExecutor's own per-task dispatch/pickling overhead if
    that overhead were paid n times instead of len(chunks) times.
    simulator_spec is resolved once per chunk (not
    once per point) for the same reason calibrate_single_trial resolves
    it once per trial rather than pickling the resolved object across
    the process boundary -- see that function's docstring.

    Args:
        x_chunk: (chunk_size, len(calib.free_keys)) slice of the full
                    sample.
        calib:   Fixed CalibrationInputs.

    Returns:
        (chunk_size,) array of objective_calibration results, same order
        as x_chunk's rows.
    """
    simulator_spec = resolve_simulator(calib.simulator_key)
    return np.array([objective_calibration(xi, calib, simulator_spec) for xi in x_chunk])


def _evaluate_parallel(
    x: np.ndarray,
    calib: CalibrationInputs,
    pool: concurrent.futures.ProcessPoolExecutor,
    should_cancel=None,
) -> np.ndarray:
    """
    Evaluate objective_calibration at every row of x (shape (n,
    len(calib.free_keys))), split into fixed-TIME chunks (see
    _SENSITIVITY_TARGET_CHUNK_S) under ProcessPoolExecutor -- shared by
    sample_morris_sensitivity and sample_sobol_sensitivity, whose only
    real difference from each other is how x itself is generated (SALib's
    Morris vs. Sobol' samplers).

    Chunk count is independent of pool size here (unlike calibrate_
    multistart's one-future-per-trial layout): normally n itself fits in
    ONE round of sample_morris_sensitivity/sample_sobol_sensitivity's own
    while loop (a replacement round only happens on an infeasible-physics
    discard), so this single call is the entire run -- with no
    should_cancel check inside it, a cancel request would just sit idle
    until every point finished regardless of chunk count. Chunking by
    TIME rather than by worker count is what gives should_cancel()
    somewhere to actually be checked mid-run, at a roughly known interval
    (see _SENSITIVITY_TARGET_CHUNK_S for the responsiveness/dispatch-
    overhead trade this size was picked for).

    Args:
        x:           (n, len(calib.free_keys)) sample matrix, columns in
                    calib.free_keys order.
        calib:       Fixed CalibrationInputs.
        pool:        An already-running ProcessPoolExecutor to submit onto
                    -- always given by both real callers (each resolves
                    its own possibly-None `pool` argument into a
                    concrete one, creating a fresh default-sized
                    ProcessPoolExecutor if the caller didn't supply one,
                    before ever reaching this function), so there is no
                    "create my own pool" fallback path here.
        should_cancel: Optional callable() -> bool, polled after each
                    chunk completes (SensitivityWorker passes its own
                    QThread.isInterruptionRequested). On a True, every
                    not-yet-started chunk's Future.cancel() is called
                    (already-running chunks still finish, their results
                    just go unused) and CalibrationCancelled is raised --
                    NOT active_pool.shutdown(), unlike calibrate_
                    multistart's should_cancel handling, since pool here
                    is routinely a caller-owned pool meant to outlive
                    this one call (see pool's own docstring above).

    Returns:
        (n,) array of objective_calibration results, in x's own row order
        (chunk results are reassembled by chunk index, not completion
        order).

    Raises:
        CalibrationCancelled: If should_cancel() returned True before
                    every chunk had completed.
    """
    n = x.shape[0]
    if n == 0:
        return np.array([])

    chunk_size = max(1, round(_SENSITIVITY_TARGET_CHUNK_S / _SENSITIVITY_EST_EVAL_S))
    chunks = [x[i:i + chunk_size] for i in range(0, n, chunk_size)]

    fun_chunks: list[np.ndarray | None] = [None] * len(chunks)
    with contextlib.nullcontext(pool) as active_pool:
        futures = {
            active_pool.submit(_evaluate_x_chunk, chunk, calib): i
            for i, chunk in enumerate(chunks)
        }
        for fut in concurrent.futures.as_completed(futures):
            fun_chunks[futures[fut]] = fut.result()
            if should_cancel is not None and should_cancel():
                # Per-future cancel(), NOT active_pool.shutdown() -- pool
                # may be a caller-owned, long-lived pool shared across
                # many calls (see eidos.apps.analyzer.window.
                # TTAnalyzerWindow's persistent Sensitivity pool),
                # shutdown() on THAT would kill it for every future call,
                # not just this one. cancel() only drops futures that
                # haven't started running yet -- same net effect (nothing
                # queued keeps running), pool stays alive either way.
                for fut_to_cancel in futures:
                    fut_to_cancel.cancel()
                raise CalibrationCancelled(
                    f"Sensitivity evaluation cancelled after "
                    f"{sum(c is not None for c in fun_chunks)}/{len(chunks)} chunks."
                )
    # Every index was submitted as a future above and as_completed()
    # only returns once every one of them has yielded a result (a
    # should_cancel() hit raises before this point instead) -- so by
    # here every slot has been overwritten with _evaluate_x_chunk's
    # real ndarray, never left at its initial None.
    verified_chunks: list[np.ndarray] = []
    for chunk_result in fun_chunks:
        assert chunk_result is not None
        verified_chunks.append(chunk_result)
    return np.concatenate(verified_chunks)


def _free_key_bounds_arrays(simulator_key: str, free_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) arrays, one entry per free_keys, in that order -- shared by
    sample_morris_sensitivity/sample_sobol_sensitivity's [0, 1]-normalized
    sampling (see their own docstrings for why bounds are normalized)."""
    bounds = [bounds_from_schema(simulator_key, k) for k in free_keys]
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def _find_infeasible_entry_key(
    traj_x: np.ndarray, traj_y: np.ndarray, free_keys: list[str],
) -> str | None:
    """
    Walk one discarded Morris trajectory's own traj_size consecutive
    points in order and return the free_key whose one-at-a-time step
    first crossed from feasible to infeasible (fun >= _FEASIBLE_FUN_CUTOFF_MPS)
    -- the specific step actually responsible for that trajectory's
    infeasibility, not just "some point in it was infeasible." The
    free_key whose aggregate mu_star reads highest is not necessarily the
    one whose step actually triggered infeasibility -- a guess from the
    aggregate statistic alone can be wrong.

    A Morris trajectory changes EXACTLY one free_key between consecutive
    points, by construction (SALib's own sampler, no groups used here) --
    trusted here rather than defensively re-checked, same as every other
    "this shape is guaranteed by construction" assumption in this module.

    Args:
        traj_x:    (traj_size, n_free) this ONE trajectory's own points,
                    in step order.
        traj_y:    (traj_size,) this ONE trajectory's own fun values, same
                    order.
        free_keys: Column names for traj_x, in order.

    Returns:
        The free_key whose step first entered infeasibility, or None if
        the trajectory's own BASELINE (its first point, before any step)
        was already infeasible -- there is no step to attribute that to,
        only the starting point itself (see
        MorrisSensitivityTrials.n_baseline_infeasible).
    """
    infeasible = traj_y >= _FEASIBLE_FUN_CUTOFF_MPS
    if infeasible[0]:
        return None
    for i in range(1, len(traj_y)):
        if infeasible[i] and not infeasible[i - 1]:
            # argmax needs no threshold at all: the "exactly one column
            # changes" invariant (see docstring) holds bit-exactly under
            # SALib's own sampler (every other column's diff is exact
            # 0.0, not merely tiny), so the changed column is simply
            # whichever has the largest
            # |diff|. A magnitude threshold would be both unnecessary and
            # risky here: a real step smaller than the threshold would
            # leave a filtered candidate set empty and crash.
            changed_col = int(np.argmax(np.abs(traj_x[i] - traj_x[i - 1])))
            return free_keys[changed_col]
    # Every trajectory passed to this function is known (by the caller's
    # own infeasible.any() check) to contain at least one infeasible
    # point past a feasible one -- reaching here without returning would
    # mean that invariant didn't hold.
    raise AssertionError(
        "_find_infeasible_entry_key: no feasible->infeasible transition "
        "found in a trajectory the caller flagged as containing one."
    )


def _find_infeasible_ab_keys(rep_y: np.ndarray, free_keys: list[str]) -> tuple[list[str], bool, bool]:
    """
    For one discarded Sobol' replicate (rep_y: (2*n_free+4,) fun values in
    SALib's own [A, AB_1..AB_D, AB_theta, BA_1..BA_D, BA_theta, B] order --
    calc_second_order=True's layout over free_keys PLUS the trailing tie-
    breaker variable theta, see sample_sobol_sensitivity's docstring),
    identify which free_keys are implicated. AB_theta/BA_theta are always
    skipped here, never attributed to any free_key: theta never reaches
    the simulator (only Y(X)'s own free_keys do -- theta only perturbs the
    RMSE value afterward, see _tie_breaker), so AB_theta/BA_theta's fun
    value is always bit-identical to A's/B's own -- infeasible there iff
    A/B already was, which a_infeasible/b_infeasible already cover.

    Weaker attribution than _find_infeasible_entry_key's (Sobol'
    replicates are two INDEPENDENT random points A/B plus 2D cross-samples,
    not a sequential one-at-a-time walk, so there is no single "step"
    that caused entry into infeasibility here): AB_i is "free_key i at
    this replicate's own B-sample value, every other key at its A-sample
    value" (BA_i the mirror -- free_key i at its A-sample value, every
    other key at B's), so an infeasible AB_i or BA_i while A and B are
    each individually feasible directly implicates free_key i's specific
    value IN THAT CONTEXT -- not a proof it alone would be infeasible with
    a different context. If the base point A or B is itself infeasible,
    that isn't attributable to any single key (every key's own value in
    that whole combination could be involved at once), so it's reported
    separately.

    Args:
        rep_y:     (2*n_free+4,) this ONE replicate's own fun values.
        free_keys: Names for rep_y's D real AB_i/BA_i entries, in order
                    (NOT including theta, always the trailing variable).

    Returns:
        (implicated_keys, a_infeasible, b_infeasible) -- implicated_keys
        is the list of free_keys whose own AB_i point was infeasible
        (possibly more than one, possibly empty even though the
        replicate itself is a "bad" one, if the infeasibility is
        entirely in A or B rather than any single AB_i).
    """
    cutoff = _FEASIBLE_FUN_CUTOFF_MPS
    n_free = len(free_keys)
    a_val = rep_y[0]
    b_val = rep_y[-1]
    ab_vals = rep_y[1:1 + n_free]                        # AB_1..AB_D (AB_theta at 1+n_free skipped)
    ba_vals = rep_y[2 + n_free:2 + 2 * n_free]            # BA_1..BA_D (BA_theta at 2+2*n_free skipped)
    implicated = [
        free_keys[i] for i in range(n_free) if ab_vals[i] >= cutoff or ba_vals[i] >= cutoff
    ]
    return implicated, bool(a_val >= cutoff), bool(b_val >= cutoff)


@dataclass
class MorrisSensitivityTrials:
    """
    Morris (1991)/Campolongo et al. (2007) elementary-effects statistics
    from sample_morris_sensitivity, per free_key: mu (signed mean -- can
    cancel toward zero for a non-monotonic effect, kept for completeness),
    mu_star (mean of ABSOLUTE elementary effects -- the usual screening
    ranking, immune to that cancellation), sigma (their standard
    deviation -- large relative to mu_star flags a nonlinear or
    interaction-driven effect), and mu_star_conf (SALib's own bootstrap
    confidence interval half-width on mu_star).

    mu_star/sigma/mu_star_conf are in "RMSE change per full sweep of this
    free_key's own [lo, hi] range" units -- directly comparable ACROSS
    free_keys regardless of their individual magnitudes or physical units
    (m/s, kg, dimensionless, ...) -- see sample_morris_sensitivity's own
    docstring for why (elementary effects computed against each key's raw
    bounds instead would make a narrow-range key like crr read as
    artifactually far more "sensitive" than a wide-range one like
    wind_speed -- a pure unit-scale artifact, not a real difference).

    Attributes:
        mu, mu_star, sigma, mu_star_conf: dict mapping each free_key to
                    its own scalar statistic, always computed from exactly
                    n_trajectories_requested VALID trajectories -- see
                    n_trajectories_discarded below and
                    sample_morris_sensitivity's docstring for how that's
                    guaranteed (replacement, not just exclusion).
        n_trajectories_requested: The r originally asked for -- always
                    how many trajectories mu/mu_star/sigma were actually
                    computed from too (barring the ValueError case where
                    that couldn't be reached at all).
        n_trajectories_discarded: How many EXTRA trajectories, beyond
                    n_trajectories_requested, had to be drawn and thrown
                    away because they contained an infeasible-physics-
                    penalty evaluation -- see sample_morris_sensitivity's
                    docstring for why WHOLE trajectories, not individual
                    points. 0 in the common case. Reported rather than
                    silently absorbed, same "no silent fallback"
                    reasoning CalibrationResult.n_trials/n_converged
                    already follows for a different kind of shortfall.
        infeasible_entry_counts: Per free_key, how many of the discarded
                    trajectories that key's OWN one-at-a-time step was
                    the one that first crossed from feasible into
                    infeasible physics -- see
                    _find_infeasible_entry_key. Every free_key present
                    (0 for one never implicated), so this is safe to
                    read/sum without a membership check. Deliberately NOT
                    the same thing as "which key's bar reads high" -- see
                    _find_infeasible_entry_key's docstring for a real
                    case where those two disagreed.
        n_baseline_infeasible: How many discarded trajectories were
                    already infeasible at their own baseline (first
                    point, before any step) -- not attributable to any
                    one free_key's step, so not counted in
                    infeasible_entry_counts.
        sample_x:   Per free_key, EVERY evaluated point's own value (real
                    schema units) across all r trajectories actually used
                    (post infeasible-physics replacement). Same point
                    order/length as sample_y and every other free_key's
                    own array -- the raw (X_i, Y) data behind
                    eidos.apps.analyzer's click-to-scatter popup (see
                    SobolSensitivityTrials.sample_x for the identical
                    shape/reasoning; Morris has no S2-equivalent, so this
                    is ALL the plottable data Morris offers, used the same
                    way regardless of whether mu_star's or sigma's own row
                    was clicked).
        sample_y:   Every evaluated point's own RMSE, same order as
                    sample_x's arrays.
    """
    mu: dict[str, float]
    mu_star: dict[str, float]
    sigma: dict[str, float]
    mu_star_conf: dict[str, float]
    sample_x: dict[str, np.ndarray]
    sample_y: np.ndarray
    n_trajectories_requested: int
    n_trajectories_discarded: int
    infeasible_entry_counts: dict[str, int]
    n_baseline_infeasible: int


def sample_morris_sensitivity(
    calib: CalibrationInputs,
    r: int = DEFAULT_MORRIS_R,
    pool: concurrent.futures.ProcessPoolExecutor | None = None,
    should_cancel=None,
) -> MorrisSensitivityTrials:
    """
    Formal trajectory-based Morris screening over calib.free_keys' full
    schema bounds, via SALib.sample.morris/SALib.analyze.morris -- one of
    two interchangeable pre-Auto-Fit sensitivity views (see this
    section's own header comment for why this and sample_sobol_
    sensitivity are independent alternatives, not a mandatory funnel):
    cheap (r*(n_free+1) evaluations), and its mu*/sigma answer a
    genuinely different question than Sobol' S1/ST (see
    MorrisSensitivityTrials' own docstring) even where both are cheap
    enough to just run directly.

    Sampled and analyzed in [0, 1]-NORMALIZED units per free_key, not each
    key's own raw schema bounds -- SALib's problem["bounds"] is [[0,1],
    ...] here, with the generated sample rescaled into real bounds only
    for the objective_calibration evaluation itself
    (_evaluate_parallel(x_real, ...)), and si["mu"/"mu_star"/"sigma"] read
    back against the ORIGINAL normalized x. This matters because a Morris
    elementary effect is a finite difference (delta_y / delta_x): computed
    against RAW bounds, a free_key with a naturally tiny absolute range
    (e.g. crr, ~0.001-0.02) gets divided by a tiny delta_x and its mu_star
    comes out inflated by orders of magnitude relative to a free_key with
    a wide range (e.g. wind_speed, 0-25 m/s) -- a pure unit-scale artifact
    of which parameter happens to use small numbers, not a real
    difference in importance (Sobol' S1/ST, variance-based, are already
    scale-free by construction and show no such inflation). Normalizing
    so every free_key's own full range
    maps to the same [0, 1] step size fixes this: mu_star/sigma now read
    as "RMSE change per full sweep of this key's own range," directly
    comparable across free_keys regardless of their individual units or
    magnitudes.

    Trajectories containing an infeasible-physics-penalty evaluation
    (objective_calibration returning _INFEASIBLE_PHYSICS_PENALTY_MPS,
    not a real RMSE -- see that constant's docstring) are discarded and
    REPLACED, not just dropped: draws a fresh replacement trajectory (a
    new seed) for each one discarded, repeating until r fully-feasible
    trajectories are collected (or _INFEASIBLE_RESAMPLE_ATTEMPT_FACTOR*r
    total attempts is reached -- see Raises). This always spends the full
    r trajectories' worth of statistical power the caller asked for,
    rather than silently handing back a smaller, weaker screen. Whole
    trajectories, not individual points, are discarded/replaced: unlike
    calibration_diagnostics._pool_trials (a flat, unstructured pool where
    dropping individual rows is harmless), a Morris trajectory is a
    structured sequence SALib's analyze() needs INTACT to compute one
    valid elementary effect per parameter from it, so the only safe
    granularity is "the whole trajectory." Not just a defensive guard
    against something that can't happen: infeasible braking physics is
    an expected, frequent region of the search space (see
    objective_calibration's own docstring) that DE's blind search hits
    routinely -- left unhandled, that corner's literal 10000 penalty
    value would dominate whichever free_key's trajectory step happened
    to touch it, inflating its mu_star into the thousands while every
    other free_key reads normally. See MorrisSensitivityTrials.
    n_trajectories_discarded for how a caller learns this happened even
    though the final r is always met.

    Args:
        calib:      Fixed CalibrationInputs -- calib.free_keys is the set
                    of parameters being screened; every OTHER
                    calibratable key stays at calib.fixed_overrides.
        r:          Number of Morris trajectories. See DEFAULT_MORRIS_R.
        pool:       See _evaluate_parallel -- genuinely reused across
                    many calls by TTAnalyzerWindow's persistent
                    Sensitivity pool, unlike the fixed constants below.
        should_cancel: See _evaluate_parallel -- polled roughly once per
                    _SENSITIVITY_TARGET_CHUNK_S inside each round's
                    _evaluate_parallel call (normally the only round; an
                    infeasible-physics replacement round gets its own
                    check the same way).

    Grid resolution (num_levels, DEFAULT_MORRIS_NUM_LEVELS) and the RNG
    seed (0) are fixed -- no caller has ever varied either. Caching is
    always under core.calibration_cache's own
    DEFAULT_SENSITIVITY_CACHE_DIR.

    Every call checks core.calibration_cache before running anything and
    returns a cached MorrisSensitivityTrials on a hit instead of
    recomputing, then saves a fresh result to the cache after a miss --
    no caller has ever needed to bypass this. A hit is only ever
    returned for a byte-for-byte-equivalent call (calib, r, num_levels,
    seed, and a source-code fingerprint -- see calibration_cache.
    compute_sensitivity_cache_key).

    Returns:
        A MorrisSensitivityTrials with one mu/mu_star/sigma/mu_star_conf
        value per free_key, always computed from exactly r trajectories
        (n_trajectories_discarded reports how many extra had to be drawn
        along the way).

    Raises:
        ValueError: If _INFEASIBLE_RESAMPLE_ATTEMPT_FACTOR*r total
                    trajectories were attempted without ever reaching r
                    usable ones -- this free_keys/fixed_overrides
                    combination is overwhelmingly infeasible, a real
                    finding worth investigating directly, not something
                    to paper over with a fabricated result.
        CalibrationCancelled: If should_cancel() fired mid-run -- see
                    _evaluate_parallel. Not cached (the cache save only
                    runs after a normal return).
    """
    from core import calibration_cache

    # Fixed constants -- no caller has ever varied either (see this
    # module's general "no default arguments unless genuinely varied"
    # convention).
    num_levels = DEFAULT_MORRIS_NUM_LEVELS
    seed = 0

    cache_key = calibration_cache.compute_sensitivity_cache_key(
        calib, "morris", code_fingerprint=_source_fingerprint(calib.simulator_key),
        r=r, num_levels=num_levels, seed=seed,
    )
    cached = calibration_cache.load_cached_sensitivity(cache_key, calibration_cache.DEFAULT_SENSITIVITY_CACHE_DIR)
    if cached is not None:
        return cached

    from SALib.analyze import morris as morris_analyze
    from SALib.sample import morris as morris_sample

    n_free = len(calib.free_keys)
    lo, hi = _free_key_bounds_arrays(calib.simulator_key, calib.free_keys)
    problem = {
        "num_vars": n_free,
        "names": calib.free_keys,
        "bounds": [[0.0, 1.0]] * n_free,
    }
    # Each trajectory is exactly n_free+1 consecutive rows in x/y (SALib's
    # own sample layout, with no groups/optimal_trajectories used here),
    # so a simple reshape recovers trajectory boundaries.
    traj_size = n_free + 1

    x_norm_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    n_collected = 0
    n_discarded = 0
    n_attempted = 0
    max_attempted = _INFEASIBLE_RESAMPLE_ATTEMPT_FACTOR * r
    round_seed = seed
    entry_counts: dict[str, int] = {k: 0 for k in calib.free_keys}
    n_baseline_infeasible = 0

    t0 = time.monotonic()
    # A single pool spans every round below (not just one _evaluate_parallel
    # call) -- reused across a replacement round so it doesn't pay the
    # ProcessPoolExecutor cold-start tax again for what's normally just a
    # handful of extra trajectories.
    pool_ctx = contextlib.nullcontext(pool) if pool is not None else concurrent.futures.ProcessPoolExecutor()
    with pool_ctx as active_pool:
        while n_collected < r:
            shortfall = r - n_collected
            x_norm_batch = morris_sample.sample(problem, N=shortfall, num_levels=num_levels, seed=round_seed)
            x_real_batch = lo + x_norm_batch * (hi - lo)
            y_batch = _evaluate_parallel(x_real_batch, calib, active_pool, should_cancel)
            n_attempted += shortfall

            infeasible = y_batch >= _FEASIBLE_FUN_CUTOFF_MPS
            x_norm_3d = x_norm_batch.reshape(shortfall, traj_size, n_free)
            y_2d = y_batch.reshape(shortfall, traj_size)
            bad_traj = infeasible.reshape(shortfall, traj_size).any(axis=1)
            n_bad = int(bad_traj.sum())
            n_discarded += n_bad

            # Attribute each discarded trajectory to the specific free_key
            # whose step actually caused it, not just "some point in it
            # was infeasible" -- see _find_infeasible_entry_key's docstring.
            for ti in np.where(bad_traj)[0]:
                entry_key = _find_infeasible_entry_key(x_norm_3d[ti], y_2d[ti], calib.free_keys)
                if entry_key is None:
                    n_baseline_infeasible += 1
                else:
                    entry_counts[entry_key] += 1

            keep = ~bad_traj
            if keep.any():
                x_norm_parts.append(x_norm_3d[keep].reshape(-1, n_free))
                y_parts.append(y_2d[keep].reshape(-1))
            n_collected += shortfall - n_bad

            if n_collected < r and n_attempted >= max_attempted:
                raise ValueError(
                    f"sample_morris_sensitivity: gave up after {n_attempted} attempted "
                    f"trajectories ({n_discarded} hit infeasible physics, fun >= "
                    f"{_FEASIBLE_FUN_CUTOFF_MPS}) -- only {n_collected}/{r} usable. "
                    f"This free_keys/fixed_overrides combination is overwhelmingly "
                    f"infeasible within the swept bounds; investigate the "
                    f"fixed_overrides (e.g. an already-extreme spinbox value) or "
                    f"narrow free_keys' bounds rather than trusting a result built "
                    f"mostly from penalty values."
                )
            round_seed += 1

    if n_discarded:
        entry_breakdown = ", ".join(f"{k}={v}" for k, v in entry_counts.items() if v) or "none"
        logger.warning(
            "sample_morris_sensitivity: %d infeasible-physics trajector%s "
            "discarded and replaced (fun >= %.1f) -- attempted %d total to "
            "reach r=%d usable. Entry-step causes: %s%s.",
            n_discarded, "y" if n_discarded == 1 else "ies", _FEASIBLE_FUN_CUTOFF_MPS,
            n_attempted, r, entry_breakdown,
            f" ({n_baseline_infeasible} already infeasible at baseline)" if n_baseline_infeasible else "",
        )
    elapsed = time.monotonic() - t0
    logger.debug(
        "sample_morris_sensitivity: n_free=%d r=%d n_evals=%d elapsed=%.1fs",
        n_free, r, n_attempted * traj_size, elapsed,
    )

    x_norm_used = np.concatenate(x_norm_parts, axis=0)
    x_real_used = lo + x_norm_used * (hi - lo)
    y_used = np.concatenate(y_parts, axis=0)
    si = morris_analyze.analyze(problem, x_norm_used, y_used, num_levels=num_levels, seed=seed)
    result = MorrisSensitivityTrials(
        mu={k: float(v) for k, v in zip(calib.free_keys, si["mu"])},
        mu_star={k: float(v) for k, v in zip(calib.free_keys, si["mu_star"])},
        sigma={k: float(v) for k, v in zip(calib.free_keys, si["sigma"])},
        mu_star_conf={k: float(v) for k, v in zip(calib.free_keys, si["mu_star_conf"])},
        sample_x={k: x_real_used[:, i].copy() for i, k in enumerate(calib.free_keys)},
        sample_y=y_used.copy(),
        n_trajectories_requested=r,
        n_trajectories_discarded=n_discarded,
        infeasible_entry_counts=entry_counts,
        n_baseline_infeasible=n_baseline_infeasible,
    )

    calibration_cache.save_cached_sensitivity(cache_key, result, calibration_cache.DEFAULT_SENSITIVITY_CACHE_DIR)

    return result


@dataclass
class SobolSensitivityTrials:
    """
    Sobol' (2001) variance-based sensitivity indices from
    sample_sobol_sensitivity, per free_key: s1 (first-order -- this
    parameter's own marginal contribution to RMSE's variance) and st
    (total-order -- s1 plus every interaction this parameter takes part
    in). s1 << st for a parameter means most of its influence runs
    through interaction with other free parameters, not its value alone.
    Second-order (pairwise interaction) indices ARE computed (calc_
    second_order=True -- see sample_sobol_sensitivity), specifically so a
    high-st/low-s1 free_key's specific interaction partner(s) can be
    read off directly -- see eidos.apps.analyzer.dialogs'
    SobolS2DetailsDialog, which shows the full S1/S2 matrix and opens an
    interaction scatter for whichever cell the viewer clicks.

    Attributes:
        s1, st, s1_conf, st_conf: dict mapping each free_key to its own
                    scalar statistic (s1_conf/st_conf: SALib's own
                    bootstrap confidence interval half-widths).
        s2, s2_conf: dict mapping each UNORDERED pair of free_keys
                    (key_a, key_b) with key_a < key_b lexicographically --
                    read directly by eidos.apps.analyzer.dialogs'
                    SobolS2DetailsDialog -- to SALib's second-order index
                    for that pair (s2_conf: its bootstrap CI half-width).
                    Only pairs with a finite S2 (SALib returns NaN
                    below/on its own upper-triangular matrix's diagonal)
                    are present.
        sample_x:   Per free_key, EVERY evaluated point's own value (real
                    schema units, not the [0, 1]-normalized units this is
                    sampled/analyzed in internally) across all n
                    replicates actually used (post infeasible-physics
                    replacement -- see n_replicates_discarded). Same
                    point order/length as sample_y and every other
                    free_key's own array, so sample_x[key_a][i]/
                    sample_x[key_b][i]/sample_y[i] are one evaluated
                    point's (X_a, X_b, Y) triple -- the raw data behind
                    both the S1-click 2D scatter and the ST-click 3D
                    interaction scatter.
        sample_y:   Every evaluated point's own RMSE, same order as
                    sample_x's arrays -- the real Y(X), NOT Y'(X, theta)
                    (see sample_sobol_sensitivity, _tie_breaker) that
                    s1/st/s2 were actually computed from.
        n_replicates_requested, n_replicates_discarded: See
                    MorrisSensitivityTrials' n_trajectories_requested/
                    n_trajectories_discarded -- same "always meet the
                    requested count via replacement, report the discard
                    count" precedent, for sample_sobol_sensitivity's own
                    infeasible-physics replacement (see that function's
                    docstring). A "replicate" here is one full
                    A/AB_1..D/BA_1..D/B group (2*n_free+4 evaluations,
                    including the tie-breaker's own AB_theta/BA_theta --
                    see _find_infeasible_ab_keys) from SALib's Saltelli
                    sample -- the Sobol' equivalent of one Morris
                    trajectory: the smallest unit that can be discarded
                    without breaking analyze()'s required sample shape.
        infeasible_ab_counts: Per free_key, how many of the discarded
                    replicates that key's own AB_i/BA_i cross-sampled
                    point was itself infeasible -- see
                    _find_infeasible_ab_keys for exactly what that
                    implicates (weaker than Morris' infeasible_entry_
                    counts: a replicate can implicate more than one key at
                    once, or none, if the infeasibility is in the base
                    A/B point instead -- see n_base_point_infeasible).
                    Every free_key present (0 for one never implicated).
        n_base_point_infeasible: How many discarded replicates' own base
                    A or B point (not any single AB_i/BA_i) was itself
                    infeasible -- not attributable to one free_key, since
                    every key's value in that whole combination could be
                    involved at once. Counts A and B separately (a
                    replicate with both infeasible counts as 2).
    """
    s1: dict[str, float]
    st: dict[str, float]
    s1_conf: dict[str, float]
    st_conf: dict[str, float]
    s2: dict[tuple[str, str], float]
    s2_conf: dict[tuple[str, str], float]
    sample_x: dict[str, np.ndarray]
    sample_y: np.ndarray
    n_replicates_requested: int
    n_replicates_discarded: int
    infeasible_ab_counts: dict[str, int]
    n_base_point_infeasible: int


# theta's own name in the Sobol' problem dict -- always the LAST variable
# (see sample_sobol_sensitivity), never a real free_key
# (calibratable_physical_keys' own keys are all plain physics-param
# names, never leading underscore), and never surfaced to the user
# (see _tie_breaker).
_TIE_BREAKER_NAME = "_tie_breaker_theta"

# e(theta)'s amplitude -- an ABSOLUTE floor (RMSE's own units, m/s),
# NOT relative to whatever Y(X) happens to be this run: the only actual
# constraint is "small enough to stay far below any real physical
# signal's own precision, large enough that adding it doesn't itself
# introduce numerical error" -- a computation-precision question, not a
# question about this particular Y(X)'s magnitude. Sourced from
# core.FLOAT_TIE_BREAKER_EPS -- this is the same "probability-0-in-the-
# reals, reachable-under-float64" tie-breaking problem as any other
# exact-coincidence degeneracy in this codebase, not a value specific to
# Sobol'/RMSE (see that constant's own docstring).
_TIE_BREAKER_AMPLITUDE = FLOAT_TIE_BREAKER_EPS


def _tie_breaker(theta: np.ndarray) -> np.ndarray:
    """
    e(theta): a deterministic, tiny-amplitude function of theta -- Sobol'
    S1/ST/S2 are computed against Y'(X, theta) = Y(X) + e(theta), NOT
    Y(X) alone (see sample_sobol_sensitivity, where theta is sampled as
    an ordinary (n_free+1)-th Sobol' input, not injected behind SALib's
    back).

    Why this is needed at all: Y is a real number in the physical model,
    so two independently-evaluated real Y values coinciding exactly has
    probability 0 -- but this module's simulator is float arithmetic, so
    exact ties DO happen (Y(X) = C, a constant simulator output over the
    whole swept region, is the extreme case). SALib's own Y = (Y-Y.mean())
    /Y.std() normalization divides 0/0 on that tie. e(theta) restores the
    "almost surely distinct" property floats broke, by construction --
    NOT a resolution of an indeterminate S1/ST by picking a value: since
    theta is an ordinary Sobol' input like any free_key, S1_i/ST_i for
    every real free_key are the literal, standard Sobol' indices of
    Y'(X, theta) -- e.g. Y(X) = C identically makes Y'(X, theta) a
    function of theta ALONE, so every free_key's S1/ST = 0 by ordinary
    Sobol' decomposition (a function that doesn't depend on X_i has zero
    sensitivity to X_i), not a special-cased substitution.

    e(theta) is a physical representation model of an unobservable,
    infinitesimal degree of freedom -- unresolved micro-scale variation
    the swept free_keys don't and can't parameterize, but that a
    real-valued physical quantity would always have, breaking the exact
    float coincidence.

    e's specific functional form is arbitrary by construction -- a plain
    linear ramp (e(theta) = _TIE_BREAKER_AMPLITUDE * (2*theta - 1), odd
    about theta's own midpoint) chosen over any other non-constant shape
    specifically for NOT suggesting this unobservable degree of freedom
    behaves in any particular way (unlike, say, a periodic shape, which
    would cosmetically imply oscillation) -- linear is the least
    assumption-laden non-constant choice, and strictly monotonic
    (injective), so no two distinct theta draws ever land on the same
    e(theta). Only non-constant is actually required (a constant e(theta)
    would reintroduce the same zero-variance failure one level up); the
    slope matters less than the shape, since only e's own VARIANCE over
    theta's domain (scaling with _TIE_BREAKER_AMPLITUDE^2 regardless of
    shape) sets how large a real signal has to be before it dominates
    e(theta) -- see sample_sobol_sensitivity's real_y_std/
    _TIE_BREAKER_AMPLITUDE debug log.

    Not applied to sample_morris_sensitivity: SALib's Morris estimator is
    a plain elementary-effect difference with no Y.std()-based
    normalization step, so it already returns exact 0.0 mu_star/sigma for
    a constant y without needing a tie-breaker at all.

    Args:
        theta: This replicate set's own sampled theta column, already in
                    [0, 1) (SALib samples it exactly like any other
                    free_key -- see sample_sobol_sensitivity).

    Returns:
        e(theta), same shape as theta -- amplitude _TIE_BREAKER_AMPLITUDE.
    """
    return _TIE_BREAKER_AMPLITUDE * (2.0 * theta - 1.0)


def sample_sobol_sensitivity(
    calib: CalibrationInputs,
    n: int = DEFAULT_SOBOL_N,
    pool: concurrent.futures.ProcessPoolExecutor | None = None,
    should_cancel=None,
) -> SobolSensitivityTrials:
    """
    Sobol' S1/ST (+S2, see calc_second_order below) via SALib.sample.sobol/
    SALib.analyze.sobol -- the other of the two interchangeable pre-Auto-
    Fit sensitivity views (see this section's own header comment).
    calib.free_keys can be any subset in principle; TTAnalyzerWindow's
    inline sensitivity row passes whichever Auto Fit keys are currently
    checked, not a Morris-narrowed one -- N*(2*n_free+4) evaluations
    (see DEFAULT_SOBOL_N) scales cheaply enough that narrowing first
    buys nothing.

    Actually run against n_free+1 variables, not n_free: a tie-breaker
    variable theta is appended as an ordinary Sobol' input alongside
    calib.free_keys, and S1/ST/S2 are computed for Y'(X, theta) = Y(X) +
    e(theta), not Y(X) directly -- see _tie_breaker's own docstring for
    why (Y(X) = C, a constant simulator output, is an exact float tie
    that SALib's own Y.std() normalization can't divide by; theta/e
    restore the "almost surely distinct" property real-valued Y would
    have had). Only calib.free_keys' own S1/ST/S2 are exposed on the
    returned SobolSensitivityTrials -- theta's own row/column is dropped
    before construction, never shown to a caller.

    calc_second_order=True throughout, paid on every call, not just when
    a caller happens to want S2: S1/ST alone answer "how much does this
    parameter matter," but a high-ST/low-S1 free_key's own S2 row is what
    identifies WHICH other free_key it's actually interacting with -- the
    piece that turns "ST is high" into a specific, plottable claim
    (eidos.apps.analyzer.dialogs' SobolS2DetailsDialog), not just a
    warning that something, somewhere, is going on. S2 can't be added
    post-hoc to a calc_second_order=False
    sample -- SALib's own sample.sobol needs the extra AB_j columns
    present from the start (N*(2*n_free+2) points vs N*(n_free+2)
    without them) -- so gating it behind a second, separately-sampled
    call would mean paying for the base N*(n_free+2) evaluations twice
    over, not saving anything.

    Sampled in [0, 1]-normalized units per free_key, rescaled into real
    bounds only for the objective_calibration evaluation itself -- same
    pattern as sample_morris_sensitivity, though for a different reason
    here: Sobol' S1/ST are a variance-decomposition FRACTION (Var(E[Y|
    X_i])/Var(Y)), already invariant to a linear rescaling of X_i's own
    units, so this isn't fixing a scale artifact the way it is for
    Morris -- it's kept anyway so both sampling functions share one
    convention rather than one normalizing and the other not for no
    principled reason.

    Replicates containing an infeasible-physics-penalty evaluation are
    discarded and REPLACED (a fresh replacement replicate drawn with a
    new seed), not just dropped, so this always spends the full N
    replicates' worth of statistical power requested -- same mechanism
    and same "whole structured unit, not individual points" constraint as
    sample_morris_sensitivity's own trajectory replacement. Even more
    important to replace here than for Morris: SALib's sobol.analyze
    centers/scales Y by ITS OWN mean and standard
    deviation internally before decomposing variance, so even a single
    ~10000 penalty value left in Y would distort every free_key's S1/ST
    (via that shared mean/std), not just the one whose replicate hit it.

    Args:
        calib:      Fixed CalibrationInputs -- calib.free_keys is the
                    (already-narrowed) set of parameters being screened.
        n:          Sobol' base sample size. See DEFAULT_SOBOL_N.
        pool:       See sample_morris_sensitivity's identically-named
                    parameter / _evaluate_parallel.
        should_cancel: See sample_morris_sensitivity's identically-named
                    parameter / _evaluate_parallel.

    The RNG seed (0) and the cache directory (core.calibration_cache's
    own DEFAULT_SENSITIVITY_CACHE_DIR) are both fixed -- no caller has
    ever varied either. Caching is keyed separately per method (see
    calibration_cache.compute_sensitivity_cache_key) so a Morris and
    Sobol' call against the same calib never collide.

    Returns:
        A SobolSensitivityTrials with one s1/st/s1_conf/st_conf value per
        free_key, always computed from exactly n replicates
        (n_replicates_discarded reports how many extra had to be drawn
        along the way). Never NaN even if RMSE came back identical at
        every one of those replicates (every free_key equally unable to
        move the outcome) -- see _tie_breaker's docstring for why and how
        that's handled; sample_y still holds the real Y(X), not Y'(X,
        theta).

    Raises:
        ValueError: If _INFEASIBLE_RESAMPLE_ATTEMPT_FACTOR*n total
                    replicates were attempted without ever reaching n
                    usable ones -- see sample_morris_sensitivity's
                    identical Raises entry; same reasoning applies here.
        CalibrationCancelled: See sample_morris_sensitivity's identical
                    Raises entry.
    """
    from core import calibration_cache

    # Fixed -- no caller has ever varied it.
    seed = 0

    cache_key = calibration_cache.compute_sensitivity_cache_key(
        calib, "sobol", code_fingerprint=_source_fingerprint(calib.simulator_key),
        n=n, seed=seed,
    )
    cached = calibration_cache.load_cached_sensitivity(cache_key, calibration_cache.DEFAULT_SENSITIVITY_CACHE_DIR)
    if cached is not None:
        return cached

    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    n_free = len(calib.free_keys)
    lo, hi = _free_key_bounds_arrays(calib.simulator_key, calib.free_keys)
    # theta (see _tie_breaker) is an ordinary (n_free+1)-th Sobol' input,
    # always LAST in names/bounds -- everything downstream that slices by
    # position (x_norm_batch's columns, _find_infeasible_ab_keys) assumes
    # that ordering.
    problem = {
        "num_vars": n_free + 1,
        "names": calib.free_keys + [_TIE_BREAKER_NAME],
        "bounds": [[0.0, 1.0]] * (n_free + 1),
    }
    # One replicate is exactly 2*(n_free+1)+2 consecutive rows in x/y
    # (SALib's own Saltelli sample layout with calc_second_order=True:
    # one A eval, (n_free+1) AB_i evals, (n_free+1) BA_i evals, one B
    # eval, per replicate, no groups used here) -- the trailing AB/BA
    # entry each is theta's own, always bit-identical to A's/B's own
    # fun value since theta never reaches the simulator (see
    # _find_infeasible_ab_keys).
    step = 2 * (n_free + 1) + 2

    x_norm_parts: list[np.ndarray] = []
    theta_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    n_collected = 0
    n_discarded = 0
    n_attempted = 0
    max_attempted = _INFEASIBLE_RESAMPLE_ATTEMPT_FACTOR * n
    round_seed = seed
    ab_counts: dict[str, int] = {k: 0 for k in calib.free_keys}
    n_base_point_infeasible = 0

    t0 = time.monotonic()
    # See sample_morris_sensitivity's identical pool-sharing comment --
    # same reasoning, one pool spans every replacement round here too.
    pool_ctx = contextlib.nullcontext(pool) if pool is not None else concurrent.futures.ProcessPoolExecutor()
    with pool_ctx as active_pool:
        while n_collected < n:
            shortfall = n - n_collected
            x_norm_batch = sobol_sample.sample(problem, N=shortfall, calc_second_order=True, seed=round_seed)
            # theta is always the trailing column (see problem's own
            # comment) -- the simulator only ever sees the real n_free
            # columns; theta only perturbs y AFTER evaluation (_tie_breaker).
            x_norm_x_batch = x_norm_batch[:, :n_free]
            theta_batch = x_norm_batch[:, n_free]
            x_real_batch = lo + x_norm_x_batch * (hi - lo)
            y_batch = _evaluate_parallel(x_real_batch, calib, active_pool, should_cancel)
            n_attempted += shortfall

            infeasible = y_batch >= _FEASIBLE_FUN_CUTOFF_MPS
            x_norm_3d = x_norm_x_batch.reshape(shortfall, step, n_free)
            theta_2d = theta_batch.reshape(shortfall, step)
            y_2d = y_batch.reshape(shortfall, step)
            bad_rep = infeasible.reshape(shortfall, step).any(axis=1)
            n_bad = int(bad_rep.sum())
            n_discarded += n_bad

            # See sample_morris_sensitivity's identical attribution
            # comment -- same reasoning, _find_infeasible_ab_keys' own
            # docstring covers what this can and can't pin down for a
            # Sobol' replicate specifically.
            for ri in np.where(bad_rep)[0]:
                implicated, a_bad, b_bad = _find_infeasible_ab_keys(y_2d[ri], calib.free_keys)
                for k in implicated:
                    ab_counts[k] += 1
                n_base_point_infeasible += int(a_bad) + int(b_bad)

            keep = ~bad_rep
            if keep.any():
                x_norm_parts.append(x_norm_3d[keep].reshape(-1, n_free))
                theta_parts.append(theta_2d[keep].reshape(-1))
                y_parts.append(y_2d[keep].reshape(-1))
            n_collected += shortfall - n_bad

            if n_collected < n and n_attempted >= max_attempted:
                raise ValueError(
                    f"sample_sobol_sensitivity: gave up after {n_attempted} attempted "
                    f"replicates ({n_discarded} hit infeasible physics, fun >= "
                    f"{_FEASIBLE_FUN_CUTOFF_MPS}) -- only {n_collected}/{n} usable. "
                    f"This free_keys/fixed_overrides combination is overwhelmingly "
                    f"infeasible within the swept bounds; investigate the "
                    f"fixed_overrides (e.g. an already-extreme spinbox value) or "
                    f"narrow free_keys' bounds rather than trusting a result built "
                    f"mostly from penalty values."
                )
            round_seed += 1

    if n_discarded:
        ab_breakdown = ", ".join(f"{k}={v}" for k, v in ab_counts.items() if v) or "none"
        logger.warning(
            "sample_sobol_sensitivity: %d infeasible-physics replicate%s "
            "discarded and replaced (fun >= %.1f) -- attempted %d total to "
            "reach N=%d usable. AB-point implications: %s%s.",
            n_discarded, "" if n_discarded == 1 else "s", _FEASIBLE_FUN_CUTOFF_MPS,
            n_attempted, n, ab_breakdown,
            f" ({n_base_point_infeasible} base A/B point(s) infeasible)" if n_base_point_infeasible else "",
        )
    elapsed = time.monotonic() - t0

    x_norm_used = np.concatenate(x_norm_parts, axis=0)
    x_real_used = lo + x_norm_used * (hi - lo)
    y_used = np.concatenate(y_parts, axis=0)
    theta_used = np.concatenate(theta_parts, axis=0)
    # real_y_std vs. _TIE_BREAKER_AMPLITUDE is exactly the "was this run
    # near-degenerate" comparison (see _tie_breaker's docstring) -- not
    # surfaced in the UI (see core.calibrator's own "IV. Pre-Auto-
    # Fit sensitivity screening" section docstring for why), but worth
    # being able to check after the fact from a log without recomputing
    # anything: real_y_std << _TIE_BREAKER_AMPLITUDE means S1/ST for
    # every free_key are describing the tie-breaker's own near-total
    # share of Y'(X, theta)'s variance, not a genuine "this parameter
    # doesn't matter" finding.
    real_y_std = float(np.std(y_used))
    logger.debug(
        "sample_sobol_sensitivity: n_free=%d N=%d n_evals=%d elapsed=%.1fs "
        "real_y_std=%.3g tie_breaker_amplitude=%.3g",
        n_free, n, n_attempted * step, elapsed, real_y_std, _TIE_BREAKER_AMPLITUDE,
    )

    y_prime = y_used + _tie_breaker(theta_used)
    si = sobol_analyze.analyze(problem, y_prime, calc_second_order=True, seed=seed)

    # SALib's S2/S2_conf are (n_free, n_free) with only the i<j upper
    # triangle populated (NaN elsewhere) -- flattened to a plain dict
    # keyed by the unordered (key_a, key_b) pair (key_a < key_b), one
    # entry per finite pairing (see SobolSensitivityTrials.s2's own
    # docstring for who reads this).
    s2: dict[tuple[str, str], float] = {}
    s2_conf: dict[tuple[str, str], float] = {}
    for i in range(n_free):
        for j in range(i + 1, n_free):
            val = float(si["S2"][i, j])
            if np.isfinite(val):
                key_a, key_b = calib.free_keys[i], calib.free_keys[j]
                pair = (key_a, key_b) if key_a < key_b else (key_b, key_a)
                s2[pair] = val
                s2_conf[pair] = float(si["S2_conf"][i, j])

    result = SobolSensitivityTrials(
        s1={k: float(v) for k, v in zip(calib.free_keys, si["S1"])},
        st={k: float(v) for k, v in zip(calib.free_keys, si["ST"])},
        s1_conf={k: float(v) for k, v in zip(calib.free_keys, si["S1_conf"])},
        st_conf={k: float(v) for k, v in zip(calib.free_keys, si["ST_conf"])},
        s2=s2,
        s2_conf=s2_conf,
        sample_x={k: x_real_used[:, i].copy() for i, k in enumerate(calib.free_keys)},
        sample_y=y_used.copy(),
        n_replicates_requested=n,
        n_replicates_discarded=n_discarded,
        infeasible_ab_counts=ab_counts,
        n_base_point_infeasible=n_base_point_infeasible,
    )

    calibration_cache.save_cached_sensitivity(cache_key, result, calibration_cache.DEFAULT_SENSITIVITY_CACHE_DIR)

    return result


# ---------------------------------------------------------------------------
# V. Result + top-level entry point
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Final calibration output, ready for the Analyzer UI to display and
    add as a normal Rebuild (via the existing _add_rebuild path).

    Attributes:
        free_keys:      Calibrated parameter names, same order as
                        x_best.
        x_best:         Calibrated values.
        rmse_mps:       Velocity RMSE [m/s] at x_best (see module
                        docstring's "Objective function" section — not a
                        ΔTime RMSE).
        physics_overrides: Full overrides dict (fixed_overrides merged
                        with the calibrated free values) — pass this
                        straight to eidos.apps.analyzer's Scenario /
                        _add_rebuild, same shape as a manual Rebuild's
                        physics_overrides.
        n_trials:       Number of independent DE->NM trials that actually
                        returned a result (trials whose process raised
                        are excluded — see calibrate_multistart). May be
                        less than calculate_num_trials(len(free_keys)).
        n_converged:    Of those n_trials, how many had their own DE
                        stage (.de_success) stop because it satisfied its
                        own tolerance, not because it hit maxiter. Also
                        requires .success when the NM polish actually
                        improved on DE (polish_applied=True) -- but when
                        the polish regressed, calibrate_single_trial
                        discards it and final.success is just DE's own
                        de_result.success again (the same value as
                        .de_success), so this does not independently
                        confirm the NM stage converged in that case.
                        n_converged < n_trials does not necessarily mean
                        the result is wrong, but it does mean at least
                        one trial gave up at an iteration cap rather than
                        settling -- treat x_best with more caution than
                        usual if so.
        x_std:          Per free_key, the std of that parameter's value
                        across all_trials' .x (one value per independent
                        trial -- every trial here is equally DE+NM-
                        polished, so this pool is representative of the
                        full trial count, not just a subset). Large x_std
                        relative to that parameter's bounds range,
                        alongside a small spread in .fun across the same
                        trials, is the signature of a flat/degenerate
                        objective landscape: trials agree on how good the
                        fit is but not on which parameter values produced
                        it. NOT itself a sign calibrate() is broken; it
                        means "this particular combination of free_keys
                        can't be pinned down by this data alone", which is
                        information about the calibration setup, not a
                        bug.
        all_trials:     Every successful independent trial's
                        OptimizeResult (including the de_raw_rmse_mps/
                        de_success/polish_applied attributes
                        calibrate_single_trial attaches — see that
                        function's docstring), for further diagnostic
                        inspection (see eidos.lib.calibration_diagnostics).
        from_cache:     True if this result was loaded from
                        core.calibration_cache instead of freshly
                        computed. Every field the cache key actually hashes is
                        identical either way (a cache hit is only ever
                        returned for a byte-for-byte equivalent run over
                        those fields -- see that module's docstring);
                        this is purely informational; e.g. for a caller to
                        note "(cached)" somewhere in the UI. physics_
                        overrides is the one exception -- see calibrate()'s
                        cache-hit branch for why it's rebuilt fresh rather
                        than reused as-is.
        cached_at:      ISO 8601 timestamp of when this result was
                        originally computed -- set unconditionally by
                        calibrate(), not only on a cache hit; a hit's
                        value is whenever the ORIGINAL run finished, not
                        when this particular read happened.
    """
    free_keys: list[str]
    x_best: np.ndarray
    rmse_mps: float
    physics_overrides: dict
    n_trials: int
    n_converged: int
    x_std: dict
    all_trials: list = field(default_factory=list)
    from_cache: bool = False
    cached_at: str | None = None


def _source_fingerprint(simulator_key: str) -> str:
    """
    Hash the source of every module calibrate()'s (and sample_morris_
    sensitivity's/sample_sobol_sensitivity's) actual computation
    touches: this module itself, core.schema (bounds_from_field/
    PhysicsParams etc.), core.activity_parser (build_zoh_power_blocks/
    _make_activity_record's gps_speed_ms), core.physics_overrides
    (build_overridden_params, which delegates v_limit/wind-geometry
    recomputation to simulator_spec.recompute_course_physics -- that
    physics lives inside each simulator's own module, e.g. core.
    simulators.sim_kiritsubo's compute_course_physics_sim_kiritsubo/
    recompute_course_physics_sim_kiritsubo, so it's already covered by
    the dynamic simulator-module hash below; core.course_geometry is
    hashed anyway since fit_course_geometry_profile is still a real,
    shared computation dependency), and whichever simulator module
    simulator_key resolves to -- resolved dynamically, not hand-listed,
    so a future new simulator is covered automatically.

    ANY change to any of these -- even a comment -- invalidates every
    cache entry; that is the intended, safe-by-default failure
    direction. A missed real behavior change silently serving a stale
    result would be a much worse failure than an occasional unnecessary
    recompute.

    Lives here, not in core.calibration_cache, specifically so that
    module never needs to import this one back: core.calibration_cache
    stays a fully generic key/cache utility with no knowledge of what
    it's caching or why -- calibrate()/sample_morris_sensitivity/
    sample_sobol_sensitivity each compute this once and pass it in as a
    plain string, the same way they already pass in every other
    run-config scalar core.calibration_cache's own functions don't
    otherwise know how to derive.

    core.course_geometry is imported locally here, not at this module's
    own top level, since it's needed only for this fingerprint --
    calibrate_single_trial's ProcessPoolExecutor workers re-import this
    whole module to unpickle it (see this module's own docstring) and
    never call this function, so a top-level import would cost every
    worker something it never uses.

    Args:
        simulator_key: core.simulators.SIMULATOR_REGISTRY key to resolve
                    which simulator module to hash.

    Returns:
        A hex digest string.
    """
    import hashlib
    import inspect
    import sys

    import core.activity_parser
    import core.course_geometry
    import core.physics_overrides
    import core.schema

    h = hashlib.sha256()
    for module in (
        sys.modules[__name__], core.schema, core.activity_parser,
        core.course_geometry, core.physics_overrides,
    ):
        h.update(inspect.getsource(module).encode())

    simulator_spec = resolve_simulator(simulator_key)
    kernel_module = inspect.getmodule(simulator_spec.kernel)
    if kernel_module is not None:
        h.update(inspect.getsource(kernel_module).encode())

    return h.hexdigest()


def calibrate(
    course_distance_m: float,
    simulator_key: str,
    base_physics: PhysicsParams,
    raw_physical: dict,
    raw_physiological: dict,
    raw_run: dict,
    course_profile: CourseProfile,
    fixed_overrides: dict,
    activity_raw: ActivityRecord,
    free_keys: list[str],
    progress_callback=None,
    should_cancel=None,
) -> CalibrationResult:
    """
    Top-level calibration entry point: build inputs, run multistart
    DE->NM, package the result as a ready-to-apply overrides dict.

    Total independent DE->NM trials = calculate_num_trials(len(free_keys))
    — see calibrate_multistart / calibrate_single_trial. Trial count is
    (AUTO_FIT_N_SEEDS_FACTOR*len(free_keys)+1)*SEED_MULTIPLIER and scales
    linearly with len(free_keys).

    No sensitivity sampling happens here: the Morris/Sobol' screening in
    this module's section IV (sample_morris_sensitivity/
    sample_sobol_sensitivity) is a separate, caller-driven step meant to
    run BEFORE this function, on whichever free_keys subset the caller is
    still deciding on -- see TTAnalyzerWindow's inline Sobol'/Morris
    sensitivity bars. Auto Fit itself only ever runs DE->NM multistart.

    Args:
        course_distance_m: Strategy's total road distance [m].
        simulator_key: core.simulators.SIMULATOR_REGISTRY key to
                    calibrate against -- normally the same simulator that
                    produced the strategy being calibrated (e.g.
                    eidos.apps.analyzer passes its StrategyRecord's own
                    resolved simulator_spec.key).
        base_physics, raw_physical, raw_physiological, raw_run, course_profile:
                    Strategy baseline, passed through to
                    build_overridden_params on every evaluation.
        fixed_overrides: Overrides dict held constant across the run
                    (e.g. the Analyzer's spinbox state at Auto Fit
                    click time).
        activity_raw: Raw ActivityRecord for the FIT ride to calibrate
                    against.
        free_keys:  Subset of core.simulators.calibratable_physical_keys(
                    simulator_key) to calibrate.
        progress_callback, should_cancel: See calibrate_multistart.

    Every call checks core.calibration_cache before running anything and
    returns a cached CalibrationResult on a hit (result.from_cache=True)
    instead of recomputing, then saves a fresh result to the cache after
    a miss -- no caller has ever needed to bypass this, so it is not a
    parameter. A hit is only ever returned for a byte-for-byte-equivalent
    run -- see core.calibration_cache's module docstring for exactly what
    "equivalent" means (every CalibrationInputs field, the seed/trial-
    count args above, and a source-code fingerprint of everything the
    computation touches). Always cached under core.calibration_cache's
    own DEFAULT_CACHE_DIR -- no caller has ever needed a different
    location.

    Returns:
        CalibrationResult with the best trial's parameters, velocity RMSE,
        convergence diagnostics, and a ready-to-use physics_overrides
        dict. from_cache/cached_at indicate whether this came from cache
        -- see CalibrationResult's docstring.

    Raises:
        ValueError: If free_keys is empty.
        CalibrationCancelled: If should_cancel() fired mid-run.
    """
    if not free_keys:
        raise ValueError("free_keys is empty — nothing to calibrate.")

    calib = build_calibration_inputs(
        course_distance_m, simulator_key, base_physics, raw_physical, raw_physiological, raw_run,
        course_profile, fixed_overrides, activity_raw, free_keys,
    )

    # Local import: core.calibration_cache is main-process-only
    # tooling (pickle/hashlib/inspect over an already-built
    # CalibrationInputs) that every ProcessPoolExecutor worker would
    # otherwise re-import for nothing.
    from core import calibration_cache

    # initial_base_seed=0, seed_factor=AUTO_FIT_N_SEEDS_FACTOR,
    # seed_multiplier=SEED_MULTIPLIER: fixed -- see calibrate_multistart/
    # calculate_num_trials, which hardcode the same.
    cache_key = calibration_cache.compute_cache_key(
        calib, 0, AUTO_FIT_N_SEEDS_FACTOR, SEED_MULTIPLIER,
        code_fingerprint=_source_fingerprint(simulator_key),
    )
    cached = calibration_cache.load_cached_result(cache_key, calibration_cache.DEFAULT_CACHE_DIR)
    if cached is not None:
        # compute_cache_key deliberately excludes W' Balance-only
        # fixed_overrides entries from the hash (they never reach
        # the ODE this fit optimizes -- see that function's
        # docstring), so a cache hit can legitimately come from a
        # PREVIOUS call whose fixed_overrides held different W'
        # Balance values than THIS call's. Only free_keys' own
        # optimized values (x_best) are actually cache-worthy;
        # physics_overrides itself is rebuilt fresh from THIS call's
        # fixed_overrides, same shape as the non-cached branch below
        # -- otherwise a cache hit would silently re-serve a stale W'
        # Balance value the user just edited (e.g. CP) straight back
        # into the new Rebuild.
        cached.physics_overrides = {
            **fixed_overrides, **dict(zip(cached.free_keys, cached.x_best))
        }
        return cached

    best, all_trials = calibrate_multistart(
        calib, progress_callback=progress_callback, should_cancel=should_cancel,
    )

    physics_overrides = {**fixed_overrides, **dict(zip(free_keys, best.x))}

    n_converged = sum(
        1 for t in all_trials
        if getattr(t, "success", False) and getattr(t, "de_success", False)
    )
    x_matrix = np.array([t.x for t in all_trials])  # shape (n_trials, n_free)
    x_std = {k: float(np.std(x_matrix[:, i])) for i, k in enumerate(free_keys)}

    result = CalibrationResult(
        free_keys=list(free_keys),
        x_best=np.asarray(best.x),
        rmse_mps=float(best.fun),
        physics_overrides=physics_overrides,
        n_trials=len(all_trials),
        n_converged=n_converged,
        x_std=x_std,
        all_trials=all_trials,
        cached_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )

    calibration_cache.save_cached_result(cache_key, result, calibration_cache.DEFAULT_CACHE_DIR)

    return result