####################
# eidos/lib/optimizers/opt_tenchi.py
####################
"""
opt_tenchi optimizer: joint (power, length) N-stage geometric-dt-
refinement DE -> Nelder-Mead multistart search. Registered in
eidos.lib.optimizer.OPTIMIZER_REGISTRY as "opt_tenchi".

The decision vector is joint (power, length-weight); DE hyperparameters
(de_strategy/de_mutation_*/de_recombination/de_popsize/de_tol/de_maxiter)
are shared identically across every stage. A single trial runs
time_step_refine_stages+1 chained differential_evolution() calls against
the SAME course/physiology, each at a different simulation time
resolution, geometrically stepped from a coarse start down to the real
(fine) dt:

    dt_i = time_step * initl_time_step_scale ** ((N - i) / N)
    for i in [0, N], where N = time_step_refine_stages

so dt_0 = time_step * initl_time_step_scale (the coarsest stage) and
dt_N = time_step exactly (the last stage always runs at the real,
production dt regardless of scale/stage count). Each call's own final
population is handed to the next via scipy's `init=` (accepts a
real-space (popsize*dim, dim) array directly, not just the
'latinhypercube'/'random'/'sobol' keyword strings), continuing the
search rather than resolving it from scratch, and every call reuses the
SAME `seed` value.

Coarse-dt evaluations are cheaper than fine-dt ones, roughly in
proportion to the ratio of their own time_step values, since
sim_kiritsubo's own integration cost scales with how many time steps one
finish_time simulation takes. A coarser dt doesn't relocate where the
optimum roughly is, it only adds numerical imprecision around it, which
each successively finer stage (and the existing Nelder-Mead polish stage
after the last one, unchanged) is positioned to clean up -- so most of a
trial's own cost can happen in the cheap regime, with only progressively
shorter, finer continuations needed to land precisely.

Reusing `seed` across every stage is deliberate, not an oversight:
differential_evolution still uses `seed` to drive every generation's own
mutation/crossover draws even when `init=` supplies an explicit starting
population, so a later call reads from the same position in that seed's
own random stream an earlier call's own initial Latin Hypercube
population draw already read from. That earlier LHS draw and a later
call's mutation/crossover draws are temporally and causally separate
consumers of the same stream (one builds coordinates for a population
that no longer exists as such by the time the later call runs; the
other drives independent per-generation search decisions), so reusing
those same numeric values for an unrelated purpose introduces no
correlation that would bias either call's own outcome. What actually
matters for the multistart search's own statistical validity --
independence ACROSS sub-seeds -- is guaranteed by
optimize_power_and_length_de_wrapper's own sub_seed derivation
(main_seed*1000+i, i in [0, sub_seed_count)), not by anything at this
level.

time_step_refine_stages has a floor of 1, not 0: initl_time_step_scale=
1.0 with time_step_refine_stages=1 already collapses every stage to the
same (real) dt -- the second call's own tol check is satisfied within a
single generation off an already-converged input population -- so a
separate 0 floor for "run fine dt only, once" would just be a second way
to express the same outcome.
"""
import concurrent.futures
import logging
import os
from typing import Literal

import numpy as np
from numba import njit
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import differential_evolution, minimize

from core.logging_setup import configure_logging, log_banner
from core.schema import OptimizationStrategy, PhysicsParams, PowerBlocks
from core.simulators import resolve_simulator
from eidos.lib.optimizers import OPEN_LOWER_BOUND_EPS, SeedResult

logger = logging.getLogger(__name__)

# --------------------------------------------------
# I. Version
# --------------------------------------------------
# Bump for a change worth being able to look back and identify later --
# not for pure refactors/renames. Enforced by the pre-commit framework
# (scripts/check_code_version_bump.sh, see .pre-commit-config.yaml),
# which blocks a commit touching this file unless this line is part of
# the same commit -- use `git commit --no-verify` for a deliberate
# no-bump change.
OPTIMIZER_VERSION = "v2.1.0"

# --- Constants ---
EPSILON_WEIGHT = OPEN_LOWER_BOUND_EPS


class TenchiParams(BaseModel):
    """opt_tenchi's own tunable parameters -- OPTIMIZER_REGISTRY's
    "opt_tenchi" entry's param_model.

    Every field is required (Field(...)), not given a Pydantic default,
    even though each one's own json_schema_extra={"preset": ...} carries
    a reasonable starting value: an optimizer with real tunable
    parameters should always show every one of them for explicit user
    review in eidos.apps.manager's GUI Form Editor (see core.schema.
    preset_value_from_field's own docstring for the full reasoning),
    never let a config JSON quietly omit one and run with a value nobody
    actually looked at. A config JSON's own Engine.optimizer_params must
    therefore always specify every field below when Engine.optimizer is
    "opt_tenchi" -- the GUI itself seeds and highlights each one from its
    own preset the instant this optimizer is selected, so a config built
    through the GUI already satisfies this without the user typing
    anything by hand.

    de_strategy/de_mutation_min/de_mutation_max/de_recombination/
    de_popsize/de_tol/de_maxiter apply to EVERY DE stage identically --
    only the simulation time resolution differs between stages -- see
    this module's own docstring."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    sub_seed_count: int = Field(..., ge=1, title="Sub-seed count", json_schema_extra={"preset": 20}, description="Independent DE populations run per outer seed before keeping the best [-]")
    # The fixed set scipy.optimize.differential_evolution accepts (its
    # _binomial/_exponential strategy dicts) -- an invalid name is rejected
    # at config-validation time rather than surfacing deep inside scipy.
    # eidos.apps.manager's GUI Form Editor also renders any Literal field
    # as a selection dropdown generically (ConfigurationEditorPane.
    # _build_section_group_box).
    de_strategy: Literal[
        "best1bin", "best1exp", "best2bin", "best2exp",
        "currenttobest1bin", "currenttobest1exp",
        "rand1bin", "rand1exp", "rand2bin", "rand2exp",
        "randtobest1bin", "randtobest1exp",
    ] = Field(..., title="DE strategy", json_schema_extra={"preset": "randtobest1bin"}, description="scipy.optimize.differential_evolution mutation strategy (every stage)")
    de_mutation_min: float = Field(..., ge=0.0, le=2.0, title="DE mutation min", json_schema_extra={"preset": 0.1}, description="DE mutation factor range lower bound (every stage) [-]")
    de_mutation_max: float = Field(..., ge=0.0, le=2.0, title="DE mutation max", json_schema_extra={"preset": 1.9}, description="DE mutation factor range upper bound (every stage) [-]")
    de_recombination: float = Field(..., ge=0.0, le=1.0, title="DE recombination", json_schema_extra={"preset": 0.9}, description="DE crossover probability (every stage) [-]")
    de_popsize: int = Field(..., ge=1, title="DE popsize", json_schema_extra={"preset": 5}, description="DE population size multiplier (every stage) [-]")
    de_tol: float = Field(..., gt=0.0, title="DE tol", json_schema_extra={"preset": 0.001, "decimals": 6}, description="DE convergence tolerance (every stage) [-]")
    # Shared across every stage -- there is no separate per-stage budget
    # (a fixed time_step_refine_stages+1 chain of otherwise-identical DE
    # calls, differing only in which dt they run at -- see this module's
    # own docstring).
    de_maxiter: int = Field(..., ge=1, title="DE maxiter", json_schema_extra={"preset": 300}, description="DE maximum generations per stage [-]")
    # >= 1.0, not > 0.0: a multiplier below 1 would make the "coarse"
    # start FINER (more expensive) than the real dt this geometric
    # schedule is supposed to descend to, inverting the whole point of
    # the schedule.
    initl_time_step_scale: float = Field(..., ge=1.0, title="Initl time step scale", json_schema_extra={"preset": 10.0, "decimals": 2}, description="First stage's coarse dt = this x the real (fine) time_step [-]")
    # ge=1, not ge=0: see this module's own docstring -- a 0 floor
    # ("run fine dt only, once") is redundant with
    # initl_time_step_scale=1.0 at this same floor of 1.
    time_step_refine_stages: int = Field(..., ge=1, title="Time step refine stages", json_schema_extra={"preset": 3}, description="Geometric dt-reduction steps from the coarse start down to fine dt -- DE runs this many + 1 times [-]")
    refine_xatol_w: float = Field(..., gt=0.0, title="Refine xatol", json_schema_extra={"unit": "W", "preset": 1.0, "decimals": 3}, description="Nelder-Mead power-simplex convergence tolerance [W]")
    refine_fatol_s: float = Field(..., gt=0.0, title="Refine fatol", json_schema_extra={"unit": "s", "preset": 0.01, "decimals": 4}, description="Nelder-Mead single-run convergence tolerance [s]")
    refine_restart_eps_s: float = Field(..., gt=0.0, title="Refine restart eps", json_schema_extra={"unit": "s", "preset": 0.01 * 1e-3, "decimals": 8}, description="Refine-loop restart-worthwhile threshold [s]")
    refine_max_restarts: int = Field(..., ge=1, title="Refine max restarts", json_schema_extra={"preset": 100}, description="Cap on refine_powers_locally's Nelder-Mead restart loop [-]")


# --------------------------------------------------
# Local power refinement
# --------------------------------------------------
@njit
def objective_powers_only(powers: np.ndarray, fixed_lengths: np.ndarray, physics: PhysicsParams, kernel):
    """Objective function for power-only optimization with fixed segment lengths.

    kernel is the @njit physics kernel to score against (a SimulatorSpec.kernel
    resolved by the caller) -- passed in as a first-class njit function argument
    rather than imported by name, so this scores strategies against whichever
    simulator the config actually selected, not a hardcoded one."""
    power_blocks = PowerBlocks(power=powers, length=fixed_lengths)
    output = kernel(0.0, power_blocks, physics, False, False, True)
    return output.finish_time * output.penalty_factor

def refine_powers_locally(current_x: np.ndarray, n_seg: int, course_distance: float, l_min: float,
                          physics: PhysicsParams, strategy: OptimizationStrategy, kernel, params: TenchiParams):
    """
    Polish power allocation with Nelder-Mead while holding DE-found segment lengths fixed.
    Runs against the real (fine-dt) physics.

    Rather than seeking convergence in a single run, iteratively restarts the simplex
    to break free from n_seg-dimensional interference. Each individual restart typically
    improves on the last by far less than params.refine_fatol_s (a single run's own fatol
    below) -- the real gain comes from accumulating many such small improvements
    across dozens of restarts, not from any one restart resolving a refine_fatol_s-
    sized difference by itself. Terminates only once improvement falls below the
    much finer params.refine_restart_eps_s: stopping at refine_fatol_s itself would
    cut this accumulation off after just 1-2 restarts.
    """
    current_blocks = extract_target_power_and_length(current_x, n_seg, course_distance, l_min)
    fixed_lengths = current_blocks.length
    initial_powers = current_blocks.power

    def objective_unscaled(p_actual):
        """Clip actual powers to [seg_power_min, seg_power_max] and evaluate the finish-time objective.

    Wrapper passed to Nelder-Mead; clipping prevents the simplex from
    exploring physically invalid power values."""
        p_clipped = np.clip(p_actual, strategy.seg_power_min, strategy.seg_power_max)
        return objective_powers_only(p_clipped, fixed_lengths, physics, kernel)

    p_bounds = [(strategy.seg_power_min, strategy.seg_power_max)] * n_seg

    # --- Nelder-Mead iterative restart ---
    current_p = initial_powers
    best_time = float('inf')

    for i in range(params.refine_max_restarts):
        res_refine = minimize(
            fun=objective_unscaled,
            x0=current_p,
            method='Nelder-Mead',
            bounds=p_bounds,
            options={
                'adaptive': True,
                'xatol': params.refine_xatol_w,  # convergence tolerance [W]
                'fatol': params.refine_fatol_s,  # convergence tolerance [s]
            }
        )

        improvement = best_time - res_refine.fun

        if res_refine.fun < best_time:
            best_time = res_refine.fun
            current_p = res_refine.x

        # params.refine_restart_eps_s, not params.refine_fatol_s -- see this
        # function's own docstring for why these need to be different, not shared.
        if i > 0 and improvement < params.refine_restart_eps_s:
            break

    final_p = np.clip(current_p, strategy.seg_power_min, strategy.seg_power_max)
    return PowerBlocks(power=final_p, length=fixed_lengths), best_time

# --------------------------------------------------
# I. Core optimization and wrapper
# --------------------------------------------------
@njit
def extract_target_power_and_length(x_combined: np.ndarray, n_seg: int, course_distance: float, l_min: float):
    """
    Decode the DE solution vector into a PowerBlocks instance.

    The first n_seg elements are target powers [W]. The remaining n_seg elements
    are raw weights whose squared values are normalized to distribute the remaining
    distance (course_distance - n_seg * l_min) proportionally.
    """
    seg_powers = x_combined[:n_seg]
    raw_weights = x_combined[n_seg:]
    raw_weights_2 = np.square(raw_weights)
    sum_w = np.sum(raw_weights_2)
    l_share = course_distance - n_seg * l_min
    seg_lengths = (raw_weights_2 / sum_w) * l_share + l_min
    return PowerBlocks(power=seg_powers, length=seg_lengths)

def decode(x_combined: np.ndarray, n_seg: int, course_distance: float, l_min: float, params: TenchiParams) -> PowerBlocks:
    """OptimizerSpec.decode -- registry-facing wrapper around the @njit
    extract_target_power_and_length. Accepts `params` for OptimizerSpec.decode's
    shared call signature (a future optimizer's decode could plausibly need its
    own tunables), but opt_tenchi's own decoding is pure geometry with nothing
    in TenchiParams affecting it, so it's unused here."""
    return extract_target_power_and_length(x_combined, n_seg, course_distance, l_min)

@njit
def objective_with_length_optimization(x_combined: np.ndarray, n_seg: int, course_distance: float, l_min: float, physics: PhysicsParams, kernel):
    """Objective function for joint power and length optimization.

    kernel: see objective_powers_only's docstring."""
    power_blocks = extract_target_power_and_length(x_combined, n_seg, course_distance, l_min)
    output = kernel(0.0, power_blocks, physics, False, False, True)
    return output.finish_time * output.penalty_factor

def optimize_power_and_length_de_core(seed: int, physics: PhysicsParams, strategy: OptimizationStrategy, course_distance: float, kernel, params: TenchiParams):
    """Run a single time_step_refine_stages+1-stage geometric-dt DE trial
    with the given seed. Returns the LAST stage's own scipy
    OptimizeResult.

    kernel: see objective_powers_only's docstring. workers=1 in every
    stage: scipy never needs to pickle it across a process boundary --
    args stays in-process (outer parallelism already happens one level
    up, across main seeds -- see optimize_power_and_length_de_multistart).

    physics.time_step (a plain PhysicsParams field the sim_kiritsubo
    kernel unpacks directly as its own integration dt) is this chain's
    own "fine dt" endpoint -- physics itself is never mutated, only
    per-stage physics._replace(time_step=...) copies are (a cheap, local
    swap; nothing else -- course geometry, distance_step, physiology --
    depends on it).
    """
    n_seg = strategy.n_seg
    l_min = strategy.seg_length_min
    bounds = ([(strategy.seg_power_min, strategy.seg_power_max)] * n_seg + [(EPSILON_WEIGHT, 1.0)] * n_seg)

    de_kwargs_common = dict(
        bounds=bounds,
        strategy=params.de_strategy,
        mutation=(params.de_mutation_min, params.de_mutation_max),
        recombination=params.de_recombination,
        popsize=params.de_popsize,
        tol=params.de_tol,
        maxiter=params.de_maxiter,
        polish=False,
        seed=seed,
        disp=False,
        updating='immediate',
        workers=1,
    )

    fine_dt = physics.time_step
    n_stages = params.time_step_refine_stages
    res = None
    init_population = None
    # i=0 is the coarsest stage (exponent=1 -> dt=fine_dt*multiplier);
    # i=n_stages is the last (exponent=0 -> dt=fine_dt exactly). No
    # special-case branch for n_stages==1 or for the very first
    # iteration (init_population is None): differential_evolution's own
    # `init=` keyword accepts None-as-"omit" naturally via **kwargs
    # below only being added once a real population exists.
    for i in range(n_stages + 1):
        stage_dt = fine_dt * (params.initl_time_step_scale ** ((n_stages - i) / n_stages))
        physics_stage = physics._replace(time_step=stage_dt)
        stage_kwargs = dict(de_kwargs_common)
        if init_population is not None:
            stage_kwargs['init'] = init_population
        res = differential_evolution(
            func=objective_with_length_optimization,
            args=(n_seg, course_distance, l_min, physics_stage, kernel),
            **stage_kwargs,
        )
        init_population = res.population
    return res

def optimize_power_and_length_de_wrapper(main_seed, physics, strategy, course_distance, simulator_key: str, params: TenchiParams) -> SeedResult:
    """
    Run params.sub_seed_count geometric-dt-refinement DE trials from sub-seeds derived from main_seed,
    keep the best, then polish its power allocation with Nelder-Mead.

    simulator_key is resolved to its @njit kernel fresh in this process (this
    function is what actually runs inside each ProcessPoolExecutor worker --
    see optimize_power_and_length_de_multistart) rather than the kernel object
    itself being passed across the process boundary and pickled.

    Returns a SeedResult -- see that type's own docstring for why this
    doesn't return the raw OptimizeResult (with its de_raw_time/
    refined_power_blocks extra attributes) across the registry boundary.

    configure_logging() is called again here, defensively, because this
    runs inside a ProcessPoolExecutor worker: on the 'spawn' start method
    (macOS/Windows default), a worker is a fresh interpreter that never
    executed generator.py's main() and so never ran configure_logging()
    itself -- without this, the per-seed logger.info() below would hit
    Python logging's unconfigured-root-logger fallback (WARNING+ only)
    and silently vanish instead of reaching the Manager's Execution Log.
    Cheap and idempotent to call again even where it isn't strictly
    needed (e.g. the 'fork' start method, where it already would have
    been inherited).
    """
    configure_logging()
    kernel = resolve_simulator(simulator_key).kernel

    # --- Phase 1: geometric-dt-refinement DE ---
    best_de_res = None
    best_de_obj = np.inf

    for i in range(params.sub_seed_count):
        sub_seed = main_seed * 1000 + i
        res_de = optimize_power_and_length_de_core(sub_seed, physics, strategy, course_distance, kernel, params)
        de_raw_time = float(res_de.fun)

        if de_raw_time < best_de_obj:
            best_de_obj = de_raw_time
            best_de_res = res_de

    # sub_seed_count is Field(ge=1), so the loop above always runs at
    # least once; best_de_res is only ever None here if every sub-seed's
    # objective came back non-finite (nan), which the fail-fast
    # infeasibility penalty elsewhere is designed to prevent.
    assert best_de_res is not None, "DE produced no finite result across sub_seed_count trials"

    # --- Phase 2: Polishing ---
    refined_blocks, refined_obj = refine_powers_locally(
        best_de_res.x, strategy.n_seg, course_distance, strategy.seg_length_min, physics, strategy, kernel, params
    )

    # --- Phase 3: Safety check (keep DE result if polishing regressed) ---
    if refined_obj < best_de_obj:
        final_blocks = refined_blocks
        final_score = refined_obj
    else:
        final_blocks = extract_target_power_and_length(best_de_res.x, strategy.n_seg, course_distance, strategy.seg_length_min)
        final_score = best_de_obj

    logger.info(
        "Seed %d: Optimized:%.3fs -> Refined:%.3fs (Gain: %.4fs)",
        main_seed, best_de_obj, final_score, best_de_obj - final_score,
    )

    # --- Phase 4: Write polished powers back into the result vector ---
    x_final = best_de_res.x.copy()
    x_final[:strategy.n_seg] = final_blocks.power
    return SeedResult(seed=main_seed, x=x_final, success=bool(best_de_res.success))

# --------------------------------------------------
# II. Multi-start entry points
# --------------------------------------------------

def calculate_num_seeds(n_seg: int, seed_factor: int) -> int:
    """Compute the number of independent optimization runs for a given n_seg.

    Internal to this module's own multi-seed loop below -- NOT exposed via
    OPTIMIZER_REGISTRY. eidos.apps.generator.save_experiment_results
    counts len(results) directly rather than re-predicting the count; see
    eidos.lib.optimizer's module docstring appendix."""
    return (seed_factor * (n_seg - 1)) + 1

def optimize_power_and_length_de_multistart(physics: PhysicsParams, strategy: OptimizationStrategy, course_distance: float, initial_base_seed: int, seed_factor: int, simulator_key: str, params: TenchiParams) -> list[SeedResult]:
    """
    Run multi-start geometric-dt-refinement DE optimization in parallel and return every seed's result.

    simulator_key (a core.simulators.SIMULATOR_REGISTRY key) is passed to each
    worker as a plain string, not the resolved kernel itself -- see
    optimize_power_and_length_de_wrapper's docstring for why. params
    (TenchiParams, picklable) is passed to each worker the same way.

    Returns:
        list of SeedResult, one per seed -- no separate "best" return
        value (see eidos.lib.optimizer's module docstring appendix).
    """
    all_seed_results: list[SeedResult] = []

    num_seeds = calculate_num_seeds(strategy.n_seg, seed_factor)
    seeds_list = list(range(initial_base_seed, initial_base_seed + num_seeds))

    logger.info("Starting %d seeds, each with %d sub-seeds", num_seeds, params.sub_seed_count)

    max_workers = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(optimize_power_and_length_de_wrapper, s, physics, strategy, course_distance, simulator_key, params): s
            for s in seeds_list
        }

        for item in concurrent.futures.as_completed(futures):
            current_seed = futures[item]
            try:
                all_seed_results.append(item.result())
            except Exception as e:
                logger.error("Error in Seed %s: %s", current_seed, e)

    return all_seed_results

def sequential_optimize_Nseg(physics: PhysicsParams, course_distance: float, N_seg_min: int, N_seg_max: int, strategy_base: OptimizationStrategy, initial_base_seed: int, seed_factor: int, simulator_key: str, params: TenchiParams) -> dict[int, list[SeedResult]]:
    """
    Run multi-start geometric-dt-refinement DE optimization for each n_seg in [N_seg_min, N_seg_max] sequentially.
    This is OPTIMIZER_REGISTRY["opt_tenchi"].run.

    simulator_key: core.simulators.SIMULATOR_REGISTRY key to search against --
    threaded through to optimize_power_and_length_de_multistart so the DE/
    Nelder-Mead search itself scores strategies against the same simulator
    eidos.apps.generator resolved from the config's Engine section, not a
    hardcoded one.
    params: TenchiParams -- validated eidos.lib.optimizer.EngineSettings.
    optimizer_params, resolved once by the caller and threaded through to
    every worker.

    Returns:
        dict mapping n_seg -> list of SeedResult (every seed's own result,
        not just the best).
    """
    all_seed_results_by_Nseg: dict[int, list[SeedResult]] = {}

    for n_seg in range(N_seg_min, N_seg_max + 1):
        log_banner(logger, f"N_seg = {n_seg}")
        strategy_current = OptimizationStrategy(
            n_seg=n_seg,
            seg_power_min=strategy_base.seg_power_min,
            seg_power_max=strategy_base.seg_power_max,
            seg_length_min=strategy_base.seg_length_min
        )
        all_seed_results_by_Nseg[n_seg] = optimize_power_and_length_de_multistart(
            physics, strategy_current, course_distance, initial_base_seed, seed_factor, simulator_key, params
        )

    return all_seed_results_by_Nseg
