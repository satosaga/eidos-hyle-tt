####################
# eidos/lib/optimizers/opt_stub.py
####################
"""
opt_stub optimizer: a deliberately simple example
optimizer, NOT tuned or validated for real strategy-planning research --
do not use it to decide an actual pacing plan without first checking
whether skipping the polish stage below costs meaningful solution
quality.

Exists to prove eidos.lib.optimizer.OPTIMIZER_REGISTRY genuinely
supports a second, independently-behaved entry. Deliberately simple:

  - One DE trial per seed -- no inner tier of independent populations
    before keeping the best.
  - No Nelder-Mead local-polish stage at all.
  - Its own param_model (OptStubParams) has no fields at all: this
    stub's handful of DE constants stay hardcoded module constants --
    there's no real-research reason to make them config-tunable for a
    stub that's explicitly not meant for real strategy planning.

Self-contained: every registry entry should stay independently
modifiable without hidden coupling to another entry's internals.
"""
import concurrent.futures
import logging
import os

import numpy as np
from numba import njit
from pydantic import BaseModel, ConfigDict
from scipy.optimize import differential_evolution

from core.logging_setup import configure_logging, log_banner
from core.schema import OptimizationStrategy, PhysicsParams, PowerBlocks
from core.simulators import resolve_simulator
from eidos.lib.optimizers import SeedResult

logger = logging.getLogger(__name__)

# --------------------------------------------------
# I. Version
# --------------------------------------------------
# Bump for a change worth being able to look back and identify later --
# not for pure refactors/renames. Enforced by the pre-commit framework
# (scripts/check_code_version_bump.sh, see .pre-commit-config.yaml),
# which blocks a commit touching this file unless this line is part of
# the same commit -- use `git commit --no-verify` for a deliberate
# no-bump change. Not part of core.git_info's automatic reproducibility
# tracking (the optimizer never runs while re-simulating an already-
# decided strategy -- see that module's docstring); this string's only
# job is a readable milestone/variant label.
OPTIMIZER_VERSION = "de-only-v7"

EPSILON_WEIGHT = 1e-6


class OptStubParams(BaseModel):
    """opt_stub's own tunable parameters -- OPTIMIZER_REGISTRY's "opt_stub"
    entry's param_model. No fields: see this module's own docstring for why
    (a stub that's explicitly not for real research has nothing worth
    exposing as config-tunable)."""
    model_config = ConfigDict(extra="forbid")


@njit
def extract_target_power_and_length(x_combined: np.ndarray, n_seg: int, course_distance: float, l_min: float):
    """Decode the DE solution vector into a PowerBlocks instance: the
    first n_seg elements are target powers, the remaining n_seg are
    length weights."""
    seg_powers = x_combined[:n_seg]
    raw_weights = x_combined[n_seg:]
    raw_weights_2 = np.square(raw_weights)
    sum_w = np.sum(raw_weights_2)
    l_share = course_distance - n_seg * l_min
    seg_lengths = (raw_weights_2 / sum_w) * l_share + l_min
    return PowerBlocks(power=seg_powers, length=seg_lengths)


def decode(x_combined: np.ndarray, n_seg: int, course_distance: float, l_min: float, params: OptStubParams) -> PowerBlocks:
    """OptimizerSpec.decode -- registry-facing wrapper around the @njit
    extract_target_power_and_length. Accepts `params` for OptimizerSpec.decode's
    shared call signature; unused here (OptStubParams has no fields)."""
    return extract_target_power_and_length(x_combined, n_seg, course_distance, l_min)


@njit
def objective_with_length_optimization(x_combined: np.ndarray, n_seg: int, course_distance: float, l_min: float, physics: PhysicsParams, kernel):
    """Objective function for joint power and length optimization.

    kernel is the @njit physics kernel to score against (a SimulatorSpec.kernel
    resolved by the caller) -- passed in as a first-class njit function argument
    rather than imported by name, so this scores strategies against whichever
    simulator the config actually selected, not a hardcoded one."""
    power_blocks = extract_target_power_and_length(x_combined, n_seg, course_distance, l_min)
    output = kernel(0.0, power_blocks, physics, False, False, True)
    return output.finish_time * output.penalty_factor


def optimize_power_and_length_de_core(seed: int, physics: PhysicsParams, strategy: OptimizationStrategy, course_distance: float, kernel):
    """Run a single DE trial with the given seed. Returns the scipy OptimizeResult.

    kernel: see objective_with_length_optimization's docstring. workers=1
    below means scipy never needs to pickle it across a process boundary."""
    n_seg = strategy.n_seg
    l_min = strategy.seg_length_min
    bounds = ([(strategy.seg_power_min, strategy.seg_power_max)] * n_seg + [(EPSILON_WEIGHT, 1.0)] * n_seg)

    return differential_evolution(
        func=objective_with_length_optimization,
        args=(n_seg, course_distance, l_min, physics, kernel),
        bounds=bounds,
        strategy="randtobest1bin",
        mutation=(0.1, 1.9),
        recombination=0.9,
        popsize=5,
        maxiter=300,
        tol=0.001,
        polish=False,
        seed=seed,
        disp=False,
        updating='immediate',
        workers=1,
    )


def calculate_num_seeds(n_seg: int, seed_factor: int) -> int:
    """Compute the number of independent DE trials for a given n_seg -- one
    trial per seed (no sub-seed tier). Internal to this module's own
    multi-seed loop -- NOT exposed via OPTIMIZER_REGISTRY."""
    return (seed_factor * (n_seg - 1)) + 1


def optimize_power_and_length_de_wrapper(main_seed, physics, strategy, course_distance, simulator_key: str) -> SeedResult:
    """One seed's full unit of work: a single DE trial, no polish. simulator_key
    is resolved to its @njit kernel fresh in this process (this function is
    what actually runs inside each ProcessPoolExecutor worker -- see
    optimize_power_and_length_de_multistart) rather than the kernel object
    itself being passed across the process boundary and pickled.

    configure_logging() is called again here, defensively, because this
    runs inside a ProcessPoolExecutor worker: on the 'spawn' start method
    (macOS/Windows default), a worker is a fresh interpreter that never
    executed generator.py's main() and so never ran configure_logging()
    itself -- without this, the logger.info() call below would hit
    Python logging's unconfigured-root-logger fallback (WARNING+ only)
    and this per-seed line would silently vanish instead of reaching the
    Manager's Execution Log. Cheap and idempotent to call again even
    where it isn't strictly needed (e.g. the 'fork' start method, where
    it already would have been inherited).
    """
    configure_logging()
    kernel = resolve_simulator(simulator_key).kernel
    res = optimize_power_and_length_de_core(main_seed, physics, strategy, course_distance, kernel)
    logger.info("Seed %d: DE:%.3fs", main_seed, res.fun)
    return SeedResult(seed=main_seed, x=res.x, success=bool(res.success))


def optimize_power_and_length_de_multistart(physics: PhysicsParams, strategy: OptimizationStrategy, course_distance: float, initial_base_seed: int, seed_factor: int, simulator_key: str) -> list[SeedResult]:
    """
    Run multi-start DE optimization in parallel and return every seed's result.

    Returns:
        list of SeedResult, one per seed -- no separate "best" return
        value (see eidos.lib.optimizer's module docstring appendix).
    """
    all_seed_results: list[SeedResult] = []

    num_seeds = calculate_num_seeds(strategy.n_seg, seed_factor)
    seeds_list = list(range(initial_base_seed, initial_base_seed + num_seeds))

    logger.info("Starting %d seeds", num_seeds)

    max_workers = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(optimize_power_and_length_de_wrapper, s, physics, strategy, course_distance, simulator_key): s
            for s in seeds_list
        }

        for item in concurrent.futures.as_completed(futures):
            current_seed = futures[item]
            try:
                all_seed_results.append(item.result())
            except Exception as e:
                logger.error("Error in Seed %s: %s", current_seed, e)

    return all_seed_results


def sequential_optimize_Nseg(physics: PhysicsParams, course_distance: float, N_seg_min: int, N_seg_max: int, strategy_base: OptimizationStrategy, initial_base_seed: int, seed_factor: int, simulator_key: str, params: OptStubParams) -> dict[int, list[SeedResult]]:
    """
    Run multi-start DE optimization for each n_seg in [N_seg_min, N_seg_max] sequentially.
    This is OPTIMIZER_REGISTRY["opt_stub"].run.

    `params` accepted for OptimizerSpec.run's shared call signature; unused
    (OptStubParams has no fields -- see module docstring).

    Returns:
        dict mapping n_seg -> list of SeedResult (every seed's own result).
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
            physics, strategy_current, course_distance, initial_base_seed, seed_factor, simulator_key
        )

    return all_seed_results_by_Nseg
