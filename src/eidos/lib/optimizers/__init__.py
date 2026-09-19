####################
# eidos/lib/optimizers/__init__.py
####################
"""
Optimizer implementations, one module per eidos.lib.optimizer.OPTIMIZER_REGISTRY
entry. The registry itself lives in eidos.lib.optimizer, not here -- see that
module's docstring.
"""
from typing import NamedTuple

import numpy as np


class SeedResult(NamedTuple):
    """One seed's search result -- the element type of the dict[int,
    list[SeedResult]] OptimizerSpec.run returns (one list per n_seg).

    Deliberately lean (3 fields) rather than passing scipy.optimize's own
    OptimizeResult around: every OPTIMIZER_REGISTRY caller (today,
    eidos.apps.generator.save_experiment_results) only ever reads .x and
    .success off a raw OptimizeResult -- .fun and any implementation-specific
    extra attributes (e.g. opt_tenchi's de_raw_time/refined_power_blocks)
    never cross the registry boundary. Defined here (eidos.lib.optimizers'
    package init), not in eidos.lib.optimizer alongside OptimizerSpec, so
    that eidos.lib.optimizer (which imports FROM opt_tenchi.py/opt_stub.py
    to build OPTIMIZER_REGISTRY) and opt_tenchi.py/opt_stub.py (which
    construct SeedResult instances) can both import it from a common
    ancestor with no import cycle -- eidos.lib.optimizer re-exports it for
    any caller that wants `eidos.lib.optimizer.SeedResult`.

    Attributes:
        seed:    RNG seed that produced this result.
        x:       Raw solution vector (scipy.optimize's own encoding) --
                 what OptimizerSpec.decode consumes.
        success: Whether the underlying search reported convergence.
    """
    seed: int
    x: np.ndarray
    success: bool

# scipy.optimize.differential_evolution's bounds are always closed ([a, b]),
# with no way to express the open lower bound (a, b] a search variable that
# must logically stay > 0 actually wants. This is the shared stand-in for
# "as close to that open bound as we need" -- not a float-comparison
# precision floor (see calibrator.py's own note on why those aren't shared
# across modules), a different problem with its own shared answer. Not
# promoted to a per-optimizer param_model field: this is a DE
# search-space implementation detail, not a value a user would ever want
# to tune.
OPEN_LOWER_BOUND_EPS = 1e-9
