import os
import sys

import numpy as np

# Division by zero, 0/0, inf-inf etc. must fail loudly, not silently
# produce nan/inf that then propagates unnoticed through downstream
# calculations. Set here (core is imported by essentially every
# eidos/hyle app) so it takes effect before any app-level numpy work runs.
np.seterr(divide="raise", invalid="raise")


def _fail_fast_excepthook(exc_type, exc_value, exc_tb):
    """Print the traceback the normal way, then kill the process outright.

    PySide6 does not propagate an exception raised inside a Qt slot back
    through app.exec() -- it invokes sys.excepthook and then resumes the
    event loop, so the default excepthook alone leaves the app running in
    a possibly inconsistent state with only a printed traceback to show
    for it. os._exit() rather than sys.exit(): raising SystemExit from
    here doesn't unwind through Qt's C++ event loop the way it would
    through an ordinary Python call stack, and does not stop app.exec()
    either -- only a hard process exit is reliable across every entry
    point (PySide6 GUIs, plain scripts).

    os._exit() skips the normal interpreter shutdown, which includes
    flushing buffered stdout/stderr -- without an explicit flush here,
    output written just before the crash (the very moment it matters
    most) can be silently lost, e.g. under Manager's Execution Log,
    which captures a subprocess's stdout/stderr rather than an
    interactive terminal.
    """
    sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _fail_fast_excepthook

# Two independent real numbers coinciding exactly (e.g. two distances
# both being 0, or a Sobol' RMSE(X) landing on the same value twice) has
# probability 0 in the real-number model this code is simulating -- but
# float64 quantizes that continuum, so the exact-tie event these models
# assign probability 0 is reachable in practice (a frozen/repeated GPS
# fix, a degenerate optimizer input, etc.), not hypothetical. The
# principled fix is never an `if` branch special-casing the tie -- that
# bakes in a hidden, arbitrary interpretive choice about what "should"
# happen at the tie (see core.calibrator's own Sobol' tie-breaker
# for the fuller version of this argument). Instead add this as a fixed,
# deterministic perturbation directly into the computation so the
# degenerate case is just the continuous limit of the same formula
# everyone else uses, never a separate code path.
FLOAT_TIE_BREAKER_EPS = 1e-9
