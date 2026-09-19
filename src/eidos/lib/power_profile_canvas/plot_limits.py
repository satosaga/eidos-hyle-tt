"""
eidos.lib.power_profile_canvas.plot_limits -- Small shared constants/types.

PlotLimits (axis-limit container used by all five PowerProfileCanvas
subplots) and the module-level color/key constants. Split out first
since these have no dependency on PowerProfileCanvas itself, and the
three mixins (data/layout/draw) and canvas.py all need at least one of
them.
"""

import numpy as np

import eidos.lib.visual_profile as vp

PROFILE_KEYS = vp.PROFILE_KEYS
DESIGNER_COLOR_HEX = "#04ECF8"
ACTIVE_COLOR_HEX = '#00CC00'

class PlotLimits:
    """Container for axis limits (X, Y1, Y2) used by all five subplots."""
    def __init__(self):
        self.X_min, self.X_max = 0.0, 1.0
        self.Y1_min, self.Y1_max = 0.0, 1.0
        self.Y2_min, self.Y2_max = 0.0, 1.0


def nice_ticks(start: float, stop: float, step: float) -> np.ndarray:
    """
    Evenly spaced tick values from start to stop (both inclusive) at
    step intervals -- start itself must already be the tick grid's own
    origin (e.g. math.ceil(axis_min / step) * step), this only fills in
    the rest of the grid up to stop.

    np.arange(start, stop, step) is not used for this: with a
    non-integer stop, its own point count isn't reliable under float
    accumulation (see core's own docstring on this). The count is
    computed once via round() (robust to a tiny float excess/deficit
    right at an integer step count, unlike floor/ceil) and the grid is
    then built from an int arange scaled by step, matching every other
    "n evenly-spaced points" computation in this codebase.

    Returns an empty array if stop < start (a degenerate/empty axis
    range), matching plain arange's own behaviour in that case.
    """
    n_steps = round((stop - start) / step)
    if n_steps < 0:
        return np.array([])
    return start + np.arange(n_steps + 1) * step

