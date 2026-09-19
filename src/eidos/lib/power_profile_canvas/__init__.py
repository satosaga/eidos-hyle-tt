"""
eidos.lib.power_profile_canvas -- PowerProfileCanvas, the shared strategy-plot widget.

Used by eidos.apps.viewer.window.TTSimulatorViewer (imports just
PowerProfileCanvas) and eidos.apps.designer (StrategyDesignerController,
imports PowerProfileCanvas and DESIGNER_COLOR_HEX).
"""

from eidos.lib.power_profile_canvas.canvas import PowerProfileCanvas
from eidos.lib.power_profile_canvas.plot_limits import (
    ACTIVE_COLOR_HEX,
    DESIGNER_COLOR_HEX,
    PROFILE_KEYS,
    PlotLimits,
)

__all__ = [
    "PowerProfileCanvas",
    "PlotLimits",
    "DESIGNER_COLOR_HEX",
    "ACTIVE_COLOR_HEX",
    "PROFILE_KEYS",
]
