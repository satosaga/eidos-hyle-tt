"""
eidos.lib.power_profile_canvas._canvas_selftype -- TYPE_CHECKING-only
stand-in for PowerProfileCanvas, used solely to type `self` inside the
three mixins in this package.

Background: PowerProfileCanvas (canvas.py) is composed from three mixins
(_PowerProfileDataMixin, _PowerProfileLayoutMixin, _PowerProfileDrawMixin)
that each call methods and read attributes defined on *one of the other
mixins* or on PowerProfileCanvas.__init__ itself -- see canvas.py's
docstring for why the mixins can't just inherit QWidget (and each other's
concrete classes) directly. mypy can't see across mixins on its own, so
without help every such call/attribute access looks like "has no
attribute X" even though it's correct once the three mixins and
PowerProfileCanvas are combined.

Importing the real PowerProfileCanvas from canvas.py into the mixins
would create an actual import cycle (canvas.py imports all three mixins
at module level to build the class), which mypy can't resolve even under
`if TYPE_CHECKING:`. This module exists purely to break that cycle: it
declares the union of attributes/methods the mixins borrow from each
other, with no import of canvas.py or the mixins themselves, so it can be
imported by any of them without creating a cycle.

Not part of any real class's MRO. Only ever referenced inside
`if TYPE_CHECKING:` blocks; has zero effect at runtime.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRect
from PySide6.QtWidgets import QWidget

from eidos.lib.power_profile_canvas.plot_limits import PlotLimits


class _CanvasSelfType(QWidget):
    # --- Class-level constants (declared on PowerProfileCanvas itself) ---
    MARGIN_LEFT: int
    MARGIN_RIGHT: int
    MARGIN_TOP: int
    MARGIN_BOTTOM: int
    PLOT_GAP: int
    LEGEND_DEFINITIONS: Dict[str, Dict[str, Any]]

    # --- Instance attributes (set in canvas.py.__init__ or by a mixin) ---
    limits: Dict[str, PlotLimits]
    subplot_rects: Dict[str, QRect]
    is_time_mode: bool
    current_dist: float
    records: Sequence[Dict[str, Any]]
    data_cache: Tuple[Dict[str, np.ndarray], ...]

    # --- Methods defined on _PowerProfileLayoutMixin, used by the others ---
    # (bodies raise rather than `...` -- mypy's empty-body check is
    # inconsistent about which return types tolerate `...`, and this class
    # is never instantiated at runtime anyway; see module docstring.)
    def map_data_to_widget(
        self,
        x_data: float,
        y_data: float,
        plot_rect: QRect,
        limits: PlotLimits,
        use_y2: bool = False,
    ) -> QPointF:
        raise NotImplementedError

    def _get_x_ticks(self, limits: PlotLimits) -> np.ndarray:
        raise NotImplementedError

    def _get_y_ticks(
        self, plot_name: str, limits: PlotLimits
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # --- Methods defined on _PowerProfileDataMixin, used by the others ---
    def _get_plot_arrays(self, data: Dict[str, Any], plot_name: str) -> Any:
        raise NotImplementedError

    def _get_legend_config(self, plot_name: str) -> Any:
        raise NotImplementedError

    # --- Methods defined on _PowerProfileDrawMixin, used by the others ---
    def request_refresh(self) -> None:
        raise NotImplementedError
