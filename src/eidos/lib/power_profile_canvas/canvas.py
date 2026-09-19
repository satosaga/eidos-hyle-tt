"""
eidos.lib.power_profile_canvas.canvas -- PowerProfileCanvas, the composite widget.

PowerProfileCanvas itself is intentionally thin: the class declaration,
its class-level constants (MARGIN_*, LEGEND_DEFINITIONS), and __init__.
All of its actual behavior comes from three mixins, combined via multiple
inheritance:

    - _PowerProfileDataMixin (data_mixin.py) -- data loading, axis limits
    - _PowerProfileLayoutMixin (layout_mixin.py) -- coordinate mapping, geometry
    - _PowerProfileDrawMixin (draw_mixin.py) -- QPainter rendering

None of the three mixins inherit from QWidget themselves (deliberately --
each mixin inheriting QWidget separately would create a real diamond
inheritance conflict with Qt's C++-backed metaclass). They're plain
Python classes that assume they'll end up mixed into a class that also
inherits QWidget, which is exactly what happens below. Because none of
the mixins define their own __init__, `super().__init__(parent)` inside
PowerProfileCanvas.__init__ walks straight through the MRO to
QWidget.__init__ as if the mixins weren't there.

This is fundamentally one enormous class, unlike eidos.apps.analyzer/
viewer's own package splits (which each had several genuinely
independent classes to separate into files) -- so this package splits
via mixins grouped by responsibility instead.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QSizePolicy, QWidget

import eidos.lib.visual_profile as vp
from eidos.lib.power_profile_canvas.data_mixin import _PowerProfileDataMixin
from eidos.lib.power_profile_canvas.draw_mixin import _PowerProfileDrawMixin
from eidos.lib.power_profile_canvas.layout_mixin import _PowerProfileLayoutMixin
from eidos.lib.power_profile_canvas.plot_limits import PlotLimits


class PowerProfileCanvas(_PowerProfileDataMixin, _PowerProfileLayoutMixin, _PowerProfileDrawMixin, QWidget):
    """
    Composite multi-subplot widget that visualises power strategy results.

    Renders five vertically stacked subplots: Course (grade/altitude), Power (W),
    Velocity (km/h), W' balance (anaerobic energy reserve, J), and Cumulative
    (distance or elapsed time). Supports both distance-axis and time-axis modes,
    interactive cursor tracking, and a cached background layer for performance.
    """
    MARGIN_LEFT = 95 
    MARGIN_RIGHT = 60 
    MARGIN_TOP = 20
    MARGIN_BOTTOM = 40 
    PLOT_GAP = 10 
    # --- Legend item definitions (units and Ref. removed) ---
    LEGEND_DEFINITIONS = {
        # (1) Course plot
        'Grade': {'style': Qt.PenStyle.DotLine, 'width': vp.LINE_WIDTH_PROF, 'color': Qt.GlobalColor.black, 'label': 'Grade'},
        'Altitude': {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_PROF, 'color': Qt.GlobalColor.black, 'label': 'Altitude'},
        # (2) Power plot
        'Target Power': {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_TARGET, 'color': Qt.GlobalColor.gray, 'label': 'Target Power'}, 
        'Actual Power': {'style': Qt.PenStyle.DotLine, 'width': vp.LINE_WIDTH_PROF, 'color': Qt.GlobalColor.black, 'label': 'Actual Power'},
        'CP': {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_CP_WPRIME, 'color': Qt.GlobalColor.black, 'label': 'CP'}, # Ref. removed
        # (3) Velocity plot
        'Velocity': {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_PROF, 'color': Qt.GlobalColor.black, 'label': 'Velocity'},
        # (4) WPrime plot
        "W' Balance": {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_PROF, 'color': Qt.GlobalColor.black, 'label': "W' Balance"},
        "W' Max": {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_CP_WPRIME, 'color': Qt.GlobalColor.black, 'label': "W' Max"},
        'Cumulative': {'style': Qt.PenStyle.SolidLine, 'width': vp.LINE_WIDTH_PROF, 'color': Qt.GlobalColor.black, 'label': 'Cumulative'},
    }

    # ------------------------------------------------
    # 1. Initialization
    # ------------------------------------------------
    def __init__(self, model: Optional[QObject], parent: Optional[QWidget] = None):
        # model=None is a valid, deliberate transitional state -- see
        # eidos.apps.designer.DesignerWindow.__init__, which constructs the
        # canvas before its real model exists and assigns the real one
        # immediately after. refresh_from_model() below (via
        # _PowerProfileDataMixin) already handles self.model being None via
        # getattr(self.model, "_records", []).
        super().__init__(parent)
        self.model = model

        self.limits = {
            'Course': PlotLimits(),
            'Power': PlotLimits(),
            'Velocity': PlotLimits(),
            'WPrime': PlotLimits(),
            'Cumulative': PlotLimits()
        }
        # Neither is ever mutated in place, only reassigned wholesale, so
        # read-only Sequence/Tuple (not List) reflects actual usage.
        # self.records is Sequence rather than Tuple specifically because
        # _rebuild_data_cache provisionally sets it to the raw input List
        # before the loop runs (so a mid-loop exception leaves *something*
        # usable rather than stale data), then replaces it with a Tuple
        # once sorting/zipping finishes.
        self.records: Sequence[Dict[str, Any]] = ()
        self.data_cache: Tuple[Dict[str, np.ndarray], ...] = ()
            
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Initialize on first launch
        self._graph_cache = None
        self._cache_dirty = True
        self.cursor_dist = None
        self.is_time_mode = False

        # Build explicit initial dataset
        self.refresh_from_model()

        # Entry point for Designer connection (normally None)
        self.external_painter = None        
        self.external_mouse_handler = None  
        self.subplot_rects = {}             

