"""
eidos.apps.manager.validators -- Small QValidator/QDoubleSpinBox subclasses
used by ConfigurationEditorPane and FilterEditorPane's dynamically-built
forms (see eidos.apps.manager.editor_panes).
"""

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox


class OptionalIntValidator(QValidator):
    """Integer validator that also accepts an empty string (None)."""
    def validate(self, input_str: str, pos: int) -> tuple[QValidator.State, str, int]:
        """Return Acceptable for an empty string or valid integer; Invalid otherwise."""
        if not input_str:
            return (QValidator.State.Acceptable, input_str, pos)
        try:
            if '.' in input_str:
                return (QValidator.State.Invalid, input_str, pos)
            int(input_str)
            return (QValidator.State.Acceptable, input_str, pos)
        except ValueError:
            return (QValidator.State.Invalid, input_str, pos)

class OptionalFloatValidator(QValidator):
    """Float validator that also accepts an empty string (None)."""
    def validate(self, input_str: str, pos: int) -> tuple[QValidator.State, str, int]:
        """Return Acceptable for an empty string or valid float; Invalid otherwise."""
        if not input_str:
            return (QValidator.State.Acceptable, input_str, pos)
        try:
            float(input_str)
            return (QValidator.State.Acceptable, input_str, pos)
        except ValueError:
            return (QValidator.State.Invalid, input_str, pos)

# --------------------------------------------------
# Custom widgets
# --------------------------------------------------
class FixedDoubleSpinBox(QDoubleSpinBox):
    """SpinBox where only typing a value changes it -- buttons, mouse
    wheel, and Up/Down arrow keys are all disabled.

    This form has no live/reactive feedback to a nudged value (unlike,
    say, a real-time preview), so there is no upside to incremental
    step-based interaction over just typing the number, and every one of
    those is also an easy way to silently mutate a value without meaning
    to -- a real risk: a mouse wheel scroll while merely hovering over a
    plain QSpinBox can change its value unnoticed. See eidos.apps.analyzer.
    widgets.NoScrollDoubleSpinBox/NoScrollSpinBox for the same reasoning
    and mechanism applied to Analyzer's own controls -- not shared code
    with this class (Analyzer's own module pulls in core.simulators/
    numba at import time, which eidos.apps.manager deliberately keeps out
    of its own process until a config is actually opened -- see
    ConfigurationEditorPane._get_schema_map's docstring) but the same
    stepBy()-no-op/wheelEvent()-ignore mechanism, independently applied
    here.

    stepBy() is the single method Qt routes ALL step-based interactions
    through (spin arrows, mouse wheel, and Up/Down arrow keys), so
    overriding it as a no-op blocks all three at once. wheelEvent() is
    still overridden separately to ignore (not just no-op) the event, so
    the scroll passes through to the parent scroll area instead of being
    swallowed here. setButtonSymbols(NoButtons) additionally hides the
    now-inert spin arrows outright, rather than leaving a visibly present
    control that silently does nothing when clicked.
    """
    def __init__(self, parent=None):
        """Initialize with unlimited range and step buttons disabled."""
        super().__init__(parent)
        self.setRange(-1e18, 1e18)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

    def stepBy(self, steps):
        pass

    def wheelEvent(self, event):
        """Ignore mouse wheel events to prevent accidental value changes."""
        event.ignore()


class FixedIntSpinBox(QSpinBox):
    """Integer sibling of FixedDoubleSpinBox -- same reasoning, same
    stepBy()-no-op/wheelEvent()-ignore/NoButtons treatment, for
    ConfigurationEditorPane's integer-valued form fields (e.g.
    RunSettings.n_seg_min/n_seg_max/initial_base_seed/seed_factor)."""
    def __init__(self, parent=None):
        """Initialize with step buttons disabled (range set by the caller)."""
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

    def stepBy(self, steps):
        pass

    def wheelEvent(self, event):
        """Ignore mouse wheel events to prevent accidental value changes."""
        event.ignore()
