"""
Guards the mypy-noise triage in docs/ARCHITECTURE.md's "mypy adoption"
section: every error mypy currently reports on `src/` must be one of
the specific, pre-triaged (file, code, message) tuples below. An error
mypy reports that is NOT in this baseline is either a real new bug or
mypy's message wording drifted (e.g. after upgrading the mypy version
pinned in pyproject.toml's `dev` extra) -- either way it needs a human
to look at it and update this baseline deliberately, not silently pass
or silently start failing every future commit.

Nothing else runs mypy in pre-commit or CI (see ARCHITECTURE.md's own
note on this) -- this test is that guard. It intentionally does NOT
assert an exact count or an exact set (baseline entries no longer
reported are fine -- that just means something got fixed); it only
fails on an error that ISN'T in the baseline.

KNOWN_MYPY_ERRORS groups into the five categories docs/ARCHITECTURE.md
describes in prose (PySide6 stub gaps on Qt model/delegate overrides;
flattened old-style Qt enum access; registry Callable variance;
matplotlib stub gaps; the logging.Logger.banner monkeypatch) -- see
that section for why each is known-safe rather than repeating the
reasoning here.

Requires the project's real runtime dependencies (mypy included, via
the `dev` extra) installed in the active venv:

    pip install -e ".[dev]"
    pytest tests/test_mypy_known_errors.py

To regenerate this baseline after a deliberate change (a real fix, a
mypy version bump, new code that legitimately hits the same known
stub/override gaps), run `mypy src`, inspect any new/changed lines by
hand to confirm they are still one of the five known categories, and
update KNOWN_MYPY_ERRORS to match.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_ERROR_LINE = re.compile(r"^(?P<file>[^:]+):\d+: error: (?P<message>.+?)\s*\[(?P<code>[a-z-]+)\]\s*$")

# (repo-relative file, mypy error code, message text with file/line/code
# stripped) -- see docs/ARCHITECTURE.md's "mypy adoption" for the five
# categories these group into.
KNOWN_MYPY_ERRORS: set[tuple[str, str, str]] = {
    # PySide6 stub gaps on Qt model/delegate overrides
    ("src/eidos/lib/record_delegate.py", "attr-defined", '"QEvent" has no attribute "pos"'),
    ("src/eidos/lib/record_delegate.py", "attr-defined", '"QStyleOptionButton" has no attribute "rect"'),
    ("src/eidos/lib/record_delegate.py", "attr-defined", '"QStyleOptionButton" has no attribute "state"'),
    ("src/eidos/lib/record_delegate.py", "attr-defined", '"QStyleOptionViewItem" has no attribute "palette"'),
    ("src/eidos/lib/record_delegate.py", "attr-defined", '"QStyleOptionViewItem" has no attribute "rect"'),
    ("src/eidos/lib/record_delegate.py", "attr-defined", '"QStyleOptionViewItem" has no attribute "state"'),
    ("src/eidos/lib/record_delegate.py", "override", 'Argument 2 of "sizeHint" is incompatible with supertype "PySide6.QtWidgets.QAbstractItemDelegate"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_delegate.py", "override", 'Argument 2 of "sizeHint" is incompatible with supertype "PySide6.QtWidgets.QStyledItemDelegate"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_delegate.py", "override", 'Argument 3 of "paint" is incompatible with supertype "PySide6.QtWidgets.QAbstractItemDelegate"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_delegate.py", "override", 'Argument 3 of "paint" is incompatible with supertype "PySide6.QtWidgets.QStyledItemDelegate"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_delegate.py", "override", 'Argument 4 of "editorEvent" is incompatible with supertype "PySide6.QtWidgets.QAbstractItemDelegate"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_delegate.py", "override", 'Argument 4 of "editorEvent" is incompatible with supertype "PySide6.QtWidgets.QStyledItemDelegate"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Argument 1 of "columnCount" is incompatible with supertype "PySide6.QtCore.QAbstractItemModel"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Argument 1 of "data" is incompatible with supertype "PySide6.QtCore.QAbstractItemModel"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Argument 1 of "flags" is incompatible with supertype "PySide6.QtCore.QAbstractItemModel"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Argument 1 of "flags" is incompatible with supertype "PySide6.QtCore.QAbstractListModel"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Argument 1 of "rowCount" is incompatible with supertype "PySide6.QtCore.QAbstractItemModel"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Argument 1 of "setData" is incompatible with supertype "PySide6.QtCore.QAbstractItemModel"; supertype defines the argument type as "QModelIndex | QPersistentModelIndex"'),
    ("src/eidos/lib/record_model.py", "override", 'Signature of "columnCount" incompatible with supertype "PySide6.QtCore.QAbstractListModel"'),

    # Flattened old-style Qt enum access
    ("src/eidos/apps/analyzer/window.py", "attr-defined", '"type[QAbstractSpinBox]" has no attribute "NoButtons"'),
    ("src/eidos/apps/analyzer/widgets.py", "attr-defined", '"type[QAbstractSpinBox]" has no attribute "NoButtons"'),
    ("src/eidos/apps/analyzer/widgets.py", "arg-type", 'Argument 1 to "setGraphicsEffect" of "QWidget" has incompatible type "None"; expected "QGraphicsEffect"'),
    ("src/eidos/apps/designer.py", "attr-defined", '"type[Qt]" has no attribute "WaitCursor"'),
    ("src/eidos/apps/viewer/window.py", "attr-defined", '"type[QAbstractItemView]" has no attribute "PositionAtCenter"'),

    # Registry Callable variance
    ("src/core/simulators/__init__.py", "arg-type", 'Argument "build_physics_params" to "SimulatorSpec" has incompatible type "Callable[[PhysicalSettings, PhysiologicalSettings, RunSettings, CourseProfile], DummyPhysicsParams]"; expected "Callable[[BaseModel, BaseModel, RunSettings, CourseProfile], Any]"'),
    ("src/core/simulators/__init__.py", "arg-type", 'Argument "build_physics_params" to "SimulatorSpec" has incompatible type "Callable[[PhysicalSettings, PhysiologicalSettings, RunSettings, CourseProfile], PhysicsParams]"; expected "Callable[[BaseModel, BaseModel, RunSettings, CourseProfile], Any]"'),
    ("src/core/simulators/__init__.py", "arg-type", 'Argument "compute_course_physics" to "SimulatorSpec" has incompatible type "Callable[[CoursePoints, PhysicalSettings, RunSettings], CourseProfile]"; expected "Callable[[CoursePoints, BaseModel, RunSettings], CourseProfile]"'),
    ("src/core/simulators/__init__.py", "arg-type", 'Argument "recompute_course_physics" to "SimulatorSpec" has incompatible type "Callable[[CourseProfile, PhysicalSettings], CourseProfile]"; expected "Callable[[CourseProfile, BaseModel], CourseProfile]"'),
    ("src/eidos/lib/optimizer.py", "arg-type", 'Argument "decode" to "OptimizerSpec" has incompatible type "Callable[[ndarray[tuple[Any, ...], dtype[Any]], int, float, float, OptStubParams], PowerBlocks]"; expected "Callable[[ndarray[tuple[Any, ...], dtype[Any]], int, float, float, BaseModel], PowerBlocks]"'),
    ("src/eidos/lib/optimizer.py", "arg-type", 'Argument "decode" to "OptimizerSpec" has incompatible type "Callable[[ndarray[tuple[Any, ...], dtype[Any]], int, float, float, TenchiParams], PowerBlocks]"; expected "Callable[[ndarray[tuple[Any, ...], dtype[Any]], int, float, float, BaseModel], PowerBlocks]"'),
    ("src/hyle/apps/fit2gpx_converter/__init__.py", "attr-defined", '"BaseModel" has no attribute "cda_yaw_table_filename"'),

    # matplotlib stub gaps
    ("src/eidos/lib/calibration_diagnostics.py", "attr-defined", '"FigureCanvasBase" has no attribute "get_renderer"'),
    ("src/eidos/lib/calibration_diagnostics.py", "call-arg", 'Unexpected keyword argument "rect" for "set" of "LayoutEngine"'),

    # logging.Logger.banner monkeypatch
    ("src/core/logging_setup.py", "attr-defined", '"Logger" has no attribute "banner"'),
    ("src/core/logging_setup.py", "attr-defined", '"type[Logger]" has no attribute "banner"'),
}


def _run_mypy() -> list[tuple[str, str, str]]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    errors = []
    for line in result.stdout.splitlines():
        m = _ERROR_LINE.match(line)
        if m:
            errors.append((m.group("file"), m.group("code"), m.group("message")))
    return errors


def test_every_mypy_error_is_a_known_triaged_one():
    unknown = sorted(set(_run_mypy()) - KNOWN_MYPY_ERRORS)
    assert not unknown, (
        "mypy src reported error(s) not in KNOWN_MYPY_ERRORS -- either a "
        "real new bug (fix it) or mypy's message wording changed, e.g. "
        "after a mypy version bump (update the baseline to match, after "
        "confirming by eye these are still the same known-safe gaps):\n"
        + "\n".join(f"{f}: [{c}] {m}" for f, c, m in unknown)
    )
