# Architecture

This is a living reference to the design rules this codebase actually
follows today — not a history of how it got here. This document does
not track that history (what changed, when, why, in what order); it
describes only the current state. This document is meant to be updated
whenever a rule described here changes.

Read this when deciding where new code belongs or whether a change
would contradict an existing design rule — not as a first read for
using the app. Each section below maps to a specific decision point
(e.g. "should this new module live in `core/`?", "is this mypy error
new or known noise?"); jump to whichever one matches the decision in
front of you rather than reading start to end.

## EIDOS/HYLE symmetry

The project name, `(EIDOS/HYLE)^TT`, splits along the same line the
words themselves do — eidos (form) and hyle (matter) — on purpose:
EIDOS^TT (`src/eidos/`) is the strategy-computation side (Generator,
Viewer, Designer, Exporter, Trainer, Navigator, Analyzer, Manager —
all Qt/PySide6 GUIs); HYLE^TT (`src/hyle/`) is the supporting-logistics
side (course_checker, cpmodel_estimator, fit2gpx_converter,
fit_combiner, strategy_doctor — standalone tools, three of them
browser-UI-based, two CLI-only).

The one rule this symmetry actually enforces: **`hyle/` never imports
from `eidos/`**, in either direction of "which one depends on which."
Neither is the "real" side with the other bolted on.

`core/` (below) is the one place both sides are allowed to depend on.
When a HYLE^TT tool turns out to need functionality that currently
lives in `eidos/lib/` — as `hyle.apps.fit2gpx_converter`'s
altitude-offset refinement did, needing `core.calibrator`'s Auto Fit
search to fit real physics parameters against a ride rather than
trusting generic defaults — the fix is moving that code into `core/`,
not adding an exception to this rule.

## What belongs in `core/`

`core/` holds modules genuinely used by *both* `eidos/` and `hyle/` —
not "reusable in principle," but actually imported by both today.
Current members and why each qualifies:

- `schema.py` — Pydantic data structures (`RunSettings`, `PhysicsParams`,
  `CourseProfile`, etc.) that both sides construct and pass around.
  `PhysicalSettings`/`PhysiologicalSettings` are *not* here — each
  `core.simulators.SIMULATOR_REGISTRY` entry defines its own (see
  "Reproducibility tracking" below and `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md`).
  `EngineSettings`/`EngineValidationModel` (simulator/optimizer registry
  selection) are *not* here either, despite the name matching `RunSettings`'
  pattern — `optimizer`/`optimizer_params` are an EIDOS-only concept (HYLE
  never selects an optimizer), so they live in `eidos/lib/optimizer.py`
  instead; keeping them here would mean `core/` importing from `eidos/lib/`.
- `course_geometry.py` — course-shape preprocessing shared across every
  simulator version (GPX B-spline fitting, curvature, apparent-wind
  trigonometry). Speed-limit physics (braking/cornering) live
  per-simulator instead, alongside that simulator's other physics — see
  `SimulatorSpec.compute_course_physics` in
  `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md`. Generated once by EIDOS^TT,
  read back into HYLE^TT tooling. (The physics kernel itself lives under
  `core/simulators/`, not here — see below.)
- `simulators/` — the physics-kernel registry (`SIMULATOR_REGISTRY`,
  `resolve_simulator`, `resolve_physical_params`/
  `resolve_physiological_params`) plus each registered Numba-accelerated
  kernel implementation (`sim_kiritsubo.py`, `sim_stub.py`). Both sides
  resolve a kernel through this registry: EIDOS^TT for generation and
  every re-simulation consumer (Designer, Trainer, Viewer, Exporter,
  Analyzer), HYLE^TT's `fit2gpx_converter` for its own physics-based
  altitude-offset refinement (via `core.calibrator`, below). See
  `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md` for the registry shape and
  the `SimulatorSpec` contract.
- `data_manager.py` — GPX loading, strategy JSON I/O, course-profile
  compression.
- `io_config.py` — path/directory configuration (`resources/`, `temp/`,
  env-var overrides) both sides resolve against.
- `activity_parser.py` — FIT-file parsing, used by EIDOS^TT's Analyzer
  and HYLE^TT's fit2gpx_converter (see "FIT parsing" below).
  Establishes the convention `core.calibrator` also follows when
  comparing a real ride against a simulated one: raw measured data is
  never resampled, only the simulated side is — see
  `build_zoh_power_blocks`'s own docstring for the full reasoning
  (including why the FIT file's own odometer field goes unused).
- `pydantic_mapper.py` — flat-record extraction / `_index.parquet`
  schema. `extract_flat_data()`/`ExperimentIndexModel` are consumed by
  both `hyle.apps.strategy_doctor` and EIDOS^TT.
- `git_info.py` — git commit/dirty introspection for the reproducibility
  check (see "Reproducibility tracking" below). Placed directly in
  `core/` since Viewer/Exporter/Trainer (EIDOS^TT) are its consumers
  today, but the mechanism itself isn't EIDOS-specific.
- `fit_combiner.py` — merges a directory of `*.fit` trial files into one
  spec-compliant activity FIT file; used in-process by EIDOS^TT's
  Trainer at the end of a live session and standalone by HYLE^TT's
  fit_combiner tool.
- `calibrator.py` — differential-evolution physics-parameter search
  ("Auto Fit") plus Morris/Sobol' sensitivity screening. Used by
  EIDOS^TT's Analyzer (Auto Fit / Sensitivity columns) and by HYLE^TT's
  fit2gpx_converter, which runs the same search to refine its
  altitude-offset (tau) estimate against a ride's own physics.
- `physics_overrides.py`, `activity_correspondence.py`,
  `calibration_cache.py` — not imported directly by HYLE^TT today, but
  all three are dependencies of `calibrator.py` above, and `core/` may
  not depend on `eidos/lib/` or `hyle/lib/` — a qualifying module's own
  dependencies have to follow it into `core/`, not stay behind and pull
  `core/` back across the boundary. See "Test before adding something
  here" below.
- `logging_setup.py` — shared console logging configuration
  (`configure_logging()`), called once near the top of every
  `eidos.apps.*`/`hyle.apps.*` entry point's `main()` so timestamp/level
  formatting and stdout routing are identical everywhere.

**Test before adding something here**: is it actually imported by code
on both sides right now, or is it a dependency of a module that already
passes that test (`core/` may not import from `eidos/lib/`,
`eidos/apps/`, `hyle/lib/`, or `hyle/apps/`, so a qualifying module's
own dependencies have to move into `core/` with it)? "Might be useful
to hyle someday" is not sufficient — that's exactly the asymmetry the
EIDOS/HYLE split exists to prevent. A module used by only one side,
with no dependents already in `core/`, belongs in that side's `lib/`
(`eidos/lib/` or `hyle/lib/`), even if it looks generically reusable.

## `apps/` package shape

`hyle/apps/*` is always a package, not a flat module — even the
smallest tools (course_checker, fit_combiner, strategy_doctor) — so an
entry point and its `.html` template, when it has one, share one
self-contained directory instead of sitting loose among unrelated
files:

```
some_app/
├── __init__.py       # the actual implementation, exposes main()
├── __main__.py        # thin shim: `from pkg.some_app import main; main()`
└── some_app.html      # only for browser-UI tools (course_checker,
                        # cpmodel_estimator, fit2gpx_converter)
```

`eidos/apps/*` becomes a package only once a single file grows too
large to manage as one — analyzer, viewer, and manager were each split
this way, into models/workers/widgets/window-style files sharing the
layout above, once they passed roughly 1500-2500 lines. designer.py,
exporter.py, generator.py, navigator.py, and trainer.py are all still
comfortably under 900 lines and remain plain, flat `.py` modules —
there's no standing requirement to package a small app just for
consistency with the rest of `eidos/apps/`.

`__main__.py` exists only so `python -m pkg.some_app` works — a
package's own `__init__.py` doesn't run as `__main__` the way a flat
`module.py` does. `pyproject.toml`'s `[project.scripts]` entries all
point at `pkg.some_app:main`, and the command name always matches the
module name exactly (`hyle-cpmodel-estimator` = `hyle.apps.
cpmodel_estimator`, `hyle-strategy-doctor` = `hyle.apps.strategy_doctor`,
etc. — underscores become hyphens, nothing else changes). Keep new
entries consistent with this convention (`tests/test_entry_points.py`'s
parametrized test checks every entry resolves; its
`test_expected_number_of_apps_present` needs its expected count bumped
too) rather than introducing a shortened name.

Browser-UI tools (`hyle.apps.cpmodel_estimator`, `hyle.apps.
fit2gpx_converter`) share a local-HTTP-server pattern —
`hyle.lib.common.QuietHTTPHandler`/`run_local_server` — because the
browser sandbox can't read/write local files on its own; the Python
side serves data and receives writes over `127.0.0.1` on a random free
port, and the page calls `navigator.sendBeacon('/shutdown')` on
`pagehide` to let the process exit once the tab closes.
`hyle.apps.course_checker` doesn't need this (it renders a static,
one-shot HTML file and never needs a second round-trip to Python).
`hyle.apps.strategy_doctor` has no browser UI at all — it's a one-shot
CLI report (`brew doctor`-style), not an interactive tool, so a GUI
would add nothing.

## Strategy JSONs are self-contained

A strategy JSON (`resources/strategies/<strategy_set_dir>/strategy_*.json`)
embeds everything needed to reconstruct its own course geometry
(distance/lat/lon/altitude/kappa/slope/v_limit, computed once at
generation time) rather than referencing the source GPX file by name
only. On disk this data sits zlib+pickle+base64-compressed in
`input.data.course_profile.compressed_packet` (`distance_step` is the
only plain scalar left outside the packet) — see `core.data_manager.
_pack_internal`/`unpack_input_data` for the compress/decompress pair.
This is deliberate: it means an existing strategy's course shape can
never change out from under it, even if the GPX file in
`resources/gpx/` is later edited or replaced. Every consumer that
reconstructs a course (Viewer, Exporter, Trainer, Designer, Analyzer)
decodes this embedded data via `core.data_manager.
unpack_input_data()` followed by `build_course_profile()` — none of
them re-read the raw GPX.

GPX-file versioning is therefore not a concern: it's already solved by
this embedding, for every strategy that already exists.

`_index.parquet` (one per strategy_set_dir) is the opposite: a derived,
disposable speed cache built from the JSON files via
`core.pydantic_mapper.extract_flat_data()`/`ExperimentIndexModel`,
never the source of truth. `eidos.lib.strategy_selector.
load_records_via_index()` reads it for fast Viewer loading;
`hyle.apps.strategy_doctor` can always fully rebuild it from the JSON
files alone (see `docs/RUNBOOK.md`).

## FIT parsing happens once, in Python

`core.activity_parser.parse_fit_file()` is the single FIT-parsing
implementation, used by both `eidos.apps.analyzer` (three-stage course
extraction from a real ride) and `hyle.apps.fit2gpx_converter`.
fit2gpx_converter parses once in Python at server startup and serves
the parsed record stream to the browser as JSON (`/parsed.json`); the
page is a pure display/trim-selection front end, with no independent
FIT parser of its own. If a future HYLE tool needs FIT data, it should
call `core.activity_parser.parse_fit_file()` rather than adding a
second parser anywhere, in any language.

## Qt mixin composition: the `_selftype` pattern

When a Qt/PySide6 widget is composed from mixins (to avoid the C++
diamond-inheritance issues a conventional multiple-inheritance widget
hierarchy would hit) and a mixin needs to reference attributes/methods
that only exist on the *composed* class, mypy cannot resolve that
dependency through a normal `TYPE_CHECKING`-only import of the
composite class if the composite module already imports the mixins at
module level — that creates a genuine import cycle. The fix: a
dependency-free stand-in class declaring exactly the attributes/methods
a mixin needs to assume exist on `self`, imported only under
`TYPE_CHECKING`:

```python
if TYPE_CHECKING:
    from eidos.lib.power_profile_canvas._canvas_selftype import _CanvasSelfType
    _DrawMixinBase = _CanvasSelfType
else:
    _DrawMixinBase = object

class _PowerProfileDrawMixin(_DrawMixinBase):
    ...
```

At runtime this is just `class _PowerProfileDrawMixin(object)` — zero
behavior change. mypy, type-checking a mixin file in isolation, sees
the stand-in class's declared attributes and can check `self.records`,
`self.request_refresh()`, etc. without needing to resolve the real
composite class.

**Example**: `eidos.lib.power_profile_canvas.PowerProfileCanvas` is
composed from three mixins (`_PowerProfileDataMixin`,
`_PowerProfileDrawMixin`, `_PowerProfileLayoutMixin`) this way; its
stand-in class is `power_profile_canvas/_canvas_selftype.py`'s
`_CanvasSelfType`. Reach for this same pattern again if another
mixin-based widget hits the same cycle — don't reintroduce a
`TYPE_CHECKING` import of the composite class directly.

## Reproducibility tracking (version management)

Three separate, deliberately non-overlapping mechanisms exist around
"what code produced this strategy":

**1. Hand-maintained milestone labels** — `DATA_MANAGER_VERSION`
(`core/data_manager.py`), `SIMULATOR_VERSION` (one per
`core/simulators/*.py` implementation), `OPTIMIZER_VERSION` (one per
`eidos/lib/optimizers/*.py` implementation -- the registry itself,
`eidos/lib/optimizer.py`, has no version of its own). Human-readable,
free-form (e.g. a future `v1.3.0-SCALE-Isolated` naming an algorithm
variant). Their only job is being a readable label — they are *not*
meant to precisely track "did this file's logic actually change," and
nothing in the codebase relies on them for that. Bumping them is
enforced by a pre-commit hook (`scripts/check_code_version_bump.sh`,
wired in via `.pre-commit-config.yaml` -- see `docs/RUNBOOK.md`), not
by these files being self-verifying.

**2. `core.git_info`** — precise, fully automatic. Every strategy JSON
records `input.git_state.commit_hash` (the whole repo's HEAD at
generation time) and `is_dirty` (whether any registered simulator's own
kernel file under `core/simulators/`, `core/data_manager.py`, or
`core/schema.py` specifically — see `core.git_info.
REPRODUCIBILITY_RELEVANT_PATHS` for the literal current list, kept
honest against every registered simulator by `tests/
test_registry_consistency.py` — had uncommitted changes at that
moment). Note `core/course_geometry.py` (the geometric preprocessing shared
across simulator versions -- GPX course fitting, curvature, apparent-wind
trigonometry; speed-limit physics live per-simulator, see above) is
*not* in this list: `build_course_profile` decodes a
strategy's course geometry from what's already stored in the JSON rather
than recomputing it, so that file has no bearing on re-simulating an
already-decided strategy. This exists because `eidos.apps.viewer`
(rebuilding a strategy's plot data), `eidos.apps.exporter` (FIT/ZWO
export), and `eidos.apps.trainer` (live "physics-identical" replay) all
re-run `core.simulators.sim_kiritsubo.simulate_power_profile_separated_blocks`
against a strategy's *stored* strategy data using whatever
`core/simulators/sim_kiritsubo.py` is running *today* — if it's changed since
generation, the freshly re-simulated values can silently diverge from
the strategy's own stored KPIs (`total_time_s` etc., which are never
recomputed).
`check_reproducibility()` compares stored vs. current state and
returns a warning string (or `None`) for these three call sites to
print/log. It's deliberately scoped to just those three files — a
commit that only touches `hyle.apps.cpmodel_estimator` or `README.md`
must not trigger a warning, and `eidos/lib/optimizer.py` /
`eidos/lib/optimizers/*.py` are deliberately *not* in scope either,
because none of the three re-simulation call sites ever invoke the
optimizer (it only runs while originally *searching for* a strategy).
`eidos.apps.designer` and `core.calibrator` are not consumers of this
check: both intentionally re-simulate with today's code (a new
strategy variant; a physics fit against real ride data), not a
reproduction of an existing strategy.

**Why PhysicalSettings/PhysiologicalSettings live where they do.**
Each `SIMULATOR_REGISTRY` entry defines its own `PhysicalSettings`/
`PhysiologicalSettings` Pydantic models (e.g.
`core/simulators/sim_kiritsubo.py`, `core/simulators/sim_stub.py`), not
a single global pair in `core/schema.py`. The split is along a causal
axis, not a "who owns the equipment" one: `PhysicalSettings` holds every
value that's causally connected to speed calculation even when power is
driven externally (`is_target_power=False` -- the mode calibration/replay
always runs in); `PhysiologicalSettings` holds only what a kernel's own
physiological power-availability clamp needs, active solely when
`is_target_power=True`, and is never itself a calibration target (see
`core.simulators.calibratable_physical_keys`, which derives calibration
targets from `physical_param_model` alone). See `sim_kiritsubo.
PhysicalSettings`'s own docstring for this verified directly against a
real kernel, not just asserted.

Because both models live inside each simulator's own file, they're
already covered by mechanism 2 above without a separate
`REPRODUCIBILITY_RELEVANT_PATHS` entry: a change to `sim_kiritsubo`'s
`PhysicalSettings` (a new field, a changed bound) is a change to
`core/simulators/sim_kiritsubo.py`, the same file its kernel and
`SIMULATOR_VERSION` already live in and that file is already tracked
under. `core/schema.py` remains tracked in its own right (it still holds
`RunSettings`, `PowerBlocks`, `PhysicsParams`, and the shared
`bounds_from_field`/`field_display_label`/`validate_flat_filename`
helpers every simulator's own models call into), not because it still
owns the physical/physiological split itself.

**Keeping the hand-maintained lists honest.** Both mechanism 1's
`VERSIONED_PAIRS` (in `scripts/check_code_version_bump.sh`) and
mechanism 2's `REPRODUCIBILITY_RELEVANT_PATHS` hardcode file paths
independently of `SIMULATOR_REGISTRY`/`OPTIMIZER_REGISTRY`
(`core/simulators/__init__.py` / `eidos/lib/optimizer.py`) — a new
registry entry whose file is never added to either list would silently
escape both checks (see each list's own comment). `tests/test_registry_consistency.py`,
wired into `.pre-commit-config.yaml` alongside the version-bump check
and ruff, cross-checks the registry (the actual, executable source of
truth for "what implementations exist") against both lists on every
commit, so this specific omission is caught automatically rather than
depending on a human rereading this section at the right moment.

**3. `git worktree`, manual** — the actual escape hatch when a warning
fires and exact fidelity is genuinely needed:
`git worktree add <dir> <commit>` checks out that commit into a
separate directory without disturbing the current working tree, so the
relevant app can be run from there once. Deliberately *not* automated
(no dynamic loading of an old `core/simulators/sim_kiritsubo.py` at runtime) — that path
was considered and rejected: it would need `core/schema.py` and
`core/data_manager.py` pulled from the same old commit too (a partial
reload risks a silent schema mismatch, not just a crash), it would
re-trigger Numba's JIT compile cost on every use, and even matching
source doesn't guarantee matching results if numpy/scipy/numba's own
installed versions have since changed. Full details and the reasoning
trail: `core/git_info.py`'s module docstring.

## Testing philosophy

Two kinds of automated tests exist, deliberately narrow:

- **Entry-point / wiring smoke tests** (`tests/test_entry_points.py`,
  `tests/test_registry_consistency.py`, `tests/
  test_doc_code_references.py`, `tests/test_mypy_known_errors.py`):
  parse a dynamic source of truth (`pyproject.toml`'s `[project.scripts]`
  table; `SIMULATOR_REGISTRY`/`OPTIMIZER_REGISTRY`; the actual repo file
  tree; `mypy src`'s real current output) and assert some other
  hand-maintained list stays consistent with it, rather than comparing
  two hardcoded lists against each other. Catches a real failure mode
  (`test_entry_points.py`: a module gets moved/renamed and some lookup
  table, e.g. `SCRIPT_MODULE_MAP`, keeps pointing at the old target) and
  the same class of bug in
  `scripts/check_code_version_bump.sh`'s `VERSIONED_PAIRS` /
  `core/git_info.py`'s `REPRODUCIBILITY_RELEVANT_PATHS`
  (`test_registry_consistency.py`: a new registry entry never added to
  either list), a `docs/*.md` file-path reference left pointing at
  something renamed or deleted (`test_doc_code_references.py`), and a
  genuinely new mypy error hiding among the pre-triaged known-noise
  baseline (`test_mypy_known_errors.py`) — in every case nothing
  *looks* broken because nothing tried to resolve or cross-check it.
- **Regression tests pinned to a specific documented past bug**
  (`tests/test_manager_labels.py` is the current example — pins
  `SCRIPT_DISPLAY_MAP`'s keys to `GENERATOR_SCRIPT`/`VIEWER_SCRIPT`'s
  own `Path(...).stem`, so the labels can't silently fall back to
  naive title-casing again).

**Deliberately not present**: golden-case regression tests over
`core.simulators`/the optimizer's actual physics/strategy output. For a
system this complex, a golden-case test risks freezing a bug as "correct" and
needs real domain expertise to construct meaningfully; actual operation
surfaces problems faster and more reliably than a battery of frozen
numeric assertions would. Don't reintroduce this class of test without
revisiting that reasoning first.

## Where documentation lives

This document holds generalizable, project-wide design rules — the
kind relevant to a *future* decision (where new code belongs, whether
a change would contradict an existing pattern). One-off detail specific
to a single implementation, verified directly against that
implementation rather than asserted, belongs in that implementation's
own docstring instead — e.g. `sim_kiritsubo.PhysicalSettings`'s own
docstring, not this document, is where "which fields the velocity/
position update can actually reach in that mode" is checked against
that kernel specifically.

Docstrings and code comments in `src/` never reference this document
or any other `docs/*.md` file, at any granularity — not a specific
section, not even the whole document. Seven real references were
found and removed during a 2026-09 documentation audit, across three
distinct failure modes. One pointed at a specific section that was
later deleted with nothing left in its place — fixed by moving the
fact it explained into the relevant docstring instead (see
`core.activity_parser.find_course_matches`). Five more, spread across
`core/course_geometry.py`, `core/simulators/__init__.py`,
`core/simulators/sim_kiritsubo.py`, `eidos/lib/optimizer.py`, and
`eidos/lib/optimizers/opt_tenchi.py`, were each just a trailing
mid-sentence pointer that made its own sentence harder to read without
adding anything the discovery path below doesn't already cover. One
more, in `core/schema.py`, was a reference split across a line-wrapped
comment (`...which docs/` / `ARCHITECTURE.md's "Test before..."`) that
`tests/test_doc_code_references.py`'s original regex missed entirely
until a fresh audit pass caught it by hand — the regex now normalizes
comment line-wraps before matching, specifically because of this case.
Anyone reading code who wants the "how do I add a new
simulator/optimizer" walkthrough already has a path to it —
`CONTRIBUTING.md` points to `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md`
— so a second, code-embedded pointer to the same document is redundant
as well as fragile: a doc section can be renamed or deleted out from
under a reference nobody remembers is there. This document and the
other `docs/*.md` files may still freely reference source/config files
by name; `tests/test_doc_code_references.py`, wired into
`.pre-commit-config.yaml`, enforces both sides of this on every commit
— no `docs/*.md` reference anywhere under `src/`, and every file-path
reference inside `docs/*.md` (and `README.md`/`CONTRIBUTING.md`/
`SECURITY.md`) still resolves to a real file.

## mypy adoption

Gradual, not strict (`pyproject.toml`'s `[tool.mypy]`):
`ignore_missing_imports = true`, no `disallow_untyped_defs`. `mypy src`
reports errors that fall into five known, understood categories, none
worth fixing further. `tests/test_mypy_known_errors.py` (wired into
`.pre-commit-config.yaml`) pins the exact set of known-safe `(file,
error code, message)` tuples and fails on any mypy error outside that
set, the same guard-against-silent-drift approach `tests/
test_registry_consistency.py` uses for the registry lists above — so
unlike a plain count in prose, this specific list of five categories
is kept honest on every commit rather than trusted at face value:

- **PySide6 stub gaps on Qt model/delegate overrides**
  (`eidos/lib/record_delegate.py`, `eidos/lib/record_model.py`) —
  Liskov-substitution complaints because the stubs type
  `QAbstractItemModel`/`QStyledItemDelegate`'s `index`/`parent`
  parameters as `QModelIndex | QPersistentModelIndex` while the working
  code only ever needs `QModelIndex`; missing-attribute complaints on
  `QStyleOptionViewItem`/`QStyleOptionButton`/`QEvent` (`.state`,
  `.rect`, `.palette`, `.pos`) that exist at runtime but aren't in the
  stub.
- **Flattened old-style Qt enum access**
  (`eidos/apps/analyzer/window.py`, `eidos/apps/analyzer/widgets.py`,
  `eidos/apps/designer.py`, `eidos/apps/viewer/window.py`) —
  `QAbstractSpinBox.NoButtons`, `Qt.WaitCursor`,
  `QAbstractItemView.PositionAtCenter`, and
  `QWidget.setGraphicsEffect(None)` (Qt's own valid way to clear an
  effect) are all real at runtime; the stub only exposes the newer
  namespaced form (`QAbstractSpinBox.ButtonSymbols.NoButtons`, etc.),
  or, for `setGraphicsEffect`, declares the parameter as non-Optional.
- **Registry Callable variance** (`core/simulators/__init__.py`,
  `eidos/lib/optimizer.py`, `hyle/apps/fit2gpx_converter/__init__.py`) —
  `SimulatorSpec`/`OptimizerSpec` type their callback
  fields against the common `BaseModel` so any simulator/optimizer can
  register one, but each concrete implementation's own
  `PhysicalSettings`/`TenchiParams`/etc. subclass is naturally narrower
  (and, in fit2gpx_converter, an attribute like
  `cda_yaw_table_filename` only exists on some simulators' subclass) --
  an unavoidable consequence of using a common base type for a family
  of non-uniform concrete types, not a bug.
- **matplotlib stub gaps** (`eidos/lib/calibration_diagnostics.py`) —
  `FigureCanvasBase.get_renderer` (present on the real
  Qt/Agg backend, not the stub's base class) and `LayoutEngine.set`'s
  `rect` keyword (only accepted by the `ConstrainedLayoutEngine`
  subclass this code actually constructs, not the generic stub
  signature).
- **`logging.Logger.banner` monkeypatch** (`core/logging_setup.py`) —
  a custom `BANNER` log level patched onto the stdlib
  `Logger` class at import time (see that file's own comment); mypy's
  stdlib stub has no way to know about a runtime-added method.

Any *new* mypy error outside these five known categories is real and
should be fixed or explicitly triaged, not assumed to be more of the
same noise.
