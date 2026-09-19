# Runbook

Task-oriented: what to do when something specific comes up. For "why is
it built this way," see `docs/ARCHITECTURE.md` instead — this document
assumes that context and doesn't re-explain it.

Entries below mix two audiences — riders hitting a runtime symptom
(a missing index, a reproducibility warning, the ANT+ dongle) and
contributors hitting a dev-workflow one (a blocked commit, first-time
setup, tests/lint) — jump to whichever entry matches what you hit
rather than reading start to end.

## Viewer shows fewer strategies than expected, or a strategy_set_dir is silently missing

Cause: `_index.parquet` (a per-strategy_set_dir cache Viewer reads for speed) is
missing, or out of sync with the actual `strategy_*.json` files in that
directory. This can happen if `eidos-generator` was killed mid-run
(it only writes the index after *all* of a run's JSONs are saved), or
if a JSON file was deleted/added by hand, or if `eidos-designer`/
`eidos-viewer`'s own index-update step partially failed.

Fix:

```bash
hyle-strategy-doctor          # report only, no changes made
hyle-strategy-doctor --fix    # rebuilds the index for every strategy_set_dir with a problem
```

`--fix` rebuilds a strategy_set_dir's index from its JSON files directly
(JSON is always the source of truth) and keeps the previous index alongside
as `_index.parquet.bak` before overwriting. Safe to run repeatedly;
re-running after a `--fix` should report no remaining problems for
whatever it just fixed. A JSON file that fails to parse, or parses but
doesn't match the expected schema (e.g. a much older strategy missing a
field), is reported and skipped rather than aborting the rest of that
strategy_set_dir's rebuild — those specific files need manual attention
(fix the JSON by hand, or accept that record just won't be indexed).

## Analyzer fails to match a recorded FIT file to its course

Symptom: the Analyzer can't find any course match for a ride, or logs
candidates being rejected during start/end detection, even though the
ride clearly covers the course.

Cause: `core.activity_parser.find_course_matches` needs real recorded
history well before the lap start and well after the finish, not just
the lap itself — a candidate whose FIT file doesn't extend far enough
on either side is rejected rather than matched approximately:

- Before the start: at least `LEAD_IN_TIME_S` (currently 30s) of
  recording before the actual departure instant. This chains several
  requirements together — the search window for the standing-start
  stillness boundary, the stillness window itself, the departure-time
  fit's own search range, and a GPS-spline fitting margin — see that
  constant's own comment in `core/activity_parser.py` for the exact
  breakdown. In practice a real standing-start TT procedure already
  covers most of this on its own (the rider sits still on the bike,
  held by a holder, for several seconds before the countdown) — the
  part that actually needs care is starting the recording well before
  getting into position, not right as the countdown begins.
- After the finish: at least `TRAIL_OUT_TIME_S` (currently 25s) of
  recording past the finish line.

Fix: nothing to fix after the fact — a FIT file that already lacks this
margin can't be recovered. Re-record with more lead-in/trail-out next
time: start the recording before getting into the start-line holder's
grip, and keep it running for at least half a minute after crossing
the finish.

## Viewer / Exporter / Trainer prints a reproducibility warning

Example: `src/core/simulators/sim_kiritsubo.py, src/core/simulators/sim_stub.py,
src/core/data_manager.py, src/core/schema.py may have changed since this
strategy was generated...`

What it means: the tool re-simulates this strategy's *stored* strategy
data using whichever `core/simulators/sim_kiritsubo.py` is running right now, not
the one that actually produced it. The warning fires when those
specific files (see `docs/ARCHITECTURE.md`'s "Reproducibility
tracking") have changed, or had uncommitted changes, since this strategy
was generated.

What's still trustworthy regardless of the warning: the strategy's own
stored numbers — `total_time_s`, the pacing strategy itself
(`target_power_list`/`target_length_list`) — were saved at generation time
and are never recomputed. Only *supplementary* re-simulated output
(a Viewer graph, an exported FIT/ZWO pacing curve, Trainer's live
replay) is at risk of not exactly matching.

What to do:

- Usually: nothing. The core numbers are fine; treat the re-simulated
  graph/replay as approximate rather than pixel-exact.
- If you need the *exact* original behavior (e.g. debugging a
  suspected regression), check out the exact commit the strategy was
  generated at without disturbing your current work:

  ```bash
  git worktree add /tmp/old_run <commit-from-the-warning>
  cd /tmp/old_run
  # run the relevant app from here, pointed at the same strategy file
  cd -
  git worktree remove /tmp/old_run
  ```

  The commit hash is in the warning message (and in the strategy JSON's
  own `input.git_state.commit_hash`, in full).

## A `git commit` is blocked by "wasn't bumped in this commit"

You're staging a change to a simulator under `core/simulators/`, an
optimizer under `eidos/lib/optimizers/`, or `core/data_manager.py`,
without also touching that file's own `*_VERSION` string in the same
commit — see `docs/ARCHITECTURE.md`'s "Reproducibility tracking" part 1
and each file's own "I. Version" comment for the bump policy.

Two options:

- **Real change** — bump the version string (any change to the value
  is enough to satisfy the hook; there's no required numbering scheme
  beyond "something you'd recognize later").
- **Pure refactor/rename, no behavior change** — skip the check:

  ```bash
  git commit --no-verify
  ```

  `--no-verify` is a git-level flag, not a per-hook one: it skips ALL
  five pre-commit hooks in this same commit (see "First-time setup"
  above), not just this version-bump check. In particular it also
  skips the registry-consistency check
  (`tests/test_registry_consistency.py`) -- the one thing that catches
  a new `SIMULATOR_REGISTRY`/`OPTIMIZER_REGISTRY` entry never added to
  `REPRODUCIBILITY_RELEVANT_PATHS` (see `docs/ARCHITECTURE.md`'s
  "Reproducibility tracking"), which would then silently escape
  reproducibility tracking with no error at commit time. Only reach
  for `--no-verify` when you're confident nothing else staged in the
  same commit needs any of the other four hooks either.

If the hook doesn't seem to be running at all (no error, no block, on
a commit that should trigger it), it's probably not enabled yet in
this checkout — see "First-time setup" below.

## First-time setup (fresh clone, or hooks not yet enabled)

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # enables every hook in .pre-commit-config.yaml
```

`pre-commit run --all-files` runs every hook on demand without
committing. `.pre-commit-config.yaml` wires up five: the version-bump
check (`scripts/check_code_version_bump.sh`, its own header comment
has the full policy), the registry-consistency check
(`tests/test_registry_consistency.py`, run via `pytest` -- it
cross-checks `SIMULATOR_REGISTRY`/`OPTIMIZER_REGISTRY` against the
version-bump check's `VERSIONED_PAIRS` and `core/git_info.py`'s
`REPRODUCIBILITY_RELEVANT_PATHS`, so a new registry entry never added
to either list is caught instead of silently escaping both), the
doc-code reference check and the mypy known-errors check (both
`pytest`-run tests -- see `docs/ARCHITECTURE.md`'s "Where
documentation lives" and "mypy adoption"), and ruff (see "Linting
with ruff" below).

## ANT+ dongle never initializes (`PHYSICAL_NODE_MISSING` in the log)

`eidos-trainer` needs `libusb` (`brew install libusb`) in addition to
the `pyusb`/`openant` Python packages `pip install -e .` already
installs -- `libusb` is a system library, not something pip can
provide. Without it, ANT+ node initialization in
`eidos/lib/ant_receiver.py` fails and the Execution Log shows
`PHYSICAL_NODE_MISSING | Operating in Virtual-Only mode.` This is not
an error state: Trainer still starts normally with the Virtual Power
Meter selected and fully usable via W/S. A successful physical
connection logs a different, more detailed sequence instead (driver
class, serial number, ANT+ version), so the two cases aren't
ambiguous -- if you have a real dongle and expect it to be used but
see `PHYSICAL_NODE_MISSING`, check `brew list libusb`.

## Isolating personal data for screenshots or a clean demo

Manager/Viewer/etc. list `resources/strategies/` by directory name, so
launching them normally shows whatever real projects already exist
there. `EIDOS_TT_RESOURCES_DIR` (`core.io_config`) redirects all of
`resources/` (`gpx/`, `strategies/`, `configs/`, `filters/`,
`activities/`, `exports/`, `cda_yaw_tables/`) to another directory in
one shot, so pointing it at a fresh directory seeded only with the
git-tracked sample templates runs the app against sample data only,
with no real project ever in view:

```bash
mkdir -p ~/eidos_demo_resources
git archive HEAD resources/ | tar -x -C ~/eidos_demo_resources --strip-components=1
export EIDOS_TT_RESOURCES_DIR=~/eidos_demo_resources
eidos-manager
```

Anything written back out (a fresh Generate run, an edited config) lands
under `~/eidos_demo_resources` too, not the real `resources/` — safe to
delete the whole directory afterward. Unset the env var (or open a new
shell) to go back to the real data.

One exception `EIDOS_TT_RESOURCES_DIR` does NOT cover: Auto Fit's result
cache (`core.calibration_cache`) always reads/writes
`~/.cache/eidos_hyle_tt/` regardless of this env var. Running Auto Fit
during a demo session both leaves demo-run entries in that real,
non-demo directory and (the cache key permitting) could serve a result
cached from your own real data. Clear it too if Auto Fit was used:
`rm -rf ~/.cache/eidos_hyle_tt`.

## Running tests / type-checking

```bash
pytest        # tests/ -- entry-point wiring + a few regression tests, fast
mypy src      # gradual-adoption config; known-noise errors are OK,
              # see docs/ARCHITECTURE.md's "mypy adoption" for exactly which
```

Any mypy error *outside* the five documented noise categories there is
real and should be triaged, not assumed to be more of the same.

## Measuring test coverage

```bash
pytest --cov --cov-report=term-missing   # summary in the terminal
pytest --cov --cov-report=html && open htmlcov/index.html   # line-by-line, browsable
```

There's no fail-under and none is planned -- `tests/` is deliberately
narrow (see `docs/ARCHITECTURE.md`'s "Testing philosophy": wiring
smoke tests + a couple of pinned regressions, not a general suite over
`core.simulators`/the optimizer's physics). Use this to see what the
existing tests actually exercise, not as a number to chase upward.

Measured at 18% (32 tests). Almost entirely
import-time coverage from `test_entry_points.py` loading every
`eidos.apps.*`/`hyle.apps.*` module -- not function/branch coverage of
their actual logic. A large drop from 18% on a future run is still
worth a look (e.g. a module that stopped being importable), but 18%
itself is the expected shape given what these tests are for, not a gap
to close.

## Building the Sphinx API reference

```bash
pip install -e ".[docs]"
sphinx-apidoc -f -e -o docs/sphinx/source/api src
sphinx-build -b html docs/sphinx/source docs/sphinx/_build/html
open docs/sphinx/_build/html/index.html   # macOS; or just open it in a browser
```

`docs/sphinx/source/api/*.rst` and `docs/sphinx/_build/` are both
gitignored and regenerated on demand by the two commands above — never
commit them. Only `docs/sphinx/source/conf.py` and `index.rst` are
tracked.

`conf.py` auto-mocks whatever heavy/hardware dependencies (PySide6,
numba, openant, ...) aren't importable in the environment the build
runs in, so it also works in a minimal environment without the full
`env312_arm64` venv. In such an environment expect a handful of
"failed to import module" warnings for `eidos.apps.viewer` and
`eidos.lib.record_model`/`eidos.lib.record_delegate` — those modules do
class-body arithmetic on a PySide6 enum at import time, which only
breaks when PySide6 itself is mocked (see `conf.py`'s comment). Not a
real bug; doesn't happen when PySide6 is genuinely installed.

## Checking docstring coverage

```bash
pip install -e ".[lint]"
interrogate src   # reads [tool.interrogate] in pyproject.toml automatically
interrogate -v src   # per-file breakdown
```

Fails if coverage drops below the floor in `pyproject.toml`'s
`[tool.interrogate]` (`fail-under = 85`) — this is a regression guard,
not a target to hit 100% on. `__init__` methods/modules and magic
methods are excluded from the count (this codebase documents most
classes at the class docstring instead).

## Linting with ruff

```bash
pip install -e ".[lint]"
ruff check src tests
ruff check --fix src tests   # applies safe/mechanical fixes (mostly import sorting)
```

`[tool.ruff.lint]` in `pyproject.toml` intentionally selects a small,
high-signal rule set for now (pyflakes real-bug checks + import
sorting), not ruff's full catalogue — same gradual-adoption approach as
`[tool.mypy]`/`[tool.interrogate]`. `E701`/`E702` (multiple statements
per line) are ignored project-wide because they're pervasive existing
style spread across 16 files (134 occurrences total; `designer.py`/
`view_opengl.py` are the two biggest at 50/15, but together still
under half of all occurrences), not scattered bugs; `F403`/`F405`
(star imports) are ignored per-file for
`trainer.py`/`view_opengl.py` because `from OpenGL.GL import *` is
PyOpenGL's own conventional usage. Broader rule categories (bugbear,
pyupgrade, simplify, ...) can be turned on incrementally later, the
same way mypy's strictness is being tightened package by package.

## Adding a new `eidos.apps.*` or `hyle.apps.*` entry point

For `hyle.apps.*`: always a package (see `docs/ARCHITECTURE.md`'s
"`apps/` package shape") — a directory with `__init__.py`
(implementation + `main()`), `__main__.py` (thin `python -m` shim), and
an `.html` template only if it's a browser-UI tool.

For `eidos.apps.*`: start as a single flat `.py` module, the same
shape as `designer.py`/`exporter.py`/`generator.py`/`navigator.py`/
`trainer.py` — only split into the package shape above once it grows
too large to manage as one file (see the same ARCHITECTURE.md section
for where `analyzer`/`viewer`/`manager` drew that line).

Either way, add the `[project.scripts]` entry in `pyproject.toml` with
the command name matching the module name exactly (underscores →
hyphens, nothing shortened or reworded) — `tests/test_entry_points.py`'s
`test_expected_number_of_apps_present` will need its expected count
updated to match the new total.

## Adding a new simulator or optimizer registry entry

For the full walkthrough — the `SimulatorSpec`/`OptimizerSpec`
field-by-field shape, a `sim_stub.py`/`opt_stub.py`-based
copy-paste template, and how to verify a new implementation before
committing it — see `docs/ADDING_A_SIMULATOR_OR_OPTIMIZER.md`. What
follows here is the quick checklist.

Simulator variants live under `core/simulators/`, optimizer variants
under `eidos/lib/optimizers/` — one module per registry entry, following
`sim_kiritsubo.py`/`sim_stub.py` (simulators) or `opt_tenchi.py`/`opt_stub.py`
(optimizers): a hand-maintained `SIMULATOR_VERSION`/`OPTIMIZER_VERSION`
string, the `@njit` kernel or search-strategy entry points, and a
builder function. Four things need to happen, in order:

1. Write the implementation file itself.
2. Register it in `core.simulators.SIMULATOR_REGISTRY`
   (`core/simulators/__init__.py`) or
   `eidos.lib.optimizer.OPTIMIZER_REGISTRY` (`eidos/lib/optimizer.py`).
3. Add the new file to `scripts/check_code_version_bump.sh`'s
   `VERSIONED_PAIRS` (both simulators and optimizers go here).
4. Simulators only: also add the new file to `core/git_info.py`'s
   `REPRODUCIBILITY_RELEVANT_PATHS` (optimizers are deliberately
   excluded from this one — see `docs/ARCHITECTURE.md`'s
   "Reproducibility tracking").

Steps 3 and 4 used to be easy to forget silently — both lists say so in
their own comments. `tests/test_registry_consistency.py` now catches a
forgotten step 3/4 automatically, as a pre-commit hook: run
`pytest tests/test_registry_consistency.py` (or just try to commit) and
it will name exactly which file is missing from which list.

Nothing further is needed for config JSON: `eidos.lib.optimizer.EngineValidationModel`
checks a config's `Engine.simulator`/`Engine.optimizer` against the live
registries directly, not a separate hardcoded list, so a new registry
key is usable in a config the moment step 2 is done.

Naming a new entry: not a version number. These registries hold
parallel, independently comparable variants, not a linear v1→v2
succession, so baking a version number into the identity slot both
implies a false hierarchy and is guaranteed to drift from that file's
own free-form `SIMULATOR_VERSION`/`OPTIMIZER_VERSION` label as it gets
bumped. Pick the next unused codename in sequence instead: simulators follow the
opening chapters of Genji Monogatari (kiritsubo, hahakigi, utsusemi,
...), optimizers follow the first poems of the Ogura Hyakunin Isshu
(tenchi, jito, hitomaro, akahito, ...). A structural test-fixture entry
(architecture-verification only, not a real variant -- see
`core.simulators.sim_stub`/`eidos.lib.optimizers.opt_stub`'s own module
docstrings) stays out of the codename pool entirely and keeps the
plain `sim_stub`/`opt_stub` name instead, so it's never mistaken for a
peer variant.

First-commit note: since the implementation file is new, its entire
diff (including the `*_VERSION` line) is staged automatically, so the
version-bump hook passes without extra effort on that first commit — a
deliberate bump is only required on later commits that change that
file's logic.

