# Adding a New Simulator or Optimizer

Audience: a developer (possibly future-you) adding a new physics kernel
under `core/simulators/` or a new search strategy under
`eidos/lib/optimizers/`. This is the how-to. For *why* the registry is
shaped this way — the reproducibility-tracking design, the three
non-overlapping version-tracking mechanisms, the testing philosophy —
see `docs/ARCHITECTURE.md`'s "Reproducibility tracking" and "Testing
philosophy" sections. For troubleshooting an existing warning or
blocked commit, see `docs/RUNBOOK.md`.

## The registry pattern

Two independent registries exist, one per axis:

- `core.simulators.SIMULATOR_REGISTRY` (`core/simulators/__init__.py`)
  — maps a config's `Engine.simulator` key to a `SimulatorSpec`.
- `eidos.lib.optimizer.OPTIMIZER_REGISTRY` (`eidos/lib/optimizer.py`)
  — maps a config's `Engine.optimizer` key to an `OptimizerSpec`.

Both are plain `dict[str, NamedTuple]` — no dynamic discovery, no
plugin loading. Adding an entry means writing a new implementation
module and adding one line to the registry dict.

`SimulatorSpec` bundles six things per entry, not just the physics
kernel itself:

- `key` / `version` — `key` is the registry's own lookup key (what a
  config's `Engine.simulator` names, e.g. `"sim_kiritsubo"`); `version`
  is that file's independent, hand-maintained `SIMULATOR_VERSION`
  string (e.g. `"v1.6.1"`). The two track different things by
  design (see the naming convention below) and are never guaranteed to
  stay in sync with each other — code that needs "which implementation"
  should resolve by `key`, not by parsing `version`.
- `physical_param_model` / `physiological_param_model` — this entry's
  own Pydantic models for the two config sections named
  `PhysicalSettings` / `PhysiologicalSettings` (see "The
  physical/physiological split" below). Bundled with the kernel, not
  shared across entries, because a future kernel is not guaranteed to
  need the same physiological field set `sim_kiritsubo`'s does (see
  `core.simulators.sim_stub.PhysiologicalSettings` for a real example —
  no fields beyond the inherited `cp`/`w_prime`, since its kernel's
  2-parameter model has no Pmax/recovery-rate/vitality-loss-rate
  concept to carry a field for).
- `kernel` — the `@njit` physics function itself.
- `build_physics_params` — a builder that assembles this kernel's own
  parameter `NamedTuple` from `PhysicalSettings` / `PhysiologicalSettings`
  / `RunSettings` / `CourseProfile`. This is bundled with the
  kernel, not shared across entries, because a future kernel is not
  guaranteed to use the same parameter shape as `PhysicsParams` (see
  `core.simulators.sim_stub.DummyPhysicsParams` for a real example of a
  divergent shape — `cp`/`w_prime` are still there (every kernel must
  carry them, see "The physical/physiological split" below), but no
  `p_max`/wind fields).
- `compute_course_physics` — builds a fresh `CourseProfile` from raw
  `CoursePoints` (course-shape fit, plus whatever further
  physics-parameter-dependent stages — speed limits, wind geometry —
  this entry's own kernel needs; see "Course physics" below).
- `recompute_course_physics` — the cheap counterpart: given an
  already-fitted `CourseProfile` and a (possibly overridden)
  `PhysicalSettings`, returns a new one with just the physics-dependent
  fields (`v_limit`/`cos_phi`/`sin_phi`) updated, without repeating the
  expensive geometry fit. Used by Analyzer's manual Rebuild, Auto Fit,
  and FIT/ZWO export re-simulation whenever physical settings differ
  from a strategy's own baseline.

`OptimizerSpec` mirrors this with four things instead of six —
`key`/`version`, `param_model`, `run`, `decode`. There is no `refine`/
`refine_improvement_epsilon` field: `eidos.apps.designer`'s Refine
button uses its own independent Nelder-Mead polish (its module-level
`_refine_powers_locally`), deliberately unrelated to whichever
optimizer, if any, originally produced the strategy being edited — a
per-optimizer `refine` would have to differ arbitrarily between
entries anyway, since editing happens on a *copy* disconnected from
the original search. See `eidos/lib/optimizer.py`'s own docstring and
the `OptimizerSpec` field-by-field comments for what each callable's
call signature must match; that file is the source of truth for the
exact types, not this document.

## Naming convention: not a version number

Pick the next unused codename, not a version string, as the registry
key. These registries hold parallel, independently comparable
variants, not a linear v1→v2 succession — baking a version number into
the identity slot implies a false hierarchy, and is guaranteed to
drift from that file's own free-form
`SIMULATOR_VERSION`/`OPTIMIZER_VERSION` label as it gets bumped
independently.

- Simulators: the opening chapters of *Genji Monogatari* (`kiritsubo`,
  `hahakigi`, `utsusemi`, ...) — `kiritsubo` is the only one used so far.
- Optimizers: the poets of the opening poems in the *Ogura Hyakunin
  Isshu* (`tenchi`, `jito`, `hitomaro`, `akahito`, ...) — `tenchi` is
  the only one used so far.
- A structural test-fixture entry (architecture-verification only, not
  a real variant — see `sim_stub.py`/`opt_stub.py`'s own module
  docstrings) stays out of the codename pool entirely and keeps the
  plain `sim_stub`/`opt_stub` name, so it's never mistaken for a real
  candidate to use in actual research.

## Course physics

Course-shape fitting (B-spline geometry, curvature, heading, altitude —
`core.course_geometry.fit_course_geometry_profile`) is genuinely
simulator-agnostic and **is** shared, since it depends only on the raw
GPX points and `RunSettings`, never on any physical parameter or kernel
choice. Speed-limit physics (braking/cornering limits) and wind
geometry, by contrast, are **not** shared — each simulator implements
its own `compute_course_physics_<name>`/`recompute_course_physics_<name>`,
because they encode a genuine
physics-modeling choice specific to that simulator (e.g.
`sim_kiritsubo`'s `brake_usability` — how much of the theoretical
friction-limited deceleration a rider can actually achieve — is a
modeling assumption a different kernel could reasonably make
differently, or not make at all). Living inside each simulator's own
file also means a change to this physics is covered by that file's own
`SIMULATOR_VERSION` bump policy (`scripts/check_code_version_bump.sh`) —
a separate, unversioned shared module could not guarantee that.

A kernel with no braking-limit or wind model at all (`sim_stub`) simply
never calls the speed-limit/wind-geometry math in its own
`compute_course_physics_sim_stub` — it fills `v_limit`/`cos_phi`/`sin_phi`
with fixed, physics-independent placeholders instead (see that
function), and its own `recompute_course_physics_sim_stub` is
accordingly a no-op (there is nothing to recompute — no `PhysicalSettings`
field feeds either output). If your own kernel genuinely needs
braking-limit physics, `core.simulators.sim_kiritsubo`'s own
`_compute_speed_limits_core`/`_compute_speed_limits_sim_kiritsubo` show
the pattern (an `@njit` core taking plain scalars, plus a thin Python
wrapper that unpacks `PhysicalSettings` and turns a failure sentinel into
a `ValueError`) — copy it into your own module rather than importing it,
the same "each entry stays independently modifiable" precedent the rest
of this document sets. Wind geometry (`core.course_geometry.
_compute_wind_geometry`) IS still shared, since apparent-wind trigonometry
has no simulator-specific modeling choice to make — call it directly if
your kernel has a wind model.

## The physical/physiological split

Every simulator's settings split along a causal axis, not a "who owns
the equipment" one:

- **`PhysicalSettings`** — every value causally connected to speed
  calculation even when power is driven externally
  (`is_target_power=False`, the mode calibration/replay always runs
  in). `PhysicalSettings` shrinks to exactly what YOUR kernel (and your
  own `compute_course_physics_<name>` — see "Course physics" above)
  actually reads, nothing more:
  `sim_stub`'s own `PhysicalSettings` has just 7 fields, not
  `sim_kiritsubo`'s 14, because it has no braking-limit or wind model to
  feed. **One exception**: `cda` and `cda_yaw_table_filename` are always
  a pair, never independently optional — a kernel that models aero drag
  at all resolves it through the same `cda_ratios[yaw_idx]` mechanism
  regardless of whether that resolution is trivial (`sim_stub` always
  evaluates at yaw=0, since it has no wind model, but still loads and
  applies `cda_ratios[0]` — see its own `build_physics_params_sim_stub`).
  A kernel with no aero-drag term at all is the only case entitled to
  have neither field.
  `tests/test_registry_consistency.py::test_every_registered_simulator_pairs_cda_with_its_yaw_table`
  enforces this.
- **`PhysiologicalSettings`** — mostly what your kernel's own
  physiological power-availability clamp needs, active solely when
  `is_target_power=True`, plus two fields required regardless of
  whether your kernel's own physics touches them: **`cp` and `w_prime`
  are mandatory on every entry** (your `PhysiologicalSettings` must
  subclass `core.schema.PhysiologicalSettingsBase`, which defines
  them) — not because every kernel needs them physically, but because
  they have real significance OUTSIDE any one kernel: FIT/ZWO export
  encodes target power as a fraction of CP (there's no meaningful way
  to export without one), and every simulator has a responsibility to
  report a coherent `w_traj` (`core.schema.SimulationOutput.w_traj`),
  even a constant one where there's no real depletion/recovery to
  model. `sim_stub`'s own `PhysiologicalSettings` is a real example of
  the first half of this — it adds no fields beyond the required `cp`/
  `w_prime` (its 2-parameter model has no Pmax/recovery-rate/
  vitality-loss-rate concept to carry a field for) — and its kernel is
  itself a real, if deliberately naive, W'-balance model: `w_traj`
  genuinely depletes whenever power exceeds `cp` and never recovers
  below it; only `cp_eff_traj` is reported constant (at `cp` itself),
  since this kernel has no time-varying effective-CP concept to report
  instead. This is a deliberate design choice made after an earlier
  version left `cp`/`w_prime` genuinely optional and let each
  downstream consumer (export, Designer, Trainer, Analyzer, PDF
  generation) defend against their absence separately — see git
  history on `core/data_manager.py`/`eidos/apps/trainer.py`/
  `eidos/apps/designer.py`/`eidos/apps/analyzer/canvas.py` for the
  scattered fallbacks (and two real crashes) that produced before this
  was pushed up to a single, type-enforced precondition instead.
  Fields beyond `cp`/`w_prime` genuinely can still shrink or reshape
  freely per entry, and are never a calibration target:
  `core.simulators.calibratable_physical_keys` derives calibratable
  keys from `physical_param_model` only.

When in doubt about which side a new field belongs on: would this value
still affect the simulated velocity if you replayed a real recorded
power trace through your kernel with `is_target_power=False`? If yes,
`PhysicalSettings`. If it only matters when the kernel itself decides
how much power is available, `PhysiologicalSettings`. Either way,
`cp`/`w_prime` must be accepted into your own params NamedTuple/builder
and threaded through to the trajectory regardless of whether your
kernel's own physics reads them for anything —
`PhysiologicalSettingsBase` enforces this at the config-model level
(see above), independent of what any one kernel's math actually does
with the values.

## Adding a new simulator

1. Pick your starting file based on how close your new kernel is to
   the existing ones:
   - **A genuinely different or more minimal physics model** (fewer
     phenomena modeled, a different parameter shape) — copy
     `core/simulators/sim_stub.py`. It exists specifically to be a
     minimal, self-contained template: a deliberately simplified
     physics model (rolling resistance + gravity + quadratic aero drag
     only, no wind, a naive non-recovering W' balance) with its own
     `DummyPhysicsParams` NamedTuple and its own minimal
     `PhysiologicalSettings`. Read its module docstring for the full
     list of what it deliberately omits.
   - **A variant that shares most of `sim_kiritsubo`'s physical model**
     (braking, wind, cornering, aero-yaw table) with one piece changed
     — copy `core/simulators/sim_kiritsubo.py` instead. Reconstructing
     its 14 `PhysicalSettings` fields and speed-limit/wind-geometry
     machinery by hand from `sim_stub`'s skeleton, while
     cross-referencing `sim_kiritsubo.py` for every value anyway, is
     more error-prone than starting from a working copy and changing
     only what's different.
2. Define your own `PhysicalSettings`/`PhysiologicalSettings` Pydantic
   models at module level (`model_config = ConfigDict(extra="forbid",
   frozen=True)`, `Field(..., ge=..., le=..., title=..., description=...)`
   per field — see "The physical/physiological split" above for which
   fields go where, and `sim_kiritsubo.py`'s own models for the
   established field set/bounds/units if your kernel shares
   `sim_kiritsubo`'s physical model). **`PhysiologicalSettings` must
   subclass `core.schema.PhysiologicalSettingsBase`**, not plain
   `BaseModel` — this is what makes `cp`/`w_prime` mandatory; skipping it
   is a real error the registry-consistency test below will catch, not a
   style preference. Do not import `PhysicalSettings`/`PhysiologicalSettings`
   themselves from another simulator module even if the shape is
   identical — copy the definition instead, the same precedent
   `opt_stub.py` not importing from `opt_tenchi.py` sets for optimizers.
   `validate_flat_filename` (`core.schema`) is available if a field needs
   a flat-filename check
   like `cda_yaw_table_filename`'s CSV validator (remember the `cda`/
   `cda_yaw_table_filename` pairing rule above if your kernel models aero
   drag at all).
3. Give your kernel function the same six-argument call shape as the
   existing kernels (`v_init`, `PowerBlocks`, your params NamedTuple,
   `use_sync_hook`, `is_target_power`, plus whatever else
   `SimulatorSpec.kernel`'s type hint in `core/simulators/__init__.py`
   currently specifies) — `use_sync_hook` and `is_target_power` must be
   *accepted* for call-signature compatibility even if your kernel
   doesn't meaningfully support them (see `sim_stub.py`'s own comment
   on this); `v_init` (the starting value of v) must be genuinely
   supported by every kernel.
4. Write your own `compute_course_physics_<name>(points, physical, run)`
   and `recompute_course_physics_<name>(course_profile, physical)` (see
   "Course physics" above). The simplest correct version, if your kernel
   has no braking-limit or wind model: `compute_course_physics_<name>`
   calls `core.course_geometry.fit_course_geometry_profile` and fills
   `v_limit`/`cos_phi`/`sin_phi` with fixed placeholders (copy
   `sim_stub.py`'s own `compute_course_physics_sim_stub` and its
   `_UNCONSTRAINED_V_LIMIT_MPS` constant); `recompute_course_physics_<name>`
   is then just `return course_profile` unchanged, the same as
   `sim_stub.py`'s own `recompute_course_physics_sim_stub`. If your
   kernel DOES need real speed-limit physics, copy `sim_kiritsubo.py`'s
   `_compute_speed_limits_core`/`_compute_speed_limits_sim_kiritsubo`
   pattern instead of `sim_stub`'s.
5. Write your own `build_physics_params_<name>(physical, physiological,
   run, course)` builder returning your params NamedTuple — `physical`/
   `physiological` are instances of the models from step 2. If your
   kernel models aero drag, load `physical.cda_yaw_table_filename`
   yourself here (`core.data_manager.load_cda_yaw_table`) — this builder
   is the one place responsible for turning that filename into the
   `cda_ratios` array your kernel actually uses (see `sim_stub.py`'s own
   `build_physics_params_sim_stub` for the pattern, including how it
   applies `cda_ratios[0]` even with no yaw variation to speak of). Do
   not reuse `sim_kiritsubo.py`'s `PhysicsParams` unless your model
   genuinely has the same parameter shape — if it doesn't, define your
   own NamedTuple the way `DummyPhysicsParams` does.
6. Set your own `SIMULATOR_VERSION` string at module level.
7. Register the entry in `SIMULATOR_REGISTRY` in
   `core/simulators/__init__.py`:

   ```python
   "sim_<codename>": SimulatorSpec(
       key="sim_<codename>",
       version=_SIM_<CODENAME>_VERSION,
       physical_param_model=PhysicalSettings,          # imported from your module
       physiological_param_model=PhysiologicalSettings, # imported from your module
       kernel=_simulate_sim_<codename>,
       build_physics_params=build_physics_params_sim_<codename>,
       compute_course_physics=compute_course_physics_sim_<codename>,
       recompute_course_physics=recompute_course_physics_sim_<codename>,
   ),
   ```

## Adding a new optimizer

1. Copy `eidos/lib/optimizers/opt_stub.py` as your starting point.
   Like `sim_stub.py`, it's deliberately simpler than `opt_tenchi.py`
   (a single DE trial per seed, no sub-seed tier, no Nelder-Mead
   polish stage in its own `run()` pipeline) and is self-contained: it
   does not import from `opt_tenchi.py`, and your new module shouldn't
   import from either stub or `opt_tenchi.py` either. Each registry
   entry should stay independently modifiable.
2. Define your own `param_model` — a Pydantic `BaseModel`
   (`model_config = ConfigDict(extra="forbid", frozen=True)`) holding
   whatever your optimizer's own tunable parameters are. If your
   optimizer doesn't have tunables worth exposing (a stub/
   architecture-verification entry), an empty model is fine — see
   `opt_stub.py`'s own `OptStubParams`. If it does, make every field
   required (`Field(...)`, no Pydantic default) with its own
   `json_schema_extra={"preset": <value that reproduces today's
   hardcoded behavior>}` — the same rule `PhysicalSettings`/
   `PhysiologicalSettings` fields already follow: an optimizer with real
   parameters should always show every one of them for explicit user
   review in `eidos.apps.
   manager`'s GUI Form Editor, never let a config JSON quietly omit one
   and run with a value nobody looked at — see `opt_tenchi.py`'s own
   `TenchiParams` and `core.schema.preset_value_from_field`'s docstring
   for the full reasoning. This is what a config JSON's
   `Engine.optimizer_params` validates against (see
   `eidos.lib.optimizer.EngineValidationModel`).
3. Implement the two registry entry points `opt_stub.py` implements —
   `sequential_optimize_Nseg` (top-level multi-`n_seg`/multi-seed
   search — what `eidos.apps.generator` calls to plan a strategy from
   scratch; returns `dict[int, list[SeedResult]]`, one `SeedResult`
   per seed per `n_seg`) and `decode` (a thin wrapper around your own
   solution-vector decoding, taking your validated `param_model`
   instance as its last argument even if unused — see
   `opt_tenchi.decode`'s own docstring for why the argument exists
   regardless). `run`/`decode` must each accept your `param_model` instance
   as their final argument — see `OptimizerSpec`'s own field comments
   in `eidos/lib/optimizer.py` for exactly what each is called with and
   why it must come from the same implementation that produced the
   result being decoded (a future optimizer's own solution-vector
   encoding or seeding scheme is not guaranteed to match
   `opt_tenchi.py`'s). `opt_stub.py` also has its own
   `calculate_num_seeds` — a purely internal helper for its own
   multi-seed loop, not part of the registry contract (see "No
   `count_seeds` registry entry point" below); write your own
   equivalent if your `sequential_optimize_Nseg` needs one, following
   `opt_stub.py`/`opt_tenchi.py`'s own private pattern rather than
   importing theirs. No `refine` entry point to implement — see "The
   registry pattern" above for why Refine lives entirely in
   `eidos.apps.designer` instead. If your own `run()` wants an internal
   local-polish stage the way `opt_tenchi.py`'s does (DE search, then
   Nelder-Mead polish before returning each seed's result), that's an
   implementation detail of your own `sequential_optimize_Nseg` — not
   something the registry contract requires or exposes.
4. Set your own `OPTIMIZER_VERSION` string at module level.
5. Register the entry in `OPTIMIZER_REGISTRY` in
   `eidos/lib/optimizer.py`, following the `"opt_stub"` entry as a
   template.

**No `count_seeds` registry entry point.** Its only caller
(`eidos.apps.generator.save_experiment_results`) already has the
actual list of seed results `run()` returned and just counts
`len(results)` instead of re-predicting the count — a future optimizer
that decides its own seed count dynamically couldn't honestly
implement a "count before running" entry point anyway. Your own
`calculate_num_seeds`-style helper, if you have one, is purely internal
to your own multi-seed loop (see `opt_tenchi.py`/`opt_stub.py`'s own
private ones), not something the registry calls.

## The checklist — four places, one automated backstop

Both `core/simulators/__init__.py`/`eidos/lib/optimizer.py` (the
registries) and two other files need to know about a new entry, and
none of this is wired together automatically:

1. Write the implementation file (above).
2. Register it in `SIMULATOR_REGISTRY` or `OPTIMIZER_REGISTRY`.
3. Add the new file to `scripts/check_code_version_bump.sh`'s
   `VERSIONED_PAIRS` — otherwise a real change to that file's logic
   can be committed without ever bumping its `SIMULATOR_VERSION` /
   `OPTIMIZER_VERSION`, and the pre-commit hook won't catch it.
4. **Simulators only:** also add the new file to `core/git_info.py`'s
   `REPRODUCIBILITY_RELEVANT_PATHS`. Optimizers are deliberately
   excluded from this list — no re-simulation call site
   (`eidos.apps.viewer`/`exporter`/`trainer`) ever invokes the
   optimizer, since it only runs while originally searching for a
   strategy, so an optimizer file has no bearing on reproducing an
   already-decided strategy's output. Note your `PhysicalSettings`/
   `PhysiologicalSettings` models live in the same file as your kernel,
   so they're already covered by this same entry — no separate listing
   needed (see `docs/ARCHITECTURE.md`'s "Reproducibility tracking"
   section).

Forgetting step 3 or 4 used to fail silently — nothing *looks* broken,
because nothing tried to resolve or cross-check the new entry against
those hardcoded lists. `tests/test_registry_consistency.py` (wired
into `.pre-commit-config.yaml` alongside the version-bump hook and
ruff) exists specifically to close that gap: it reads
`SIMULATOR_REGISTRY`/`OPTIMIZER_REGISTRY` as the actual source of
truth for "what implementations exist" and cross-checks both hardcoded
lists against it on every commit, so a registry entry that was never
added to `VERSIONED_PAIRS` or `REPRODUCIBILITY_RELEVANT_PATHS` is
caught automatically rather than depending on a human rereading this
checklist at the right moment. It also checks that every entry has real
`physical_param_model`/`physiological_param_model`/`param_model` values
(not `None` or the wrong type), that a simulator's physical and
physiological field names never overlap, that every
`physiological_param_model` actually subclasses
`core.schema.PhysiologicalSettingsBase` (i.e. has `cp`/`w_prime`), and
that `build_physics_params`/`compute_course_physics`/
`recompute_course_physics`/`run`/`decode` accept the argument shapes
described above. It does not, however, replace steps 1–2 — you
still have to write the implementation and register it yourself.

## Wiring it into a config

A config JSON's top level names which registry entries to use, plus
that entry's own settings — required, no default for `Engine`'s
`simulator`/`optimizer` fields (see `eidos.lib.optimizer.EngineValidationModel`,
`extra="forbid"`, both required):

```json
{
    "PhysicalSettings": { "...": "...this simulator's own physical_param_model fields..." },
    "PhysiologicalSettings": { "...": "...this simulator's own physiological_param_model fields..." },
    "RunSettings": { "...": "..." },
    "Engine": {
        "simulator": "sim_kiritsubo",
        "optimizer": "opt_tenchi",
        "optimizer_params": { "...": "...this optimizer's own param_model fields (required if it has any -- see below)..." }
    }
}
```

`PhysicalSettings`/`PhysiologicalSettings` are validated against
whichever simulator `Engine.simulator` names (see
`core.simulators.resolve_physical_params`/`resolve_physiological_params`)
— a config naming an unregistered `simulator`/`optimizer` key fails
Pydantic validation and is skipped with a warning by
`load_config_jsons()` — it does not fall back to a default, by design.
`optimizer_params` itself may be omitted entirely from a config (defaults
to `{}`), but whether its OWN fields tolerate being individually omitted
is entirely up to your `param_model`'s own field definitions (see
`eidos.lib.optimizer.EngineValidationModel.validate_and_default_optimizer_params`,
which just runs `param_model.model_validate(optimizer_params)` -- no
separate default-filling step of its own). This project's own convention:
if your optimizer has real tunable parameters, make every one
of them required (`Field(...)`, no Pydantic default) with its own
`json_schema_extra={"preset": ...}` for `eidos.apps.manager`'s GUI Form
Editor to seed it with -- the same rule `PhysicalSettings`/
`PhysiologicalSettings` already follow (see `core.schema.
preset_value_from_field`'s own docstring for the full reasoning). A
config JSON must then always specify every one of those fields
explicitly when it selects your optimizer; a config built through the GUI
already does, since the form seeds and highlights each one the instant
your optimizer is selected. `eidos.lib.optimizers.opt_stub.OptStubParams`
(no fields at all) is the one case this doesn't apply to -- nothing to
require.

## Verifying your new implementation

There is no golden-case regression test over simulator/optimizer
numeric output in this repo, and that is a deliberate choice (see
`docs/ARCHITECTURE.md`'s "Testing philosophy" — for a system this
complex, freezing specific numbers as "correct" risks freezing a bug
instead). What you do get for free is `tests/test_registry_consistency.py`
catching wiring mistakes (a forgotten registration, an unbumped
version, a missing/malformed param_model) before they reach a commit —
`tests/test_entry_points.py` is a separate, unrelated check on
`pyproject.toml`'s console-script wiring, not the registries.

Beyond that, verification is manual and expected to be manual:

- Run your new implementation end-to-end via a config that names it in
  `Engine`, and sanity-check the resulting strategy makes physical
  sense (finish time, W' trajectory if applicable) before relying on
  it for real research.
- If your change is a modification to an *existing* registered file
  rather than a brand-new entry, compare output against the same
  config/seed before and after — `sim_stub.py`/`opt_stub.py` exist
  precisely so a second, independently-shaped entry can be exercised
  end-to-end without touching `sim_kiritsubo.py`/`opt_tenchi.py` at
  all while you're validating the registry mechanism itself.
- Once you're satisfied, bump `SIMULATOR_VERSION`/`OPTIMIZER_VERSION`
  in the same commit (the pre-commit hook enforces this for real
  changes; see `docs/RUNBOOK.md` for the `--no-verify` escape hatch for
  pure refactors/renames with no behavior change).
