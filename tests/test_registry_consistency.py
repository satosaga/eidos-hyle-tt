"""
Guards against the class of bug where a new SIMULATOR_REGISTRY or
OPTIMIZER_REGISTRY entry's implementation file is added without also
registering it in scripts/check_code_version_bump.sh's VERSIONED_PAIRS
and (simulators only) core.git_info.REPRODUCIBILITY_RELEVANT_PATHS --
both of those lists say, in their own comments, that a forgotten entry
would let edits to that file silently escape the check.

This test makes SIMULATOR_REGISTRY/OPTIMIZER_REGISTRY -- the actual,
executable source of truth for "what implementations exist" -- the
basis for checking the other two hand-maintained lists, instead of
relying on a human to keep three independent lists in sync by memory.

core.git_info.REPRODUCIBILITY_RELEVANT_PATHS deliberately excludes
optimizer files (see that module's docstring: the optimizer only runs
while *searching* for a strategy, never while re-simulating one
already decided), so only simulator files are checked against it.
VERSIONED_PAIRS, which backs the pre-commit version-bump check, tracks
both simulators and optimizers, so both are checked against it.

Requires the project's real runtime dependencies to be installed (see
tests/test_entry_points.py). Run inside the project's own venv:

    pip install -e ".[dev]"
    pytest tests/test_registry_consistency.py
"""

import inspect
import re
from pathlib import Path

from pydantic import BaseModel

from core.git_info import REPRODUCIBILITY_RELEVANT_PATHS
from core.simulators import SIMULATOR_REGISTRY
from eidos.lib.optimizer import OPTIMIZER_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
BUMP_SCRIPT_PATH = REPO_ROOT / "scripts" / "check_code_version_bump.sh"


def _source_path(func) -> str:
    """Repo-relative, posix-style path of the file defining `func`.

    Simulator kernels are @njit-decorated (numba CPUDispatcher objects),
    which inspect.getsourcefile doesn't handle directly -- unwrap to the
    original Python function via .py_func first, same as numba's own
    inspect.getsource() support does internally.
    """
    func = getattr(func, "py_func", func)
    return Path(inspect.getsourcefile(func)).resolve().relative_to(REPO_ROOT).as_posix()


def _simulator_files() -> set[str]:
    return {_source_path(spec.kernel) for spec in SIMULATOR_REGISTRY.values()}


def _optimizer_files() -> set[str]:
    return {_source_path(spec.run) for spec in OPTIMIZER_REGISTRY.values()}


def _versioned_pairs_files() -> set[str]:
    """Parse VERSIONED_PAIRS out of check_code_version_bump.sh's heredoc-style
    block (one `path:VERSION_CONST` per line) -- see that script for the
    format this depends on."""
    text = BUMP_SCRIPT_PATH.read_text()
    return set(re.findall(r"^(\S+\.py):\w+\s*$", text, re.MULTILINE))


def test_every_registered_simulator_is_reproducibility_tracked():
    missing = _simulator_files() - set(REPRODUCIBILITY_RELEVANT_PATHS)
    assert not missing, (
        f"{sorted(missing)} in SIMULATOR_REGISTRY but missing from "
        "core.git_info.REPRODUCIBILITY_RELEVANT_PATHS -- edits to it "
        "would silently escape the reproducibility check."
    )


def test_every_registered_simulator_has_a_version_bump_check():
    missing = _simulator_files() - _versioned_pairs_files()
    assert not missing, (
        f"{sorted(missing)} in SIMULATOR_REGISTRY but missing from "
        "VERSIONED_PAIRS in scripts/check_code_version_bump.sh."
    )


def test_every_registered_optimizer_has_a_version_bump_check():
    missing = _optimizer_files() - _versioned_pairs_files()
    assert not missing, (
        f"{sorted(missing)} in OPTIMIZER_REGISTRY but missing from "
        "VERSIONED_PAIRS in scripts/check_code_version_bump.sh."
    )


# ---------------------------------------------------------------------------
# param_model presence, call-signature shape, and field-namespace checks --
# guards against the class of bug where a new/edited registry entry's
# param_model is missing or shaped wrong, or its entry-point functions fall
# out of sync with what OptimizerSpec/SimulatorSpec's own callers actually
# pass.
# ---------------------------------------------------------------------------

def test_every_registered_simulator_has_param_models():
    """SimulatorSpec.physical_param_model / physiological_param_model must
    each be set to an actual Pydantic BaseModel subclass -- these are what
    core.simulators.resolve_physical_params/resolve_physiological_params
    validate config JSON sections against, and what
    core.simulators.calibratable_physical_keys introspects for calibration
    bounds. A registry entry with either left as None or some non-BaseModel
    placeholder would fail confusingly deep inside a caller instead of here."""
    for key, spec in SIMULATOR_REGISTRY.items():
        assert isinstance(spec.physical_param_model, type) and issubclass(spec.physical_param_model, BaseModel), (
            f"SIMULATOR_REGISTRY['{key}'].physical_param_model is not a Pydantic BaseModel subclass: "
            f"{spec.physical_param_model!r}"
        )
        assert isinstance(spec.physiological_param_model, type) and issubclass(spec.physiological_param_model, BaseModel), (
            f"SIMULATOR_REGISTRY['{key}'].physiological_param_model is not a Pydantic BaseModel subclass: "
            f"{spec.physiological_param_model!r}"
        )


def test_every_registered_optimizer_has_a_param_model():
    """OptimizerSpec.param_model must be an actual Pydantic BaseModel
    subclass -- what eidos.lib.optimizer.EngineValidationModel validates
    Engine.optimizer_params against (via resolve_optimizer_params). An
    optimizer with no tunables at all (e.g.
    "opt_stub") still needs a real (if field-less) BaseModel here, not None
    -- see eidos.lib.optimizers.opt_stub.OptStubParams."""
    for key, spec in OPTIMIZER_REGISTRY.items():
        assert isinstance(spec.param_model, type) and issubclass(spec.param_model, BaseModel), (
            f"OPTIMIZER_REGISTRY['{key}'].param_model is not a Pydantic BaseModel subclass: {spec.param_model!r}"
        )


def test_simulator_physical_and_physiological_fields_dont_overlap():
    """A SIMULATOR_REGISTRY entry's physical_param_model and
    physiological_param_model must share no field names.
    core.physics_overrides.build_overridden_params routes a single combined
    `overrides` dict to whichever of the two models actually has each key
    (`k in physical_fields` / `k in physiological_fields`) -- an
    overlapping field name would make that routing ambiguous (silently
    applying an override to only one of the two, or both, depending on
    dict-comprehension iteration order) instead of failing loudly."""
    for key, spec in SIMULATOR_REGISTRY.items():
        physical_fields = set(spec.physical_param_model.model_fields)
        physiological_fields = set(spec.physiological_param_model.model_fields)
        overlap = physical_fields & physiological_fields
        assert not overlap, (
            f"SIMULATOR_REGISTRY['{key}']: physical_param_model and "
            f"physiological_param_model share field name(s) {sorted(overlap)}"
        )


def test_every_registered_simulator_physiological_settings_has_cp_and_w_prime():
    """Every SIMULATOR_REGISTRY entry's physiological_param_model must be a
    subclass of core.schema.PhysiologicalSettingsBase (guaranteeing cp/
    w_prime), regardless of whether that simulator's own kernel physics
    reads either field -- see that class's own docstring for why: cp is
    load-bearing for FIT/ZWO export (target power encoded as a fraction of
    CP), and every simulator has a responsibility to report a coherent (if
    constant) W' balance trajectory. Downstream code throughout
    core.data_manager/eidos.apps.designer/trainer/generator/eidos.apps.
    analyzer.canvas/eidos.lib.pdf_exporter reads physiological_settings['cp']/
    ['w_prime'] unconditionally on the strength of this guarantee -- a
    registry entry that violated it would silently reintroduce the crashes
    this test exists to prevent."""
    from core.schema import PhysiologicalSettingsBase
    for key, spec in SIMULATOR_REGISTRY.items():
        assert issubclass(spec.physiological_param_model, PhysiologicalSettingsBase), (
            f"SIMULATOR_REGISTRY['{key}'].physiological_param_model "
            f"({spec.physiological_param_model!r}) does not inherit from "
            "core.schema.PhysiologicalSettingsBase -- cp/w_prime are not guaranteed."
        )


def test_every_registered_simulator_build_physics_params_accepts_new_signature():
    """SimulatorSpec.build_physics_params must accept exactly the 4-argument
    (physical, physiological, run, course) shape every real caller
    (eidos.apps.generator/designer/exporter/trainer/viewer/analyzer,
    core.physics_overrides, hyle.apps.fit2gpx_converter) uses -- a
    positional-arg count mismatch here would fail at every one of those
    call sites, not just one. No cda_ratios argument: each simulator's
    own build_physics_params loads physical.cda_yaw_table_filename
    itself, if it even has that field -- see core.simulators' module
    docstring."""
    for key, spec in SIMULATOR_REGISTRY.items():
        params = list(inspect.signature(spec.build_physics_params).parameters)
        assert len(params) == 4, (
            f"SIMULATOR_REGISTRY['{key}'].build_physics_params has {len(params)} "
            f"parameters {params}, expected 4: (physical, physiological, run, course)"
        )


def test_every_registered_simulator_has_course_physics_callables():
    """SimulatorSpec.compute_course_physics/recompute_course_physics must
    each be set and accept the right arity -- (points, physical, run) and
    (course_profile, physical) respectively. Course physics is computed
    by each simulator's own module rather than a single shared function,
    so every registry entry must supply both -- see core.simulators'
    module docstring."""
    for key, spec in SIMULATOR_REGISTRY.items():
        compute_params = list(inspect.signature(spec.compute_course_physics).parameters)
        assert len(compute_params) == 3, (
            f"SIMULATOR_REGISTRY['{key}'].compute_course_physics has {len(compute_params)} "
            f"parameters {compute_params}, expected 3: (points, physical, run)"
        )
        recompute_params = list(inspect.signature(spec.recompute_course_physics).parameters)
        assert len(recompute_params) == 2, (
            f"SIMULATOR_REGISTRY['{key}'].recompute_course_physics has {len(recompute_params)} "
            f"parameters {recompute_params}, expected 2: (course_profile, physical)"
        )


def test_every_registered_simulator_course_physics_lives_with_its_kernel():
    """compute_course_physics/recompute_course_physics must be defined in
    the SAME file as that entry's own kernel -- the whole point of moving
    course-physics out of the old shared core.course_geometry.
    calculate_course_physics was so a change to it is covered by that
    file's own SIMULATOR_VERSION bump policy (scripts/
    check_code_version_bump.sh); living in a separate file would silently
    reopen the exact reproducibility-tracking gap this refactor closed."""
    for key, spec in SIMULATOR_REGISTRY.items():
        kernel_file = _source_path(spec.kernel)
        compute_file = _source_path(spec.compute_course_physics)
        recompute_file = _source_path(spec.recompute_course_physics)
        assert compute_file == kernel_file, (
            f"SIMULATOR_REGISTRY['{key}'].compute_course_physics lives in "
            f"{compute_file}, not alongside its own kernel in {kernel_file}"
        )
        assert recompute_file == kernel_file, (
            f"SIMULATOR_REGISTRY['{key}'].recompute_course_physics lives in "
            f"{recompute_file}, not alongside its own kernel in {kernel_file}"
        )


def test_every_registered_simulator_pairs_cda_with_its_yaw_table():
    """A SIMULATOR_REGISTRY entry's physical_param_model may not have a
    `cda` field without also having `cda_yaw_table_filename`, or vice
    versa -- CdA and its yaw-multiplier table are always a pair, never two
    independently-optional concerns (a simulator with no aero-drag term
    at all is the only case entitled to have neither -- see core.
    simulators.sim_stub's own section-0 comment for the full reasoning).
    A simulator that models CdA but skips the yaw table would have no way
    to resolve cda_ratios in its own build_physics_params; one with the
    table but no CdA field would have nothing to apply the multiplier to."""
    for key, spec in SIMULATOR_REGISTRY.items():
        fields = set(spec.physical_param_model.model_fields)
        has_cda = "cda" in fields
        has_table = "cda_yaw_table_filename" in fields
        assert has_cda == has_table, (
            f"SIMULATOR_REGISTRY['{key}'].physical_param_model has cda={has_cda} "
            f"but cda_yaw_table_filename={has_table} -- these must always be paired."
        )


def test_every_registered_optimizer_entry_points_accept_new_signature():
    """OptimizerSpec.run/decode must each accept a trailing `params`
    argument (the optimizer's own validated param_model instance) -- see
    eidos.lib.optimizer's module docstring for why this was added to every
    entry point in the config/registry refoundation. decode is additionally
    checked for its full 5-argument shape (x_combined, n_seg,
    course_distance, l_min, params), matching every real caller
    (eidos.apps.generator.save_experiment_results). OptimizerSpec has no
    `refine` field -- eidos.apps.designer's Refine button uses its own
    independent Nelder-Mead polish (see designer.py's module-level
    _refine_powers_locally), not a per-optimizer registry entry point."""
    for key, spec in OPTIMIZER_REGISTRY.items():
        run_params = inspect.signature(spec.run).parameters
        assert "params" in run_params, f"OPTIMIZER_REGISTRY['{key}'].run is missing a `params` argument"

        decode_params = list(inspect.signature(spec.decode).parameters)
        assert len(decode_params) == 5, (
            f"OPTIMIZER_REGISTRY['{key}'].decode has {len(decode_params)} parameters "
            f"{decode_params}, expected 5: (x_combined, n_seg, course_distance, l_min, params)"
        )
