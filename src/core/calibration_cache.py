##########################
# calibration_cache.py
##########################
"""
On-disk cache for core.calibrator.calibrate() results (compute_cache_key/
load_cached_result/save_cached_result) and, separately, for its
sample_morris_sensitivity/sample_sobol_sensitivity pre-Auto-Fit sensitivity-
screening results (compute_sensitivity_cache_key/load_cached_sensitivity/
save_cached_sensitivity) -- both keyed on every input that actually affects
their own output (CalibrationInputs' own fields, the run's own config
scalars, and a source-code fingerprint of every module the computation
touches) so a cache hit is only ever returned for a byte-for-byte-equivalent
run — never an approximate or best-effort match. See calibrate()'s and
sample_morris_sensitivity's/sample_sobol_sensitivity's own docstrings for
how each is wired in.

This module has no knowledge of what it's caching or why -- the source-
code fingerprint is computed by calibrator._source_fingerprint and passed
in as a plain code_fingerprint string, the same way every other run-
config scalar is passed in, rather than this module importing
core.calibrator back to compute it itself. Keeps this module a fully
generic key/cache utility with no dependency on core.calibrator at all.

Deliberately excludes anything the respective computation itself never
reads — most notably a strategy's planned_power_blocks (the DE-optimized
pacing plan). Both calibrate() and the sensitivity samplers replay the FIT
activity's OWN recorded power, not the strategy's planned one (see
calibrator.py's "Power source" docstring section), so it never appears in
CalibrationInputs at all. Two strategy_*.json exports that share a
run_set_id but differ only in N_seg/seed (i.e. only in planned_power_blocks)
therefore hash to the SAME key and correctly share one cache entry instead
of each forcing its own recompute.

No automatic eviction: entries are small (bounded arrays -- a few hundred
KB to ~1MB each), so at realistic usage volumes each of these directories
stays in the tens-to-low-hundreds-of-MB range. Each is a plain directory of
independent files; delete it, or any file in it, by hand at any time — a
missing entry is just a cache miss, never a correctness risk.
"""

import dataclasses
import hashlib
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/eidos_hyle_tt/autofit")

# Separate directory from DEFAULT_CACHE_DIR -- Morris/Sobol' sensitivity
# screening results are a different kind of object (MorrisSensitivityTrials/
# SobolSensitivityTrials, not CalibrationResult) with no from_cache/
# cached_at fields of their own (see load_cached_sensitivity), so keeping
# them separate means each can be manually cleared independently.
DEFAULT_SENSITIVITY_CACHE_DIR = os.path.expanduser("~/.cache/eidos_hyle_tt/sensitivity")


def _update_hash(h, value) -> None:
    """
    Recursively fold value into h — generic over CalibrationInputs'
    actual field types (dataclasses, NamedTuples, dicts, lists/tuples,
    numpy arrays, plain scalars) rather than a hand-enumerated field
    list, so a field ADDED to any of those structures later is
    automatically picked up here too, instead of silently falling out of
    the cache key the way a hand-maintained list would.

    Args:
        h:     A hashlib hash object, updated in place.
        value: Anything reachable from a CalibrationInputs.
    """
    if isinstance(value, np.ndarray):
        h.update(b"ndarray:")
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        h.update(b"dict:")
        for k in sorted(value.keys(), key=str):
            h.update(str(k).encode())
            _update_hash(h, value[k])
    elif hasattr(value, "_fields"):  # NamedTuple (PhysicsParams, PowerBlocks, ...)
        h.update(b"namedtuple:")
        h.update(type(value).__name__.encode())
        for name in value._fields:
            h.update(name.encode())
            _update_hash(h, getattr(value, name))
    elif isinstance(value, (list, tuple)):
        h.update(b"seq:")
        for item in value:
            _update_hash(h, item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        h.update(b"dataclass:")
        h.update(type(value).__name__.encode())
        for f in dataclasses.fields(value):
            h.update(f.name.encode())
            _update_hash(h, getattr(value, f.name))
    else:
        h.update(repr(value).encode())


def compute_cache_key(
    calib,
    initial_base_seed: int,
    seed_factor: int,
    seed_multiplier: int,
    *,
    code_fingerprint: str,
) -> str:
    """
    Deterministic key for a calibrate() call — identical for two calls
    that would produce identical output, different for any that might
    not. See module docstring for what "identical" means here (every
    CalibrationInputs field, the run-config scalars, and a source-code
    fingerprint) and what it deliberately excludes (planned_power_blocks,
    which never reaches CalibrationInputs in the first place).

    Also excludes two categories of calib.fixed_overrides entry that
    don't actually affect calibrate()'s output:

    - Any key ALSO in calib.free_keys: calibrator.pack_physics always
      builds its per-evaluation overrides as `{**calib.fixed_overrides,
      **dict(zip(calib.free_keys, x_free))}`, so free_keys uncondition-
      ally win regardless of what fixed_overrides holds for them. In the
      Analyzer GUI, fixed_overrides is the PhysicsOverridePanel's
      current spinbox values for EVERY row, including "Auto Fit"-checked
      ones whose displayed number is just whatever the previous run
      happened to calibrate (dead data, not an input) -- hashing it
      unfiltered would spuriously miss on two otherwise-identical runs.
    - Any key in core.simulators.physiological_only_keys(calib.
      simulator_key) (W' Balance-only fields): these only affect the
      physiologically-clamped power path (is_target_power=True), never
      the externally-driven replay path calibrate() always runs
      (is_target_power=False, see objective_calibration). Same
      PhysicsOverridePanel.WBAL_ONLY_KEYS reasoning (eidos.apps.
      analyzer.widgets) that panel applies to its own "fit confirmed"
      styling. calibrate()'s own cache-hit branch still rebuilds
      physics_overrides fresh from the CURRENT call's fixed_overrides
      for these keys -- see that function's docstring.

    Does not take sensitivity_n_samples/local_sensitivity_n_baselines/
    local_sensitivity_n_sweep_points -- calibrate() itself runs no
    sensitivity sampling (see calibrator.py's module docstring); that
    happens BEFORE Auto Fit, via sample_morris_sensitivity/
    sample_sobol_sensitivity, cached separately by
    compute_sensitivity_cache_key below.

    Args:
        calib: The CalibrationInputs calibrate() built for this run —
                    already bundles every data-relevant input (course
                    geometry, physics baseline, activity-derived power/
                    speed grids, fixed_overrides, free_keys) in one
                    object, so nothing else data-side needs passing here.
        initial_base_seed, seed_factor, seed_multiplier: The run-config
                    scalars calibrate() itself doesn't bundle into calib.
        code_fingerprint: A hex digest covering every module calibrate()'s
                    actual computation touches -- computed by
                    calibrator._source_fingerprint, not by this module:
                    this module has no knowledge of what it's caching or
                    why, only that two calls whose fingerprint differs
                    must never share a cache entry. ANY change to any of
                    those modules -- even a comment -- therefore
                    invalidates every cache entry; that is the intended,
                    safe-by-default failure direction. A missed real
                    behavior change silently serving a stale result would
                    be a much worse failure than an occasional
                    unnecessary recompute.

    Returns:
        A hex digest string, safe to use as a filename.
    """
    # Local import: core.simulators has no dependency back on this
    # module, so no circularity risk -- just deferred to keep this
    # module's own top-level import list minimal.
    import core.simulators

    h = hashlib.sha256()

    # fixed_overrides, filtered per the docstring above, hashed
    # separately from the rest of calib's fields below.
    wbal_only = core.simulators.physiological_only_keys(calib.simulator_key)
    relevant_fixed_overrides = {
        k: v for k, v in calib.fixed_overrides.items()
        if k not in calib.free_keys and k not in wbal_only
    }
    h.update(b"fixed_overrides:")
    _update_hash(h, relevant_fixed_overrides)

    for f in dataclasses.fields(calib):
        if f.name == "fixed_overrides":
            continue  # already handled, filtered, above
        h.update(f.name.encode())
        _update_hash(h, getattr(calib, f.name))

    h.update(str(initial_base_seed).encode())
    h.update(str(seed_factor).encode())
    h.update(str(seed_multiplier).encode())
    h.update(code_fingerprint.encode())
    return h.hexdigest()


def compute_sensitivity_cache_key(calib, method: str, *, code_fingerprint: str, **method_kwargs) -> str:
    """
    Deterministic key for a sample_morris_sensitivity/
    sample_sobol_sensitivity call -- same principle as compute_cache_key
    (every CalibrationInputs field plus a source-code fingerprint), but
    for the pre-Auto-Fit sensitivity-screening step instead of calibrate()
    itself. calib.free_keys here is whatever (possibly narrowed) subset
    is being screened, not necessarily every calibratable key -- already
    covered by the generic per-field hashing below, same as calibrate()'s
    own key treats free_keys.

    fixed_overrides is filtered the same way compute_cache_key filters
    it, and for the same reason: build_calibration_inputs does not
    reject a key appearing in both free_keys and fixed_overrides, and
    TTAnalyzerWindow's inline sensitivity controls do pass fixed_overrides
    as every PhysicsOverridePanel spinbox's current value, including
    rows that are also being screened (i.e. also in calib.free_keys) --
    objective_calibration's own per-evaluation overrides (via
    _evaluate_parallel) always let calib.free_keys win regardless of
    what fixed_overrides holds for those same keys, so hashing that
    dead data unfiltered would spuriously miss on two otherwise-
    identical runs. W'-Balance-only keys are excluded for the same
    reason compute_cache_key excludes them: _evaluate_parallel runs
    objective_calibration with is_target_power=False, so those keys
    never affect the result either.

    Args:
        calib:   The CalibrationInputs the sensitivity call was built
                    from.
        method:  "morris" or "sobol" -- folded into the key so the two
                    methods' cache entries can never collide even if
                    some run-config scalar happened to coincide.
        code_fingerprint: See compute_cache_key's identically-named
                    parameter -- computed by calibrator._source_
                    fingerprint the same way.
        **method_kwargs: The method's own run-config scalars (r/
                    num_levels/seed for Morris; n/seed for Sobol) --
                    passed as keywords, sorted before hashing, so a
                    scalar added later is automatically covered instead
                    of silently falling out of the key.

    Returns:
        A hex digest string, safe to use as a filename.
    """
    # Local import: same deferred-import reasoning as compute_cache_key's
    # own import of core.simulators above.
    import core.simulators

    h = hashlib.sha256()
    h.update(b"method:")
    h.update(method.encode())

    # fixed_overrides, filtered per the docstring above, hashed
    # separately from the rest of calib's fields below -- same pattern
    # as compute_cache_key.
    wbal_only = core.simulators.physiological_only_keys(calib.simulator_key)
    relevant_fixed_overrides = {
        k: v for k, v in calib.fixed_overrides.items()
        if k not in calib.free_keys and k not in wbal_only
    }
    h.update(b"fixed_overrides:")
    _update_hash(h, relevant_fixed_overrides)

    for f in dataclasses.fields(calib):
        if f.name == "fixed_overrides":
            continue  # already handled, filtered, above
        h.update(f.name.encode())
        _update_hash(h, getattr(calib, f.name))

    for k in sorted(method_kwargs):
        h.update(k.encode())
        _update_hash(h, method_kwargs[k])

    h.update(code_fingerprint.encode())
    return h.hexdigest()


def _cache_path(key: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{key}.pkl")


def load_cached_result(key: str, cache_dir: str = DEFAULT_CACHE_DIR):
    """
    Return the cached CalibrationResult for key, or None on a miss —
    including a miss on any deserialization failure (a corrupt/partial
    file, or a pickle written by an incompatible numpy/scipy version):
    treated exactly like an ordinary cache miss (log a warning, let the
    caller recompute) rather than raised, since a stale/unreadable cache
    file is never a reason to fail an Auto Fit run outright.

    Args:
        key:       From compute_cache_key.
        cache_dir: Directory cache files live in.

    Returns:
        The cached CalibrationResult (with from_cache=True), or None.
    """
    path = _cache_path(key, cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
    except Exception:
        logger.warning(
            "calibration_cache: failed to load %s -- treating as a cache miss", path, exc_info=True,
        )
        return None
    # The pickled object's own from_cache is whatever it was when SAVED
    # (False -- see calibrate(), which only sets it after this call
    # returns None) -- set True here, on the read side, since that's what
    # actually makes this a cache hit rather than a fresh computation.
    result.from_cache = True
    logger.debug("calibration_cache: HIT %s (originally computed %s)", path, result.cached_at)
    return result


def save_cached_result(key: str, result, cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """
    Persist result under key. Writes to a per-process temp file first and
    os.replace()s it into place — atomic, so a crash mid-write never
    leaves a corrupt file at the real path for a later load to trip over.

    Exceptions propagate (e.g. a full disk): a failed cache WRITE should
    be visible, not silently swallowed. It never blocks returning result
    to the original caller though — see calibrate(), which calls this
    only after result is already fully built.

    Args:
        key:       From compute_cache_key.
        result:    A CalibrationResult.
        cache_dir: Directory to write into; created if missing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(key, cache_dir)
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    logger.debug("calibration_cache: saved %s", path)


def load_cached_sensitivity(key: str, cache_dir: str = DEFAULT_SENSITIVITY_CACHE_DIR):
    """
    Return the cached MorrisSensitivityTrials/SobolSensitivityTrials for
    key, or None on a miss -- same read/atomic-write/no-eviction shape as
    load_cached_result, but for sample_morris_sensitivity's/
    sample_sobol_sensitivity's own results, which (unlike
    CalibrationResult) have no from_cache/cached_at fields of their own,
    so this doesn't set anything on the returned object -- just returns
    it, or None, plainly.

    Args:
        key:       From compute_sensitivity_cache_key.
        cache_dir: Directory cache files live in.

    Returns:
        The cached MorrisSensitivityTrials/SobolSensitivityTrials, or
        None -- including on any deserialization failure (a corrupt/
        partial file, or a pickle written by an incompatible numpy/SALib
        version), treated exactly like an ordinary cache miss (log a
        warning, let the caller recompute) rather than raised, same
        reasoning as load_cached_result.
    """
    path = _cache_path(key, cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
    except Exception:
        logger.warning(
            "calibration_cache: failed to load %s -- treating as a cache miss", path, exc_info=True,
        )
        return None
    logger.debug("calibration_cache: sensitivity HIT %s", path)
    return result


def save_cached_sensitivity(key: str, result, cache_dir: str = DEFAULT_SENSITIVITY_CACHE_DIR) -> None:
    """
    Persist result (a MorrisSensitivityTrials or SobolSensitivityTrials)
    under key -- same atomic-write shape as save_cached_result.

    Args:
        key:       From compute_sensitivity_cache_key.
        result:    A MorrisSensitivityTrials or SobolSensitivityTrials.
        cache_dir: Directory to write into; created if missing.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(key, cache_dir)
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    logger.debug("calibration_cache: saved sensitivity %s", path)
