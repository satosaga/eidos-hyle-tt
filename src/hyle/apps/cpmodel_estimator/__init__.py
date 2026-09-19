#!/usr/bin/env python3
"""
HYLE - hyle.apps.cpmodel_estimator

Estimates a rider's Morton 3-parameter critical power model (CP, W', Pmax)
from their full GoldenCheetah ride history. Opens a browser UI where the
user picks a period (recent N weeks/months, or an absolute year-month
range); the tool builds a Mean Maximal Power (MMP) curve over that period
and fits the 3p model to it, showing the fitted numbers overlaid on the MMP
curve. Screen-display only -- no file is written as a "result" (see the
note below for the one deliberate exception: an internal perf cache).

Usage
-----
    hyle-cpmodel-estimator

No CLI arguments: the input directory is resources/GC_activities/ at the
repo root by default, or the EIDOS_TT_GC_ACTIVITIES_DIR environment
variable when set -- matching the rest of the hyle.apps.* series'
"GUI-only, no path picker" convention (the directory is chosen once, via
env var or symlink, not per-run).

Scope
-----
This tool estimates exactly three parameters: cp_w, w_prime_j,
p_max_physio_w. It does NOT estimate w_prime_recovery_rate (K) or
v_slope_ratio -- a pure MMP-curve fit has no access to the on/off-CP
dynamic response those two describe, and there's currently no estimation
method for either. Feeding the estimated cp_w/w_prime_j/p_max_physio_w
values back into a rider config JSON is an intentionally separate, later
feature -- this tool only displays them.

Architecture
------------
Like hyle.apps.fit2gpx_converter (and unlike hyle.apps.course_checker), this
tool keeps a small local HTTP server (127.0.0.1, random free port) running
after the browser opens, because the period selector needs to trigger a
fresh computation on the Python side every time the user clicks "Estimate"
-- a static, one-shot HTML file can't do that on its own.

Note (perf cache, not a "result" file)
---------------------------------------
resources/GC_activities/ is several GB across ~2600 JSON files, so parsing
all of it at startup is the one part of this tool that's slow.
temp/hyle_cpmodel_estimator_cache.json (gitignored) caches each file's
parsed MMP vector keyed by (mtime, size), so only new/changed rides get
re-parsed on subsequent runs. This is a performance cache the tool manages
for itself, not a result export -- the "no file output" requirement in
this tool's spec is about not requiring the user to save/export their
CP/W'/Pmax results, and does not cover this cache.
It lives under temp/, not resources/, because it's non-domain bookkeeping
the tool keeps for its own sake (see core.io_config's BASE_TEMP_DIR
comment) -- unlike GC_activities/, which is the actual domain data this
tool reads.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.optimize import differential_evolution, minimize

from core.logging_setup import configure_logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
# This file lives at <repo_root>/src/hyle/apps/cpmodel_estimator/__init__.py,
# with its HTML template sitting right next to it. GC_activities/'s default
# location (see ENV_VAR_GC_ACTIVITIES_DIR below) and the on-disk perf cache
# both intentionally stay at the repo root, not inside src/, so those two
# need REPO_ROOT rather than SCRIPT_DIR.
# GC_activities is domain data (the tool's actual input), so it's gathered
# under resources/ alongside everything core.io_config manages, even
# though it isn't one of that module's own constants (it's specific to
# this tool, not shared). The perf cache is the opposite: it's non-domain
# bookkeeping the tool keeps only for its own sake, so it lives under
# temp/ instead -- the same "group by role" reasoning, landing in a
# different sibling directory because its role is different (see
# core.io_config's RESOURCES_DIR / BASE_TEMP_DIR comments).
# Four .parent hops: cpmodel_estimator/ -> apps/ -> hyle/ -> src/ -> repo root.
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
HTML_FILE = SCRIPT_DIR / "cpmodel_estimator.html"

# GC_activities/ is each user's own pre-existing GoldenCheetah ride history,
# not something this tool generates -- unlike core.io_config's
# ENV_VAR_STRATEGIES_DIR (an optional override of a default that already
# works with no setup), there's no working default here, so the env var is
# the primary way to point at it; the resources/GC_activities/ symlink
# (this repo's own machine, an external volume) is kept only as this
# installation's fallback, not a convention every user is expected to
# replicate.
ENV_VAR_GC_ACTIVITIES_DIR = "EIDOS_TT_GC_ACTIVITIES_DIR"
ACTIVITIES_DIR = Path(os.environ.get(ENV_VAR_GC_ACTIVITIES_DIR, str(REPO_ROOT / "resources" / "GC_activities")))
CACHE_FILE = REPO_ROOT / "temp" / "hyle_cpmodel_estimator_cache.json"
CACHE_VERSION = 2  # bump to invalidate all cached entries when _parse_one_file's extraction logic changes

# Fixed MMP sample durations [s]. Deliberately a sparse, log-spaced set
# (not every second) -- this doubles as the fit's default point set, so no
# extra reweighting is needed to keep short durations from numerically
# dominating the objective (see fit_morton_3p).
DURATIONS_S: tuple[int, ...] = (
    5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240,
    300, 360, 480, 600, 720, 900, 1200, 1500, 1800,
)

MIN_FIT_POINTS = 4
_CURVE_N_POINTS = 200      # smooth-curve sample count for the chart overlay

# Envelope-fit search bounds -- generous physiological ranges. DE explores
# the whole box directly, so these don't need to be anchored to observed
# data the way curve_fit's initial guess used to.
_FIT_BOUNDS = {
    "cp_w": (50.0, 600.0),
    "w_prime_j": (500.0, 200000.0),
    "p_max_physio_w": (200.0, 3000.0),
}
_FIT_SEED = 42                       # fixed -- deterministic DE given the same data/period
_INFEASIBLE_PENALTY = 1.0e6          # CP >= Pmax (unphysical: tau0 undefined/negative)
_UNDERSHOOT_PENALTY_WEIGHT = 1.0e6   # squared-Watt penalty per unit of envelope violation -- must
                                      # dominate the sum-of-squares-scale overshoot cost term below
                                      # (see _envelope_objective's own docstring for the full rationale)

DEFAULT_PERIOD_MODE = "recent_weeks"
DEFAULT_PERIOD_WEEKS = 6


@dataclass(frozen=True)
class ParsedActivity:
    start_time: datetime  # naive, UTC (parsed from STARTTIME)
    mmp_w: np.ndarray      # shape (len(DURATIONS_S),); NaN where no run reaches that duration


@dataclass(frozen=True)
class ScanStats:
    total_files: int
    parsed_ok: int
    skipped_no_watts: int
    skipped_bad_recintsecs: int
    skipped_malformed: int
    skipped_estimated_power: int


@dataclass(frozen=True)
class PeriodSelection:
    start: datetime  # inclusive
    end: datetime     # exclusive
    label: str


# --------------------------------------------------------------------
# GC_activities/*.json parsing
# --------------------------------------------------------------------

def _split_into_runs(secs: np.ndarray, watts: np.ndarray, rec_int_s: int) -> list[np.ndarray]:
    """Maximal contiguous WATTS-only runs. Drops NaN-WATTS samples first,
    then splits what remains on any SECS step != rec_int_s -- this handles
    both a genuine recording gap (e.g. consecutive SECS diffs of
    {1, 2, 16238} within one real file, i.e. a paused/resumed recording)
    and a mid-file WATTS dropout in one pass, since removing a NaN sample
    also breaks time-contiguity across it. Returns watts-only arrays (SECS
    itself is no longer needed once contiguity is established)."""
    valid = ~np.isnan(watts)
    watts_v = watts[valid]
    if len(watts_v) <= 1:
        return [watts_v] if len(watts_v) else []
    secs_v = secs[valid]
    breaks = np.where(np.diff(secs_v) != rec_int_s)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(secs_v)]))
    return [watts_v[s:e] for s, e in zip(starts, ends)]


def _activity_mmp(runs: list[np.ndarray]) -> np.ndarray:
    """This activity's own MMP value per duration: max D-second contiguous
    rolling average across all its runs (never bridging a run boundary),
    via an O(n) cumsum per run. Precomputed once per activity (at parse
    time) so a period change at request time is just an elementwise max
    over cached arrays, not a re-scan of raw power samples. Durations are
    always DURATIONS_S -- this function's only caller never varies it."""
    result = np.full(len(DURATIONS_S), np.nan)
    for run in runs:
        n = len(run)
        if n == 0:
            continue
        cumsum = np.concatenate(([0.0], np.cumsum(run)))
        for i, d in enumerate(DURATIONS_S):
            if n < d:
                break  # DURATIONS_S is ascending -- no larger d fits either
            window_sums = cumsum[d:] - cumsum[:-d]
            best = float(window_sums.max()) / d
            if np.isnan(result[i]) or best > result[i]:
                result[i] = best
    return result


def _parse_one_file(path: Path) -> tuple[ParsedActivity | None, str]:
    """Parse one GC_activities/*.json file. Returns (activity_or_None, reason),
    reason in {"ok", "no_watts", "bad_recintsecs", "estimated_power",
    "malformed"}. Never raises -- every failure path is caught here so one
    bad file can't crash the whole directory scan."""
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        ride = data["RIDE"]
        start_time = datetime.strptime(ride["STARTTIME"].strip(), "%Y/%m/%d %H:%M:%S UTC")
        rec_int_s = int(ride["RECINTSECS"])
        samples = ride["SAMPLES"]
        if not samples:
            return None, "malformed"
        secs = np.array([s["SECS"] for s in samples], dtype=float)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError):
        return None, "malformed"

    if rec_int_s != 1:
        # No observed instances of this in the dataset (a 200-file random
        # sample was uniformly RECINTSECS==1) -- not worth building general
        # resampling logic for a case that's never actually occurred.
        return None, "bad_recintsecs"

    # GoldenCheetah records an "Estimate Power" action in TAGS['Change
    # History'] whenever a ride's WATTS came from its own speed/elevation
    # model (typically synced from Strava for a ride recorded on a bike
    # with no power meter) rather than a real power meter. Estimated
    # power isn't a real measurement -- an algorithmic estimate can
    # produce an implausible MMP outlier that skews the fit -- so these
    # rides are excluded from the MMP pool entirely, same as rides with
    # no WATTS.
    if "Estimate Power" in ride.get("TAGS", {}).get("Change History", ""):
        return None, "estimated_power"

    if np.isnan(secs).any():
        return None, "malformed"  # SECS must always be present; a hole here is corruption

    watts = np.array(
        [np.nan if s.get("WATTS") is None else float(s["WATTS"]) for s in samples], dtype=float
    )
    if np.isnan(watts).all():
        return None, "no_watts"

    runs = _split_into_runs(secs.astype(np.int64), watts, rec_int_s)
    mmp = _activity_mmp(runs)
    return ParsedActivity(start_time=start_time, mmp_w=mmp), "ok"


# --------------------------------------------------------------------
# Startup scan, backed by a persistent on-disk cache
# --------------------------------------------------------------------

def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != CACHE_VERSION:
            return {}
        return data.get("entries", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def _save_cache(entries: dict) -> None:
    payload = {"version": CACHE_VERSION, "entries": entries}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)  # temp/ may not exist yet
    tmp = CACHE_FILE.with_name(CACHE_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(CACHE_FILE)  # atomic on the same filesystem


def _activities_from_entries(entries: dict) -> tuple[list[ParsedActivity], ScanStats]:
    """Reconstructs the (activities, stats) result from a raw cache-entries
    dict -- shared by the normal scan path (over freshly-built entries) and
    the "directory looks inaccessible" fallback path (over the untouched
    old cache) below, so the counting logic only lives in one place."""
    activities: list[ParsedActivity] = []
    parsed_ok = skipped_no_watts = skipped_bad_recintsecs = skipped_malformed = skipped_estimated_power = 0
    for entry in entries.values():
        reason = entry["reason"]
        if reason == "ok":
            parsed_ok += 1
            activities.append(ParsedActivity(
                start_time=datetime.fromisoformat(entry["start_time"]),
                mmp_w=np.array([np.nan if v is None else v for v in entry["mmp_w"]], dtype=float),
            ))
        elif reason == "no_watts":
            skipped_no_watts += 1
        elif reason == "bad_recintsecs":
            skipped_bad_recintsecs += 1
        elif reason == "estimated_power":
            skipped_estimated_power += 1
        else:
            skipped_malformed += 1
    activities.sort(key=lambda a: a.start_time)
    stats = ScanStats(len(entries), parsed_ok, skipped_no_watts, skipped_bad_recintsecs,
                       skipped_malformed, skipped_estimated_power)
    return activities, stats


def scan_activities(activities_dir: Path) -> tuple[list[ParsedActivity], ScanStats]:
    """Read + parse every ``GC_activities/*.json`` ONCE per run. A file whose
    (mtime, size) matches its cache entry is never reopened at all -- only
    genuinely new or changed files (and any file from a version-mismatched
    or missing cache) get a full parse. Skip reasons are cached too, so a
    permanently-unusable file (e.g. a pre-power-meter ride) is never
    reopened on any future run either, not just usable ones. Always
    caches to CACHE_FILE -- this function's only caller never varies it."""
    old_entries = _load_cache()
    paths = sorted(activities_dir.glob("*.json"))
    total = len(paths)

    # Guard against activities_dir being transiently inaccessible (it's a
    # symlink to an external network share that has been observed to drop
    # mid-session): if we suddenly see far fewer files than the cache
    # remembers, that's "can't see the real directory right now," not
    # "most rides were deleted" -- fall back to the existing cache as-is
    # rather than overwriting it with the little (or nothing) we found.
    if old_entries and total < len(old_entries) * 0.5:
        logger.warning(
            "only found %d file(s) but the cache has %d entries -- %s may be "
            "temporarily inaccessible (e.g. a dropped network mount). Using "
            "the existing cache as-is instead of overwriting it.",
            total, len(old_entries), activities_dir,
        )
        return _activities_from_entries(old_entries)

    new_entries: dict = {}
    for i, path in enumerate(paths, 1):
        if i % 250 == 0:
            logger.info("scanned %d/%d files...", i, total)

        try:
            st = path.stat()
        except OSError:
            new_entries[path.name] = {
                "mtime": None, "size": None, "reason": "malformed",
                "start_time": None, "mmp_w": None,
            }
            continue

        cached = old_entries.get(path.name)
        if cached is not None and cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size:
            entry = cached
        else:
            activity, reason = _parse_one_file(path)
            if reason == "ok":
                assert activity is not None, "_parse_one_file: reason == 'ok' must come with a parsed activity"
                start_iso = activity.start_time.isoformat()
                mmp_list = [None if np.isnan(v) else float(v) for v in activity.mmp_w]
            else:
                start_iso, mmp_list = None, None
            entry = {
                "mtime": st.st_mtime, "size": st.st_size, "reason": reason,
                "start_time": start_iso, "mmp_w": mmp_list,
            }
        new_entries[path.name] = entry

    _save_cache(new_entries)
    return _activities_from_entries(new_entries)


# --------------------------------------------------------------------
# Period selection
# --------------------------------------------------------------------

def _add_months(dt: datetime, delta_months: int) -> datetime:
    total = dt.year * 12 + (dt.month - 1) + delta_months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])  # clamp e.g. Mar 31 - 1mo -> Feb 28/29
    return dt.replace(year=year, month=month, day=day)


def _period_from_payload(payload: dict, now: datetime) -> PeriodSelection:
    """Raises KeyError/ValueError/TypeError on bad input -- the caller turns
    that into a clean {"ok": False, ...} response, never a crash."""
    mode = payload.get("mode")
    if mode == "recent_weeks":
        weeks = int(payload["weeks"])
        if weeks <= 0:
            raise ValueError("weeks must be > 0")
        return PeriodSelection(start=now - timedelta(weeks=weeks), end=now, label=f"recent {weeks} week(s)")

    if mode == "recent_months":
        months = int(payload["months"])
        if months <= 0:
            raise ValueError("months must be > 0")
        return PeriodSelection(start=_add_months(now, -months), end=now, label=f"recent {months} month(s)")

    if mode == "absolute_range":
        sy, sm = int(payload["start_year"]), int(payload["start_month"])
        ey, em = int(payload["end_year"]), int(payload["end_month"])
        if not (1 <= sm <= 12 and 1 <= em <= 12):
            raise ValueError("month must be in 1..12")
        start = datetime(sy, sm, 1)
        end = _add_months(datetime(ey, em, 1), 1)  # exclusive upper bound: first day of the month AFTER the end month
        if end <= start:
            raise ValueError("end period must not be before start period")
        return PeriodSelection(start=start, end=end, label=f"{sy}-{sm:02d} to {ey}-{em:02d}")

    raise ValueError(f"unknown period mode: {mode!r}")


def compute_period_mmp(selected: list[ParsedActivity]) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    """Elementwise max across selected activities' precomputed MMP arrays.
    Durations with no valid value across the whole selection are dropped
    (never treated as 0). This is the only per-request numeric work needed
    for the MMP curve -- deliberately cheap since parsing already happened
    at startup. Durations are always DURATIONS_S -- this function's only
    caller never varies it.

    Also returns, per surviving duration, the start_time of whichever
    activity actually set that duration's record -- lets the UI show which
    specific ride produced each observed point, so an outlier can be
    traced back to its source file."""
    if not selected:
        return np.array([]), np.array([]), []
    stacked = np.vstack([a.mmp_w for a in selected])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "All-NaN slice" is expected, not a bug
        col_max = np.nanmax(stacked, axis=0)
    valid = ~np.isnan(col_max)
    source_times = [
        selected[int(np.nanargmax(stacked[:, i]))].start_time
        for i in np.flatnonzero(valid)
    ]
    return np.array(DURATIONS_S, dtype=float)[valid], col_max[valid], source_times


# --------------------------------------------------------------------
# Morton 3-parameter critical power model fit
# --------------------------------------------------------------------

def _morton_p_of_t(t: np.ndarray, cp_w: float, w_prime_j: float, p_max_w: float) -> np.ndarray:
    """Morton 3-param CP model, closed-form P(t).

    Derivation: the model's defining relation is
        t = W'/(P-CP) + W'/(CP-Pmax)
    Let tau0 = W'/(Pmax-CP) > 0 (since Pmax > CP); note that
    -W'/(CP-Pmax) = W'/(Pmax-CP) = tau0, so solving for P gives the
    equivalent closed form used here: P(t) = CP + W'/(t + tau0).
    Sanity checks: P(0) = CP + W'/tau0 = CP + (Pmax-CP) = Pmax, and
    P(t->inf) -> CP -- both match the model's physiological meaning
    (instantaneous ceiling at t=0, asymptote to CP for long durations).
    """
    tau0 = w_prime_j / (p_max_w - cp_w)
    return cp_w + w_prime_j / (t + tau0)


_PARAM_NAMES = ("cp_w", "w_prime_j", "p_max_physio_w")


def _envelope_objective(x_free: np.ndarray, free_names: tuple[str, ...], fixed: dict, t: np.ndarray, p_obs: np.ndarray) -> float:
    """One-sided least-squares envelope objective: minimize the TOTAL
    (squared) overshoot of the fitted curve above every observed MMP point,
    while heavily penalizing any point the curve falls BELOW (an
    undershoot is physiologically nonsensical -- MMP is power the rider
    actually produced, so a "power ceiling" model can never claim less
    than that at the same duration).

    A pure minimax/Chebyshev fit (minimizing only the WORST-case
    overshoot) also satisfies the never-undershoot requirement, but a
    3-parameter monotonic curve only needs to touch ~3 points to achieve
    the minimal worst-case gap -- on real data, a minimax fit's curve
    touched only the 5s/30s/45s points, while every duration from 60s to
    1800s (including everything that should inform CP) sat 12-34W below
    the curve with zero influence on the fit. Minimizing TOTAL squared
    overshoot instead means every
    point, including the long-duration ones that actually determine CP,
    pulls the curve down and contributes to the result -- not just
    whichever few points happen to be locally hardest to clear.

    Structured as (total overshoot) + (squared-undershoot penalty),
    mirroring this codebase's existing house convention for soft nonlinear
    constraints baked into a plain scalar objective (see
    core.calibrator's _INFEASIBLE_PHYSICS_PENALTY_MPS hard-region
    penalty and core.simulators' penalty_factor accumulation) rather than
    scipy.optimize.NonlinearConstraint/SLSQP, which nothing in this
    codebase uses.
    """
    vals = dict(zip(free_names, x_free))
    vals.update(fixed)
    cp, wprime, pmax = vals["cp_w"], vals["w_prime_j"], vals["p_max_physio_w"]
    if pmax <= cp or wprime <= 0:
        return _INFEASIBLE_PENALTY  # unphysical region: tau0 = W'/(Pmax-CP) undefined or negative

    gap = _morton_p_of_t(t, cp, wprime, pmax) - p_obs
    overshoot_cost = float(np.sum(np.clip(gap, 0.0, None) ** 2))
    undershoot_penalty = _UNDERSHOOT_PENALTY_WEIGHT * float(np.sum(np.clip(-gap, 0.0, None) ** 2))
    return overshoot_cost + undershoot_penalty


def _envelope_objective_normalized(x_norm: np.ndarray, free_names: tuple[str, ...], fixed: dict, t: np.ndarray, p_obs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """_envelope_objective, evaluated at x_norm's real point (x_norm in
    [0, 1] per free_name, rescaled here via lo + x_norm*(hi-lo)).

    Exists so fit_morton_3p_envelope's Nelder-Mead polish can run in
    [0,1]-normalized space instead of _FIT_BOUNDS' raw, wildly mismatched
    physical units/scales: cp_w's range is 550 W, w_prime_j's is 199500 J,
    p_max_physio_w's is 2800 W, but xatol/fatol below are a single
    absolute tolerance applied across every free dimension at once (see
    core.calibrator's own `_objective_calibration_normalized` for the
    fuller argument and its confirmation against scipy's own Nelder-Mead
    source) -- xatol=1e-4 is between 0.0000036% and 0.00000005% of these
    three ranges, dimensionally meaningless as a shared tolerance.
    Same fix, same reason, as that function.
    """
    x_real = lo + x_norm * (hi - lo)
    return _envelope_objective(x_real, free_names, fixed, t, p_obs)


def fit_morton_3p_envelope(durations_s: np.ndarray, mmp_w: np.ndarray, fixed: dict | None = None) -> dict:
    """Fits cp_w, w_prime_j, p_max_physio_w as the Morton curve with minimal
    total (squared) overshoot above every observed MMP point, while never
    sitting below any of them (a constrained, one-sided least-squares
    envelope fit -- see _envelope_objective for why "minimal total
    overshoot" replaced an earlier pure-minimax version: minimax only needs
    ~3 touching points to be optimal, which let it ignore every
    long-duration point that should inform CP).

    `fixed` optionally pins any subset of {"cp_w", "w_prime_j",
    "p_max_physio_w"} to a caller-supplied value instead of estimating it
    (used for clamping experiments, e.g. "assume W' is 20000 J from a
    longer baseline, what CP/Pmax fit the recent MMP curve under that
    assumption?"). Absent/empty fixed reproduces full 3-parameter
    estimation.

    Solved via the same DE (global search) -> Nelder-Mead (local polish)
    two-stage convention already used in core.calibrator, with identical
    hyperparameters, but a single fixed-seed run rather than that file's
    full outer x inner multi-seed ensemble (justified here by the much
    lower dimensionality: <=3 free parameters vs. that file's 4-12).
    Deterministic given the same data/period, since the seed is fixed.

    Returns {"ok": True, "cp_w", "w_prime_j", "p_max_physio_w", "n_points",
    "max_gap_w", "mean_gap_w", "fixed_params",
    "fit_curve": [{"t_s","p_w"}, ...]} on success, or
    {"ok": False, "error", "n_points"} otherwise -- never raises.
    """
    fixed = fixed or {}
    n = len(durations_s)
    if n < MIN_FIT_POINTS:
        return {"ok": False, "error": f"insufficient data: only {n} valid MMP points (need >= {MIN_FIT_POINTS})", "n_points": n}

    order = np.argsort(durations_s)
    t, p = durations_s[order].astype(float), mmp_w[order].astype(float)

    free_names = tuple(name for name in _PARAM_NAMES if name not in fixed)

    if not free_names:
        vals = dict(fixed)
    else:
        bounds = [_FIT_BOUNDS[name] for name in free_names]
        try:
            de_result = differential_evolution(
                func=_envelope_objective,
                args=(free_names, fixed, t, p),
                bounds=bounds,
                strategy="randtobest1bin",
                mutation=(0.1, 1.9),
                recombination=0.9,
                popsize=5,
                maxiter=300,
                tol=0.001,
                polish=False,
                seed=_FIT_SEED,
                workers=1,
            )
            # NM polish runs in [0,1]-normalized space, not bounds' own raw
            # units -- see _envelope_objective_normalized's docstring.
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            nm_result = minimize(
                _envelope_objective_normalized, x0=(de_result.x - lo) / (hi - lo),
                args=(free_names, fixed, t, p, lo, hi),
                method="Nelder-Mead", bounds=[(0.0, 1.0)] * len(free_names),
                options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000},
            )
            nm_result.x = lo + nm_result.x * (hi - lo)  # back to real units for every caller downstream
        except (RuntimeError, ValueError) as e:
            return {"ok": False, "error": f"fit did not converge: {e}", "n_points": n}

        final = nm_result if nm_result.fun < de_result.fun else de_result
        vals = dict(zip(free_names, (float(x) for x in final.x)))
        vals.update(fixed)

    cp_fit, wprime_fit, pmax_fit = vals["cp_w"], vals["w_prime_j"], vals["p_max_physio_w"]
    if pmax_fit <= cp_fit or wprime_fit <= 0:
        return {"ok": False, "error": "degenerate parameters: p_max_physio_w must exceed cp_w and w_prime_j must be > 0", "n_points": n}

    gap = _morton_p_of_t(t, cp_fit, wprime_fit, pmax_fit) - p
    max_gap = float(np.max(gap))
    mean_gap = float(np.mean(gap))

    t_curve = np.logspace(np.log10(t.min()), np.log10(t.max()), _CURVE_N_POINTS)
    p_curve = _morton_p_of_t(t_curve, cp_fit, wprime_fit, pmax_fit)

    return {
        "ok": True, "cp_w": cp_fit, "w_prime_j": wprime_fit, "p_max_physio_w": pmax_fit,
        "n_points": n, "max_gap_w": max_gap, "mean_gap_w": mean_gap,
        "fixed_params": sorted(fixed.keys()),
        "fit_curve": [{"t_s": float(tt), "p_w": float(pp)} for tt, pp in zip(t_curve, p_curve)],
    }


# --------------------------------------------------------------------
# Request handling
# --------------------------------------------------------------------

def handle_estimate(payload: dict, activities: list[ParsedActivity]) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        period = _period_from_payload(payload, now)
    except (KeyError, ValueError, TypeError) as e:
        return {"ok": False, "error": f"invalid period selection: {e}"}

    selected = [a for a in activities if period.start <= a.start_time < period.end]
    durations, mmp_values, source_times = compute_period_mmp(selected)

    base = {
        "period_label": period.label,
        "n_activities_used": len(selected),
        "n_mmp_points": len(durations),
        "mmp_points": [
            {"t_s": float(d), "p_w": float(p), "date": st.isoformat() + "Z"}
            for d, p, st in zip(durations, mmp_values, source_times)
        ],
    }
    if not selected:
        return {"ok": False, "error": "no activities found in this period", **base}

    try:
        fixed = _parse_fixed_payload(payload.get("fixed"))
    except (KeyError, ValueError, TypeError) as e:
        return {"ok": False, "error": f"invalid fixed-parameter value: {e}", **base}

    fit = fit_morton_3p_envelope(durations, mmp_values, fixed=fixed)
    if not fit["ok"]:
        return {"ok": False, "error": fit["error"], **base}
    return {"ok": True, **base, "fit": fit}


def _parse_fixed_payload(raw: dict | None) -> dict:
    """Validates an optional {"cp_w"?, "w_prime_j"?, "p_max_physio_w"?}
    dict from the request body -- each present value must be a finite,
    positive number. Raises ValueError/TypeError on bad input (caught by
    the caller); returns {} for None/empty input (estimate all 3, today's
    default behavior)."""
    if not raw:
        return {}
    fixed = {}
    for name in _PARAM_NAMES:
        if name not in raw or raw[name] is None or raw[name] == "":
            continue
        value = float(raw[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number, got {raw[name]!r}")
        fixed[name] = value
    return fixed


def make_handler(activities: list[ParsedActivity], stats: ScanStats):
    from hyle.lib.common import QuietHTTPHandler

    html_bytes = HTML_FILE.read_bytes()

    class Handler(QuietHTTPHandler):
        def do_GET(self):  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, html_bytes, "text/html; charset=utf-8")
            elif path == "/meta":
                body = json.dumps({
                    "total_files": stats.total_files,
                    "parsed_ok": stats.parsed_ok,
                    "skipped_no_watts": stats.skipped_no_watts,
                    "skipped_bad_recintsecs": stats.skipped_bad_recintsecs,
                    "skipped_estimated_power": stats.skipped_estimated_power,
                    "skipped_malformed": stats.skipped_malformed,
                    "earliest_activity": activities[0].start_time.isoformat() if activities else None,
                    "latest_activity": activities[-1].start_time.isoformat() if activities else None,
                    "default_period": {"mode": DEFAULT_PERIOD_MODE, "weeks": DEFAULT_PERIOD_WEEKS},
                }).encode("utf-8")
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            parsed = urlsplit(self.path)

            if parsed.path == "/shutdown":
                # See hyle.lib.common.QuietHTTPHandler._handle_shutdown: this tool
                # also keeps a server running after the browser tab opens,
                # so it needs its own signal that the user is done, sent via
                # navigator.sendBeacon() on the page's pagehide event.
                self._handle_shutdown()
                return

            if parsed.path == "/estimate":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = handle_estimate(payload, activities)
                except Exception as e:  # noqa: BLE001 -- report any failure to the browser, don't crash the server
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                body = json.dumps(result).encode("utf-8")
                self._send(200 if result.get("ok") else 400, body, "application/json")
                return

            self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    configure_logging()
    from hyle.lib.common import run_local_server

    if not ACTIVITIES_DIR.exists():
        sys.exit(
            f"error: {ACTIVITIES_DIR} not found -- point {ENV_VAR_GC_ACTIVITIES_DIR} at your "
            "GoldenCheetah activities directory, or place it (or a symlink to it) at "
            "resources/GC_activities"
        )
    if not HTML_FILE.exists():
        sys.exit(f"error: {HTML_FILE.name} not found next to this script")

    logger.info("scanning %s ...", ACTIVITIES_DIR)
    t0 = time.time()
    activities, stats = scan_activities(ACTIVITIES_DIR)
    logger.info(
        "scanned %d files in %.1fs: %d usable, %d no WATTS, "
        "%d unsupported RECINTSECS, %d estimated power, %d malformed",
        stats.total_files, time.time() - t0, stats.parsed_ok, stats.skipped_no_watts,
        stats.skipped_bad_recintsecs, stats.skipped_estimated_power, stats.skipped_malformed,
    )

    run_local_server(make_handler(activities, stats), serving_message="serving at {url}")


if __name__ == "__main__":
    main()
