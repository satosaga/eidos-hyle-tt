#!/usr/bin/env python3
"""
HYLE - hyle.apps.fit2gpx_converter

FIT -> GPX converter with an optional interactive trim range. Opens a
browser UI (Leaflet map, speed/power chart, draggable trim handles) for
selecting a start/end range in a FIT activity, then writes the selected
segment as a GPX file next to the original FIT file. Left untouched, the
selection defaults to the full activity, so simply opening and exporting
acts as a plain FIT -> GPX converter -- trimming is an optional feature
layered on top of that, not the tool's defining behavior (hence the name;
"gpx_trimmer" would suggest a GPX file being trimmed, which is not what
this does).

Usage
-----
    hyle-fit2gpx-converter

Architecture
------------
FIT parsing happens exactly once, in Python, via
core.activity_parser.parse_fit_file -- not duplicated in the browser,
which would risk silently diverging from Python's own timestamp/
power-dropout handling. This script starts a small local HTTP server
(127.0.0.1, random free port) that:

  - serves fit2gpx_converter.html itself
  - serves the parsed record stream (lat/lon/altitude/time/power/speed,
    plus filename) as JSON at /parsed.json -- computed once in main()
    before the server starts, so a bad FIT file (or core/EIDOS^TT not
    being importable) fails fast in the terminal rather than as a
    browser-side error after opening a tab
  - accepts the resulting GPX back via POST /save and writes it directly
    into the same directory as the input FIT file

fit2gpx_converter.html is a pure visualization / trim-selection front
end: it builds its point list directly from /parsed.json's arrays, with
no FIT parsing of its own.

The POST /save step above is the actual reason a local server exists at
all: a plain static HTML page has no way to write a file to a chosen
directory on disk (browser sandboxing intentionally prevents that) --
the best a pure client-side page can do is trigger a generic "Save As"
download, which lands wherever the browser's downloads folder happens to
be. Since Python already knows the input FIT's directory, routing the
save through it is what makes "GPX ends up next to the FIT" possible at
all.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

# Lazy/defensive, like this file's other core.* imports (see main() and
# calc_altitude_offset). Doesn't make the tool work without core -- FIT
# parsing itself requires core.activity_parser, so main() still exits
# before starting the server if core isn't importable (see
# _parse_fit_for_browser's docstring). It only keeps the *module* itself
# importing cleanly either way: a top-level `from core.logging_setup
# import ...` would turn "core missing" into a raw ImportError at import
# time, before main()'s own clear sys.exit() message gets a chance to
# run. Falls back to a plain logging.basicConfig() with the same format
# string, kept in sync by hand with core.logging_setup's LOG_FORMAT/
# LOG_DATEFMT.
try:
    from core.logging_setup import configure_logging
except ImportError:
    def configure_logging(level: int = logging.INFO) -> None:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_FILE = SCRIPT_DIR / "fit2gpx_converter.html"

# --------------------------------------------------------------------
# Altitude-offset auto-calculation (physics-based)
# --------------------------------------------------------------------
# Fixed "average" physics parameters used only for this auto-calc feature.
# Hardcoded here (not read from a reference config file at runtime) since
# no config file is guaranteed to exist wherever this script runs. May
# need retuning later.
#
# Rationale for the method: a candidate
# offset tau is scored by shifting the FIT altitude channel by tau (same
# ele(t+tau) convention as the GPX export / speed-chart overlay), rebuilding
# course geometry from it, replaying the FIT's *actual recorded power*
# through _simulator (recorded-power replay: is_target_power=False), and
# correlating the resulting simulated speed's ACCELERATION against the
# acceleration of the FIT's GPS-derived recorded speed (see
# sim_speed_corr's own comment for why acceleration rather than speed).
# Using power directly (rather than speed vs. altitude alone) avoids
# the confound where speed also depends on how hard the rider was pedaling,
# not just on grade -- a plain speed/altitude correlation can't distinguish
# the two. Because this is a correlation (not an absolute-error fit),
# moderate error in the fixed physics parameters below is tolerated; only
# the *shape* (phase) of the simulated trajectory needs to line up with
# reality, not its exact magnitude.
_OFFSET_CALC_PHYSICAL: dict[str, Any] = {
    "rider_weight": 65.0,
    "cda": 0.35,
    "f_max": 400.0,
    "brake_lookahead": 1.2,
    "brake_usability": 0.7,
    "bike_weight": 9.0,
    "gravity_accel": 9.80665,
    "air_density": 1.225,
    "crr": 0.004,
    "mu": 0.8,
    # min_corner_radius is core.simulators.sim_kiritsubo.PhysicalSettings'
    # own field, not RunSettings' -- see that field's own comment.
    "min_corner_radius": 15.0,
    "wind_speed": 0.0,
    "wind_direction": 0.0,
    "cda_yaw_table_filename": "constant_model.csv",
}
_OFFSET_CALC_PHYSIOLOGICAL: dict[str, Any] = {
    "cp": 247.0,
    "w_prime": 20800.0,
    "p_max": 637.0,
    "w_prime_recovery_rate": 1.28,
    "vitality_loss_rate": 0.0,
}
_OFFSET_CALC_RUN: dict[str, Any] = {
    # gpx_filename is unused below (course geometry here is built directly
    # from the FIT segment, not loaded from any GPX file) but the field is
    # required to construct RunSettings; left blank since RunSettings is
    # instantiated directly (bypassing RunSettings.from_dict's Pydantic
    # validation, which would otherwise require this path to exist on disk).
    "gpx_filename": "",
    "n_seg_min": 1,
    "n_seg_max": 6,
    "initial_base_seed": 42,
    "seed_factor": 3,
    "time_step": 0.1,
    "seg_power_max": 1000.0,
    "seg_power_min": 0.0,
    "seg_length_min": 1.0,
    "distance_step": 1.0,
}

_OFFSET_SEARCH_MIN_S = 0.0
# Upper bound is core.activity_parser.ALTITUDE_LAG_MAX_S (imported lazily
# where used, matching this module's own core-optional import style) --
# shared with eidos.apps.analyzer.window's Altitude-lag spinbox, since
# both search/adjust the same barometric-lag quantity.
_OFFSET_SEARCH_STEP_S = 0.5
# GPS position fixes right after the trim's own start are noisier than the
# rest of the segment (settling time after a cold/warm GPS fix, or just a
# rider stationary/slow-rolling before the real launch, both of which
# distort the GPS-spline distance/speed fit locally). This many metres of
# recorded distance are excluded from BOTH the physics replay (which
# starts its own x=0 at this point, not the trim's own start, carrying the
# GPS-derived speed already reached there as the kernel's v_init) and the
# correlation score itself -- see calc_altitude_offset's own cut_idx
# comment for how.
_LAUNCH_TRIM_DISTANCE_M = 10.0
# Every _TRAJ_SUBSAMPLE-th candidate's full simulated-speed trajectory is
# kept for the offset-analysis popup (offset_s step * this = the spacing
# between overlaid curves). 1 (no subsampling) since the 0-15s/0.5s range
# (31 candidates) is small enough to send every candidate's trajectory --
# the popup's drag-to-select needs an exact, precomputed trajectory at
# every step it can land on, not an interpolated/nearest guess.
_TRAJ_SUBSAMPLE = 1

# calc_altitude_offset_autofit's free parameters: of core.simulators.
# calibratable_physical_keys('sim_kiritsubo')'s 13 keys, only these four
# have a genuine real-world reason to differ ride-to-ride on the SAME
# device/course (actual body/gear weight, aero position, weather).
# bike_weight and the other 8 keys stay at _OFFSET_CALC_PHYSICAL's
# generic defaults.
_AUTOFIT_FREE_KEYS = ["rider_weight", "cda", "wind_speed", "wind_direction"]
# Upper bound on (course geometry @ tau -> AutoFit -> speed-RMSE tau)
# iteration rounds; a safety bound on the fixed-point loop below, not
# expected to bind in practice.
_AUTOFIT_MAX_ROUNDS = 6

# Live progress for whichever calc_altitude_offset_autofit call is
# currently running, polled by the browser's GET /calc_offset_progress
# (see make_handler) while its own POST /calc_offset sits blocked on the
# same computation -- same "status text, no progress bar" idiom
# eidos.apps.analyzer's own Auto Fit uses (AutoFitWorker's progress
# label). Single shared dict, not per-request: this tool only ever has
# one /calc_offset in flight at a time (the browser's own Cancel-toggle
# button prevents a second click from starting another), so there is no
# real "whose progress is this" ambiguity to key by request. Locked
# because it's genuinely written from the POST request's own thread and
# read from a DIFFERENT thread on every GET poll (ThreadingHTTPServer
# gives each request its own thread) -- relying on the GIL to make that
# safe would be implicit, not structural.
_autofit_progress_lock = threading.Lock()
_autofit_progress: dict[str, Any] = {"active": False}


def _reset_autofit_progress() -> None:
    with _autofit_progress_lock:
        _autofit_progress.clear()
        _autofit_progress["active"] = False


def _update_autofit_progress(**kwargs: Any) -> None:
    with _autofit_progress_lock:
        _autofit_progress.update(active=True, **kwargs)


def _read_autofit_progress() -> dict[str, Any]:
    with _autofit_progress_lock:
        return dict(_autofit_progress)


def _nan_to_none(arr) -> list[float | None]:
    """NaN isn't valid JSON; None (-> JSON null) is. Shared by
    _parse_fit_for_browser and calc_altitude_offset's response building."""
    import numpy as np

    return [None if not np.isfinite(v) else float(v) for v in arr]


def _parse_fit_for_browser(fit_path: Path) -> dict:
    """
    Parse fit_path once, in Python, and return a JSON-ready dict of
    parallel arrays for the browser to build its point list from directly
    -- no FIT parsing in the browser required.

    Field names deliberately match the payload shape /calc_offset already
    expects (see calc_altitude_offset's docstring) and core.activity_parser's
    own FlatFITData attribute names, so this is a straight passthrough
    rather than a third independent naming scheme.

    Returns {"ok": False, "error": ...} if the file has no usable records
    (fewer than 2 samples with timestamp + distance + speed all present --
    see parse_fit_file) or if core.activity_parser can't be imported at
    all. Note this is NOT an "optional feature" -- FIT parsing is this
    tool's entire job, and main() calls sys.exit() on this failure before
    run_local_server() ever runs, so without core.activity_parser the
    tool does nothing useful. What the try/except here (and the other
    lazily-imported core.* calls in this file) actually buys is narrower:
    the *module* still imports cleanly regardless, so a missing/broken
    core surfaces as this function's own clean {"ok": False, "error": ...}
    -> main()'s single-line sys.exit() message, not a raw ImportError
    traceback at module-import time.
    """
    try:
        from core.activity_parser import parse_fit_file
    except ImportError as e:
        return {"ok": False, "error": f"core.activity_parser not importable: {e}"}

    import numpy as np

    flat = parse_fit_file(str(fit_path))
    if flat is None:
        return {"ok": False, "error": "FIT file has fewer than 2 valid timestamp+distance+speed records"}

    # parse_fit_file() now re-grids lat_deg/lon_deg onto a gap-free
    # uniform 1Hz axis file-wide (see _normalize_lat_lon_to_uniform_1hz),
    # inserting a synthesized row for any second the device didn't
    # record at all. Only lat_deg/lon_deg get a real value there;
    # distance_m/speed_ms (deliberately never file-wide interpolated,
    # same policy as power_w) are NaN at those rows -- which plain
    # float(v) would serialize as a bare `NaN` JSON token the browser's
    # strict r.json() rejects outright. Rather than null-ing them out
    # (which would break buildPoints's own "every sample here already
    # has timestamp+distance+speed" invariant downstream), drop those
    # synthesized rows entirely: this tool visualizes/exports real
    # recorded trackpoints, and a gap-filled row has no real speed/
    # distance measurement behind it.
    real = np.isfinite(flat.distance_m) & np.isfinite(flat.speed_ms)

    return {
        "ok": True,
        "filename": fit_path.name,
        "timestamp_ms": [float(v) for v in flat.timestamp_ms[real]],
        "time_s": [float(v) for v in flat.time_s[real]],
        "distance_m": [float(v) for v in flat.distance_m[real]],
        "speed_ms": [float(v) for v in flat.speed_ms[real]],
        "power_w": _nan_to_none(flat.power_w[real]),
        "altitude_m": _nan_to_none(flat.altitude_m[real]),
        "lat_deg": _nan_to_none(flat.lat_deg[real]),
        "lon_deg": _nan_to_none(flat.lon_deg[real]),
    }


def _pearson_corr(a, b):
    """Pearson correlation over the finite-valued positions of a and b, or None if degenerate."""
    import numpy as np

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return None
    a_m, b_m = a[mask], b[mask]
    if np.std(a_m) == 0 or np.std(b_m) == 0:
        return None
    return float(np.corrcoef(a_m, b_m)[0, 1])


def _rmse(a, b):
    """RMSE over the finite-valued positions of a and b, or None if
    degenerate. Same finite-position masking convention as _pearson_corr,
    used by calc_altitude_offset_autofit's speed-domain scoring (unlike
    _pearson_corr, scale/offset-SENSITIVE -- see that function's own
    docstring for why that's the point there)."""
    import numpy as np

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return None
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


@dataclass
class _OffsetSearchInputs:
    """Payload-derived inputs shared by every offset candidate in
    calc_altitude_offset's search over sim_speed_corr, and by any other
    caller that wants to build a course_profile for a chosen tau via
    _build_replay_course_and_blocks without re-deriving these itself.
    Built once by _prepare_offset_search_inputs -- independent of which
    offset_s or which physical/physiological parameter values are being
    tried."""

    time_s: Any
    alt_lookup_t: Any
    alt_lookup_z: Any
    lat_deg_clean: Any
    lon_deg_clean: Any
    cut_idx: int
    record: Any  # core.activity_parser.ActivityRecord
    distance_m_from_cut: Any
    v_init: float
    gps_speed_ms: Any
    gps_accel_ms2: Any


def _prepare_offset_search_inputs(payload: dict) -> _OffsetSearchInputs:
    """
    Parse/validate calc_altitude_offset's payload, fit the GPS distance/
    speed curves, apply the launch-GPS trim, and build the ActivityRecord/
    cleaned-track inputs every offset candidate's course construction
    (_build_replay_course_and_blocks) then shares -- split out so this
    computation isn't duplicated by other callers of
    _build_replay_course_and_blocks.

    Raises:
        ValueError: with the same messages calc_altitude_offset returns as
            {"ok": False, "error": ...} for each failure case below --
            calc_altitude_offset itself catches this and rebuilds that
            same dict, so its external error contract is unchanged.
    """
    import numpy as np

    from core.activity_parser import (
        ActivityRecord,
        _compute_gps_distance_speed_m,
        _fill_nan_next_valid,
        _fix_leading_zero_power,
        _interpolate_dropped_and_frozen_gps_fixes,
    )

    try:
        time_s = np.array(payload["time_s"], dtype=np.float64)
        # Read (and shape-validated) for the payload-contract check below
        # only -- see the comment above distance_m's own computation for
        # why the odometer/wheel-sensor field itself is never used as data.
        _ = np.array(payload["distance_m"], dtype=np.float64)
        power_raw = np.array(
            [np.nan if p is None else float(p) for p in payload["power_w"]], dtype=np.float64
        )
        altitude_m = np.array(payload["altitude_m"], dtype=np.float64)
        lat_deg = np.array(payload["lat_deg"], dtype=np.float64)
        lon_deg = np.array(payload["lon_deg"], dtype=np.float64)
        speed_ms = np.array(payload["speed_ms"], dtype=np.float64)

        if payload.get("alt_lookup_time_s") and payload.get("alt_lookup_altitude_m"):
            alt_lookup_t = np.array(payload["alt_lookup_time_s"], dtype=np.float64)
            alt_lookup_z = np.array(payload["alt_lookup_altitude_m"], dtype=np.float64)
        else:
            alt_lookup_t, alt_lookup_z = time_s, altitude_m

        if payload.get("gps_lookup_time_s") and payload.get("gps_lookup_lat_deg") and payload.get("gps_lookup_lon_deg"):
            gps_lookup_t = np.array(payload["gps_lookup_time_s"], dtype=np.float64)
            gps_lookup_lat = np.array(payload["gps_lookup_lat_deg"], dtype=np.float64)
            gps_lookup_lon = np.array(payload["gps_lookup_lon_deg"], dtype=np.float64)
            # altitude only feeds _compute_gps_distance_speed_m's own
            # NaN-mean fallback (see _compute_gps_distance_speed_time_
            # param's docstring) -- unlike lat/lon, it's fine for this to
            # come up short/absent.
            gps_lookup_alt_raw = payload.get("gps_lookup_altitude_m")
            gps_lookup_alt = (
                np.array(gps_lookup_alt_raw, dtype=np.float64)
                if gps_lookup_alt_raw else np.full_like(gps_lookup_lat, np.nan)
            )
        else:
            gps_lookup_t, gps_lookup_lat, gps_lookup_lon, gps_lookup_alt = None, None, None, None
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid payload: {e}") from e

    # GPS-spline distance, NOT the FIT file's own odometer/wheel-sensor
    # `distance` field (payload["distance_m"] above, read only for
    # payload-shape validation, never as data) — same
    # _compute_gps_distance_speed_m helper as core.activity_parser.
    # _make_activity_record's own GPS-position fit on the eidos side: the
    # odometer disagrees with GPS-derived distance and drifts
    # non-monotonically against the rider's true GPS position, which
    # would feed the wrong local course gradient into the recorded-power
    # replay below (via
    # build_zoh_power_blocks -> _rescale_distance_to_target, which only
    # corrects the TOTAL, not this per-block local drift). Computed
    # once, here, from the untouched (not tau-shifted) recorded altitude
    # — this is the Activity's own geometry, independent of which
    # altitude-lag hypothesis is being tested below, so it doesn't need
    # to be redone inside the offsets loop. This same fit also yields a
    # speed curve (narrow_speed_ms below) as a byproduct of the unified
    # x(t)/y(t)/z(t) B-spline fit.
    narrow_result = _compute_gps_distance_speed_m(time_s, lat_deg, lon_deg, altitude_m)
    if narrow_result is None:
        raise ValueError("could not fit a GPS distance curve for this segment "
                          "(too little/degenerate GPS data)")
    distance_m, narrow_speed_ms = narrow_result[0], narrow_result[1]

    # The tau-selection metric (Pearson correlation, taken over this
    # curve's own ACCELERATION -- see calc_altitude_offset's sim_speed_corr
    # for why) is scored against this higher-resolution GPS-derived speed,
    # not raw FIT speed_ms -- the latter is more heavily device-smoothed
    # than a direct GPS-position derivative, which would bias which tau
    # scores best against a tau-shifted-altitude physics replay.
    #
    # Fit on the WIDER gps_lookup_* window (see calc_altitude_offset's own
    # docstring) when available, evaluated only at the trim's own time_s
    # -- not fit-and-evaluated on the trim directly, which is truncated
    # exactly at the trim boundary and so leaves the unconstrained
    # quintic spline with no real data past the cut edge to anchor
    # against (see core.activity_parser.
    # _compute_gps_distance_speed_time_param's eval_t/
    # _fit_time_b_spline's own docstrings for the full account -- same
    # fix applied for eidos.apps.analyzer.canvas). A speed derivative
    # doesn't care about the two distance curves sharing an origin (only
    # the shared time_s axis matters), so this wide-window fit needs no
    # d_origin reconciliation with distance_m above -- only its SPEED
    # half is used; its own distance half is discarded.
    gps_speed_ms = None
    if gps_lookup_t is not None:
        assert gps_lookup_lat is not None and gps_lookup_lon is not None and gps_lookup_alt is not None  # all four travel together, see their assignment above
        wide_result = _compute_gps_distance_speed_m(
            gps_lookup_t, gps_lookup_lat, gps_lookup_lon, gps_lookup_alt, eval_t=time_s,
        )
        gps_speed_ms = wide_result[1] if wide_result is not None else None
    if gps_speed_ms is None:
        # No gps_lookup_* payload, or its own GPS fit failed (e.g. too-
        # degenerate a window) -- fall back to the SAME narrow-window fit
        # distance_m above already came from (this unified fit already
        # produced a speed curve as a byproduct of computing distance_m,
        # so no second fit is needed).
        gps_speed_ms = narrow_speed_ms
    if gps_speed_ms is None:
        raise ValueError("could not fit a GPS speed curve for this segment "
                          "(trim shorter than the GPS-speed knot spacing)")

    # Candidates are scored on ACCELERATION (d/dt of speed), not speed
    # itself -- see calc_altitude_offset's sim_speed_corr for why. Computed
    # once here (gps_speed_ms doesn't depend on the candidate offset),
    # same rationale as gps_speed_ms/distance_m/lat_deg_clean above.
    gps_accel_ms2 = np.gradient(gps_speed_ms, time_s)

    n = len(time_s)
    if n < 20 or not (len(distance_m) == len(power_raw) == len(altitude_m)
                       == len(lat_deg) == len(lon_deg) == len(speed_ms) == n):
        raise ValueError("segment too short or arrays misaligned")
    if len(alt_lookup_t) != len(alt_lookup_z) or len(alt_lookup_t) < 2:
        raise ValueError("alt_lookup arrays missing or misaligned")

    # cut_idx: first sample at or past _LAUNCH_TRIM_DISTANCE_M of recorded
    # distance -- everything before it (unstable launch GPS, see that
    # constant's own comment) is excluded below from both the physics
    # replay's own course geometry and the correlation score. The score
    # exclusion needs no separate masking: sim_speed_on_record_grid stays
    # np.nan for indices < cut_idx by construction (see
    # calc_altitude_offset's sim_speed_corr), and _pearson_corr already
    # drops any index where either input is non-finite.
    cut_idx = int(np.searchsorted(distance_m, _LAUNCH_TRIM_DISTANCE_M))
    if n - cut_idx < 20:
        raise ValueError(f"segment too short after excluding the first "
                          f"{_LAUNCH_TRIM_DISTANCE_M:.0f}m (unstable launch GPS)")

    # Same leading-zero-power / NaN-dropout fixups core.activity_parser
    # applies when building an ActivityRecord from a FIT slice, reused here
    # rather than reimplemented. power_raw itself (from the browser's
    # /calc_offset payload) is still the raw, unfixed power_w that
    # _parse_fit_for_browser served -- that function is a straight
    # passthrough of parse_fit_file's output, deliberately not applying
    # ActivityRecord-level fixups, so this is the one place they happen.
    # Applied to the FULL (not yet cut_idx-sliced) array -- the leading-
    # zero-power detection looks for a genuine pattern at the ride's own
    # start, which would misfire if run on an already-truncated array
    # starting mid-ride.
    power_fixed = _fill_nan_next_valid(_fix_leading_zero_power(power_raw))

    # distance_m_from_cut/v_init: the physics replay's own inputs, re-
    # origined at cut_idx (GPS-derived distance re-zeroed there, kernel
    # v_init taken from the GPS-derived speed already reached there) --
    # both offset-invariant (cut_idx/gps_speed_ms don't depend on the
    # candidate altitude-shift), so computed once here rather than inside
    # sim_speed_corr's per-candidate loop.
    distance_m_from_cut = distance_m[cut_idx:] - distance_m[cut_idx]
    v_init = float(gps_speed_ms[cut_idx])

    # record: sliced to [cut_idx:] and re-origined the same way, so
    # build_zoh_power_blocks's own d_src[0]==0 requirement holds and its
    # power blocks line up with distance_m_from_cut/the course geometry
    # _build_replay_course_and_blocks builds (also cut at cut_idx) -- not
    # the trim's own un-cut start.
    record = ActivityRecord(
        source_path="<hyle.apps.fit2gpx_converter trim>",
        start_time=datetime(1970, 1, 1),
        elapsed_time_s=float(time_s[-1] - time_s[cut_idx]),
        total_distance_m=float(distance_m_from_cut[-1]),
        time_s=time_s[cut_idx:] - time_s[cut_idx],
        distance_m=distance_m_from_cut,
        power_w=power_fixed[cut_idx:],
        speed_ms=speed_ms[cut_idx:],
        altitude_m=altitude_m[cut_idx:],
        lat_deg=lat_deg[cut_idx:],
        lon_deg=lon_deg[cut_idx:],
        heart_rate_bpm=None,
        # core.calibrator's Auto Fit (see calc_altitude_offset_autofit)
        # requires this unconditionally -- sliced/re-origined the same
        # way as every other field above, from the SAME gps_speed_ms
        # already computed above (no second fit).
        gps_speed_ms=gps_speed_ms[cut_idx:],
    )

    # Dropped/frozen-fix cleaning -- same
    # _interpolate_dropped_and_frozen_gps_fixes helper the distance_m/
    # gps_speed_ms fit above already gets internally (via
    # _compute_gps_distance_speed_m), applied here too so
    # _build_replay_course_and_blocks's own course-geometry construction
    # doesn't build a course straight from a raw, uncleaned lat/lon track
    # -- a bit-exact frozen fix followed by an oversized catch-up jump
    # (see that helper's own docstring) would otherwise show up as a
    # spurious kink in every tau candidate's course_profile, biasing
    # curvature/v_limit near that point for all of them alike. A
    # SEPARATE cleaned copy, not a reassignment of lat_deg/lon_deg
    # themselves -- those still feed the raw ActivityRecord construction
    # above, which is meant to carry the rider's actual recorded track,
    # not a cleaned one (same convention core.activity_parser.
    # ActivityRecord.lat_deg/lon_deg already follow). Computed once,
    # outside the offsets loop, same rationale as distance_m_from_cut/
    # v_init above. If cleaning fails (fewer than 5 valid samples), fall
    # back to the raw track unchanged -- distance_m/gps_speed_ms's own
    # fit above already succeeded by this point using the SAME lat_deg/
    # lon_deg, so this should not actually be reachable in practice, but
    # _build_replay_course_and_blocks's own course construction has no
    # other guard against a None return here to fall back to.
    cleaned = _interpolate_dropped_and_frozen_gps_fixes(lat_deg, lon_deg)
    lat_deg_clean, lon_deg_clean = cleaned if cleaned is not None else (lat_deg, lon_deg)

    return _OffsetSearchInputs(
        time_s=time_s, alt_lookup_t=alt_lookup_t, alt_lookup_z=alt_lookup_z,
        lat_deg_clean=lat_deg_clean, lon_deg_clean=lon_deg_clean, cut_idx=cut_idx,
        record=record, distance_m_from_cut=distance_m_from_cut, v_init=v_init,
        gps_speed_ms=gps_speed_ms, gps_accel_ms2=gps_accel_ms2,
    )


def _build_replay_course_and_blocks(offset_s: float, prep: _OffsetSearchInputs, physical, run, simulator_spec):
    """
    Build one offset candidate's course_profile + PowerBlocks: shift
    prep's altitude lookup table by offset_s (same ele(t+offset)
    convention as the GPX export / speed-chart overlay), rebuild course
    geometry from the shifted altitude plus prep's cleaned lat/lon, and
    rescale PowerBlocks to that course's own distance axis.

    Shared by calc_altitude_offset's own sim_speed_corr (one call per tau
    candidate, using its own _OFFSET_CALC_PHYSICAL-derived physical/run)
    and by core.calibrator-based callers that want a FIXED
    course_profile for one chosen tau, built with a DIFFERENT (e.g.
    calibrated) physical/run. Extracted so both stay byte-identical
    instead of two independently-maintained copies of this
    course-construction logic.

    Args:
        offset_s: candidate altitude-lead offset [s].
        prep: this payload's _OffsetSearchInputs (see
            _prepare_offset_search_inputs).
        physical, run: the physical/RunSettings instances to build this
            candidate's course physics with -- the caller's own, not
            necessarily calc_altitude_offset's _OFFSET_CALC_PHYSICAL
            defaults.
        simulator_spec: resolve_simulator(...)'s result for the simulator
            being scored.

    Returns:
        (course_profile, power_blocks), or (None, None) if this
        candidate's course geometry is degenerate.
    """
    import numpy as np

    from core.activity_parser import build_zoh_power_blocks
    from core.data_manager import _clean_and_project
    from core.schema import CoursePoints

    # Same ele(t+offset) lookup convention as the GPX export / speed
    # overlay in fit2gpx_converter.html -- but looked up against the
    # WIDER alt_lookup_* margin arrays, not the trimmed time_s/altitude_m,
    # so shifted queries near the trim boundaries land on real recorded
    # altitude instead of np.interp's flat-clamped extrapolation (see the
    # alt_lookup_* note in calc_altitude_offset's own docstring).
    z_shifted = np.interp(prep.time_s + offset_s, prep.alt_lookup_t, prep.alt_lookup_z)

    # _clean_and_project is the single authoritative lat/lon -> local
    # Cartesian projection (also used for GPX course loading); reused here
    # rather than reimplemented. Its point-merge step depends only on
    # horizontal distance, never on z, so this is safe to call fresh per
    # candidate even though only z_shifted changes each time.
    # lat_deg_clean/lon_deg_clean (not raw lat_deg/lon_deg) -- see
    # _prepare_offset_search_inputs' own frozen-fix cleaning comment.
    # Sliced to [cut_idx:] -- the course geometry (and so the replay
    # itself) starts at cut_idx, not the trim's own start, same rationale
    # as prep.record's own [cut_idx:] slicing.
    raw_pts = np.column_stack([
        prep.lat_deg_clean[prep.cut_idx:], prep.lon_deg_clean[prep.cut_idx:], z_shifted[prep.cut_idx:],
    ])
    x_c, y_c, z_c, s_h_c, s_p_c = _clean_and_project(raw_pts, 1.0)
    if len(s_p_c) < 5 or s_p_c[-1] <= 0:
        return None, None

    points = CoursePoints(
        x=x_c, y=y_c, z=z_c, s_h=s_h_c, s_p=s_p_c,
        distance=float(s_p_c[-1]),
        origin_lat=float(prep.lat_deg_clean[prep.cut_idx]), origin_lon=float(prep.lon_deg_clean[prep.cut_idx]),
    )
    course_profile = simulator_spec.compute_course_physics(points, physical, run)

    # PowerBlocks must be rescaled to THIS candidate's own
    # course_profile.distance, not built once outside the loop.
    # course_profile.distance is the 3D geometric arc length recomputed
    # from _clean_and_project(lat, lon, z_shifted) fresh for every tau,
    # which does not exactly equal the FIT's raw recorded distance
    # ("odometer drift" -- see _rescale_distance_to_target). If the two
    # distance axes disagree, the njit loop's position x (driven by the
    # power-block distance axis) can run past the end of the
    # course_profile arrays; get_interpolated_value() then clamps to the
    # LAST slope/v_limit sample and holds it constant for the rest of the
    # ride (see core.simulators). Rescaling PowerBlocks to
    # course_profile.distance every time keeps both axes exactly
    # aligned, so the clamp is never reached.
    power_blocks = build_zoh_power_blocks(prep.record, target_distance_m=course_profile.distance)

    return course_profile, power_blocks


def _simulate_speed_for_offset(offset_s: float, prep: _OffsetSearchInputs, physical, physiological, run, simulator_spec):
    """
    Run the recorded-power replay for one candidate offset and return its
    simulated speed on prep's own time_s grid (np.nan before prep.cut_idx),
    or None if this candidate is degenerate. Shared by both
    calc_altitude_offset's correlation-domain search and
    calc_altitude_offset_autofit's speed-RMSE-domain search, so "build
    course + replay kernel + rescale onto the record grid" has one
    implementation instead of two copies.

    Args:
        offset_s: Candidate altitude-timestamp offset [s] being tried.
        prep: _OffsetSearchInputs this candidate is evaluated against.
        physical, physiological: this candidate's physics settings (the
            caller's own -- e.g. calibrator-overridden values -- not
            necessarily calc_altitude_offset's own _OFFSET_CALC_PHYSICAL
            defaults).
        run: RunSettings for the replay.
        simulator_spec: Resolved SimulatorSpec supplying the kernel and
            course-physics builders this replay runs against.
    """
    import numpy as np

    from core.activity_parser import _rescale_distance_to_target

    course_profile, power_blocks = _build_replay_course_and_blocks(offset_s, prep, physical, run, simulator_spec)
    if course_profile is None:
        return None

    params = simulator_spec.build_physics_params(physical, physiological, run, course_profile)

    # Recorded-power replay: use_sync_hook=False, is_target_power=False
    # (drives the physics from the actual recorded power, unclamped).
    # v_init (not 0.0): the kernel starts at cut_idx's own position, not a
    # standing start, carrying whatever speed the rider had already
    # reached there (see prep.v_init's own comment in
    # _prepare_offset_search_inputs).
    output = simulator_spec.kernel(prep.v_init, power_blocks, params, True, False, False)
    if len(output.t_traj) < 2:
        return None

    # output.x_traj lives in the RESCALED distance coordinate system
    # build_zoh_power_blocks() built the power blocks in (distance_m
    # uniformly stretched so its last sample lands exactly on
    # course_profile.distance -- see _rescale_distance_to_target). The
    # un-rescaled distance array (GPS-spline-derived -- see
    # _compute_gps_distance_speed_m in _prepare_offset_search_inputs) is
    # NOT that same coordinate system: the two agree at their shared
    # origin by construction but drift apart toward the end of the
    # segment. Querying output.x_traj with the un-rescaled distance would
    # sample the WRONG point on the simulated trajectory, increasingly so
    # near the end. Rescale prep.distance_m_from_cut (already re-origined
    # at cut_idx, matching course_profile's own origin) the exact same way
    # before using it as the query axis.
    d_scaled, _ = _rescale_distance_to_target(prep.distance_m_from_cut, course_profile.distance)

    # Full time_s-length, not n-cut_idx: indices < cut_idx stay np.nan
    # (never written below), which is what excludes them from the score
    # without any separate masking -- see cut_idx's own comment in
    # _prepare_offset_search_inputs.
    sim_speed_on_record_grid = np.full_like(prep.time_s, np.nan)
    in_range = d_scaled <= output.x_traj[-1]
    if in_range.sum() < 5:
        return None
    tail = np.full(len(prep.time_s) - prep.cut_idx, np.nan)
    tail[in_range] = np.interp(d_scaled[in_range], output.x_traj, output.v_traj)
    sim_speed_on_record_grid[prep.cut_idx:] = tail
    return sim_speed_on_record_grid


def calc_altitude_offset(
    payload: dict, physical_overrides: dict | None = None, physiological_overrides: dict | None = None,
) -> dict:
    """
    Search for the altitude-lead offset tau [s] that makes recorded-power
    replay's (_simulator, actual FIT power, tau-shifted FIT altitude as
    course geometry) ACCELERATION best match the acceleration of the FIT's
    GPS-derived recorded speed.

    Args:
        payload: dict with equal-length lists (already sliced to the exact
            GPX export range, elapsed/distance both starting at 0):
            time_s, distance_m, power_w (null where missing), altitude_m,
            lat_deg, lon_deg, speed_ms.

            Also: alt_lookup_time_s / alt_lookup_altitude_m -- a WIDER pair
            of arrays (same time origin as time_s, but extending margin
            seconds past both ends of the trim, drawn from the full,
            untrimmed activity where available) used only as the altitude
            lookup table for the tau shift. Without this margin,
            np.interp's flat-value extrapolation beyond a candidate's
            shifted-time range silently flattens altitude near the trim
            boundaries for large ``|tau|``, which biases the search
            toward the edges of the offset range on some courses. Falls
            back to time_s / altitude_m if omitted (then
            large ``|tau|`` candidates near the trim edges are less trustworthy).

            Also: gps_lookup_time_s / gps_lookup_lat_deg /
            gps_lookup_lon_deg / gps_lookup_altitude_m -- the same
            WIDER-window idea as alt_lookup_*, applied to GPS position
            instead of altitude. core.activity_parser's GPS distance/speed
            fit (_compute_gps_distance_speed_m) uses an unconstrained
            spline with no boundary condition at either end (see
            _fit_time_b_spline's docstring) -- a design that relies on
            every caller fitting over a window padded past its actual
            segment of interest, so the fit's own less-constrained edge
            never lands inside the range actually used. Fitting directly
            on the trim-truncated lat/lon/time (this function's own
            behaviour when these are omitted) leaves the GPS distance/
            speed curves exposed to instability right at the trim
            boundary. Falls back to lat_deg / lon_deg / altitude_m /
            time_s if omitted (then large-|tau| candidates near the trim
            edges are less trustworthy, same caveat as alt_lookup_*
            above).
        physical_overrides, physiological_overrides: optional dicts merged
            onto _OFFSET_CALC_PHYSICAL/_OFFSET_CALC_PHYSIOLOGICAL before
            constructing this search's physical/physiological settings --
            None (default, the browser's own /calc_offset caller) uses
            those generic fixed values unchanged. Exists so a caller that
            has already fit this rider's own physical parameters (e.g.
            via core.calibrator) can re-run this same tau search against
            them instead of the generic defaults, without a second,
            divergent copy of the search itself.

    Returns:
        On success: a dict with keys "ok" (True), "best_offset_s",
        "best_corr", "curve", "trajectories", and "recorded_speed_ms"
        (the GPS-derived speed each candidate's acceleration score is
        derived from -- kept in the speed domain here, for the browser's
        diagnostic chart to plot as its "Recorded" reference line instead
        of its own local speed_ms copy). On failure (segment unusable): a
        dict with keys "ok" (False) and "error".
    """
    import numpy as np

    # Imported lazily (not at module scope) so this script still runs as a
    # plain FIT->GPX converter even in an environment where the
    # core / EIDOS^TT package isn't on the path -- only this auto-calc
    # feature needs it.
    from core.activity_parser import ALTITUDE_LAG_MAX_S
    from core.data_manager import load_cda_yaw_table
    from core.schema import RunSettings
    from core.simulators import DEFAULT_SIMULATOR_KEY, resolve_simulator

    simulator_spec = resolve_simulator(DEFAULT_SIMULATOR_KEY)

    try:
        prep = _prepare_offset_search_inputs(payload)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    physical = simulator_spec.physical_param_model(**{**_OFFSET_CALC_PHYSICAL, **(physical_overrides or {})})
    physiological = simulator_spec.physiological_param_model(
        **{**_OFFSET_CALC_PHYSIOLOGICAL, **(physiological_overrides or {})}
    )
    run = RunSettings(**_OFFSET_CALC_RUN)
    # Validation only, upfront (a friendly {"ok": False, ...} error instead
    # of a raw exception from deep inside sim_speed_corr's per-candidate
    # loop below) -- simulator_spec.build_physics_params loads this table
    # itself (see core.simulators' module docstring), so the array itself
    # is discarded here.
    try:
        load_cda_yaw_table(physical.cda_yaw_table_filename)
    except OSError as e:
        return {"ok": False, "error": f"could not load CdA yaw table '{physical.cda_yaw_table_filename}': {e}"}

    def sim_speed_corr(offset_s: float):
        """Returns (corr, sim_speed_on_record_grid) for one candidate offset,
        or (None, None). sim_speed_on_record_grid is the SPEED trajectory
        (kept in this domain purely for the browser's tau-overlay
        diagnostic chart, where a speed curve is what a human can actually
        judge by eye); corr itself is computed over its ACCELERATION
        (np.gradient against prep.gps_accel_ms2), not the speed curve
        directly. Differentiating removes the slow, broadband component
        (grade-driven speed change) that two merely similarly-shaped speed
        curves share across a wide range of candidate time shifts, which
        otherwise leaves the correlation-vs-offset curve nearly flat and
        the true optimum hard to localize. tau is a property of the
        recording device (altitude-sensor/GPS timing lag), not of any one
        course, so a sharper, more localized optimum is what lets repeated
        estimates on different courses actually agree with each other.
        """
        sim_speed_on_record_grid = _simulate_speed_for_offset(
            offset_s, prep, physical, physiological, run, simulator_spec
        )
        if sim_speed_on_record_grid is None:
            return None, None
        sim_accel_ms2 = np.gradient(sim_speed_on_record_grid, prep.time_s)
        corr = _pearson_corr(sim_accel_ms2, prep.gps_accel_ms2)
        return corr, sim_speed_on_record_grid

    # _nan_to_none is defined at module scope (shared with
    # _parse_fit_for_browser) -- JS side already filters/skips null values
    # in chart data.
    # Exact _OFFSET_SEARCH_STEP_S multiples (not merely close to it, unlike
    # course_geometry's dense/fine grids -- the diagnostic popup's drag-to-
    # select needs an exact, precomputed trajectory at every step it can
    # land on): compute the point count up front and use linspace, never
    # arange with a non-integer stop (float step accumulation makes
    # arange's actual point count/endpoint unreliable).
    n_offset_steps = round((ALTITUDE_LAG_MAX_S - _OFFSET_SEARCH_MIN_S) / _OFFSET_SEARCH_STEP_S)
    offsets = np.linspace(_OFFSET_SEARCH_MIN_S, ALTITUDE_LAG_MAX_S, n_offset_steps + 1)
    curve: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []  # subsampled sim-speed curves, for the tau-overlay diagnostic chart
    best_offset, best_corr, best_sim_speed, best_i = None, None, None, None
    for i, off in enumerate(offsets):
        off_s = float(off)
        corr, sim_speed = sim_speed_corr(off_s)
        curve.append({"offset_s": off_s, "corr": corr})
        if corr is not None and (best_corr is None or corr > best_corr):
            best_offset, best_corr, best_sim_speed, best_i = off_s, corr, sim_speed, i
        # Keep every _TRAJ_SUBSAMPLE-th candidate's full trajectory (dense
        # curve above is cheap -- a scalar per candidate -- but sending
        # every candidate's full speed trace would bloat the response for
        # long segments, so only a diagnostic subsample is kept here).
        if sim_speed is not None and i % _TRAJ_SUBSAMPLE == 0:
            trajectories.append({"offset_s": off_s, "sim_speed_ms": _nan_to_none(sim_speed)})

    if best_offset is None:
        return {"ok": False, "error": "no offset candidate produced a valid correlation"}
    assert best_i is not None  # travels with best_offset -- see the tuple assignment above

    # Always include the best offset's own trajectory even if it fell
    # between subsample points. best_i's trajectory is already in
    # `trajectories` iff best_i was itself a subsample point -- an exact
    # index check, not a float comparison: trajectories' offset_s and
    # best_offset both trace back to the same offsets[i] float (see the
    # loop above), so this was never actually a "do two independent reals
    # coincide" question to begin with.
    if best_i % _TRAJ_SUBSAMPLE != 0:
        trajectories.append({"offset_s": best_offset, "sim_speed_ms": _nan_to_none(best_sim_speed)})
        trajectories.sort(key=lambda t: t["offset_s"])

    # No heuristic reliability warnings here (plateau width, edge-of-range,
    # flat-curve checks, etc.) -- the full curve and trajectory overlays are
    # handed to the user via the analysis popup precisely so they can judge
    # confidence/ambiguity themselves. Per this project's own premise (see
    # README: "the 'Intelligent' in the name refers to the user, not the
    # software"), the system doesn't second-guess that judgment.
    return {
        "ok": True,
        "best_offset_s": best_offset,
        "best_corr": best_corr,
        "curve": curve,
        "trajectories": trajectories,
        # The signal every candidate above was actually scored against
        # (see the "tau-selection metric" comment above) -- sent back so
        # the browser's diagnostic chart can plot the SAME "Recorded"
        # reference line the search itself used, instead of silently
        # falling back to its own local copy of speed_ms (the biased
        # signal this switch exists to stop scoring against).
        "recorded_speed_ms": _nan_to_none(prep.gps_speed_ms),
    }


def calc_altitude_offset_autofit(payload: dict) -> dict:
    """
    Refine calc_altitude_offset's tau estimate by jointly fitting this
    ride's own physical parameters (_AUTOFIT_FREE_KEYS, via
    core.calibrator's "Auto Fit") instead of trusting calc_altitude_offset's
    generic _OFFSET_CALC_PHYSICAL defaults.

    Alternates two steps to a fixed point:
      1. Build course geometry at the current tau, then run Auto Fit
         (core.calibrator.calibrate) against it -- this step's own
         objective is RMSE between simulated and GPS-derived speed (see
         that module's "Objective function" docstring section), not
         correlation, because Auto Fit is fitting the ABSOLUTE MAGNITUDE-
         determining physical constants themselves, not just tau's own
         timing.
      2. Re-run the full offset grid with the now-calibrated physics,
         scored by that SAME speed-domain RMSE metric (not
         calc_altitude_offset's acceleration-domain Pearson correlation,
         which is scale/offset-invariant by design -- appropriate when
         the physics are still the generic, uncalibrated defaults, but
         once the physics are calibrated the magnitude-sensitive RMSE
         metric is the one that matches what Auto Fit itself just
         optimized against).
    Stops once step 2's own best offset stops changing between rounds, or
    after _AUTOFIT_MAX_ROUNDS rounds.

    tau_0 (the starting point, and the course geometry the first Auto Fit
    round is built against) comes from an ordinary calc_altitude_offset(
    payload) call -- generic physics -- reused rather than reimplemented.

    Args:
        payload: same shape as calc_altitude_offset's own payload arg.

    Returns:
        On success: a dict with keys "ok" (True), "best_offset_s",
        "best_score" (the winning offset's speed-domain RMSE [m/s], lower
        is better -- NOT a correlation coefficient, see "curve" below),
        "curve" (list of {"offset_s", "score"} from the FINAL round only
        -- "score" is this same RMSE metric, deliberately named
        differently from calc_altitude_offset's "corr" so a caller can't
        mistake one metric for the other), "trajectories" and
        "recorded_speed_ms" (same shape/meaning as calc_altitude_offset's
        own), "physics_overrides" (the final round's Auto Fit result --
        not surfaced in the browser UI, kept here for debugging/
        inspection), and "n_rounds". On failure: "ok" (False) and
        "error", same convention as calc_altitude_offset.
    """
    import numpy as np

    from core.activity_parser import ALTITUDE_LAG_MAX_S
    from core.calibrator import calibrate
    from core.schema import RunSettings
    from core.simulators import DEFAULT_SIMULATOR_KEY, resolve_simulator

    _reset_autofit_progress()
    try:
        _update_autofit_progress(stage="initial_tau", round=0, max_rounds=_AUTOFIT_MAX_ROUNDS)
        out0 = calc_altitude_offset(payload)
        if not out0["ok"]:
            return out0

        try:
            prep = _prepare_offset_search_inputs(payload)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        simulator_spec = resolve_simulator(DEFAULT_SIMULATOR_KEY)
        run = RunSettings(**_OFFSET_CALC_RUN)

        tau = out0["best_offset_s"]
        physical_overrides: dict = {}
        n_offset_steps = round((ALTITUDE_LAG_MAX_S - _OFFSET_SEARCH_MIN_S) / _OFFSET_SEARCH_STEP_S)
        offsets = np.linspace(_OFFSET_SEARCH_MIN_S, ALTITUDE_LAG_MAX_S, n_offset_steps + 1)

        curve: list[dict[str, Any]] = []
        trajectories: list[dict[str, Any]] = []
        best_score: float | None = None
        round_i = 0
        for round_i in range(_AUTOFIT_MAX_ROUNDS):
            physical = simulator_spec.physical_param_model(**{**_OFFSET_CALC_PHYSICAL, **physical_overrides})
            physiological = simulator_spec.physiological_param_model(**_OFFSET_CALC_PHYSIOLOGICAL)
            course_profile, _ = _build_replay_course_and_blocks(tau, prep, physical, run, simulator_spec)
            if course_profile is None:
                return {"ok": False, "error": f"could not build course geometry at tau={tau:.1f}s"}
            base_physics = simulator_spec.build_physics_params(physical, physiological, run, course_profile)

            round_start = time.monotonic()
            _update_autofit_progress(stage="calibrating", round=round_i + 1, trial=0, n_trials=0,
                                      best_rmse=None, elapsed_s=0.0)

            def _on_calib_progress(n_done, n_total, best_rmse_so_far, _round=round_i + 1, _t0=round_start):
                _update_autofit_progress(stage="calibrating", round=_round, trial=n_done, n_trials=n_total,
                                          best_rmse=best_rmse_so_far, elapsed_s=time.monotonic() - _t0)

            calib_result = calibrate(
                course_distance_m=course_profile.distance,
                simulator_key=simulator_spec.key,
                base_physics=base_physics,
                raw_physical=_OFFSET_CALC_PHYSICAL,
                raw_physiological=_OFFSET_CALC_PHYSIOLOGICAL,
                raw_run=_OFFSET_CALC_RUN,
                course_profile=course_profile,
                fixed_overrides={},
                activity_raw=prep.record,
                free_keys=_AUTOFIT_FREE_KEYS,
                progress_callback=_on_calib_progress,
            )
            # float(): calib_result.physics_overrides' values are numpy
            # scalars (from zip()-ing free_keys against a numpy array of
            # optimized values) -- json.dumps() in the /calc_offset HTTP
            # handler can't serialize those directly.
            physical_overrides = {k: float(v) for k, v in calib_result.physics_overrides.items()}
            physical = simulator_spec.physical_param_model(**{**_OFFSET_CALC_PHYSICAL, **physical_overrides})

            _update_autofit_progress(stage="rescoring", round=round_i + 1)
            curve = []
            trajectories = []
            best_offset, best_score, best_sim_speed, best_i = None, None, None, None
            for i, off in enumerate(offsets):
                off_s = float(off)
                sim_speed = _simulate_speed_for_offset(off_s, prep, physical, physiological, run, simulator_spec)
                score = _rmse(sim_speed, prep.gps_speed_ms) if sim_speed is not None else None
                curve.append({"offset_s": off_s, "score": score})
                if score is not None and (best_score is None or score < best_score):
                    best_offset, best_score, best_sim_speed, best_i = off_s, score, sim_speed, i
                if sim_speed is not None and i % _TRAJ_SUBSAMPLE == 0:
                    trajectories.append({"offset_s": off_s, "sim_speed_ms": _nan_to_none(sim_speed)})

            if best_offset is None:
                return {"ok": False, "error": "no offset candidate produced a valid RMSE"}
            assert best_i is not None  # travels with best_offset -- see the tuple assignment above

            # Same "always include the best offset's own trajectory" fixup as
            # calc_altitude_offset's own grid-search loop.
            if best_i % _TRAJ_SUBSAMPLE != 0:
                trajectories.append({"offset_s": best_offset, "sim_speed_ms": _nan_to_none(best_sim_speed)})
                trajectories.sort(key=lambda t: t["offset_s"])

            if best_offset == tau:
                tau = best_offset
                break
            tau = best_offset

        return {
            "ok": True,
            "best_offset_s": tau,
            "best_score": best_score,
            "curve": curve,
            "trajectories": trajectories,
            "recorded_speed_ms": _nan_to_none(prep.gps_speed_ms),
            "physics_overrides": physical_overrides,
            "n_rounds": round_i + 1,
        }
    finally:
        _reset_autofit_progress()


def make_handler(fit_path: Path, parsed_payload_bytes: bytes):
    from hyle.lib.common import QuietHTTPHandler

    html_bytes = HTML_FILE.read_bytes()

    class Handler(QuietHTTPHandler):
        def do_GET(self):  # noqa: N802 -- required BaseHTTPRequestHandler name
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, html_bytes, "text/html; charset=utf-8")
            elif path == "/parsed.json":
                # Parsed once in main() before the server even starts (see
                # there) -- this is a plain cached-bytes serve, not a
                # per-request re-parse.
                self._send(200, parsed_payload_bytes, "application/json")
            elif path == "/calc_offset_progress":
                # Polled by the browser while its own POST /calc_offset
                # sits blocked on calc_altitude_offset_autofit -- served
                # on a DIFFERENT thread than that POST (ThreadingHTTPServer
                # gives every request its own thread), which is exactly
                # why _autofit_progress needs its own lock rather than
                # relying on the GIL.
                body = json.dumps(_read_autofit_progress()).encode("utf-8")
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            parsed = urlsplit(self.path)

            if parsed.path == "/shutdown":
                # See hyle.lib.common.QuietHTTPHandler._handle_shutdown: unlike the
                # eidos.apps.* GUIs (closing the Qt window ends the event loop,
                # process exits) or hyle.apps.course_checker (no server -- it
                # writes a file and returns immediately), this tool keeps a
                # server running after the browser tab opens, so it needs
                # its own signal that the user is done, sent via
                # navigator.sendBeacon() on the page's pagehide event.
                self._handle_shutdown()
                return

            if parsed.path == "/calc_offset":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    result = calc_altitude_offset_autofit(payload)
                except Exception as e:  # noqa: BLE001 -- report any failure to the browser, don't crash the server
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                body = json.dumps(result).encode("utf-8")
                self._send(200 if result.get("ok") else 400, body, "application/json")
                return

            if parsed.path != "/save":
                self._send(404, b"not found", "text/plain")
                return

            filename = parse_qs(parsed.query).get("filename", [None])[0]
            if not filename:
                self._send(400, b"missing ?filename=", "text/plain")
                return

            length = int(self.headers.get("Content-Length", 0))
            gpx_text = self.rfile.read(length).decode("utf-8")

            output_path = fit_path.parent / filename
            try:
                output_path.write_text(gpx_text, encoding="utf-8")
            except OSError as e:
                self._send(500, str(e).encode("utf-8"), "text/plain")
                return

            # Remember whatever altOffsetInput held at export time (see
            # fit2gpx_converter.html's exportBtn handler, which appends
            # ?offset=<altOffset> to this same request) so the NEXT
            # hyle.apps.fit2gpx_converter or eidos.apps.analyzer session
            # starts from it instead of DEFAULT_POWER_OFFSET_S -- see
            # core.io_config.load_power_offset_s. Imported lazily, same
            # reason as the parsed["power_offset_s"] seeding in main(). By
            # this point core.activity_parser has already succeeded
            # (that's the only way we got here -- see main()'s sys.exit()
            # on a failed parse), so this is genuinely optional: if
            # core.io_config specifically can't be imported, or the offset
            # value is unparseable, this just leaves the existing
            # remembered value (if any) untouched rather than failing the
            # save itself -- unlike the FIT-parsing step, this one really
            # is a best-effort nicety, not the tool's core job.
            offset_str = parse_qs(parsed.query).get("offset", [None])[0]
            if offset_str is not None:
                try:
                    from core.io_config import save_power_offset_s
                    save_power_offset_s(float(offset_str))
                except (ImportError, ValueError):
                    pass

            body = json.dumps({"ok": True, "path": str(output_path)}).encode("utf-8")
            self._send(200, body, "application/json")
            logger.info("wrote %s", output_path)

    return Handler


def main() -> None:
    configure_logging()
    from hyle.lib.common import prompt_via_drag_drop, run_local_server, window_title

    fit_path = prompt_via_drag_drop(
        message="Drop a .fit file here",
        win_title=window_title("FIT → GPX Converter"),
    )
    if fit_path is None:
        sys.exit("no file selected")

    if not fit_path.exists():
        sys.exit(f"error: {fit_path} not found")
    if not HTML_FILE.exists():
        sys.exit(f"error: {HTML_FILE.name} not found next to this script")

    # Parsed once, up front -- fail fast in the terminal with a clear
    # message rather than starting a server whose only symptom of a bad
    # FIT file (or a missing core/EIDOS^TT package) would be a browser-side
    # error after opening a tab.
    parsed = _parse_fit_for_browser(fit_path)
    if not parsed.get("ok"):
        sys.exit(f"error: {parsed.get('error', 'failed to parse FIT file')}")

    # Seeds altOffsetInput's initial value in the browser (see
    # fit2gpx_converter.html's buildPoints) with whatever offset was last
    # remembered here or in eidos.apps.analyzer's own Altitude lag
    # spinbox -- see core.io_config.load_power_offset_s -- rather than a
    # value hardcoded into the HTML. Falls back to DEFAULT_POWER_OFFSET_S
    # (6.0s) the first time either tool ever runs. Imported lazily, like
    # calc_altitude_offset's own core.* imports above. We already know
    # core.activity_parser is importable at this point (the sys.exit()
    # above already checked parsed["ok"]), so this specifically guards
    # against core.io_config alone being unavailable/broken -- a genuinely
    # optional nicety (a remembered UI default), unlike the FIT parsing
    # step itself.
    try:
        from core.io_config import load_power_offset_s
        parsed["power_offset_s"] = load_power_offset_s()
    except ImportError:
        pass
    parsed_payload_bytes = json.dumps(parsed).encode("utf-8")

    run_local_server(
        make_handler(fit_path, parsed_payload_bytes),
        serving_message=f"serving {fit_path.name} at {{url}}",
        extra_messages=[f"GPX exports will be written to {fit_path.parent}/"],
    )


if __name__ == "__main__":
    main()