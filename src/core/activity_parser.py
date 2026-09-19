########################
# activity_parser.py
########################
"""
Parse real-activity FIT files for use in the EIDOS^TT Analyzer.

Course extraction pipeline (find_course_matches):

    Stage 1 - _generate_candidates():
        Anchor on goal passages (unambiguous: one crossing per lap, at
        speed), then scan backward from each for the true standing-start
        stop (speed_ms at or below stop_speed_threshold_ms). A forward
        scan from the start line can't distinguish the real departure
        from pre-race jostling near the line, so this scans backward
        from the goal instead.
        Goal: high Recall — never miss the actual race lap.

    Stage 2 - _refine_candidates():
        Sub-sample-refine the end index only (GPS-nearest point to the
        goal, within the passage Stage 1 already identified) -- the start
        index needs no refinement here (see Stage 1's own docstring).
        Goal: accurate lap-end determination.

    Stage 2.5 (inside find_course_matches):
        Reject any candidate whose GPS-spline distance disagrees with
        course_distance_m by more than a fixed 10%.

    Stage 3 - _validate_candidates():
        Compute trajectory similarity (discrete Fréchet distance) between
        the candidate track and the course polyline.
        Goal: reject coincidentally matching laps on different courses.

Each accepted candidate's start boundary is then itself sub-sample
corrected via find_stationary_cluster()/estimate_start_time() (a
stillness-boundary detection plus constant-acceleration fit) rather than
Stage 1's own provisional speed-threshold index -- see those functions'
own docstrings.

Public API:
    parse_fit_file()      -> FlatFITData       raw FIT samples as flat arrays
    find_course_matches() -> list[ActivityCandidate]
"""

import bisect
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import overload

import numpy as np
from fit_tool.field import Field
from fit_tool.fit_file import FitFile
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.record_message import RecordMessage
from scipy.interpolate import BSpline, interp1d

from core import FLOAT_TIE_BREAKER_EPS
from core.schema import PowerBlocks

logger = logging.getLogger(__name__)

# Seconds between the Unix epoch (1970-01-01 00:00:00 UTC) and the FIT
# epoch (1989-12-31 00:00:00 UTC). fit_tool's TimestampField applies this
# automatically (offset=-631065600000, scale=0.001) so msg.timestamp comes
# back as ms-since-1970. ActivityLocalTimestampField does NOT apply any
# such conversion (offset=0, scale=1) — msg.local_timestamp is the raw
# FIT-epoch second count. The two must be put on the same footing before
# diffing them for a UTC offset, or the "offset" is nonsense.
FIT_EPOCH_OFFSET_S = 631065600

# Devices and this project's own FIT writers alike record at 1 Hz;
# parse_fit_file() drops any record within this many ms of the last KEPT
# record's timestamp (exact-duplicate or near-duplicate timestamps are a
# known device/writer glitch, never a genuine sub-second sample here).
RECORD_MIN_SPACING_MS = 500

# fit_tool's Field.read_strings_from_bytes() hard-codes strict UTF-8
# decoding for STRING-typed fields (e.g. UserProfileMessage's device
# nickname). Some devices write these fields in a non-UTF-8 encoding,
# which crashes parsing of the whole file before the record data this
# module actually needs is ever reached. These string fields are unused
# here, so decoding them losslessly is not required, only not crashing
# on them. Patched here rather than in fit_tool itself, since this
# project doesn't control that dependency.
def _read_strings_from_bytes_tolerant(self, bytes_buffer: bytes, offset: int = 0, size: int | None = None) -> None:
    # offset/size kwargs match fit_tool==0.9.16's Field.read_all_from_bytes
    # call site (self.read_strings_from_bytes(bytes_buffer, offset=offset, size=self.size)).
    end = None if size is None else offset + size
    bytes_buffer = bytes(bytes_buffer[offset:end])
    string_container = bytes_buffer.decode('utf-8', errors='replace')
    strings = string_container.split('\u0000')
    strings = strings[:-1]
    strings = [x for x in strings if x]
    self.encoded_values = []
    self.encoded_values.extend(strings)


Field.read_strings_from_bytes = _read_strings_from_bytes_tolerant

# ---------------------------------------------------------------------------
# I. Data structures
# ---------------------------------------------------------------------------

@dataclass
class FlatFITData:
    """
    All RecordMessage samples from a FIT file as parallel numpy arrays.

    Attributes:
        source_path:  Absolute path of the originating FIT file.
        timestamp_ms: Wall-clock timestamps [ms, UTC].
        time_s:       Elapsed time from first sample [s].
        distance_m:   Cumulative road distance [m].
        speed_ms:     Speed [m/s].
        power_w:      Power [W], NaN where not recorded.
        cadence_rpm:  Cadence [rpm], NaN where not recorded per-sample,
                      None if the device never reported cadence at all
                      for this file.
        altitude_m:   Altitude [m], NaN where not recorded.
        lat_deg:      Latitude [deg], NaN where not recorded.
        lon_deg:      Longitude [deg], NaN where not recorded.
        heart_rate:   Heart rate [bpm], None if channel absent.
        utc_offset_s: The recording device's local-time offset from UTC
                      [s], read from the FIT Activity message's
                      local_timestamp - timestamp. None if the file has no
                      Activity message or it lacks local_timestamp.
        n:            Number of samples.
    """
    source_path: str
    timestamp_ms: np.ndarray
    time_s: np.ndarray
    distance_m: np.ndarray
    speed_ms: np.ndarray
    power_w: np.ndarray
    cadence_rpm: np.ndarray | None
    altitude_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    heart_rate: np.ndarray | None
    utc_offset_s: float | None

    @property
    def n(self) -> int:
        return len(self.time_s)


# find_stationary_cluster's own search-back budget: how far before the
# Stage-1 provisional start point (si) to look for the standing-start
# stillness boundary -- covers a FIT device whose own speed_ms channel
# takes a few seconds to react after GPS position has already started
# moving. Device-dependent.
STATIONARY_SEARCH_BACK_S = 10.0

# find_stationary_cluster's minimum-stillness window and outlier
# threshold. window_s=7.0s reflects real standing-start TT procedure: a
# rider is in position with a holder steadying the bike well before the
# gun, and the countdown itself runs the last few seconds, so several
# seconds of genuine stillness are guaranteed by how a start is run.
# mad_k=3.0 (with the standard 1.4826 MAD-to-sigma scale factor) is a
# conventional robust-outlier threshold.
STATIONARY_WINDOW_S = 7.0
STATIONARY_MAD_K = 3.0

# estimate_start_time's acceleration-cluster size and t0 grid search:
# fits a constant-acceleration parabola to n_fit real samples after the
# stationary cluster's own boundary, then grid-searches the parabola's
# t0 (true departure instant) up to t0_search_back_s before / t0_search_
# fwd_s after that boundary's own timestamp. n_fit encodes an assumed
# minimum standing-start sprint duration in seconds (n_fit-1, since
# points are 1 Hz). t0_search_back_s is bounded by GPS position
# accuracy, not device or race-procedure behavior.
ACCELERATION_N_FIT = 5
_START_TIME_T0_SEARCH_BACK_S = 3.0
_START_TIME_T0_SEARCH_FWD_S = 2.0
_START_TIME_T0_GRID_STEP_S = 0.005

# Barometric altitude lags GPS/speed by up to this many seconds -- never
# leads, which device architecture rules out -- device- and course-
# dependent. Shared by eidos.apps.analyzer.window's Altitude-lag spinbox
# (manual correction) and hyle.apps.fit2gpx_converter's own delay-
# ESTIMATION search, so both apply the same bound.
ALTITUDE_LAG_MAX_S = 15.0

# Margin an unconstrained (no boundary condition) quintic B-spline fit
# (_fit_time_b_spline, used throughout this module's GPS distance/speed
# pipeline) needs beyond whatever point its output must be trusted at,
# before that point is free of edge disagreement -- see
# _fit_time_b_spline's own docstring for why the fit has no boundary
# condition.
GPS_TIME_PARAM_EDGE_MARGIN_S = 10.0

# Real seconds of FIT history find_course_matches needs before the lap
# start (si); ActivityRecord.pad_time_s/pad_distance_m/pad_altitude_m/
# pad_lat_deg/pad_lon_deg extend before it by this much. A chain, not a
# sum of four independent "seconds before si" requirements -- each term's
# reference point is the PREVIOUS term's own worst case, not si directly:
#   1. STATIONARY_SEARCH_BACK_S: how far back find_stationary_cluster
#      searches from si for a candidate stillness boundary.
#   2. + STATIONARY_WINDOW_S: that boundary, in the worst case found
#      right at search_back_s's own edge, needs this much MORE real
#      history behind it to pass the stillness test.
#   3. + _START_TIME_T0_SEARCH_BACK_S: estimate_start_time's t0 fit can
#      place the true departure instant (t_start_point_s) up to this
#      much earlier than the stillness boundary above.
#   4. + GPS_TIME_PARAM_EDGE_MARGIN_S: d_start_m (_make_activity_record)
#      evaluates the GPS-spline distance exactly AT t_start_point_s, so
#      trusting that evaluation needs the spline's own fitting window to
#      extend this much further still.
# find_stationary_cluster/estimate_start_time work on raw lat/lon only,
# never the GPS-spline (see find_stationary_cluster's own docstring), so
# only the last term needs a spline-fitting margin.
LEAD_IN_TIME_S = (
    STATIONARY_SEARCH_BACK_S + STATIONARY_WINDOW_S
    + _START_TIME_T0_SEARCH_BACK_S + GPS_TIME_PARAM_EDGE_MARGIN_S
)

# Real seconds of FIT history needed AFTER the lap goal (ei), for the
# same consumers as LEAD_IN_TIME_S above. Two chained terms:
#   1. ALTITUDE_LAG_MAX_S: eidos.apps.analyzer's Altitude-lag correction
#      re-plots an altitude sample recorded at time t at distance_at(t -
#      lag) -- covering the full lap distance range for the largest lag
#      the UI allows requires altitude SAMPLES recorded up to
#      ALTITUDE_LAG_MAX_S seconds past the goal (never before the start:
#      a sample recorded before the lap start only ever maps to a
#      negative, off-course display position, which is simply clipped).
#   2. + GPS_TIME_PARAM_EDGE_MARGIN_S: that sample's own display position
#      comes from the same GPS-spline distance pipeline (pad_distance_m),
#      evaluated at goal_time + ALTITUDE_LAG_MAX_S -- same edge-margin
#      need as LEAD_IN_TIME_S's own last term.
TRAIL_OUT_TIME_S = ALTITUDE_LAG_MAX_S + GPS_TIME_PARAM_EDGE_MARGIN_S


@dataclass
class ActivityRecord:
    """
    A single extracted lap as parallel arrays, ready for the simulator.

    Arrays are at the FIT file's own (variable-interval) sample spacing.

    Attributes:
        source_path:        Absolute path of the originating FIT file.
        start_time:         UTC wall-clock time at the start of this lap.
        elapsed_time_s:     Lap duration [s].
        total_distance_m:   Lap road distance [m].
        time_s:             Elapsed time from lap start [s].
        distance_m:         Cumulative distance from lap start [m].
        power_w:            Power [W], NaN filled with the next valid
                            sample (0 W if none remains).
        speed_ms:           Speed [m/s].
        altitude_m:         Altitude [m].
        lat_deg:            Latitude [deg] -- from the SAME GPS-spline curve
                            distance_m was chord-summed from (see
                            _make_activity_record), not the raw recorded
                            position, so a point and its own distance_m
                            label always agree on where it is.
        lon_deg:            Longitude [deg]; see lat_deg.
        heart_rate_bpm:     Heart rate [bpm], None if absent.
        t_start_point_s:    The fitted true departure time [s], same origin
                            as FlatFITData.time_s -- see estimate_start_time().
        utc_offset_s:       Recording device's local-time offset from UTC [s],
                            carried through from FlatFITData. None if unknown.
        pad_time_s:         time_s extended with LEAD_IN_TIME_S seconds
                            before the lap start and TRAIL_OUT_TIME_S
                            seconds after the goal (see those constants;
                            the two margins differ, so this isn't
                            symmetric). Same t=0 origin as time_s, so
                            entries run negative before the start and past
                            elapsed_time_s after the goal. None if the
                            source FlatFITData wasn't available. Used by
                            _make_activity_record's d_start_m evaluation and
                            by eidos.apps.analyzer's Altitude-lag display
                            shift (AnalysisCanvas.update_plots); never by
                            build_zoh_power_blocks or any other
                            physics-relevant consumer, which use
                            time_s/distance_m/power_w instead.
        pad_distance_m:     distance_m's counterpart to pad_time_s, same
                            start-line-relative origin.
        pad_altitude_m:     altitude_m's counterpart to pad_time_s.
        pad_lat_deg:        lat_deg's counterpart to pad_time_s -- same
                            GPS-spline curve as lat_deg, not raw position.
        pad_lon_deg:        lon_deg's counterpart to pad_time_s; see
                            pad_lat_deg.
        dense_distance_m:   distance_m's own curve, re-evaluated at
                            GPS_TRACK_DENSE_STEP_S spacing instead of the
                            FIT's ~1Hz sample spacing, so a polyline
                            through it follows the fitted curve's actual
                            shape rather than straight chords between
                            sparse samples. None if unavailable (falls
                            back to distance_m).
        dense_lat_deg:      lat_deg's own curve at that same dense
                            spacing; see dense_distance_m.
        dense_lon_deg:      lon_deg's own curve at that same dense
                            spacing; see dense_distance_m.
        gps_speed_ms:       GPS-position-derived speed [m/s] -- distinct
                            from speed_ms (the FIT sensor's own, more
                            heavily device-smoothed reading). Populated
                            by every real ActivityRecord constructor
                            (this module's own _make_activity_record, from
                            the SAME GPS-spline curve as lat_deg/
                            distance_m; hyle.apps.fit2gpx_converter's own,
                            from the speed fit it already computes for its
                            own offset search) -- never None in practice.
                            Read directly by consumers that need this
                            speed (eidos.apps.analyzer's Velocity panel,
                            core.calibrator's Auto Fit ground truth)
                            instead of each re-fitting their own spline,
                            which for a find_course_matches-built record
                            would double-smooth on top of its
                            already-smoothed lat_deg/lon_deg.
    """
    source_path: str
    start_time: datetime
    elapsed_time_s: float
    total_distance_m: float
    time_s: np.ndarray
    distance_m: np.ndarray
    power_w: np.ndarray
    speed_ms: np.ndarray
    altitude_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    heart_rate_bpm: np.ndarray | None
    t_start_point_s: float = 0.0
    utc_offset_s: float | None = None
    pad_time_s: np.ndarray | None = None
    pad_distance_m: np.ndarray | None = None
    pad_altitude_m: np.ndarray | None = None
    pad_lat_deg: np.ndarray | None = None
    pad_lon_deg: np.ndarray | None = None
    dense_distance_m: np.ndarray | None = None
    dense_lat_deg: np.ndarray | None = None
    dense_lon_deg: np.ndarray | None = None
    gps_speed_ms: np.ndarray | None = None


@dataclass
class ActivityCandidate:
    """
    A validated course-match candidate.

    Attributes:
        record:          Extracted ActivityRecord for this lap.
        combined_score:  Final ranking score [0-1].
    """
    record: ActivityRecord
    combined_score: float


# ---------------------------------------------------------------------------
# II. FIT parsing
# ---------------------------------------------------------------------------

def parse_fit_file(fit_path: str) -> FlatFITData | None:
    """
    Load a FIT file and return all RecordMessage samples as flat arrays.

    A record within RECORD_MIN_SPACING_MS of the immediately preceding
    KEPT record (exact-duplicate or near-duplicate timestamp -- a known
    device/writer glitch, since recording is 1 Hz) is dropped. Comparison
    is always against the last KEPT record, never the last-seen one, so a
    run of close-together records collapses to its first member rather
    than letting slow drift accumulate past the threshold one small step
    at a time. Every downstream consumer can then assume any two samples
    in time_s are more than RECORD_MIN_SPACING_MS/1000 seconds apart.

    Returns None if the file contains fewer than 2 valid records.

    Args:
        fit_path: Path to the FIT file.
    """
    fit_file = FitFile.from_file(fit_path)

    timestamps, distances, speeds, powers, cadences = [], [], [], [], []
    altitudes, lats, lons, heart_rates = [], [], [], []
    has_hr = False
    has_cadence = False
    utc_offset_s: float | None = None
    # RecordMessage.timestamp is an int (ms since epoch, see fit_tool),
    # never a float -- so this stays an exact integer comparison even
    # though it's a threshold rather than an equality check. prev_ts is
    # only ever reassigned in the KEEP branch below (see docstring on why).
    prev_ts: int | None = None
    n_thinned_ts = 0

    for record in fit_file.records:
        msg = record.message

        if isinstance(msg, ActivityMessage):
            # Local wall-clock offset the recording device was set to.
            # timestamp is ms since the Unix epoch (fit_tool's
            # TimestampField converts it); local_timestamp is left as a
            # raw FIT-epoch second count (see FIT_EPOCH_OFFSET_S) shifted
            # by the device's UTC offset at record time -- both must be
            # put on the same epoch/units before diffing, which is what
            # FIT_EPOCH_OFFSET_S and the /1000.0 below do. Typically one
            # Activity message per file -- if more than one, the last one
            # wins, same as any other FIT field overwrite.
            ts_ms = msg.timestamp
            local_raw = msg.local_timestamp
            if ts_ms is not None and local_raw is not None:
                local_unix_s = local_raw + FIT_EPOCH_OFFSET_S
                utc_offset_s = local_unix_s - (ts_ms / 1000.0)
            continue

        if not isinstance(msg, RecordMessage):
            continue
        ts   = msg.timestamp
        dist = msg.distance
        # Some FIT writers (e.g. this project's own eidos.apps.exporter, via
        # save_strategy_fit()) populate only enhanced_speed and never write
        # the legacy speed field; other devices do the reverse. The two
        # fields are not reliably both present, so fall back rather than
        # requiring one specific field.
        spd = msg.speed
        if spd is None:
            spd = msg.enhanced_speed
        if ts is None or dist is None or spd is None:
            continue
        if prev_ts is not None and ts - prev_ts <= RECORD_MIN_SPACING_MS:
            n_thinned_ts += 1
            continue
        prev_ts = ts

        timestamps.append(float(ts))
        distances.append(float(dist))
        speeds.append(float(spd))
        powers.append(float(msg.power)    if msg.power    is not None else float('nan'))
        cad = msg.cadence
        if cad is not None:
            has_cadence = True
        cadences.append(float(cad) if cad is not None else float('nan'))
        altitudes.append(float(msg.altitude) if msg.altitude is not None else float('nan'))
        lat = msg.position_lat
        lon = msg.position_long
        lats.append(float(lat) if lat is not None else float('nan'))
        lons.append(float(lon) if lon is not None else float('nan'))
        hr = msg.heart_rate
        if hr is not None:
            has_hr = True
        heart_rates.append(float(hr) if hr is not None else float('nan'))

    if n_thinned_ts:
        logger.info(
            "parse_fit_file: dropped %d record(s) within %dms of the "
            "preceding kept record in %s",
            n_thinned_ts, RECORD_MIN_SPACING_MS, fit_path,
        )

    if len(timestamps) < 2:
        logger.warning("parse_fit_file: fewer than 2 valid records in %s", fit_path)
        return None

    t_ms  = np.array(timestamps,  dtype=np.float64)
    t_s   = (t_ms - t_ms[0]) / 1000.0

    # power_arr keeps NaN as-is (no file-wide interpolation). File-wide gaps
    # (e.g. minutes of stationary pre-race waiting with the power meter off)
    # must not be smeared across the eventual race segment boundary; any
    # NaN/leading-zero fixing is done later, scoped to the extracted race
    # segment only
    # (see _make_activity_record / _fix_leading_zero_power / _fill_nan_next_valid).
    power_arr = np.array(powers, dtype=np.float64)

    (t_s, t_ms, distance_arr, speed_arr, power_arr, cadence_arr, altitude_arr,
     lat_arr, lon_arr, hr_arr) = _normalize_lat_lon_to_uniform_1hz(
        t_s, t_ms,
        np.array(distances, dtype=np.float64),
        np.array(speeds,    dtype=np.float64),
        power_arr,
        np.array(cadences, dtype=np.float64) if has_cadence else None,
        np.array(altitudes, dtype=np.float64),
        np.array(lats, dtype=np.float64),
        np.array(lons, dtype=np.float64),
        np.array(heart_rates, dtype=np.float64) if has_hr else None,
    )

    return FlatFITData(
        source_path  = fit_path,
        timestamp_ms = t_ms,
        time_s       = t_s,
        distance_m   = distance_arr,
        speed_ms     = speed_arr,
        power_w      = power_arr,
        cadence_rpm  = cadence_arr,
        altitude_m   = altitude_arr,
        lat_deg      = lat_arr,
        lon_deg      = lon_arr,
        heart_rate   = hr_arr,
        utc_offset_s = utc_offset_s,
    )


def _normalize_lat_lon_to_uniform_1hz(
    t_s: np.ndarray,
    timestamp_ms: np.ndarray,
    distance_m: np.ndarray,
    speed_ms: np.ndarray,
    power_w: np.ndarray,
    cadence_rpm: np.ndarray | None,
    altitude_m: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    heart_rate: np.ndarray | None,
) -> tuple:
    """
    Re-grid every per-sample array onto a strict 1 Hz time axis (integer
    seconds from t_s[0] to t_s[-1], no gaps). Real device recordings can
    drop whole seconds outright (e.g. an auto-pause that stops writing
    position/speed for its duration) -- a different defect from the
    per-sample NaN/frozen-fix cases _interpolate_dropped_and_frozen_gps_
    fixes handles, which assume a sample exists at every index.

    Only lat_deg/lon_deg get a real value at a synthesized (previously
    missing) second, via linear interpolation from the nearest valid real
    samples on each side (np.interp). Every other field stays NaN at a
    synthesized second, preserving this module's file-wide gap policy for
    them (see this function's call site's comment on power_w): position
    has no physics consumer downstream (lat_deg/lon_deg only ever feed a
    minimap display, a GPX/JSON export, or a speed_ms re-derivation), so
    smoothing over a gap there is harmless, whereas smearing a fabricated
    power_w/speed_ms/altitude_m across a real gap is not.

    A candidate whose analysis window overlaps a large interpolated
    lat/lon stretch is not specially flagged here -- find_course_matches's
    Stage 2.5 (course-distance) and Stage 3 (Fréchet-distance) checks
    already reject a candidate whose reconstructed shape doesn't match
    the real course, which is exactly what a fabricated straight-line
    interpolation across real curved terrain would produce.

    Returns the same 10 arrays (t_s/timestamp_ms replaced by the exact
    uniform grid; every other array's real values carried over
    unchanged at their own original second, NaN at any synthesized
    second) in the same order as this function's own signature.
    """
    n_uniform = int(round(t_s[-1] - t_s[0])) + 1
    t_uniform = t_s[0] + np.arange(n_uniform, dtype=np.float64)

    # Assumes real GPS devices record close to 1 Hz, so each original
    # sample lands on its own integer-second slot in practice --
    # RECORD_MIN_SPACING_MS only guarantees kept samples are >0.5s
    # apart, not >~1.0s, so two samples straddling a rounding boundary
    # (e.g. 0.51s apart) could in principle round to the same slot and
    # silently overwrite one another below; not observed against real
    # device data so far. slot is used both to place real values
    # (regrid) and to test for gaps (a slot count of 10 for 10 uniform
    # seconds means no gap, 8 for 10 means 2 missing).
    slot = np.round(t_s - t_s[0]).astype(np.int64)

    def regrid(values: np.ndarray) -> np.ndarray:
        out = np.full(n_uniform, np.nan, dtype=np.float64)
        out[slot] = values
        return out

    valid = np.isfinite(lat_deg) & np.isfinite(lon_deg)
    if valid.sum() < 2:
        lat_u, lon_u = regrid(lat_deg), regrid(lon_deg)
    else:
        lat_u = np.interp(t_uniform, t_s[valid], lat_deg[valid])
        lon_u = np.interp(t_uniform, t_s[valid], lon_deg[valid])

    timestamp_ms_u = timestamp_ms[0] + t_uniform * 1000.0

    return (
        t_uniform,
        timestamp_ms_u,
        regrid(distance_m),
        regrid(speed_ms),
        regrid(power_w),
        regrid(cadence_rpm) if cadence_rpm is not None else None,
        regrid(altitude_m),
        lat_u,
        lon_u,
        regrid(heart_rate) if heart_rate is not None else None,
    )


# ---------------------------------------------------------------------------
# III. Stage 1 — Candidate generation (high Recall)
# ---------------------------------------------------------------------------

def _geo_dist_m(
    lat1: "float | np.floating", lon1: "float | np.floating",
    lat2: "float | np.floating", lon2: "float | np.floating",
) -> float:
    """
    Approximate flat-earth distance in metres between two lat/lon points.

    Uses a local tangent plane projection around the midpoint latitude.
    Accurate to within ~0.1% for distances up to a few km.
    """
    lat_mid = np.radians((lat1 + lat2) / 2.0)
    m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
    m_per_lon = 111412.84 * np.cos(lat_mid)
    dy = (lat1 - lat2) * m_per_lat
    dx = (lon1 - lon2) * m_per_lon
    return float(np.sqrt(dx**2 + dy**2))


def _interpolate_dropped_and_frozen_gps_fixes(
    lat_deg: np.ndarray, lon_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Clean a raw (lat_deg, lon_deg) track of two known device-side GPS
    defects, in place of the raw values: dropped fixes (NaN) and frozen
    fixes (a sample bit-exactly repeating its predecessor). Returns
    (lat_arr, lon_arr) — same length as the input, same index order, no
    resampling — or None if fewer than 5 samples remain valid.

    Dropped fix (NaN lat/lon): lat/lon must stay index-aligned 1:1 with
    the input, so a dropped fix is linearly interpolated from its nearest
    valid neighbours rather than filtered out (edge NaNs fall back to the
    nearest valid value via np.interp's default). An unfilled NaN would
    otherwise poison every downstream cumsum entry from that sample on.

    Frozen fix: a sample bit-exactly repeating the immediately preceding
    (valid) sample's lat/lon is a device GPS-fix "freeze", not a real
    momentary stop (the fix repeats for one sample, then the next sample
    "catches up" with an oversized jump, while speed_ms shows the rider
    still moving at full speed throughout). No run-length threshold is
    needed to separate "brief glitch" from "genuine stop": this codebase's
    domain assumption is that a standing-start TT race never stops
    mid-race (see _generate_candidates' docstring), so a genuine pre-race
    stop still interpolates harmlessly (a run of bit-identical fixes on
    both sides of an interpolated gap has the same start/end position
    regardless). Folded into the same np.interp pass as the NaN handling,
    so an adjacent frozen run and NaN run interpolate across the combined
    gap in one shot, and a longer run of repeats falls out naturally too
    (each repeat after the first also compares equal to its own immediate
    predecessor, so the whole run is marked invalid).
    """
    n = len(lat_deg)
    lat_arr = np.asarray(lat_deg, dtype=float)
    lon_arr = np.asarray(lon_deg, dtype=float)
    valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)

    frozen = np.zeros(n, dtype=bool)
    frozen[1:] = valid[1:] & valid[:-1] & (lat_arr[1:] == lat_arr[:-1]) & (lon_arr[1:] == lon_arr[:-1])
    valid = valid & ~frozen

    if not valid.all():
        if valid.sum() < 5:
            return None
        idx = np.arange(n)
        lat_arr = np.interp(idx, idx[valid], lat_arr[valid])
        lon_arr = np.interp(idx, idx[valid], lon_arr[valid])
    return lat_arr, lon_arr


def _interpolate_dropped_altitude_fixes(altitude_m: np.ndarray) -> np.ndarray | None:
    """
    Clean a raw altitude_m track of dropped barometer fixes (NaN), the
    same way _interpolate_dropped_and_frozen_gps_fixes cleans dropped GPS
    fixes: linear interpolation from the nearest valid neighbours (edge
    NaNs fall back to the nearest valid value via np.interp's default),
    index-aligned 1:1 with the input, no resampling. No frozen-fix analog
    here -- an altitude sensor legitimately reports the same value for
    many consecutive samples on flat ground, unlike a GPS fix repeating
    bit-exactly.

    Returns None if fewer than 5 samples are valid.
    """
    n = len(altitude_m)
    alt = np.asarray(altitude_m, dtype=float)
    valid = np.isfinite(alt)
    if not valid.all():
        if valid.sum() < 5:
            return None
        idx = np.arange(n)
        alt = np.interp(idx, idx[valid], alt[valid])
    return alt


def _fit_time_b_spline(
    t: np.ndarray, values: np.ndarray, knot_interval_s: float,
    smoothing_lambda: float, smoothing_diff_order: int,
) -> BSpline:
    """
    Least-squares fit of `values` against elapsed time `t` using a
    uniform, always-degree=5/quintic B-spline with knots spaced at
    knot_interval_s [s]. Unconstrained -- no boundary condition at either
    endpoint: an endpoint constraint has no physical meaning for a
    rider's speed curve (nothing requires acceleration to be flat at the
    edges of an arbitrarily-cut time window), and forcing one measurably
    hurts the fit right where it matters most, at a standing-start
    launch. The GPS_TIME_PARAM_EDGE_MARGIN_S margin every caller carries
    past whatever point its output must actually be trusted at (see
    LEAD_IN_TIME_S/TRAIL_OUT_TIME_S and _make_activity_record's own
    padded fit window) absorbs whatever edge disagreement this produces.

    Textually independent from core.course_geometry.fit_uniform_b_spline
    (a structurally similar fit) rather than sharing code with it: that
    function fits course geometry (position against arc length) for the
    physics simulator and may evolve on physics-simulation-specific
    grounds with no bearing on this module's time-axis speed derivative,
    and vice versa.

    No ordering requirement on t: this is a least-squares fit, not an
    interpolation walk, so it's insensitive to sample order. t[0]/t[-1]
    are still taken as the domain bounds -- t is expected to be overall
    chronological.

    Args:
        t:                Independent variable (e.g. elapsed time [s]).
        values:           Dependent variable, index-aligned with t.
        knot_interval_s:  B-spline knot spacing in t's own units.
        smoothing_lambda: P-spline/Eilers-Marx roughness penalty (ported
                          from
                          core.course_geometry.fit_uniform_b_spline):
                          knot_interval alone sets a fitted curve's ripple
                          WAVELENGTH, not just its amplitude, a fitting
                          artifact rather than real signal. When > 0,
                          adds a roughness penalty on the control points:
                              solve (A^T A + smoothing_lambda * D^T D) c
                                    = A^T y
                          where D is smoothing_diff_order's finite-
                          difference operator; the fit stays unconstrained
                          at both endpoints regardless.
                          core.course_geometry.COURSE_SMOOTHING_LAMBDA_XY/
                          Z were tuned against GPX course data at a
                          different point density than a FIT activity's
                          ~1Hz sampling, so not assumed to transfer.
        smoothing_diff_order: Which finite-difference order
                          smoothing_lambda's penalty D matrix uses:
                          2 penalizes CURVATURE (2nd derivative) of the
                          control-point sequence (matches
                          core.course_geometry.fit_uniform_b_spline
                          exactly); 3 penalizes JERK (3rd derivative); 4
                          penalizes SNAP (4th derivative) -- the highest
                          order a degree=5 (quintic) basis can represent
                          without a knot discontinuity in the penalized
                          derivative (a quintic spline is C^4 across
                          interior knots). 3/4 exist only here, not in
                          core.course_geometry (which fits static shape,
                          not motion): real acceleration changes are
                          large but physically bounded in JERK (a
                          human/bike can't change acceleration
                          instantaneously), whereas GPS noise shows up as
                          erratic, effectively unbounded jerk. A curvature
                          penalty can't tell a smooth deceleration from a
                          noise spike; jerk targets the erratic component
                          but its null space is quadratic (constant
                          acceleration), so a large lambda fights real
                          rapid acceleration changes (e.g. a hard
                          braking-then-reacceleration corner) with global
                          ringing; snap's null space is cubic (jerk can
                          vary freely), tolerating a sudden braking
                          onset/release without that fight. Raises
                          ValueError if smoothing_lambda is set and this
                          isn't 2, 3, or 4.
    The fitting sample set is always `values` linearly interpolated onto
    a synthetic grid of 2*num_control_points+1 uniformly-spaced points,
    not the raw (t, values) samples -- guards against under-sampling
    relative to the number of unknowns at small n_intervals, at the cost
    of diluting the noise-to-signal ratio the fit sees (every synthetic
    point strictly between two real samples is a noiseless linear
    interpolation, while every real sample carries real jitter). No
    caller has ever needed to fit directly against the raw samples
    instead.

    Returns:
        Fitted BSpline over t's domain.
    """
    degree = 5  # quintic -- fixed, never varied by this function's 3 real call sites
    x_start, x_end = t[0], t[-1]
    domain = x_end - x_start

    # Knot count rounded UP so the actual spacing is <= knot_interval_s and
    # the grid spans exactly [x_start, x_end] -- no gap past the true
    # endpoint. knot_interval_s is therefore a target, not exact, spacing.
    n_intervals = max(1, int(np.ceil(domain / knot_interval_s)))
    eff_interval = domain / n_intervals

    # Build uniform knot vector spanning exactly [x_start, x_end]
    # (extended by degree on each side)
    inner_knots = np.concatenate(
        [x_start + np.arange(n_intervals) * eff_interval, [x_end]]
    )
    prefix_knots = x_start - np.arange(degree, 0, -1) * eff_interval
    suffix_knots = inner_knots[-1] + np.arange(1, degree + 1) * eff_interval
    knots = np.concatenate([prefix_knots, inner_knots, suffix_knots])

    num_control_points = len(knots) - degree - 1

    f_linear = interp1d(t, values, kind='linear')
    fit_t = np.linspace(x_start, x_end, 2 * num_control_points + 1)
    fit_v = f_linear(fit_t)

    # Build design matrix A — no boundary constraints (unconstrained least squares)
    A = BSpline.design_matrix(fit_t, knots, degree).toarray()

    if smoothing_lambda:
        # P-spline roughness penalty -- see smoothing_lambda/
        # smoothing_diff_order's own docstrings above.
        if smoothing_diff_order not in (2, 3, 4):
            raise ValueError(
                f"smoothing_diff_order must be 2, 3, or 4, got {smoothing_diff_order}"
            )
        if num_control_points < smoothing_diff_order + 1:
            raise ValueError(
                f"smoothing_lambda requires at least "
                f"{smoothing_diff_order + 1} control points to form an "
                f"order-{smoothing_diff_order} difference penalty, got "
                f"{num_control_points} (domain={domain:.3f}s, "
                f"knot_interval_s={knot_interval_s}s, "
                f"n_intervals={n_intervals})"
            )
        n_rows = num_control_points - smoothing_diff_order
        D = np.zeros((n_rows, num_control_points))
        if smoothing_diff_order == 2:
            # Curvature (2nd derivative) of the control-point sequence.
            for i in range(n_rows):
                D[i, i] = 1.0
                D[i, i + 1] = -2.0
                D[i, i + 2] = 1.0
        elif smoothing_diff_order == 3:
            # Jerk (3rd derivative) of the control-point sequence.
            for i in range(n_rows):
                D[i, i] = 1.0
                D[i, i + 1] = -3.0
                D[i, i + 2] = 3.0
                D[i, i + 3] = -1.0
        else:
            # Snap (4th derivative) of the control-point sequence.
            for i in range(n_rows):
                D[i, i] = 1.0
                D[i, i + 1] = -4.0
                D[i, i + 2] = 6.0
                D[i, i + 3] = -4.0
                D[i, i + 4] = 1.0
        P = D.T @ D
        ata = A.T @ A + smoothing_lambda * P
        c = np.linalg.solve(ata, A.T @ fit_v)
    else:
        # Unconstrained least squares: (A^T A) c = A^T y
        c, _, _, _ = np.linalg.lstsq(A, fit_v, rcond=None)
    return BSpline(knots, c[:num_control_points], degree)


# Fits position directly against elapsed TIME rather than arc length: a
# FIT recording's natural independent variable is time, and each
# sample's timestamp is essentially exact, whereas an arc-length
# parametrization would itself be built from the same noisy raw GPS
# positions the fit exists to smooth. Fitting x(t)/y(t)/z(t) directly
# against time also removes any "which arc-length does this sample
# correspond to" nearest-point-search ambiguity, and yields distance and
# speed from the same three fits: distance by evaluating position and
# chord-summing, speed by their analytic time-derivative (no second
# fit). See _fit_time_b_spline's own docstring for why this module keeps
# that function textually independent from core.course_geometry's own
# arc-length fit (a GPX course file has no meaningful time axis).
@dataclass
class _GpsPositionCurve:
    """
    One fitted x(t)/y(t)/z(t) GPS-position curve (see
    _fit_gps_position_curve), evaluable at any number of different time
    grids -- distance, speed, and position at an arbitrary eval_t -- all
    read from the SAME fit. Exists so a caller that needs several
    different views of one candidate's GPS track (e.g. _make_activity_record's
    lap-trimmed/padded/dense arrays and its precise start/goal-instant
    lookups) fits once and evaluates repeatedly, rather than re-fitting
    per view.
    """
    bs_x: BSpline
    bs_y: BSpline
    bs_z: BSpline
    t_dense: np.ndarray
    cumulative_dense: np.ndarray
    lon0: float
    lat0: float
    m_per_lon: float
    m_per_lat: float

    @overload
    def distance_m(self, eval_t: float) -> float: ...
    @overload
    def distance_m(self, eval_t: np.ndarray) -> np.ndarray: ...
    def distance_m(self, eval_t):
        """Chord-summed distance [m] along the fitted curve at eval_t."""
        return np.interp(eval_t, self.t_dense, self.cumulative_dense)

    @overload
    def speed_ms(self, eval_t: float) -> float: ...
    @overload
    def speed_ms(self, eval_t: np.ndarray) -> np.ndarray: ...
    def speed_ms(self, eval_t):
        """Analytic speed [m/s] (clipped at 0) at eval_t -- no second fit."""
        vx = self.bs_x.derivative(nu=1)(eval_t)
        vy = self.bs_y.derivative(nu=1)(eval_t)
        vz = self.bs_z.derivative(nu=1)(eval_t)
        return np.clip(np.sqrt(vx**2 + vy**2 + vz**2), 0.0, None)

    @overload
    def position_deg(self, eval_t: float) -> tuple[float, float]: ...
    @overload
    def position_deg(self, eval_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def position_deg(self, eval_t):
        """(lat_deg, lon_deg) at eval_t, projected back from the fitted x(t)/y(t)."""
        x_eval, y_eval = self.bs_x(eval_t), self.bs_y(eval_t)
        lon_eval = self.lon0 + x_eval / self.m_per_lon
        lat_eval = self.lat0 + y_eval / self.m_per_lat
        return lat_eval, lon_eval

    def scaled_to_length(self, target_length_m: float, t_a: float, t_b: float) -> "_GpsPositionCurve":
        """
        Return a new curve whose [t_a, t_b] segment has EXACTLY
        target_length_m of arc length, by uniformly scaling this curve's
        x/y/z control points about their own centroid -- not by
        re-optimizing anything (see the SLSQP joint-fit attempt this
        replaced: ~1300x slower and didn't even converge at this curve's
        real control-point count).

        Exact because a B-spline's basis functions sum to 1 everywhere
        (partition of unity), so replacing every control point c_i with
        centroid + s*(c_i - centroid) replaces the curve B(t) with
        s*B(t) + (1-s)*centroid EVERYWHERE, for any centroid and any
        single shared scale factor s across x, y, and z -- not just at
        the control points. Differencing two points on the curve cancels
        the (1-s)*centroid term, so every chord (and every sub-interval's
        arc length, in the limit) scales by exactly s regardless of
        centroid -- this curve's own dense chord-sum (cumulative_dense)
        can therefore just be scaled by s too, rather than
        re-accumulated from a fresh dense evaluation. The same
        cancellation means the new curve's speed_ms comes out correctly
        scaled by s automatically too (derivative kills the constant
        centroid term), with no separate handling needed.

        centroid is the mean of this curve's own control points -- any
        point works equally well for the length-scaling property above;
        the centroid is just a reasonable, shape-neutral choice of where
        to hold the curve roughly in place while its length changes,
        rather than letting it drift from the domain's own origin.

        Args:
            target_length_m: Desired arc length [m] of the [t_a, t_b]
                segment (e.g. the course's own known distance).
            t_a, t_b: The segment (e.g. t_start_point_s/t_goal_point_s)
                whose length is being matched -- NOT necessarily this
                curve's own fitting domain; every other point on the
                curve (including outside [t_a, t_b], e.g. the padded
                margin) scales by the same factor regardless.
        """
        current_length_m = self.distance_m(t_b) - self.distance_m(t_a)
        s = target_length_m / current_length_m
        centroid = np.array([self.bs_x.c.mean(), self.bs_y.c.mean(), self.bs_z.c.mean()])
        new_bs_x = BSpline(self.bs_x.t, centroid[0] + s * (self.bs_x.c - centroid[0]), self.bs_x.k)
        new_bs_y = BSpline(self.bs_y.t, centroid[1] + s * (self.bs_y.c - centroid[1]), self.bs_y.k)
        new_bs_z = BSpline(self.bs_z.t, centroid[2] + s * (self.bs_z.c - centroid[2]), self.bs_z.k)
        return _GpsPositionCurve(
            bs_x=new_bs_x, bs_y=new_bs_y, bs_z=new_bs_z,
            t_dense=self.t_dense, cumulative_dense=s * self.cumulative_dense,
            lon0=self.lon0, lat0=self.lat0, m_per_lon=self.m_per_lon, m_per_lat=self.m_per_lat,
        )


def _fit_gps_position_curve(
    time_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
    knot_interval_s: float,
    smoothing_lambda_xy: float,
    smoothing_lambda_z: float,
    smoothing_diff_order: int,
) -> _GpsPositionCurve | None:
    """
    Fit x(t)/y(t)/z(t) once against elapsed time (see this function's own
    module-level comment for the time-vs-arc-length rationale) and return
    a _GpsPositionCurve reusable for distance/speed/position at any
    eval_t, without refitting.

    Does not use core.data_manager._clean_and_project -- that function's
    duplicate-point merging (by distance, d_epsilon) would break the 1:1
    index alignment with time_s this function's x(t)/y(t)/z(t) fit needs
    (every fitting sample needs its own (t, x, y, z), not a merged/
    dropped one). Only _interpolate_dropped_and_frozen_gps_fixes's NaN/
    frozen-fix cleaning (lat/lon) and _interpolate_dropped_altitude_fixes's
    NaN cleaning (altitude) are applied, both preserving length and order.

    Args:
        time_s:           Elapsed time [s] for each FITTING sample.
        lat_deg/lon_deg/altitude_m: Raw GPS track, index-aligned 1:1 with
                           time_s.
        knot_interval_s:  B-spline knot spacing [s] on the time axis,
                           shared by x(t)/y(t)/z(t).
        smoothing_lambda_xy/smoothing_lambda_z: forwarded to
                           _fit_time_b_spline's own smoothing_lambda --
                           lambda_xy for the x(t)/y(t) (path-shape)
                           channels, lambda_z for z(t) (altitude), kept
                           separate by the same precedent as
                           core.course_geometry.COURSE_SMOOTHING_LAMBDA_XY/Z
                           (penalizing x/y curvature cuts real hairpin
                           corners, shrinking distance and inflating
                           grade).
        smoothing_diff_order: forwarded to _fit_time_b_spline's own
                           smoothing_diff_order for all three of
                           x(t)/y(t)/z(t) -- see that parameter's own
                           docstring for the curvature-vs-jerk-vs-snap
                           rationale.

    Returns:
        A _GpsPositionCurve, or None if there's too little data to fit
        (fewer than 5 samples, or too short a time span for even one
        interior knot, or a mismatched-length input).
    """
    n = len(time_s)
    if n < 5 or len(lat_deg) != n or len(lon_deg) != n or len(altitude_m) != n:
        return None
    t = np.asarray(time_s, dtype=float)
    if t[-1] - t[0] <= knot_interval_s:
        return None

    cleaned = _interpolate_dropped_and_frozen_gps_fixes(lat_deg, lon_deg)
    if cleaned is None:
        return None
    lat, lon = cleaned

    alt = _interpolate_dropped_altitude_fixes(altitude_m)
    if alt is None:
        return None

    lat_mid = np.radians(np.mean(lat))
    m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
    m_per_lon = 111412.84 * np.cos(lat_mid)
    x = (lon - lon[0]) * m_per_lon
    y = (lat - lat[0]) * m_per_lat
    z = alt

    try:
        bs_x = _fit_time_b_spline(t, x, knot_interval_s, smoothing_lambda=smoothing_lambda_xy, smoothing_diff_order=smoothing_diff_order)
        bs_y = _fit_time_b_spline(t, y, knot_interval_s, smoothing_lambda=smoothing_lambda_xy, smoothing_diff_order=smoothing_diff_order)
        bs_z = _fit_time_b_spline(t, z, knot_interval_s, smoothing_lambda=smoothing_lambda_z, smoothing_diff_order=smoothing_diff_order)
    except np.linalg.LinAlgError:
        return None

    # Dense chord-sum over the fitting domain, computed once here rather
    # than per eval_t call (mirrors core.course_geometry's own
    # x_fine/y_fine/z_fine/s_p_fine dense-then-interpolate pattern, on the
    # time axis instead of arc length).
    n_dense = max(1, int(np.ceil((t[-1] - t[0]) / 0.1)))  # dense_step_s: 0.1s is far finer than any distance change this pipeline needs to resolve
    t_dense = np.linspace(t[0], t[-1], n_dense + 1)
    x_dense, y_dense, z_dense = bs_x(t_dense), bs_y(t_dense), bs_z(t_dense)
    step = np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2 + np.diff(z_dense) ** 2)
    cumulative_dense = np.insert(np.cumsum(step), 0, 0.0)

    return _GpsPositionCurve(
        bs_x=bs_x, bs_y=bs_y, bs_z=bs_z,
        t_dense=t_dense, cumulative_dense=cumulative_dense,
        lon0=float(lon[0]), lat0=float(lat[0]), m_per_lon=m_per_lon, m_per_lat=m_per_lat,
    )


def _compute_gps_distance_speed_time_param(
    time_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
    knot_interval_s: float,
    smoothing_lambda_xy: float,
    smoothing_lambda_z: float,
    smoothing_diff_order: int,
    eval_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    GPS-position-derived distance [m], speed [m/s], AND position (lat/lon
    [deg], the same fitted curve distance/speed came from), at every
    eval_t sample (defaults to time_s itself) -- a single-eval convenience
    wrapper over _fit_gps_position_curve for a caller that only needs one
    evaluation grid (see that function's own docstring for the fitting
    contract/Args). A caller needing several different grids from the
    same fit (e.g. _make_activity_record) should call
    _fit_gps_position_curve directly instead, and reuse its
    _GpsPositionCurve, rather than calling this repeatedly (each call
    refits from scratch).

    Returns:
        (distance_m, speed_ms, lat_deg, lon_deg) at every eval_t sample,
        all four arrays -- speed_ms clipped at 0, lat_deg/lon_deg the
        fitted x(t)/y(t) curve's own position (the same curve distance_m
        was chord-summed from), not the raw recorded position. None if
        _fit_gps_position_curve couldn't fit (see its own docstring).
    """
    curve = _fit_gps_position_curve(
        time_s, lat_deg, lon_deg, altitude_m,
        knot_interval_s, smoothing_lambda_xy, smoothing_lambda_z, smoothing_diff_order,
    )
    if curve is None:
        return None
    eval_pts = np.asarray(time_s, dtype=float) if eval_t is None else np.asarray(eval_t, dtype=float)
    distance_m = curve.distance_m(eval_pts)
    speed_ms = curve.speed_ms(eval_pts)
    lat_eval, lon_eval = curve.position_deg(eval_pts)
    return distance_m, speed_ms, lat_eval, lon_eval


# Production defaults for _compute_gps_distance_speed_time_param.
# knot_interval_s=1.0 (a tight knot spacing) preserves real post-braking
# speed recovery/reacceleration shape without cutting corners.
# smoothing_diff_order=4 (a snap penalty) tolerates a sudden brake
# onset/release that lower-order (curvature/jerk) penalties fight, since
# its null space (cubic control-point sequences) lets jerk vary freely.
# lambda_xy/lambda_z share one value here, unlike core.course_geometry's
# own COURSE_SMOOTHING_LAMBDA_XY/Z split.
GPS_TIME_PARAM_KNOT_INTERVAL_S = 1.0
GPS_TIME_PARAM_SMOOTHING_LAMBDA_XY = 5.0
GPS_TIME_PARAM_SMOOTHING_LAMBDA_Z = 5.0
GPS_TIME_PARAM_SMOOTHING_DIFF_ORDER = 4

# Time step [s] used to draw an ActivityRecord's dense track (see
# ActivityRecord.dense_lat_deg) -- far finer than the FIT's own ~1Hz
# sample spacing, so a polyline through this array follows the fitted
# curve's actual shape instead of hiding it behind straight chords
# between sparse real samples.
GPS_TRACK_DENSE_STEP_S = 0.1


def _compute_gps_distance_speed_m(
    time_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_m: np.ndarray,
    eval_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Production entry point for the time-parametrized GPS distance/speed
    (and matching position) pipeline -- thin wrapper over
    _compute_gps_distance_speed_time_param with this module's
    GPS_TIME_PARAM_* defaults baked in (same precedent as
    core.course_geometry.fit_course_geometry_profile wrapping
    fit_uniform_b_spline with COURSE_SMOOTHING_LAMBDA_XY/Z).

    Args/Returns: identical contract to
    _compute_gps_distance_speed_time_param (see that function's own
    docstring), with knot_interval_s/smoothing_lambda_xy/
    smoothing_lambda_z/smoothing_diff_order fixed at this module's
    GPS_TIME_PARAM_* constants (degree and dense_step_s are hardcoded
    even deeper in the chain, inside _fit_time_b_spline/
    _compute_gps_distance_speed_time_param themselves -- not exposed as
    parameters anywhere in this call chain at all) -- only
    time_s/lat_deg/lon_deg/altitude_m/eval_t are exposed here.
    """
    return _compute_gps_distance_speed_time_param(
        time_s, lat_deg, lon_deg, altitude_m,
        knot_interval_s=GPS_TIME_PARAM_KNOT_INTERVAL_S,
        smoothing_lambda_xy=GPS_TIME_PARAM_SMOOTHING_LAMBDA_XY,
        smoothing_lambda_z=GPS_TIME_PARAM_SMOOTHING_LAMBDA_Z,
        smoothing_diff_order=GPS_TIME_PARAM_SMOOTHING_DIFF_ORDER,
        eval_t=eval_t,
    )


def _generate_candidates(
    fit: FlatFITData,
    goal_lat: float,
    goal_lon: float,
    course_distance_m: float,
) -> list[tuple[int, int]]:
    """
    Stage 1: scan for goal passages, then scan backward for the true stop.

    A standing-start TT race never stops mid-race by definition, so instead
    of forward-scanning from a start-line GPS match (which cannot
    distinguish the real standing-start departure from pre-race jostling
    while queued near the start line), this anchors on the goal passage,
    which is unambiguous (one crossing per lap, at speed), and scans
    backward from there for the last sample at or below
    stop_speed_threshold_ms. A non-standing-start passage (warm-up lap,
    rolling start) simply yields no stop within max_backscan_factor and is
    skipped: correctly, there is no race segment to analyze there.

    Args:
        fit:                 FlatFITData from parse_fit_file().
        goal_lat/lon:        Course goal coordinates [deg].
        course_distance_m:   Expected lap distance [m].

    Goal-passage detection uses fixed constants, never varied by any
    caller: a 30m proximity threshold (wider than a start-line threshold
    because goal passages happen at race speed, ~50 km/h+, where a tight
    radius risks skipping over the only nearby sample entirely), a 60s
    cluster gap (maximum time between consecutive near-goal samples to
    treat them as the same passage/lap -- must be well under the
    shortest realistic lap time, or two laps get merged into one
    cluster), a backscan distance of 1.2x course_distance_m (how far
    back to search for a stop before giving up on a goal passage), and a
    0.3 m/s stop-speed threshold (a sample at or below this speed counts
    as the stop, scanning backward from the goal passage).

    Returns:
        List of (i_stop, j) raw index pairs into fit arrays, where i_stop
        is a provisional start point (speed_ms-based; refined later by
        find_stationary_cluster/estimate_start_time -- see those
        functions' own docstrings for why speed_ms alone isn't trusted
        as the true departure) and j is the closest-approach sample
        within the goal passage (subsample-refined later by
        estimate_end_offset).
    """
    n = fit.n
    lat  = fit.lat_deg
    lon  = fit.lon_deg
    dist = fit.distance_m

    # 1) all samples within 30m (goal_proximity_m) of the goal coordinate
    near_goal = [
        i for i in range(n)
        if not (np.isnan(lat[i]) or np.isnan(lon[i]))
        and _geo_dist_m(lat[i], lon[i], goal_lat, goal_lon) <= 30.0
    ]
    if not near_goal:
        logger.debug("Stage1: no samples near goal coordinate")
        return []

    # 2) group into passages: one contiguous-in-time cluster == one lap
    clusters: list[list[int]] = [[near_goal[0]]]
    for idx in near_goal[1:]:
        if fit.time_s[idx] - fit.time_s[clusters[-1][-1]] > 60.0:  # goal_cluster_gap_s
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    candidates: list[tuple[int, int]] = []
    max_backscan_m = course_distance_m * 1.2  # max_backscan_factor

    for cluster in clusters:
        # closest approach to the goal coordinate within this passage
        j = min(cluster, key=lambda k: _geo_dist_m(float(lat[k]), float(lon[k]), goal_lat, goal_lon))

        # 3) scan backward for the true stop (last sample at or below
        #    stop_speed_threshold_ms)
        i_stop = None
        for k in range(j, -1, -1):
            if dist[j] - dist[k] > max_backscan_m:
                break
            if fit.speed_ms[k] <= 0.3:  # stop_speed_threshold_ms
                i_stop = k
                break

        if i_stop is None:
            logger.debug(
                "  Stage1: goal passage j=%d — no stop found within %.0fm back, skipping",
                j, max_backscan_m,
            )
            continue

        candidates.append((i_stop, j))
        logger.debug(
            "  Stage1: candidate start_idx=%d end_idx=%d dist=%.1fm",
            i_stop, j, dist[j] - dist[i_stop],
        )

    logger.debug("Stage1: %d candidate(s) generated", len(candidates))
    return candidates


def _fix_leading_zero_power(power_seg: np.ndarray) -> np.ndarray:
    """
    Replace a segment's leading run of NaN/0W power with the first
    genuinely positive sample that follows.

    A standing start is, by design, close to a maximal effort from the
    first pedal stroke, so power should be high — not 0 W — from sample
    0 onward. In practice the power channel reports NaN or a premature
    "0.0 W" for the first stroke or two before settling, which is
    physically implausible for a standing-start departure and can be
    corrected directly from that premise.

    Anchored specifically on "this is the leading edge of a
    standing-start segment" rather than any mid-ride signal like cadence:
    speed/cadence cannot distinguish a premature 0 W (crank not yet
    resolved) from a genuine 0 W (e.g. coasting, where cadence
    legitimately reaches 0 too), so a signal-based mask would let
    premature 0 W readings through uncaught elsewhere in the segment.
    Anchoring on position (index 0 only) sidesteps that ambiguity
    entirely: coasting elsewhere in the segment is untouched by
    construction.

    Args:
        power_seg: Power [W] for a single race segment (si..ei slice),
                   possibly containing NaN and/or 0.0 at the leading edge.

    Returns:
        power_seg with the leading NaN/0 W run replaced by the first
        positive sample found. If no positive sample exists anywhere in
        the segment (pathological/corrupt file), NaN is filled with 0.0
        and the segment is otherwise returned unchanged. Never raises.
    """
    fixed = power_seg.copy()
    n = len(fixed)

    m = 0
    while m < n and (np.isnan(fixed[m]) or fixed[m] == 0.0):
        m += 1

    if m == 0:
        return fixed  # first sample is already positive — nothing to do

    if m == n:
        # No positive sample anywhere in the segment: nothing to replace
        # the leading run with. Don't raise — just clear NaN to 0.0.
        fixed[np.isnan(fixed)] = 0.0
        return fixed

    fixed[:m] = fixed[m]
    return fixed


def _fill_nan_next_valid(power_seg: np.ndarray) -> np.ndarray:
    """
    Fill NaN power samples with the next valid (non-NaN) sample in the
    segment. Trailing NaN with no valid sample remaining are filled with
    0 W.

    Scoped to a single race segment (si..ei slice) — file-wide gaps
    (e.g. minutes of stationary pre-race waiting) never enter this
    function, since power_w is never interpolated across the whole file
    (see parse_fit_file), so "next valid sample" is always a real
    reading, never a fabricated one.

    Args:
        power_seg: Power [W] for a single race segment (si..ei slice),
                   possibly containing NaN.

    Returns:
        power_seg with NaN filled per the rule above. Untouched elsewhere.
    """
    filled = power_seg.copy()
    nan_mask = np.isnan(filled)
    if not nan_mask.any():
        return filled

    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        filled[:] = 0.0
        return filled

    pos = np.searchsorted(valid_idx, np.arange(len(filled)))
    has_next = pos < len(valid_idx)
    next_idx = valid_idx[np.clip(pos, 0, len(valid_idx) - 1)]
    fill_vals = np.where(has_next, filled[next_idx], 0.0)

    filled[nan_mask] = fill_vals[nan_mask]
    return filled


def _project_local_m(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a small window of lat/lon onto a local planar frame [m], referenced to its own first point."""
    lat_mid = np.radians(np.mean(lats))
    m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
    m_per_lon = 111412.84 * np.cos(lat_mid)
    x = (lons - lons[0]) * m_per_lon
    y = (lats - lats[0]) * m_per_lat
    return x, y


def _pc1_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Project a window's points onto their own first principal component,
    returning one scalar per point.

    If every point in the window is bit-identical (a real, common case --
    see find_stationary_cluster's own docstring), the covariance matrix
    is exactly zero and its eigenvectors are mathematically indeterminate
    -- np.linalg.eigh still returns some valid orthonormal basis rather
    than erroring, but which one is LAPACK-implementation-defined. The
    indeterminacy is harmless here (every point's deviation from the mean
    is exactly 0 regardless of direction), but relying on
    implementation-defined behaviour even when currently harmless is a
    code smell, so (following this codebase's own pattern for this class
    of problem: core.FLOAT_TIE_BREAKER_EPS, core.calibrator.
    _tie_breaker) a tiny (1e-9 m) deterministic, monotonic-in-sample-order
    ramp is added to x before computing the covariance matrix. This keeps
    it never exactly singular, so PC1 is always well-posed; the ramp's
    amplitude is far below any real GPS signal, so it never affects the
    result in a non-degenerate window.
    """
    n = len(x)
    theta = np.arange(n) / max(n - 1, 1)
    x = x + FLOAT_TIE_BREAKER_EPS * (2.0 * theta - 1.0)

    xc, yc = x - np.mean(x), y - np.mean(y)
    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, -1]  # largest-eigenvalue eigenvector = PC1
    return xc * u[0] + yc * u[1]


def find_stationary_cluster(
    fit: FlatFITData,
    provisional_start_idx: int,
) -> int | None:
    """
    Find the stationary cluster preceding a Stage-1 provisional start
    point, and return its boundary (the end nearest the provisional
    start point) as a fit index.

    Works on raw lat_deg/lon_deg only, never a GPS-spline coordinate: a
    spline can drift smoothly during a genuinely stationary period (a
    repeated/near-repeated raw GPS fix still produces a slowly, smoothly
    increasing spline-distance reconstruction, a curve-fitting artifact
    of smoothing through noise, not real motion), which would corrupt a
    threshold- or outlier-based stillness test applied to it.

    The outlier test itself (see _pc1_score) compares each point in a
    candidate window against the window's own robust median (~50%
    breakdown point), which correctly isolates a minority of moving
    samples from a majority-still window even when the moving tail's own
    growth is smooth and monotonic (so it wouldn't stand out as a single
    outlier point under a naive threshold test).

    Method: starting from a window whose right edge is
    provisional_start_idx, slide the window one sample at a time toward
    the past, testing at each position whether the window's own points --
    projected onto their own first principal component -- contain an
    outlier relative to the window's own median (threshold: median ±
    mad_k * 1.4826 * MAD, the standard scale factor converting a normal
    distribution's MAD to an equivalent sigma). The first (nearest to
    provisional_start_idx) window with no such outlier is accepted; its
    right edge is the returned boundary, never grown further once
    accepted.

    No minimum-sample-count guard on a candidate window: FlatFITData's
    lat_deg/lon_deg are always uniform-1Hz and gap-free by construction
    (see _normalize_lat_lon_to_uniform_1hz), so the only way a window can
    run short is running past the FILE's own recorded start -- which
    callers must reject upstream by requiring LEAD_IN_TIME_S (not just
    search_back_s alone -- see that constant's own docstring) of real
    history before provisional_start_idx.

    Args:
        fit:                    FlatFITData (lat_deg/lon_deg uniform-1Hz,
                                 gap-free by construction).
        provisional_start_idx:  Stage-1 backward-scan provisional start
                                 point (speed_ms-based; used only as the
                                 search's own starting point, not
                                 trusted directly).

    window_s/search_back_s/mad_k are fixed at this module's
    STATIONARY_WINDOW_S/STATIONARY_SEARCH_BACK_S/STATIONARY_MAD_K
    constants, never varied by any caller (window_s: minimum stillness
    duration [s] a candidate window must span -- 5-10s covers a holder
    physically steadying the bike through a starter's countdown;
    search_back_s: how far before provisional_start_idx to search
    before giving up; mad_k: outlier-threshold multiplier, see Method
    above).

    Returns:
        Fit index of the stationary cluster's boundary (the end nearest
        provisional_start_idx), or None if no qualifying window was
        found within STATIONARY_SEARCH_BACK_S.
    """
    t_psi = fit.time_s[provisional_start_idx]
    limit_t = t_psi - STATIONARY_SEARCH_BACK_S
    e = provisional_start_idx
    while e >= 0 and fit.time_s[e] >= limit_t:
        t_e = fit.time_s[e]
        lo = e
        while lo > 0 and fit.time_s[lo - 1] >= t_e - STATIONARY_WINDOW_S:
            lo -= 1
        idxs = np.arange(lo, e + 1)
        lats, lons = fit.lat_deg[idxs], fit.lon_deg[idxs]
        if not (np.any(np.isnan(lats)) or np.any(np.isnan(lons))):
            s = _pc1_score(*_project_local_m(lats, lons))
            med = float(np.median(s))
            half_width = STATIONARY_MAD_K * 1.4826 * float(np.median(np.abs(s - med)))
            if np.max(np.abs(s - med)) <= half_width:
                logger.debug(
                    "  StationaryCluster: psi=%d -> boundary=%d (t=%.1fs, window=[%d,%d])",
                    provisional_start_idx, e, fit.time_s[e], lo, e,
                )
                return int(e)
        e -= 1
    logger.debug(
        "  StationaryCluster: psi=%d no qualifying window within %.0fs back",
        provisional_start_idx, STATIONARY_SEARCH_BACK_S,
    )
    return None


def estimate_start_time(
    fit: FlatFITData,
    stationary_position_idx: int,
) -> float | None:
    """
    Fit a constant-acceleration model d = a*(t-t0)^2 to the n_fit real
    samples immediately after the stationary cluster's boundary, and
    return the fitted start time t0 (absolute, same time origin as
    fit.time_s).

    d is measured as the raw geodesic distance (_geo_dist_m) from each of
    the n_fit samples to the stationary cluster's boundary's own recorded
    position, never a spline coordinate (see find_stationary_cluster's
    own docstring for why start-detection stays spline-free throughout).
    Because d is a physical distance, it's non-negative by construction:
    for any candidate t0, the closed-form least-squares solution
    a = sum(x_i*d_i)/sum(x_i^2) (x_i=(t_i-t0)^2) is a ratio of two sums
    of non-negative terms, so a >= 0 always. a=0 is likewise structurally
    unreachable: find_stationary_cluster only ever accepts the window
    nearest provisional_start_idx that passes its own outlier test, so if
    the very next sample after the accepted boundary were also at
    distance 0 from it, the window ending one sample later would have
    passed that same test first and been accepted instead.

    Args:
        fit:                      FlatFITData.
        stationary_position_idx:  find_stationary_cluster's own return
                                   value -- the fixed origin the fitting
                                   samples' distances are measured from.

    n_fit/t0_search_back_s/t0_search_fwd_s/t0_grid_step_s are fixed at
    this module's ACCELERATION_N_FIT/_START_TIME_T0_SEARCH_BACK_S/
    _START_TIME_T0_SEARCH_FWD_S/_START_TIME_T0_GRID_STEP_S constants,
    never varied by any caller (n_fit: number of consecutive real
    samples after stationary_position_idx used to fit (t0, a) -- an
    assumed minimum standing-start sprint duration in seconds, points
    are 1Hz; t0_search_back_s/fwd_s: t0 grid-search range, centred on
    stationary_position_idx's own timestamp; t0_grid_step_s:
    grid-search resolution [s]).

    Returns:
        Fitted start time t0, or None if fewer than ACCELERATION_N_FIT
        real samples follow stationary_position_idx.
    """
    fit_lo = stationary_position_idx + 1
    fit_hi = min(fit_lo + ACCELERATION_N_FIT, fit.n)
    if fit_hi - fit_lo < ACCELERATION_N_FIT:
        logger.debug(
            "  StartTime: boundary=%d fewer than n_fit=%d samples available after it",
            stationary_position_idx, ACCELERATION_N_FIT,
        )
        return None
    fit_idxs = np.arange(fit_lo, fit_hi)
    if np.any(np.isnan(fit.lat_deg[fit_idxs])) or np.any(np.isnan(fit.lon_deg[fit_idxs])):
        return None

    pos_lat = float(fit.lat_deg[stationary_position_idx])
    pos_lon = float(fit.lon_deg[stationary_position_idx])
    tw = fit.time_s[fit_idxs]
    dw = np.array([
        _geo_dist_m(float(fit.lat_deg[k]), float(fit.lon_deg[k]), pos_lat, pos_lon)
        for k in fit_idxs
    ])

    t_center = float(fit.time_s[stationary_position_idx])
    t0_lo, t0_hi = t_center - _START_TIME_T0_SEARCH_BACK_S, t_center + _START_TIME_T0_SEARCH_FWD_S
    n_steps = round((t0_hi - t0_lo) / _START_TIME_T0_GRID_STEP_S)

    best_t0 = best_a = best_sse = None
    for t0_cand in np.linspace(t0_lo, t0_hi, n_steps + 1):
        x = (tw - t0_cand) ** 2
        a_cand = float(np.dot(x, dw) / np.dot(x, x))
        resid = dw - a_cand * x
        sse = float(np.dot(resid, resid))
        if best_sse is None or sse < best_sse:
            best_t0, best_a, best_sse = float(t0_cand), a_cand, sse

    logger.debug(
        "  StartTime: boundary=%d (t=%.1fs) fit=t[%.0f..%.0f] t0=%.3fs a=%.2fm/s^2 sse=%.4f",
        stationary_position_idx, t_center, tw[0], tw[-1], best_t0, best_a, best_sse,
    )
    return best_t0


def estimate_end_offset(
    fit: FlatFITData,
    ei: int,
    goal_lat: float,
    goal_lon: float,
) -> float:
    """
    Estimate the sub-second goal-crossing timestamp via GPS interpolation.

    ei (Stage 2's GPS-nearest-to-goal sample) only identifies WHICH second
    the crossing likely falls in, not whether the crossing itself is
    before or after ei, nor its sub-second instant. This function
    resolves both: it compares ei's two immediate neighbours' own
    distance to the goal to decide which side brackets the crossing (the
    neighbour still approaching the goal, vs. the one already past it),
    then linearly interpolates between that adjacent pair — appropriate
    here because the rider is moving at speed through the crossing, so
    motion is locally near-linear in time. find_course_matches's own
    padding-availability check guarantees ei-1/ei+1 exist as real,
    non-NaN samples before this runs (see that check's own docstring),
    so no bracket-formation failure case exists here.

    The start side needs an analogous correction too, but not this same
    method (see estimate_start_time()): near a standing start the motion
    is strongly nonlinear in time (accelerating from rest), so linear
    bracket interpolation doesn't apply there -- estimate_start_time()
    instead fits a constant-acceleration parabola anchored at the
    stationary cluster's own boundary and solves for its v=0 vertex.

    Args:
        fit:           FlatFITData.
        ei:            Stage-2 refined index (GPS-nearest sample to goal).
        goal_lat/lon:  Course goal coordinates [deg].

    Returns:
        The interpolated goal-crossing time [s], same origin as
        fit.time_s. Callers needing the corresponding real sample index
        (the last one at or before the crossing) or GPS-spline distance
        should derive them from this timestamp directly (bisect on
        fit.time_s; re-evaluate the fitted spline at this instant), not
        from ei, which may itself be on either side of the crossing.
    """
    d_hint = _geo_dist_m(fit.lat_deg[ei], fit.lon_deg[ei], goal_lat, goal_lon)
    d_before = _geo_dist_m(fit.lat_deg[ei - 1], fit.lon_deg[ei - 1], goal_lat, goal_lon)
    d_after = _geo_dist_m(fit.lat_deg[ei + 1], fit.lon_deg[ei + 1], goal_lat, goal_lon)

    # Bracket the crossing using the side with the smaller neighbouring
    # distance (closer approach to the goal point). d_a/d_b >= 0 always
    # (geographic distances), so i_a is always at/before the crossing
    # and i_b always at/after it, regardless of which branch is taken.
    if d_before <= d_after:
        i_a, i_b, d_a, d_b = ei - 1, ei, d_before, d_hint
    else:
        i_a, i_b, d_a, d_b = ei, ei + 1, d_hint, d_after

    # d_a == d_b == 0 (two distinct samples both exactly on the goal
    # point) has probability 0 for a rider actually moving through
    # continuous space, but is reachable under float64 (e.g. a frozen/
    # repeated GPS fix) -- see core.FLOAT_TIE_BREAKER_EPS's own docstring
    # for why this gets a fixed perturbation, not an `if` branch.
    frac = d_a / (d_a + d_b + FLOAT_TIE_BREAKER_EPS)
    t_goal_point_s = fit.time_s[i_a] + frac * (fit.time_s[i_b] - fit.time_s[i_a])

    logger.debug(
        "  EndOffset: ei=%d bracket=(%d,%d) d_a=%.1fm d_b=%.1fm frac=%.3f t_goal=%.3fs",
        ei, i_a, i_b, d_a, d_b, frac, t_goal_point_s,
    )
    return float(t_goal_point_s)


# ---------------------------------------------------------------------------
# IV. Stage 2 — Boundary refinement
# ---------------------------------------------------------------------------

def _refine_end(
    fit: FlatFITData,
    idx: int,
    target_lat: float,
    target_lon: float,
) -> int:
    """
    Refine end index: GPS-nearest point within a small window around idx.

    Search window is a fixed 60 samples -- this function's only caller,
    _refine_candidates, always passes that same value.
    """
    search_window = 60
    lo = max(0, idx - search_window)
    hi = min(fit.n - 1, idx + search_window)
    best_idx = idx
    best_dist = float('inf')
    for k in range(lo, hi + 1):
        if np.isnan(fit.lat_deg[k]) or np.isnan(fit.lon_deg[k]):
            continue
        d = _geo_dist_m(fit.lat_deg[k], fit.lon_deg[k], target_lat, target_lon)
        if d < best_dist:
            best_dist = d
            best_idx = k
    return best_idx


def _refine_candidates(
    fit: FlatFITData,
    raw_candidates: list[tuple[int, int]],
    goal_lat: float,
    goal_lon: float,
) -> list[tuple[int, int]]:
    """
    Stage 2: refine the end index only.

    Start: no refinement — Stage 1 already anchors on the true stop
    (speed_ms at or below stop_speed_threshold_ms), found by scanning
    backward from the goal passage (see _generate_candidates). A
    GPS-nearest search here would be ambiguous instead: queued/jostling
    samples near the start line are also GPS-close to it.
    End:   GPS-nearest point to course goal, within the goal passage
    cluster already identified by Stage 1.

    Runs 3 refinement iterations -- a fixed constant, never varied by
    any caller.
    """
    refined = []
    for s_idx, e_idx in raw_candidates:
        si, ei = s_idx, e_idx
        for _ in range(3):
            ei = _refine_end(fit, ei, goal_lat, goal_lon)
        logger.debug("  Stage2: refined (%d,%d) -> (%d,%d)", s_idx, e_idx, si, ei)
        refined.append((si, ei))
    return refined


# ---------------------------------------------------------------------------
# V. Stage 3 — Validation (trajectory similarity)
# ---------------------------------------------------------------------------

def _discrete_frechet(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete Fréchet distance between two (N,2) and (M,2) polylines."""
    n, m = len(p), len(q)
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] >= 0:
            return ca[i, j]
        d = float(np.linalg.norm(p[i] - q[j]))
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(c(0, j-1), d)
        elif j == 0:
            ca[i, j] = max(c(i-1, 0), d)
        else:
            ca[i, j] = max(min(c(i-1,j), c(i-1,j-1), c(i,j-1)), d)
        return ca[i, j]

    return c(n-1, m-1)


def _validate_candidates(
    fit: FlatFITData,
    refined: list[tuple[int, int]],
    course_latlons: list[tuple[float, float]],
) -> list[float | None]:
    """
    Stage 3: compute spatial similarity score for each refined candidate.

    Args:
        fit:            FlatFITData.
        refined:        Output of _refine_candidates().
        course_latlons: Course polyline as [(lat,lon), ...].

    Downsample target for Fréchet computation is a fixed 200 samples,
    never varied by this function's only caller.

    Returns:
        List of spatial scores [0-1] in the same order as refined, or None
        for a candidate with fewer than 2 valid GPS points to score --
        there's no real spatial evidence to compute a Frechet distance
        from, so this isn't a "middling" match, it's an unscoreable one;
        the caller rejects it like any other REJECTED candidate rather
        than let a made-up score compete against real ones for the
        top-ranked match.
    """
    course_pts = np.array(course_latlons, dtype=np.float64)
    step_c = max(1, len(course_pts) // 200)  # max_samples
    course_ds = course_pts[::step_c]

    # Bounding-box diagonal for normalisation
    lat_range = course_pts[:, 0].max() - course_pts[:, 0].min()
    lon_range = course_pts[:, 1].max() - course_pts[:, 1].min()
    bbox_diag = float(np.sqrt(lat_range**2 + lon_range**2))
    if bbox_diag == 0.0:
        # Every course point identical -- a real GPX course never has zero
        # geographic extent, so this means course_latlons itself is corrupt.
        # Fail loudly rather than silently substituting a fake normalisation
        # scale (1.0) and returning a spatial score that looks plausible but
        # is meaningless.
        raise ValueError(
            "Course polyline has zero geographic extent (all points "
            "identical) -- cannot compute a spatial similarity score "
            "against a degenerate course."
        )

    scores: list[float | None] = []
    for si, ei in refined:
        seg_lat = fit.lat_deg[si:ei+1]
        seg_lon = fit.lon_deg[si:ei+1]
        valid = ~(np.isnan(seg_lat) | np.isnan(seg_lon))
        seg_pts = np.column_stack([seg_lat[valid], seg_lon[valid]])

        if len(seg_pts) < 2:
            scores.append(None)
            continue

        step_s = max(1, len(seg_pts) // 200)  # max_samples
        seg_ds = seg_pts[::step_s]
        frechet = _discrete_frechet(seg_ds, course_ds)
        score = float(np.clip(1.0 - frechet / bbox_diag, 0.0, 1.0))
        logger.debug("  Stage3: frechet=%.6f bbox=%.6f score=%.3f", frechet, bbox_diag, score)
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# VI. Public API
# ---------------------------------------------------------------------------

def find_course_matches(
    fit: FlatFITData,
    course_distance_m: float,
    course_latlons: list[tuple[float, float]],
) -> list[ActivityCandidate]:
    """
    Run the four-stage pipeline and return validated ActivityCandidates.

    Args:
        fit:               FlatFITData from parse_fit_file().
        course_distance_m: Expected lap distance [m].
        course_latlons:    Course polyline as [(lat, lon), ...].

    Stage 2.5's relative-error tolerance for the odometer-vs-course-
    distance check is a fixed ±30%, never varied by this function's only
    caller -- a rough cutoff to catch a wrong-course/mis-detected
    candidate, not a precision check, so the odometer field's own known
    imprecision (see the Stage 2.5 block below) doesn't matter here.
    Candidates exceeding it are rejected outright.

    A fully accepted candidate's own GPS-spline distance (computed later,
    once, in _make_activity_record) is deliberately pinned to EXACTLY
    course_distance_m (see _GpsPositionCurve.scaled_to_length) rather
    than left to disagree with it by the real ~0-1% a rider's actual
    line through corners would otherwise produce (apex-cutting shortens
    it, a wide/swinging line lengthens it) -- a real difference in
    physical path length, not a measurement error, but one this module's
    consumers need expressed as a uniform rescale of the whole track
    rather than a raw distance value that disagrees with the course's
    own, the same reasoning _rescale_distance_to_target already applies
    to an Activity's recorded power.

    Returns:
        List of ActivityCandidate sorted by combined_score descending.
        Empty list if no match found (including: every candidate rejected
        at the Stage 2.5 distance check).
    """
    latlons = np.array(course_latlons)
    start_lat, start_lon = float(latlons[0, 0]),  float(latlons[0, 1])
    goal_lat,  goal_lon  = float(latlons[-1, 0]), float(latlons[-1, 1])

    logger.debug(
        "find_course_matches: course=%.1fm start=(%.5f,%.5f) goal=(%.5f,%.5f)",
        course_distance_m, start_lat, start_lon, goal_lat, goal_lon,
    )

    # Stage 1
    raw = _generate_candidates(fit, goal_lat, goal_lon, course_distance_m)
    if not raw:
        logger.debug("find_course_matches: no candidates after Stage 1")
        return []

    # Stage 2
    refined = _refine_candidates(
        fit, raw, goal_lat, goal_lon,
    )

    # Stage 2.5: reject any candidate whose distance disagrees wildly
    # with the expected course distance. A rough cutoff only (30%
    # tolerance) to catch a wrong-course/mis-detected candidate, not a
    # precision check -- gates on the FIT file's own odometer field
    # (fit.distance_m), the same field Stage 1's backscan budget already
    # uses. The odometer's known 5-15m/few-hundred-metres drift against
    # true GPS position matters for physics-replay gradients (see
    # _make_activity_record, which fits a proper GPS-spline for that),
    # not for a 30%-tolerance sanity filter.
    accepted: list[tuple[int, int]] = []
    for si, ei in refined:
        if ei <= si:
            # Degenerate candidate (e.g. a single-point Stage-1/2 match) —
            # same guard _make_activity_record applies later, just moved
            # earlier.
            logger.debug("  DistCheck: (%d,%d) degenerate (ei<=si) — REJECTED", si, ei)
            continue
        odo_delta = float(fit.distance_m[ei] - fit.distance_m[si])
        rel_err = abs(odo_delta - course_distance_m) / course_distance_m
        if rel_err > 0.30:  # distance_tolerance -- rough cutoff only
            logger.debug(
                "  DistCheck: (%d,%d) odo_delta=%.1fm course=%.1fm rel_err=%.1f%% > tolerance=%.0f%% — REJECTED",
                si, ei, odo_delta, course_distance_m, rel_err * 100, 0.30 * 100,
            )
            continue
        logger.debug(
            "  DistCheck: (%d,%d) odo_delta=%.1fm course=%.1fm rel_err=%.1f%% — OK",
            si, ei, odo_delta, course_distance_m, rel_err * 100,
        )
        accepted.append((si, ei))

    if not accepted:
        logger.debug("find_course_matches: all candidates rejected at distance check")
        return []
    refined = accepted

    # Stage 3
    spatial_scores = _validate_candidates(fit, refined, course_latlons)

    # Build ActivityCandidate list
    candidates: list[ActivityCandidate] = []
    for (si, ei), score in zip(refined, spatial_scores):
        # Stage 3 couldn't score this candidate at all (fewer than 2 valid
        # GPS points in its segment) -- see _validate_candidates' own
        # docstring for why that's rejected rather than given a made-up
        # score.
        if score is None:
            logger.debug("  Stage3: si=%d too few valid GPS points to score — REJECTED", si)
            continue

        # Padding-availability check: find_stationary_cluster's own search
        # plus estimate_start_time's own t0 fit can together place what
        # this loop needs up to LEAD_IN_TIME_S seconds before si (not just
        # STATIONARY_SEARCH_BACK_S alone -- see that constant's own
        # docstring) -- reject rather than let the search run out of data
        # near the FIT's own recorded start.
        if fit.time_s[0] > fit.time_s[si] - LEAD_IN_TIME_S:
            logger.debug(
                "  StartDetect: si=%d insufficient history before si (need %.0fs) — REJECTED",
                si, LEAD_IN_TIME_S,
            )
            continue

        # Same check, mirrored on the goal side: estimate_end_offset()
        # below needs a real sample on each side of ei to bracket the
        # crossing, and ActivityRecord.pad_time_s needs TRAIL_OUT_TIME_S
        # seconds of real history after ei. Reject rather than let either
        # run out of data near the FIT's own recorded end.
        if fit.time_s[-1] < fit.time_s[ei] + TRAIL_OUT_TIME_S:
            logger.debug(
                "  EndDetect: ei=%d insufficient history after ei (need %.0fs) — REJECTED",
                ei, TRAIL_OUT_TIME_S,
            )
            continue

        stationary_position_idx = find_stationary_cluster(fit, si)
        if stationary_position_idx is None:
            logger.debug("  StartDetect: si=%d no stationary cluster found — REJECTED", si)
            continue

        t_start_point_s = estimate_start_time(fit, stationary_position_idx)
        if t_start_point_s is None:
            logger.debug(
                "  StartDetect: si=%d boundary=%d insufficient acceleration-domain samples — REJECTED",
                si, stationary_position_idx,
            )
            continue

        t_goal_point_s = estimate_end_offset(fit, ei, goal_lat, goal_lon)

        rec = _make_activity_record(fit, si, t_start_point_s, stationary_position_idx, t_goal_point_s, course_distance_m)
        if rec is None:
            continue
        candidates.append(ActivityCandidate(
            record=rec,
            combined_score=score,
        ))

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    logger.debug("find_course_matches: %d candidate(s) returned", len(candidates))
    return candidates


def _make_activity_record(
    fit: FlatFITData,
    si: int,
    t_start_point_s: float,
    stationary_position_idx: int,
    t_goal_point_s: float,
    course_distance_m: float,
) -> ActivityRecord | None:
    """
    Build an ActivityRecord from a slice of FlatFITData.

    ei is not a caller-supplied argument: it's derived here, from
    t_goal_point_s, as the last real fit sample at or before the crossing
    (bisect on fit.time_s). Stage 2's own GPS-nearest-to-goal index (also
    called ei elsewhere in this module) is NOT used for this -- it only
    identifies which second the crossing likely falls in, not whether
    it's itself before or after the true crossing (see
    estimate_end_offset), and per build_zoh_power_blocks's ANT+/BLE
    convention power_w[i] is the mean power over the interval ENDING at
    distance_m[i] -- so every per-sample channel sliced up to ei must
    itself be a real sample at or before the goal, or a fabricated
    post-goal reading would wrongly govern the interval ending at the
    goal.

    Fits ONE GPS-position curve per candidate (_fit_gps_position_curve),
    over a window padded LEAD_IN_TIME_S before si and TRAIL_OUT_TIME_S
    after ei -- wide enough that an unconstrained quintic spline has real
    data to anchor against past both boundaries, and that
    t_start_point_s/t_goal_point_s always fall inside its domain
    (GPS_TIME_PARAM_EDGE_MARGIN_S to spare). That curve is then rescaled
    (see _GpsPositionCurve.scaled_to_length) so its own
    [t_start_point_s, t_goal_point_s] segment has EXACTLY course_distance_m
    of length -- an exact, near-free control-polygon rescale, not a
    re-fit -- before anything below reads distance/position from it.
    Every distance/position value below -- the lap-trimmed body, the
    padded arrays, the dense display track, and the precise start/goal-
    instant lookups -- is read from that ONE (now length-pinned) curve;
    nothing here refits.

    distance_m/lat_deg/lon_deg (and their pad_/dense_ counterparts) come
    from this GPS-spline curve, not from fit.distance_m/fit.lat_deg/
    fit.lon_deg (the FIT file's own odometer/raw recorded position): the
    odometer disagrees non-monotonically with the rider's true position
    by 5-15 m over a few hundred metres and can feed the wrong local
    gradient into the physics replay on a switchback course, and a
    displayed position must describe the same curve its own distance
    label was chord-summed from, or the two silently drift apart through
    the spline's own smoothing bias. gps_speed_ms is this same curve's
    own analytic derivative (see ActivityRecord.gps_speed_ms) -- exposed
    so a consumer needing GPS-derived speed reads it directly instead of
    re-fitting a second spline on top of the already-smoothed lat_deg/
    lon_deg, which would double-smooth.

    Applies start time offset and NaN power filling:
    - time_s is adjusted so t=0 corresponds to t_start_point_s (the
      fitted true departure — see estimate_start_time()).
    - power_w: the segment's leading NaN/0W run (pre-crank-resolution
      artifact at a standing start) is replaced per
      _fix_leading_zero_power(), then any remaining NaN (genuine sensor
      dropout) is filled per _fill_nan_next_valid(). File-wide gaps never
      reach this point (see parse_fit_file).
    - The final time_s AND distance_m/lat_deg/lon_deg/gps_speed_ms
      samples are all replaced by the curve evaluated exactly at the goal
      crossing (t_goal_point_s — see estimate_end_offset()), so
      elapsed_time_s/total_distance_m reflect the same sub-second-accurate
      instant rather than leaving distance at the nearest whole-second
      sample's value while time is corrected, which would put the
      channels out of sync at the last point.
    - Symmetrically, a synthetic leading sample at exactly (time_s=0,
      distance_m=0, using the curve's own position at t_start_point_s) is
      prepended to every per-sample array — a segment has to physically
      contain its own declared origin, or "distance since start" isn't
      meaningful. si itself is not reused for this, since it can sit
      several seconds/metres past the true departure (see
      find_stationary_cluster's docstring). Altitude/power/heart_rate
      have no continuous curve to query, so the synthetic sample borrows
      stationary_position_idx's own recorded values instead (a real
      sample typically within ~1s of t_start_point_s); power inherits the
      first real sample's own (already leading-zero-corrected) value
      instead, so no artificial discontinuity is introduced before it.
      gps_speed_ms's own leading sample is hardcoded 0.0 rather than the
      curve's own derivative at t_start_point_s (unlike lat_deg/lon_deg,
      which DO use the curve there): a standing start is exactly 0 m/s by
      definition at the departure instant, so whatever near-zero value
      the curve's derivative happens to give there would be fitting
      noise, not real signal.
    """
    ei = bisect.bisect_right(fit.time_s, t_goal_point_s) - 1
    if ei <= si:
        return None
    sl = slice(si, ei + 1)
    t_seg = fit.time_s[sl] - t_start_point_s
    t_seg[-1] = t_goal_point_s - t_start_point_s

    p_raw_seg = fit.power_w[sl]
    p_leading_fixed = _fix_leading_zero_power(p_raw_seg)
    p_corrected = _fill_nan_next_valid(p_leading_fixed)

    speed_seg = fit.speed_ms[sl].copy()
    altitude_seg = fit.altitude_m[sl].copy()
    hr_seg = fit.heart_rate[sl].copy() if fit.heart_rate is not None else None

    pad_lo = max(0, bisect.bisect_right(fit.time_s, fit.time_s[si] - LEAD_IN_TIME_S) - 1)
    pad_hi = bisect.bisect_right(fit.time_s, fit.time_s[ei] + TRAIL_OUT_TIME_S)
    pad_sl = slice(pad_lo, pad_hi)
    curve = _fit_gps_position_curve(
        fit.time_s[pad_sl], fit.lat_deg[pad_sl], fit.lon_deg[pad_sl], fit.altitude_m[pad_sl],
        GPS_TIME_PARAM_KNOT_INTERVAL_S,
        GPS_TIME_PARAM_SMOOTHING_LAMBDA_XY, GPS_TIME_PARAM_SMOOTHING_LAMBDA_Z,
        GPS_TIME_PARAM_SMOOTHING_DIFF_ORDER,
    )
    if curve is None:
        return None
    # Pin the lap's own [t_start_point_s, t_goal_point_s] length to the
    # course's own known distance -- see _GpsPositionCurve.scaled_to_length's
    # own docstring for why this is an exact, ~free rescale of the control
    # polygon rather than a re-fit. Corrects for the unconstrained fit's
    # own smoothing bias (corner-cutting/widening shifts distance a real
    # 0-1%, same magnitude as the rider's own real line-choice
    # difference from the GPX reference -- see find_course_matches's own
    # docstring) rather than leaving the two uncorrelated with each
    # other. Applied to the curve as a whole, so pad_*/dense_* below
    # inherit the SAME correction, not just the lap-trimmed body.
    curve = curve.scaled_to_length(course_distance_m, t_start_point_s, t_goal_point_s)

    d_start_m = float(curve.distance_m(t_start_point_s))
    d_end_goal_m = float(curve.distance_m(t_goal_point_s))
    lat_start, lon_start = (float(v) for v in curve.position_deg(t_start_point_s))
    lat_at_goal, lon_at_goal = (float(v) for v in curve.position_deg(t_goal_point_s))

    d_seg = curve.distance_m(fit.time_s[sl]) - d_start_m
    d_seg[-1] = d_end_goal_m - d_start_m
    lat_seg, lon_seg = curve.position_deg(fit.time_s[sl])
    lat_seg[-1] = lat_at_goal
    lon_seg[-1] = lon_at_goal

    # GPS-derived speed, read directly off this same curve (see
    # ActivityRecord.gps_speed_ms's own docstring) -- 0.0 at the synthetic
    # departure sample below, same standing-start convention as speed_seg,
    # rather than whatever near-zero fitting noise the curve's own
    # derivative happens to give exactly at t_start_point_s.
    gps_speed_seg = curve.speed_ms(fit.time_s[sl])
    gps_speed_seg[-1] = float(curve.speed_ms(t_goal_point_s))

    # Prepend the synthetic true-departure sample -- see docstring.
    t_seg = np.concatenate([[0.0], t_seg])
    d_seg = np.concatenate([[0.0], d_seg])
    speed_seg = np.concatenate([[0.0], speed_seg])
    gps_speed_seg = np.concatenate([[0.0], gps_speed_seg])
    altitude_seg = np.concatenate([[fit.altitude_m[stationary_position_idx]], altitude_seg])
    lat_seg = np.concatenate([[lat_start], lat_seg])
    lon_seg = np.concatenate([[lon_start], lon_seg])
    p_corrected = np.concatenate([[p_corrected[0]], p_corrected])
    if hr_seg is not None:
        assert fit.heart_rate is not None  # hr_seg was derived from it above
        hr_seg = np.concatenate([[fit.heart_rate[stationary_position_idx]], hr_seg])

    # t_start_point_s is elapsed seconds since fit.timestamp_ms[0] (same
    # origin as fit.time_s), so the record's own wall-clock start is
    # simply that origin plus t_start_point_s -- consumed as the record's
    # start elsewhere (e.g. the analyzer window's "Start: HH:MM" display).
    # Fails fast with a clear message rather than silently substituting a
    # fake epoch start_time: an out-of-range timestamp here means
    # fit.timestamp_ms[0] itself is corrupt, a real data problem that
    # should surface, not a display value worth papering over.
    try:
        start_time = datetime.utcfromtimestamp(
            fit.timestamp_ms[0] / 1000.0 + t_start_point_s
        )
    except (OSError, OverflowError, ValueError) as e:
        raise ValueError(
            f"invalid start_time timestamp (fit.timestamp_ms[0]="
            f"{fit.timestamp_ms[0]}, t_start_point_s={t_start_point_s}): {e}"
        ) from e

    # Dense track for drawing (see ActivityRecord.dense_lat_deg): the SAME
    # curve, re-evaluated at GPS_TRACK_DENSE_STEP_S spacing over
    # [0, elapsed_time_s] -- the lap-trimmed range, not the wider pad_*
    # one, since a display polyline has no need for that margin.
    dense_t_rel = np.arange(0.0, t_seg[-1], GPS_TRACK_DENSE_STEP_S)
    dense_t_rel = np.append(dense_t_rel, t_seg[-1])
    dense_distance_m = curve.distance_m(dense_t_rel + t_start_point_s) - d_start_m
    dense_lat_deg, dense_lon_deg = curve.position_deg(dense_t_rel + t_start_point_s)

    # Padded arrays (see ActivityRecord.pad_time_s's docstring): index
    # range extended LEAD_IN_TIME_S before si and TRAIL_OUT_TIME_S past
    # ei, clamped to FlatFITData's own bounds (a lap starting/ending
    # within that margin of the FIT file's own first/last sample just
    # gets whatever's actually available, no synthetic extrapolation).
    # Same t=0/d=0 origin as time_s/distance_m above, so pad_time_s/
    # pad_distance_m agree with them on what "0" means.
    pad_lat_deg, pad_lon_deg = curve.position_deg(fit.time_s[pad_sl])

    return ActivityRecord(
        source_path       = fit.source_path,
        start_time        = start_time,
        elapsed_time_s    = float(t_seg[-1]),
        total_distance_m  = float(d_seg[-1]),
        time_s            = t_seg.copy(),
        distance_m        = d_seg.copy(),
        power_w           = p_corrected,
        speed_ms          = speed_seg,
        altitude_m        = altitude_seg,
        lat_deg           = lat_seg,
        lon_deg           = lon_seg,
        heart_rate_bpm    = hr_seg,
        t_start_point_s   = t_start_point_s,
        utc_offset_s      = fit.utc_offset_s,
        pad_time_s        = fit.time_s[pad_sl] - t_start_point_s,
        pad_distance_m    = curve.distance_m(fit.time_s[pad_sl]) - d_start_m,
        pad_altitude_m    = fit.altitude_m[pad_sl].copy(),
        pad_lat_deg       = pad_lat_deg,
        pad_lon_deg       = pad_lon_deg,
        dense_distance_m  = dense_distance_m,
        dense_lat_deg     = dense_lat_deg,
        dense_lon_deg     = dense_lon_deg,
        gps_speed_ms      = gps_speed_seg,
    )


def _rescale_distance_to_target(
    d_src: np.ndarray, target_distance_m: float | None
) -> tuple[np.ndarray, float]:
    """
    Uniformly stretch/shrink d_src so its last sample lands exactly on
    target_distance_m.

    Pin d_src[-1] exactly to end_dist by stretching/shrinking the whole
    distance axis by one constant factor, rather than extrapolating only
    the last segment's local pace across whatever distance the recording
    falls short (or over) by. d_src is GPS-derived, so any small
    residual gap remaining is attributable to the rider's real
    line-choice through corners (apex-cutting vs. a wide/swinging line)
    rather than a sensor artifact — a genuine, roughly-uniformly-
    distributed difference, not a jump concentrated at one point, so a
    uniform rescale matches its actual distribution.

    find_course_matches()'s Stage 2.5 already rejects any candidate whose
    GPS-derived distance disagrees with course_distance_m by more than a
    fixed 10% before this is ever called, so the scale factor here is
    always small — this corrects line-choice residual, it does not
    paper over a fundamentally wrong match.

    Args:
        d_src: Raw cumulative distance array [m], d_src[0] == 0.
        target_distance_m: Course length to pin the last sample to. If
            None, d_src is returned unscaled (end_dist = d_src[-1]).

    Returns:
        (d_scaled, end_dist): the rescaled distance array and the pinned
        end distance actually used.
    """
    end_dist = target_distance_m if target_distance_m is not None else d_src[-1]
    if d_src[-1] > 0 and end_dist > 0:
        d_scaled = d_src * (end_dist / d_src[-1])
        d_scaled[-1] = end_dist  # eliminate float residual from the multiply
        return d_scaled, end_dist
    return d_src.copy(), end_dist


def build_zoh_power_blocks(record: ActivityRecord, target_distance_m: float | None = None):
    """
    Build a PowerBlocks instance directly from raw (un-resampled)
    ActivityRecord samples, for use as the physics engine's actual-power
    replay input.

    This intentionally does NOT resample onto a uniform distance grid.
    Any such grid's spacing (e.g. 10 m) has no physical relationship to
    the FIT file's own sampling structure. Feeding a resampled series into
    the physics engine would mean smoothing the recorded power to that
    grid and only then holding it constant per block — the system's power
    model is a zero-order-hold on the raw sample structure throughout, and
    no linear interpolation of power should occur anywhere in that path.
    Block boundaries therefore sit at the FIT file's own (variable,
    ~1s-spaced) sample distances, not on any fixed grid.

    The same odometer-drift rescale hyle.apps.fit2gpx_converter uses is
    applied here too (via _rescale_distance_to_target), so a replayed
    course reaches the same total distance as the course-physics
    reference — but only as a uniform distance-axis correction, not as a
    resample.

    Per the ANT+/BLE cycling power convention, power_w[i] is the mean
    power over the raw interval ending at distance_m[i] (i.e.
    [t_{i-1}, t_i)), not an instantaneous value AT distance_m[i]. Block i
    therefore holds power_w[i+1] (the interval starting at distance_m[i])
    over a length of distance_m[i+1] - distance_m[i]; the leading sample
    power_w[0] has no preceding interval within this segment and is
    discarded (N samples -> N-1 blocks).

    Args:
        record: Raw ActivityRecord (e.g. ActivityCandidate.record) at the
                FIT file's own (un-resampled) sample spacing.
        target_distance_m: Course length to rescale the recorded distance
            to (odometer-drift correction). If None, the recorded total
            distance is used as-is.

    Returns:
        PowerBlocks with variable-length blocks matching the FIT file's
        own sample spacing.
    """
    d_scaled, _ = _rescale_distance_to_target(record.distance_m, target_distance_m)
    return PowerBlocks(
        power=record.power_w[1:],
        length=np.diff(d_scaled),
    )
