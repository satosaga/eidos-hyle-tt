##########################
# activity_correspondence.py
##########################
"""
Percent-distance correspondence between an Activity (FIT recording) and
the Strategy/Rebuild course it's compared against.

pct := activity.distance_m / activity.total_distance_m -- an Activity's
own distance as a fraction of its own final distance, not a raw distance
value. Activity's total distance is GPS-spline-derived and Strategy/
Rebuild's is course geometry; the two rarely agree, so comparing at the
same raw distance conflates two different measurements of "how far".
Comparing at the same pct instead treats both as spanning their own
course start-to-finish (the same assumption build_zoh_power_blocks makes
when it rescales an Activity's distance onto a course length before
replaying power through the simulator).

Deliberately not part of core.activity_parser: that module has no notion
of a course or its length, only a FIT recording's own distance/time.
pct only becomes meaningful once an ActivityRecord is placed alongside a
course of known length -- eidos.apps.analyzer and core.calibrator
are the two consumers that do so.

pct is monotonic and invertible (activity.distance_m increases
monotonically by construction), but looked up in different directions
per side: Activity's own channels (power_w, speed_ms, altitude_m, ...)
are indexed by time, so pct_to_activity_time() below converts pct to
elapsed time for looking a channel up there. Strategy/Rebuild's channels
(course_s_p, x_traj, ...) are indexed by distance, so pct maps to them
by plain multiplication -- see pct_to_distance().
"""

import numpy as np

from core.activity_parser import ActivityRecord


def activity_pct(activity: ActivityRecord) -> np.ndarray:
    """The Activity's own recorded samples as pct instead of distance_m."""
    return activity.distance_m / activity.total_distance_m


def pct_to_activity_time(activity: ActivityRecord, pct: "float | np.ndarray") -> "float | np.ndarray":
    """
    The Activity's own elapsed time [s] at the given pct, via linear
    interpolation on its (monotonic) pct -> time_s mapping. Use this to
    evaluate the Activity at a pct that didn't come from its own samples
    (e.g. a shared Strategy/Rebuild pct grid), then look up whichever
    channel is needed at the returned time.
    """
    return np.interp(pct, activity_pct(activity), activity.time_s)


def pct_to_distance(pct: "float | np.ndarray", total_distance_m: float) -> "float | np.ndarray":
    """
    Strategy/Rebuild-side counterpart to pct_to_activity_time(): these
    traces are already indexed by distance, so pct maps to a distance by
    plain multiplication. total_distance_m is the COURSE length for a
    Strategy (course_distance_m) or a scenario's own finish distance for
    a Rebuild (trace.x_traj[-1]) -- never the Activity's own.
    """
    return pct * total_distance_m
