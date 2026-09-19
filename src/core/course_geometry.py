####################
# course_geometry.py
####################
import logging

import numpy as np
from scipy.interpolate import BSpline, interp1d

from core.schema import CoursePoints, RunSettings

logger = logging.getLogger(__name__)

# --------------------------------------------------
# I. Geometric and physical preprocessing
# --------------------------------------------------
# Hardcoded course-geometry B-spline fit parameters -- a numerical-
# fitting concern (how well the smoothed course geometry tracks the raw
# GPX shape vs. how much high-frequency ripple survives into
# slope/kappa), not a per-run RunSettings/config field a rider/course-
# designer should have to reason about. fit_course_geometry_profile uses
# these directly rather than reading knot_interval/degree/smoothing off
# RunSettings.
#
# A tight, fixed knot spacing (COURSE_KNOT_INTERVAL_M) alone can only
# trade a fitted-ripple artifact's amplitude against shape fidelity, not
# fix its wavelength tracking knot_interval itself -- a numerical
# B-spline-fitting signature, not real terrain (a real switchback's
# spacing wouldn't track an arbitrary fitting parameter that way). A
# generous, fixed knot spacing plus a P-spline-style roughness penalty on
# the control points (smoothing_lambda_xy/z, smoothing_diff_order) fixes
# this directly, per a smoothing spline's actual textbook design (Eilers
# & Marx P-splines: pick knots generously, let the penalty do the
# smoothing).
#
# smoothing_diff_order=4 (a 4th-difference/"snap" penalty, not the more
# common 2nd-difference/curvature one) is a deliberate physical choice:
# a 2nd-difference penalty's null space is a straight line, which cuts
# real hairpin corners at a lambda large enough to actually damp ripple;
# a 4th-difference penalty's null space is a CLOTHOID (Euler spiral,
# curvature varying linearly with arc length) -- the actual
# transition-curve shape real road design uses between a straight and a
# constant-radius corner (see fit_uniform_b_spline's own
# smoothing_diff_order docstring for the full math), so a large lambda
# at order=4 pulls the fit toward a family real course geometry already
# resembles rather than fights.
#
# smoothing_lambda_xy stays much smaller than smoothing_lambda_z: an
# equal penalty on path shape (x/y) as on altitude (z) cuts real hairpin
# corners, shortening total course distance and inflating grade rather
# than smoothing it.
COURSE_KNOT_INTERVAL_M = 5.0
COURSE_SPLINE_DEGREE = 5
COURSE_SMOOTHING_LAMBDA_XY = 4.0
COURSE_SMOOTHING_LAMBDA_Z = 200.0
COURSE_SMOOTHING_DIFF_ORDER = 4
# Shared across all simulator versions in core.simulators.SIMULATOR_REGISTRY
# -- depends only on course shape, not on any particular physics kernel or
# physical parameters (unlike speed-limit/wind-geometry physics, which
# moved into each simulator's own module -- see fit_course_geometry_profile's
# own docstring).
def fit_uniform_b_spline(
    s_p, values, knot_interval,
    smoothing_lambda: float = 0.0,
):
    """
    Least-squares fit using a uniform B-spline of fixed degree 5
    (quintic), with knots spaced at (at most) knot_interval [m].
    Unconstrained least squares -- no boundary condition is imposed at
    either endpoint. degree/dense_margin/smoothing_diff_order are fixed
    constants (5/2/4), never varied across any of this function's 4
    real call sites in _fit_course_geometry -- only knot_interval and
    smoothing_lambda genuinely differ per call (smoothing_lambda: 4.0
    for the x/y/s_h channels, 200.0 for z -- see _fit_course_geometry's
    own docstring for why kept separate).

    Quintic degree, paired with knot_interval widened to
    COURSE_KNOT_INTERVAL_M=5m (see fit_course_geometry_profile), preserves
    course-shape fidelity while noticeably reducing slope ripple.

    knot_interval is a target spacing, not an exact one: the number of knot
    intervals is rounded UP to the smallest integer that keeps the actual
    spacing <= knot_interval, so the knot grid always spans exactly
    [s_p[0], s_p[-1]] with no unsupported gap past the true endpoint.

    Note on well-posedness: dense_s is sized off num_control_points (2x
    margin, see below), which keeps A full column rank -- and therefore
    ata invertible -- in every case tested, including n_intervals=1 (a
    single cubic segment across the whole domain).

    smoothing_lambda: 0.0 (off) is this function's own signature default
    (never actually reached by _fit_course_geometry, its only real
    caller, which always passes 4.0 or 200.0 explicitly), kept as a real
    parameter since it's the one value that genuinely differs per call.
    When > 0, adds a roughness penalty on the control points to the
    normal-equations matrix (P-spline / penalized-least-squares style,
    Eilers & Marx), turning this from a plain least-squares B-spline
    into a genuine smoothing spline:
        solve (A^T A + smoothing_lambda * D^T D) c = A^T y
    where D is a fixed order-4 finite-difference operator (see below).
    Added specifically because knot_interval alone was found, on real
    course data, to set the WAVELENGTH of a spurious ripple in the
    fitted curve's own derivative (slope), not just its amplitude --
    peak-to-peak spacing scaled almost exactly linearly with
    knot_interval (ratio ~2.3-2.6x across knot_interval=10-60m), the
    unmistakable signature of a numerical B-spline-fitting artifact, not
    real terrain: sweeping knot_interval alone can trade artifact
    amplitude against shape fidelity, but can't change its wavelength's
    dependence on knot_interval itself. A roughness penalty targets this
    directly by damping curvature independent of knot placement.

    D (the finite-difference penalty operator) is fixed at order 4:
    D[i] = c[i] - 4*c[i+1] + 6*c[i+2] - 4*c[i+3] + c[i+4] -- penalizes
    d^2CURVATURE/ds^2. Null space is curvature varying LINEARLY with arc
    length -- the exact defining property of a CLOTHOID (Euler spiral),
    the transition-curve shape real road design actually uses between a
    straight and a constant-radius corner. A large lambda at this order
    therefore pulls the fit toward a family that already resembles real
    road geometry, rather than toward a straight line (order 2's null
    space) or a circular arc (order 3's) that real transitions aren't --
    the highest order this function's fixed degree-5 (quintic) basis can
    still represent without hitting a knot discontinuity in the
    penalized derivative itself (a quintic spline is C^4 across interior
    knots -- its 5th derivative is where the piecewise polynomial pieces
    actually break). See core.activity_parser._fit_time_b_spline's own
    smoothing_diff_order docstring for the analogous (time-parametrized,
    not arc-length) argument that motivated adding this here too -- 4
    (snap, there) needed roughly 10x more lambda than 3 (jerk) before
    showing comparable distortion on real activity data, consistent with
    a higher-order penalty's null space tolerating real features (there:
    sudden braking; here: real transition curves) that a lower-order
    penalty actively fights.

    Raises:
        ValueError: if the course domain is non-positive, or if the
        resulting least-squares system turns out to be singular for some
        other degenerate reason. Fails fast with a clear message rather
        than deep inside interp1d/BSpline/np.linalg.solve with an opaque
        error.
    """
    x_start, x_end = s_p[0], s_p[-1]
    domain = x_end - x_start

    if domain <= 0:
        raise ValueError(
            f"course domain must be positive, got {domain} "
            f"(s_p[0]={x_start}, s_p[-1]={x_end})"
        )

    n_intervals = int(np.ceil(domain / knot_interval))
    eff_interval = domain / n_intervals

    # Build uniform knot vector spanning exactly [x_start, x_end]
    # (extended by the fixed degree=5 on each side)
    inner_knots = np.concatenate(
        [x_start + np.arange(n_intervals) * eff_interval, [x_end]]
    )
    prefix_knots = x_start - np.arange(COURSE_SPLINE_DEGREE, 0, -1) * eff_interval
    suffix_knots = inner_knots[-1] + np.arange(1, COURSE_SPLINE_DEGREE + 1) * eff_interval
    knots = np.concatenate([prefix_knots, inner_knots, suffix_knots])

    num_control_points = len(knots) - COURSE_SPLINE_DEGREE - 1

    # Linear interpolation to fill sparse intervals (prevents rank deficiency).
    # Sample count is sized off num_control_points (with a fixed 2x
    # margin), not off n_intervals alone -- at small n_intervals,
    # num_control_points = n_intervals + degree is dominated by the +degree
    # term, so tying the sample count to n_intervals alone under-samples
    # relative to the actual number of unknowns (verified: this was the root
    # cause of the earlier "n_intervals=1 looks singular" finding, not any
    # inherent B-spline instability). The 2x margin only controls how many
    # extra *interpolated* points get resampled between the existing raw
    # data -- it does not add new information beyond what's already in
    # (s_p, values), so it mainly helps conditioning/quantization, not
    # genuine boundary extrapolation risk.
    f_linear = interp1d(s_p, values, kind='linear')
    dense_s = np.linspace(x_start, x_end, 2 * num_control_points + 1)
    dense_z = f_linear(dense_s)

    A = BSpline.design_matrix(dense_s, knots, COURSE_SPLINE_DEGREE).toarray()
    ata = A.T @ A

    if smoothing_lambda:
        if num_control_points < COURSE_SMOOTHING_DIFF_ORDER + 1:
            raise ValueError(
                f"smoothing_lambda requires at least "
                f"{COURSE_SMOOTHING_DIFF_ORDER + 1} control points to form an "
                f"order-{COURSE_SMOOTHING_DIFF_ORDER} difference "
                f"penalty, got {num_control_points} (domain={domain:.3f}m, "
                f"knot_interval={knot_interval}m, n_intervals={n_intervals})"
            )
        n_rows = num_control_points - COURSE_SMOOTHING_DIFF_ORDER
        D = np.zeros((n_rows, num_control_points))
        for i in range(n_rows):
            D[i, i] = 1.0
            D[i, i + 1] = -4.0
            D[i, i + 2] = 6.0
            D[i, i + 3] = -4.0
            D[i, i + 4] = 1.0
        P = D.T @ D
        ata = ata + smoothing_lambda * P

    try:
        solution = np.linalg.solve(ata, A.T @ dense_z)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"B-spline least-squares system is singular (domain={domain:.3f}m, "
            f"knot_interval={knot_interval}m, n_intervals={n_intervals}): {e}"
        ) from e
    return BSpline(knots, solution[:num_control_points], COURSE_SPLINE_DEGREE)

def project_meters_to_latlon(x_m, y_m, origin_lat, origin_lon):
    """Back-project local Cartesian coordinates [m] to geographic coordinates [deg]."""
    lat_mid = np.radians(origin_lat)
    m_per_lat = 111132.92 - 559.82 * np.cos(2 * lat_mid)
    m_per_lon = 111412.84 * np.cos(lat_mid)
    lat = (y_m / m_per_lat) + origin_lat
    lon = (x_m / m_per_lon) + origin_lon
    return lat, lon

def _fit_course_geometry(
    points: CoursePoints,
    stride: float,
    knot_interval: float,
) -> dict:
    """
    Fit the raw course points to a uniform B-spline and resample onto an
    exact-arc-length grid at fixed stride.

    Depends only on the course's raw geometry (points, stride,
    knot_interval) — not on any physical parameter. This is the
    expensive stage (B-spline fit + dense arc-length resampling +
    finite-difference derivatives); its output can be cached and reused
    across physics-parameter changes (see each simulator's own
    compute_course_physics_* / recompute_course_physics_*, e.g.
    core.simulators.sim_kiritsubo's).

    degree/dense_margin/smoothing_diff_order are fixed at
    fit_uniform_b_spline's own constants (5/2/4). smoothing_lambda_xy/z
    are fixed at 4.0/200.0 (this module's COURSE_SMOOTHING_LAMBDA_XY/Z
    constants) -- forwarded to fit_uniform_b_spline's own
    smoothing_lambda, xy for the x/y/s_h fits (path shape) and z for the
    z (altitude) fit only (which is what actually drives slope ripple).
    Kept as two separate constants deliberately: applying the same
    penalty to x/y as to z was found, on real course data, to cut real
    hairpin corners, shortening total course distance and inflating
    grade rather than smoothing it. None of these five are varied by
    this function's only caller, fit_course_geometry_profile.

    Args:
        points:        Raw course points (CoursePoints).
        stride:        Evaluation grid spacing [m].
        knot_interval: B-spline knot interval [m].

    Returns:
        Dict with keys: s_p_fine, x_fine, y_fine, z_fine, s_h_fine,
        lat_fine, lon_fine, dx, dy, dz, kappa, slope, heading.
    """
    # 1. Smooth x, y, z, s_h via uniform B-spline, using the original (non-uniform)
    #    cumulative road distance (points.s_p) as the independent variable
    bs_x = fit_uniform_b_spline(points.s_p, points.x, knot_interval, COURSE_SMOOTHING_LAMBDA_XY)
    bs_y = fit_uniform_b_spline(points.s_p, points.y, knot_interval, COURSE_SMOOTHING_LAMBDA_XY)
    bs_z = fit_uniform_b_spline(points.s_p, points.z, knot_interval, COURSE_SMOOTHING_LAMBDA_Z)
    bs_s_h = fit_uniform_b_spline(points.s_p, points.s_h, knot_interval, COURSE_SMOOTHING_LAMBDA_XY)

    # 2. Dense temporary sampling for arc-length tracking (~0.1m step -- only
    # used to integrate arc length via consecutive differences below, so the
    # exact step size doesn't matter, just that it stays close to
    # dense_step). Equal-length division, both endpoints included exactly --
    # same reasoning as step 4's s_p_fine grid below.
    dense_step = 0.1
    n_dense_segments = max(1, round((points.s_p[-1] - points.s_p[0]) / dense_step))
    s_p_dense = np.linspace(points.s_p[0], points.s_p[-1], n_dense_segments + 1)

    x_dense = bs_x(s_p_dense)
    y_dense = bs_y(s_p_dense)
    z_dense = bs_z(s_p_dense)

    # 3. Compute the exact cumulative geometric arc length along the smoothed 3D curve
    s_dense = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2 + np.diff(z_dense)**2))])
    total_road_length = s_dense[-1]

    # 4. Cut out the uniformly-spaced evaluation grid over the true domain (0m to finish),
    # both endpoints included exactly. n_segments segments of EQUAL length (the
    # closest achievable equal-length division to stride, via rounding), not
    # stride-length segments plus one leftover-length final segment: an
    # arange-then-append-the-endpoint approach can make that final segment
    # arbitrarily short (down to a rounding-noise sliver, were the true
    # endpoint to land a hair past an arange step), which np.gradient's
    # edge_order=2 (below) would amplify into a spurious tail kappa/heading/
    # slope spike -- baked permanently into course JSON on save, since a
    # loaded strategy reads its saved geometry back rather than recomputing
    # it. linspace can't produce a degenerate segment: every one, including
    # the last, is exactly total_road_length/n_segments long.
    n_segments = max(1, round(total_road_length / stride))
    s_p_fine = np.linspace(0.0, total_road_length, n_segments + 1)

    # Back-solve for the coordinates in spline-independent-variable space (s_p_mapped)
    # from geometric arc-length space
    s_p_mapped = np.interp(s_p_fine, s_dense, s_p_dense)

    # 5. Extract the true-domain 3D point sequence and metadata precisely
    x_fine = bs_x(s_p_mapped)
    y_fine = bs_y(s_p_mapped)
    z_fine = bs_z(s_p_mapped)
    s_h_fine = bs_s_h(s_p_mapped)  # metadata (for embedding in the strategy file)

    # 6. [Second-order-accurate finite difference in arc-length space]
    # Pass the independent variable array s_p_fine directly so that the fractional
    # interval at the tail is also differentiated exactly
    dx = np.gradient(x_fine, s_p_fine, edge_order=2)
    dy = np.gradient(y_fine, s_p_fine, edge_order=2)
    dz = np.gradient(z_fine, s_p_fine, edge_order=2)

    ddx = np.gradient(dx, s_p_fine, edge_order=2)
    ddy = np.gradient(dy, s_p_fine, edge_order=2)
    ddz = np.gradient(dz, s_p_fine, edge_order=2)

    # Evaluate the norm of the 3D derivative vector
    v_norm = np.sqrt(dx**2 + dy**2 + dz**2)

    logger.debug(
        "[GEOMETRY FINITE DIFFERENCE] np.gradient(edge_order=2) applied (tail-exact version)"
    )
    logger.debug(
        "[GEOMETRY FINITE DIFFERENCE] v_norm: min=%.6f, max=%.6f, mean=%.6f",
        v_norm.min(), v_norm.max(), v_norm.mean(),
    )
    logger.debug("[GEOMETRY FINITE DIFFERENCE] v_norm check at last 3 points: %s", v_norm[-3:])
    logger.debug(
        "[GEOMETRY FINITE DIFFERENCE] total parameter length = %.3f m (exact total=%.3f m)",
        s_p_fine[-1], total_road_length,
    )
    if logger.isEnabledFor(logging.DEBUG):
        min_idx = np.argmin(v_norm)
        start_look = max(0, min_idx - 2)
        end_look = min(len(v_norm), min_idx + 3)
        logger.debug("[V_NORM MINIMUM IN 3D TRACK]")
        logger.debug("  - Minimum v_norm: %.6f", v_norm[min_idx])
        logger.debug("  - Index: %d / %d", min_idx, len(v_norm))
        logger.debug("  - Distance from start: %.3f m", s_p_fine[min_idx])
        logger.debug(
            "  - Coordinates: x=%.3f, y=%.3f, z=%.3f",
            x_fine[min_idx], y_fine[min_idx], z_fine[min_idx],
        )
        logger.debug(
            "  - 1st Derivs (dx, dy, dz): (%.4f, %.4f, %.4f)",
            dx[min_idx], dy[min_idx], dz[min_idx],
        )
        logger.debug(
            "  - 2nd Derivs (ddx, ddy, ddz): (%.4f, %.4f, %.4f)",
            ddx[min_idx], ddy[min_idx], ddz[min_idx],
        )
        logger.debug("  - Neighborhood v_norm: %s", v_norm[start_look:end_look])
        logger.debug("  - Neighborhood z (alt): %s", z_fine[start_look:end_look])

    # 7. Project to geographic coordinates
    lat_fine, lon_fine = project_meters_to_latlon(
        x_fine, y_fine, points.origin_lat, points.origin_lon
    )

    # 8. 3D curvature: kappa = |r' x r''| / |r'|^3
    v_norm_sq = v_norm**2
    cross_x = dy * ddz - dz * ddy
    cross_y = dz * ddx - dx * ddz
    cross_z = dx * ddy - dy * ddx
    numerator = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    kappa_raw = numerator / (v_norm_sq * v_norm)

    # 10. Slope: back-solved exactly from dz in arc-length parameter space
    slope = np.arcsin(np.clip(dz, -1.0, 1.0))

    # 13. Compute wind geometry data (heading). cos_phi/sin_phi are computed in
    # _compute_wind_geometry() given the wind direction parameter.
    heading = np.arctan2(dx, dy)

    return {
        "s_p_fine": s_p_fine,
        "x_fine": x_fine,
        "y_fine": y_fine,
        "z_fine": z_fine,
        "s_h_fine": s_h_fine,
        "lat_fine": lat_fine,
        "lon_fine": lon_fine,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "kappa_raw": kappa_raw,
        "slope": slope,
        "heading": heading,
    }


def _compute_wind_geometry(
    heading: np.ndarray, wind_direction: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cos_phi/sin_phi (apparent-wind geometry factors) from course
    heading and wind direction.

    Depends only on course heading (from _fit_course_geometry) and the
    scalar wind direction — cheap to re-run whenever wind_direction changes,
    without repeating the geometry fit or the speed-limit backward pass.
    This is the single source of truth for this calculation; callers
    (e.g. eidos.apps.analyzer's physics-override rebuild) should import and
    call this rather than reimplementing the formula.

    Args:
        heading:    Course heading [rad] at each s_p_fine sample.
        wind_direction: Wind direction [deg], meteorological convention
                    (direction the wind is blowing FROM).

    Returns:
        (cos_phi, sin_phi): apparent-wind angle cosine/sine, same length
        as heading.
    """
    d_wind_to = np.radians(wind_direction) + np.pi
    phi = heading - d_wind_to
    return np.cos(phi), np.sin(phi)


def fit_course_geometry_profile(
    points: CoursePoints,
    run: RunSettings,
) -> dict:
    """
    Fit the course-shape-only geometry (distance/slope/kappa_raw/heading/
    lat/lon/altitude) -- the one genuinely simulator-agnostic stage of
    the course-physics pipeline. Each core.simulators.SIMULATOR_REGISTRY
    entry's own module owns whichever further physics-parameter-dependent
    stages (curvature clamping, speed limits, wind geometry) it actually
    needs -- see core.simulators.sim_kiritsubo.
    compute_course_physics_sim_kiritsubo for the "full" pipeline this
    feeds into. That physics lives in each simulator's own file, not here
    as a shared utility, because this project version-tracks a
    simulator's own physics model via that file's own SIMULATOR_VERSION
    constant, which a shared, untracked helper here would bypass.

    Curvature is returned RAW (this function's own kappa_raw key, not
    clamped to any minimum cornering radius) -- that clamp only matters
    to core.simulators.sim_kiritsubo's own cornering speed-limit physics
    (core.simulators.sim_stub has no cornering-limit model at all), so it
    lives on sim_kiritsubo.PhysicalSettings.min_corner_radius alongside
    mu/brake_usability/gravity_accel. Each simulator's own
    compute_course_physics_<name> decides for itself whether (and how) to
    clamp kappa_raw before building its own CourseProfile.

    The B-spline knot interval and smoothing_lambda_xy/z (forwarded into
    _fit_course_geometry, which hardcodes them further -- see that
    function's own docstring) come from the module-level
    COURSE_KNOT_INTERVAL_M/COURSE_SMOOTHING_LAMBDA_XY/
    COURSE_SMOOTHING_LAMBDA_Z constants, not from RunSettings/config -- a
    numerical-fitting concern, not something worth making every user tune
    per run (see RunValidationModel's docstring; see those constants' own
    module comment for the current values' rationale/validation).
    """
    return _fit_course_geometry(points, run.distance_step, COURSE_KNOT_INTERVAL_M)
