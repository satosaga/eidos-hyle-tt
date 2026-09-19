##########################
# calibration_diagnostics.py
##########################
"""
Trade-off / non-identifiability visualizations for
core.calibrator.CalibrationResult.

Every plot here is built from information calibrate() already computed --
free_keys, x_best, x_std, all_trials -- no additional simulation is ever
run.

Kept as a module separate from calibrator.py on purpose: calibrator.py's
calibrate_single_trial runs inside ProcessPoolExecutor workers, and those
workers re-import the whole module to unpickle it (see
calibrator.calibrate_multistart). Anything imported at calibrator.py's top
level is therefore paid for by every DE worker whether it plots anything
or not -- so matplotlib and scipy.stats live here instead, imported only
by whichever process actually wants a diagnostic figure (normally the
main process, after calibrate() has already returned).

Pooling trials
---------------
_pool_trials reads result.all_trials, drops infeasible-physics-penalty
hits (calibrator._INFEASIBLE_PHYSICS_PENALTY_MPS), then restricts the
pool to a near-best-fit "level set" via pooling_method: "rmse_tolerance"
(an RMSE-space margin, resolved internally from the fixed
DEFAULT_RMSE_TOLERANCE_REL -- see _pool_for_plot) or "mahalanobis" (a
parameter-space confidence ellipsoid around a robust fit to the pooled
points -- see _mahalanobis_filter). x_best is plotted directly wherever
it's shown (e.g. plot_single_pair_scatter's star marker) regardless of
whether it falls inside this ellipsoid. This restriction happens before
computing anything -- see plot_parameter_correlation_heatmap.
plot_single_pair_scatter applies the SAME filter (pooling_method is the
only knob left; every other pooling constant is fixed) as whichever
heatmap it was opened from, so its title's Pearson r always matches
that cell.

If "wind_direction" is free, _pool_trials also unwraps its column around
result.x_best's own value (before EITHER filter runs) so a near-best-fit
wind direction near the 0/360 seam (e.g. a north wind) doesn't get split
into two spuriously-far clusters -- see _pool_trials' docstring.

Sensitivity screening (which parameters matter to the fit at all) is
answered BEFORE Auto Fit runs, via core.calibrator.sample_morris_
sensitivity/sample_sobol_sensitivity and eidos.apps.analyzer's inline
Sensitivity bars -- plot_calibration_diagnostics' own
panels below are a different, later question: whether the free
parameters actually chosen for Auto Fit trade off against each other in
the resulting fit. plot_sensitivity_effect_scatter/plot_sensitivity_
interaction_scatter/plot_morris_mustar_sigma_scatter are the one
exception living in this module anyway -- on-demand scatter popups for
those inline Sensitivity bars, grouped here with plot_single_pair_scatter
(matplotlib/scipy imported
only where something is actually plotted, see above) rather than in
calibrator.py itself, even though they render PRE-Auto-Fit data.

Even after pooling, the correlation heatmap's per-cell annotations are a
plain, UNCORRECTED significance gate (see plot_parameter_correlation_heatmap),
not a validated finding -- treat a cell marked significant as "worth a
second look," not as proof of a real trade-off.
"""

import logging
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats
from sklearn.covariance import MinCovDet

from core.calibrator import (
    _INFEASIBLE_PHYSICS_PENALTY_MPS,
    CalibrationResult,
    SobolSensitivityTrials,
)
from core.schema import field_display_label
from core.simulators import DEFAULT_SIMULATOR_KEY, resolve_simulator

logger = logging.getLogger(__name__)


def _display_label(key: str) -> str:
    """Pretty title for one free_keys entry, for plot AXIS LABELS/point
    annotations only -- every other use of free_keys in this module
    (pair-key construction into result.s2, .index() lookups, error-
    message identifiers) keeps the raw key itself unchanged. Matches the
    same pretty title eidos.apps.analyzer.widgets' own PhysicsOverridePanel
    shows for each row, so a matrix/scatter axis label cross-references
    directly against that panel.

    Resolved against DEFAULT_SIMULATOR_KEY, not a simulator_key threaded
    down from whichever strategy this diagnostic plot actually belongs
    to: calibratable_physical_keys is per-simulator, but Auto Fit/
    calibration against a real FIT ride
    is only ever meaningful for the real physics model (sim_kiritsubo) in
    practice -- see core.simulators.sim_stub's own docstring ("do not use
    for real training-strategy research"). Threading the actual
    simulator_key through this module's whole plotting call chain (8+
    function signatures across eidos.apps.analyzer.window/dialogs) for a
    label lookup that only ever matters for one simulator today would be
    a disproportionate internal-logic redesign."""
    model_cls = resolve_simulator(DEFAULT_SIMULATOR_KEY).physical_param_model
    return field_display_label(model_cls, key)


def _display_labels(keys: list[str]) -> list[str]:
    """_display_label, mapped over a free_keys list."""
    return [_display_label(k) for k in keys]


def _max_text_width_in(labels: list[str], fontsize: float, dpi: float) -> float:
    """Widest of `labels` at `fontsize`, in inches at `dpi` -- measured
    with matplotlib's own Agg renderer (real font metrics), not guessed
    from character counts. Used to pad a heatmap's fixed figsize (see
    plot_parameter_correlation_heatmap/plot_sobol_s2_heatmap) enough for
    the actual longest y-tick label -- these figures are shown in a
    dialogs.py QDialog whose canvas is a FIXED size pinned to the
    Figure's own nominal size (see that module's docstring), so unlike
    an interactive matplotlib window, there's no later resize pass that
    could otherwise save a too-tight margin from clipping tick text.
    Switching from raw free_keys (short, roughly uniform length) to
    pretty titles (some much longer, e.g. "Wind direction [deg]") is
    what made that margin too tight to keep hardcoding as a flat
    n-only-based constant.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    tmp_fig = Figure()
    FigureCanvasAgg(tmp_fig)
    renderer = tmp_fig.canvas.get_renderer()
    widths_px = [
        tmp_fig.text(0, 0, label, fontsize=fontsize).get_window_extent(renderer).width
        for label in labels
    ]
    return (max(widths_px) if widths_px else 0.0) / dpi

# Uncorrected per-cell significance threshold for the correlation heatmap
# -- see plot_parameter_correlation_heatmap's docstring for why this is
# deliberately not Bonferroni/FDR-corrected for the C(n_free,2)
# simultaneous comparisons.
DEFAULT_ALPHA = 0.05

# Default relative RMSE margin (fraction of result.rmse_mps) defining the
# "near-best-fit" level set a trial must fall within to be pooled for the
# correlation heatmap. A starting point, not a validated constant.
DEFAULT_RMSE_TOLERANCE_REL = 0.05

# Default pooling method -- see _pool_trials' docstring.
DEFAULT_POOLING_METHOD = "rmse_tolerance"

# Chi-squared confidence level for pooling_method="mahalanobis": a pooled
# point is kept if its squared Mahalanobis distance from the robust-fit
# center is below chi2.ppf(DEFAULT_MAHALANOBIS_ALPHA, df=n_free).
DEFAULT_MAHALANOBIS_ALPHA = 0.95

# Margin (inches) reserved past the figure's right edge so VIFs' title
# doesn't clip -- do not remove, no test catches the clipping.
_RIGHT_MARGIN_IN = 0.25

# gridspec `wspace` for plot_parameter_correlation_heatmap's nested
# column groups. Not interchangeable: `wspace` is a fraction of each
# gridspec's OWN mean column width, and the inner (Correlations/Loadings)
# and outer (corr/load block vs. VIFs) grids have different column-width
# mixes, so the same number doesn't buy the same pixel gap in both --
# values below were reached empirically, not derived.
_OUTER_WSPACE = 0.02
_INNER_WSPACE = 0.035

# Single-hue (green) sequential ramp for every highlight-threshold bar in
# this figure, light->dark = small->large magnitude -- deliberately not
# red/blue/orange (red and blue are already the Correlations/Loadings
# RdBu_r colormap's own colors in the same figure; orange sits close
# enough to RdBu_r's red end to be confused with it in a quick glance)
# and deliberately ONE hue rather than a categorical palette, so a
# viewer can read "small vs large" from lightness alone without having
# to already know which hue means what.
_BELOW_THRESHOLD_COLOR = "#a9dca3"
_MEDIUM_THRESHOLD_COLOR = "#4bb062"
_ABOVE_THRESHOLD_COLOR = "#077331"

# Any pooled trial with fun at or above this is an objective_calibration
# infeasible-physics-penalty hit (calibrator._INFEASIBLE_PHYSICS_PENALTY_MPS),
# not a real fit -- dropped from every pooled set unconditionally,
# regardless of RMSE tolerance.
_FEASIBLE_FUN_CUTOFF_MPS = _INFEASIBLE_PHYSICS_PENALTY_MPS / 2.0


# ---------------------------------------------------------------------------
# I. Pooling (shared by the correlation heatmap and the single-pair scatter)
# ---------------------------------------------------------------------------

@dataclass
class _PooledTrials:
    """
    Parameter vectors + RMSE pooled from result.all_trials -- see this
    module's docstring.

    Attributes:
        x:   (n_points, n_free) array, columns in free_keys order.
        fun: (n_points,) velocity RMSE [m/s] per point.
    """
    x: np.ndarray
    fun: np.ndarray


def _mahalanobis_filter(
    x: np.ndarray,
    fun: np.ndarray,
    *,
    x_best: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    pooling_method="mahalanobis"'s filter: restricts pooling to a
    PARAMETER-space neighborhood instead of an RMSE-space one.

    Fits a robust (location, covariance) pair to `x` via
    sklearn.covariance.MinCovDet (FastMCD), then keeps points whose
    squared Mahalanobis distance from that center is below
    chi2.ppf(alpha, df=n_free) -- the standard robust-Mahalanobis-distance
    outlier test (Rousseeuw & Van Driessen 1999). Works for n_free=1 too:
    MinCovDet's "covariance" there is a scalar variance, and squared
    Mahalanobis distance is ((x-location)/scale)^2, a robust z-score
    squared whose null distribution is chi2(df=1) -- no separate 1D case
    needed. MCD (not the plain sample covariance) so a second cluster or
    a long tail in one free_key can't inflate the fitted ellipsoid to
    re-include the points it should exclude.

    The center is MCD's own robust location, not x_best (contrast the
    wind_direction unwrap below, which IS centered on x_best). x_best's
    own membership in the resulting ellipsoid is not checked or required
    here -- it's plotted directly by callers (e.g.
    plot_single_pair_scatter's star marker) regardless of whether it
    survived this filter.

    Args:
        x:      (n, n_free) pooled parameter vectors -- already
                    feasibility-filtered and wind_direction-unwrapped by
                    the caller (_pool_trials).
        fun:    (n,) matching RMSE values.
        x_best: This run's best-fit parameter vector, free_keys order --
                    not used by this function; kept for signature
                    symmetry with _pool_trials.
        alpha:  Chi-squared confidence level -- see DEFAULT_MAHALANOBIS_ALPHA.

    MinCovDet's own support_fraction is always None (its own auto-
    selected default) -- no caller has ever needed to override it.

    Returns:
        (x, fun) filtered to points within the robust-fit ellipsoid --
        x_best itself may or may not be among them.

    Raises:
        ValueError: If n <= n_free -- MinCovDet needs strictly more
                    points than dimensions to fit a non-singular
                    covariance.
    """
    n, n_free = x.shape
    if n <= n_free:
        raise ValueError(
            f"_mahalanobis_filter: only {n} feasible pooled point(s) for "
            f"{n_free} free parameters -- MinCovDet needs strictly more "
            f"points than dimensions to fit a non-singular covariance. Run "
            f"more Auto Fit seeds, or use pooling_method='rmse_tolerance'."
        )

    mcd = MinCovDet(support_fraction=None, random_state=0).fit(x)
    threshold = float(stats.chi2.ppf(alpha, df=n_free))
    d2 = mcd.mahalanobis(x)
    keep = d2 <= threshold
    return x[keep], fun[keep]


def _pool_trials(
    result: CalibrationResult,
    *,
    pooling_method: str = DEFAULT_POOLING_METHOD,
    rmse_tolerance_mps: float,
) -> _PooledTrials:
    """
    Gather (x, fun) from result.all_trials, dropping infeasible-physics-
    penalty hits, then apply the near-best-fit level-set filter chosen
    via pooling_method:

    - "rmse_tolerance": drop any point whose fun exceeds
      result.rmse_mps + rmse_tolerance_mps. Filters in RMSE space only.
    - "mahalanobis": drop any point outside a robust-fit confidence
      ellipsoid in PARAMETER space -- see _mahalanobis_filter.

    If "wind_direction" is a free key, its column is unwrapped around
    result.x_best's own wind_direction value before either filter runs:
    DE/NM's bounds treat [0, 360) as a plain interval, not a circle, so a
    near-best-fit wind direction near the 0/360 seam (e.g. a north wind)
    can produce level-set trials on BOTH sides of the seam -- physically
    a few degrees apart, but ~360 apart in raw degrees. Left alone, that
    seam corrupts every consumer of this module's pooled x for that
    column (Correlations, Loadings/Eigenvalues, VIFs,
    plot_single_pair_scatter, and now the Mahalanobis distance itself)
    with an artifact of where the optimizer landed, not a real
    relationship. Re-centering on x_best into (center-180, center+180]
    removes the seam without changing units. Only works because the
    pooled set is a near-best-fit set around one point to begin with.

    Args:
        result: A CalibrationResult.
        pooling_method: "rmse_tolerance" or "mahalanobis" -- see above.
        rmse_tolerance_mps: Used only if pooling_method="rmse_tolerance".
                    Drop pooled points whose fun is more than this much
                    worse than result.rmse_mps. Always resolved to a
                    concrete value by this function's only caller,
                    _pool_for_plot, from the fixed DEFAULT_RMSE_
                    TOLERANCE_REL -- no caller has ever needed an
                    unrestricted ("keep everything") pool.

    pooling_method="mahalanobis" always uses _mahalanobis_filter's own
    fixed alpha (DEFAULT_MAHALANOBIS_ALPHA) and MinCovDet's own auto-
    selected support_fraction -- neither has ever been varied by this
    function's only caller.

    Returns:
        A _PooledTrials (possibly with zero points for
        pooling_method="rmse_tolerance", e.g. if every trial happened to
        be an infeasible-physics hit -- callers check length).

    Raises:
        ValueError: If pooling_method is neither "rmse_tolerance" nor
                    "mahalanobis", or (pooling_method="mahalanobis" only)
                    anything _mahalanobis_filter raises -- see that
                    function's docstring.
    """
    n_free = len(result.free_keys)
    xs: list[np.ndarray] = []
    funs: list[float] = []

    for t in result.all_trials:
        fun_val = float(t.fun)
        if fun_val < _FEASIBLE_FUN_CUTOFF_MPS:
            xs.append(np.asarray(t.x, dtype=float))
            funs.append(fun_val)

    x = np.array(xs) if xs else np.empty((0, n_free))
    fun = np.array(funs)

    if "wind_direction" in result.free_keys and len(x):
        idx = result.free_keys.index("wind_direction")
        center = float(result.x_best[idx])
        x[:, idx] = center + ((x[:, idx] - center + 180.0) % 360.0 - 180.0)

    if pooling_method == "rmse_tolerance":
        if len(fun):
            keep = fun <= (result.rmse_mps + rmse_tolerance_mps)
            x, fun = x[keep], fun[keep]
    elif pooling_method == "mahalanobis":
        if len(fun):
            x, fun = _mahalanobis_filter(
                x, fun, x_best=np.asarray(result.x_best, dtype=float),
                alpha=DEFAULT_MAHALANOBIS_ALPHA,
            )
    else:
        raise ValueError(
            f"_pool_trials: unknown pooling_method {pooling_method!r} -- "
            f"expected 'rmse_tolerance' or 'mahalanobis'."
        )

    return _PooledTrials(x=x, fun=fun)


def _pool_for_plot(
    result: CalibrationResult,
    *,
    pooling_method: str,
) -> _PooledTrials:
    """
    Shared by plot_parameter_correlation_heatmap/plot_single_pair_scatter:
    resolves the fixed DEFAULT_RMSE_TOLERANCE_REL into an absolute RMSE
    tolerance (used only for pooling_method="rmse_tolerance"; harmless to
    compute unconditionally since it's cheap) and calls _pool_trials.
    Pulled out to one place rather than each of those two functions
    repeating the same line -- see this module's general no-duplication
    convention.
    """
    tol = result.rmse_mps * DEFAULT_RMSE_TOLERANCE_REL
    return _pool_trials(result, pooling_method=pooling_method, rmse_tolerance_mps=tol)


def _fmt_signed(v: float) -> str:
    """
    "{v:.2f}", with a proper Unicode minus sign (U+2212) in place of the
    ASCII hyphen-minus Python's own float formatting produces for
    negative values. Used for every Correlations/Loadings cell
    annotation: the ASCII glyph renders short and sits flush against the
    digit at this figure's small cell fontsize, easy to misread as part
    of the number itself rather than its sign.
    """
    return f"{v:.2f}".replace("-", "−")


# ---------------------------------------------------------------------------
# II. Parameter correlation heatmap
# ---------------------------------------------------------------------------

def plot_parameter_correlation_heatmap(
    result: CalibrationResult,
    *,
    pooling_method: str = DEFAULT_POOLING_METHOD,
) -> Figure | None:
    """
    Four row-aligned panels, indexed by the SAME free_keys in the SAME
    top-to-bottom row order (sharey=True across all of them): a Pearson
    correlation heatmap (Correlations), an identifiability-directions
    loadings heatmap (Loadings -- PCA on the same correlation matrix, see
    _eigendecompose_correlation), and a variance-inflation-factor (VIF)
    bar chart (VIFs). The fourth panel (Eigenvalues, eigenvalue bars)
    sits directly above Loadings, sharing ITS x-axis (principal direction
    1..n_free) instead of the free_keys row-axis, since it's the one
    panel indexed by direction rather than by parameter.

    Eigenvalues/Loadings eigendecompose the pooled parameters'
    CORRELATION matrix (each parameter z-scored to unit variance first,
    equivalent to np.corrcoef), not a bounds-normalized covariance
    matrix: two parameters can be equally uncertain in absolute terms
    (e.g. trials agree to within +/-3-4kg on both rider_weight and
    bike_weight, a shared "total mass" degeneracy) yet a covariance-based
    view would still let whichever one has the larger relative std
    (fraction of its own bounds span) dominate the leading eigenvector --
    purely because its bounds span happens to be narrower, not because it
    is the real driver. Standardizing to correlation removes each
    parameter's own variance scale from the picture, so the leading
    direction reflects actual co-movement instead. Correlation-matrix
    eigenvalues sum to n_free and are bounded below by 0; under complete
    independence every eigenvalue is exactly 1, so an eigenvalue >= 1
    (Kaiser's criterion, the fixed direction_highlight_threshold below)
    means that direction is soaking up more variance than a single
    independent parameter could -- a genuine joint degeneracy, not
    measurement noise. A given eigenvector's overall SIGN is arbitrary
    (eigh may flip it between runs) -- only the relative loadings within
    one direction (which parameters, how large, same vs. opposite sign)
    are meaningful.

    Whether a parameter matters to the fit AT ALL is a separate, earlier
    question this figure no longer answers -- see core.calibrator.
    sample_morris_sensitivity/sample_sobol_sensitivity and
    TTAnalyzerWindow's inline Sobol'/Morris sensitivity bars (rendered
    directly into PhysicsOverridePanel, next to the Auto Fit checkboxes),
    run BEFORE Auto Fit to choose which parameters to free in the first
    place. What's left here is a
    later, narrower question: whether the free parameters actually
    CHOSEN trade off against each other in the resulting fit.

    Each off-diagonal Correlations cell is annotated with its Pearson r
    and, via scipy.stats.pearsonr, its two-sided p-value against the null
    of zero correlation. Cells that do NOT clear an UNCORRECTED p < alpha
    are visually muted (a semi-transparent white overlay, non-bold gray
    text) rather than hidden, so the viewer sees both the point estimate
    and its (lack of) statistical support in the same picture. This is
    deliberately uncorrected for the C(n_free,2) simultaneous comparisons
    being eyeballed at once (a Bonferroni/FDR pass would raise the bar
    further, e.g. from |r|>=0.40 to |r|>=0.63 at n=25 for a 12-parameter
    run) -- read a "significant" cell as "worth a second look," not as a
    validated finding. Correlations and Loadings intentionally SHARE one
    colorbar (both use the identical RdBu_r colormap over the identical
    [-1, 1] range -- a Pearson r and an eigenvector loading are different
    quantities, but a value of e.g. 0.7 means the same color in both, so
    one colorbar serves both without losing information).

    VIFs are the diagonal of the correlation matrix's inverse (a standard
    identity: VIF_k = 1 / (1 - R_k^2), where R_k^2 comes from regressing
    parameter k on every OTHER free parameter, equals [R^-1]_kk when R is
    the correlation matrix -- avoids n_free separate regressions since
    r_mat is already built for the Correlations panel above). A pairwise
    Correlations cell or Directions loading can miss a parameter that's
    only entangled through a COMBINATION of others (e.g. k trades off
    against (i - j), not against either alone) -- VIF catches that
    because it regresses on all other parameters simultaneously. VIF is
    mathematically >= 1 (1 = no multicollinearity at all); the
    conventional rule-of-thumb concern threshold is 5-10 -- this defaults
    to the more sensitive end (5.0, matching this module's general "worth
    a second look" bias, e.g. alpha's deliberately-uncorrected default --
    see above). If r_mat is exactly singular (perfect multicollinearity,
    a true VIF of infinity for at least one parameter), this raises
    ValueError instead of computing an approximate lower bound via
    pseudo-inverse (see Raises below): a singular correlation matrix
    means those parameters are not independently identifiable from this
    run's near-best-fit level set at all, a real finding worth surfacing
    directly rather than a number to quietly under-report.

    Args:
        result: A CalibrationResult.
        pooling_method: "rmse_tolerance" or "mahalanobis" -- which
                    near-best-fit level-set filter selects the pooled
                    points every panel here is built from. See
                    _pool_trials' docstring for what each one does and
                    why both exist side by side for now.
    The RMSE-space pooling margin (pooling_method="rmse_tolerance") is
    always the fixed DEFAULT_RMSE_TOLERANCE_REL fraction of
    result.rmse_mps -- see _pool_for_plot. The parameter-space pooling
    ellipsoid (pooling_method="mahalanobis") always uses
    _mahalanobis_filter's own fixed alpha (DEFAULT_MAHALANOBIS_ALPHA)
    and MinCovDet's own auto-selected support_fraction. The per-cell
    significance threshold (DEFAULT_ALPHA), the VIF panel's two color
    thresholds (5.0/10.0), and the Eigenvalues panel's warning cutoff
    (1.0, Kaiser's criterion) are fixed constants below. No caller of
    this function has ever varied any of these.

    Returns:
        A matplotlib Figure, or None (with a logged warning) if fewer
        than 4 pooled points survive filtering -- 3 is the mathematical
        minimum for a defined correlation, 4 is still nowhere near
        "enough," just where this stops being pure noise-fitting.
        plot_calibration_diagnostics (this module's higher-level entry
        point) turns a None here into a raised ValueError rather than
        silently substituting anything (see that function's docstring)
        -- a caller of THIS function directly just gets the None back,
        as documented, and decides for itself what to do about it.

    Raises:
        ValueError:
                    - If any pairwise correlation comes out NaN -- a
                      pooled free parameter never moved within its
                      sample, making that correlation mathematically
                      undefined (0/0), not zero.
                    - If r_mat is exactly singular (perfect
                      multicollinearity -- true VIF is infinite for at
                      least one parameter).
                    - pooling_method="mahalanobis" only: anything
                      _mahalanobis_filter raises -- see that function's
                      docstring.
    """
    # Fixed constants -- no caller of this function has ever varied any
    # of these (see this module's general "no default arguments unless
    # genuinely varied" convention).
    alpha = DEFAULT_ALPHA
    vif_highlight_threshold = 5.0
    vif_severe_threshold = 10.0
    direction_highlight_threshold = 1.0

    pooled = _pool_for_plot(result, pooling_method=pooling_method)
    n = len(pooled.fun)
    keys = result.free_keys
    n_free = len(keys)

    # n_free == 1 is a legitimate, if trivial, case (a 1x1 "correlation
    # matrix", r=1.00 on the diagonal and nothing else -- see
    # _eigendecompose_correlation's atleast_2d comment for the numpy
    # shape numpy.corrcoef needs for this) and renders normally below.
    # n_free == 0 genuinely has nothing to plot -- not a realistic
    # calibrate() outcome (free_keys is never empty), but guarded anyway
    # rather than letting an empty-array shape mismatch surface deeper in
    # this function.
    if n_free < 1:
        logger.warning(
            "plot_parameter_correlation_heatmap: 0 free parameters -- nothing "
            "to plot -- skipping.",
        )
        return None
    if n < 4:
        logger.warning(
            "plot_parameter_correlation_heatmap: only %d pooled point(s) after "
            "feasibility/RMSE-tolerance filtering (need >= 4) -- skipping.", n,
        )
        return None
    if n <= n_free:
        logger.warning(
            "plot_parameter_correlation_heatmap: only %d pooled point(s) for %d "
            "free parameters -- the Directions panel's correlation matrix is "
            "rank-deficient, so some of its eigenvalues will read as exactly 0 "
            "for lack of data, not because those directions are actually "
            "perfectly identified. Treat this run's smallest eigenvalue bars "
            "with extra caution.", n, n_free,
        )

    r_mat = np.eye(n_free)
    p_mat = np.zeros((n_free, n_free))
    for i in range(n_free):
        for j in range(i + 1, n_free):
            r, p = stats.pearsonr(pooled.x[:, i], pooled.x[:, j])
            r_mat[i, j] = r_mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p
    # pearsonr on a zero-variance column (a free parameter that never
    # moved within the near-best-fit level set) silently returns NaN
    # rather than erroring -- checked here, not guessed at in advance.
    # Identified by EVERY off-diagonal entry in a row being NaN (not
    # just any), since one degenerate column poisons every other row's
    # single paired entry too -- "any NaN" would misname every
    # parameter as an offender instead of just the actual one.
    if np.any(np.isnan(r_mat)):
        offenders = [keys[i] for i in np.where(np.sum(np.isnan(r_mat), axis=1) == n_free - 1)[0]]
        raise ValueError(
            f"{offenders} produced NaN correlation(s) -- likely zero "
            f"variance across the {n} pooled points (never moved within "
            f"the near-best-fit level set). Investigate that parameter's "
            f"bounds/search space for this run rather than trusting a "
            f"Correlations panel built by silently papering over this."
        )

    eigvals, eigvecs = _eigendecompose_correlation(pooled, n_free, keys)

    # VIF_k = [R^-1]_kk -- see this function's docstring for the identity
    # this uses (avoids n_free separate "regress k on everything else"
    # fits). r_mat is well-conditioned in the overwhelmingly common case
    # (real near-best-fit samples essentially never hit EXACT
    # multicollinearity -- e.g. a rider/bike mass-split trade-off
    # typically shows r=-1.00 only to 2 decimal places, not exactly). A
    # singular matrix means true
    # VIF is infinite for at least one parameter -- that is a real finding
    # about this run (those parameters are not independently identifiable
    # at all within the near-best-fit level set), not a numerical
    # inconvenience to route around with a pseudo-inverse "approximate
    # lower bound" and a VIFs panel that quietly under-reports it as some
    # large-but-finite number. Fail fast instead.
    try:
        inv_r = np.linalg.inv(r_mat)
    except np.linalg.LinAlgError as exc:
        near_singular_pairs = [
            (keys[i], keys[j])
            for i in range(n_free) for j in range(i + 1, n_free)
            if abs(r_mat[i, j]) >= 0.999
        ]
        hint = (
            f" Near-perfectly-correlated pair(s) (|r|>=0.999): "
            f"{near_singular_pairs} -- start there."
            if near_singular_pairs else
            " No single pair reaches |r|>=0.999, so the singularity is "
            "likely a joint (3+ parameter) linear dependency rather than a "
            "simple pairwise one -- check the Loadings panel's dominant "
            "eigenvector(s) instead."
        )
        raise ValueError(
            f"Correlation matrix is exactly singular (perfect "
            f"multicollinearity among some parameters) -- true VIF is "
            f"infinite for at least one of {keys}, not just large."
            f"{hint} This means those parameters are not independently "
            f"constrained by this run's near-best-fit level set at all; "
            f"investigate the model/bounds rather than trusting a VIFs "
            f"panel built from a pseudo-inverse approximation."
        ) from exc
    # VIF_k = [R^-1]_kk is mathematically >= 1 for any valid correlation
    # matrix R. Deliberately unguarded beyond the LinAlgError catch above
    # (an exactly-singular r_mat): no tolerance-clipped floor here -- this
    # codebase doesn't guess magic-number thresholds to absorb
    # hypothetical float noise upstream can't actually produce.
    vifs = np.diag(inv_r)

    # layout="constrained", not tight_layout() -- the caller embeds this
    # Figure in a Qt canvas that gets redrawn at different sizes (window
    # resize); tight_layout() computes spacing once and goes stale.
    #
    # dir_h/bar_w are the figure's "free" dimensions (not tied to
    # n_free the way main_h/heat_w are): Eigenvalues' bars extend upward
    # (height = value axis), VIFs' bars extend rightward (width = value
    # axis).
    heat_w = max(5.62, 0.412 * n_free)
    main_h = max(4.12, 0.412 * n_free)
    # heat_w/main_h are independently-tuned minimums and were never
    # naturally equal; forcing both to their max keeps Correlations/
    # Loadings square (paired with box_aspect(1) below) instead of a
    # stretched rectangle.
    heat_w = main_h = max(heat_w, main_h)
    dir_h = 0.85
    # Equal to dir_h: any narrower and the vif_highlight_threshold/
    # vif_severe_threshold tick labels (5 and 10, only 0.3 decades apart
    # on the log axis) render too close together and visually overlap.
    # Not a tradeoff against wasted space -- this function trims the
    # whole FIGURE down to its real content extent (below), which works
    # regardless of bar_w.
    bar_w = dir_h
    # _RIGHT_MARGIN_IN reserves real blank space past VIFs (the rightmost
    # panel) for its title's overhang -- do not remove, no test catches
    # the resulting clipped title.
    plot_w = heat_w * 2 + bar_w
    fig_w = plot_w + _RIGHT_MARGIN_IN
    fig = Figure(figsize=(fig_w, main_h + dir_h), layout="constrained")
    layout_engine = fig.get_layout_engine()
    assert layout_engine is not None  # guaranteed by layout="constrained" above
    layout_engine.set(rect=(0, 0, plot_w / fig_w, 1))
    # Nested gridspec, not one flat 2x3: a flat gridspec's wspace is one
    # scalar shared by every column pair, so Correlations<->Loadings
    # couldn't be tuned independently of Loadings<->VIFs. This split
    # gives Correlations<->Loadings its own wspace (_INNER_WSPACE) while
    # Loadings<->VIFs uses the outer gridspec's own wspace (_OUTER_WSPACE).
    # height_ratios=[dir_h, main_h] is repeated identically on both grids
    # so row-1 lines up across them.
    outer_gs = fig.add_gridspec(
        2, 2,
        height_ratios=[dir_h, main_h],
        width_ratios=[heat_w * 2, bar_w],
        wspace=_OUTER_WSPACE,
    )
    corr_load_gs = outer_gs[:, 0].subgridspec(
        2, 2,
        height_ratios=[dir_h, main_h],
        width_ratios=[heat_w, heat_w],
        wspace=_INNER_WSPACE,
    )
    ax = fig.add_subplot(corr_load_gs[1, 0])                    # Correlations
    # Labeled so a caller with only the Figure (e.g. dialogs' click-a-cell
    # popup) can find this axes via fig.axes without depending on
    # creation order.
    ax.set_label("correlations_heatmap")
    # set_box_aspect(1): equal gridspec ratios alone do NOT make
    # Correlations/Loadings square -- their tick-label margins differ
    # (Correlations has long rotated y-labels, Loadings has none), so
    # constrained_layout carves out different amounts of space from each.
    # Do not remove; no test catches the resulting non-square heatmaps.
    ax.set_box_aspect(1)
    ax_eig = fig.add_subplot(corr_load_gs[0, 1])                # Eigenvalues
    ax_eig.set_label("eigenvalues_bar")
    ax_load = fig.add_subplot(corr_load_gs[1, 1], sharex=ax_eig, sharey=ax)  # Loadings
    ax_load.set_label("loadings_heatmap")
    ax_load.set_box_aspect(1)
    # ax_eig box_aspect: sharex ties data limits, not physical width.
    # Loadings' box_aspect(1) narrows it below its gridspec column, so
    # without a matching box_aspect here, Eigenvalues' bars drift out of
    # alignment with Loadings' columns below them. dir_h/main_h makes
    # Eigenvalues track whatever width Loadings actually renders at,
    # instead of a hardcoded pixel value that breaks when n_free changes.
    ax_eig.set_box_aspect(dir_h / main_h)
    ax_vif = fig.add_subplot(outer_gs[1, 1], sharey=ax)         # VIFs
    ax_vif.set_label("vifs_bar")
    # corr_load_gs[0, 0] holds the shared colorbar (below). outer_gs[0, 1]
    # (above VIFs) is left empty on purpose: Eigenvalues has no free_keys
    # row-axis to align with it. VIFs' own title (below) sits on ITS OWN
    # axes via plain ax.set_title() instead, same as Eigenvalues/Loadings/
    # Correlations -- a title belongs directly above the panel it labels,
    # not floated up in row 0 next to Eigenvalues. dir_h and main_h are
    # independent gridspec rows, so a multi-line title growing row 1
    # (main_h, where VIFs lives) never affects row 0's height.
    ax_cbar = fig.add_subplot(corr_load_gs[0, 0])

    # aspect="auto": imshow defaults to aspect="equal", which letterboxes
    # the image inside its box whenever the box isn't exactly square,
    # insetting it vertically -- while ax_vif's bars fill their box edge
    # to edge. Without "auto", shared y-limits (sharey) would still land
    # the same data value at a different pixel row per panel. Do not
    # remove; no test catches rows drifting out of alignment.
    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n_free):
        for j in range(n_free):
            significant = (i == j) or (p_mat[i, j] < alpha)
            if not significant:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="white", alpha=0.55, edgecolor="none",
                ))
            ax.text(
                j, i, _fmt_signed(r_mat[i, j]),
                ha="center", va="center", fontsize=5.5,
                fontweight="bold" if significant else "normal",
                color="black" if significant else "gray",
            )

    ax.set_xticks(range(n_free))
    ax.set_yticks(range(n_free))
    display_keys = _display_labels(keys)
    ax.set_xticklabels(display_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_keys, fontsize=8)
    # loc="left" + fontsize 10 to match Eigenvalues/Loadings. The dropped
    # detail (uncorrected p >= alpha means muted) is in this function's
    # own docstring.
    ax.set_title(f"Pearson Correlations (n={n})", fontsize=10, loc="left")

    positions = range(n_free)
    eig_colors = [_ABOVE_THRESHOLD_COLOR if v >= direction_highlight_threshold else _BELOW_THRESHOLD_COLOR for v in eigvals]
    ax_eig.bar(positions, eigvals, color=eig_colors, edgecolor="black", linewidth=0.8)
    # No legend box -- bar color already encodes the >= threshold
    # distinction the dashed line marks.
    ax_eig.axhline(direction_highlight_threshold, color="gray", linestyle="--", linewidth=1)
    ax_eig.tick_params(labelbottom=False, labelsize=7)
    ax_eig.set_ylabel("Eigenvalue", fontsize=8)
    ax_eig.set_title("Eigenvalues", fontsize=10, loc="left")  # loc="left": matplotlib default (centered) reads as just another axis label here

    # Shares Correlations' colorbar (im, below) -- same cmap/vmin/vmax.
    ax_load.imshow(eigvecs, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_load.set_xticks(list(positions))
    ax_load.set_xticklabels([str(i) for i in range(1, n_free + 1)], fontsize=7)
    ax_load.set_xlabel("Principal Component", fontsize=8)
    ax_load.set_title("Loadings", fontsize=10, loc="left")
    ax_load.tick_params(labelleft=False)  # row labels already on ax (shared y-axis)
    for i in range(n_free):
        for j in range(n_free):
            v = eigvecs[i, j]
            if abs(v) >= 0.3:  # only annotate meaningfully-loaded cells
                ax_load.text(j, i, _fmt_signed(v), ha="center", va="center", fontsize=5.5,
                             color="white" if abs(v) > 0.6 else "black")

    # Colorbar for the shared [-1,1] scale (Pearson r in Correlations,
    # eigenvector loading in Loadings) -- placed in the otherwise-unused
    # cell above Correlations since it isn't "attached" to either panel
    # specifically. ax_cbar reserves the cell's layout space (kept
    # invisible); an inset_axes spanning the middle 50% of its width and
    # height holds the actual colorbar.
    ax_cbar.axis("off")
    cbar_box = ax_cbar.inset_axes((0.25, 0.25, 0.5, 0.5))
    cbar = fig.colorbar(im, cax=cbar_box, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)

    # VIF panel: log-scaled x-axis, floored at 1 (VIF's own mathematical
    # floor -- see the docstring's VIF identity), 3-way colored (VIF has
    # no natural upper bound and commonly spans over an order of
    # magnitude, unlike the 2-way panels elsewhere here).
    vif_colors = [
        _ABOVE_THRESHOLD_COLOR if v >= vif_severe_threshold
        else _MEDIUM_THRESHOLD_COLOR if v >= vif_highlight_threshold
        else _BELOW_THRESHOLD_COLOR
        for v in vifs
    ]
    ax_vif.barh(range(n_free), vifs, color=vif_colors, edgecolor="black", linewidth=0.8)
    ax_vif.set_xscale("log")
    # FIXED range, 10^0 to 10^3 -- not auto-scaled to this run's actual
    # VIFs. Auto-scaling fails in both directions: a run with no
    # multicollinearity (every VIF near 1) would zoom the axis down to
    # ~[1, 1.1], pushing both threshold lines off-canvas; a near-singular
    # run (VIF ~1e10) would stretch the log axis across ten decades,
    # crowding 1/5/10 against an unreadable tick jumble. 1-1000
    # comfortably covers "no concern" through "solidly severe"; a bar
    # past 1000 just runs off the right edge, itself an unambiguous
    # "far past severe" signal.
    ax_vif.set_xlim(1.0, 1000.0)  # left=1.0: VIF's own floor, not a bar chart's usual 0 (log(0) is undefined anyway)
    # Explicit ticks, not matplotlib's own log-scale locator (which,
    # unconstrained, spaces minor ticks at 2/3/4/.../9 per decade --
    # three decades of those is far more than this panel's narrow width
    # can label without overlapping). 1/5/10 are the numbers that
    # actually matter (VIF's own floor and the two thresholds); 100/1000
    # just mark the two decades of headroom above "severe" so the fixed
    # range doesn't read as an arbitrary crop.
    ax_vif.set_xticks(sorted({1.0, vif_highlight_threshold, vif_severe_threshold, 100.0, 1000.0}))
    ax_vif.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax_vif.xaxis.set_minor_formatter(NullFormatter())
    ax_vif.axvline(vif_highlight_threshold, color="gray", linestyle="--", linewidth=1)
    ax_vif.axvline(vif_severe_threshold, color="gray", linestyle="--", linewidth=1)
    ax_vif.set_xlabel("VIF", fontsize=8)
    ax_vif.set_title("VIFs", fontsize=10, loc="left")
    # rotation=90: "5" and "10" sit close enough together on a log scale
    # (0.3 decades apart, vs. 1's own 0.7-decade gap from "5") that at
    # this panel's width their horizontal labels run together into "510".
    # Vertical labels take only their own thin column regardless of how
    # close two tick positions are, so this fixes it at any width.
    ax_vif.tick_params(axis="x", labelsize=7, rotation=90)
    ax_vif.tick_params(labelleft=False)  # row labels already on ax (shared y-axis)

    # Explicit, not left to sharey's/sharex's own autoscale resolution --
    # imshow's default extent puts row/col 0 at one end (an inverted
    # range for y, exact for x) while bar()/barh()'s default view pads
    # its data range with matplotlib's own 5% autoscale margin -- close
    # but NOT identical to imshow's exact extent, and which one "wins" on
    # a shared axis isn't something to rely on. Pin both explicitly
    # rather than trust different plot types' autoscale requests to
    # resolve to the same thing (this is the same class of bug the
    # y-axis pin below and the x-axis pin were each added to fix, the
    # first time this module had a heatmap sharing an axis with a
    # non-heatmap panel).
    ax.set_ylim(n_free - 0.5, -0.5)
    ax_load.set_xlim(-0.5, n_free - 0.5)
    return fig


def equalize_correlation_heatmap_gaps(fig: Figure) -> None:
    """
    Close the Correlations<->Loadings and Loadings<->VIFs horizontal gaps
    in a plot_parameter_correlation_heatmap Figure down to one consistent
    width, and pin Loadings/Eigenvalues/VIFs' row-1 band (y0, height) to
    exactly match Correlations' own. Each panel's own title (plain
    ax.set_title(), same as Eigenvalues/Loadings/Correlations) rides
    along with its axes' set_position() call below automatically -- no
    separate title-cell repositioning needed.

    Gridspec `wspace` tuning alone isn't sufficient: Correlations/
    Loadings both use set_box_aspect(1), and how much of their gridspec
    column that square actually fills depends on the real renderer's
    font metrics, which a dev-time render can't predict -- left alone,
    the resulting Correlations<->Loadings and Loadings<->VIFs gaps come
    out visibly uneven (VIFs has no box_aspect, so IT already sits
    exactly where the outer gridspec intends). Rather than measuring an
    already-rendered "clean" pair to infer what gap the layout intended
    (fragile once there's only one non-box_aspect panel left in the
    row), this reads the target gap directly off the outer gridspec's
    own geometry (get_grid_positions) -- exact, and unaffected by any
    individual axes' box_aspect, since it's pure gridspec arithmetic.
    That one gap is then applied to BOTH the Correlations<->Loadings and
    Loadings<->VIFs repositioning below, so every visible gap in the row
    reads as the same width even though _INNER_WSPACE and _OUTER_WSPACE
    differ. sharey alone doesn't guarantee matching physical row
    positions when axes have different heights (box_aspect vs. none), so
    y0/height are pinned too.

    Must be called AFTER the Figure has been drawn at least once by its
    REAL target canvas (e.g. `canvas.draw()` on the FigureCanvasQTAgg
    CalibrationDiagnosticsDialog wraps this figure in) -- calling it
    before a real draw measures and equalizes the wrong renderer's gaps.
    Permanently disables the figure's layout engine afterward (it would
    otherwise re-solve and undo this on the next draw); not safe to call
    on a figure that still needs to survive being resized.

    Args:
        fig: A Figure from plot_parameter_correlation_heatmap, already
                    drawn once by its real target canvas.

    Returns:
        None -- repositions fig's axes in place.
    """
    by_label = {a.get_label(): a for a in fig.axes}
    needed = ["correlations_heatmap", "loadings_heatmap", "eigenvalues_bar", "vifs_bar"]
    missing = [k for k in needed if k not in by_label]
    if missing:
        logger.warning(
            "equalize_correlation_heatmap_gaps: expected axes %s not found on "
            "this Figure (found labels: %s) -- was this actually built by "
            "plot_parameter_correlation_heatmap? Skipping.",
            missing, list(by_label),
        )
        return

    ax = by_label["correlations_heatmap"]
    ax_load = by_label["loadings_heatmap"]
    ax_eig = by_label["eigenvalues_bar"]
    ax_vif = by_label["vifs_bar"]

    corr_pos = ax.get_position()
    load_pos = ax_load.get_position()
    eig_pos = ax_eig.get_position()
    vif_pos = ax_vif.get_position()

    # Re-assert Correlations' own position before touching box_aspect
    # below -- get_position() returned box_aspect's ADJUSTED box, which
    # apply_aspect() re-derives from the wider raw gridspec cell on every
    # draw. Without pinning this first, clearing box_aspect below would
    # snap Correlations back to that wider cell on the next draw,
    # overlapping Loadings.
    ax.set_position((corr_pos.x0, corr_pos.y0, corr_pos.width, corr_pos.height))

    # The target gap, read directly off the outer gridspec (corr_load_gs's
    # own column vs. VIFs' column) -- see this function's docstring for
    # why this is exact where a measured "clean pair" would be fragile
    # with only one box_aspect-free panel left in the row.
    vif_gridspec = ax_vif.get_gridspec()
    assert vif_gridspec is not None  # ax_vif is built from a gridspec-based subplot layout
    _, _, outer_lefts, outer_rights = vif_gridspec.get_grid_positions(fig)
    gap = outer_lefts[1] - outer_rights[0]

    # Correlations' own row-1 band -- every other panel's y0/height gets
    # pinned to this.
    row_y0, row_h = corr_pos.y0, corr_pos.height

    new_load_x0 = corr_pos.x1 + gap
    ax_load.set_position((new_load_x0, row_y0, load_pos.width, row_h))
    # Eigenvalues sits in the figure's OTHER row (dir_h, above row-1), so
    # its own y0/height (not Correlations') are kept; its width is reset
    # to Loadings' (they're meant to be pixel-identical already).
    ax_eig.set_position((new_load_x0, eig_pos.y0, load_pos.width, eig_pos.height))

    new_vif_x0 = new_load_x0 + load_pos.width + gap
    ax_vif.set_position((new_vif_x0, row_y0, vif_pos.width, row_h))

    # Clear box_aspect on every axes that had one: apply_aspect()
    # re-enforces it on EVERY draw regardless of layout engine state, by
    # shrinking/re-centering within whatever position it's given -- left
    # set, the next draw() would undo the manual positions just set
    # above. Safe to drop now that those positions bake in its result as
    # fixed numbers.
    ax.set_box_aspect(None)
    ax_load.set_box_aspect(None)
    ax_eig.set_box_aspect(None)

    # Trim the figure's own WIDTH down to the content just repositioned,
    # plus one _RIGHT_MARGIN_IN -- Correlations/Loadings' box_aspect(1)
    # squares render notably narrower than their nominal heat_w gridspec
    # column once each panel's own title eats into the row's real
    # available height (font-metric-dependent, same reason
    # equalize_correlation_heatmap_gaps exists at all -- see its own
    # docstring), so plot_parameter_correlation_heatmap's own fig_w
    # ends up a wide overestimate. Left uncorrected, that shows up as a
    # dead strip of blank figure canvas past VIFs' right edge -- fully
    # inside the Qt canvas (dialogs.py sizes the canvas to fig.dpi *
    # get_size_inches() AFTER this function runs), not a Qt/dialog
    # sizing issue, so no dialogs.py-side chrome/margin constant could
    # ever have fixed it.
    #
    # Every axes' x0/width, not just these four -- the colorbar axes
    # (corr_load_gs[0, 0]) sits at Correlations' own x0/width but was
    # never in `by_label` above, so it needs the identical rescale to
    # stay aligned once the figure narrows.
    content_right_frac = new_vif_x0 + vif_pos.width
    old_fig_w_in, old_fig_h_in = fig.get_size_inches()
    new_fig_w_in = content_right_frac * old_fig_w_in + _RIGHT_MARGIN_IN

    # Also guarantee real room at the BOTTOM for Correlations'/Loadings'
    # own x-tick labels. Correlations' are long and rotated 45 deg,
    # ha="right" -- anchored at the tick, extending down-LEFT, so a
    # long label's OWN NAME (its first characters in reading order,
    # e.g. "Wind" of "Wind direction [deg]"), not its trailing "[unit]"
    # suffix, is what sits lowest and is first to go missing. Measured
    # 4px above the canvas's own bottom edge in dev-time testing --
    # real enough of a margin there, but a real system's slightly wider
    # font metrics clips those leading characters outright. Measured
    # here the same way as _max_text_width_in (the real renderer, not a
    # character-count guess) for the same reason: this row's own
    # required height is exactly as font-metric-dependent as that
    # function's own column width was.
    renderer = fig.canvas.get_renderer()
    candidate_labels = ax.get_xticklabels() + ax_load.get_xticklabels()
    min_label_y0_px = min(
        (t.get_window_extent(renderer).y0 for t in candidate_labels),
        default=old_fig_h_in * fig.dpi,
    )
    safe_margin_px = 20.0
    extra_h_in = max(0.0, safe_margin_px - min_label_y0_px) / fig.dpi
    new_fig_h_in = old_fig_h_in + extra_h_in

    # Every axes' position, not just these four -- the colorbar axes
    # (corr_load_gs[0, 0]) sits at Correlations' own x0/width but was
    # never in `by_label` above, so it needs the identical rescale to
    # stay aligned once the figure's own size changes. Width and height
    # are independent transforms (scale_w shrinks in place; the height
    # term instead shifts every axes up by the same absolute extra_h_in
    # so the newly inserted room lands entirely below everything, not
    # split awkwardly above and below).
    scale_w = old_fig_w_in / new_fig_w_in
    for a in fig.axes:
        pos = a.get_position()
        new_y0_in = pos.y0 * old_fig_h_in + extra_h_in
        new_height_in = pos.height * old_fig_h_in
        a.set_position((
            pos.x0 * scale_w, new_y0_in / new_fig_h_in,
            pos.width * scale_w, new_height_in / new_fig_h_in,
        ))
    fig.set_size_inches(new_fig_w_in, new_fig_h_in, forward=False)

    fig.set_layout_engine(None)


# ---------------------------------------------------------------------------
# III. Identifiability directions (PCA on the pooled correlation matrix)
# ---------------------------------------------------------------------------

def _eigendecompose_correlation(
    pooled: "_PooledTrials", n_free: int, free_keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of the pooled parameters' standardized correlation
    matrix -- used by plot_parameter_correlation_heatmap's combined view.
    See that function's own docstring for the full reasoning this
    operationalizes (why correlation, not covariance; Kaiser's
    criterion; etc) -- kept there rather than duplicated here since this
    helper is pure numerics with no caller-facing meaning of its own.

    Args:
        pooled: Pooled (x, fun) from _pool_trials -- only .x is used here.
        n_free: Number of free parameters (columns in pooled.x).
        free_keys: Names for pooled.x's columns, in order -- used only to
                    name the offending parameter(s) if this raises (see
                    Raises below).

    Returns:
        (eigvals, eigvecs) in descending eigenvalue order (most-entangled
        direction first), eigvecs columns matching eigvals order.

    Raises:
        ValueError: If any pooled parameter has exactly zero variance --
                    np.corrcoef divides by each column's own std, so
                    this shows up as NaN entries, checked for below.
    """
    # np.corrcoef, not a bounds/span-normalized covariance -- see
    # plot_parameter_correlation_heatmap's docstring for why span-
    # normalization alone doesn't equalize each parameter's contribution.
    corr = np.corrcoef(pooled.x, rowvar=False)
    # np.corrcoef degenerates to a 0-d scalar (rather than a 1x1 matrix)
    # when pooled.x has a single column (n_free == 1, a legitimate, if
    # trivial, case: one free parameter trivially "correlates" with
    # itself at r=1) -- atleast_2d restores the 2D shape the rest of this
    # function (and np.fill_diagonal specifically) requires, without
    # changing anything for n_free >= 2, where corrcoef already returns a
    # proper 2D array.
    corr = np.atleast_2d(corr)
    # A zero-variance column (a free parameter that never moved across
    # the pooled points) makes np.corrcoef divide by that column's own
    # (zero) std, silently producing NaN rather than erroring -- checked
    # here, not guessed at in advance. Identified by a row being ENTIRELY
    # NaN, including its own diagonal (corrcoef gives a degenerate
    # column's self-correlation as 0/0 too, unlike the pairwise-loop
    # r_mat elsewhere in this module, which starts from np.eye and so
    # keeps a clean diagonal) -- a collateral row (paired with the
    # actual offender) only picks up exactly one NaN, not a whole row.
    if np.any(np.isnan(corr)):
        offenders = [free_keys[i] for i in np.where(np.sum(np.isnan(corr), axis=1) == n_free)[0]]
        raise ValueError(
            f"{offenders} produced NaN correlation(s) -- likely zero "
            f"variance across the {pooled.x.shape[0]} pooled points "
            f"(never moved within the near-best-fit level set). "
            f"Investigate that parameter's bounds/search space for this "
            f"run rather than trusting a Directions panel built by "
            f"silently papering over this."
        )
    # Every diagonal entry is mathematically exactly 1 (self-correlation
    # of a nonzero-variance column, guaranteed by the NaN check above) --
    # this just cleans up float rounding noise (e.g. 0.9999999999998)
    # around that known-exact value, not a data fallback.
    np.fill_diagonal(corr, 1.0)

    # eigh (not the generic eig) because a correlation matrix is symmetric
    # by construction -- guarantees real eigenvalues and orthonormal
    # eigenvectors, and is the numerically appropriate routine for a
    # symmetric input.
    # A correlation matrix is positive semi-definite by construction, so
    # every eigenvalue is mathematically >= 0. Deliberately unguarded --
    # no tolerance-clipped floor here -- this codebase doesn't guess
    # magic-number thresholds to absorb hypothetical float noise upstream
    # can't actually produce.
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]  # descending: most-entangled direction first
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


# ---------------------------------------------------------------------------
# IV. Single-pair scatter popup
# ---------------------------------------------------------------------------

def plot_single_pair_scatter(
    result: CalibrationResult,
    key_a: str,
    key_b: str,
    *,
    pooling_method: str = DEFAULT_POOLING_METHOD,
) -> Figure | None:
    """
    One parameter pair's scatter, on demand -- used by
    eidos.apps.analyzer.dialogs.CalibrationDiagnosticsDialog's click-a-
    Correlations-cell popup so a viewer can check any pair they click,
    without this module pre-selecting or pre-ranking which pairs are
    worth looking at: every pair is one click away on the Correlations
    heatmap, and this function draws whichever one the viewer actually
    asked for -- a curated/fixed pair list would be exactly the kind of
    unsolicited, opinionated pre-judgment (EIDOS/HYLE)^TT avoids.

    Pools with the SAME near-best-fit level-set restriction (pooling_method,
    the only knob left -- see plot_parameter_correlation_heatmap for why
    its own other pooling knobs are fixed constants) as the Correlations
    cell the viewer just clicked, so the points drawn here and the
    title's Pearson r below are exactly what that cell's own r was
    computed from.

    Args:
        result: A CalibrationResult.
        key_a, key_b: Free parameter keys (must both be in
                    result.free_keys) -- key_a on the x-axis, key_b on y.
        pooling_method: See plot_parameter_correlation_heatmap -- same
                    pooling, so this MUST be called with whatever value
                    the Correlations heatmap itself was built with for
                    the r in the title to match that heatmap's own cell.

    Returns:
        A matplotlib Figure, or None (with a logged warning) if either
        key isn't in result.free_keys or (pooling_method="rmse_tolerance"
        only) fewer than 2 pooled points survive feasibility/RMSE-
        tolerance filtering.

    Raises:
        ValueError: pooling_method="mahalanobis" only -- see
                    plot_parameter_correlation_heatmap's identical note.
                    In practice this should not fire here if the caller
                    passes the same pooling_method/params the Correlations
                    heatmap itself was already successfully built with
                    (that build would have raised first).
    """
    keys = result.free_keys
    if key_a not in keys or key_b not in keys:
        logger.warning(
            "plot_single_pair_scatter: %r not both in free_keys=%s -- skipping.",
            (key_a, key_b), keys,
        )
        return None
    ia, ib = keys.index(key_a), keys.index(key_b)

    pooled = _pool_for_plot(result, pooling_method=pooling_method)
    if len(pooled.fun) < 2:
        logger.warning(
            "plot_single_pair_scatter: only %d pooled point(s) after "
            "feasibility/RMSE-tolerance filtering -- skipping.", len(pooled.fun),
        )
        return None

    fig = Figure(figsize=(3.7, 3.3), layout="constrained")
    ax = fig.add_subplot(111)
    vmin, vmax = float(np.min(pooled.fun)), float(np.max(pooled.fun))
    # Draw worst (largest) RMSE first, best (smallest) last -- so the
    # near-best-fit points a viewer most cares about land on top of the
    # stack instead of being buried under whichever point happened to be
    # sampled last.
    order = np.argsort(pooled.fun)[::-1]
    sc = ax.scatter(
        pooled.x[order, ia], pooled.x[order, ib],
        c=pooled.fun[order], cmap="viridis_r", vmin=vmin, vmax=vmax,
        s=45, alpha=0.9, edgecolors="black", linewidths=0.5, marker="o",
    )
    best_marker = ax.scatter(
        [result.x_best[ia]], [result.x_best[ib]],
        marker="*", s=220, c="red", edgecolors="black", linewidths=0.8,
        zorder=5,
    )
    ax.set_xlabel(_display_label(key_a), fontsize=8)
    ax.set_ylabel(_display_label(key_b), fontsize=8)
    r, _ = stats.pearsonr(pooled.x[:, ia], pooled.x[:, ib])
    ax.set_title(f"Pearson r={r:.2f} (n={len(pooled.fun)})", fontsize=7)
    # pooled.x's wind_direction column (and result.x_best's own value below)
    # is unwrapped around x_best by _pool_trials to keep a near-seam
    # (e.g. north-wind) cluster contiguous -- see its docstring. That
    # unwrapping must stay in the actual plotted coordinates (re-wrapping
    # the DATA back into [0, 360) here would re-split the cluster across
    # the axis, exactly what _pool_trials exists to avoid), so only the
    # tick LABEL text is re-wrapped mod 360, cosmetic-only.
    if key_a == "wind_direction":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v % 360:g}"))
    if key_b == "wind_direction":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v % 360:g}"))
    ax.tick_params(labelsize=7)
    # shrink=0.5 -- a full-axes-height colorbar (matplotlib's own
    # default) reads as too long/prominent for a small popup that's
    # really just there to decode each point's color. Halves its length,
    # centered on the same vertical span it already occupied (shrink
    # centers by default).
    cbar = fig.colorbar(sc, ax=ax, label="Velocity RMSE [m/s]", shrink=0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Velocity RMSE [m/s]", fontsize=8)
    # Anchored to cbar.ax (not `ax`) and placed just below it, entirely
    # outside the scatter -- so the legend's swatch can't be confused
    # with the real star marker.
    cbar.ax.legend(
        [best_marker], ["Best fit"],
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        fontsize=7, frameon=False, handletextpad=0.4, borderaxespad=0,
    )
    # set_box_aspect(1): this popup's canvas is a fixed pixel size
    # regardless of which pair was clicked, but without box_aspect,
    # layout="constrained" solves margins per-pair based on that pair's
    # own tick-label lengths (e.g. "crr" vs "brake_lookahead"),
    # so the same window would show a different plot aspect ratio from
    # one popup to the next.
    ax.set_box_aspect(1)
    return fig


def plot_sensitivity_effect_scatter(
    key: str, x: np.ndarray, y: np.ndarray, title: str | None = None,
) -> Figure:
    """
    One free parameter's raw sampled value vs. RMSE, from Sobol's OWN
    evaluated points (calibrator.SobolSensitivityTrials.sample_x/
    sample_y -- NOT all_trials/a post-fit CalibrationResult; this is the
    PRE-Auto-Fit screening data) -- used by eidos.apps.analyzer.window's
    press-a-Sobol'-bar popup, as the S1 half of a pair: pressing either
    lane opens this figure together with plot_sensitivity_interaction_
    scatter's own ST view side by side, not one or the other depending
    on which lane was pressed (S1 and ST answer two different questions
    about the same parameter, so both are shown every time, together).

    A clean trend here (points falling near a curve) says this
    parameter's effect shows up on its own; a pure-noise scatter here
    with a clear pattern in the OTHER (interaction) view instead means
    the real story only shows up together with another parameter.

    RMSE is mapped to point color (viridis_r, worst drawn first so the
    near-best-fit points land on top) -- the same colormap/ordering
    plot_sensitivity_interaction_scatter's own S2 view uses, so the two
    figures read as one consistent pair (same color = same RMSE) rather
    than a plain-color S1 view next to a colored S2 one.

    Morris does NOT use this function -- see plot_morris_mustar_sigma_
    scatter for its own (different-shaped) popup.

    Args:
        key:   The free parameter's own name (x-axis label).
        x:     That free_key's own sampled value at every evaluated
                    point (real schema units).
        y:     RMSE at each of those same points, same order as x.
        title: Caller-supplied title text (e.g. "Sobol' S1 = 0.12±0.03",
                    plain text -- see the S1/ST notation comment in
                    eidos.apps.analyzer.window for why not mathtext) --
                    same role plot_single_pair_scatter's own "Pearson r=..."
                    title plays: the specific statistic number this
                    scatter is illustrating, stated directly on the
                    figure rather than left for the viewer to go back
                    and re-read off the bar's own tooltip. None omits
                    the title (e.g. for callers with no such number).

    Returns:
        A matplotlib Figure. Always succeeds -- x/y come straight from
        an already-computed SobolSensitivityTrials (guaranteed >= 2
        points by sample_sobol_sensitivity's own n minimum), unlike
        plot_single_pair_scatter's post-fit pooling, which can filter
        down to too few.
    """
    fig = Figure(figsize=(3.7, 3.3), layout="constrained")
    ax = fig.add_subplot(111)
    order = np.argsort(y)[::-1]
    vmin, vmax = float(np.min(y)), float(np.max(y))
    sc = ax.scatter(
        x[order], y[order], c=y[order], cmap="viridis_r", vmin=vmin, vmax=vmax,
        s=25, alpha=0.85, edgecolors="black", linewidths=0.4, marker="o",
    )
    ax.set_xlabel(_display_label(key), fontsize=8)
    ax.set_ylabel("Velocity RMSE [m/s]", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=7)
    # Fixed 0-5 m/s floor/ceiling when the run's own RMSE stays under it
    # (keeps small-RMSE runs from being drawn with a misleadingly
    # stretched axis), autoscaled from 0 otherwise.
    if float(np.max(y)) < 5.0:
        ax.set_ylim(0.0, 5.0)
    else:
        ax.set_ylim(bottom=0.0)
    cbar = fig.colorbar(sc, ax=ax, label="Velocity RMSE [m/s]", shrink=0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Velocity RMSE [m/s]", fontsize=8)
    ax.set_box_aspect(1)
    return fig


def plot_morris_mustar_sigma_scatter(
    mu_star: dict[str, float], sigma: dict[str, float],
    mu_star_conf: dict[str, float] | None = None, highlight_key: str | None = None,
) -> Figure:
    """
    The classic Morris plot: every free parameter's own (mu_star, sigma)
    -- X=mu_star, Y=sigma -- as one point each in a single 2D scatter,
    used by eidos.apps.analyzer.window's press-a-Morris-bar popup.

    Unlike plot_sensitivity_effect_scatter (one parameter's raw evaluated
    points), this shows every free parameter's own SUMMARY STATISTIC at
    once: a Morris screen is inherently a whole-run comparison, with no
    single-parameter version of it to fall back to -- pressing ANY
    Morris bar (either lane) opens this same figure; highlight_key
    (that bar's own parameter) is marked with a red star so the viewer
    can still find "their" row in the wider picture, and titled with
    that same parameter's own mu_star/sigma numbers -- same role plot_
    single_pair_scatter's own "Pearson r=..." title plays: the specific
    statistic this popup is illustrating, stated directly on the figure.

    The sigma = mu_star diagonal is the standard Morris-plot reference:
    points below it are dominated by their own mean effect (roughly
    linear/additive, mu_star > sigma); points above it have sigma
    exceeding mu_star itself -- a nonlinear or interaction-driven effect
    strong enough to outweigh the average.

    Args:
        mu_star, sigma: MorrisSensitivityTrials.mu_star/.sigma (same
                    free_keys in both).
        mu_star_conf: MorrisSensitivityTrials.mu_star_conf, for the
                    title's own ±. None (e.g. no data for highlight_key)
                    omits the ± from the title.
        highlight_key: A free parameter key to mark with a red star and
                    title (typically whichever bar the viewer pressed).
                    None draws every point the same way, no title.

    Returns:
        A matplotlib Figure.
    """
    keys = list(mu_star.keys())
    mustar_vals = [mu_star[k] for k in keys]
    sigma_vals = [sigma[k] for k in keys]

    fig = Figure(figsize=(3.9, 3.6), layout="constrained")
    ax = fig.add_subplot(111)
    data_max = max(max(mustar_vals, default=0.0), max(sigma_vals, default=0.0))
    lim = data_max * 1.1 if data_max > 0 else 1.0
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.scatter(
        mustar_vals, sigma_vals, s=40, alpha=0.85, color="#4c72b0",
        edgecolors="black", linewidths=0.5, marker="o", zorder=3,
    )
    for k, mx, sy in zip(keys, mustar_vals, sigma_vals):
        ax.annotate(_display_label(k), (mx, sy), fontsize=6, xytext=(4, 3), textcoords="offset points")
    if highlight_key is not None and highlight_key in mu_star:
        ax.scatter(
            [mu_star[highlight_key]], [sigma[highlight_key]],
            marker="*", s=220, c="red", edgecolors="black", linewidths=0.8,
            zorder=5, label=_display_label(highlight_key),
        )
        ax.legend(fontsize=7, loc="best")
        conf = mu_star_conf.get(highlight_key) if mu_star_conf else None
        mu_str = f"μ* = {mu_star[highlight_key]:.3g}±{conf:.3g}" if conf is not None \
            else f"μ* = {mu_star[highlight_key]:.3g}"
        # Same "MethodName stat = value[, stat = value]" style
        # plot_single_pair_scatter's own "Pearson r=..." title uses --
        # the point's own label (star + legend entry above) already
        # names WHICH parameter this is, so the title states only the
        # numbers, not the key again.
        ax.set_title(f"Morris {mu_str}, σ = {sigma[highlight_key]:.3g}", fontsize=7)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("μ*", fontsize=8)
    ax.set_ylabel("σ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect(1)
    return fig


def plot_sensitivity_interaction_scatter(
    key_a: str, key_b: str, x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
    title: str | None = None,
) -> Figure:
    """
    A 2D scatter of one free parameter (key_a) against another (key_b),
    RMSE mapped to point color -- used by eidos.apps.analyzer.dialogs'
    SobolS2DetailsDialog, which opens this for whichever cell (key_a,
    key_b) the viewer clicks in the S1/S2 matrix.

    Exists because a high-ST/low-S1 free_key's OWN 2D effect scatter
    (plot_sensitivity_effect_scatter) reads as pure noise: its influence
    only shows up jointly with another parameter, not on its own. Adding
    that second parameter as this plot's OWN y-axis (RMSE moved to color
    instead) turns that noise into a legible pattern -- a Sobol'
    interaction generally shows up as RMSE varying with the joint
    pattern of the two parameters' values together, which a flat scatter
    of either parameter alone cannot separate from noise.

    Drawn worst (largest) RMSE first, best (smallest) last -- same
    ordering plot_single_pair_scatter's own colored scatter uses -- so
    the near-best-fit points a viewer most cares about land on top of
    the stack instead of being buried under whichever point happened to
    be sampled last.

    Args:
        key_a, key_b: The clicked S1/S2 matrix cell's row/column
                    parameter names (x/y-axis labels).
        x_a, x_b:   key_a's/key_b's own sampled value at every evaluated
                    point (real schema units) -- SAME point order as y
                    (both come from the same SobolSensitivityTrials.
                    sample_x, so this is guaranteed by construction as
                    long as both are read from the same result).
        y:          RMSE at each of those same points.
        title:      Caller-supplied title text (e.g. "Sobol' ST =
                    0.45±0.02") -- see plot_sensitivity_effect_scatter's
                    identical parameter for the reasoning. None omits
                    the title.

    Returns:
        A matplotlib Figure.
    """
    fig = Figure(figsize=(3.7, 3.3), layout="constrained")
    ax = fig.add_subplot(111)
    order = np.argsort(y)[::-1]
    vmin, vmax = float(np.min(y)), float(np.max(y))
    sc = ax.scatter(
        x_a[order], x_b[order], c=y[order], cmap="viridis_r", vmin=vmin, vmax=vmax,
        s=25, alpha=0.85, edgecolors="black", linewidths=0.4, marker="o",
    )
    ax.set_xlabel(_display_label(key_a), fontsize=8)
    ax.set_ylabel(_display_label(key_b), fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=7)
    cbar = fig.colorbar(sc, ax=ax, label="Velocity RMSE [m/s]", shrink=0.5)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Velocity RMSE [m/s]", fontsize=8)
    ax.set_box_aspect(1)
    return fig


# ---------------------------------------------------------------------------
# V. Sobol' S1/S2 interaction matrix
# ---------------------------------------------------------------------------

def plot_sobol_s2_heatmap(result: SobolSensitivityTrials) -> Figure | None:
    """
    An NxN matrix of Sobol' SECOND-order indices (S2, result.s2) over
    every screened free parameter -- built from a finished Sobol'
    sensitivity SCREEN (calibrator.sample_sobol_sensitivity's
    SobolSensitivityTrials), not a post-fit CalibrationResult: this
    answers "which parameter PAIRS matter jointly," the question
    eidos.apps.analyzer's inline Sobol' bars only summarize down to one
    number per parameter (its own S1/ST) -- used by eidos.apps.analyzer.
    dialogs.SobolS2DetailsDialog, opened by that panel's "Check S2"
    button.

    The diagonal is N/A, not filled with anything -- S2_ij is only
    DEFINED for two DISTINCT free_keys (a real pairwise interaction
    term in the Sobol' variance decomposition); there is no "S2_ii", a
    parameter cannot pairwise-interact with itself. Putting S1
    (result.s1, the FIRST-order index) in the diagonal slot as a
    convenience would be actively wrong, not just unhelpful: S1 is a
    different-order statistic (different meaning, different typical
    magnitude), and placing it on what reads as this matrix's own
    diagonal would falsely imply it belongs to the same object -- the
    same mistake a covariance matrix would make by writing a correlation
    into its own diagonal. S1 is already shown elsewhere (the inline
    Sensitivity bar's own S1 lane); this figure stays a genuine, honest
    S2-only matrix, and each diagonal cell is drawn hatched with "N/A"
    -- visually distinct from an off-diagonal
    cell that HAS a defined S2 SALib simply didn't return a finite value
    for (see below), since those two are different kinds of absence.
    Clicking a diagonal cell is a deliberate no-op (see
    SobolS2DetailsDialog._wire_cell_popup): with no S2 value defined for
    a parameter paired with itself, there is nothing for a click there
    to be about. Only an off-diagonal cell opens that pair's own S2
    interaction scatter.

    Every finite off-diagonal cell is shown at full strength --
    deliberately UNLIKE plot_parameter_correlation_heatmap's own p<alpha
    muting: a significance-based dim-and-gray overlay on every cell of a
    matrix this size reads as visual noise obscuring the numbers, not a
    useful signal, so this omits it. result.s2_conf still exists on
    `result` for a caller that wants it; this figure just doesn't
    editorialize with it. A pair with no finite S2 at all (SALib's own
    upper-triangular NaN below/on the diagonal -- see
    SobolSensitivityTrials.s2's docstring) renders as a flat, opaque
    gray cell with no annotation -- a DEFINED quantity SALib didn't
    return a usable value for, not the same absence as the diagonal's
    hatched N/A (which has no defined quantity to be missing in the
    first place): a fake "S2=0" for either would misrepresent absent
    data as a real zero, but conflating the two kinds of absence into
    one visual would misrepresent WHY each is absent.

    Args:
        result: A finished SobolSensitivityTrials (calibrator.
                    sample_sobol_sensitivity, always run with
                    calc_second_order=True by this codebase's own caller
                    -- see that function).

    Returns:
        A matplotlib Figure, or None (with a logged warning) if fewer
        than 2 free parameters were screened -- S2 is undefined for a
        single parameter (nothing to pair it with).
    """
    keys = list(result.s1.keys())
    n = len(keys)
    if n < 2:
        logger.warning(
            "plot_sobol_s2_heatmap: only %d free parameter(s) screened -- "
            "no pairwise interaction to plot -- skipping.", n,
        )
        return None

    offdiag_mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            pair = (keys[i], keys[j]) if keys[i] < keys[j] else (keys[j], keys[i])
            val = result.s2.get(pair)
            if val is None or not np.isfinite(val):
                continue  # No finite S2 for this pair -- flat gray cell, no text (see docstring).
            offdiag_mat[i, j] = offdiag_mat[j, i] = val

    display_keys = _display_labels(keys)
    dpi = mpl.rcParams["figure.dpi"]
    # +0.9 in the original n-only formula was sized for raw free_keys
    # (short, roughly uniform length) -- pretty titles vary a lot more
    # (e.g. "mu" vs "Wind direction [deg]"), so the longest one actually
    # present is measured and padded for directly instead of guessing a
    # wider flat constant. See _max_text_width_in's own docstring for why
    # this matters here specifically (fixed-size QDialog canvas).
    label_margin_in = max(0.9, _max_text_width_in(display_keys, fontsize=8, dpi=dpi) + 0.3)
    fig_side = max(4.2, 0.42 * n)
    fig = Figure(figsize=(fig_side + label_margin_in, fig_side + label_margin_in), dpi=dpi, layout="constrained")
    ax = fig.add_subplot(111)
    # Labeled so a caller with only the Figure (SobolS2DetailsDialog's
    # click-a-cell popup) can find this axes via fig.axes without
    # depending on creation order -- same convention as
    # plot_parameter_correlation_heatmap's "correlations_heatmap" label.
    ax.set_label("sobol_s2_heatmap")
    ax.set_box_aspect(1)

    offdiag_finite = offdiag_mat[np.isfinite(offdiag_mat)]
    vmax = float(np.max(offdiag_finite)) if offdiag_finite.size and np.max(offdiag_finite) > 0 else 1.0
    im = ax.imshow(
        np.ma.masked_invalid(offdiag_mat), cmap="Oranges", vmin=0, vmax=vmax, aspect="auto",
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="#888888",
                    hatch="////", linewidth=0.5,
                ))
                ax.text(
                    j, i, "N/A", ha="center", va="center", fontsize=5.5,
                    style="italic", color="#888888",
                )
                continue
            v = offdiag_mat[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor="#808080", edgecolor="none",
                ))
                continue
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="white" if v > vmax * 0.6 else "black",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_keys, fontsize=8)
    ax.set_title("Sobol' S2 Interaction Matrix (diagonal: N/A)", fontsize=10, loc="left")
    ax.set_ylim(n - 0.5, -0.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("S2 (pairwise interaction)", fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# VI. Convenience wrapper
# ---------------------------------------------------------------------------

def plot_calibration_diagnostics(
    result: CalibrationResult,
    *,
    pooling_method: str = DEFAULT_POOLING_METHOD,
) -> dict[str, Figure]:
    """
    Build the trade-off/non-identifiability diagnostic figure(s) -- see
    plot_parameter_correlation_heatmap (Correlations + Eigenvalues +
    Loadings + VIFs, all in one figure) for what each panel shows and its
    caveats. Per-pair scatter views are not built here --
    eidos.apps.analyzer.dialogs.CalibrationDiagnosticsDialog's click-and-
    hold-a-Correlations-cell popup calls plot_single_pair_scatter
    directly, on demand, for whichever pair the viewer cares about.

    Raises rather than falling back to a lower-fidelity view if there
    isn't enough data for the correlation heatmap -- fail-fast, matching
    this codebase's general "no silent fallback" convention (see e.g.
    core.schema.bounds_from_schema's and
    core.calibrator.bounds_from_schema's "fail fast rather than
    silently falling back to an unbounded range/search").

    Args:
        result: A CalibrationResult from calibrator.calibrate().
        pooling_method: See plot_parameter_correlation_heatmap. Every
                    other knob it takes is a fixed constant there -- no
                    caller of this function has ever varied any of them,
                    so none are threaded through here either.

    Returns:
        dict with exactly one key, "correlation_heatmap".

    Raises:
        ValueError: If result.all_trials has too few usable points to
                    build the correlation heatmap -- see
                    plot_parameter_correlation_heatmap (fewer than 4
                    pooled points after feasibility/RMSE-tolerance
                    filtering, or 0 free parameters). The root cause is
                    always that the near-best-fit filtering itself isn't
                    usable for this run (see the message for specifics)
                    -- there is no lower-fidelity view to fall back to.
                    pooling_method="mahalanobis" only: also see
                    plot_parameter_correlation_heatmap's identical note.

    Example:
        >>> figs = plot_calibration_diagnostics(result)
        >>> figs["correlation_heatmap"].savefig("correlation_heatmap.png")
    """
    heatmap = plot_parameter_correlation_heatmap(result, pooling_method=pooling_method)
    if heatmap is None:
        raise ValueError(
            f"plot_calibration_diagnostics: too few usable trials to build "
            f"the correlation heatmap ({len(result.all_trials)} trial(s) in "
            f"result.all_trials -- see the warning(s) logged by "
            f"plot_parameter_correlation_heatmap above for the exact reason)."
        )

    return {"correlation_heatmap": heatmap}


if __name__ == "__main__":
    # Minimal, self-contained demo against a synthetic CalibrationResult
    # (no real FIT activity / course / calibrate() run needed) -- built to
    # show a genuine cda/air_density trade-off so the correlation heatmap
    # and scatter grid have something real to display. Requires the same
    # import chain as calibrator.py itself (core.schema etc.) since
    # bounds_from_schema reads live Pydantic Field constraints.
    #
    # Every trial here is independent and directly comparable, matching
    # how CalibrationResult.all_trials is actually populated: n_total
    # flat DE+NM-polished trials, no outer/sub-seed grouping.
    import types

    rng = np.random.default_rng(0)
    free_keys = ["cda", "air_density", "crr", "wind_speed"]
    n_total = 180  # matches calculate_num_trials(4, 2, SEED_MULTIPLIER)

    all_trials = []
    for _ in range(n_total):
        # A synthetic degenerate ridge: cda and air_density trade off
        # (their product held roughly constant) while crr/wind_speed
        # vary independently -- mimics the CdA x air_density x wind
        # confound this module's docstring discusses. Every point is
        # drawn fully independently (no shared per-trial "true center")
        # to match how real calibrate() trials actually behave --
        # anchoring points to a shared center per group would be a
        # pseudoreplication trap.
        cda = rng.uniform(0.22, 0.32)
        air_density = 0.075 / cda + rng.normal(0, 0.03)  # keeps cda*air_density ~ 0.075
        x = np.array([
            cda,
            air_density,
            rng.uniform(0.003, 0.006),
            rng.uniform(-2.0, 2.0),
        ])
        fun = 0.3 + abs(rng.normal(0, 0.05))
        all_trials.append(types.SimpleNamespace(x=x, fun=fun, success=True, de_success=True))

    x_matrix = np.array([t.x for t in all_trials])
    best = min(all_trials, key=lambda t: t.fun)

    result = CalibrationResult(
        free_keys=free_keys,
        x_best=np.asarray(best.x),
        rmse_mps=float(best.fun),
        physics_overrides={},
        n_trials=len(all_trials),
        n_converged=len(all_trials),
        x_std={k: float(np.std(x_matrix[:, i])) for i, k in enumerate(free_keys)},
        all_trials=all_trials,
    )

    figs = plot_calibration_diagnostics(result)
    for name, fig in figs.items():
        out_path = f"/tmp/calibration_diagnostics_demo_{name}.png"
        fig.savefig(out_path, dpi=110)
        print(f"wrote {out_path}")
