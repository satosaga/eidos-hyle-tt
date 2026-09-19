# Analyzer: Auto Fit

Auto Fit is an optional feature of the Analyzer's Rebuild-and-compare
workbench (see the README's Analyze step): an automated search over
selected physics parameters, plus two panels to help judge which
parameters are worth fitting and how well-identified the result is.

## What Auto Fit does

Check one or more parameters' **Fit** boxes in the Physics panel
and press **Add Rebuild**: those checked parameters are searched
automatically so the Rebuild's simulated velocity best matches the
loaded Activity's recorded velocity, while every other (unchecked)
parameter stays fixed at its current spinbox value. The result is added
to the Rebuild list like any other Rebuild.

With no boxes checked, **Add Rebuild** instead does a plain manual
rebuild: every parameter is used exactly as its spinbox currently
shows, no search involved.

## Sensitivity: which parameters are worth fitting at all (before a Rebuild)

Before running any Rebuild, the **Sensitivity** column next to the
**Fit** checkboxes always shows a live sensitivity screen for whichever
parameters are currently checked — there's no separate button to
launch it. Checking/unchecking a box, editing a spinbox value, or
selecting an existing Rebuild all trigger an automatic recompute
(briefly debounced against a burst of changes).

Pick the method (Sobol' or Morris) with the radio buttons in the column
header, and its sample size (Sobol's `N`, Morris's `r`) in the field
next to it — press Enter there, or switch method, to recompute
immediately instead of waiting for the debounce.

Each checked parameter's row shows a two-lane bar: Sobol' shows S1
(top) and ST (bottom); Morris shows mu_star (top) and sigma (bottom).
Click-and-hold any bar to pop up its underlying scatter plot; release
to close it.

For Sobol' specifically, **Check S2** (next to the header) opens a
separate window with the full S2 interaction matrix for the currently
checked parameters. Click-and-hold any off-diagonal cell there to pop
up that pair's own interaction scatter; release to close it.

## Diagnostics: how identifiable were the fitted parameters (after a Rebuild)

Highlight a Rebuild that was made by Auto Fit (not a manual one) in the
Rebuild list, then press **Diagnose Fit** to open the "Auto Fit
Diagnostics" window. It pools every trial from that Auto Fit search
whose fit came close enough to the best one found ("near-best-fit")
and looks for trade-offs between parameters across that pool, not just
the single winning combination. Four panels, row-aligned by that
Rebuild's free parameters (Eigenvalues indexed by direction instead):

1. **Correlations** — pairwise Pearson r heatmap; cells not even
   uncorrected-p<0.05 significant are grayed out.
2. **Eigenvalues + Loadings** — rows with eigenvalue >= 1 are
   highlighted (dashed line, dark green bar); each has a Loadings column
   below it showing which parameters have `|loading| >= 0.3`
   (annotated) and their sign. A loading's sign is arbitrary per run —
   don't compare signs between two different windows.
3. **VIFs** — one bar per parameter, log-scaled, fixed at [1, 1000].
   Light green < 5, medium 5-10, dark >= 10.
4. **Click a Correlations cell** (including the diagonal) to pop up
   that pair's scatter, colored by RMSE, with `x_best` marked on the
   plot and Pearson `r` in the title.

The **Pooling:** combo switches which trials count as "near-best":
`rmse_tolerance` (default, an RMSE-space margin) or `mahalanobis` (a
robust fit in parameter space) — useful when similar-RMSE trials have
visibly different parameter values.
