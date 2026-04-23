# Frozen-axis dropdown for 3D sweep heatmaps

**Date:** 2026-04-23

## Problem

In the 3D sweep "Frozen Heat" / "Frozen Ctr" views, the third metric is always
the frozen axis (axis index 2). The slider sits at the bottom of the page and
is labelled with the static text **"Frozen axis"** — even though there is a
callback that overwrites the label with the metric name, in practice the user
still perceives it as a fixed role. There is no way to choose *which* of the
three metrics is the frozen one without rearranging the metric dropdowns on
the left.

## Goal

Replace the static label with a small dropdown that

1. shows the **name** of the metric currently frozen,
2. lets the user pick **any** of the three sweep axes to freeze, and
3. **opens upward** so the menu does not run off the bottom of the viewport.

The user's choice is remembered for the rest of the session (across re-runs of
the same sweep) as long as the metric set is unchanged.

## Approach

### UI

* Remove `html.Span#frozen-slider-label`.
* Add `dcc.Dropdown#frozen-axis-dropdown` in its place. Width capped (~160 px),
  custom CSS class `dse-dropdown-up` so the menu uses `bottom: 100%; top: auto;`.
* Options are derived from `sweep_data["metric_keys"]` → metric labels via
  `METRIC_BY_KEY`.

### State

* New `dcc.Store#frozen-axis-store` holds the chosen axis index (default `2`).
* The dropdown writes to this store; the store feeds back into the figure-build
  pipeline.

### Permutation strategy

Add a helper `permute_sweep_for_frozen(sweep_data, frozen_idx)` in
`gui/interpolation.py`. It returns a *new* sweep-data-shaped dict with `xs`,
`ys`, `zs`, `grid`, and `metric_keys` rearranged so that `frozen_idx` ends up at
position 2 and the other two axes keep their relative order at positions 0/1.

Downstream code (`build_figure`, `sweep_to_interp_grid`, `frozen_slice`, the
clientside `qusimInterp.frozenSlice` JS) is **unchanged** — they continue to
assume axis 2 is frozen.

### Wiring

* The main sweep callback already produces the figure, the interp store, and
  the slider min/max. Extend it to take `frozen-axis-store` as `State` and
  apply the permutation before `sweep_to_interp_grid` / `build_figure`.
* A new lightweight callback fires on `frozen-axis-dropdown.value` →
  writes `frozen-axis-store`. To re-render after a dropdown change without
  re-running the sweep, route through the existing `sweep-trigger` mechanism
  (the same one used after parameter edits) so we re-enter the main callback
  with the cached `sweep-result-store` data.
* When a sweep returns a *new* set of metric keys, validate the stored index:
  if it is in range and the metric at that index is still the same, keep it;
  otherwise reset to `2`.

### CSS

`gui/assets/styles.css` (or wherever existing dropdown styles live) gets:

```css
.dse-dropdown-up .Select-menu-outer,
.dse-dropdown-up .Select__menu {
    top: auto !important;
    bottom: 100% !important;
    margin-bottom: 4px;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.12);
}
```

## Trade-offs

* **Permute-and-forget vs. parameterise downstream.** Permutation keeps the
  change blast-radius small but pays an O(n³) copy of the grid each rebuild.
  Acceptable: cap is ~5 k points and we already pass over the grid in
  `sweep_to_interp_grid`.
* **CSS flip vs. custom dropdown component.** A scoped CSS rule is one block
  and avoids a new component. Risk: dcc.Dropdown's internals change. Mitigated
  by the `dse-dropdown-up` scope.
* **Persistence policy.** Remember within session as long as the metric set is
  unchanged. Surprises if a sweep replaces the metrics — handled by the
  validation check above.

## Out of scope

* N-D sweeps (>3 axes) with multiple frozen sliders. Today's `frozen-slider*`
  IDs are singular; the existing N-D plumbing is unused for the heatmap views.
  This change keeps the same singular shape.
