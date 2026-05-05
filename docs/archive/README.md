# `docs/archive/`

Implementation handoffs / view-spec plans whose work has shipped.
Kept here for historical context — file paths, identifiers, and
"current state" sections in these documents are point-in-time and
should NOT be used as a guide to the present codebase.

For the current architecture, see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) and the per-package
READMEs:

- [`../../python/qusim/dse/README.md`](../../python/qusim/dse/README.md)
- [`../../python/qusim/analysis/README.md`](../../python/qusim/analysis/README.md)
- [`../../gui/README.md`](../../gui/README.md)

## Contents

| File | Originally | Status |
|---|---|---|
| `2026-04-handoff_progress_bar.md` | `HANDOFF.md` at repo root — agent handoff for adding a sweep progress bar | Implemented (`SweepProgress` + `dcc.Store` polling) |
| `2026-04-dse_view_implementation_plan.md` | `DSE_VIEW_IMPLEMENTATION_PLAN.md` at repo root — full GUI view specification | Implemented; live catalogue lives in [`../DSE_VIEWS.md`](../DSE_VIEWS.md) |
| `2026-04-21-handoff_qualitative_variables.md` | `docs/handoff_qualitative_variables.md` — categorical sweep axis support | Implemented (see `CATEGORICAL_METRICS` in `qusim.dse.axes`) |

The full implementation plans for each completed milestone (the
`docs/plans/2026-*` series) are kept in [`../plans/`](../plans/).
