/**
 * Plot help icon — anchors a "?" button just to the left of the Plotly
 * modebar of #main-plot and shows a popup explaining the active view.
 *
 * The button is NOT inserted into the modebar (Plotly's modebar groups are
 * floated and adding a button into the camera's group can push the camera
 * onto a new floated row). Instead it lives as a sibling of `.modebar`
 * inside `_modebardiv`, absolutely positioned with `right` recomputed as the
 * modebar resizes — so the icon always sits 4 px to the left of the modebar's
 * leftmost button (the camera) regardless of how many buttons Plotly renders
 * for the active view.
 *
 * Help text is keyed by view-type-store value; Merit further keys by
 * merit-mode-store ("merit:<mode>"). The active key is pushed in from a Dash
 * clientside callback via window.qusimUpdatePlotHelp. Each entry has a
 * `body` (what the plot shows) and a `useFor` (questions it answers / metrics
 * it makes easy to compare).
 */
(function () {
    "use strict";

    var BTN_ID = "modebar-help-icon";
    var TIP_ID = "modebar-help-tip";
    var PLOT_ID = "main-plot";

    var PLOT_HELP = {
        "line": {
            title: "Line",
            body: "Output metric vs the swept axis. Iso-level lines mark where the metric crosses each threshold.",
            useFor: "What value of the swept parameter hits a target performance? At what point does the metric saturate or fall off a cliff?"
        },
        "heatmap": {
            title: "Heatmap",
            body: "Output metric on a 2D color grid over the two swept axes. Reads absolute values at any (X, Y); enable Iso-levels to overlay threshold contours.",
            useFor: "Which (X, Y) corner gives the best metric? Where is the basin of acceptable performance, and how steeply does it drop off the edges?"
        },
        "contour": {
            title: "Contour",
            body: "Smoothed output metric drawn as filled iso-bands over two swept axes.",
            useFor: "Where does the metric cross a threshold (the contour)? Which direction in (X, Y) gives the steepest improvement? Compare gradient magnitudes across the plane."
        },
        "scatter3d": {
            title: "3D Scatter",
            body: "Every swept point plotted in (X, Y, Z) and colored by the output metric.",
            useFor: "Inspect the raw point cloud before fitting an Isosurface. Spot outliers, gaps in the sweep grid, or clusters where one axis dominates."
        },
        "isosurface": {
            title: "Isosurface",
            body: "Nested 3D surfaces drawn at each iso-level threshold.",
            useFor: "What 3D parameter region keeps the metric above a target? How does the volume of \"good\" parameter space shrink as you raise the threshold?"
        },
        "frozen_heatmap": {
            title: "Frozen Heatmap",
            body: "2D heatmap on the X–Y axes with the third axis pinned to the slider value below.",
            useFor: "How does the (X, Y) landscape change as the third parameter sweeps? Useful for finding the slice of Z where the (X, Y) basin is widest."
        },
        "frozen_contour": {
            title: "Frozen Contour",
            body: "Filled iso-bands on the X–Y axes with the third axis pinned via the slider.",
            useFor: "Same Z-slicing as Frozen Heat but emphasises threshold contours — find the Z value at which a target iso-line first appears or disappears."
        },
        "parallel": {
            title: "Parallel Coordinates",
            body: "One polyline per swept point, hitting every axis at its parameter value. Drag along axes to filter; lines are colored by the output metric.",
            useFor: "Which combinations of all axes satisfy multiple constraints at once? Spot redundant axes (parallel bundles) and trade-offs (crossing bundles)."
        },
        "slices": {
            title: "1D Slices",
            body: "Output metric along each axis individually, with all other axes fixed to the median.",
            useFor: "Compare how sensitive the metric is to each axis in isolation. Decide which axis to tighten the sweep on next, and which is safe to ignore."
        },
        "importance": {
            title: "Feature Importance",
            body: "Variance-based ranking of how much each axis drives the output metric.",
            useFor: "Which knobs matter most? Use this to rank parameters before a finer sweep, or to justify dropping low-impact axes from a follow-up run."
        },
        "pareto": {
            title: "Pareto Front",
            body: "Two-objective trade-off plot. Highlighted points are non-dominated; gray points are dominated. Pick X/Y from the dropdowns above.",
            useFor: "Comparing two metrics that pull in opposite directions (e.g. fidelity vs runtime) — what's the cheapest point that still meets a fidelity floor?"
        },
        "correlation": {
            title: "Correlation Matrix",
            body: "Pairwise Pearson correlations between sweep axes and output metrics.",
            useFor: "Which axes are redundant (|r|≈1)? Which axes drive which metric most strongly? Spot positive vs negative couplings before designing a follow-up sweep."
        },
        "merit:heatmap": {
            title: "Merit · Heatmap",
            body: "2D heatmap of the figure-of-merit defined in the formula editor below the plot.",
            useFor: "How does a weighted combination of fidelity / runtime / depth / shots / etc. behave across two axes? Tune the FOM formula and watch the basin shift."
        },
        "merit:3d": {
            title: "Merit · 3D",
            body: "3D surface of the figure-of-merit over two swept axes.",
            useFor: "Same composite as the Heatmap but with a tilted view that emphasises landscape shape — easier to spot ridges, cliffs, and saddle points than in flat color."
        },
        "merit:pareto": {
            title: "Merit · Pareto",
            body: "Pareto front for the merit components. The colour-by dropdown highlights points by a third metric.",
            useFor: "Balance competing metrics in the FOM directly: see which points dominate on the chosen X/Y, then colour by a third metric to break ties."
        },
        "topology": {
            title: "Topology",
            body: "Interactive graph of the multi-core qubit architecture. Nodes are qubits, edges are couplings. Without a sweep loaded, comm qubits are red and data qubits are blue. With a sweep, nodes color by per-qubit fidelity (see the legend at the bottom right).",
            useFor: "Sanity-check the architecture: where are the comm bottlenecks, how are the cores connected, which qubits limit overall fidelity? Use the slider strip on the left to step through the sweep grid; the Re-layout button refreshes the force-directed positions."
        },
        "__default__": {
            title: "Plot",
            body: "Run a sweep with at least one axis to see results.",
            useFor: "The View tabs above switch between line / heatmap / contour / 3D / analysis renderings of the same data — pick one and the help text here explains what it answers."
        }
    };

    var ICON_SVG =
        '<svg viewBox="0 0 24 24" height="1em" width="1em" class="icon" aria-hidden="true">' +
        '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>' +
        '</svg>';

    var current = PLOT_HELP["__default__"];
    var resizeObserver = null;

    function helpFor(viewType, meritMode) {
        if (viewType === "merit") {
            var key = "merit:" + (meritMode || "heatmap");
            return PLOT_HELP[key] || PLOT_HELP["__default__"];
        }
        return PLOT_HELP[viewType] || PLOT_HELP["__default__"];
    }

    function ensureTip() {
        var tip = document.getElementById(TIP_ID);
        if (tip) return tip;
        tip = document.createElement("div");
        tip.id = TIP_ID;
        tip.className = "modebar-help-tip";
        tip.style.display = "none";
        tip.innerHTML =
            '<div class="modebar-help-tip__title"></div>' +
            '<div class="modebar-help-tip__body"></div>' +
            '<div class="modebar-help-tip__sep"></div>' +
            '<div class="modebar-help-tip__use-label">Use it for</div>' +
            '<div class="modebar-help-tip__use"></div>';
        document.body.appendChild(tip);
        return tip;
    }

    function renderTip(tip) {
        tip.querySelector(".modebar-help-tip__title").textContent = current.title;
        tip.querySelector(".modebar-help-tip__body").textContent = current.body;
        tip.querySelector(".modebar-help-tip__use").textContent = current.useFor || "";
    }

    function positionTip(tip, btn) {
        var br = btn.getBoundingClientRect();
        var top = br.bottom + 8;
        var maxLeft = window.innerWidth - tip.offsetWidth - 8;
        var left = Math.min(br.right - tip.offsetWidth, maxLeft);
        if (left < 8) left = 8;
        tip.style.left = left + "px";
        tip.style.top = top + "px";
    }

    function showTip(btn) {
        var tip = ensureTip();
        renderTip(tip);
        tip.style.display = "block";
        positionTip(tip, btn);
    }

    function hideTip() {
        var tip = document.getElementById(TIP_ID);
        if (tip) tip.style.display = "none";
    }

    function repositionButton(btn, modebar) {
        // _modebardiv (modebar's parent) is full-width with position:relative.
        // .modebar is position:absolute; right:2px. Anchor our button at
        // right:(modebar.width + 4)px so it sits 4px left of the camera.
        var width = modebar.offsetWidth || 0;
        btn.style.right = (width + 4) + "px";
    }

    function bindTrigger(el, helpKey) {
        if (!el || el.dataset.qusimHelpBound === "1") return;
        el.dataset.qusimHelpBound = "1";
        // Strip the browser-native tooltip — we render our own popup on
        // hover and showing both at once is noisy.
        if (el.hasAttribute("title")) el.removeAttribute("title");
        var lookup = function () {
            // For static keys (e.g. "topology"), show that entry directly.
            // For the modebar icon we follow `current`, which the Dash
            // clientside callback keeps in sync with view-type-store.
            return helpKey ? (PLOT_HELP[helpKey] || PLOT_HELP["__default__"]) : current;
        };
        el.addEventListener("mouseenter", function () {
            current = lookup();
            showTip(el);
        });
        el.addEventListener("mouseleave", hideTip);
        el.addEventListener("click", function (e) {
            e.preventDefault();
            e.stopPropagation();
            var tip = document.getElementById(TIP_ID);
            if (tip && tip.style.display === "block") {
                hideTip();
                return;
            }
            current = lookup();
            showTip(el);
        });
    }

    function injectHelp() {
        var plot = document.getElementById(PLOT_ID);
        if (!plot) return;
        if (plot.querySelector("#" + BTN_ID)) return;
        var modebar = plot.querySelector(".modebar");
        if (!modebar) return;
        var paperdiv = modebar.parentElement;  // Plotly's _modebardiv
        if (!paperdiv) return;

        var btn = document.createElement("a");
        btn.id = BTN_ID;
        btn.className = "modebar-btn modebar-help-btn";
        btn.setAttribute("rel", "tooltip");
        btn.innerHTML = ICON_SVG;
        paperdiv.appendChild(btn);
        bindTrigger(btn, null);
        repositionButton(btn, modebar);

        if (resizeObserver) {
            try { resizeObserver.disconnect(); } catch (_) { /* ignore */ }
        }
        if (typeof ResizeObserver !== "undefined") {
            resizeObserver = new ResizeObserver(function () {
                repositionButton(btn, modebar);
            });
            resizeObserver.observe(modebar);
        }
    }

    function bindStaticTriggers() {
        bindTrigger(document.getElementById("topology-help-icon"), "topology");
    }

    window.qusimUpdatePlotHelp = function (viewType, meritMode) {
        current = helpFor(viewType, meritMode);
        var tip = document.getElementById(TIP_ID);
        if (tip && tip.style.display === "block") renderTip(tip);
    };

    function attach() {
        injectHelp();
        bindStaticTriggers();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attach);
    } else {
        attach();
    }

    // Plotly tears down and rebuilds the modebar on every figure update — the
    // observer re-injects the button. injectHelp is idempotent (early-return
    // if the button already exists) so the firehose of body mutations is
    // cheap. Each successful re-injection also re-binds the ResizeObserver.
    var mo = new MutationObserver(function () {
        injectHelp();
        bindStaticTriggers();
    });
    mo.observe(document.body, { childList: true, subtree: true });
})();
