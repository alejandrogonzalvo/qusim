/**
 * Vertical resize handle for the right-sidebar split between the
 * Circuit/Noise/Thresholds tabs (upper) and the Performance panel (lower).
 *
 * The handle element (#right-panel-divider) is rendered once; the rest is
 * pure DOM — no Dash callbacks — so dragging is immediate and doesn't
 * round-trip to Python.
 */
(function () {
    "use strict";

    var DIVIDER_ID = "right-panel-divider";
    var UPPER_ID = "fixed-config-container";
    var LOWER_ID = "performance-container";
    var STORAGE_KEY = "quadris.rightPanelSplit";
    var MIN_PX = 60;

    function getNodes() {
        return {
            divider: document.getElementById(DIVIDER_ID),
            upper: document.getElementById(UPPER_ID),
            lower: document.getElementById(LOWER_ID),
        };
    }

    function applySplit(upperPx, nodes) {
        var parent = nodes.upper.parentElement;
        if (!parent) return;
        var total = parent.clientHeight;
        var dividerH = nodes.divider.offsetHeight || 6;
        var available = total - dividerH;
        if (available < MIN_PX * 2) return;
        var upper = Math.max(MIN_PX, Math.min(available - MIN_PX, upperPx));
        var lower = available - upper;
        nodes.upper.style.flex = "0 0 " + upper + "px";
        nodes.lower.style.flex = "0 0 " + lower + "px";
    }

    function restoreSplit(nodes) {
        var saved = parseFloat(localStorage.getItem(STORAGE_KEY));
        if (!isNaN(saved) && saved > 0) {
            applySplit(saved, nodes);
        }
    }

    function attach() {
        var nodes = getNodes();
        if (!nodes.divider || !nodes.upper || !nodes.lower) return false;
        if (nodes.divider.dataset.resizeBound === "1") return true;
        nodes.divider.dataset.resizeBound = "1";

        restoreSplit(nodes);

        nodes.divider.addEventListener("mousedown", function (ev) {
            ev.preventDefault();
            var parent = nodes.upper.parentElement;
            var startY = ev.clientY;
            var startUpper = nodes.upper.getBoundingClientRect().height;
            var prevCursor = document.body.style.cursor;
            var prevSelect = document.body.style.userSelect;
            document.body.style.cursor = "row-resize";
            document.body.style.userSelect = "none";

            function onMove(e) {
                var dy = e.clientY - startY;
                var next = startUpper + dy;
                applySplit(next, nodes);
            }
            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                document.body.style.cursor = prevCursor;
                document.body.style.userSelect = prevSelect;
                try {
                    var finalUpper = nodes.upper.getBoundingClientRect().height;
                    localStorage.setItem(STORAGE_KEY, String(finalUpper));
                } catch (_) { /* ignore quota */ }
            }
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
        });
        return true;
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attach);
    } else {
        attach();
    }

    // The divider is present at page load, but Dash can re-render the sidebar
    // if callbacks swap children — re-bind on mutations.
    var mo = new MutationObserver(function () { attach(); });
    mo.observe(document.body, { childList: true, subtree: true });
})();
