/**
 * Sweep progress overlay — polls /api/progress and renders a bar + parameters.
 *
 * The overlay appears on top of the plot when a sweep is running.
 * It shows: progress bar, percentage, completed/total, and current parameters
 * arranged in columns (slowest-changing on the left, fastest on the right).
 */
(function () {
    "use strict";

    var POLL_MS = 150;
    var DELAY_SHOW_MS = 1000; // only show overlay if sweep takes >1s
    var _timer = null;
    var _lastRunning = false;
    var _runningStartedAt = null; // timestamp when running first detected

    function formatValue(v) {
        if (typeof v !== "number") return String(v);
        if (v === 0) return "0";
        var abs = Math.abs(v);
        if (abs >= 1000 || abs < 0.01) {
            return v.toExponential(2);
        }
        // Up to 4 significant digits
        return parseFloat(v.toPrecision(4)).toString();
    }

    function renderOverlay(data) {
        var overlay = document.getElementById("sweep-progress-overlay");
        if (!overlay) return;

        if (!data.running) {
            overlay.style.display = "none";
            _runningStartedAt = null;
            return;
        }

        // Track when the sweep started; skip rendering until delay elapses
        var now = Date.now();
        if (_runningStartedAt === null) {
            _runningStartedAt = now;
        }
        if (now - _runningStartedAt < DELAY_SHOW_MS) {
            return;
        }

        var pct = data.percentage || 0;
        var completed = data.completed || 0;
        var total = data.total || 0;
        var params = data.current_params || {};
        var coldDone = data.cold_completed || 0;
        var coldTotal = data.cold_total || 0;

        // Build parameter columns HTML
        var paramKeys = Object.keys(params);
        var paramHtml = "";
        if (paramKeys.length > 0) {
            paramHtml = '<div style="display:flex;gap:24px;justify-content:center;margin-top:8px">';
            for (var i = 0; i < paramKeys.length; i++) {
                var k = paramKeys[i];
                var v = params[k];
                paramHtml +=
                    '<div style="text-align:center;min-width:80px">' +
                    '<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px">' +
                    k +
                    "</div>" +
                    '<div style="font-size:13px;font-weight:600;color:#2B2B2B;font-family:\'SF Mono\',\'JetBrains Mono\',monospace">' +
                    formatValue(v) +
                    "</div>" +
                    "</div>";
            }
            paramHtml += "</div>";
        }

        // Cold-compilation counter (only shown when the sweep has cold groups).
        var coldHtml = "";
        if (coldTotal > 0) {
            coldHtml =
                '<div style="text-align:center;margin-top:10px;font-size:11px;color:#888">' +
                'Cold compilations: ' +
                '<span style="font-weight:600;color:#2B2B2B;font-family:\'SF Mono\',\'JetBrains Mono\',monospace">' +
                coldDone + ' / ' + coldTotal +
                '</span>' +
                '</div>';
        }

        overlay.style.display = "flex";
        overlay.style.position = "absolute";
        overlay.style.inset = "0";
        overlay.style.background = "rgba(255,255,255,0.88)";
        overlay.style.backdropFilter = "blur(2px)";
        overlay.style.zIndex = "50";
        overlay.style.flexDirection = "column";
        overlay.style.alignItems = "center";
        overlay.style.justifyContent = "center";
        overlay.style.transition = "opacity 0.15s";

        overlay.innerHTML =
            '<div style="width:320px;max-width:90%">' +
            // Percentage label
            '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">' +
            '<span style="font-size:13px;font-weight:600;color:#2B2B2B">' +
            Math.round(pct) +
            "%</span>" +
            '<span style="font-size:11px;color:#888">' +
            completed +
            " / " +
            total +
            "</span>" +
            "</div>" +
            // Progress bar track
            '<div style="width:100%;height:6px;background:#E5E5E5;border-radius:3px;overflow:hidden">' +
            '<div style="width:' +
            pct +
            "%;height:100%;background:#2B2B2B;border-radius:3px;transition:width 0.1s linear" +
            '"></div>' +
            "</div>" +
            // Parameters
            paramHtml +
            // Cold compilation counter
            coldHtml +
            "</div>";
    }

    function poll() {
        fetch("/api/progress")
            .then(function (r) {
                return r.json();
            })
            .then(function (data) {
                renderOverlay(data);
                _lastRunning = data.running;
            })
            .catch(function () {
                // Server busy (sweep in progress), keep polling
            });
    }

    // Start polling when document is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }

    function start() {
        _timer = setInterval(poll, POLL_MS);
    }
})();
