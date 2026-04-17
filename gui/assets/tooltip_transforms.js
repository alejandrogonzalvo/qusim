/**
 * Dash dcc.Slider tooltip transform functions.
 *
 * Each function receives the raw slider value and returns the
 * formatted string shown in the tooltip bubble.
 *
 * Register under window.dccFunctions so Dash can find them via
 * tooltip={"transform": "<functionName>"}.
 */
window.dccFunctions = window.dccFunctions || {};

/* ── helpers ─────────────────────────────────────────────────────── */

function _fmtNs(ns) {
    var abs = Math.abs(ns);
    if (abs === 0)   return "0 ns";
    if (abs < 1e3)   return +ns.toPrecision(3) + " ns";
    if (abs < 1e6)   return +(ns / 1e3).toPrecision(3) + " \u00B5s";
    if (abs < 1e9)   return +(ns / 1e6).toPrecision(3) + " ms";
    return +(ns / 1e9).toPrecision(3) + " s";
}

function _fmtHz(hz) {
    var abs = Math.abs(hz);
    if (abs === 0)   return "0 Hz";
    if (abs < 1e3)   return +hz.toPrecision(3) + " Hz";
    if (abs < 1e6)   return +(hz / 1e3).toPrecision(3) + " kHz";
    if (abs < 1e9)   return +(hz / 1e6).toPrecision(3) + " MHz";
    return +(hz / 1e9).toPrecision(3) + " GHz";
}

function _fmtRate(v) {
    if (v === 0) return "0";
    if (Math.abs(v) >= 0.01 && Math.abs(v) < 100) return +v.toPrecision(3) + "";
    return v.toExponential(2);
}

/* ── log-scale transforms (slider value = exponent) ──────────────── */

window.dccFunctions.logNs = function (value) {
    return _fmtNs(Math.pow(10, value));
};

window.dccFunctions.logHz = function (value) {
    return _fmtHz(Math.pow(10, value));
};

window.dccFunctions.logRate = function (value) {
    return _fmtRate(Math.pow(10, value));
};

/* ── linear-scale transforms ─────────────────────────────────────── */

window.dccFunctions.linearWires = function (value) {
    return Math.round(value) + " wires";
};

window.dccFunctions.linearCycles = function (value) {
    return Math.round(value) + " cycles";
};

window.dccFunctions.linearFraction = function (value) {
    return value.toFixed(2);
};

window.dccFunctions.linearQubits = function (value) {
    return Math.round(value) + " qubits";
};

window.dccFunctions.linearCores = function (value) {
    return Math.round(value) + " cores";
};
