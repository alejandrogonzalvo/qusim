/**
 * Client-side interpolation for the frozen-slider pattern.
 *
 * Mirrors the Python implementations in gui/interpolation.py.
 * Runs entirely in the browser — no server round-trip on slider drag.
 */

window.qusimInterp = (function () {
  "use strict";

  function bisect(xs, x) {
    var n = xs.length;
    if (n < 2) return 0;
    if (x <= xs[0]) return 0;
    if (x >= xs[n - 1]) return n - 2;
    var lo = 0, hi = n - 1;
    while (lo < hi - 1) {
      var mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  function clamp01(t) {
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  function frozenSlice(grid3d, xs, ys, zs, zValue) {
    var nz = zs.length, ny = ys.length, nx = xs.length;
    var k = bisect(zs, zValue);
    var dz = zs[k + 1] - zs[k];
    var fz = dz !== 0 ? clamp01((zValue - zs[k]) / dz) : 0;

    var result = new Array(ny);
    for (var j = 0; j < ny; j++) {
      result[j] = new Array(nx);
      for (var i = 0; i < nx; i++) {
        var v0 = grid3d[k][j][i];
        var v1 = grid3d[k + 1][j][i];
        result[j][i] = v0 * (1 - fz) + v1 * fz;
      }
    }
    return result;
  }

  function buildHeatmapUpdate(slice2d, xs, ys, isLogX, isLogY) {
    var xPlot = isLogX ? xs.map(Math.log10) : xs;
    var yPlot = isLogY ? ys.map(Math.log10) : ys;
    return {
      x: [xPlot],
      y: [yPlot],
      z: [slice2d],
    };
  }

  return {
    bisect: bisect,
    frozenSlice: frozenSlice,
    buildHeatmapUpdate: buildHeatmapUpdate,
  };
})();
