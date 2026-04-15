"""Tests for hot-reload view preservation and initial load trigger bugs."""

import pytest

from gui.app import resolve_view_type, should_skip_poll


# ---------------------------------------------------------------------------
# Bug 1: resolve_view_type should preserve user's view selection on hot reload
# ---------------------------------------------------------------------------

class TestResolveViewType:
    """resolve_view_type(current_view, num_active) returns the view to display."""

    def test_preserves_user_selected_view_same_dimensionality(self):
        """If user switched to contour in 2D, hot reload should keep contour."""
        assert resolve_view_type("contour", 2) == "contour"

    def test_preserves_scatter3d_when_user_selected_it(self):
        """If user switched to scatter3d in 3D, hot reload should keep it."""
        assert resolve_view_type("scatter3d", 3) == "scatter3d"

    def test_preserves_heatmap_in_2d(self):
        assert resolve_view_type("heatmap", 2) == "heatmap"

    def test_preserves_parallel_view(self):
        """Analysis views like parallel should be preserved across hot reloads."""
        assert resolve_view_type("parallel", 2) == "parallel"

    def test_falls_back_to_default_when_dimensionality_changes(self):
        """Switching from 2D to 3D should reset to the 3D default (isosurface)."""
        assert resolve_view_type("contour", 3) == "isosurface"

    def test_falls_back_to_default_when_3d_to_2d(self):
        """Switching from 3D to 2D should reset to the 2D default (contour)."""
        assert resolve_view_type("isosurface", 2) == "contour"

    def test_falls_back_to_default_when_view_is_none(self):
        """First run (no previous view) should use the default."""
        assert resolve_view_type(None, 3) == "isosurface"

    def test_falls_back_to_default_for_1d(self):
        assert resolve_view_type(None, 1) == "line"

    def test_preserves_analysis_view_across_dimensionalities(self):
        """Analysis views (parallel, slices, importance, etc.) work for any dimensionality."""
        assert resolve_view_type("slices", 3) == "slices"
        assert resolve_view_type("importance", 2) == "importance"
        assert resolve_view_type("pareto", 1) == "pareto"


# ---------------------------------------------------------------------------
# Bug 2: should_skip_poll must not swallow auto-run-trigger
# ---------------------------------------------------------------------------

class TestShouldSkipPoll:
    """should_skip_poll(triggered_ids, hot_reload, dirty, processed) -> bool"""

    def test_skips_when_only_poll_and_not_dirty(self):
        """Normal poll tick with nothing dirty should skip."""
        assert should_skip_poll(["sweep-poll"], ["on"], 0, 0) is True

    def test_does_not_skip_when_poll_and_dirty(self):
        """Poll tick with dirty > processed should not skip."""
        assert should_skip_poll(["sweep-poll"], ["on"], 5, 3) is False

    def test_does_not_skip_when_auto_run_trigger_present(self):
        """auto-run-trigger must never be skipped, even if dirty == processed."""
        assert should_skip_poll(["auto-run-trigger"], ["on"], 0, 0) is False

    def test_does_not_skip_when_both_poll_and_auto_run_fire(self):
        """Race condition: both poll and auto-run fire simultaneously."""
        assert should_skip_poll(["sweep-poll", "auto-run-trigger"], ["on"], 0, 0) is False

    def test_does_not_skip_on_run_button(self):
        assert should_skip_poll(["run-btn"], ["on"], 0, 0) is False

    def test_skips_when_hot_reload_off_and_only_poll(self):
        """With hot reload off, poll should skip."""
        assert should_skip_poll(["sweep-poll"], [], 5, 3) is True

    def test_does_not_skip_auto_run_even_with_hot_reload_off(self):
        """auto-run-trigger is the initial load — must run regardless of hot reload toggle."""
        assert should_skip_poll(["auto-run-trigger"], [], 0, 0) is False
