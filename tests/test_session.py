"""Tests for save/load sessions (serialization only; Dash wiring is separate)."""

import pytest


# ---------------------------------------------------------------------------
# collect_session: build a JSON-shaped dict from raw control/view/sweep values
# ---------------------------------------------------------------------------

class TestCollectSession:
    def _minimal_controls(self):
        return {
            "num_metrics": 3,
            "axes": [
                {"key": "t1", "slider": [4.0, 6.0], "checklist": None},
                {"key": "t2", "slider": [4.0, 6.0], "checklist": None},
                {"key": "two_gate_time", "slider": [1.0, 2.0], "checklist": None},
            ],
            "circuit": {
                "num_qubits": 16, "num_cores": 4, "seed": 42,
                "circuit_type": "qft", "topology_type": "ring",
                "intracore_topology": "all_to_all", "placement": "random",
                "routing_algorithm": "hqa_sabre", "dynamic_decoupling": False,
            },
            "noise": {"single_gate_error": 1e-4},
            "thresholds": {
                "output_metric": "overall_fidelity",
                "enable": True, "num_thresholds": 3,
                "values": [0.5, 0.7, 0.9, None, None],
                "colors": ["#ff0000", "#ffaa00", "#00ff00", None, None],
            },
            "performance": {"max_cold": None, "max_hot": None, "max_workers": None},
            "hot_reload": False,
        }

    def _minimal_view(self):
        return {"view_type": "isosurface", "frozen_axis": 2, "frozen_slider_value": 0.5}

    def test_includes_schema_version(self):
        from gui.session import collect_session, SCHEMA_VERSION
        s = collect_session(self._minimal_controls(), self._minimal_view(), None)
        assert s["schema_version"] == SCHEMA_VERSION

    def test_includes_timestamp(self):
        from gui.session import collect_session
        s = collect_session(self._minimal_controls(), self._minimal_view(), None)
        assert "saved_at" in s
        assert s["saved_at"].endswith("Z") or "+" in s["saved_at"]

    def test_sweep_absent_when_none(self):
        from gui.session import collect_session
        s = collect_session(self._minimal_controls(), self._minimal_view(), None)
        assert s["sweep"]["present"] is False
        assert "grid" not in s["sweep"]

    def test_sweep_present_with_data(self):
        from gui.session import collect_session
        sweep = {
            "metric_keys": ["t1", "t2"],
            "xs": [1.0, 2.0], "ys": [3.0, 4.0],
            "shape": [2, 2],
            "grid": [[{"overall_fidelity": 0.9}, {"overall_fidelity": 0.8}],
                     [{"overall_fidelity": 0.7}, {"overall_fidelity": 0.6}]],
        }
        s = collect_session(self._minimal_controls(), self._minimal_view(), sweep)
        assert s["sweep"]["present"] is True
        assert s["sweep"]["metric_keys"] == ["t1", "t2"]
        assert s["sweep"]["grid"] == sweep["grid"]

    def test_roundtrips_controls_verbatim(self):
        from gui.session import collect_session
        ctrls = self._minimal_controls()
        s = collect_session(ctrls, self._minimal_view(), None)
        assert s["controls"] == ctrls

    def test_roundtrips_view_verbatim(self):
        from gui.session import collect_session
        view = self._minimal_view()
        s = collect_session(self._minimal_controls(), view, None)
        assert s["view"] == view
