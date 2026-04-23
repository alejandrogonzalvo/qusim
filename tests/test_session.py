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


# ---------------------------------------------------------------------------
# dump / load: gzipped-JSON round-trip
# ---------------------------------------------------------------------------

class TestDumpLoad:
    def _example(self):
        from gui.session import collect_session
        return collect_session(
            TestCollectSession()._minimal_controls(),
            TestCollectSession()._minimal_view(),
            None,
        )

    def test_dump_returns_gzip_bytes(self):
        from gui.session import dump
        raw = dump(self._example())
        assert isinstance(raw, bytes)
        # gzip magic: first two bytes
        assert raw[:2] == b"\x1f\x8b"

    def test_load_roundtrip_from_gzip_bytes(self):
        from gui.session import dump, load
        original = self._example()
        back = load(dump(original))
        assert back == original

    def test_load_accepts_plain_json_string(self):
        from gui.session import load
        import json as _json
        payload = {"schema_version": 1, "controls": {}, "view": {}, "sweep": {"present": False}, "saved_at": "x", "app": {"name": "qusim-dse"}}
        back = load(_json.dumps(payload))
        assert back == payload

    def test_load_accepts_plain_json_bytes(self):
        from gui.session import load
        import json as _json
        payload = {"schema_version": 1, "controls": {}, "view": {}, "sweep": {"present": False}, "saved_at": "x", "app": {"name": "qusim-dse"}}
        back = load(_json.dumps(payload).encode("utf-8"))
        assert back == payload


# ---------------------------------------------------------------------------
# validate: schema_version + required top-level keys
# ---------------------------------------------------------------------------

class TestValidate:
    def _good(self):
        return {
            "schema_version": 1,
            "saved_at": "2026-04-23T00:00:00Z",
            "app": {"name": "qusim-dse"},
            "controls": {}, "view": {},
            "sweep": {"present": False},
        }

    def test_good_session_passes(self):
        from gui.session import validate
        validate(self._good())  # no exception

    def test_missing_schema_version_raises(self):
        from gui.session import validate, SessionError
        bad = self._good()
        del bad["schema_version"]
        with pytest.raises(SessionError, match="schema_version"):
            validate(bad)

    def test_wrong_schema_version_raises(self):
        from gui.session import validate, SessionError
        bad = self._good()
        bad["schema_version"] = 999
        with pytest.raises(SessionError, match="version"):
            validate(bad)

    def test_missing_controls_raises(self):
        from gui.session import validate, SessionError
        bad = self._good()
        del bad["controls"]
        with pytest.raises(SessionError, match="controls"):
            validate(bad)

    def test_non_dict_raises(self):
        from gui.session import validate, SessionError
        with pytest.raises(SessionError):
            validate([1, 2, 3])
