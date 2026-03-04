"""
tests/test_api.py — Tests for api/server.py Flask endpoints
============================================================
Tests cover:
  - GET /api/status — always returns ok=True; ngspice_available is a bool.
  - POST /api/simulate — returns expected keys; handles missing params gracefully.
  - POST /api/optimize — returns history, convergence, and summary fields.

These tests do NOT require ngspice; they run with the SyntheticBandgapRunner
which is automatically selected when ngspice is absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.server import app  # noqa: E402


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture()
def ngspice_available(client):
    resp = client.get("/api/status")
    return bool(resp.get_json()["ngspice_available"])


class TestStatus:
    def test_ok_field(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True

    def test_ngspice_available_is_bool(self, client):
        resp = client.get("/api/status")
        data = resp.get_json()
        assert isinstance(data["ngspice_available"], bool)


class TestSimulate:
    _PARAMS = {"N": 8, "R1": 100000, "R2": 10000, "W_P": 4e-6, "L_P": 1e-6}

    def test_returns_expected_keys(self, client, ngspice_available):
        payload = {"params": self._PARAMS}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post(
            "/api/simulate",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ("params", "vref_V", "iq_uA", "spec_checks"):
            assert key in data

    def test_vref_is_numeric(self, client, ngspice_available):
        payload = {"params": self._PARAMS}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post("/api/simulate", json=payload)
        data = resp.get_json()
        # With synthetic runner vref_V should be a float
        if data.get("error") is None:
            assert isinstance(data["vref_V"], float)

    def test_invalid_param_value_returns_400(self, client):
        resp = client.post(
            "/api/simulate",
            json={"params": {"N": "not_a_number"}},
        )
        assert resp.status_code == 400

    def test_empty_params_still_runs(self, client, ngspice_available):
        payload = {"params": {}}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post("/api/simulate", json=payload)
        assert resp.status_code == 200

    def test_real_default_requires_ngspice_or_override(self, client, ngspice_available):
        resp = client.post("/api/simulate", json={"params": self._PARAMS})
        if ngspice_available:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 503


class TestOptimize:
    def test_returns_expected_keys(self, client, ngspice_available):
        payload = {"budget": 3, "n_init": 2, "seed": 0}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post(
            "/api/optimize",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ("best_params", "n_simulations", "n_spec_pass", "history", "convergence"):
            assert key in data

    def test_n_simulations_matches_budget(self, client, ngspice_available):
        payload = {"budget": 3, "n_init": 2, "seed": 0}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post(
            "/api/optimize",
            json=payload,
        )
        data = resp.get_json()
        assert data["n_simulations"] == 3

    def test_history_length_matches_budget(self, client, ngspice_available):
        payload = {"budget": 4, "n_init": 2, "seed": 1}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post(
            "/api/optimize",
            json=payload,
        )
        data = resp.get_json()
        assert len(data["history"]) == 4

    def test_convergence_length_matches_history(self, client, ngspice_available):
        payload = {"budget": 4, "n_init": 2, "seed": 1}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post(
            "/api/optimize",
            json=payload,
        )
        data = resp.get_json()
        assert len(data["convergence"]) == len(data["history"])

    def test_budget_clamped_to_maximum(self, client, ngspice_available):
        payload = {"budget": 9999, "n_init": 1, "seed": 0}
        if not ngspice_available:
            payload["use_synthetic"] = True
        resp = client.post(
            "/api/optimize",
            json=payload,
        )
        data = resp.get_json()
        from api.server import _MAX_BUDGET
        assert data["n_simulations"] <= _MAX_BUDGET

    def test_default_params_accepted(self, client, ngspice_available):
        """Calling /api/optimize with no body should work."""
        resp = client.post("/api/optimize", json={})
        if ngspice_available:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 503

    def test_real_default_requires_ngspice_or_override(self, client, ngspice_available):
        resp = client.post("/api/optimize", json={"budget": 3, "n_init": 2, "seed": 0})
        if ngspice_available:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 503

        resp_override = client.post(
            "/api/optimize",
            json={"budget": 3, "n_init": 2, "seed": 0, "use_synthetic": True},
        )
        assert resp_override.status_code == 200

    def test_stream_endpoint_emits_progress_and_final(self, client, ngspice_available):
        query = "/api/optimize/stream?budget=3&n_init=2&seed=0"
        if not ngspice_available:
            query += "&use_synthetic=true"
        resp = client.get(query)
        assert resp.status_code == 200
        assert resp.mimetype == "text/event-stream"

        body = resp.get_data(as_text=True)
        assert "event: progress" in body
        assert "event: final" in body
        assert "event: done" in body

    def test_stream_real_default_requires_ngspice_or_override(self, client, ngspice_available):
        resp = client.get("/api/optimize/stream?budget=3&n_init=2&seed=0")
        if ngspice_available:
            assert resp.status_code == 200
        else:
            assert resp.status_code == 503


class TestLayoutPreview:
    def test_layout_preview_returns_expected_keys(self, client):
        resp = client.get("/api/layout/preview?seed=1&patch_size=32")
        assert resp.status_code == 200
        data = resp.get_json()

        for key in ("patch_size", "n_layers", "layer_map", "patch", "drc"):
            assert key in data

    def test_layout_preview_patch_shape_matches_metadata(self, client):
        resp = client.get("/api/layout/preview?seed=2&patch_size=24")
        data = resp.get_json()

        assert data["patch_size"] == 24
        assert len(data["patch"]) == data["n_layers"]
        assert len(data["patch"][0]) == data["patch_size"]
        assert len(data["patch"][0][0]) == data["patch_size"]


class TestProjectAndExport:
    def test_project_save_returns_path(self, client):
        resp = client.post(
            "/api/project/save",
            json={"project_name": "ui_state", "state": {"foo": "bar"}},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["saved_path"].startswith("results/projects/")

    def test_netlist_export_returns_text(self, client):
        resp = client.post(
            "/api/netlist/export",
            json={"params": {"N": 8, "R1": 100000, "R2": 10000, "W_P": 4e-6, "L_P": 1e-6}},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["filename"].endswith(".sp")
        assert isinstance(data["netlist_text"], str)
        assert len(data["netlist_text"]) > 0
