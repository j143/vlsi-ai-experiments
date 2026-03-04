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

    def test_returns_expected_keys(self, client):
        resp = client.post(
            "/api/simulate",
            json={"params": self._PARAMS},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ("params", "vref_V", "iq_uA", "spec_checks"):
            assert key in data

    def test_vref_is_numeric(self, client):
        resp = client.post("/api/simulate", json={"params": self._PARAMS})
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

    def test_empty_params_still_runs(self, client):
        resp = client.post("/api/simulate", json={"params": {}})
        assert resp.status_code == 200


class TestOptimize:
    def test_returns_expected_keys(self, client):
        resp = client.post(
            "/api/optimize",
            json={"budget": 3, "n_init": 2, "seed": 0},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ("best_params", "n_simulations", "n_spec_pass", "history", "convergence"):
            assert key in data

    def test_n_simulations_matches_budget(self, client):
        resp = client.post(
            "/api/optimize",
            json={"budget": 3, "n_init": 2, "seed": 0},
        )
        data = resp.get_json()
        assert data["n_simulations"] == 3

    def test_history_length_matches_budget(self, client):
        resp = client.post(
            "/api/optimize",
            json={"budget": 4, "n_init": 2, "seed": 1},
        )
        data = resp.get_json()
        assert len(data["history"]) == 4

    def test_convergence_length_matches_history(self, client):
        resp = client.post(
            "/api/optimize",
            json={"budget": 4, "n_init": 2, "seed": 1},
        )
        data = resp.get_json()
        assert len(data["convergence"]) == len(data["history"])

    def test_budget_clamped_to_maximum(self, client):
        resp = client.post(
            "/api/optimize",
            json={"budget": 9999, "n_init": 1, "seed": 0},
        )
        data = resp.get_json()
        assert data["n_simulations"] <= 100

    def test_default_params_accepted(self, client):
        """Calling /api/optimize with no body should work."""
        resp = client.post("/api/optimize", json={})
        assert resp.status_code == 200
