"""
tests/test_data_gen.py — Tests for data_gen/sweep_bandgap.py
==============================================================
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_gen.sweep_bandgap import (  # noqa: E402
    PARAM_SPACE,
    _make_grid_samples,
    _make_lhs_samples,
    run_sweep,
)


class TestMakeGridSamples:
    def test_invalid_n_per_dim_raises(self):
        with pytest.raises(ValueError):
            _make_grid_samples(n_per_dim=0)

    def test_returns_list_of_dicts(self):
        samples = _make_grid_samples(n_per_dim=2)
        assert isinstance(samples, list)
        assert all(isinstance(s, dict) for s in samples)

    def test_correct_number_of_samples(self):
        n = 2
        samples = _make_grid_samples(n_per_dim=n)
        assert len(samples) == n ** len(PARAM_SPACE)

    def test_all_params_present(self):
        samples = _make_grid_samples(n_per_dim=2)
        expected_keys = {name for name, *_ in PARAM_SPACE}
        for s in samples:
            assert set(s.keys()) == expected_keys

    def test_integer_params_are_int(self):
        samples = _make_grid_samples(n_per_dim=3)
        for s in samples:
            for name, lo, hi, scale in PARAM_SPACE:
                if scale == "int":
                    assert isinstance(s[name], int), f"{name} should be int"

    def test_values_within_bounds(self):
        samples = _make_grid_samples(n_per_dim=3)
        for s in samples:
            for name, lo, hi, scale in PARAM_SPACE:
                # Use small tolerance for floating-point rounding from logspace
                rtol = 1e-9
                assert lo * (1 - rtol) <= s[name] <= hi * (1 + rtol), (
                    f"{name}={s[name]} out of [{lo}, {hi}]"
                )


class TestMakeLHSSamples:
    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError):
            _make_lhs_samples(n_samples=0)

    def test_returns_correct_count(self):
        samples = _make_lhs_samples(n_samples=20)
        assert len(samples) == 20

    def test_all_params_present(self):
        samples = _make_lhs_samples(n_samples=5)
        expected_keys = {name for name, *_ in PARAM_SPACE}
        for s in samples:
            assert set(s.keys()) == expected_keys

    def test_values_within_bounds(self):
        samples = _make_lhs_samples(n_samples=50)
        for s in samples:
            for name, lo, hi, scale in PARAM_SPACE:
                assert lo <= s[name] <= hi, f"{name}={s[name]} out of [{lo}, {hi}]"

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(seed=0)
        rng2 = np.random.default_rng(seed=0)
        s1 = _make_lhs_samples(n_samples=10, rng=rng1)
        s2 = _make_lhs_samples(n_samples=10, rng=rng2)
        for a, b in zip(s1, s2):
            for key in a:
                assert a[key] == b[key]

    def test_different_seeds_give_different_samples(self):
        s1 = _make_lhs_samples(n_samples=20, rng=np.random.default_rng(0))
        s2 = _make_lhs_samples(n_samples=20, rng=np.random.default_rng(1))
        # At least one sample should differ
        any_diff = any(
            any(a[k] != b[k] for k in a)
            for a, b in zip(s1, s2)
        )
        assert any_diff


class TestRunSweep:
    def test_runner_exception_is_captured(self, tmp_path):
        mock_runner = MagicMock()
        mock_runner.run.side_effect = RuntimeError("boom")
        mock_runner.is_ngspice_available.return_value = True
        samples = _make_lhs_samples(n_samples=2)
        df = run_sweep(samples, out_dir=tmp_path, runner=mock_runner)
        assert (df["error"] == "boom").all()
        for name, *_ in PARAM_SPACE:
            assert name in df.columns
            assert df[name].notna().all()

    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        """run_sweep should return a DataFrame even when ngspice is unavailable."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "vref_V": 1.2,
            "iq_uA": 25.0,
            "spec_checks": {"vref": True, "iq": True},
            "error": None,
        }
        mock_runner.is_ngspice_available.return_value = True

        samples = _make_lhs_samples(n_samples=3)
        df = run_sweep(samples, out_dir=tmp_path, runner=mock_runner)

        assert len(df) == 3
        assert "vref_V" in df.columns
        assert "iq_uA" in df.columns
        assert "error" in df.columns
        assert "sim_time_s" in df.columns

    def test_saves_csv_file(self, tmp_path):
        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "vref_V": None,
            "iq_uA": None,
            "spec_checks": {},
            "error": "ngspice not found",
        }
        mock_runner.is_ngspice_available.return_value = False

        samples = _make_lhs_samples(n_samples=2)
        run_sweep(samples, out_dir=tmp_path, runner=mock_runner)

        csv_files = list(tmp_path.glob("bandgap_sweep_*.csv"))
        assert len(csv_files) == 1

    def test_errors_are_captured_not_raised(self, tmp_path):
        """Simulation errors should be logged, not propagate as exceptions."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "vref_V": None,
            "iq_uA": None,
            "spec_checks": {},
            "error": "timeout",
        }
        mock_runner.is_ngspice_available.return_value = True

        samples = _make_lhs_samples(n_samples=2)
        df = run_sweep(samples, out_dir=tmp_path, runner=mock_runner)
        assert (df["error"] == "timeout").all()
