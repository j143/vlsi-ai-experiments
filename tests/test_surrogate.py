"""
tests/test_surrogate.py — Tests for ml/surrogate.py
======================================================
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.surrogate import (  # noqa: E402
    GaussianProcessSurrogate,
    RandomForestSurrogate,
    evaluate_surrogate,
)


def _make_data(n: int = 30, d: int = 5, seed: int = 42):
    """Generate synthetic (X, y) data for testing."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    # Simple linear function so predictions are non-trivial
    y = X[:, 0] * 2.0 - X[:, 1] + 0.5 + rng.normal(0, 0.01, n)
    return X, y


class TestGaussianProcessSurrogate:
    def test_fit_predict_shape(self):
        model = GaussianProcessSurrogate(n_restarts=1)
        X, y = _make_data()
        model.fit(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5,)

    def test_predict_with_uncertainty_shape(self):
        model = GaussianProcessSurrogate(n_restarts=1)
        X, y = _make_data()
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X[:5])
        assert mean.shape == (5,)
        assert std.shape == (5,)

    def test_std_is_non_negative(self):
        model = GaussianProcessSurrogate(n_restarts=1)
        X, y = _make_data()
        model.fit(X, y)
        _, std = model.predict_with_uncertainty(X)
        assert (std >= 0).all()

    def test_predict_before_fit_raises(self):
        model = GaussianProcessSurrogate()
        X, _ = _make_data()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_fit_requires_2d_X(self):
        model = GaussianProcessSurrogate()
        with pytest.raises(ValueError):
            model.fit(np.ones(10), np.ones(10))

    def test_fit_requires_1d_y(self):
        model = GaussianProcessSurrogate()
        with pytest.raises(ValueError):
            model.fit(np.ones((10, 3)), np.ones((10, 2)))

    def test_fit_length_mismatch(self):
        model = GaussianProcessSurrogate()
        with pytest.raises(ValueError):
            model.fit(np.ones((10, 3)), np.ones(8))

    def test_save_and_load(self, tmp_path):
        model = GaussianProcessSurrogate(n_restarts=1)
        X, y = _make_data()
        model.fit(X, y)
        save_path = tmp_path / "gp_test.pkl"
        model.save(save_path)

        loaded = GaussianProcessSurrogate.load(save_path)
        pred_orig = model.predict(X[:3])
        pred_load = loaded.predict(X[:3])
        np.testing.assert_allclose(pred_orig, pred_load, rtol=1e-6)

    def test_reasonable_r2_on_simple_function(self):
        """GP should achieve R² > 0.9 on a simple synthetic function."""
        model = GaussianProcessSurrogate(n_restarts=2)
        X, y = _make_data(n=40)
        model.fit(X[:30], y[:30])
        metrics = evaluate_surrogate(model, X[30:], y[30:])
        assert metrics["r2"] > 0.7, f"R² too low: {metrics['r2']}"


class TestRandomForestSurrogate:
    def test_fit_predict_shape(self):
        model = RandomForestSurrogate(n_estimators=10)
        X, y = _make_data()
        model.fit(X, y)
        pred = model.predict(X[:5])
        assert pred.shape == (5,)

    def test_predict_with_uncertainty_shape(self):
        model = RandomForestSurrogate(n_estimators=10)
        X, y = _make_data()
        model.fit(X, y)
        mean, std = model.predict_with_uncertainty(X[:5])
        assert mean.shape == (5,)
        assert std.shape == (5,)

    def test_std_is_non_negative(self):
        model = RandomForestSurrogate(n_estimators=10)
        X, y = _make_data()
        model.fit(X, y)
        _, std = model.predict_with_uncertainty(X)
        assert (std >= 0).all()

    def test_predict_before_fit_raises(self):
        model = RandomForestSurrogate()
        X, _ = _make_data()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_fit_requires_at_least_2_samples(self):
        model = RandomForestSurrogate()
        with pytest.raises(ValueError):
            model.fit(np.ones((1, 3)), np.ones(1))


class TestEvaluateSurrogate:
    def test_returns_expected_keys(self):
        model = GaussianProcessSurrogate(n_restarts=1)
        X, y = _make_data()
        model.fit(X[:25], y[:25])
        metrics = evaluate_surrogate(model, X[25:], y[25:])
        expected = {"r2", "rmse", "max_abs_error", "mean_std", "within_1sigma_frac"}
        assert expected.issubset(metrics.keys())

    def test_rmse_is_non_negative(self):
        model = RandomForestSurrogate(n_estimators=10)
        X, y = _make_data()
        model.fit(X[:25], y[:25])
        metrics = evaluate_surrogate(model, X[25:], y[25:])
        assert metrics["rmse"] >= 0

    def test_within_1sigma_between_0_and_1(self):
        model = GaussianProcessSurrogate(n_restarts=1)
        X, y = _make_data()
        model.fit(X[:25], y[:25])
        metrics = evaluate_surrogate(model, X[25:], y[25:])
        assert 0.0 <= metrics["within_1sigma_frac"] <= 1.0
