"""
ml/surrogate.py — Surrogate Model for Bandgap Reference Design
================================================================
Provides a surrogate that predicts bandgap outputs (Vref, TC, PSRR, Iq) from
design variables, enabling fast evaluation without ngspice.

Models provided:
  - GaussianProcessSurrogate: GP-based model with uncertainty estimates (recommended).
  - RandomForestSurrogate: Ensemble model, faster training, less calibrated uncertainty.

Usage::

    import pandas as pd
    from ml.surrogate import GaussianProcessSurrogate

    df = pd.read_csv("datasets/bandgap_sweep_<timestamp>.csv")
    FEATURES = ["N", "R1", "R2", "W_P", "L_P"]
    TARGETS = ["vref_V", "iq_uA"]

    X = df[FEATURES].values
    y = df["vref_V"].values
    valid = ~pd.isna(df["vref_V"])  # skip rows where simulation failed

    model = GaussianProcessSurrogate()
    model.fit(X[valid], y[valid])
    mean, std = model.predict_with_uncertainty(X[:5])
    print(mean, std)

Design decision:
    GP is the default because:
    - Uncertainty estimates are calibrated out of the box.
    - Well-suited for small datasets (< 500 points).
    - Training cost is manageable for D ≤ 10 design variables.
    For larger datasets, switch to RandomForestSurrogate or a neural network.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class _BaseSurrogate:
    """Abstract base class for surrogate models.

    Subclasses must implement ``_train`` and ``_predict_raw``.
    """

    def __init__(self) -> None:
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseSurrogate":
        """Fit the surrogate model.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features). Each row is a design point.
        y:
            Target vector, shape (n_samples,). Should be a single output (e.g., vref_V).

        Returns
        -------
        self
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        if len(X) < 2:
            raise ValueError("Need at least 2 samples to fit.")

        X_scaled = self._scaler_X.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self._train(X_scaled, y_scaled)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted mean for each row of X.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values, shape (n_samples,), in original (unscaled) units.
        """
        self._check_fitted()
        X_scaled = self._scaler_X.transform(X)
        y_scaled = self._predict_mean(X_scaled)
        return self._scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted mean and standard deviation for each row of X.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        tuple of (mean, std)
            Both arrays have shape (n_samples,) and are in original units.
        """
        self._check_fitted()
        X_scaled = self._scaler_X.transform(X)
        mean_scaled, std_scaled = self._predict_with_uncertainty_scaled(X_scaled)
        mean = self._scaler_y.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        # std scales by the same factor as mean (no shift for std)
        std = std_scaled * self._scaler_y.scale_[0]
        return mean, std

    def save(self, path: str | Path) -> None:
        """Save the fitted model to a pickle file.

        Parameters
        ----------
        path:
            Output file path (e.g., ``ml/checkpoints/gp_vref.pkl``).
        """
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "_BaseSurrogate":
        """Load a previously saved model.

        Parameters
        ----------
        path:
            Path to the pickle file.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    # Subclasses implement these:
    def _train(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        raise NotImplementedError

    def _predict_mean(self, X_scaled: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _predict_with_uncertainty_scaled(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class GaussianProcessSurrogate(_BaseSurrogate):
    """Gaussian Process surrogate with Matérn-5/2 kernel.

    Provides calibrated uncertainty estimates, making it the recommended model
    for Bayesian optimization loops with small datasets (< 500 samples).

    Parameters
    ----------
    n_restarts:
        Number of optimizer restarts for hyperparameter fitting. Higher = better
        fit, slower training. Default 5.
    noise_level:
        Initial noise variance for the WhiteKernel. If 0, use a noiseless GP
        (only appropriate for near-noise-free simulation data).
    """

    def __init__(self, n_restarts: int = 5, noise_level: float = 1e-6) -> None:
        super().__init__()
        self.n_restarts = n_restarts
        self.noise_level = noise_level
        self._model: GaussianProcessRegressor | None = None

    def _train(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(X_scaled.shape[1]))
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            alpha=self.noise_level,
            normalize_y=False,  # We handle normalization ourselves
        )
        self._model.fit(X_scaled, y_scaled)
        logger.info("GP fitted. Kernel: %s", self._model.kernel_)

    def _predict_mean(self, X_scaled: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X_scaled, return_std=False)

    def _predict_with_uncertainty_scaled(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._model is not None
        mean, std = self._model.predict(X_scaled, return_std=True)
        return mean, std


class RandomForestSurrogate(_BaseSurrogate):
    """Random Forest surrogate model.

    Faster training than GP; uncertainty estimated from tree variance.
    Recommended for larger datasets (> 500 samples) or high-dimensional spaces.

    Parameters
    ----------
    n_estimators:
        Number of trees. Default 100.
    max_depth:
        Maximum tree depth. Default None (unlimited).
    """

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model: RandomForestRegressor | None = None

    def _train(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._model.fit(X_scaled, y_scaled)

    def _predict_mean(self, X_scaled: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(X_scaled)

    def _predict_with_uncertainty_scaled(
        self, X_scaled: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._model is not None
        # Uncertainty from tree prediction variance
        tree_preds = np.array([
            tree.predict(X_scaled) for tree in self._model.estimators_
        ])
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean, std


def evaluate_surrogate(
    model: _BaseSurrogate,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fitted surrogate on a test set and return metrics.

    Parameters
    ----------
    model:
        A fitted surrogate model.
    X_test:
        Test feature matrix, shape (n_test, n_features).
    y_test:
        Test targets, shape (n_test,).

    Returns
    -------
    dict
        Contains: ``r2``, ``rmse``, ``max_abs_error``, ``mean_std``,
        ``within_1sigma_frac`` (fraction of test points where |error| < 1σ).
    """
    from sklearn.metrics import r2_score

    mean, std = model.predict_with_uncertainty(X_test)
    errors = np.abs(mean - y_test)

    r2 = float(r2_score(y_test, mean))
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))
    max_err = float(errors.max())
    mean_std = float(std.mean())
    within_1sigma = float((errors < std).mean())

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "max_abs_error": max_err,
        "mean_std": mean_std,
        "within_1sigma_frac": within_1sigma,
    }
    logger.info(
        "Surrogate eval — R²: %.4f  RMSE: %.4e  max|err|: %.4e  "
        "mean σ: %.4e  within 1σ: %.1f%%",
        r2, rmse, max_err, mean_std, 100 * within_1sigma,
    )
    return metrics
