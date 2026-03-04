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

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Feature columns expected in the sweep CSV
FEATURES = ["N", "R1", "R2", "W_P", "L_P"]

# Numerical stability constants used in the synthetic data generator
_MIN_N = 1.01          # minimum BJT area ratio to keep log(N) > 0
_MIN_VREF_FOR_TC = 0.1  # minimum Vref [V] used as denominator in TC calculation


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
    """Gaussian Process surrogate with Matérn-3/2 kernel.

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
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            nu=1.5,
            length_scale=np.ones(X_scaled.shape[1]),
            length_scale_bounds=(1e-3, 1e3),
        )
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
    from sklearn.metrics import r2_score, mean_absolute_error

    mean, std = model.predict_with_uncertainty(X_test)
    errors = np.abs(mean - y_test)

    r2 = float(r2_score(y_test, mean))
    mae = float(mean_absolute_error(y_test, mean))
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))
    max_err = float(errors.max())
    mean_std = float(std.mean())
    within_1sigma = float((errors < std).mean())

    metrics = {
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
        "max_abs_error": max_err,
        "mean_std": mean_std,
        "within_1sigma_frac": within_1sigma,
        "coverage_90": float((errors < 1.645 * std).mean()),
    }
    logger.info(
        "Surrogate eval — R²: %.4f  MAE: %.4e  RMSE: %.4e  max|err|: %.4e  "
        "mean σ: %.4e  within 1σ: %.1f%%",
        r2, mae, rmse, max_err, mean_std, 100 * within_1sigma,
    )
    return metrics


def _generate_synthetic_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bandgap data using the analytical Brokaw formula.

    Used as a fallback when no ngspice sweep CSV is available.

    Vref ≈ Vbe + (R1/R2) * VT * ln(N)   (simplified Brokaw equation)

    Parameters
    ----------
    n:
        Number of samples to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: N, R1, R2, W_P, L_P, vref_V, tc_ppm_C, psrr_dB, iq_uA, error.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_gen.sweep_bandgap import _make_lhs_samples  # noqa: E402

    K_B = 1.380649e-23  # J/K
    Q_E = 1.602176634e-19  # C
    T_NOM = 300.0  # K

    rng = np.random.default_rng(seed)
    samples = _make_lhs_samples(n_samples=n, rng=rng)

    rows = []
    for p in samples:
        VT = 0.02585  # Thermal voltage at 300 K [V]
        Vbe = 0.650 + rng.normal(0, 0.002)  # ±2 mV process variation
        # Brokaw formula: Vref ≈ Vbe + (R1/R2) * VT * ln(N)
        Vref = Vbe + (p["R1"] / p["R2"]) * VT * float(np.log(max(p["N"], _MIN_N)))
        Vref += rng.normal(0, 0.002)  # ±2 mV simulation noise
        Vref = float(np.clip(Vref, 0.55, 3.4))
        iq_uA = Vref / p["R1"] * 1e6

        # Approximate TC using two-temperature method
        dvbe_dt = -2.0e-3
        T_cold, T_hot = 233.0, 398.0
        N = max(p["N"], _MIN_N)
        vref_cold = float(np.clip(
            (0.65 + dvbe_dt * (T_cold - T_NOM))
            + (p["R1"] / p["R2"]) * (K_B * T_cold / Q_E) * np.log(N),
            0.3, 2.5))
        vref_hot = float(np.clip(
            (0.65 + dvbe_dt * (T_hot - T_NOM))
            + (p["R1"] / p["R2"]) * (K_B * T_hot / Q_E) * np.log(N),
            0.3, 2.5))
        tc_ppm_C = round(
            abs(vref_hot - vref_cold) / (max(Vref, _MIN_VREF_FOR_TC) * (T_hot - T_cold)) * 1e6,
            2,
        )

        psrr_dB = round(-58.0 - 1.2 * np.log10(max(N, _MIN_N)), 2)

        rows.append({
            **p,
            "vref_V": float(Vref),
            "tc_ppm_C": tc_ppm_C,
            "psrr_dB": psrr_dB,
            "iq_uA": float(iq_uA),
            "error": "",
        })

    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point: train surrogate on sweep data, save checkpoint and eval plots.

    Usage::

        python -m ml.surrogate [--datasets-dir datasets] [--results-dir results]
                               [--model gp|rf] [--target vref_V]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train surrogate model on bandgap sweep data."
    )
    parser.add_argument("--datasets-dir", default="datasets",
                        help="Directory containing bandgap_sweep_*.csv files.")
    parser.add_argument("--results-dir", default="results",
                        help="Output directory for plots and metrics JSON.")
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints",
                        help="Directory to save the fitted model checkpoint.")
    parser.add_argument("--model", choices=["gp", "rf"], default="gp",
                        help="Surrogate model type: 'gp' (Gaussian Process) or 'rf' (RF).")
    parser.add_argument("--target", default="vref_V",
                        help="Target column to model (must exist in the sweep CSV).")
    parser.add_argument("--n-synthetic", type=int, default=200,
                        help="Synthetic samples to use when no valid sweep data is found.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load sweep data (or fall back to synthetic)
    # -----------------------------------------------------------------------
    datasets_dir = Path(args.datasets_dir)
    csv_files = sorted(datasets_dir.glob("bandgap_sweep_*.csv")) if datasets_dir.exists() else []

    df_valid = pd.DataFrame()
    if csv_files:
        df = pd.read_csv(csv_files[-1])
        if args.target in df.columns:
            valid_mask = (df["error"].isna() | (df["error"] == "")) & df[args.target].notna()
            df_valid = df[valid_mask]
            logger.info("Loaded %d valid rows from %s", len(df_valid), csv_files[-1])
        else:
            logger.warning("Target column '%s' not found in CSV.", args.target)

    MIN_ROWS = 10
    if len(df_valid) < MIN_ROWS:
        logger.info(
            "Fewer than %d valid rows available (%d). "
            "Generating %d synthetic samples (analytical Brokaw model).",
            MIN_ROWS, len(df_valid), args.n_synthetic,
        )
        df_valid = _generate_synthetic_data(n=args.n_synthetic)

    # -----------------------------------------------------------------------
    # 2. Build feature / target arrays
    # -----------------------------------------------------------------------
    X = df_valid[FEATURES].values
    y = df_valid[args.target].values

    n = len(X)
    n_train = max(2, int(0.8 * n))
    idx = np.random.default_rng(42).permutation(n)
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    # -----------------------------------------------------------------------
    # 3. Train
    # -----------------------------------------------------------------------
    if args.model == "gp":
        model: _BaseSurrogate = GaussianProcessSurrogate(n_restarts=5)
    else:
        model = RandomForestSurrogate(n_estimators=100)

    logger.info("Training %s surrogate on %d samples…", args.model.upper(), n_train)
    model.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # 4. Evaluate
    # -----------------------------------------------------------------------
    metrics: dict[str, Any] = {}
    if len(X_test) >= 2:
        metrics = evaluate_surrogate(model, X_test, y_test)

    # -----------------------------------------------------------------------
    # 5. Save checkpoint
    # -----------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_{args.target}.pkl"
    model.save(ckpt_path)

    # -----------------------------------------------------------------------
    # 6. Save metrics JSON
    # -----------------------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model": args.model,
        "target": args.target,
        "n_train": n_train,
        "n_test": len(X_test),
        **metrics,
    }
    metrics_path = results_dir / "surrogate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Surrogate metrics saved to %s", metrics_path)

    # -----------------------------------------------------------------------
    # 7. Plots
    # -----------------------------------------------------------------------
    if len(X_test) >= 2:
        mean, std = model.predict_with_uncertainty(X_test)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        ax.scatter(y_test, mean, alpha=0.7, edgecolors="k", linewidths=0.5)
        lo = min(float(y_test.min()), float(mean.min()))
        hi = max(float(y_test.max()), float(mean.max()))
        ax.plot([lo, hi], [lo, hi], "r--", label="ideal")
        r2_val = metrics.get("r2", float("nan"))
        ax.set_xlabel(f"Actual {args.target}")
        ax.set_ylabel(f"Predicted {args.target}")
        ax.set_title(f"Surrogate ({args.model.upper()}) — R²={r2_val:.3f}")
        ax.legend()

        ax = axes[1]
        ax.hist(std, bins=20, edgecolor="k")
        ax.set_xlabel("Predicted uncertainty (σ)")
        ax.set_ylabel("Count")
        ax.set_title("Uncertainty distribution")

        plt.tight_layout()
        plot_path = results_dir / "surrogate_eval.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Surrogate eval plot saved to %s", plot_path)

    logger.info("Done. Checkpoint: %s  Metrics: %s", ckpt_path, metrics_path)


if __name__ == "__main__":
    main()
