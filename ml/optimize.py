"""
ml/optimize.py — Bayesian Optimization Loop for Bandgap Design
================================================================
Wraps a surrogate model in a Bayesian Optimization (BO) loop that:
  1. Proposes new design points via an acquisition function (EI).
  2. Selectively calls ngspice for high-value candidates.
  3. Logs each iteration to a structured JSON/CSV file.
  4. Reports total simulations used, spec satisfaction rate, and best design.

Usage::

    from ml.optimize import BayesianOptimizer
    from bandgap.runner import BandgapRunner

    runner = BandgapRunner()
    opt = BayesianOptimizer(runner=runner, budget=30)
    result = opt.run()
    print(result.best_params, result.best_vref)

Algorithm
---------
- Acquisition function: Expected Improvement (EI) — balances exploration/exploitation.
- Initialization: Latin-Hypercube Samples (LHS) for the first `n_init` points.
- Surrogate update: Re-fit GP after every ngspice call.
- Termination: When `budget` ngspice calls are exhausted or spec is met for
  `early_stop_streak` consecutive iterations.

References
----------
[1] Brochu et al., "A Tutorial on Bayesian Optimization," arXiv 2010.
[2] Snoek et al., "Practical Bayesian Optimization of ML algorithms," NeurIPS 2012.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Add repo root to sys.path if running as a script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_gen.sweep_bandgap import PARAM_SPACE, _make_lhs_samples  # noqa: E402
from ml.surrogate import GaussianProcessSurrogate  # noqa: E402

logger = logging.getLogger(__name__)

# Spec target (loaded from bandgap/specs.yaml at runtime)
_REPO_ROOT = Path(__file__).parent.parent
SPECS_FILE = _REPO_ROOT / "bandgap" / "specs.yaml"


@dataclass
class OptimizationResult:
    """Summary of a completed optimization run."""

    best_params: dict[str, Any]
    best_vref_V: float | None
    n_simulations: int
    n_spec_pass: int
    history: list[dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def spec_pass_rate(self) -> float:
        """Fraction of valid simulations that passed the Vref spec."""
        if self.n_simulations == 0:
            return 0.0
        return self.n_spec_pass / self.n_simulations

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_vref_V": self.best_vref_V,
            "n_simulations": self.n_simulations,
            "n_spec_pass": self.n_spec_pass,
            "spec_pass_rate": self.spec_pass_rate(),
            "history": self.history,
            "timestamp": self.timestamp,
        }

    def save(self, path: str | Path) -> None:
        """Save result to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Optimization result saved to %s", path)


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_so_far: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute Expected Improvement acquisition function.

    Parameters
    ----------
    mean:
        Predicted mean values, shape (n,).
    std:
        Predicted std values, shape (n,).
    best_so_far:
        Best target value observed so far (we minimize |vref - target|).
    xi:
        Exploration-exploitation trade-off parameter. Higher = more exploration.

    Returns
    -------
    np.ndarray
        EI values, shape (n,). Higher is better.
    """
    from scipy.stats import norm

    # We are minimizing the absolute error: f(x) = |vref(x) - target|
    # EI = E[max(best - f(x), 0)]
    improvement = best_so_far - mean - xi
    Z = improvement / (std + 1e-9)
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std <= 0] = 0.0
    return ei


def _params_to_array(params: dict[str, Any]) -> np.ndarray:
    """Convert a parameter dict to a 1D array in PARAM_SPACE order."""
    return np.array([float(params[name]) for name, *_ in PARAM_SPACE])


def _array_to_params(arr: np.ndarray) -> dict[str, Any]:
    """Convert a 1D array to a parameter dict."""
    result = {}
    for j, (name, lo, hi, scale) in enumerate(PARAM_SPACE):
        val = float(np.clip(arr[j], lo, hi))
        if scale == "int":
            val = int(round(val))
        result[name] = val
    return result


class BayesianOptimizer:
    """Bayesian Optimization loop for bandgap design.

    Parameters
    ----------
    runner:
        BandgapRunner instance for ngspice simulation.
    budget:
        Maximum number of ngspice calls allowed.
    n_init:
        Number of initial LHS points before BO starts.
    n_candidates:
        Number of random candidates evaluated per BO iteration.
    xi:
        EI exploration parameter.
    specs_file:
        Path to bandgap/specs.yaml.
    results_dir:
        Directory to save run logs.
    """

    def __init__(
        self,
        runner: Any,
        budget: int = 30,
        n_init: int = 10,
        n_candidates: int = 1000,
        xi: float = 0.01,
        specs_file: Path | str = SPECS_FILE,
        results_dir: Path | str = "results",
    ) -> None:
        self.runner = runner
        self.budget = budget
        self.n_init = min(n_init, budget)
        self.n_candidates = n_candidates
        self.xi = xi
        self.results_dir = Path(results_dir)

        with open(specs_file) as f:
            self.specs = yaml.safe_load(f)

        self._vref_target = self.specs["vref"]["target_V"]
        self._vref_tol = self.specs["vref"]["tolerance_V"]

    def run(self, seed: int = 42) -> OptimizationResult:
        """Execute the Bayesian Optimization loop.

        Parameters
        ----------
        seed:
            Random seed for reproducibility.

        Returns
        -------
        OptimizationResult
        """
        rng = np.random.default_rng(seed=seed)
        history: list[dict[str, Any]] = []
        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []  # |vref - target| — we minimize this
        n_sim = 0
        n_pass = 0
        best_error = np.inf
        best_params: dict[str, Any] = {}

        # Phase 1: LHS initialization
        logger.info("=== BO Phase 1: LHS initialization (%d points) ===", self.n_init)
        init_samples = _make_lhs_samples(n_samples=self.n_init, rng=rng)

        for params in init_samples:
            if n_sim >= self.budget:
                break
            entry = self._simulate_and_log(params, iteration=n_sim, source="lhs")
            history.append(entry)
            n_sim += 1

            if entry["vref_V"] is not None:
                err = abs(entry["vref_V"] - self._vref_target)
                X_obs.append(_params_to_array(params))
                y_obs.append(err)
                if entry.get("spec_vref_pass"):
                    n_pass += 1
                if err < best_error:
                    best_error = err
                    best_params = params

        # Phase 2: BO loop
        logger.info("=== BO Phase 2: Bayesian loop (budget %d) ===", self.budget - n_sim)
        model = GaussianProcessSurrogate(n_restarts=3)

        while n_sim < self.budget:
            if len(X_obs) < 3:
                # Not enough data yet — fall back to random
                candidate_params = _make_lhs_samples(n_samples=1, rng=rng)[0]
                acquisition_score = float("nan")
            else:
                # Fit/refit surrogate on all observations
                X_arr = np.array(X_obs)
                y_arr = np.array(y_obs)
                try:
                    model.fit(X_arr, y_arr)
                except Exception as exc:
                    logger.warning("GP fit failed: %s. Falling back to random.", exc)
                    candidate_params = _make_lhs_samples(n_samples=1, rng=rng)[0]
                    acquisition_score = float("nan")
                else:
                    # Evaluate EI over random candidates
                    cand_samples = _make_lhs_samples(n_samples=self.n_candidates, rng=rng)
                    C = np.array([_params_to_array(p) for p in cand_samples])
                    mu, sigma = model.predict_with_uncertainty(C)
                    ei = _expected_improvement(mu, sigma, best_error, xi=self.xi)
                    best_idx = int(np.argmax(ei))
                    candidate_params = cand_samples[best_idx]
                    acquisition_score = float(ei[best_idx])

            entry = self._simulate_and_log(
                candidate_params, iteration=n_sim, source="bo",
                acquisition_score=acquisition_score if len(X_obs) >= 3 else float("nan"),
            )
            history.append(entry)
            n_sim += 1

            if entry["vref_V"] is not None:
                err = abs(entry["vref_V"] - self._vref_target)
                X_obs.append(_params_to_array(candidate_params))
                y_obs.append(err)
                if entry.get("spec_vref_pass"):
                    n_pass += 1
                if err < best_error:
                    best_error = err
                    best_params = candidate_params
                    logger.info(
                        "New best: vref=%.4f V, error=%.4f V (iter %d)",
                        entry["vref_V"], err, n_sim,
                    )

        best_vref = None
        if best_params:
            last_best = next(
                (h for h in reversed(history) if h["params"] == best_params), None
            )
            if last_best:
                best_vref = last_best.get("vref_V")

        result = OptimizationResult(
            best_params=best_params,
            best_vref_V=best_vref,
            n_simulations=n_sim,
            n_spec_pass=n_pass,
            history=history,
        )

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        result.save(self.results_dir / f"bo_run_{timestamp}.json")

        logger.info(
            "=== BO complete: %d simulations, %d spec-pass (%.0f%%), best error=%.4f V ===",
            n_sim, n_pass, 100 * result.spec_pass_rate(), best_error,
        )
        return result

    def _simulate_and_log(
        self,
        params: dict[str, Any],
        iteration: int,
        source: str,
        acquisition_score: float = float("nan"),
    ) -> dict[str, Any]:
        """Run one simulation and return a log entry."""
        t0 = time.perf_counter()
        sim_result = self.runner.run(params)
        elapsed = time.perf_counter() - t0

        vref = sim_result.get("vref_V")
        spec_pass = sim_result.get("spec_checks", {}).get("vref", False)

        entry = {
            "iteration": iteration,
            "source": source,
            "params": params,
            "vref_V": vref,
            "iq_uA": sim_result.get("iq_uA"),
            "spec_vref_pass": spec_pass,
            "acquisition_score": acquisition_score,
            "sim_time_s": round(elapsed, 4),
            "error": sim_result.get("error") or "",
        }
        logger.debug("Iter %d [%s] vref=%.4f spec=%s", iteration, source,
                     vref or float("nan"), spec_pass)
        return entry
