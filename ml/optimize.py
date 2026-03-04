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

import argparse
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
    # Top spec-passing candidate designs, sorted by closeness to Vref target.
    # Each entry has the same keys as a history entry.
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def spec_pass_rate(self) -> float:
        """Fraction of valid simulations that passed the Vref spec."""
        if self.n_simulations == 0:
            return 0.0
        return self.n_spec_pass / self.n_simulations

    def top_k_candidates(self, k: int = 3) -> list[dict[str, Any]]:
        """Return up to *k* spec-passing candidates sorted by closeness to Vref target.

        Parameters
        ----------
        k:
            Maximum number of candidates to return.

        Returns
        -------
        list of dict
            Each dict has at least ``params``, ``vref_V``, and ``spec_vref_pass`` keys.
        """
        return self.top_candidates[:k]

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_vref_V": self.best_vref_V,
            "n_simulations": self.n_simulations,
            "n_spec_pass": self.n_spec_pass,
            "spec_pass_rate": self.spec_pass_rate(),
            "top_candidates": self.top_candidates,
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
        passing_entries: list[dict[str, Any]] = []  # spec-passing entries for top_candidates

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
                    passing_entries.append(entry)
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
                    passing_entries.append(entry)
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

        # Build top_candidates: spec-passing designs sorted by closeness to target.
        # Deduplicate by rounding Vref to 1 mV to avoid near-identical entries.
        seen_vrefs: set[int] = set()
        top_candidates: list[dict[str, Any]] = []
        for e in sorted(passing_entries, key=lambda x: abs(x["vref_V"] - self._vref_target)):
            v_rounded = round(e["vref_V"] * 1000)  # quantize to 1 mV
            if v_rounded not in seen_vrefs:
                seen_vrefs.add(v_rounded)
                top_candidates.append(e)

        result = OptimizationResult(
            best_params=best_params,
            best_vref_V=best_vref,
            n_simulations=n_sim,
            n_spec_pass=n_pass,
            history=history,
            top_candidates=top_candidates,
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


# ---------------------------------------------------------------------------
# Synthetic runner — analytical Brokaw approximation (no ngspice required)
# ---------------------------------------------------------------------------

class SyntheticBandgapRunner:
    """Lightweight runner using the analytic Brokaw formula.

    Replaces ngspice when it is not installed, so that the full BO pipeline
    can be exercised without a real SPICE simulator.

    Vref ≈ Vbe + (R1/R2) * VT * ln(N)

    Notes
    -----
    - Results are physically plausible but NOT silicon-verified.
    - A small amount of Gaussian noise is added to simulate process variation.
    - Spec checks use the same thresholds as bandgap/specs.yaml.
    """

    def __init__(
        self,
        specs_file: Path | str = SPECS_FILE,
        noise_std: float = 0.002,
        seed: int = 0,
    ) -> None:
        with open(specs_file) as f:
            self.specs = yaml.safe_load(f)
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the analytic bandgap model at *params*."""
        N = float(params.get("N", 8))
        R1 = float(params.get("R1", 100e3))
        R2 = float(params.get("R2", 10e3))

        VT = 0.02585  # Thermal voltage at 300 K
        Vbe = 0.650 + self._rng.normal(0, self.noise_std)
        # Brokaw formula: Vref ≈ Vbe + (R1/R2) * VT * ln(N)
        Vref = Vbe + (R1 / R2) * VT * np.log(max(N, 1.0))
        Vref += self._rng.normal(0, self.noise_std)
        iq_uA = Vref / R1 * 1e6

        target = self.specs["vref"]["target_V"]
        tol = self.specs["vref"]["tolerance_V"]
        spec_vref = bool(abs(Vref - target) <= tol)
        spec_iq = bool(iq_uA <= self.specs["quiescent_current"]["max_uA"])

        return {
            "params": params,
            "vref_V": float(Vref),
            "iq_uA": float(iq_uA),
            "spec_checks": {"vref": spec_vref, "iq": spec_iq},
            "raw_output": "",
            "error": None,
        }

    def is_ngspice_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_convergence(result: OptimizationResult, results_dir: Path) -> None:
    """Plot best-so-far Vref error vs. BO iteration number."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vref_target = None
    errors: list[float] = []
    best_so_far = float("inf")

    # Read target from specs (if available), fall back to 1.2 V
    try:
        with open(SPECS_FILE) as f:
            specs = yaml.safe_load(f)
        vref_target = specs["vref"]["target_V"]
        tol = specs["vref"]["tolerance_V"]
    except Exception:
        vref_target = 1.2
        tol = 0.01

    for entry in result.history:
        v = entry.get("vref_V")
        if v is not None:
            err = abs(v - vref_target)
            best_so_far = min(best_so_far, err)
        errors.append(best_so_far if best_so_far < float("inf") else float("nan"))

    iters = list(range(1, len(errors) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, errors, marker="o", markersize=3, linewidth=1.5, label="Best |Vref − target|")
    ax.axhline(tol, color="red", linestyle="--", linewidth=1,
               label=f"Tolerance (±{tol*1000:.0f} mV)")
    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("|Vref − target| [V]")
    ax.set_title("Bayesian Optimization Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = results_dir / "bo_convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Convergence plot saved to %s", plot_path)


def _plot_comparison(summary: dict[str, Any], results_dir: Path) -> None:
    """Bar chart comparing BO vs. brute-force sweep simulation counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = summary.get("sweep", {})
    bo = summary.get("bayesian_optimization", {})

    labels = ["Brute-force\nsweep", "Bayesian\noptimization"]
    n_sims = [sweep.get("n_samples", 0), bo.get("n_simulations", 0)]
    n_pass = [sweep.get("n_spec_pass", 0), bo.get("n_spec_pass", 0)]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_total = ax.bar([i - width / 2 for i in x], n_sims, width,
                        label="Total simulations", color="steelblue", alpha=0.8)
    bars_pass = ax.bar([i + width / 2 for i in x], n_pass, width,
                       label="Spec-pass count", color="seagreen", alpha=0.8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Simulation Budget: Brute-Force vs. Bayesian Optimization")
    ax.legend()

    for bar in bars_total:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars_pass:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = results_dir / "exp01_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Comparison plot saved to %s", plot_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: run BO experiment and produce exp01_summary.json + plots.

    Usage::

        python -m ml.optimize [--budget 30] [--n-init 10] [--brute-force-n 50]
                              [--results-dir results] [--seed 42]
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization experiment (Exp 01)."
    )
    parser.add_argument("--budget", type=int, default=30,
                        help="Maximum ngspice calls for the BO run.")
    parser.add_argument("--n-init", type=int, default=10,
                        help="LHS initialisation points before BO starts.")
    parser.add_argument("--brute-force-n", type=int, default=50,
                        help="Number of random samples for the brute-force baseline.")
    parser.add_argument("--results-dir", default="results",
                        help="Output directory for summary JSON and plots.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Create runner (real ngspice or analytic synthetic fallback)
    # -----------------------------------------------------------------------
    using_synthetic = False
    try:
        from bandgap.runner import BandgapRunner, _find_ngspice  # noqa: F401
        _find_ngspice()
        runner: Any = BandgapRunner()
        logger.info("Using real ngspice runner.")
    except FileNotFoundError:
        runner = SyntheticBandgapRunner(specs_file=SPECS_FILE, seed=args.seed)
        using_synthetic = True
        logger.warning(
            "ngspice not found — using analytical synthetic runner. "
            "Results illustrate the BO algorithm; they are NOT silicon-verified."
        )

    # -----------------------------------------------------------------------
    # 2. Bayesian Optimization run
    # -----------------------------------------------------------------------
    logger.info("Starting Bayesian Optimization (budget=%d, n_init=%d)…",
                args.budget, args.n_init)
    opt = BayesianOptimizer(
        runner=runner,
        budget=args.budget,
        n_init=args.n_init,
        specs_file=SPECS_FILE,
        results_dir=results_dir,
    )
    bo_result = opt.run(seed=args.seed)

    # -----------------------------------------------------------------------
    # 3. Brute-force baseline sweep
    # -----------------------------------------------------------------------
    logger.info("Running brute-force baseline (%d random samples)…", args.brute_force_n)
    rng = np.random.default_rng(args.seed + 1)
    bf_samples = _make_lhs_samples(n_samples=args.brute_force_n, rng=rng)

    import pandas as pd
    bf_rows = []
    for params in bf_samples:
        sim = runner.run(params)
        bf_rows.append({
            **params,
            "vref_V": sim.get("vref_V"),
            "iq_uA": sim.get("iq_uA"),
            "error": sim.get("error") or "",
        })
    bf_df = pd.DataFrame(bf_rows)

    with open(SPECS_FILE) as f:
        specs = yaml.safe_load(f)
    vref_target = specs["vref"]["target_V"]
    vref_tol = specs["vref"]["tolerance_V"]

    bf_valid = bf_df["vref_V"].notna()
    bf_pass_mask = bf_valid & (bf_df["vref_V"].sub(vref_target).abs() <= vref_tol)
    bf_n_pass = int(bf_pass_mask.sum())
    bf_pass_rate = bf_n_pass / args.brute_force_n

    # Index (1-based) of first passing sample in random sweep order
    passing_indices = bf_pass_mask[bf_pass_mask].index.tolist()
    bf_first_pass_at = int(passing_indices[0]) + 1 if passing_indices else args.brute_force_n

    # -----------------------------------------------------------------------
    # 4. Load surrogate metrics (produced by ml.surrogate main)
    # -----------------------------------------------------------------------
    surrogate_metrics: dict[str, Any] = {}
    surrogate_metrics_path = results_dir / "surrogate_metrics.json"
    if surrogate_metrics_path.exists():
        with open(surrogate_metrics_path) as f:
            surrogate_metrics = json.load(f)
        logger.info("Loaded surrogate metrics from %s", surrogate_metrics_path)

    # -----------------------------------------------------------------------
    # 5. Build and save exp01_summary.json
    # -----------------------------------------------------------------------
    simulation_reduction_pct = (
        round(100.0 * (1.0 - bo_result.n_simulations / args.brute_force_n), 1)
        if args.brute_force_n > 0 else 0.0
    )

    summary: dict[str, Any] = {
        "experiment": "exp01",
        "description": "Bandgap surrogate vs. brute-force sweep",
        "timestamp": datetime.now().isoformat(),
        "using_synthetic_runner": using_synthetic,
        "sweep": {
            "n_samples": args.brute_force_n,
            "n_valid": int(bf_valid.sum()),
            "n_spec_pass": bf_n_pass,
            "spec_pass_rate": round(bf_pass_rate, 4),
            "first_pass_at_sim": bf_first_pass_at,
        },
        "bayesian_optimization": {
            "budget": args.budget,
            "n_init": args.n_init,
            "n_simulations": bo_result.n_simulations,
            "n_spec_pass": bo_result.n_spec_pass,
            "spec_pass_rate": round(bo_result.spec_pass_rate(), 4),
            "best_params": bo_result.best_params,
            "best_vref_V": bo_result.best_vref_V,
            "top_candidates": bo_result.top_k_candidates(k=3),
        },
        "surrogate": surrogate_metrics,
        "comparison": {
            "brute_force_simulations": args.brute_force_n,
            "bo_simulations": bo_result.n_simulations,
            "simulation_reduction_pct": simulation_reduction_pct,
            "bo_spec_pass_rate": round(bo_result.spec_pass_rate(), 4),
            "brute_force_spec_pass_rate": round(bf_pass_rate, 4),
        },
    }

    summary_path = results_dir / "exp01_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # -----------------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------------
    _plot_convergence(bo_result, results_dir)
    _plot_comparison(summary, results_dir)

    # -----------------------------------------------------------------------
    # 7. Console summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Experiment 01 — Bandgap: Surrogate vs. Brute-Force Sweep")
    print(f"{'='*60}")
    if using_synthetic:
        print("  [NOTE] ngspice not installed — results use analytic model.")
    print(f"  Brute-force sweep : {args.brute_force_n} sims, "
          f"{bf_n_pass} spec-pass ({100*bf_pass_rate:.0f}%)")
    print(f"  Bayesian opt      : {bo_result.n_simulations} sims, "
          f"{bo_result.n_spec_pass} spec-pass "
          f"({100*bo_result.spec_pass_rate():.0f}%)")
    print(f"  Simulation savings: {simulation_reduction_pct:.0f}%")
    if bo_result.best_vref_V is not None:
        print(f"  Best Vref         : {bo_result.best_vref_V:.4f} V")
    candidates = bo_result.top_k_candidates(k=3)
    if candidates:
        print(f"\n  Top {len(candidates)} candidate design(s) "
              f"(spec-passing, sorted by |Vref − target|):")
        for i, c in enumerate(candidates, 1):
            p = c["params"]
            print(f"    [{i}] Vref={c['vref_V']:.4f} V  "
                  f"N={p.get('N')}  R1={p.get('R1'):.0f}  R2={p.get('R2'):.0f}  "
                  f"W_P={p.get('W_P', float('nan')):.2e}  L_P={p.get('L_P', float('nan')):.2e}")
    else:
        print("  No spec-passing candidates found — try increasing budget.")
    print(f"  Output JSON       : {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
