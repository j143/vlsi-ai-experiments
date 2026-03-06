#!/usr/bin/env python3
"""
examples/bo_vs_grid.py — BO vs Grid-Search Benchmark
======================================================
Compares Bayesian Optimisation (BO) against a brute-force grid search for
finding a spec-passing SKY130 Brokaw bandgap design.

Metrics reported:
  - Best spec deviation [mV] from the 1.2 V target
  - Number of simulator calls used
  - Wall-clock time [s]

The script works WITHOUT ngspice installed: it falls back to the analytical
Brokaw surrogate (``SyntheticBandgapRunner``) automatically.

Usage::

    python examples/bo_vs_grid.py
    python examples/bo_vs_grid.py --grid-n 4 --bo-budget 40 --seed 7
    python examples/bo_vs_grid.py --out results/my_benchmark
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_gen.sweep_bandgap import _make_grid_samples, PARAM_SPACE  # noqa: E402
from ml.optimize import BayesianOptimizer, SyntheticBandgapRunner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
SPECS_FILE = _REPO_ROOT / "bandgap" / "specs.yaml"
N_DIMS = len(PARAM_SPACE)  # number of design variables (5 for SKY130 Brokaw)


def _get_runner(seed: int = 42):
    """Return a real BandgapRunner or fall back to the synthetic runner."""
    try:
        from bandgap.runner import BandgapRunner, _find_ngspice
        _find_ngspice()
        logger.info("Using real ngspice runner.")
        return BandgapRunner()
    except (ImportError, FileNotFoundError):
        logger.warning(
            "ngspice not found — using analytical SyntheticBandgapRunner. "
            "Results illustrate the algorithm; NOT silicon-verified."
        )
        return SyntheticBandgapRunner(specs_file=SPECS_FILE, seed=seed)


def run_grid_search(runner, grid_n: int, vref_target: float, vref_tol: float):
    """Run a brute-force grid search over the design space.

    Parameters
    ----------
    runner:
        Runner instance (real or synthetic).
    grid_n:
        Number of grid points per dimension.
    vref_target:
        Target Vref [V].
    vref_tol:
        Tolerance [V].

    Returns
    -------
    dict
        Summary with ``n_simulations``, ``n_spec_pass``, ``best_err_V``,
        ``wall_clock_s``, ``first_pass_at``.
    """
    samples = _make_grid_samples(n_per_dim=grid_n)
    logger.info("Grid search: %d points (%d dims × %d pts).",
                len(samples), N_DIMS, grid_n)

    n_pass = 0
    best_err = float("inf")
    first_pass_at = None

    t0 = time.perf_counter()
    for i, params in enumerate(samples):
        result = runner.run(params)
        vref = result.get("vref_V")
        if vref is not None:
            err = abs(vref - vref_target)
            if err <= vref_tol:
                n_pass += 1
                if first_pass_at is None:
                    first_pass_at = i + 1
            best_err = min(best_err, err)
    elapsed = time.perf_counter() - t0

    return {
        "n_per_dim": grid_n,
        "n_simulations": len(samples),
        "n_spec_pass": n_pass,
        "spec_pass_rate": n_pass / len(samples) if samples else 0.0,
        "best_err_V": best_err,
        "best_err_mV": round(best_err * 1000, 2),
        "first_pass_at": first_pass_at,
        "wall_clock_s": round(elapsed, 2),
    }


def run_bayesian_opt(runner, budget: int, n_init: int, seed: int, results_dir: Path):
    """Run Bayesian Optimisation.

    Parameters
    ----------
    runner:
        Runner instance.
    budget:
        Maximum number of simulator calls.
    n_init:
        LHS initialisation budget.
    seed:
        Random seed.
    results_dir:
        Directory to save BO run artefacts.

    Returns
    -------
    dict
        Summary with ``n_simulations``, ``n_spec_pass``, ``best_err_V``,
        ``wall_clock_s``.
    """
    import yaml

    with open(SPECS_FILE) as f:
        specs = yaml.safe_load(f)
    vref_target = specs["vref"]["target_V"]

    opt = BayesianOptimizer(
        runner=runner,
        budget=budget,
        n_init=n_init,
        specs_file=SPECS_FILE,
        results_dir=results_dir,
    )

    t0 = time.perf_counter()
    result = opt.run(seed=seed)
    elapsed = time.perf_counter() - t0

    best_err = (
        abs(result.best_vref_V - vref_target)
        if result.best_vref_V is not None else float("inf")
    )

    return {
        "budget": budget,
        "n_init": n_init,
        "n_simulations": result.n_simulations,
        "n_spec_pass": result.n_spec_pass,
        "spec_pass_rate": result.spec_pass_rate(),
        "best_vref_V": result.best_vref_V,
        "best_err_V": best_err,
        "best_err_mV": round(best_err * 1000, 2),
        "wall_clock_s": round(elapsed, 2),
    }


def write_summary(grid_res: dict, bo_res: dict, out_dir: Path) -> None:
    """Write a CSV and Markdown summary comparing grid vs BO results."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute simulation reduction
    n_grid = grid_res["n_simulations"]
    n_bo = bo_res["n_simulations"]
    reduction_pct = round(100.0 * (1 - n_bo / n_grid), 1) if n_grid > 0 else 0.0

    # ---- CSV ----
    csv_path = out_dir / "bo_vs_grid_summary.csv"
    rows = [
        {
            "method": f"Grid ({grid_res['n_per_dim']}^{N_DIMS})",
            "n_simulations": n_grid,
            "n_spec_pass": grid_res["n_spec_pass"],
            "best_err_mV": grid_res["best_err_mV"],
            "wall_clock_s": grid_res["wall_clock_s"],
        },
        {
            "method": f"BO (budget={bo_res['budget']})",
            "n_simulations": n_bo,
            "n_spec_pass": bo_res["n_spec_pass"],
            "best_err_mV": bo_res["best_err_mV"],
            "wall_clock_s": bo_res["wall_clock_s"],
        },
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV summary → %s", csv_path)

    # ---- JSON ----
    json_path = out_dir / "bo_vs_grid_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "grid": grid_res,
        "bayesian_optimization": bo_res,
        "comparison": {
            "simulation_reduction_pct": reduction_pct,
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("JSON summary → %s", json_path)

    # ---- Markdown ----
    md_lines = [
        "# BO vs Grid-Search Benchmark",
        "",
        f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results",
        "",
        "| Method | Simulations | Spec-Pass | Best Err (mV) | Wall-clock (s) |",
        "|--------|-------------|-----------|---------------|----------------|",
        f"| Grid ({grid_res['n_per_dim']}^{N_DIMS}) "
        f"| {n_grid} | {grid_res['n_spec_pass']} "
        f"| {grid_res['best_err_mV']:.1f} | {grid_res['wall_clock_s']:.1f} |",
        f"| BO (budget={bo_res['budget']}) "
        f"| {n_bo} | {bo_res['n_spec_pass']} "
        f"| {bo_res['best_err_mV']:.1f} | {bo_res['wall_clock_s']:.1f} |",
        "",
        f"**Simulation reduction:** {reduction_pct:.0f}% fewer simulations with BO.",
        "",
    ]
    md_path = out_dir / "bo_vs_grid_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    logger.info("Markdown summary → %s", md_path)

    # ---- Console ----
    print(f"\n{'='*60}")
    print("BO vs Grid-Search Benchmark")
    print(f"{'='*60}")
    print(f"  Grid ({grid_res['n_per_dim']}^{N_DIMS}={n_grid} sims):")
    print(f"    best err = {grid_res['best_err_mV']:.1f} mV, "
          f"{grid_res['n_spec_pass']} spec-pass, "
          f"{grid_res['wall_clock_s']:.1f}s")
    print(f"  BO (budget={bo_res['budget']}, {n_bo} sims):")
    print(f"    best err = {bo_res['best_err_mV']:.1f} mV, "
          f"{bo_res['n_spec_pass']} spec-pass, "
          f"{bo_res['wall_clock_s']:.1f}s")
    print(f"  Simulation reduction: {reduction_pct:.0f}%")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Bayesian Optimisation vs grid search for bandgap design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grid-n", type=int, default=3,
        help="Grid points per dimension (total = grid_n^5).",
    )
    parser.add_argument(
        "--bo-budget", type=int, default=30,
        help="Simulation budget for the BO run.",
    )
    parser.add_argument(
        "--n-init", type=int, default=10,
        help="LHS initialisation points before BO starts.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--out", default="results/bo_vs_grid",
        help="Output directory for CSV / Markdown summary.",
    )
    args = parser.parse_args(argv)

    import yaml
    with open(SPECS_FILE) as f:
        specs = yaml.safe_load(f)
    vref_target = specs["vref"]["target_V"]
    vref_tol = specs["vref"]["tolerance_V"]

    runner = _get_runner(seed=args.seed)
    out_dir = Path(args.out)

    # Grid search
    logger.info("=== Grid Search ===")
    grid_res = run_grid_search(runner, args.grid_n, vref_target, vref_tol)

    # Bayesian Optimisation
    logger.info("=== Bayesian Optimisation ===")
    bo_res = run_bayesian_opt(
        runner, args.bo_budget, args.n_init, args.seed, out_dir
    )

    # Summary
    write_summary(grid_res, bo_res, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
