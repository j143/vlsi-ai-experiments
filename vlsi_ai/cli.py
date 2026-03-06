"""
vlsi_ai/cli.py — vlsi-ai Command-Line Interface
=================================================
Provides the ``vlsi-ai`` entry-point with three sub-commands:

  vlsi-ai sweep    — run ngspice sweeps and log results to a CSV
  vlsi-ai optimize — train GP surrogate and run Bayesian optimisation
  vlsi-ai demo     — end-to-end: sweep (or synthetic) → surrogate → BO vs grid → report

Usage examples::

    vlsi-ai sweep --netlist bandgap/netlists/sky130_brokaw.sp \\
                  --out datasets/sky130_bandgap_real.csv --n-samples 80

    vlsi-ai optimize --dataset datasets/sky130_bandgap_real.csv \\
                     --budget 30 --out results/bo_run_$(date +%s)

    vlsi-ai demo
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure the repo root is importable when the CLI is invoked without pip-install
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
SPECS_FILE = _REPO_ROOT / "bandgap" / "specs.yaml"
FEATURES = ["N", "R1", "R2", "W_P", "L_P"]


def _get_runner(netlist: str | None = None) -> Any:
    """Return a BandgapRunner if ngspice is present, else SyntheticBandgapRunner."""
    from ml.optimize import SyntheticBandgapRunner

    try:
        from bandgap.runner import BandgapRunner, _find_ngspice
        _find_ngspice()
        kwargs = {}
        if netlist:
            kwargs["netlist_template"] = netlist
        return BandgapRunner(**kwargs)
    except (ImportError, FileNotFoundError):
        logger.warning(
            "ngspice not found — using analytical synthetic runner. "
            "Results illustrate the algorithm; they are NOT silicon-verified."
        )
        return SyntheticBandgapRunner(specs_file=SPECS_FILE)


# ---------------------------------------------------------------------------
# sweep sub-command
# ---------------------------------------------------------------------------

def cmd_sweep(args: argparse.Namespace) -> int:
    """Run an ngspice parameter sweep and write results to a CSV file."""
    import numpy as np
    import pandas as pd
    import yaml
    from data_gen.sweep_bandgap import _make_lhs_samples, _make_grid_samples

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_path = Path(args.out)
    # If the user supplied a directory, generate a timestamped filename.
    if out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_path / f"bandgap_sweep_{timestamp}.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    runner = _get_runner(netlist=args.netlist)

    with open(SPECS_FILE) as f:
        specs = yaml.safe_load(f)
    vref_target = specs["vref"]["target_V"]

    if args.mode == "grid":
        samples = _make_grid_samples(n_per_dim=args.n_samples)
        logger.info("Generated %d grid samples (mode=grid).", len(samples))
    else:
        rng = np.random.default_rng(seed=args.seed)
        samples = _make_lhs_samples(n_samples=args.n_samples, rng=rng)
        logger.info("Generated %d LHS samples (mode=lhs).", len(samples))

    rows = []
    ngspice_ok = runner.is_ngspice_available()
    if not ngspice_ok:
        logger.warning(
            "ngspice is not available — rows will use the analytical Brokaw model."
        )

    for idx, params in enumerate(samples):
        t0 = time.perf_counter()
        result = runner.run(params)
        elapsed = time.perf_counter() - t0

        row: dict[str, Any] = {**params}
        row["vref_V"] = result.get("vref_V")
        row["tc_ppm_C"] = result.get("tc_ppm_C")
        row["psrr_dB"] = result.get("psrr_dB")
        row["iq_uA"] = result.get("iq_uA")
        # Convenience: error from 1.2 V target [mV]
        vref = result.get("vref_V")
        row["err_from_target_mV"] = (
            round(abs(vref - vref_target) * 1000, 3) if vref is not None else None
        )
        for spec_name, passed in result.get("spec_checks", {}).items():
            row[f"spec_{spec_name}_pass"] = passed
        row["sim_time_s"] = round(elapsed, 4)
        row["error"] = result.get("error") or ""
        rows.append(row)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            logger.info("Progress: %d / %d", idx + 1, len(samples))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows → %s", len(df), out_path)
    print(f"Sweep complete. Output: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# optimize sub-command
# ---------------------------------------------------------------------------

def cmd_optimize(args: argparse.Namespace) -> int:
    """Train GP surrogate and run Bayesian optimisation on an existing dataset."""
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _get_runner()

    from ml.optimize import BayesianOptimizer

    logger.info("Starting Bayesian Optimisation (budget=%d)…", args.budget)
    opt = BayesianOptimizer(
        runner=runner,
        budget=args.budget,
        n_init=min(args.n_init, args.budget),
        specs_file=SPECS_FILE,
        results_dir=out_dir,
    )
    result = opt.run(seed=args.seed)

    with open(SPECS_FILE) as f:
        specs = yaml.safe_load(f)
    vref_target = specs["vref"]["target_V"]
    vref_tol = specs["vref"]["tolerance_V"]

    print(f"\n{'='*55}")
    print("vlsi-ai optimize — Bayesian Optimisation Result")
    print(f"{'='*55}")
    print(f"  Budget used     : {result.n_simulations} simulations")
    print(f"  Spec-pass count : {result.n_spec_pass} "
          f"({100*result.spec_pass_rate():.0f}%)")
    if result.best_vref_V is not None:
        err_mV = abs(result.best_vref_V - vref_target) * 1000
        print(f"  Best Vref       : {result.best_vref_V:.4f} V "
              f"(target {vref_target:.3f} V ± {vref_tol*1000:.0f} mV, "
              f"err={err_mV:.1f} mV)")
    print(f"  Output dir      : {out_dir}")
    print(f"{'='*55}\n")
    return 0


# ---------------------------------------------------------------------------
# demo sub-command
# ---------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end demo: sweep → surrogate → BO vs grid → Markdown report."""
    import numpy as np
    import pandas as pd
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = Path(args.out)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load or generate dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset) if args.dataset else None

    if dataset_path and dataset_path.exists():
        logger.info("Loading dataset from %s", dataset_path)
        df = pd.read_csv(dataset_path)
        valid_mask = (df["error"].isna() | (df["error"] == "")) & df["vref_V"].notna()
        df_valid = df[valid_mask]
        if len(df_valid) < 10:
            logger.warning("Fewer than 10 valid rows — switching to synthetic data.")
            df_valid = _generate_synthetic(args.n_samples)
    else:
        logger.info("No dataset provided — generating %d synthetic samples.", args.n_samples)
        df_valid = _generate_synthetic(args.n_samples)

    logger.info("Dataset: %d valid rows.", len(df_valid))

    # ------------------------------------------------------------------
    # 2. Train surrogate
    # ------------------------------------------------------------------
    from ml.surrogate import GaussianProcessSurrogate, evaluate_surrogate

    X = df_valid[FEATURES].values
    y = df_valid["vref_V"].values

    n = len(X)
    n_train = max(2, int(0.8 * n))
    idx = np.random.default_rng(args.seed).permutation(n)
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    logger.info("Training GP surrogate on %d samples…", n_train)
    surrogate = GaussianProcessSurrogate(n_restarts=3)
    surrogate.fit(X_train, y_train)

    surr_metrics: dict[str, Any] = {}
    if len(X_test) >= 2:
        surr_metrics = evaluate_surrogate(surrogate, X_test, y_test)

    # ------------------------------------------------------------------
    # 3. BO run
    # ------------------------------------------------------------------
    from ml.optimize import BayesianOptimizer

    runner = _get_runner()
    opt = BayesianOptimizer(
        runner=runner,
        budget=args.budget,
        n_init=min(args.n_init, args.budget),
        specs_file=SPECS_FILE,
        results_dir=results_dir,
    )
    logger.info("Running Bayesian optimisation (budget=%d)…", args.budget)
    t_bo_start = time.perf_counter()
    bo_result = opt.run(seed=args.seed)
    t_bo = time.perf_counter() - t_bo_start

    # ------------------------------------------------------------------
    # 4. Grid baseline (7×7×3 ≈ 147 points, but capped at grid_n^5)
    # ------------------------------------------------------------------
    from data_gen.sweep_bandgap import _make_grid_samples

    grid_samples = _make_grid_samples(n_per_dim=args.grid_n)
    logger.info("Running grid baseline (%d points)…", len(grid_samples))
    t_grid_start = time.perf_counter()

    with open(SPECS_FILE) as f:
        specs = yaml.safe_load(f)
    vref_target = specs["vref"]["target_V"]
    vref_tol = specs["vref"]["tolerance_V"]

    grid_pass = 0
    grid_best_err = float("inf")
    for gp in grid_samples:
        gr = runner.run(gp)
        v = gr.get("vref_V")
        if v is not None:
            err = abs(v - vref_target)
            if err <= vref_tol:
                grid_pass += 1
            grid_best_err = min(grid_best_err, err)
    t_grid = time.perf_counter() - t_grid_start

    # ------------------------------------------------------------------
    # 5. Write Markdown report
    # ------------------------------------------------------------------
    bo_best_err = (
        abs(bo_result.best_vref_V - vref_target)
        if bo_result.best_vref_V is not None else float("nan")
    )
    reduction_pct = (
        round(100.0 * (1.0 - bo_result.n_simulations / len(grid_samples)), 1)
        if grid_samples else 0.0
    )

    report_lines = [
        "# vlsi-ai Demo Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset",
        f"- Valid samples used: {len(df_valid)}",
        f"- Features: {', '.join(FEATURES)}",
        "",
        "## Surrogate (GP)",
    ]
    if surr_metrics:
        report_lines += [
            f"- R²: {surr_metrics.get('r2', float('nan')):.4f}",
            f"- MAE: {surr_metrics.get('mae', float('nan')):.4e} V",
            f"- RMSE: {surr_metrics.get('rmse', float('nan')):.4e} V",
        ]
    else:
        report_lines.append("- (too few test samples)")

    report_lines += [
        "",
        "## Comparison: Grid Search vs Bayesian Optimisation",
        "",
        "| Method | Simulations | Spec-Pass | Best Err (mV) | Wall-clock (s) |",
        "|--------|-------------|-----------|---------------|----------------|",
        f"| Grid ({args.grid_n}^5) | {len(grid_samples)} | {grid_pass} | "
        f"{grid_best_err*1000:.1f} | {t_grid:.1f} |",
        f"| BO (budget={args.budget}) | {bo_result.n_simulations} | "
        f"{bo_result.n_spec_pass} | {bo_best_err*1000:.1f} | {t_bo:.1f} |",
        "",
        f"**Simulation reduction:** {reduction_pct:.0f}% fewer simulations with BO.",
        "",
        "## Best Design (BO)",
        "",
    ]

    if bo_result.best_params:
        report_lines.append("| Parameter | Value |")
        report_lines.append("|-----------|-------|")
        for k, v in bo_result.best_params.items():
            report_lines.append(f"| {k} | {v} |")
        if bo_result.best_vref_V is not None:
            report_lines.append(f"| **Vref** | **{bo_result.best_vref_V:.4f} V** |")

    report_path = results_dir / "demo_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    logger.info("Report written to %s", report_path)

    # Also save a summary JSON for machine-readable consumption
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_rows": len(df_valid),
        "surrogate": surr_metrics,
        "grid": {
            "n_per_dim": args.grid_n,
            "n_simulations": len(grid_samples),
            "n_spec_pass": grid_pass,
            "best_err_V": grid_best_err,
            "wall_clock_s": round(t_grid, 2),
        },
        "bayesian_optimization": {
            "budget": args.budget,
            "n_simulations": bo_result.n_simulations,
            "n_spec_pass": bo_result.n_spec_pass,
            "best_vref_V": bo_result.best_vref_V,
            "best_err_V": bo_best_err if bo_result.best_vref_V is not None else None,
            "wall_clock_s": round(t_bo, 2),
        },
        "simulation_reduction_pct": reduction_pct,
    }
    summary_path = results_dir / "demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON written to %s", summary_path)

    print(f"\n{'='*55}")
    print("vlsi-ai demo — complete")
    print(f"{'='*55}")
    print(f"  Grid ({args.grid_n}^5={len(grid_samples)} sims) best err : "
          f"{grid_best_err*1000:.1f} mV, {grid_pass} pass")
    print(f"  BO (budget={args.budget}) best err       : "
          f"{bo_best_err*1000:.1f} mV, {bo_result.n_spec_pass} pass")
    print(f"  Simulation reduction                  : {reduction_pct:.0f}%")
    print(f"  Report  : {report_path}")
    print(f"  Summary : {summary_path}")
    print(f"{'='*55}\n")
    return 0


def _generate_synthetic(n: int):
    """Generate synthetic bandgap data using the analytical Brokaw model."""
    from ml.surrogate import _generate_synthetic_data
    return _generate_synthetic_data(n=n)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlsi-ai",
        description="VLSI AI Experiments — bandgap design automation toolkit.",
    )
    parser.add_argument(
        "--version", action="version", version="vlsi-ai 0.1.0"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ------------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------------
    p_sweep = sub.add_parser(
        "sweep",
        help="Run parameter sweep and log results to CSV.",
        description=(
            "Sweep the bandgap design space with ngspice (or the analytical "
            "fallback when ngspice is unavailable) and write results to a CSV file."
        ),
    )
    p_sweep.add_argument(
        "--netlist",
        default=str(_REPO_ROOT / "bandgap" / "netlists" / "bandgap_simple.sp"),
        help="Path to the SPICE netlist template.",
    )
    p_sweep.add_argument(
        "--out",
        default="datasets/sky130_bandgap_real.csv",
        help="Output CSV path (or directory — a timestamped file is created).",
    )
    p_sweep.add_argument(
        "--n-samples", type=int, default=80,
        help="Number of samples (LHS mode) or grid points per dimension (grid mode).",
    )
    p_sweep.add_argument(
        "--mode", choices=["lhs", "grid"], default="lhs",
        help="Sampling strategy: Latin-Hypercube (default) or regular grid.",
    )
    p_sweep.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    p_sweep.set_defaults(func=cmd_sweep)

    # ------------------------------------------------------------------
    # optimize
    # ------------------------------------------------------------------
    p_opt = sub.add_parser(
        "optimize",
        help="Run Bayesian optimisation on a sweep dataset.",
        description=(
            "Train a GP surrogate on a CSV dataset and run Bayesian "
            "Optimisation to find a spec-passing bandgap design."
        ),
    )
    p_opt.add_argument(
        "--dataset",
        default="datasets/sky130_bandgap_real.csv",
        help="Path to the sweep CSV produced by 'vlsi-ai sweep'.",
    )
    p_opt.add_argument(
        "--budget", type=int, default=30,
        help="Maximum number of simulator calls for the BO run.",
    )
    p_opt.add_argument(
        "--n-init", type=int, default=10,
        help="Number of LHS initialisation points before BO starts.",
    )
    p_opt.add_argument(
        "--out",
        default=f"results/bo_run_{int(time.time())}",
        help="Output directory for BO run JSON and plots.",
    )
    p_opt.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    p_opt.set_defaults(func=cmd_optimize)

    # ------------------------------------------------------------------
    # demo
    # ------------------------------------------------------------------
    p_demo = sub.add_parser(
        "demo",
        help="End-to-end demo: sweep → surrogate → BO vs grid → report.",
        description=(
            "Run the full pipeline: generate (or load) a dataset, train a GP "
            "surrogate, run Bayesian Optimisation, compare against grid search, "
            "and write a Markdown/JSON report to the output directory."
        ),
    )
    p_demo.add_argument(
        "--dataset",
        default=None,
        help="CSV dataset path. If absent, synthetic data is generated.",
    )
    p_demo.add_argument(
        "--n-samples", type=int, default=100,
        help="Synthetic samples to generate when --dataset is not provided.",
    )
    p_demo.add_argument(
        "--budget", type=int, default=30,
        help="BO simulation budget.",
    )
    p_demo.add_argument(
        "--n-init", type=int, default=10,
        help="LHS initialisation points for BO.",
    )
    p_demo.add_argument(
        "--grid-n", type=int, default=3,
        help="Grid points per dimension for the grid baseline (total = grid_n^5).",
    )
    p_demo.add_argument(
        "--out", default="results",
        help="Output directory for the report and JSON summary.",
    )
    p_demo.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    p_demo.set_defaults(func=cmd_demo)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Main entry point for the ``vlsi-ai`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
