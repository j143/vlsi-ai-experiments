#!/usr/bin/env python3
"""
examples/run_full_pipeline.py
================================
End-to-end demonstration of the ML-assisted bandgap design pipeline:

  1. Load dataset (analytical or real SPICE CSV).
  2. Train a GP surrogate model.
  3. Evaluate surrogate quality (MAE, coverage, calibration).
  4. Run Bayesian optimization.
  5. Report best design, spec pass rate, and simulation savings.

This script works with NO external dependencies beyond the repo's
requirements.txt — no ngspice, no PDK install needed.

Usage::

    # Use the bundled analytical dataset
    python examples/run_full_pipeline.py

    # Use a real SPICE dataset
    python examples/run_full_pipeline.py --dataset datasets/bandgap_sweep_*.csv

    # Larger budget
    python examples/run_full_pipeline.py --budget 50 --n-init 15
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.surrogate import GaussianProcessSurrogate, evaluate_surrogate  # noqa: E402
from ml.optimize import BayesianOptimizer  # noqa: E402
from ml.optimize import SyntheticBandgapRunner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["N", "R1", "R2", "W_P", "L_P"]
TARGET = "vref_V"
VREF_TARGET = 1.200
TOLERANCE = 0.010


def load_dataset(path: str | None) -> pd.DataFrame:
    """Load dataset from CSV, or generate the analytical one.

    Applies row filtering:
      - Drops rows where the ``error`` column is non-empty (failed simulations).
      - If a ``corner`` column is present, logs per-corner counts but keeps all
        corners so the surrogate learns process variation too.
    """
    if path and Path(path).exists():
        df = pd.read_csv(path)
        logger.info("Loaded dataset from %s (%d rows)", path, len(df))
    else:
        # Prefer the pre-generated real dataset if it exists
        real_path = Path(__file__).parent.parent / "datasets" / "bandgap_sweep_real_sky130.csv"
        if real_path.exists():
            df = pd.read_csv(str(real_path))
            logger.info("Loaded real dataset from %s (%d rows)", real_path, len(df))
        else:
            demo_path = Path(__file__).parent / "demo_dataset.csv"
            if demo_path.exists():
                df = pd.read_csv(str(demo_path))
                logger.info("Loaded demo dataset (%d rows)", len(df))
            else:
                logger.info("No dataset found — generating analytical dataset...")
                from examples.generate_reference_dataset import generate_dataset
                df = generate_dataset(n_samples=200, seed=42)
                df.to_csv(str(demo_path), index=False)
                logger.info("Generated and saved demo dataset (%d rows)", len(df))

    # Drop failed simulation rows (keeps all corners)
    if "error" in df.columns:
        before = len(df)
        df = df[df["error"].isna() | (df["error"] == "")].reset_index(drop=True)
        if len(df) < before:
            logger.info("Dropped %d failed-simulation rows.", before - len(df))

    if "corner" in df.columns:
        logger.info("Corners in dataset: %s", df["corner"].value_counts().to_dict())

    return df


def step1_explore(df: pd.DataFrame) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("STEP 1: Dataset Exploration")
    print("=" * 60)
    print(f"  Rows:         {len(df)}")
    print(f"  Features:     {FEATURES}")
    print(f"  Target:       {TARGET}")

    if "corner" in df.columns:
        corners = df["corner"].value_counts().to_dict()
        print(f"  Corners:      {corners}")

    if TARGET in df.columns:
        valid = df[TARGET].notna()
        print(f"  Valid sims:   {valid.sum()}/{len(df)}")
        print(f"  Vref range:   {df.loc[valid, TARGET].min():.4f} – "
              f"{df.loc[valid, TARGET].max():.4f} V")
        errs = (df.loc[valid, TARGET] - VREF_TARGET).abs() * 1000
        print(f"  |Err| range:  {errs.min():.2f} – {errs.max():.2f} mV")
        spec_pass = (errs <= TOLERANCE * 1000).sum()
        print(f"  Vref spec:    {spec_pass}/{valid.sum()} "
              f"({spec_pass / valid.sum():.1%}) pass")

    if "tc_ppm_C" in df.columns:
        tc = df["tc_ppm_C"].dropna()
        print(f"  TC range:     {tc.min():.1f} – {tc.max():.1f} ppm/°C")
        tc_pass = (tc <= 20).sum()
        print(f"  TC spec:      {tc_pass}/{len(tc)} ({tc_pass/len(tc):.1%}) ≤ 20 ppm/°C")

    if "psrr_dB" in df.columns:
        psrr = df["psrr_dB"].dropna()
        print(f"  PSRR range:   {psrr.min():.1f} – {psrr.max():.1f} dB")


def step2_train_surrogate(df: pd.DataFrame):
    """Train GP surrogate and evaluate."""
    print("\n" + "=" * 60)
    print("STEP 2: Train & Evaluate GP Surrogate")
    print("=" * 60)

    valid = df[TARGET].notna()
    X = df.loc[valid, FEATURES].values
    y = df.loc[valid, TARGET].values

    # 80/20 train/test split
    n_train = int(0.8 * len(X))
    idx = np.random.default_rng(42).permutation(len(X))
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]

    model = GaussianProcessSurrogate(n_restarts=3)
    model.fit(X_train, y_train)
    metrics = evaluate_surrogate(model, X_test, y_test)

    print(f"  Training set:  {n_train} points")
    print(f"  Test set:      {len(X_test)} points")
    print(f"  MAE:           {metrics['mae']:.4f} V ({metrics['mae']*1000:.2f} mV)")
    print(f"  RMSE:          {metrics['rmse']:.4f} V")
    print(f"  R²:            {metrics['r2']:.4f}")
    if "coverage_90" in metrics:
        print(f"  90% coverage:  {metrics['coverage_90']:.1%} "
              "(ideal ≈ 0.90)")

    return model, metrics


def step3_optimize():
    """Run Bayesian optimization with synthetic runner."""
    print("\n" + "=" * 60)
    print("STEP 3: Bayesian Optimization")
    print("=" * 60)

    runner = SyntheticBandgapRunner()
    opt = BayesianOptimizer(runner=runner, budget=30, n_init=10)
    result = opt.run(seed=42)

    vref = result.best_vref_V
    err_mV = abs(vref - VREF_TARGET) * 1000 if vref else float("inf")
    spec_ok = err_mV <= TOLERANCE * 1000

    print(f"  Simulations:   {result.n_simulations}")
    print(f"  Spec pass:     {result.n_spec_pass}/{result.n_simulations} "
          f"({result.spec_pass_rate():.0%})")
    print(f"  Best Vref:     {vref:.4f} V"
          if vref else "  Best Vref:     N/A")
    print(f"  Best |err|:    {err_mV:.2f} mV")
    print(f"  Spec met:      {'YES' if spec_ok else 'NO'}")
    print(f"  Best params:   {result.best_params}")

    return result


def step4_report(metrics: dict, opt_result, out_path: str) -> None:
    """Save a JSON report combining surrogate and optimizer results."""
    print("\n" + "=" * 60)
    print("STEP 4: Report")
    print("=" * 60)

    report = {
        "surrogate": {
            "mae_V": metrics["mae"],
            "rmse_V": metrics["rmse"],
            "r2": metrics["r2"],
        },
        "optimizer": {
            "n_simulations": opt_result.n_simulations,
            "n_spec_pass": opt_result.n_spec_pass,
            "spec_pass_rate": opt_result.spec_pass_rate(),
            "best_vref_V": opt_result.best_vref_V,
            "best_params": opt_result.best_params,
        },
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {out_path}")
    print(
        "\n  Summary: Surrogate MAE = "
        f"{metrics['mae']*1000:.2f} mV, "
        f"Optimizer spec-pass = {opt_result.spec_pass_rate():.0%}, "
        f"Best Vref = {opt_result.best_vref_V:.4f} V"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Full ML pipeline demo for bandgap design"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to CSV dataset (default: use analytical demo data)"
    )
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument(
        "--out", type=str, default="results/example_pipeline_report.json"
    )
    args = parser.parse_args()

    # Step 1: Load / explore
    df = load_dataset(args.dataset)
    step1_explore(df)

    # Step 2: Train surrogate
    model, metrics = step2_train_surrogate(df)

    # Step 3: Optimize
    opt_result = step3_optimize()

    # Step 4: Report
    step4_report(metrics, opt_result, args.out)

    print("\nDone. All pipeline steps completed successfully.")


if __name__ == "__main__":
    main()
