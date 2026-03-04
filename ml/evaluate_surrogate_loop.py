"""
ml/evaluate_surrogate_loop.py — Full ML surrogate evaluation loop
==================================================================
Demonstrates whether a surrogate model can reduce SPICE calls by ≥ 50% for a
fixed bandgap spec set.  The loop:

  1. Generates a labelled dataset (physics-proxy when ngspice is absent,
     real ngspice simulations when --use-ngspice is passed).
  2. Splits into 80 % train / 20 % held-out test.
  3. Trains a GaussianProcessSurrogate and a RandomForestSurrogate per output.
  4. Evaluates MAE, RMSE, R² per output spec on the held-out set.
  5. Estimates speedup: surrogate inference time vs. per-sample sim time.
  6. Prints a metric table and saves a bar-chart to results/surrogate_eval/.

Usage::

    python ml/evaluate_surrogate_loop.py
    python ml/evaluate_surrogate_loop.py --n-samples 200 --out results/surrogate_eval
    python ml/evaluate_surrogate_loop.py --use-ngspice   # requires ngspice on PATH

Speedup definition
------------------
    speedup = mean_sim_time_per_sample / mean_surrogate_predict_time_per_sample

A speedup ≥ 2× (50 % SPICE-call reduction) is the target of this experiment.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_gen.sweep_bandgap import _make_lhs_samples, run_sweep  # noqa: E402
from ml.surrogate import (  # noqa: E402
    GaussianProcessSurrogate,
    RandomForestSurrogate,
    evaluate_surrogate,
)

logger = logging.getLogger(__name__)

# Design-variable feature columns (must match PARAM_SPACE in sweep_bandgap.py)
FEATURES = ["N", "R1", "R2", "W_P", "L_P"]

# SPICE output columns to model
TARGETS = ["vref_V", "iq_uA"]

# Conservative per-sample ngspice wall-clock estimate (seconds) used when
# real simulation times are unavailable (synthetic dataset path).
_SPICE_TIME_PER_SAMPLE_S = 1.0


def generate_synthetic_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate a physics-informed synthetic bandgap dataset.

    Uses the Brokaw bandgap approximation::

        Vref ≈ Vbe + (R2 / R1) * VT * ln(N)

    where Vbe ≈ 0.65 V and VT ≈ 26 mV at 300 K.  Quiescent current::

        Iq ≈ VDD / (R1 + R2)   [simplified]

    Small Gaussian noise mimics SPICE variability from parasitics and
    placeholder model inaccuracies.

    Parameters
    ----------
    n_samples : int
        Number of design points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: N, R1, R2, W_P, L_P, vref_V, iq_uA, sim_time_s
    """
    rng = np.random.default_rng(seed)
    samples = _make_lhs_samples(n_samples=n_samples, rng=rng)

    VT = 0.02585   # thermal voltage at ~300 K  [V]
    Vbe = 0.650    # approx base-emitter voltage [V]
    VDD = 1.8      # nominal supply              [V]

    rows = []
    for params in samples:
        N = params["N"]
        R1 = params["R1"]
        R2 = params["R2"]

        # Brokaw bandgap reference voltage
        vref = Vbe + (R2 / R1) * VT * np.log(N)
        vref += rng.normal(0, 2e-3)   # ±2 mV noise

        # Quiescent current (simplified)
        iq_uA = (VDD / (R1 + R2)) * 1e6
        iq_uA = max(float(iq_uA) + float(rng.normal(0, 0.5)), 0.1)

        row = dict(params)
        row["vref_V"] = float(vref)
        row["iq_uA"] = float(iq_uA)
        row["sim_time_s"] = _SPICE_TIME_PER_SAMPLE_S
        rows.append(row)

    return pd.DataFrame(rows)


def build_and_evaluate(
    df: pd.DataFrame,
    out_dir: Path,
    test_frac: float = 0.20,
    seed: int = 42,
) -> dict:
    """Train surrogates and evaluate on a held-out test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with FEATURES + TARGETS columns.
    out_dir : Path
        Directory to write the plot and summary.
    test_frac : float
        Fraction of rows reserved for the held-out test set.
    seed : int
        Random seed for the train/test split.

    Returns
    -------
    dict
        Nested metrics::

            {
              "GP/vref_V": {"mae": ..., "rmse": ..., "r2": ..., "n_train": ..., "n_test": ...},
              "RF/vref_V": {...},
              ...
              "speedup": {"speedup_x": ..., "spice_calls_saved_pct": ..., ...},
            }
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_all = df[FEATURES].values.astype(float)
    X_train, X_test = X_all[train_idx], X_all[test_idx]

    metrics: dict = {}
    surrogate_times: list[float] = []

    for target in TARGETS:
        valid = ~pd.isna(df[target])
        if valid.sum() < 5:
            logger.warning("Too few valid rows for target '%s'; skipping.", target)
            continue

        y_all = df[target].values.astype(float)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        for model_name, model in [
            ("GP", GaussianProcessSurrogate(n_restarts=3)),
            ("RF", RandomForestSurrogate(n_estimators=50)),
        ]:
            model.fit(X_train, y_train)

            t0 = time.perf_counter()
            pred = model.predict(X_test)
            pred_time = time.perf_counter() - t0

            if len(X_test) > 0:
                surrogate_times.append(pred_time / len(X_test))

            mae = float(np.mean(np.abs(pred - y_test)))
            rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
            eval_metrics = evaluate_surrogate(model, X_test, y_test)
            r2 = float(eval_metrics["r2"])

            key = f"{model_name}/{target}"
            metrics[key] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n_train": int(len(X_train)),
                "n_test": int(n_test),
            }

    # Speedup estimate
    if "sim_time_s" in df.columns:
        mean_sim_time = float(df["sim_time_s"].mean())
    else:
        mean_sim_time = _SPICE_TIME_PER_SAMPLE_S

    mean_surr_time = float(np.mean(surrogate_times)) if surrogate_times else 1e-6
    speedup = mean_sim_time / (mean_surr_time + 1e-12)
    reduction_ratio = 1.0 - 1.0 / max(1.0, speedup)
    metrics["speedup"] = {
        "mean_sim_time_s": mean_sim_time,
        "mean_surrogate_time_s": mean_surr_time,
        "speedup_x": float(speedup),
        "spice_calls_saved_pct": float(min(100.0, reduction_ratio * 100)),
    }

    _print_table(metrics)
    _save_plot(metrics, out_dir)

    return metrics


def _print_table(metrics: dict) -> None:
    """Print a formatted metric table to stdout."""
    header = f"{'Model/Target':<25} {'MAE':>12} {'RMSE':>12} {'R²':>8} {'N_train':>8}"
    sep = "=" * 70
    print("\n" + sep)
    print(header)
    print("-" * 70)
    for key, vals in metrics.items():
        if key == "speedup":
            continue
        print(
            f"{key:<25} {vals['mae']:>12.4e} {vals['rmse']:>12.4e} "
            f"{vals['r2']:>8.4f} {vals['n_train']:>8d}"
        )
    spd = metrics.get("speedup", {})
    print(sep)
    print(
        f"Speedup estimate : {spd.get('speedup_x', 0):.1f}×  "
        f"(SPICE calls saved: {spd.get('spice_calls_saved_pct', 0):.1f}%)"
    )
    target_met = "✓ TARGET MET" if spd.get("spice_calls_saved_pct", 0) >= 50 else "✗ below target"
    print(f"50 % reduction   : {target_met}")
    print(sep + "\n")


def _save_plot(metrics: dict, out_dir: Path) -> None:
    """Save a MAE bar-chart to out_dir/mae_per_spec.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")
        return

    keys = [k for k in metrics if k != "speedup"]
    maes = [metrics[k]["mae"] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.4), 4))
    colors = ["steelblue" if k.startswith("GP") else "darkorange" for k in keys]
    bars = ax.bar(keys, maes, color=colors)
    ax.set_ylabel("MAE (original units)")
    ax.set_title("Surrogate MAE per output spec (held-out test set)")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.3e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    spd = metrics.get("speedup", {})
    fig.text(
        0.5,
        -0.04,
        f"Speedup: {spd.get('speedup_x', 0):.1f}×  |  "
        f"SPICE calls saved: {spd.get('spice_calls_saved_pct', 0):.1f}%",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout()
    plot_path = out_dir / "mae_per_spec.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to %s", plot_path)
    print(f"Plot saved → {plot_path}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="ML surrogate evaluation loop for bandgap design."
    )
    parser.add_argument(
        "--n-samples", type=int, default=150, help="Dataset size (default: 150)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/surrogate_eval",
        help="Output directory for plot and logs (default: results/surrogate_eval).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--use-ngspice",
        action="store_true",
        help="Use real ngspice simulations instead of the Brokaw physics proxy.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.use_ngspice:
        try:
            from bandgap.runner import BandgapRunner
            runner = BandgapRunner()
        except FileNotFoundError as exc:
            logger.error(
                "Could not initialise BandgapRunner: %s  "
                "Install ngspice or remove --use-ngspice.",
                exc,
            )
            raise SystemExit(1) from exc
        rng = np.random.default_rng(seed=args.seed)
        samples = _make_lhs_samples(n_samples=args.n_samples, rng=rng)
        logger.info("Running %d ngspice simulations …", args.n_samples)
        df = run_sweep(samples=samples, out_dir=out_dir / "raw", runner=runner)
        # Drop rows where simulation failed
        df = df[df["error"].fillna("") == ""].reset_index(drop=True)
        logger.info("%d valid simulation rows retained.", len(df))
    else:
        logger.info(
            "Generating %d synthetic samples via Brokaw physics proxy "
            "(pass --use-ngspice for real simulations).",
            args.n_samples,
        )
        df = generate_synthetic_dataset(n_samples=args.n_samples, seed=args.seed)

    logger.info("Dataset: %d rows, columns: %s", len(df), list(df.columns))
    build_and_evaluate(df, out_dir=out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
