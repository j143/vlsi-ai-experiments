"""
data_gen/sweep_bandgap.py — Dataset Generation by Sweeping Bandgap Design Variables
======================================================================================
Runs ngspice over a grid (or random sample) of design variable combinations and
logs all outputs to a timestamped CSV file.

Usage::

    python data_gen/sweep_bandgap.py --mode grid --n-samples 50 --out datasets/

    # Or with Latin-hypercube sampling:
    python data_gen/sweep_bandgap.py --mode lhs --n-samples 100 --out datasets/

Output CSV columns:
    N, R1, R2, W_P, L_P,           ← design variables (SI units)
    vref_V, iq_uA,                  ← primary outputs
    spec_vref_pass, spec_iq_pass,   ← per-spec pass/fail flags
    sim_time_s, error               ← metadata

Notes
-----
- Requires ngspice to be installed; rows with simulation errors are kept in the
  dataset but flagged in the 'error' column.
- Adjust the parameter ranges in PARAM_SPACE below to match your design target
  and technology constraints.
- All parameter ranges are ILLUSTRATIVE; replace with technology-appropriate
  values once PDK models are available.
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path so we can import bandgap package
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bandgap.runner import BandgapRunner  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design variable search space
# ---------------------------------------------------------------------------
# Each entry: (name, min, max, scale)
# scale = 'log' → sample uniformly in log space (good for resistors, currents)
# scale = 'lin' → sample uniformly in linear space
# scale = 'int' → sample integers (good for BJT ratio N)
#
# ILLUSTRATIVE RANGES — adjust for actual PDK constraints.
PARAM_SPACE: list[tuple[str, float, float, str]] = [
    ("N",   4,     20,    "int"),   # BJT emitter area ratio [dimensionless]
    ("R1",  30e3,  300e3, "log"),   # Top resistor [Ω]
    ("R2",  3e3,   40e3,  "log"),   # PTAT resistor [Ω]
    ("W_P", 1e-6,  20e-6, "lin"),   # PMOS width [m]
    ("L_P", 0.35e-6, 4e-6, "lin"),  # PMOS length [m]
]


def _make_grid_samples(n_per_dim: int = 3) -> list[dict]:
    """Create a regular grid of parameter combinations.

    Parameters
    ----------
    n_per_dim:
        Number of grid points per dimension (total points = n_per_dim^D).

    Returns
    -------
    list of dict
        Each dict maps parameter name → value.
    """
    grids = []
    for name, lo, hi, scale in PARAM_SPACE:
        if scale == "log":
            pts = np.logspace(np.log10(lo), np.log10(hi), n_per_dim)
        elif scale == "int":
            pts = np.linspace(lo, hi, n_per_dim, dtype=float).astype(int)
        else:
            pts = np.linspace(lo, hi, n_per_dim)
        grids.append((name, pts))

    import itertools

    samples = []
    for combo in itertools.product(*[pts for _, pts in grids]):
        sample = {name: float(v) for (name, _), v in zip(grids, combo)}
        # Round integers
        for name, _, _, scale in PARAM_SPACE:
            if scale == "int":
                sample[name] = int(round(sample[name]))
        samples.append(sample)
    return samples


def _make_lhs_samples(n_samples: int, rng: np.random.Generator | None = None) -> list[dict]:
    """Create Latin-Hypercube Samples (LHS) of the parameter space.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    rng:
        NumPy random generator (for reproducibility).

    Returns
    -------
    list of dict
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_dims = len(PARAM_SPACE)
    # LHS: stratify each dimension
    cut = np.linspace(0, 1, n_samples + 1)
    u = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        u[:, j] = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u[:, j])

    samples = []
    for i in range(n_samples):
        sample = {}
        for j, (name, lo, hi, scale) in enumerate(PARAM_SPACE):
            t = u[i, j]
            if scale == "log":
                val = np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo)))
            elif scale == "int":
                val = int(round(lo + t * (hi - lo)))
            else:
                val = lo + t * (hi - lo)
            sample[name] = float(val)
            if scale == "int":
                sample[name] = int(round(sample[name]))
        samples.append(sample)
    return samples


def run_sweep(
    samples: list[dict],
    out_dir: Path,
    runner: BandgapRunner | None = None,
) -> pd.DataFrame:
    """Run ngspice for each sample and collect results into a DataFrame.

    Parameters
    ----------
    samples:
        List of design variable dicts.
    out_dir:
        Directory to write the CSV file.
    runner:
        BandgapRunner instance. Created with defaults if None.

    Returns
    -------
    pd.DataFrame
        Dataset with design variables, outputs, and metadata columns.
    """
    if runner is None:
        runner = BandgapRunner()

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    ngspice_ok = runner.is_ngspice_available()
    if not ngspice_ok:
        logger.warning(
            "ngspice not found. Simulation results will be empty (error column set). "
            "Install ngspice or set NGSPICE_BIN to run real simulations."
        )

    for idx, params in enumerate(samples):
        t0 = time.perf_counter()
        result = runner.run(params)
        elapsed = time.perf_counter() - t0

        row = {**params}
        row["vref_V"] = result.get("vref_V")
        row["iq_uA"] = result.get("iq_uA")

        for spec_name, passed in result.get("spec_checks", {}).items():
            row[f"spec_{spec_name}_pass"] = passed

        row["sim_time_s"] = round(elapsed, 4)
        row["error"] = result.get("error") or ""
        rows.append(row)

        if (idx + 1) % 10 == 0:
            logger.info("Progress: %d / %d samples simulated.", idx + 1, len(samples))

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"bandgap_sweep_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)

    return df


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Sweep bandgap design space and log to CSV.")
    parser.add_argument(
        "--mode",
        choices=["grid", "lhs"],
        default="lhs",
        help="Sampling mode: 'grid' (regular grid) or 'lhs' (Latin-hypercube).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples (for LHS) or grid points per dimension (for grid).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for LHS sampling.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="datasets",
        help="Output directory for CSV files.",
    )
    args = parser.parse_args()

    if args.mode == "grid":
        samples = _make_grid_samples(n_per_dim=args.n_samples)
        logger.info("Generated %d grid samples.", len(samples))
    else:
        rng = np.random.default_rng(seed=args.seed)
        samples = _make_lhs_samples(n_samples=args.n_samples, rng=rng)
        logger.info("Generated %d LHS samples.", len(samples))

    run_sweep(samples=samples, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
