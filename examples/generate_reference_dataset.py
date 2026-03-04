#!/usr/bin/env python3
"""
examples/generate_reference_dataset.py
==========================================
Generates a 200-point reference dataset using the analytical Brokaw bandgap
model (no ngspice required).  The dataset can be used to:

  1. Validate the surrogate training pipeline.
  2. Benchmark the Bayesian optimizer on known-good data.
  3. Serve as a baseline before real SPICE data is available.

The analytical model:
    Vref = Vbe(T) + (R1/R2) * kT/q * ln(N)
    where Vbe ≈ 0.65 V at T=300K  (simplified)
    and kT/q ≈ 25.85 mV at T=300K

This is the EXACT same model used by the SyntheticBandgapRunner in
ml/optimize.py, so surrogate predictions trained on this data should
match the synthetic runner closely.

Usage::

    python examples/generate_reference_dataset.py
    python examples/generate_reference_dataset.py --n-samples 500 --seed 123

Output:
    examples/demo_dataset.csv   (or --out path)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_gen.sweep_bandgap import PARAM_SPACE, _make_lhs_samples  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Physical constants
K_BOLTZMANN = 1.380649e-23  # J/K
Q_ELECTRON = 1.602176634e-19  # C
T_NOM = 300.0  # K (27°C)
VT_NOM = K_BOLTZMANN * T_NOM / Q_ELECTRON  # ~25.85 mV


def analytical_vref(N: float, R1: float, R2: float, **_kwargs) -> float:
    """Compute Vref using the simplified Brokaw analytical model.

    Vref = Vbe + (R1 / R2) * VT * ln(N)

    This matches the SyntheticBandgapRunner formula in ml/optimize.py.
    R1 is the series resistor, R2 is the PTAT resistor to ground, so the
    gain factor is R1/R2.

    Parameters
    ----------
    N : BJT emitter area ratio (dimensionless, ≥ 2).
    R1 : Top/series resistor [Ω].
    R2 : PTAT resistor [Ω].
    """
    N = max(N, 1.01)  # avoid log(1) = 0
    vbe = 0.65  # nominal Vbe at 300K
    vref = vbe + (R1 / R2) * VT_NOM * np.log(N)
    # Clip to physically reasonable range
    return float(np.clip(vref, 0.3, 2.5))


def analytical_iq(R1: float, R2: float, vdd: float = 1.8, **_kwargs) -> float:
    """Estimate quiescent current [µA] from bias path.

    Approximate: I ≈ VDD / (R1 + R2) for series path.
    """
    return float(vdd / (R1 + R2) * 1e6)


def generate_dataset(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a reference dataset of design points with analytical outputs.

    Parameters
    ----------
    n_samples : Number of LHS samples.
    seed : Random seed.

    Returns
    -------
    pd.DataFrame with columns: N, R1, R2, W_P, L_P, vref_V, iq_uA,
                                err_from_target_mV, spec_vref_pass
    """
    rng = np.random.default_rng(seed)
    samples = _make_lhs_samples(n_samples=n_samples, rng=rng)

    rows = []
    target_V = 1.200
    tolerance_V = 0.010

    for params in samples:
        vref = analytical_vref(**params)
        iq = analytical_iq(**params)
        err_mV = abs(vref - target_V) * 1000
        spec_pass = err_mV <= tolerance_V * 1000

        rows.append({
            **params,
            "vref_V": round(vref, 6),
            "iq_uA": round(iq, 4),
            "err_from_target_mV": round(err_mV, 3),
            "spec_vref_pass": spec_pass,
            "source": "analytical",
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Generated %d samples. Spec pass: %d/%d (%.0f%%)",
        len(df),
        df["spec_vref_pass"].sum(),
        len(df),
        df["spec_vref_pass"].mean() * 100,
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate analytical reference dataset")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    out_path = args.out or str(Path(__file__).parent / "demo_dataset.csv")
    df = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Samples:         {len(df)}")
    print(f"Vref range:      {df['vref_V'].min():.4f} – {df['vref_V'].max():.4f} V")
    print(f"Vref mean:       {df['vref_V'].mean():.4f} V")
    print(f"Iq range:        {df['iq_uA'].min():.2f} – {df['iq_uA'].max():.2f} µA")
    print(f"Spec pass rate:  {df['spec_vref_pass'].mean():.1%}")
    print(f"Best |err|:      {df['err_from_target_mV'].min():.2f} mV")
    print(f"Output:          {out_path}")


if __name__ == "__main__":
    main()
