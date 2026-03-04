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

from data_gen.sweep_bandgap import _make_lhs_samples  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Physical constants
K_BOLTZMANN = 1.380649e-23  # J/K
Q_ELECTRON = 1.602176634e-19  # C
T_NOM = 300.0  # K (27°C)
VT_NOM = K_BOLTZMANN * T_NOM / Q_ELECTRON  # ~25.85 mV

# Minimum BJT area ratio to avoid log(1)=0 or log(<1)=negative in Brokaw formula
_MIN_N = 1.01

# Process corner definitions: values multiply dvbe_dt (-2 mV/K) to model
# process spread in the temperature-dependent Vbe slope (∂Vbe/∂T).
# Used to generate multi-corner datasets for robustness evaluation.
CORNERS = {
    "tt": 1.000,  # typical-typical — nominal ∂Vbe/∂T
    "ff": 0.980,  # fast-fast  — 2% smaller |∂Vbe/∂T|, slightly less cancellation
    "ss": 1.020,  # slow-slow  — 2% larger |∂Vbe/∂T|, slightly more cancellation
}


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
    N = max(N, _MIN_N)
    vbe = 0.65  # nominal Vbe at 300K
    vref = vbe + (R1 / R2) * VT_NOM * np.log(N)
    # Clip to physically reasonable range
    return float(np.clip(vref, 0.3, 2.5))


def analytical_tc(
    N: float, R1: float, R2: float, corner_factor: float = 1.0, **_kwargs
) -> float:
    """Estimate temperature coefficient [ppm/°C] using two-temperature method.

    Computes Vref at T_cold=-40°C and T_hot=125°C, then:
        TC = |Vref_hot - Vref_cold| / (Vref_nom * ΔT) * 1e6

    The Brokaw cell achieves first-order TC cancellation.  Residual TC arises
    from higher-order Vbe(T) curvature; this model captures the dominant linear
    term only (∂Vbe/∂T ≈ -2 mV/K).

    Parameters
    ----------
    N, R1, R2 : Design variables.
    corner_factor : Multiplier on ∂Vbe/∂T to model process spread.
    """
    N = max(N, _MIN_N)
    dvbe_dt = -2.0e-3 * corner_factor  # V/K  (typical silicon BJT, scaled by process corner)
    T_cold = 233.0  # K  (-40°C)
    T_hot = 398.0   # K  (125°C)

    def vref_at_T(T: float) -> float:
        VT = K_BOLTZMANN * T / Q_ELECTRON
        Vbe = 0.65 + dvbe_dt * (T - T_NOM)
        return float(np.clip(Vbe + (R1 / R2) * VT * np.log(N), 0.3, 2.5))

    vref_cold = vref_at_T(T_cold)
    vref_hot = vref_at_T(T_hot)
    vref_nom = analytical_vref(N, R1, R2)

    if vref_nom <= 0:
        return 999.0

    tc = abs(vref_hot - vref_cold) / (vref_nom * (T_hot - T_cold)) * 1e6
    return round(float(tc), 2)


def analytical_psrr(N: float, **_kwargs) -> float:
    """Estimate PSRR [dB] from simplified Brokaw model.

    Higher N → better PSRR due to larger loop gain.  This is an approximation;
    real PSRR requires AC analysis.
    """
    return round(float(-58.0 - 1.2 * np.log10(max(N, _MIN_N))), 2)


def analytical_iq(R1: float, R2: float, vdd: float = 1.8, **_kwargs) -> float:
    """Estimate quiescent current [µA] from bias path.

    Approximate: I ≈ VDD / (R1 + R2) for series path.
    """
    return float(vdd / (R1 + R2) * 1e6)


def generate_dataset(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a reference dataset of design points with analytical outputs.

    Generates one row per LHS sample × process corner (tt/ff/ss), giving
    3× n_samples rows total.  The ``corner`` column enables multi-corner
    surrogate training and robustness evaluation (Direction B in the roadmap).

    Parameters
    ----------
    n_samples : Number of LHS samples (per corner).
    seed : Random seed.

    Returns
    -------
    pd.DataFrame with columns:
        N, R1, R2, W_P, L_P,
        vref_V, tc_ppm_C, psrr_dB, iq_uA,
        corner, err_from_target_mV, spec_vref_pass, spec_tc_pass,
        source
    """
    rng = np.random.default_rng(seed)
    samples = _make_lhs_samples(n_samples=n_samples, rng=rng)

    rows = []
    target_V = 1.200
    tolerance_V = 0.010
    max_tc_ppm = 20  # matches bandgap/specs.yaml after the spec fix

    for corner, factor in CORNERS.items():
        for params in samples:
            vref = analytical_vref(**params)
            tc = analytical_tc(**params, corner_factor=factor)
            psrr = analytical_psrr(**params)
            iq = analytical_iq(**params)
            err_mV = abs(vref - target_V) * 1000
            spec_vref = err_mV <= tolerance_V * 1000
            spec_tc = tc <= max_tc_ppm  # one-sided max (direction: minimize)

            rows.append({
                **params,
                "vref_V": round(vref, 6),
                "tc_ppm_C": tc,
                "psrr_dB": psrr,
                "iq_uA": round(iq, 4),
                "corner": corner,
                "err_from_target_mV": round(err_mV, 3),
                "spec_vref_pass": spec_vref,
                "spec_tc_pass": spec_tc,
                "error": "",
                "source": "analytical",
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Generated %d samples (%d LHS × %d corners). Vref spec pass: %d/%d (%.0f%%). "
        "TC spec pass: %d/%d (%.0f%%)",
        len(df),
        n_samples,
        len(CORNERS),
        df["spec_vref_pass"].sum(),
        len(df),
        df["spec_vref_pass"].mean() * 100,
        df["spec_tc_pass"].sum(),
        len(df),
        df["spec_tc_pass"].mean() * 100,
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate analytical reference dataset")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of LHS samples per corner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--save-to-datasets", action="store_true",
        help="Also save to datasets/bandgap_sweep_real_sky130.csv for pipeline use",
    )
    args = parser.parse_args()

    out_path = args.out or str(Path(__file__).parent / "demo_dataset.csv")
    df = generate_dataset(n_samples=args.n_samples, seed=args.seed)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    if args.save_to_datasets:
        datasets_dir = Path(__file__).parent.parent / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        real_path = datasets_dir / "bandgap_sweep_real_sky130.csv"
        df.to_csv(str(real_path), index=False)
        logger.info("Also saved to %s (pipeline-ready dataset)", real_path)

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Samples:         {len(df)} ({args.n_samples} LHS × {len(CORNERS)} corners)")
    print(f"Corners:         {', '.join(CORNERS)}")
    print(f"Vref range:      {df['vref_V'].min():.4f} – {df['vref_V'].max():.4f} V")
    print(f"Vref mean:       {df['vref_V'].mean():.4f} V")
    print(f"TC range:        {df['tc_ppm_C'].min():.1f} – {df['tc_ppm_C'].max():.1f} ppm/°C")
    print(f"PSRR range:      {df['psrr_dB'].min():.1f} – {df['psrr_dB'].max():.1f} dB")
    print(f"Iq range:        {df['iq_uA'].min():.2f} – {df['iq_uA'].max():.2f} µA")
    print(f"Vref spec pass:  {df['spec_vref_pass'].mean():.1%}")
    print(f"TC spec pass:    {df['spec_tc_pass'].mean():.1%}")
    print(f"Best |err|:      {df['err_from_target_mV'].min():.2f} mV")
    print(f"Output:          {out_path}")


if __name__ == "__main__":
    main()
