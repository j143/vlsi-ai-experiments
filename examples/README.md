# Examples — Open Data, Netlists & Verification

This directory contains **ready-to-run examples** that connect the VLSI-AI
design flow to real, open-source data.  No proprietary PDK access is required.

---

## Quick Reference: Open Resources

| Resource | What It Gives You | URL |
|----------|-------------------|-----|
| **SkyWater SKY130** | Full 130nm CMOS PDK (SPICE models, DRC, LEF/DEF) | <https://github.com/google/skywater-pdk> |
| **open_pdks** (efabless) | Pre-built SKY130 install with ngspice-ready `.lib` | <https://github.com/RTimothyEdwards/open_pdks> |
| **GF180MCU** | GlobalFoundries 180nm CMOS PDK | <https://github.com/google/gf180mcu-pdk> |
| **IHP SG13G2** | IHP 130nm SiGe BiCMOS (has real NPN BJTs) | <https://github.com/IHP-GmbH/IHP-Open-PDK> |
| **BSIM models** (UC Berkeley) | Reference compact models | <https://bsim.berkeley.edu> |
| **Open-source bandgap refs (papers)** | Published Brokaw / sub-1V designs | See list below |

### Published Bandgap Reference Designs (open / academic)

These papers provide **fully disclosed** component values you can use to
validate or seed the optimizer:

1. **Brokaw (1974)** — "A simple three-terminal IC bandgap reference," *IEEE JSSC*.
   Classic topology; our `bandgap_simple.sp` follows this.
2. **Banba et al. (1999)** — "A CMOS bandgap reference circuit with sub-1-V operation,"
   *IEEE JSSC* 34(5). Sub-1V, all-CMOS (no BJT needed).
3. **Leung & Mok (2002)** — "A sub-1-V 15-ppm/°C CMOS bandgap voltage reference
   without requiring low threshold voltage device," *IEEE JSSC* 37(4).
4. **SkyWater community designs** — open-source tapeouts at
   <https://github.com/efabless/caravel_user_project> often include bandgap cells.
5. **IHP SG13G2 examples** — the IHP PDK ships with a reference design including
   real NPN BJT-based bandgap; see `IHP-Open-PDK/ihp-sg13g2/libs.ref/`.

### Open Simulation Datasets

| Dataset / Source | Description |
|------------------|-------------|
| **ieee-ceda.org Open-IC datasets** | Circuit-level benchmark data (some analog) |
| **CircuitNet** (ISPD contest data) | Layout+netlist dataset for ML-EDA tasks |
| **MAGICAL benchmark** (UT Austin) | Analog layout benchmarks including OTA, bandgap |
| **This repo `examples/demo_dataset.csv`** | 200-point analytically-generated reference set |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This guide |
| `sky130_bandgap.sp` | Bandgap netlist using **bundled** SKY130 TT models (works out of box) |
| `ihp_sg13g2_bandgap.sp` | Bandgap netlist adapted for IHP SG13G2 NPN BJTs |
| `generate_reference_dataset.py` | Creates a 200-point calibration dataset from the analytical model |
| `demo_dataset.csv` | Pre-generated 200-point dataset (committed for immediate use) |
| `run_full_pipeline.py` | End-to-end: load data → train surrogate → optimize → evaluate |
| `open_data_guide.md` | Deep-dive on obtaining & installing open PDKs |

> **Bundled models**: Real SKY130 SPICE models (TT corner) live in `pdk/sky130/`.
> The SKY130 netlist includes them automatically — no setup needed.

---

## How to Get Started (no ngspice needed)

```bash
# 1. Generate the reference dataset (analytical, instant)
python examples/generate_reference_dataset.py

# 2. Run the full ML pipeline on that dataset
python examples/run_full_pipeline.py

# 3. Inspect outputs
cat results/example_pipeline_report.json
```

## How to Use with Real SKY130 Simulation

The repository bundles real SKY130 SPICE models (TT corner) in `pdk/sky130/`.
The example netlist is pre-wired to use them:

```bash
# 1. Install ngspice
sudo apt-get install -y ngspice

# 2. Run the SKY130 netlist (models included, no extra setup)
ngspice examples/sky130_bandgap.sp

# 3. For full-corner simulation, install the complete PDK:
pip install volare && volare enable --pdk sky130
# Then edit sky130_bandgap.sp to switch from bundled to full .lib

# 4. Run a design sweep
python data_gen/sweep_bandgap.py --mode lhs --n-samples 50 --out datasets/ \
    --netlist examples/sky130_bandgap.sp

# 5. Train surrogate on real data
python examples/run_full_pipeline.py --dataset datasets/bandgap_sweep_*.csv
```
